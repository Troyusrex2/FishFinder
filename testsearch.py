import os
import logging
import yaml
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timezone
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI
import string

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from config.yaml
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)['app_config']

# Extract configurations
COLLECTION_NAME = config['collection_name']
DB_NAME = config['mongodb']['db_name']
SEARCH_PARAMS = config['search_params']
EMBEDDING_CONFIG = config['embedding']

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def preprocess_query(query):
    query = query.lower()
    query = query.translate(str.maketrans("", "", string.punctuation))
    tokens = query.split()
    preprocessed_query = " ".join(tokens)
    return preprocessed_query

def generate_query_vector(query):
    logger.info(f"Generating query vector for: {query}")
    embed_model = OpenAIEmbedding(model=EMBEDDING_CONFIG['model'], embed_batch_size=EMBEDDING_CONFIG['batch_size'])
    embedding = embed_model.get_text_embedding(query)
    logger.info(f"Generated embedding: {embedding[:10]}...")
    return embedding

def hybrid_search(query, collection_name, selected_channels):
    preprocessed_query = preprocess_query(query)
    query_vector = generate_query_vector(preprocessed_query)
    
    vector_pipeline = [
        {
            '$vectorSearch': {
                'index': 'vector_index', 
                'path': 'Embedding',
                'queryVector': query_vector,
                'numCandidates': SEARCH_PARAMS['num_candidates'],
                'limit': SEARCH_PARAMS['limit']
            }
        },
        {
            '$match': {
                '$or': [
                    {'Channel_Name': {'$in': selected_channels}},
                    {'Channel_Name': {'$exists': False}}  # Include documents without Channel_Name
                ]
            }
        },
        {
            '$project': {
                '_id': 1,
                'Title': 1,
                'URL': 1,
                'Transcript': 1,
                'Channel_Name': 1,
                'score': {'$meta': 'vectorSearchScore'}
            }
        }
    ]
    
    vector_results = list(db[collection_name].aggregate(vector_pipeline))
    
    text_pipeline = [
        {
            '$search': {
                'index': 'default',
                'text': {
                    'query': query,
                    'path': ['Title', 'Transcript']
                }
            }
        },
        {
            '$limit': SEARCH_PARAMS['limit']
        },
        {
            '$project': {
                '_id': 1,
                'Title': 1,
                'URL': 1,
                'Transcript': 1,
                'Channel_Name': 1,
                'score': {'$meta': 'searchScore'}
            }
        }
    ]
    
    text_results = list(db[collection_name].aggregate(text_pipeline))
    
    all_results = {}
    for i, doc in enumerate(vector_results):
        doc_id = str(doc['_id'])
        if doc_id not in all_results:
            all_results[doc_id] = {'doc': doc, 'vector_rank': i + 1, 'text_rank': SEARCH_PARAMS['limit'] + 1}
    
    for i, doc in enumerate(text_results):
        doc_id = str(doc['_id'])
        if doc_id in all_results:
            all_results[doc_id]['text_rank'] = i + 1
        else:
            all_results[doc_id] = {'doc': doc, 'vector_rank': SEARCH_PARAMS['limit'] + 1, 'text_rank': i + 1}
    
    for doc_id, info in all_results.items():
        info['rrf_score'] = (1 / (SEARCH_PARAMS['k'] + info['vector_rank'])) + (1 / (SEARCH_PARAMS['k'] + info['text_rank']))
    
    sorted_results = sorted(all_results.values(), key=lambda x: x['rrf_score'], reverse=True)
    
    final_results = [item['doc'] for item in sorted_results[:SEARCH_PARAMS['limit']]]
    
    return final_results

def print_results(results):
    for doc in results:
        print(f"Title: {doc.get('Title', 'N/A')}")
        print(f"URL: {doc.get('URL', 'N/A')}")
        if 'Channel_Name' in doc:
            print(f"Channel: {doc['Channel_Name']}")
        if 'Transcript' in doc:
            print(f"Transcript: {doc['Transcript'][:500]}...")  # Truncate long transcripts
        print("\n")

if __name__ == "__main__":
    query = "Slacktyde"
    selected_channels = []  # Replace with actual channel names or leave empty to include all

    results = hybrid_search(query, COLLECTION_NAME, selected_channels)
    print_results(results)

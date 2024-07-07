import os
from pymongo import MongoClient
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI

# Set your OpenAI API key
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = 'influencers'
COLLECTION_NAME = 'MakeUpHelper'
EMBEDDING_MODEL = "text-embedding-3-small"  # Change this to "text-embedding-3-small" when available

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

embed_model = OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=10)

def add_embeddings_to_transcripts():
    # Find all documents that do not have embeddings yet
    documents_without_embeddings = collection.find({"Embedding": {"$exists": False}})

    for doc in documents_without_embeddings:
        transcript_text = doc["Transcript"]
        
        # Generate embeddings for the transcript text
        
        embeddings = embed_model.get_text_embedding(transcript_text)

        # Update the document with the generated embeddings
        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"Embedding": embeddings}}
        )
        print(f"Added embeddings for Video ID: {doc['VideoID']}, Chunk Index: {doc['ChunkIndex']}")

if __name__ == "__main__":
    add_embeddings_to_transcripts()

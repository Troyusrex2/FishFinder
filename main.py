import streamlit as st
import asyncio
import nest_asyncio
import os
import re
import uuid
import yaml
from pymongo import MongoClient
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI
import logging
import time
import random
from collections import Counter
from dotenv import load_dotenv
from datetime import datetime, timezone
from contextlib import closing
import string

# Load configuration from config.yaml
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)['app_config']

# Extract configurations
COLLECTION_NAME = config['collection_name']
APP_TITLE = config['app_title']
DEFAULT_MESSAGE = config['default_message']
default_example_questions = config['example_questions']
PROMPT_BASE = config['prompt']
inappropriate_words = config['inappropriate_words']
disclaimer = config['disclaimer']
sidebar_links = config['sidebar_links']
donation_text = config['donation_text']
donation_url = config['donation_url']
STOP_WORDS = set(config['stop_words'])
MAX_QUESTIONS = config['max_questions']
OPENAI_MODEL = config['openai_model']
DB_NAME = config['mongodb']['db_name']
QUERY_COLLECTION_NAME = config['mongodb']['query_collection']
CACHE_COLLECTION_NAME = config['mongodb']['cache_collection']
FEEDBACK_COLLECTION_NAME = config['mongodb']['feedback_collection']
SEARCH_PARAMS = config['search_params']
EMBEDDING_CONFIG = config['embedding']
MAX_EXAMPLE_QUESTION_LENGTH = config['max_example_question_length']
NUM_EXAMPLE_QUESTIONS = config['num_example_questions']
NUM_VIDEOS_TO_DISPLAY = config['num_videos_to_display']
PROCESSING_MESSAGES = config['processing_messages']

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=config['logging_level'])
logger = logging.getLogger(__name__)

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
query_collection = db[QUERY_COLLECTION_NAME]
cache_collection = db[CACHE_COLLECTION_NAME]
feedback_collection = db[FEEDBACK_COLLECTION_NAME]

# OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize session state variables
if "current_question" not in st.session_state:
    st.session_state["current_question"] = None
if "show_example_questions" not in st.session_state:
    st.session_state["show_example_questions"] = True
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "question_count" not in st.session_state:
    st.session_state["question_count"] = 0
if "input_disabled" not in st.session_state:
    st.session_state["input_disabled"] = False
if "input_message" not in st.session_state:
    st.session_state["input_message"] = ""
if "displayed_videos" not in st.session_state:
    st.session_state.displayed_videos = set()
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": DEFAULT_MESSAGE}]
if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False
if "temp_selected_channels" not in st.session_state:
    st.session_state["temp_selected_channels"] = []

def get_distinct_channels():
    logger.info("Fetching distinct channels")
    sample_doc = collection.find_one()
    if sample_doc and 'Channel_Name' in sample_doc:
        channels = collection.distinct("Channel_Name")
        logger.info(f"Distinct channels found: {channels}")
        return sorted(channels)
    else:
        logger.warning("No channels found")
        return []  # Return an empty list if there are no channels

def preprocess_query(query):
    logger.info(f"Preprocessing query: {query}")
    query = query.lower()
    query = query.translate(str.maketrans("", "", string.punctuation))
    tokens = query.split()
    tokens = [token for token in tokens if token not in STOP_WORDS]
    preprocessed_query = " ".join(tokens)
    logger.info(f"Preprocessed query: {preprocessed_query}")
    return preprocessed_query

def generate_query_vector(query):
    logger.info(f"Generating query vector for: {query}")
    embed_model = OpenAIEmbedding(model=EMBEDDING_CONFIG['model'], embed_batch_size=EMBEDDING_CONFIG['batch_size'])
    embedding = embed_model.get_text_embedding(query)
    logger.info(f"Generated embedding: {embedding[:10]}...")
    return embedding

def get_cached_response(query, collection_name, selected_channels, all_channels):
    logger.info(f"Checking cache for query: {query}")
    if set(selected_channels) == set(all_channels):
        cached_item = cache_collection.find_one({
            'query': query, 
            'collection_name': collection_name,
            'all_channels': True
        })
        if cached_item:
            logger.info("Cache hit")
            return cached_item['response']
    logger.info("Cache miss")
    return None

def cache_response(query, preprocessed_query, response, collection_name, selected_channels, all_channels):
    logger.info(f"Caching response for query: {query}")
    if set(selected_channels) == set(all_channels):
        cache_collection.insert_one({
            'query': query,
            'preprocessed_query': preprocessed_query,
            'response': response,
            'collection_name': collection_name,
            'all_channels': True,
            'timestamp': datetime.now(timezone.utc)
        })

def hybrid_search(query, collection_name, selected_channels):
    logger.info(f"Starting hybrid search for query: {query}")
    preprocessed_query = preprocess_query(query)
    query_vector = generate_query_vector(preprocessed_query)
    
    # Check if the collection has 'Channel_Name' field
    sample_doc = db[collection_name].find_one()
    has_channel_name = 'Channel_Name' in sample_doc if sample_doc else False
    
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

    if has_channel_name:
        vector_pipeline.append({
            '$match': {
                'Channel_Name': {'$in': selected_channels}
            }
        })
    
    vector_pipeline.append({
        '$project': {
            '_id': 1,
            'Title': 1,
            'URL': 1,
            'Transcript': 1,
            'Channel_Name': 1,
            'score': {'$meta': 'vectorSearchScore'}
        }
    })
    
    vector_results = list(db[collection_name].aggregate(vector_pipeline))
    logger.info(f"Vector search results: {vector_results[:5]}")  # Log first 5 results for brevity
    
    text_pipeline = [
        {
            '$search': {
                'index': 'default',
                'text': {
                    'query': query,
                    'path': ['Title', 'Transcript'] + (['Channel_Name'] if has_channel_name else [])
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
    
    if has_channel_name:
        text_pipeline.insert(1, {
            '$match': {
                'Channel_Name': {'$in': selected_channels}
            }
        })
    
    text_results = list(db[collection_name].aggregate(text_pipeline))
    logger.info(f"Text search results: {text_results[:5]}")  # Log first 5 results for brevity
    
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
    
    # Log the structure of the first result for debugging
    if final_results:
        logger.info(f"Structure of first result: {final_results[0].keys()}")
    else:
        logger.warning("No results found in hybrid search")
    
    return final_results

def format_results(results):
    logger.info("Formatting results")
    formatted_results = ""
    for i, doc in enumerate(results):
        formatted_results += f"Title: {doc.get('Title', 'N/A')}\n"
        formatted_results += f"URL: {doc.get('URL', 'N/A')}\n"
        if 'Channel_Name' in doc:
            formatted_results += f"Channel: {doc['Channel_Name']}\n"
        if 'Transcript' in doc:
            formatted_results += f"Transcript: {doc['Transcript'][:500]}...\n"  # Truncate long transcripts
        formatted_results += "\n"
    logger.info(f"Formatted results: {formatted_results}")
    return formatted_results

def generate_output(prompt, results):
    logger.info(f"Generating output for prompt: {prompt}")
    formatted_results = format_results(results)
    
    messages = [
        {"role": "system", "content": PROMPT_BASE},
        {"role": "user", "content": f"Prompt: {prompt}\n\nResults:\n{formatted_results}"}
    ]
    
    stream = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        max_tokens=1500,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        stream=True
    )
    
    return stream

def stream_output(stream):
    logger.info("Streaming output")
    response_container = st.empty()
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            response_container.markdown(full_response + "▌")
            time.sleep(0.01)  # Small delay to allow for visual updates
    response_container.markdown(full_response)
    logger.info(f"Full response: {full_response}")
    return full_response

def extract_youtube_urls(response_text):
    logger.info("Extracting YouTube URLs from response")
    youtube_url_pattern = r'https://www\.youtube\.com/watch\?v=[\w-]+'
    matches = re.findall(youtube_url_pattern, response_text)
    logger.info(f"Found YouTube URLs: {matches}")
    return matches

def get_start_time_from_url(url):
    timestamp_pattern = r"t=(\d+)s"
    match = re.search(timestamp_pattern, url)
    if match:
        return int(match.group(1))
    return 0

def get_most_asked_questions(collection_name):
    logger.info("Fetching most asked questions")
    unique_filtered_queries = []
    
    if len(unique_filtered_queries) < NUM_EXAMPLE_QUESTIONS:
        unique_filtered_queries.extend(
            q for q in default_example_questions 
        )
    
    logger.info(f"Most asked questions: {unique_filtered_queries}")
    return unique_filtered_queries[:NUM_EXAMPLE_QUESTIONS]

def save_query_to_db(prompt, preprocessed_prompt, app_name, session_id):
    logger.info(f"Saving query to DB: {prompt}")
    query_data = {
        'prompt': prompt,
        'preprocessed_prompt': preprocessed_prompt,
        'app_name': app_name,
        'timestamp': time.time(),
        'session_id': session_id
    }
    query_collection.insert_one(query_data)
    logger.info(f"Saved query to DB: {query_data}")

def save_feedback(query, feedback_type, comment, app_name):
    logger.info(f"Saving feedback to DB: {query}, type: {feedback_type}, comment: {comment}")
    feedback_data = {
        'query': query,
        'feedback_type': feedback_type,
        'comment': comment,
        'app_name': app_name,
        'timestamp': datetime.now(timezone.utc)
    }
    feedback_collection.insert_one(feedback_data)
    logger.info(f"Saved feedback to DB: {feedback_data}")

def process_user_input(user_input):
    logger.info(f"Processing user input: {user_input}")
    st.session_state["is_processing"] = True
    progress_placeholder = st.empty()
    
    try:
        steps = [
            "Preprocessing query",
            "Searching database",
            "Generating response",
            "Extracting video links",
            "Finalizing results"
        ]
        
        for i, step in enumerate(steps):
            progress = (i + 1) / len(steps)
            progress_placeholder.progress(progress)
            with st.spinner(f"{step}... {random.choice(PROCESSING_MESSAGES)}"):
                if step == "Preprocessing query":
                    preprocessed_input = preprocess_query(user_input)
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    save_query_to_db(user_input, preprocessed_input, COLLECTION_NAME, st.session_state["session_id"])
                elif step == "Searching database":
                    cached_response = get_cached_response(user_input, COLLECTION_NAME, st.session_state.selected_channels, st.session_state['all_channels'])
                    if not cached_response:
                        results = hybrid_search(user_input, COLLECTION_NAME, st.session_state.selected_channels)
                        logger.info(f"Search results: {results}")
                elif step == "Generating response":
                    if cached_response:
                        output = cached_response
                        st.markdown(output)
                    else:
                        stream = generate_output(user_input, results)
                        output = stream_output(stream)
                        cache_response(user_input, preprocessed_input, output, COLLECTION_NAME, st.session_state.selected_channels, st.session_state['all_channels'])
                elif step == "Extracting video links":
                    youtube_urls = extract_youtube_urls(output)
                elif step == "Finalizing results":
                    st.session_state.messages.append({"role": "assistant", "content": output})
                    if youtube_urls:
                        st.write("Related Videos:")
                        cols = st.columns(NUM_VIDEOS_TO_DISPLAY)
                        col_index = 0
                        displayed_in_this_response = 0
                        for url in youtube_urls:
                            video_id = url.split("v=")[-1]
                            if video_id not in st.session_state.displayed_videos and displayed_in_this_response < NUM_VIDEOS_TO_DISPLAY:
                                cols[col_index].video(f"https://youtu.be/{video_id}")
                                st.session_state.displayed_videos.add(video_id)
                                col_index += 1
                                displayed_in_this_response += 1
                                if col_index >= NUM_VIDEOS_TO_DISPLAY:
                                    break
                    st.session_state["question_count"] += 1
    
    finally:
        st.session_state["is_processing"] = False
        progress_placeholder.empty()

def display_feedback(user_input, message_index):
    logger.info(f"Displaying feedback options for message index: {message_index}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Thumbs Up", key=f"thumbs_up_{message_index}"):
            save_feedback(user_input, "positive", "", COLLECTION_NAME)
            st.success("Thank you for your positive feedback!")
    with col2:
        if st.button("👎 Thumbs Down", key=f"thumbs_down_{message_index}"):
            comment = st.text_input("Please tell us why:", key=f"feedback_comment_{message_index}")
            if st.button("Submit Feedback", key=f"submit_feedback_{message_index}"):
                save_feedback(user_input, "negative", comment, COLLECTION_NAME)
                st.success("Thank you for your feedback!")

def update_selected_channels(channel):
    logger.info(f"Updating selected channels: {channel}")
    if not st.session_state["is_processing"]:
        if channel in st.session_state.selected_channels:
            st.session_state.selected_channels.remove(channel)
        else:
            st.session_state.selected_channels.append(channel)
        
        # Ensure at least one channel is selected
        if len(st.session_state.selected_channels) == 0:
            st.session_state.selected_channels = [channel]
        
        # Force a rerun to update the UI
        st.rerun()

# Streamlit app starts here
nest_asyncio.apply()

st.set_page_config(page_title=APP_TITLE)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .reportview-container {
        background: #ffffff;
        color: #000000;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    .stButton>button {
        background-color: transparent;
        color: #000000;
        border: 1px solid #ced4da;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #f1f1f1;
    }
    .stTextInput>div>input {
        border: 1px solid #ced4da;
        padding: 10px;
        border-radius: 5px;
    }
    .stTextInput>div>input:focus {
        border-color: #007bff;
    }
    .stCheckbox {
        padding: 2px 0;
    }
    .stCheckbox label {
        font-size: 14px;
    }
    .sidebar .stCheckbox {
        padding: 1px 0;
    }
    </style>
    """, unsafe_allow_html=True
)

# Header Section
st.image("webheader.png", use_column_width=True)
st.caption(disclaimer)

# Sidebar
st.sidebar.title("Filters")
all_channels = get_distinct_channels()

if all_channels:
    # Initialize selected_channels in session state if not present
    if 'selected_channels' not in st.session_state:
        st.session_state.selected_channels = all_channels.copy()
        st.session_state.temp_selected_channels = all_channels.copy()

    st.sidebar.write("Who's advice should I pick from?")

    for channel in all_channels:
        channel_selected = st.sidebar.checkbox(
            channel, 
            value=(channel in st.session_state.selected_channels),
            key=f"checkbox_{channel}",
            disabled=st.session_state["is_processing"],
            on_change=update_selected_channels,
            args=(channel,)
        )
    # Ensure at least one channel is selected
    if len(st.session_state.temp_selected_channels) == 0:
        st.warning("At least one channel must be selected.")
        st.session_state.temp_selected_channels = [all_channels[0]]

    # Add this line to make all_channels available throughout the script
    st.session_state['all_channels'] = all_channels
else:
    st.session_state['all_channels'] = []
    st.session_state.selected_channels = []

for link in sidebar_links:
    st.sidebar.markdown(f"[{link['text']}]({link['url']})")
st.sidebar.markdown(donation_text)
st.sidebar.markdown(f"Donate at {donation_url}")

# Render all messages stored in the session state
for i, msg in enumerate(st.session_state.messages):
    st.chat_message(msg["role"]).write(msg["content"])
    if msg["role"] == "assistant" and i > 0:
        display_feedback(st.session_state.messages[i-1]["content"], i)

# Display videos if they exist in the session state
if st.session_state.displayed_videos:
    st.write("Related Videos:")
    cols = st.columns(NUM_VIDEOS_TO_DISPLAY)
    for i, video_id in enumerate(list(st.session_state.displayed_videos)[:NUM_VIDEOS_TO_DISPLAY]):
        cols[i].video(f"https://youtu.be/{video_id}")

# Handle example questions and chat input
if st.session_state["question_count"] < MAX_QUESTIONS:
    if st.session_state.get("show_example_questions", True):
        example_questions = get_most_asked_questions(COLLECTION_NAME)
        st.subheader("Most asked questions")
        cols = st.columns(len(example_questions))

        for i, question in enumerate(example_questions):
            if cols[i].button(question, key=f"example_question_{i}", disabled=st.session_state["is_processing"]):
                st.session_state["show_example_questions"] = False
                process_user_input(question)
                st.rerun()

    if not st.session_state["input_disabled"]:
        user_input = st.chat_input("Enter your question here...", disabled=st.session_state["is_processing"])
        if user_input:
            st.session_state["show_example_questions"] = False
            process_user_input(user_input)
            st.rerun()
    else:
        st.text_input("Enter your question here...", value=st.session_state["input_message"], disabled=True)
else:
    st.session_state["input_disabled"] = True
    st.session_state["input_message"] = f"You have reached the limit of {MAX_QUESTIONS} questions for this session. Although provided for free, this service costs money to operate. If you've enjoyed this service please consider donating at {donation_url}"
    st.warning(st.session_state["input_message"])

# Ensure event loop is properly closed
with closing(asyncio.get_event_loop()) as loop:
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()

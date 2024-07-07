import os
import requests
import json
from pymongo import MongoClient
from youtube_transcript_api import YouTubeTranscriptApi
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.core import Document
from openai import OpenAI
import yaml


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)['app_config']

# Extract configurations


# Configuration
YOUTUBE_API_KEY = 'AIzaSyBLLngRq6oVOmTDtu-uUCp_G_dH8bMdqPw'
CHANNEL_ID = 'UCgvxD2vgpldEZlLMDlVaiQg'
CHANNEL_NAME = 'BassFishingHQ'
MAX_TOKENS = 1024
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = 'influencers'
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = config['collection_name']
# OpenAI setup
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

# Embedding setup
embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL, embed_batch_size=10)
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)
base_splitter = SentenceSplitter(chunk_size=512)

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

def get_video_ids(api_key, channel_id):
    videos = []
    url = f"https://www.googleapis.com/youtube/v3/search?part=id&channelId={channel_id}&maxResults=50&type=video&key={api_key}"
    while True:
        response = requests.get(url).json()
        if 'items' not in response:
            break
        videos.extend(item['id']['videoId'] for item in response['items'])
        page_token = response.get('nextPageToken')
        if not page_token:
            break
        url = f"{url}&pageToken={page_token}"
    return videos

def get_video_details(api_key, video_ids):
    details = {}
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    video_id_chunks = list(chunks(video_ids, 20))
    for chunk in video_id_chunks:
        videos_string = ','.join(chunk)
        url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={videos_string}&key={api_key}"
        response = requests.get(url).json()
        if 'items' not in response:
            print("Error in API response:", response)
            continue
        for item in response['items']:
            video_id = item['id']
            details[video_id] = {
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'publishDate': item['snippet']['publishedAt'],
                'statistics': item['statistics'],
                'tags': item['snippet'].get('tags', []),
                'url': f"https://www.youtube.com/watch?v={video_id}"
            }
    return details

def format_transcript(transcript_data):
    formatted_transcript = []
    for item in transcript_data:
        formatted_transcript.append(f"{item['start']} - {item['text']}")
    return "\n".join(formatted_transcript)

def split_into_chunks(transcript_data, max_tokens):
    chunks = []
    current_chunk = []
    current_length = 0
    for item in transcript_data:
        text = item['text']
        start_time = item['start']
        sentence_length = len(text.split()) + 1
        if current_length + sentence_length > max_tokens:
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0
        current_chunk.append({'text': text, 'start': start_time})
        current_length += sentence_length
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def store_transcript_to_mongodb(video_id, transcript_data, video_details):
    text_chunks = split_into_chunks(transcript_data, MAX_TOKENS)
    for idx, chunk in enumerate(text_chunks):
        formatted_chunk = format_transcript(chunk)
        video_entry = {
            "Title": video_details[video_id]['title'],
            "VideoID": video_id,
            "Tags": video_details[video_id].get('tags', []),
            "ViewCount": video_details[video_id]['statistics'].get('viewCount', 0),
            "LikeCount": video_details[video_id]['statistics'].get('likeCount', 0),
            "CommentCount": video_details[video_id]['statistics'].get('commentCount', 0),
            "URL": video_details[video_id]['url'],
            "Transcript": formatted_chunk,
            "ChunkIndex": idx + 1,
            "Channel_Name": CHANNEL_NAME
        }
        
        # Generate embeddings for the transcript text
        embeddings = embed_model.get_text_embedding(formatted_chunk)
        video_entry["Embedding"] = embeddings
        
        collection.insert_one(video_entry)
        print(f"Stored transcript chunk {idx + 1} with embeddings for video ID: {video_id}")

def fetch_and_store_transcripts(video_ids, video_details):
    for video_id in video_ids:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_data = [{'start': item['start'], 'text': item['text']} for item in transcript]
            store_transcript_to_mongodb(video_id, transcript_data, video_details)
        except Exception as e:
            print(f"Could not fetch transcript for video ID {video_id}: {e}")

if __name__ == "__main__":
    video_ids = get_video_ids(YOUTUBE_API_KEY, CHANNEL_ID)
    #video_ids = video_ids[:2]  # Uncomment to test with first 2 videos
    video_details = get_video_details(YOUTUBE_API_KEY, video_ids)
    fetch_and_store_transcripts(video_ids, video_details)
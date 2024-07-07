import logging
import signal
import sys
import os
import json
from bs4 import BeautifulSoup
from bs4.builder import ParserRejectedMarkup
import re
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, DocumentTooLarge
import hashlib
from datetime import datetime
import psutil
import requests
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI
import traceback

# Configuration parameters
SPIDER_API_KEY = os.environ.get("SPIDER_API_KEY", "sk-95fc79c8-cb4d-4d41-99ca-2b4377d16449")
UPDATE_EXISTING = False
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://EduReaderUser:0PMZhtYpkpdNs59o@edureader.11uhgou.mongodb.net/?retryWrites=true&w=majority&appName=EduReader")
DB_NAME = 'influencers'
COLLECTION_NAME = 'FishingHelper'
RETRY_LIMIT = 3
MAX_DOC_SIZE = 1024 * 1024  # 1MB in bytes

# OpenAI API key configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
influencers_collection = db[COLLECTION_NAME]

interrupted = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the embedding model
embed_model = OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=10, api_key=OPENAI_API_KEY)

# Initialize the OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def signal_handler(sig, frame):
    global interrupted
    interrupted = True
    logger.info("Interrupt received, exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def hash_content(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def generate_tags(content):
    prompt = "Please pull out up to 7 one or two word tags to describe this content. Provide each tag on a new line."
    
    # Truncate content to ensure we don't exceed token limit
    max_content_length = 3800  # Adjust this based on GPT-3.5's token limit
    truncated_content = content[:max_content_length] if len(content) > max_content_length else content
    
    full_prompt = f"{prompt}\n\nContent: {truncated_content}"
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates tags for content."},
                {"role": "user", "content": full_prompt}
            ]
        )
        
        tags = [tag.strip() for tag in response.choices[0].message.content.split('\n') if tag.strip()]
        return tags[:7]  # Ensure we don't exceed 7 tags
    except Exception as e:
        logger.error(f"Error generating tags: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def normalize_url(url):
    url = url.lower()  # Convert to lowercase
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url.rstrip('/')

def document_exists(url, content_hash):
    normalized_url = normalize_url(url)
    return influencers_collection.count_documents({'url': normalized_url, 'content_hash': content_hash}) > 0

def truncate_content(content, max_size):
    content_bytes = content.encode('utf-8')
    if len(content_bytes) > max_size:
        truncated_bytes = content_bytes[:max_size]
        return truncated_bytes.decode('utf-8', errors='ignore')
    return content

def insert_document(url, content, influencer_data):
    content_hash = hash_content(content)
    truncated_content = truncate_content(content, MAX_DOC_SIZE)
    if not document_exists(url, content_hash):
        try:
            # Generate embeddings for the content
            embeddings = embed_model.get_text_embedding(truncated_content)
            
            # Generate tags using GPT-3.5
            tags = generate_tags(truncated_content)
            
            influencers_collection.insert_one({
                'URL': url,
                'Title': url,
                'date_scraped': datetime.now().isoformat(),
                'Transcript': truncated_content,
                'content_hash': content_hash,
                'influencer_data': influencer_data,
                'Embedding': embeddings,
                'Channel_Name' : 'Website',
                'Tags': tags
            })
            logger.info(f"Inserted document with embeddings and tags for URL: {url}")
        except DuplicateKeyError:
            logger.warning(f"Document for URL {url} already exists. Skipping insertion.")
        except DocumentTooLarge:
            logger.error(f"Document for URL {url} is too large even after truncation. Skipping insertion.")
    else:
        logger.info(f"Skipping document insertion for URL {url} as it already exists with the same content.")

def process_url(url):
    normalized_url = normalize_url(url)
    logger.info(f"Processing: {normalized_url}")

    headers = {
        'Authorization': SPIDER_API_KEY,
        'Content-Type': 'application/json',
    }

    retry_count = 0
    while retry_count < RETRY_LIMIT and not interrupted:
        try:
            response = requests.post(
                'https://api.spider.cloud/crawl',
                headers=headers,
                json={"limit": 20, "url": normalized_url},
                stream=True
            )
            response.raise_for_status()
            
            crawl_results = []
            for line in response.iter_lines():
                if line:
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            crawl_results.append(item)
                        elif isinstance(item, list):
                            crawl_results.extend(item)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON from line: {line}")
            
            process_crawl_results(crawl_results, normalized_url)
            break  # Success, exit the retry loop
        except Exception as e:
            logger.error(f"Failed to fetch or process data for {normalized_url}: {e}")
            retry_count += 1
            if retry_count >= RETRY_LIMIT:
                logger.error(f"Max retries reached for {normalized_url}")

    logger.info(f"Finished processing {normalized_url}")

def process_crawl_results(crawl_results, base_url):
    global interrupted
    for item in crawl_results:
        if interrupted:
            return

        if not isinstance(item, dict):
            logger.warning(f"Unexpected item type in crawl results: {type(item)}")
            continue

        html_content = item.get('content')
        if html_content is None:
            logger.warning(f"No content retrieved for {item.get('url')}")
            continue

        specific_page_url = item.get('url')
        if not specific_page_url:
            logger.warning(f"No URL found in crawl result item")
            continue

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
        except ParserRejectedMarkup:
            soup = BeautifulSoup(html_content, 'lxml')  # fallback parser

        # Clean up the content
        tags_to_remove = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']
        classes_or_ids_to_remove = ['menu', 'sidebar', 'ad-section', 'navbar', 'modal', 'footer', 'masthead', 'comment', 'widget']
        for tag in tags_to_remove:
            for element in soup.find_all(tag):
                element.decompose()

        for identifier in classes_or_ids_to_remove:
            for element in soup.find_all(class_=identifier):
                element.decompose()
            for element in soup.find_all(id=identifier):
                element.decompose()

        text = soup.get_text(separator=' ')
        text = re.sub(r'[\r\n]+', '\n', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'&[a-z]+;', '', text)

        influencer_data = extract_influencer_data(soup, specific_page_url)

        try:
            insert_document(specific_page_url, text, influencer_data)
        except DocumentTooLarge:
            logger.error(f"Document for URL {specific_page_url} is too large even after truncation. Skipping insertion.")

def extract_influencer_data(soup, url):
    # Implement your logic to extract influencer data from the soup object
    # This is a placeholder implementation
    return {
        'name': soup.find('h1', class_='influencer-name').text.strip() if soup.find('h1', class_='influencer-name') else '',
        'followers': soup.find('span', class_='follower-count').text.strip() if soup.find('span', class_='follower-count') else '',
        'bio': soup.find('div', class_='bio').text.strip() if soup.find('div', class_='bio') else '',
        # Add more fields as needed
    }

def main():
    with open('pafishing.txt', 'r') as file:
        urls = file.read().splitlines()
    
    urls.reverse()

    for url in urls:
        if interrupted:
            break
        process_url(url)

    logger.info("Data scraping and storage process completed successfully.")

    # Log resource usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory Usage: RSS = {memory_info.rss / (1024 ** 2)} MB")

if __name__ == "__main__":
    main()
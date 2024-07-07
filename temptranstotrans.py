from pymongo import MongoClient
import os

# MongoDB connection details
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://EduReaderUser:0PMZhtYpkpdNs59o@edureader.11uhgou.mongodb.net/?retryWrites=true&w=majority&appName=EduReader")
DB_NAME = 'influencers'
COLLECTION_NAME = 'FishingHelper'

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

sample_doc = collection.find_one()
print(sample_doc.keys())

print(collection.index_information())

def update_field_name():
    # Find all documents that have a 'content' field
    docs_to_update = collection.find({"Channel_Name": {"$exists": False}})
    
    update_count = 0
    for doc in docs_to_update:
        # Rename 'content' to 'transcript'
        result = collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"Channel_Name": "Website"}}
        )
        if result.modified_count > 0:
            update_count += 1
    
    print(f"Updated {update_count} documents.")

def copy_url_to_title():
    # Find all documents that have a 'URL' field but no 'Title' field
    docs_to_update = collection.find({"URL": {"$exists": True}, "Title": {"$exists": False}})
    
    update_count = 0
    for doc in docs_to_update:
        # Copy 'URL' to 'Title'
        result = collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"Title": doc["URL"]}}
        )
        if result.modified_count > 0:
            update_count += 1
    
    print(f"Copied URL to Title for {update_count} documents.")

if __name__ == "__main__":
    update_field_name()
    copy_url_to_title()
    print("Field name update and URL copy process completed.")
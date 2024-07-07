from pymongo import MongoClient
import os

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client['influencers']
collection = db['MakeUpHelper']

# Get distinct Channel_Name values
distinct_channels = collection.distinct("Channel_Name")

# Print each distinct channel name
print("Distinct Channel Names:")
for channel in distinct_channels:
    print(channel)

# Print the total count
print(f"Total number of distinct channels: {len(distinct_channels)}")

# Close the connection
client.close()
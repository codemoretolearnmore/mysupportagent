import os
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")  
DB_NAME = os.getenv("DB_NAME", "ticket_classification")  

# Required collections can be passed as a comma-separated list
REQUIRED_COLLECTIONS = os.getenv("REQUIRED_COLLECTIONS", "").split(",")

# Establish connection
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
db.command("ping")

def get_db():
    return db

def get_mongo_collection(collection_name):
    """Returns the MongoDB collection. If missing, it is created."""
    if collection_name and collection_name not in db.list_collection_names():
        print(f"⚠️ Collection '{collection_name}' not found. Creating it now...")
        db.create_collection(collection_name)
    return db[collection_name]

# Ensure all required collections exist on startup
for collection in REQUIRED_COLLECTIONS:
    if collection.strip():
        get_mongo_collection(collection.strip())

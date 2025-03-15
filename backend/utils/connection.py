from pymongo import MongoClient
import json


with open("config.json", "r") as config_file:
    config = json.load(config_file)

MONGO_URI = config.get("MONGO_URI", "mongodb://localhost:27017")  # Default if missing
DB_NAME = config.get("DB_NAME", "support_tickets")
REQUIRED_COLLECTIONS = config.get("required_collections", [])

# Establish connection

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
db.command("ping")

def get_db():
    return db

def get_mongo_collection(collection_name):
    """
    Returns the MongoDB collection.
    If the collection doesn't exist, it is automatically created.
    """
    if collection_name not in db.list_collection_names():
        print(f"⚠️ Collection '{collection_name}' not found. Creating it now...")
        db.create_collection(collection_name)
    return db[collection_name]

# Ensure all required collections exist on server startup
for collection in REQUIRED_COLLECTIONS:
    get_mongo_collection(collection)

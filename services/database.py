from utils.connection import get_mongo_collection
from uuid import uuid4
from datetime import datetime, timezone
from pymongo.bson import ObjectId

def save_tickets(tickets):
    """Save multiple tickets to MongoDB."""
    collection = get_mongo_collection("tickets")
    for ticket in tickets:
        collection.update_one(
            {"ticket_id": ticket["ticket_id"]}, 
            {"$set": ticket}, 
            upsert=True
        )

def get_tickets():
    """Fetch all tickets from MongoDB."""
    collection = get_mongo_collection("tickets")
    return list(collection.find({}, {"_id": 0}))

def get_ticket_by_id(ticket_id):
    """Fetch a single ticket from MongoDB."""
    collection = get_mongo_collection("tickets")
    return collection.find_one({"ticket_id": ticket_id}, {"_id": 0})

def save_user_edit(data):
    """Update a ticket with user-edited category."""
    collection = get_mongo_collection("tickets")
    collection.update_one(
        {"ticket_id": data["ticket_id"]},
        {"$set": {
            "classified_category": data["new_category"],
            "confidence_score": 1.0,
            "mode_of_tagging": "manual_edit"
        }}
    )

def save_chatgpt_trained_tickets(tickets):
    """Save ChatGPT classified tickets to a separate collection and main database."""
    collection = get_mongo_collection("chatgpt_trained_tickets")
    main_collection = get_mongo_collection("tickets")
    
    for ticket in tickets:
        collection.update_one({"ticket_id": ticket["ticket_id"]}, {"$set": ticket}, upsert=True)
        main_collection.update_one({"ticket_id": ticket["ticket_id"]}, {"$set": ticket}, upsert=True)

def log_training(data):
    """Log model training history."""
    collection = get_mongo_collection("training_logs")
    collection.insert_one(data)

async def createClassificationJob(logger):
    job_id = str(uuid4())
    request_id = logger.extra['request_id']
    jobs_collection = get_mongo_collection("jobs")
    jobs_collection.insert_one(
        {"job_id":job_id, 
         "request_id":request_id, 
         "Status":"IN_PROGRESS",
         "createdAt":datetime.now(timezone.utc).isoformat(),
        "updatedAt":datetime.now(timezone.utc).isoformat()
        })
    return job_id

async def checkClassificationTaskStatus(logger, job_id):
    jobs_collection = get_mongo_collection("jobs")
    job = jobs_collection.find_one({"job_id":job_id})
    if not job:
        return None
    return {"message":"Fetched Job Status", "status":job['Status']}

async def getClassificationResults(job_id, logger):
    classification_collection = get_mongo_collection("tickets")
    result =  classification_collection.find({"job_id":job_id},{"vectorized_data":0})
    print(result)
    result = [serialize_mongo_document(doc) for doc in result]

    logger.info("Retrieved Classified Tickets")
    return result

def serialize_mongo_document(doc):
    """Converts MongoDB ObjectId to a string for JSON serialization."""
    return {
        **doc,
        "_id": str(doc["_id"]) if "_id" in doc else None  # Convert ObjectId to string explicitly
    }

async def updateCategoryClassification(data, logger):
    unique_id = data['id']
    updatedCategory = data['category']
    classification_collection = get_mongo_collection("tickets")
    updated_time = datetime.now(timezone.utc).isoformat()
    result = classification_collection.update_one({"_id":ObjectId(unique_id)},{"$set":{"classification_category":updatedCategory, "updatedAt":updated_time, "confidence_score":1}})
    print(result)
    return result

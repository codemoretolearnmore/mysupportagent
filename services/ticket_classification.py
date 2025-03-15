import json
from fastapi.responses import JSONResponse
import pickle
from utils.connection import get_mongo_collection
import numpy as np
from utils.embeddings import generate_embedding
from services.similar_ticket import get_most_similar_tickets
from datetime import datetime, timezone


tickets_collection = get_mongo_collection("tickets")
jobs_collection = get_mongo_collection("jobs")

def fetch_all_classified_tickets_from_db():
    """
    Fetch both LLM-classified and locally-classified tickets with their vector embeddings.
    """
    # Fetch LLM-classified tickets
    llm_collection = get_mongo_collection("classified_tickets")
    llm_tickets = list(llm_collection.find({}, {"vectorized_data": 1, "classification_category": 1, "_id": 0}))

    # Fetch locally classified tickets
    local_tickets = list(tickets_collection.find({}, {"vectorized_data": 1, "classification_category": 1, "_id": 0}))

    # Combine both datasets
    all_tickets = llm_tickets + local_tickets
    return all_tickets


async def classify_tickets(job_id, tickets, logger):
    try:
        classified_tickets = []
        stored_tickets = fetch_all_classified_tickets_from_db()
        for ticket in tickets:
            # Generate vector embedding for the ticket description
            vector_embedding = generate_embedding(ticket['description']+" in "+ticket['product'])

            # Classify the ticket using the trained model
            classification_result = await classify_ticket(vector_embedding, stored_tickets)
    
            if classification_result:
                classified_tickets.append({
                    "ticket_id": ticket['ticket_id'],
                    "description": ticket['description'],
                    "product": ticket['product'],
                    "created_date": ticket['created_date'],
                    "classification_category": classification_result["predicted_category"],
                    "confidence_score": classification_result["confidence_score"],
                    "vectorized_data": vector_embedding,  
                    "mode_of_tagging": "local_model",
                    "job_id":job_id,
                    "createdAt":datetime.now(timezone.utc).isoformat(),
                    "updatedAt":datetime.now(timezone.utc).isoformat()
                })
            else:
                print(f"Failed to classify ticket ID: {ticket['ticket_id']}")
        logger.info("Ticket classification completed")
        # Save classified tickets to MongoDB
        if classified_tickets!=[]:
            tickets_collection.insert_many(classified_tickets)
            jobs_collection.update_one({"job_id":job_id},{"$set":{"Status":"COMPLETED"}})
            logger.info("Saved tickets in tickets collection after classification")
            for ticket in classified_tickets:
                if "_id" in ticket:
                    ticket["_id"] = str(ticket["_id"])
            return JSONResponse(
                status_code=201,
                content={"message":"Classified ticket successfully", "classified_tickets":classified_tickets}
            )
        else:
            
            return JSONResponse(
                status_code="200",
                content={"message":"tickets classified", "classified_tickets":[]}
            )
    except Exception as e:
        logger.error("Error occured in ticket classification" + str(e))
        jobs_collection.update_one({"job_id":job_id},{"$set":{"Status":"FAILED"}})
        return JSONResponse(
            status_code=400,
            content={"message":"Ticket Classification failed", "classified_tickets":[]}
        )


    

async def classify_ticket(vector_embedding, stored_tickets):
    threshold=0.9
    try:
        with open("models/trained_model.pkl", "rb") as f:
            model = pickle.load(f)

        # Get predicted category
        # look for similar tickets first
        similar_tickets = get_most_similar_tickets(vector_embedding, stored_tickets)
        high_confidence_tickets = [t[0] for t in similar_tickets if t[1] >= threshold]
        # if similar tickets are found, return most similar ticket's category and cofidence score
        # if high_confidence_tickets:
        #     # Majority vote from similar tickets
        #     categories = [ticket["classification_category"] for ticket in high_confidence_tickets]
        #     predicted_category= max(set(categories), key=categories.count)
        #     confidence_score = categories.count(predicted_category) / len(categories)

        # use locally trained model
        predicted_category = model.predict([vector_embedding])[0]

        # Get confidence score using probability prediction
        if hasattr(model, "predict_proba"):  # Check if model supports probability estimation
            probabilities = model.predict_proba([vector_embedding])
            confidence_score = round(max(probabilities[0]), 4)  # Highest probability
        else:
            confidence_score = 0.95  # Default confidence if model doesn't support `predict_proba`


        return {
            "predicted_category": predicted_category,
            "confidence_score": confidence_score
        }

    except Exception as e:
        print(f"Error in classification: {e}")
        return None



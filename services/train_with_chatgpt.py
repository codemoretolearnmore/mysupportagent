from openai import OpenAI
from datetime import datetime, timezone
import json
from services.database import get_mongo_collection
from utils.clustering import cluster_tickets
from utils.embeddings import generate_embedding  # Assuming this function generates vector embeddings
from fastapi.responses import JSONResponse
import os

# Load config
API_KEY = os.getenv("API_KEY", "")  


async def train_with_chatgpt(tickets, logger):
    
    try:
        logger.info("labelling of tickets by llm model started")
        tickets = [ticket.dict() for ticket in tickets]
        clustered_tickets = await cluster_tickets(tickets, logger)
        responses = []
        
        with open("metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        # Step 2: Process each cluster
        for cluster in clustered_tickets:
            representative_ticket = cluster[0]  # Pick one ticket from cluster
            ticket_text = representative_ticket['description'] + " in " + representative_ticket['product']
            metadata_string = json.dumps(metadata, indent=2)
            # Step 3: Call ChatGPT API for classification
            prompt = f"""
        You are a support ticket classification expert. Your job is to analyze customer complaints and tag those tickets to the feature mentioned in below json metadata of feature and related potential issue that can occur. Ticket description is entered by human, so spelling mistake can happen.
        For the given ticket description, return:
        1. The most accurate **category** for classification by reading metadata pasted below.
        2. A **confidence score** between 0 to 1 (e.g., 0.85 for 85% confidence), give confidence score based on your confidence, not just some random number.
        This is the metadata of feature and description of potential issue that user can face while using product. Metadata: {metadata_string}
        Please return the response in strict JSON format:
        {{
            "category": "Best category name",
            "confidence_score": 0.0 to 1.0
        }}

        If ticket doesn't seem to belong to any of pre-defined category, tag it as Others, with low confidence score.

        Ticket Description:
        "{ticket_text}"
        """
            
            client = OpenAI(
                api_key=config["API_KEY"],  # This is the default and can be omitted
            )
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4"
            )
            response_data = json.loads(response.choices[0].message.content)
            classified_category = response_data.get("category", "Unknown")
            confidence_score = response_data.get("confidence_score", 0.0)
            
            # Step 4: Assign classification to all tickets in the cluster
            for ticket in cluster:
                text = ticket['description']+" in " + ticket['product']
                vectorized_data = generate_embedding(text)  # Generate vector embeddings

                ticket["classified_category"] = classified_category
                ticket["confidence_score"] = confidence_score  # Assuming ChatGPT confidence
                ticket["mode_of_tagging"] = "chatgpt"
                ticket["vectorized_data"] = vectorized_data

                # Step 5: Save to MongoDB
                collection = get_mongo_collection("classified_tickets")
                
                collection.insert_one(
                    {
                        "ticket_id":ticket['ticket_id'],
                        "ticket_description":ticket['description'],
                        "product":ticket['product'],
                        "created_date": ticket["created_date"],
                        "classification_category": classified_category,
                        "confidence_score":confidence_score,
                        "mode_of_tagging":"chatgpt",
                        "vectorized_data":vectorized_data,
                        "createdAt":datetime.now(timezone.utc).isoformat(),
                        "updatedAt":datetime.now(timezone.utc).isoformat()
                    }
                )
            
            responses.extend(cluster)
        logger.info("Ticket Classification done and saved in classified_ticket collection")
        return JSONResponse(
            status_code=201,
            content={"message":"Ticket classified with LLM Model", "classified_tickets":responses}
        )
    
    except Exception as e:
        
        logger.error("Error occured in classification from LLM model" + str(e))
        return JSONResponse(
            status_code=500,
            content={"message":f"Ticket classification by LLM Model failed due to {str(e)}", "classified_tickets":[]}
        )

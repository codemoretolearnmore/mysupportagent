import joblib  # Replacing pickle with joblib
from datetime import datetime, timezone
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fastapi.responses import JSONResponse, Response
from utils.connection import get_mongo_collection
import os

local_model_classification_collection = get_mongo_collection("tickets")
training_logs_collection = get_mongo_collection("training_logs")
MODEL_PATH = "models/trained_model.pkl"

def get_last_training_time():
    """Fetch the last training timestamp from logs."""
    last_log = training_logs_collection.find_one({}, {"timestamp": 1}, sort=[("timestamp", -1)])
    return last_log["timestamp"] if last_log else None

def train_model(logger):
    try:
        """Train the local model using only new tickets since the last training."""
        last_training_time = get_last_training_time()
        llm_collection = get_mongo_collection("classified_tickets")

        # Fetch new tickets since last training
        if last_training_time is None:
            llm_labeled_tickets = list(llm_collection.find({}, {"vectorized_data": 1, "classification_category": 1, "_id": 0}))
            local_model_feedback_tickets = list(local_model_classification_collection.find(
                {"$expr": {"$ne": ["$createdAt", "$updatedAt"]}},  
                {"vectorized_data": 1, "classification_category": 1, "_id": 0}
            ))
        else:
            llm_labeled_tickets = list(llm_collection.find(
                {"updatedAt": {"$gt": last_training_time}}, 
                {"vectorized_data": 1, "classification_category": 1, "_id": 0}
            ))
            local_model_feedback_tickets = list(local_model_classification_collection.find(
                {
                    "$and": [
                        {"$expr": {"$ne": ["$createdAt", "$updatedAt"]}},  
                        {"updatedAt": {"$gt": last_training_time}}  
                    ]
                }, 
                {"vectorized_data": 1, "classification_category": 1, "_id": 0}
            ))

        all_local_model_feedback_tickets = llm_labeled_tickets + local_model_feedback_tickets
        if not all_local_model_feedback_tickets:
            logger.warning("No new ticket for training")
            return Response(status_code=204)

        logger.info("Fetched all new tickets for training")

        # ✅ Ensure vectorized data is converted properly to NumPy arrays
        X = np.array([np.array(ticket["vectorized_data"], dtype=np.float32) for ticket in all_local_model_feedback_tickets])
        y = np.array([ticket["classification_category"] for ticket in all_local_model_feedback_tickets])

        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)  
            logger.info("Loaded existing model for incremental training.")
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)  
            logger.info("No existing model found. Initializing a new one.")

        # Split dataset and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        logger.info("Model training completed")

        # Evaluate accuracy
        y_pred = model.predict(X_test)
        model_accuracy = accuracy_score(y_test, y_pred)

        # Save model using joblib
        joblib.dump(model, MODEL_PATH)  
        logger.info("Saved trained model successfully")

        # Log training details
        training_logs_collection.insert_one({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_tickets": len(all_local_model_feedback_tickets),
            "method": "Local Model",
            "accuracy": model_accuracy
        })
        
        logger.info("Saved training logs")

        return JSONResponse(
            status_code=200,
            content={"message": "Model Trained Successfully", "accuracy": model_accuracy}
        )
    
    except Exception as e:
        logger.error(f"Error occurred while training model: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": "Model Training failed", "error": str(e)}
        )

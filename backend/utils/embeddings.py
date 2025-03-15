from sentence_transformers import SentenceTransformer
import numpy as np
from services.database import get_mongo_collection
import hashlib

# Load a local transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Small & fast model

def generate_embedding(text):
    
    try:
        collection = get_mongo_collection("embeddings_cache")

        # Hash the text for caching
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cached_embedding = collection.find_one({"hash": text_hash})

        if cached_embedding:
            return cached_embedding["embedding"]

        # Generate the embedding
        embedding = model.encode(text).tolist()

        # Store in DB cache
        collection.insert_one({"hash": text_hash, "embedding": embedding})

        return embedding

    except Exception as e:
        print(f"Embedding generation failed: {e}")
        return np.zeros(384).tolist()  # Returns zero vector (model size: 384)


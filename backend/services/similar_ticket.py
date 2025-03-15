from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_most_similar_tickets(vector_embedding, stored_tickets, top_k=5):
    """
    Retrieve the most similar past tickets using cosine similarity.
    """
    embeddings = np.array([ticket["vectorized_data"] for ticket in stored_tickets])
    similarities = cosine_similarity([vector_embedding], embeddings)[0]

    # Get top-k similar tickets
    top_indices = similarities.argsort()[-top_k:][::-1]
    similar_tickets = [(stored_tickets[i], similarities[i]) for i in top_indices]
    return similar_tickets
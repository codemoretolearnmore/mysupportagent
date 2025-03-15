from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

async def cluster_tickets(tickets, logger, max_clusters=5):
    logger.info("Ticket clustering Started")
    descriptions = [ticket["description"] for ticket in tickets]
    
    # Load MiniLM model for embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(descriptions, convert_to_numpy=True)
    
    # Avoid more clusters than data points
    max_clusters = min(max_clusters, len(tickets))
    
    # Find optimal clusters using silhouette score
    best_k = 2  # Minimum 2 clusters
    best_score = -1
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        
        if score > best_score:
            best_score = score
            best_k = k
    
    # Apply K-Means with the best number of clusters
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Organize tickets into clusters
    clusters = {i: [] for i in range(best_k)}
    for i, label in enumerate(labels):
        clusters[label].append(tickets[i])
    
    logger.info("Ticket clustering done")
    return list(clusters.values())

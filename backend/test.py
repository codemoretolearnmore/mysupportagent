# from utils.clustering import cluster_tickets
import asyncio  # Add this
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Run the async function properly

tickets = [
  {
    "created_date": "2024-07-01 0:00:00",
    "ticket_id": 660328,
    "product": "TDS",
    "description": "FUV - ERROR FILE"
  },
  {
    "created_date": "2024-07-01 0:00:00",
    "ticket_id": 660421,
    "product": "TDS",
    "description": "Error in Extracting Report from Income Tax Portal"
  },
  {
    "created_date": "2024-07-01 0:00:00",
    "ticket_id": 660411,
    "product": "TDS",
    "description": "Facing issue from the FUV generated for filing correction return"
  },
  {
    "created_date": "2024-07-01 0:00:00",
    "ticket_id": 660427,
    "product": "TDS",
    "description": "error in generating fvu for correction return"
  },
  {
    "created_date": "2024-07-01 0:00:00",
    "ticket_id": 660585,
    "product": "TDS",
    "description": "Error in Revised 24Q * Global Jewellery Pvt Ltd * FY 23-24"
  },
  {
    "created_date": "2024-07-01 0:00:00",
    "ticket_id": 660836,
    "product": "TDS",
    "description": "i am not getting the issues regarding the correction return"
  },
  {
    "created_date": "2024-07-01 0:00:00",
    "ticket_id": 660987,
    "product": "TDS",
    "description": "Not able to generate the fvu file"
  }
]

async def optimal_kmeans_clustering(tickets, max_clusters=5):
    descriptions = [ticket["description"] for ticket in tickets]
    
    # Convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    # Avoid more clusters than data points
    max_clusters = min(max_clusters, len(tickets))

    # Find optimal clusters using silhouette score
    best_k = 2  # Minimum 2 clusters
    best_score = -1
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, labels)
        
        if score > best_score:
            best_score = score
            best_k = k

    # Apply K-Means with the best number of clusters
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(tfidf_matrix)

    # Organize tickets into clusters
    clusters = {i: [] for i in range(best_k)}
    for i, label in enumerate(labels):
        clusters[label].append(tickets[i])
    
    return list(clusters.values())

# Test function
async def finalFunction(tickets):
    clustered_tickets = await optimal_kmeans_clustering(tickets)
    for i, cluster in enumerate(clustered_tickets):
        print(f"\n🔹 Cluster {i} ({len(cluster)} tickets)")
        for ticket in cluster:
            print(f" - {ticket['description']}")
asyncio.run(finalFunction(tickets))

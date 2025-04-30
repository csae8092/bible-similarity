import chromadb
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Step 1: Connect to ChromaDB and get embeddings
chroma_client = chromadb.PersistentClient(path="chroma_data")
collection = chroma_client.get_collection(name="passages_openai")

# Get all items from collection with embeddings and documents
results = collection.get(
    include=["embeddings", "documents"]
)
embeddings = np.array(results["embeddings"])

# Step 2: Create DataFrame
df = pd.DataFrame({
    "document": results["documents"]
})
df["embedding"] = list(embeddings)

# Step 3: Cluster with KMeans
kmeans = KMeans(n_clusters=10, random_state=42)
df["cluster"] = kmeans.fit_predict(embeddings)

# Step 4: Apply t-SNE (can take a few seconds)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_results = tsne.fit_transform(embeddings)
df["tsne_x"] = tsne_results[:, 0]
df["tsne_y"] = tsne_results[:, 1]

# Step 5: Plot
plt.figure(figsize=(10, 7))
for cluster_id in sorted(df["cluster"].unique()):
    cluster_data = df[df["cluster"] == cluster_id]
    plt.scatter(cluster_data["tsne_x"], cluster_data["tsne_y"], label=f"Cluster {cluster_id}", alpha=0.7)

plt.title("t-SNE of ChromaDB Document Embeddings")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('cluster_plot.png', dpi=300, bbox_inches='tight')
plt.close()

import os
import chromadb
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
import plotly.express as px

chroma_client = chromadb.PersistentClient(path="chroma_data")
collection = chroma_client.get_collection(name="passages_default")
results = collection.get(include=["embeddings", "documents", "metadatas"])
df = pd.DataFrame(
    {
        "jad_id": [x["jad_id"] for x in results["metadatas"]],
        "document": results["documents"],
        "author": [x["author"] for x in results["metadatas"]],
        "work": [x["work"] for x in results["metadatas"]],
    }
)
df["embeddings"] = list(np.array(results["embeddings"]))
kmeans = KMeans(n_clusters=20, random_state=42)
kmeans.fit(np.stack(df["embeddings"]))
df["cluster"] = kmeans.labels_

# Reduce dimensionality to 2D for visualization
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(np.stack(df["embeddings"]))

# Create scatter plot using Plotly
fig = px.scatter(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    color=df["cluster"].astype(str),
    hover_data={
        "jad_id": df["jad_id"],
        "author": df["author"],
        "work": df["work"]
    },
    labels={
        "x": "First Principal Component",
        "y": "Second Principal Component",
        "color": "Cluster"
    },
    title="Clusters of Passages"
)

# Update layout for better visualization
fig.update_layout(
    plot_bgcolor='white',
    width=1000,
    height=800
)

# Save as interactive HTML file
fig.write_html("clusters_interactive.html")

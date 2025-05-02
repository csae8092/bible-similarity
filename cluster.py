import chromadb
import pandas as pd
import numpy as np
import umap
from sklearn.cluster import KMeans
import plotly.express as px

# inspired by https://programminghistorian.org/en/lessons/clustering-visualizing-word-embeddings


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

dmeasure = "euclidean"  # distance metric
rdims = 4  # r-dims == Reduced dimensionality
cluster_nr = 10


# Extract the embedding from a list-type column
# in the source data frame using this function
def x_from_df(df: pd.DataFrame, col: str = "embeddings") -> pd.DataFrame:
    cols = ["E" + str(x) for x in np.arange(0, len(df[col].iloc[0]))]
    return pd.DataFrame(df[col].tolist(), columns=cols, index=df.index)


X = x_from_df(df, col="embeddings")

reducer = umap.UMAP(n_neighbors=25, min_dist=0.01, n_components=rdims, random_state=43)
X_embedded = reducer.fit_transform(X)

# Create a dictionary that is easily converted into a pandas df
embedded_dict = {}
for i in range(0, X_embedded.shape[1]):
    embedded_dict[f"Dim {i + 1}"] = X_embedded[:, i]  # D{dimension_num} (Dim 1...Dim n)

dfe = pd.DataFrame(embedded_dict, index=df.index)
del embedded_dict

dfe["umap"] = dfe[["Dim 1", "Dim 2", "Dim 3", "Dim 4"]].values.tolist()

projected = df.join(dfe).sort_values(by="author")

kmeans = KMeans(n_clusters=cluster_nr, random_state=42)
kmeans.fit(np.stack(projected["embeddings"]))
projected["cluster"] = kmeans.labels_

projected_sorted = projected.sort_values("cluster")

fig = px.scatter(
    x=projected_sorted["Dim 1"],
    y=projected_sorted["Dim 2"],
    color=projected_sorted["cluster"].astype(str),
    hover_data={
        "jad_id": projected_sorted["jad_id"],
        "author": projected_sorted["author"],
        "work": projected_sorted["work"],
    },
    labels={
        "x": "x",
        "y": "y",
        "color": "Cluster",
    },
    title="Clusters of Passages",
)

# Update layout for better visualization
fig.update_layout(plot_bgcolor="white", width=1000, height=800)

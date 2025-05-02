import chromadb
import pandas as pd

# Initialize Chroma client and collection
client = chromadb.PersistentClient(path="chroma_data")
collection = client.get_collection(name="passages_default")

# Fetch all documents
data = collection.get(include=["embeddings", "documents", "metadatas"])
ids = data["ids"]
embeddings = data["embeddings"]
documents = data["documents"]
metadatas = data["metadatas"]

# Store results
rows = []

for i in range(len(ids)):
    pid = ids[i]
    vector = embeddings[i]
    doc_text = documents[i]
    meta = metadatas[i]

    # Query similar passages (include self in result, exclude later)
    results = collection.query(
        query_embeddings=[vector],
        n_results=4,  # 3 similar + 1 (the document itself)
        include=["documents", "metadatas", "distances"]  # changed from "ids" to valid parameters
    )

    # Get the IDs from metadatas since they're stored there
    similar_ids = [item["jad_id"] for item in results["metadatas"][0]][1:]
    distances = results["distances"][0][1:]

    # Add row
    rows.append({
        "id": meta["jad_id"],
        "author": meta["author"],
        "work": meta["work"],
        "similar_ids": similar_ids,
        "distances": distances
    })

# Create DataFrame
df = pd.DataFrame(rows)
df.to_csv("similar_texts.csv", index=False)

# Optional: inspect
print(df.head())
print(results)
from collections import defaultdict

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

# count matches
data = defaultdict(list)
for i, row in df.iterrows():
    jad_id = row["id"]
    sim_text = row["similar_ids"]
    for x in sim_text:
        data[x].append(jad_id)

new_data = {}
for key, value in data.items():
    new_data[key] = {
        "items": value,
        "n": len(value)
    }
df['match_count'] = df['id'].map(lambda x: new_data.get(x, {'n': 0})['n'])
df_sorted = df.sort_values(["match_count"], ascending=False)
df_sorted.to_csv("similar_texts.csv", index=False)

# Optional: inspect
print(df_sorted.head())

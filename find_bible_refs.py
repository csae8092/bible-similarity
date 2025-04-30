import os
import json
import requests

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from tqdm import tqdm


API_KEY = os.environ.get("OPENAI_API_KEY")
emb_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=API_KEY, model_name="text-embedding-3-small"
)

url = "https://raw.githubusercontent.com/jerusalem-70-ad/jad-ai/refs/heads/main/out/all_in_one.json"

data = requests.get(url).json()


chroma_db = os.path.join("chroma_data")
chroma_client = chromadb.PersistentClient(path=chroma_db)
collection = chroma_client.get_collection(
    name="vulgata_openai", embedding_function=emb_fn
)

for key, value in tqdm(data.items()):
    for x in value:
        try:
            text = x["text"]
        except TypeError:
            continue
        results = collection.query(query_texts=[text], n_results=1)
        x["match"] = {
            "id": results["ids"][0][0],
            "bible_text": results["documents"][0][0],
            "score": results["distances"][0][0],
        }

with open("result.json", "w", encoding="utf-8") as fp:
    json.dump(data, fp, ensure_ascii=False, indent=2)

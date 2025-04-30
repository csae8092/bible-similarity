import os
import json

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from csae_pyutils import load_json, save_json
from tqdm import tqdm


API_KEY = os.environ.get("OPENAI_API_KEY")
emb_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=API_KEY, model_name="text-embedding-3-small"
)

chroma_db = os.path.join("chroma_data")
chroma_client = chromadb.PersistentClient(path=chroma_db)
collection = chroma_client.get_collection(
    name="passages_openai", embedding_function=emb_fn
)
url = "https://raw.githubusercontent.com/jerusalem-70-ad/jad-baserow-dump/refs/heads/main/json_dumps/occurrences.json"
SRC_DATA = "passages.json"
if os.path.exists(SRC_DATA):
    data = load_json(SRC_DATA)
else:
    data = load_json(url)
    save_json(data, SRC_DATA)

items = list(data.values())
data = []

print("no need to vectorize texts again because their vectors are already stored in the db")

# for x in tqdm(items):
#     if x["text_paragraph"]:
#         text = x["text_paragraph"]
#         results = collection.query(query_texts=[text], n_results=2)
#         item = {
#             "jad_id": x["jad_id"],
#             "org_text": text
#         }
#         item["match"] = {
#             "id": results["ids"][0][1],
#             "matching_text": results["documents"][0][1],
#             "score": results["distances"][0][1],
#         }
#         data.append(item)

# with open("result_passage.json", "w", encoding="utf-8") as fp:
#     json.dump(data, fp, ensure_ascii=False, indent=2)

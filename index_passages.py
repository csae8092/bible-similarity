import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from tqdm import tqdm
from csae_pyutils import load_json, save_json


API_KEY = os.environ.get("OPENAI_API_KEY")
emb_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=API_KEY, model_name="text-embedding-3-small"
)

chroma_db = os.path.join("chroma_data")
chroma_client = chromadb.PersistentClient(path=chroma_db)
collection = chroma_client.get_or_create_collection(name="passages_openai", embedding_function=emb_fn)

SRC_DATA = "passages.json"

url = "https://raw.githubusercontent.com/jerusalem-70-ad/jad-baserow-dump/refs/heads/main/json_dumps/occurrences.json"
if os.path.exists(SRC_DATA):
    data = load_json(SRC_DATA)
else:
    data = load_json(url)
    save_json(data, SRC_DATA)


docs = []
ids = []
for key, x in data.items():
    if x["text_paragraph"]:
        docs.append(x["text_paragraph"])
        ids.append(key)

# Add documents with progress bar
batch_size = 20
for i in tqdm(range(0, len(docs), batch_size)):
    batch_docs = docs[i:i + batch_size]
    batch_ids = ids[i:i + batch_size]
    try:
        collection.add(
            documents=batch_docs,
            ids=batch_ids
        )
    except:  # noqa:
        continue
print("done")

import os
import pandas as pd
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from tqdm import tqdm

API_KEY = os.environ.get("OPENAI_API_KEY")

emb_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=API_KEY, model_name="text-embedding-3-small"
)

source_file = os.path.join("data", "vul.tsv")
df = pd.read_csv(
    source_file, sep="\t", names=["book", "abbr", "chapter", "verse", "line", "text"]
)
df["ids"] = df[["abbr", "chapter", "verse", "line"]].astype(str).apply("_".join, axis=1)
docs = list(df["text"].values)
ids = list(df["ids"].values)

chroma_db = os.path.join("chroma_data")
os.makedirs(chroma_db, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=chroma_db)
collection = chroma_client.get_or_create_collection(
    name="vulgata_openai", embedding_function=emb_fn
)

# Add documents with progress bar
batch_size = 100
for i in tqdm(range(0, len(docs), batch_size)):
    batch_docs = docs[i : i + batch_size]
    batch_ids = ids[i : i + batch_size]
    collection.add(documents=batch_docs, ids=batch_ids)

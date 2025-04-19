import os
import json
import requests
import spacy
import re
import chromadb
from collections import defaultdict
from tqdm import tqdm

chroma_db = os.path.join("chroma_data")
chroma_client = chromadb.PersistentClient(path=chroma_db)
collection = chroma_client.get_collection(name="vulgata")

nlp = spacy.load("en_core_web_sm")
page_pattern = re.compile(r"p\.\s*\d+[A-D]\s*\|")
parentheses_pattern = re.compile(r'\([^)]*\)')

url = "https://raw.githubusercontent.com/jerusalem-70-ad/jad-baserow-dump/refs/heads/main/json_dumps/occurrences.json"
data = requests.get(url).json()
if os.path.exists("passages.json"):
    pass
else:
    with open("passages.json", "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


d = defaultdict(list)
for key, value in tqdm(data.items()):
    occ_id = key
    text = value["text_paragraph"]
    if text:
        # Remove page markers and parentheses content before processing
        cleaned_text = page_pattern.sub('', text)
        cleaned_text = parentheses_pattern.sub('', cleaned_text)
        cleaned_text = cleaned_text.replace("«", '').replace("»", '').replace("|", "")
        doc = nlp(cleaned_text)
        for sent in doc.sents:
            cur_text = sent.text
            item = {
                "text": cur_text
            }
            results = collection.query(query_texts=[cur_text], n_results=1)
            item["match"] = {
                "id": results["ids"][0][0],
                "bible_text": results["documents"][0][0],
                "score": results["distances"][0][0]
            }
            d[occ_id].append(item)


with open("sentences.json", "w", encoding="utf-8") as fp:
    json.dump(d, fp, ensure_ascii=False, indent=2)

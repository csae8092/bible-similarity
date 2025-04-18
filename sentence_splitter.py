import os
import json
import requests
import spacy
import re

from collections import defaultdict
from tqdm import tqdm

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
            d[occ_id].append(cur_text.strip())

with open("sentences.json", "w", encoding="utf-8") as fp:
    json.dump(d, fp, ensure_ascii=False, indent=2)

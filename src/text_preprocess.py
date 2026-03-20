import json
import re
import argparse
from collections import Counter
from nltk.stem import PorterStemmer
import os
stemmer = PorterStemmer()


def tokenize(text):
    """Lowercase, extract words via regex, stem each token."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [stemmer.stem(t) for t in tokens]


def build_documents(filepath):
    documents = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents

def build_document_term_matrix(files):
    pass



def main():
    DATASET_FOLDER = "dataset"
    OUTPUT_FILE = "src/data/term_doc_matrix.json"

    team_files = {}
    for filename in os.listdir(DATASET_FOLDER):
        if filename.endswith(".jsonl"):
            team_name = filename.replace("-utterances.jsonl", "").replace("-", " ").title()
            filepath = os.path.join(DATASET_FOLDER, filename)
            team_files[team_name] = filepath

    result = build_term_document_matrix(team_files)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)



if __name__ == "__main__":
    main()
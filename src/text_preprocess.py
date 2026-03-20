import json
import re
import os


def tokenize(text):
    """Lowercase and extract words via regex."""
    return re.findall(r"[a-z]+", text.lower())


def build_documents(filepath):
    documents = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents

def build_inverted_index(files):
    """
    Builds an inverted indexs

    files: {Sport Team Name: file Path
    }
    
    """
    inverted_index = {}
    for team_name in files:
        filePath = files[team_name]
        documents = build_documents(filePath)
        seen_team = set()
        for doc in documents:
            text = doc["text"]
            for token in tokenize(text):
                if token not in seen_team:
                    seen_team.add(token)
                    if token not in inverted_index:
                        inverted_index[token] = []
                    inverted_index[token].append(team_name)
    for token in inverted_index:
        inverted_index[token] = sorted(inverted_index[token])
    return inverted_index




def main():
    DATASET_FOLDER = "dataset"
    OUTPUT_FILE = "src/data/inverted_index_matrix.json"

    team_files = {}
    for filename in os.listdir(DATASET_FOLDER):
        if filename.endswith(".jsonl"):
            team_name = filename.replace("-utterances.jsonl", "").replace("-", " ").title()
            filepath = os.path.join(DATASET_FOLDER, filename)
            team_files[team_name] = filepath

    result = build_inverted_index(team_files)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)



if __name__ == "__main__":
    main()
import json
import re
import os
from collections import Counter
import wikipediaapi



def tokenize(text):
    """Lowercase and extract words via regex."""
    return re.findall(r"[a-z]+", text.lower())


def build_documents(filepath):
    """

    returns a list of utterances from a filepath
    """
    documents = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents

def build_inverted_index(files):
    """
    Builds an inverted index with team-level term frequencies.

    files: {Sport Team Name: file Path
    }
    
    """
    inverted_index = {}
    for team_name in files:
        filePath = files[team_name]
        documents = build_documents(filePath)
        team_term_freq = Counter()
        for doc in documents:
            text = doc.get("text", "")
            team_term_freq.update(tokenize(text))

        for token, tf in team_term_freq.items():
            if token not in inverted_index:
                inverted_index[token] = {}
            inverted_index[token][team_name] = tf

    for token in inverted_index:
        inverted_index[token] = dict(sorted(inverted_index[token].items()))
    return inverted_index

def scrape_wikipedia(team_name: str) -> str:
    """
    returns the webscraped wikipedia content
    
    """
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="SportsTeamFinder"
    )
    page = wiki.page(team_name)
    if not page.exists():
        return ""
    return page.text




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

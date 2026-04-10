import json
import re
import os
import time
from collections import Counter
import wikipediaapi
from nltk.stem import PorterStemmer
from helper import (
    european_soccrer_league_to_teams,
    americas_soccer_league_to_teams,
    basketball_teams,
    football,
    baseball,
    hockey,
    LEAGUE_ALIASES,
)

stemmer = PorterStemmer()
WIKISCRAPED_CACHE_FILE = "dataset/wiki_cache.json"
wiki = wikipediaapi.Wikipedia(language="en", user_agent="SportsTeamFinder")


team_metadata = {}
for league, teams in european_soccrer_league_to_teams.items():
    for team in teams:
        team_metadata[team] = {"sport": "soccer", "league": league}
for league, teams in americas_soccer_league_to_teams.items():
    for team in teams:
        team_metadata[team] = {"sport": "soccer", "league": league}
for league, teams in basketball_teams.items():
    for team in teams:
        team_metadata[team] = {"sport": "basketball", "league": league}
for league, teams in football.items():
    for team in teams:
        team_metadata[team] = {"sport": "football", "league": league}
for league, teams in baseball.items():
    for team in teams:
        team_metadata[team] = {"sport": "baseball", "league": league}
for league, teams in hockey.items():
    for team in teams:
        team_metadata[team] = {"sport": "hockey", "league": league}



def load_wiki_cache():
    """
    returns scarped wikipedia data
    """
    if os.path.exists(WIKISCRAPED_CACHE_FILE):
        with open(WIKISCRAPED_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_wiki_cache(cache):
    """
    dumps new data to cache
    """
    with open(WIKISCRAPED_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f)




def tokenize(text):
    """Lowercase and extract words via regex."""
    tokens = re.findall(r"[a-z]+", text.lower())
    result = [stemmer.stem(t) for t in tokens]
    return result


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


def build_documents_from_wikipedia(team_name, cache):
    """
    builds documents from scaped wikipedia
    """
    if team_name in cache:
        return cache[team_name]
  
    page = wiki.page(team_name)
    if not page.exists():
        cache[team_name] = []
        save_wiki_cache(cache)
        return []
    teamDocs = [{"text": s.strip()} for s in page.text.split(". ") if s.strip()]
    cache[team_name] = teamDocs
    save_wiki_cache(cache)
    return teamDocs


def build_team_documents(team_name, files, cache):
    """Returns all documents for a team from Reddit, Wikipedia, and metadata."""
    documents = []
    if team_name in files:
        documents = build_documents(files[team_name])
    documents += build_documents_from_wikipedia(team_name, cache)
    meta = team_metadata.get(team_name)
    if meta:
        league = meta['league']
        normalized_league = LEAGUE_ALIASES.get(league.lower(), league)
        documents.append({"text": f"{team_name} is a professional {meta['sport']} team"})
        documents.append({"text": f"{team_name} plays in {normalized_league}"})
    return documents


def build_corpus(files):
    """Builds a corpus dict {team_name: concatenated text}"""
    cache = load_wiki_cache()
    corpus = {}
    all_teams = set(files.keys()).union(set(team_metadata.keys()))
    for team_name in all_teams:
        documents = build_team_documents(team_name, files, cache)
        corpus[team_name] = " ".join(doc.get("text", "") for doc in documents)
    return corpus


def build_inverted_index(files):
    """
    Builds an inverted index with team-level term frequencies.

    files: {Sport Team Name: file Path}
    """
    inverted_index = {}
    cache = load_wiki_cache()
    all_teams = set(files.keys()).union(set(team_metadata.keys()))
    for team_name in all_teams:
        documents = build_team_documents(team_name, files, cache)
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





def main():
    DATASET_FOLDER = "dataset"
    OUTPUT_FILE = "src/data/inverted_index_matrix.json"

    team_files = {}
    for filename in os.listdir(DATASET_FOLDER):
        if filename.endswith(".jsonl"):
            team_name = filename.replace("-utterances.jsonl", "")
            filepath = os.path.join(DATASET_FOLDER, filename)
            team_files[team_name] = filepath

    print(f"Started at: {time.strftime('%H:%M:%S')}")
    start = time.time()
    result = build_inverted_index(team_files)
    print(f"Finished at: {time.strftime('%H:%M:%S')} — took {time.time() - start:.4f}s")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    corpus = build_corpus(team_files)
    with open("src/data/corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)



if __name__ == "__main__":
    main()

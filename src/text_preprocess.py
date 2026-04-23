import json
import re
import os
import time
from collections import Counter
import wikipediaapi
from helper import (
    european_soccer_league_to_teams,
    americas_soccer_league_to_teams,
    basketball_teams,
    football,
    baseball,
    hockey,
    LEAGUE_ALIASES, MULTIWORDS, tokenize, normalize_text
)

WIKISCRAPED_CACHE_FILE = "dataset/wiki_cache.json"
wiki = wikipediaapi.Wikipedia(language="en", user_agent="SportsTeamFinder")

WEIGHTS = {
    "name": 50.0,
    "league": 2.0,
    "sport": 2.0,
    "wiki": 1.0,
    "reddit": .3,
}

MAX_REDDIT_DOCS_PER_TEAM = 100


team_metadata = {}
for league, teams in european_soccer_league_to_teams.items():
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






def build_documents(filepath, cap=None):
    """

    returns a list of utterances from a filepath
    """
    documents = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
            if cap and len(documents) >= cap:
                break
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
        return []
    teamDocs = [{"text": page.text}]
    cache[team_name] = teamDocs
    return teamDocs


def build_team_documents(team_name, files, cache):
    """Returns all documents for a team from Reddit, Wikipedia, and metadata."""
    documents = []
    if team_name in files:
        documents = build_documents(files[team_name], cap=MAX_REDDIT_DOCS_PER_TEAM)
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
    all_teams = set(team_metadata.keys())
    
    for team_name in all_teams:
        team_term_freq = Counter()
        
        for token in tokenize(team_name):
            team_term_freq[token] += WEIGHTS["name"]
        
        normalized_team_name = normalize_text(team_name)
        for phrase, token in MULTIWORDS.items():
            if phrase in normalized_team_name and token not in team_term_freq:
                team_term_freq[token] += WEIGHTS["name"]
        meta = team_metadata.get(team_name)
        if meta:
            league = meta['league']
            normalized_league = LEAGUE_ALIASES.get(league.lower(), league)
            for token in tokenize(f"{team_name} is a professional {meta['sport']} team"):
                team_term_freq[token] += WEIGHTS["sport"]
            for token in tokenize(f"{team_name} plays in {normalized_league}"):
                team_term_freq[token] += WEIGHTS["league"]

        wiki_docs = build_documents_from_wikipedia(team_name, cache)
        if not wiki_docs:
            continue
        for doc in wiki_docs:
            for token in tokenize(doc.get("text", "")):
                team_term_freq[token] += WEIGHTS["wiki"]
        
        if team_name in files:
            reddit_docs = build_documents(files[team_name], cap=MAX_REDDIT_DOCS_PER_TEAM)
            for doc in reddit_docs:
                for token in tokenize(doc.get("text", "")):
                    team_term_freq[token] += WEIGHTS["reddit"]
        
        for token, tf in team_term_freq.items():
            if token not in inverted_index:
                inverted_index[token] = {}
            inverted_index[token][team_name] = tf
    
    save_wiki_cache(cache)
    for token in inverted_index:
        inverted_index[token] = dict(sorted(inverted_index[token].items()))
    return inverted_index





def build_summaries(output_path=None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "data", "team_summaries.json")
    """
    Fetches Wikipedia summaries for all teams using page.summary and writes
    them to data json file as { team: { league, sport, summary } }.
    """
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = {}

    for team, meta in team_metadata.items():
        if team in result:
            continue
        summary = ""
        try:
            page = wiki.page(team)
            if page.exists():
                raw = (page.summary or "").strip()
                if raw:
                    sentences = re.split(r"(?<=[.!?])\s+", raw)
                    summary = " ".join(sentences[:2]).strip()
        except Exception as e:
            print(f"failed summary for '{team}': {e}")

        result[team] = {
            "league": meta["league"],
            "sport": meta["sport"],
            "summary": summary,
        }
    
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

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





if __name__ == "__main__":
    main()

# Sports Team Finder

## Group members

Erik Mauricio (em882), Desmond Whitley (ddw78), Chenwei Hou (ch2352). 

## Introduction

Sports Team Finder is a Flask web app that retrieves sports teams from a prebuilt inverted index using TF-IDF cosine similarity.
Users can search for teams in real time through the web UI, and the app ranks results by textual relevance.

## Project info

- Serves a search interface at `/` for team lookup.
- Uses `src/data/inverted_index_matrix.json` as the search index.
- Runs a custom retrieval engine in `src/ir_engine.py`:
  - Tokenization
  - TF-IDF weighting
  - Cosine similarity scoring
  - Query-term coverage boost for ranking
- Provides:
  - `GET /episodes?title=<query>` for ranked search results
  - `GET /search?q=<query>` for team-name suggestions

Optional mode:
- LLM chat support can be enabled by setting `USE_LLM = True` in `src/routes.py` and providing `API_KEY` in `.env`.

## Architecture

```text
4300-Sports-Team-Finder/
├── src/
│   ├── app.py                         # Flask entry point
│   ├── routes.py                      # Search routes and template selection
│   ├── llm_routes.py                  # Optional LLM chat route (when USE_LLM=True)
│   ├── ir_engine.py                   # TF-IDF + cosine similarity retrieval engine
│   ├── text_preprocess.py             # Builds inverted index from dataset files
│   ├── data/
│   │   └── inverted_index_matrix.json # Prebuilt inverted index used at runtime
│   ├── templates/
│   │   ├── base.html                  # Search UI
│   │   └── chat.html                  # Search + chat UI
│   └── static/
│       ├── style.css
│       └── images/
├── dataset/                           # Source team utterance JSONL files
├── requirements.txt                   # Python dependencies
├── .env.example                       # API key template for optional LLM mode
└── Dockerfile                         # Container build config
```

## Installation (Conda Environment)

### Prerequisites

- Conda installed (Anaconda or Miniconda)

### Setup

1. Create a conda environment:
```bash
conda create -n sports-team-finder python=3.10 -y
```

2. Activate the environment:
```bash
conda activate sports-team-finder
```

3. Install project dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional, for LLM mode) configure environment variables:
```bash
cp .env.example .env
```
Then update `.env` with your real `API_KEY`.

## Run The App

From the project root:
```bash
python src/app.py
```

Open:
`http://localhost:5001`

## Rebuild The Inverted Index (Optional)

If you update files under `dataset/`, regenerate the index with:
```bash
python src/text_preprocess.py
```

This rewrites `src/data/inverted_index_matrix.json`.

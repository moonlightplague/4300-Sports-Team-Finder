# Sports Team Finder

Sports Team Finder is a Flask web app that retrieves sports teams from a prebuilt inverted index using TF-IDF cosine similarity.
Users can search for teams in real time through the web UI, and the app ranks results by textual relevance.

## What The Project Does

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

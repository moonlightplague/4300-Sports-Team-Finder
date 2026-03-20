"""
Routes: home page and team search.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for LLM specific routes.
"""
import json
import os
from flask import render_template, request
from ir_engine import InvertedIndexSearchEngine

# ── AI toggle ──
USE_LLM = False
# USE_LLM = True
# ───────────────

current_directory = os.path.dirname(os.path.abspath(__file__))
index_file_path = os.path.join(current_directory, "data", "inverted_index_matrix.json")
search_engine = InvertedIndexSearchEngine(index_file_path)


def team_search(query, inverted_index):
    if not query or not query.strip():
        return json.dumps([])
    tokenizeQuery = tokenize(query)

    postings = []
    for token in tokenizeQuery:
        if token in inverted_index:
            postings.append(sorted(inverted_index[token]))
    if not postings:
        return json.dumps([])
    print(postings)
    result = postings[0]
    for p in postings[1:]:
        result = merge_postings(result, p)
    return json.dumps(result)


def json_search(query):
    results = search_engine.search(query or "", top_k=20)
    return json.dumps(results)


def register_routes(app):
    @app.route("/")
    def home():
        if USE_LLM:
            return render_template('chat.html')
        return render_template('base.html')

    @app.route("/episodes")
    def episodes_search():
        text = request.args.get("title", "")
        return json_search(text)
    
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        inverted_index = json.load(f)

    @app.route("/search")
    def search():
        query = request.args.get("q", "")
        return team_search(query, inverted_index)

    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app, json_search)

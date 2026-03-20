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


def json_search(query):
    results = search_engine.search(query or "", top_k=20)
    return json.dumps(results)

def json_team_names(query):
    if not query or not query.strip():
        return json.dumps([])
    results = search_engine.search(query, top_k=20)
    team_names = [row["title"] for row in results]
    return json.dumps(team_names)


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

    @app.route("/search")
    def search():
        query = request.args.get("q", "")
        return json_team_names(query)

    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app, json_search)

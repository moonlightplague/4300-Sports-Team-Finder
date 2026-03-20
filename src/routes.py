"""
Routes: home page and episode search.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for LLM specific routes.
"""
import json
import os
from flask import render_template, request
from models import db, Episode, Review
from text_preprocess import tokenize
from analysis import merge_postings

INDEX_PATH = os.path.join(os.path.dirname(__file__), "data", "inverted_index_matrix.json")



# ── AI toggle ──
USE_LLM = False
# USE_LLM = True
# ───────────────


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
    if not query or not query.strip():
        query = "Kardashian"
    results = db.session.query(Episode, Review).join(
        Review, Episode.id == Review.id
    ).filter(
        Episode.title.ilike(f'%{query}%')
    ).all()
    matches = []
    for episode, review in results:
        matches.append({
            'title': episode.title,
            'descr': episode.descr,
            'imdb_rating': review.imdb_rating
        })
    return json.dumps(matches)


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

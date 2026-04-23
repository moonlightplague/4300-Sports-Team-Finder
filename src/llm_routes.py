"""
LLM chat route — only loaded when USE_LLM = True in routes.py.
Adds a POST /chat endpoint that performs LLM-driven RAG for Sports Team Finder.

Setup:
  1. Add API_KEY=your_key to .env
  2. Set USE_LLM = True in routes.py
"""

import json
import os
import logging
from flask import request, jsonify, Response, stream_with_context
from infosci_spark_client import LLMClient

logger = logging.getLogger(__name__)


def improveQuery(client, user_query):
    """
    Rewrite the user's natural language request into a better IR query.
    """
    system = (
        "You are a search query optimizer for a sports team finder. "
        "Rewrite the user's request into a short keyword-style search query. "
        "Focus on play style, culture, fanbase, prestige, history, geography, league, and team identity. "
        "Return ONLY the rewritten query and nothing else."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_query},
    ]

    response = client.chat(messages, stream=False, show_thinking=False)
    rewritten = (response.get("content") or "").strip()
    return rewritten if rewritten else user_query


def build_context(results, top_k=5):
    """
    Turn retrieved IR results into text context for the LLM.
    """
    if not results:
        return "No relevant teams were retrieved."

    context_chunks = []
    for row in results[:top_k]:
        title = row.get("title", "Unknown Team")
        sport = row.get("sport", "unknown")
        league = row.get("league", "Unknown league")
        summary = row.get("summary", "")
        descr = row.get("descr", "")
        score = row.get("score", "N/A")
        svd_score = row.get("svd_score", "N/A")
        matched_terms = ", ".join(row.get("matched_terms", []))

        chunk = (
            f"Team: {title}\n"
            f"Sport: {sport}\n"
            f"League: {league}\n"
            f"Summary: {summary}\n"
            f"Details: {descr}\n"
            f"Matched Terms: {matched_terms}\n"
            f"Cosine Score: {score}\n"
            f"SVD Score: {svd_score}"
        )
        context_chunks.append(chunk)

    return "\n\n---\n\n".join(context_chunks)


def register_chat_route(app, json_search):
    """
    Register the /chat SSE endpoint. Called from routes.py.
    json_search(query) should return a JSON string of IR results.
    """

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        api_key = os.getenv("API_KEY")
        if not api_key:
            return jsonify({"error": "API_KEY not set — add it to your .env file"}), 500

        client = LLMClient(api_key=api_key)

        try:
            # 1. Rewrite user query for retrieval
            rewritten_query = improveQuery(client, user_message)

            # 2. Run IR system
            results = json.loads(json_search(rewritten_query))

            # 3. Build retrieval context for LLM
            context_text = build_context(results, top_k=5)

            # 4. Final grounded answer prompt
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a sports team recommendation assistant. "
                        "Use ONLY the retrieved team results provided. "
                        "Do not invent teams or facts not present in the retrieved results. "
                        "If the results are weak or incomplete, say so clearly."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User request:\n{user_message}\n\n"
                        f"Rewritten retrieval query:\n{rewritten_query}\n\n"
                        f"Retrieved team results:\n{context_text}\n\n"
                        "Based on these retrieved results, recommend the best team or teams for the user "
                        "and explain why in a concise way."
                    ),
                },
            ]

        except Exception as e:
            logger.error(f"RAG setup error: {e}")
            return jsonify({"error": "Failed to prepare RAG response"}), 500

        def generate():
            try:
                # send rewritten query back so frontend can display retrieval
                yield f"data: {json.dumps({'search_term': rewritten_query})}\n\n"

                # stream grounded response
                for chunk in client.chat(messages, stream=True):
                    if chunk.get("content"):
                        yield f"data: {json.dumps({'content': chunk['content']})}\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': 'Streaming error occurred'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
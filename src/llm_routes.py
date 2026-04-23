"""
LLM chat route — loaded only when USE_LLM = True in routes.py.
Adds a POST /chat endpoint that performs sports-team RAG.
"""
import json
import logging
import os
from typing import Dict, List

from flask import Response, jsonify, request, stream_with_context
from infosci_spark_client import LLMClient

logger = logging.getLogger(__name__)

TEAM_SUMMARIES_PATH = os.path.join(
    os.path.dirname(__file__), "data", "team_summaries.json"
)


def _load_team_summaries() -> Dict[str, str]:
    if not os.path.exists(TEAM_SUMMARIES_PATH):
        return {}
    try:
        with open(TEAM_SUMMARIES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            team: str((entry or {}).get("summary", "")).strip()
            for team, entry in data.items()
            if team
        }
    except Exception as exc:
        logger.warning("Failed to load team summaries: %s", exc)
        return {}


TEAM_SUMMARIES = _load_team_summaries()


def _clean_text(value, max_len=500):
    text = " ".join(str(value or "").split())
    return text[:max_len]


def _build_ir_query(client, user_message):
    system_prompt = (
        "You rewrite sports-team preference prompts into search queries for an IR system. "
        "Return only a concise keyword query (about 4-12 terms), no explanation."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    try:
        response = client.chat(
            messages,
            stream=False,
            show_thinking=False,
            reasoning_level="low",
        )
        rewritten = _clean_text((response or {}).get("content", ""), max_len=200)
        return rewritten or user_message
    except Exception as exc:
        logger.warning("IR query rewrite failed; falling back to user message: %s", exc)
        return user_message


def _run_ir(json_search, ir_query) -> List[Dict]:
    try:
        results = json.loads(json_search(ir_query or ""))
        if isinstance(results, list):
            return results
    except Exception as exc:
        logger.error("IR search failed: %s", exc)
    return []


def _retrieval_context(ir_query, retrieved, max_items=8):
    if not retrieved:
        return (
            f"IR query used: {ir_query}\n"
            "No IR results were returned."
        )

    lines = [f"IR query used: {ir_query}", "Top retrieved teams:"]
    for idx, team in enumerate(retrieved[:max_items], start=1):
        title = _clean_text(team.get("title", "Unknown team"), max_len=120)
        league = _clean_text(team.get("league", ""), max_len=120)
        sport = _clean_text(team.get("sport", ""), max_len=60)
        score = team.get("score")
        matched_terms = team.get("matched_terms") or []
        if not isinstance(matched_terms, list):
            matched_terms = []
        matched_terms = [_clean_text(t, max_len=30) for t in matched_terms[:8] if t]
        summary = (
            TEAM_SUMMARIES.get(title)
            or _clean_text(team.get("summary", ""), max_len=420)
            or _clean_text(team.get("descr", ""), max_len=420)
        )
        lines.append(
            f"{idx}. {title} | sport={sport} | league={league} | score={score}"
        )
        if matched_terms:
            lines.append(f"   matched_terms: {', '.join(matched_terms)}")
        if summary:
            lines.append(f"   summary: {summary}")
    return "\n".join(lines)


def _build_generation_messages(user_message, ir_query, retrieved):
    context = _retrieval_context(ir_query, retrieved)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a sports-team recommendation assistant. "
                "Ground your answer in the retrieved IR evidence. "
                "Use only the provided retrieval context when giving factual claims, "
                "and clearly say when retrieval evidence is weak or missing."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original user request:\n{user_message}\n\n"
                f"Retrieval context:\n{context}\n\n"
                "Give a concise answer with recommended teams and why they match."
            ),
        },
    ]
    return messages


def register_chat_route(app, json_search):
    """Register the /chat SSE endpoint. Called from routes.py."""

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        api_key = os.getenv("SPARK_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            return jsonify(
                {"error": "Missing SPARK_API_KEY (or API_KEY) in environment"}
            ), 500

        client = LLMClient(api_key=api_key)
        ir_query = _build_ir_query(client, user_message)
        retrieved = _run_ir(json_search, ir_query)
        messages = _build_generation_messages(user_message, ir_query, retrieved)

        def generate():
            if ir_query:
                yield f"data: {json.dumps({'search_term': ir_query})}\n\n"
            try:
                streamed_any_content = False
                for chunk in client.chat(messages, stream=True, show_thinking=False):
                    content = chunk.get("content")
                    if content:
                        streamed_any_content = True
                        yield f"data: {json.dumps({'content': content})}\n\n"
                if not streamed_any_content:
                    yield f"data: {json.dumps({'error': 'No response from LLM service'})}\n\n"
            except Exception as exc:
                logger.error("Streaming error: %s", exc)
                yield f"data: {json.dumps({'error': 'LLM streaming error occurred'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

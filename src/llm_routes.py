"""
LLM chat route — loaded only when USE_LLM = True in routes.py.
Adds a POST /chat endpoint that performs sports-team RAG.
"""
import json
import logging
import os
import re
from typing import Dict, List, Tuple

from flask import Response, jsonify, request, stream_with_context
from infosci_spark_client import LLMClient

logger = logging.getLogger(__name__)

TEAM_SUMMARIES_PATH = os.path.join(
    os.path.dirname(__file__), "data", "team_summaries.json"
)
WORD_RE = re.compile(r"[a-z0-9_]+")


def _load_team_summaries() -> Dict[str, Dict]:
    if not os.path.exists(TEAM_SUMMARIES_PATH):
        return {}
    try:
        with open(TEAM_SUMMARIES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        cleaned = {}
        for team, entry in data.items():
            if not team or not isinstance(entry, dict):
                continue
            cleaned[team] = {
                "league": str(entry.get("league", "") or "").strip(),
                "sport": str(entry.get("sport", "") or "").strip(),
                "summary": str(entry.get("summary", "") or "").strip(),
                "extended": str(entry.get("extended", "") or "").strip(),
                "sections": entry.get("sections") if isinstance(entry.get("sections"), dict) else {},
            }
        return cleaned
    except Exception as exc:
        logger.warning("Failed to load team summaries: %s", exc)
        return {}


TEAM_SUMMARIES = _load_team_summaries()


def _clean_text(value, max_len=500):
    text = " ".join(str(value or "").split())
    return text[:max_len]


def _token_set(text):
    return set(WORD_RE.findall(str(text or "").lower()))


def _select_relevant_sections(sections, query_terms, max_sections=2):
    if not isinstance(sections, dict) or not sections:
        return []

    ranked = []
    for section_title, section_text in sections.items():
        title = _clean_text(section_title, max_len=120)
        body = _clean_text(section_text, max_len=420)
        if not title or not body:
            continue
        content_tokens = _token_set(f"{title} {body}")
        overlap = len(query_terms & content_tokens)
        ranked.append((overlap, len(body), title, body))

    if not ranked:
        return []

    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    selected = [row for row in ranked if row[0] > 0][:max_sections]
    if not selected:
        selected = ranked[:1]
    return [(title, body, overlap) for overlap, _, title, body in selected]


def _team_knowledge(title, query_terms) -> Tuple[str, str, List[Tuple[str, str, int]], str]:
    entry = TEAM_SUMMARIES.get(title) or {}
    if not isinstance(entry, dict):
        return "", "", [], "IR index summary/description"

    summary = _clean_text(entry.get("summary", ""), max_len=320)
    extended = _clean_text(entry.get("extended", ""), max_len=520)
    sections = _select_relevant_sections(entry.get("sections"), query_terms, max_sections=2)

    used_parts = []
    if summary:
        used_parts.append("summary")
    if extended:
        used_parts.append("extended")
    if sections:
        used_parts.append("sections")

    if used_parts:
        return summary, extended, sections, f"team_summaries.json ({', '.join(used_parts)})"
    return "", "", [], "IR index summary/description"


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

    lines = [
        f"IR query used: {ir_query}",
        "Top retrieved teams with source IDs:",
        "Use these source IDs for citations in the final answer.",
    ]
    base_query_terms = _token_set(ir_query)
    for idx, team in enumerate(retrieved[:max_items], start=1):
        source_id = f"S{idx}"
        title = _clean_text(team.get("title", "Unknown team"), max_len=120)
        league = _clean_text(team.get("league", ""), max_len=120)
        sport = _clean_text(team.get("sport", ""), max_len=60)
        score = team.get("score")
        matched_terms = team.get("matched_terms") or []
        top_terms = team.get("top_terms") or []
        if not isinstance(matched_terms, list):
            matched_terms = []
        if not isinstance(top_terms, list):
            top_terms = []
        matched_terms = [_clean_text(t, max_len=30) for t in matched_terms[:8] if t]
        top_terms = [_clean_text(t, max_len=30) for t in top_terms[:8] if t]
        team_query_terms = set(base_query_terms)
        team_query_terms.update(_token_set(" ".join(matched_terms)))
        team_query_terms.update(_token_set(" ".join(top_terms)))
        summary_from_json, extended_from_json, relevant_sections, summary_source = _team_knowledge(
            title, team_query_terms
        )
        summary = (
            summary_from_json
            or _clean_text(team.get("summary", ""), max_len=420)
            or _clean_text(team.get("descr", ""), max_len=420)
        )
        lines.append(
            f"[{source_id}] {title} | sport={sport} | league={league} | score={score}"
        )
        if matched_terms:
            lines.append(f"   matched_terms: {', '.join(matched_terms)}")
        if top_terms:
            lines.append(f"   top_terms: {', '.join(top_terms)}")
        if summary:
            lines.append(f"   summary: {summary}")
        if extended_from_json:
            lines.append(f"   extended: {extended_from_json}")
        for section_title, section_body, overlap in relevant_sections:
            lines.append(
                f"   section ({section_title}) [query_overlap={overlap}]: {section_body}"
            )
        lines.append(f"   source_type: {summary_source}")
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
                "and clearly say when retrieval evidence is weak or missing. "
                "When available, leverage summary/extended/section evidence from team_summaries.json. "
                "Explain in plain language why recommended teams match the user request. "
                "Only recommend teams that appear in the retrieval context source list. "
                "Avoid technical IR jargon and raw field names (for example: matched_terms, top_terms, SVD, lexical score, query overlap). "
                "Do not expose ranking math or internal retrieval statistics unless the user explicitly asks. "
                "Cite sources inline using [S#] and end with a short 'Sources: ...' line listing cited source IDs."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original user request:\n{user_message}\n\n"
                f"Retrieval context:\n{context}\n\n"
                "Output format requirements:\n"
                "1) Write one or two cohesive paragraphs in natural language for a non-technical user.\n"
                "2) Mention the strongest 1-3 team recommendations and explain fit in everyday terms.\n"
                "3) Do not use bullets, numbered lists, or markdown headings.\n"
                "4) Include inline citations like [S1] naturally in sentences.\n"
                "5) End with one line in this form: Sources: [S1], [S3]."
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

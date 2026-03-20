import json
import math
import os
import re
from collections import Counter, defaultdict


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase and tokenize text into alphanumeric terms."""
    return TOKEN_PATTERN.findall((text or "").lower())


class InvertedIndexSearchEngine:
    """
    Search engine backed only by a term -> teams inverted index.

    Retrieval method:
      - Cosine similarity on binary team vectors weighted by IDF.
      - Query uses TF-IDF weights.
      - Final rank adds a query-term coverage boost.
    """

    def __init__(self, index_path):
        self.index_path = index_path
        self.inverted_index = {}
        self.team_terms = defaultdict(set)
        self.teams = []
        self.idf = {}
        self.doc_norm = {}
        self._load_index()

    def _load_index(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"Inverted index file not found: {self.index_path}"
            )

        with open(self.index_path, "r", encoding="utf-8") as f:
            raw_index = json.load(f)

        normalized_index = {}
        for term, team_list in raw_index.items():
            if not isinstance(team_list, list):
                continue
            deduped = sorted({team for team in team_list if isinstance(team, str)})
            if not deduped:
                continue
            normalized_index[term] = set(deduped)
            for team in deduped:
                self.team_terms[team].add(term)

        # Add team-name tokens so direct team queries (e.g. "lakers") are retrievable.
        all_teams = sorted(self.team_terms.keys())
        for team in all_teams:
            for token in tokenize(team):
                normalized_index.setdefault(token, set()).add(team)
                self.team_terms[team].add(token)

        self.inverted_index = normalized_index
        self.teams = all_teams

        num_teams = max(1, len(self.teams))
        for term, teams in self.inverted_index.items():
            df = len(teams)
            self.idf[term] = 1.0 + math.log((num_teams + 1.0) / (df + 1.0))

        for team, terms in self.team_terms.items():
            norm_sq = 0.0
            for term in terms:
                weight = self.idf.get(term, 0.0)
                norm_sq += weight * weight
            self.doc_norm[team] = math.sqrt(norm_sq) if norm_sq > 0 else 1.0

    def search(self, query, top_k=20):
        query_tokens = tokenize(query)

        if not query_tokens:
            return [
                {
                    "title": team,
                    "descr": "Team from inverted index.",
                    "imdb_rating": 0.0,
                    "score": 0.0,
                    "matched_terms": [],
                }
                for team in self.teams[:top_k]
            ]

        query_tf = Counter(query_tokens)
        query_weights = {}
        for term, tf in query_tf.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            query_weights[term] = tf * idf

        if not query_weights:
            return []

        query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
        if query_norm == 0:
            return []

        candidate_teams = set()
        for term in query_weights:
            candidate_teams.update(self.inverted_index.get(term, set()))

        scored = []
        for team in candidate_teams:
            dot = 0.0
            matched_terms = []
            for term, q_weight in query_weights.items():
                if team in self.inverted_index.get(term, set()):
                    d_weight = self.idf[term]
                    dot += q_weight * d_weight
                    matched_terms.append(term)

            denom = query_norm * self.doc_norm.get(team, 1.0)
            cosine_score = dot / denom if denom else 0.0
            coverage = len(matched_terms) / max(1, len(query_weights))
            score = cosine_score + (0.1 * coverage)
            if score <= 0:
                continue

            scored.append(
                {
                    "title": team,
                    "descr": "Matched terms: " + ", ".join(matched_terms[:8]),
                    "imdb_rating": round(score, 4),
                    "score": round(score, 4),
                    "matched_terms": matched_terms,
                }
            )

        scored.sort(key=lambda x: (-x["score"], x["title"].lower()))
        return scored[:top_k]

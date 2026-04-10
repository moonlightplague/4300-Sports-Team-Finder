import json
import math
import os
import re
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

QUERY_EXPANSIONS = {
    "exciting": ["energetic", "lethal", "attacking", "explosive"],
    "young": ["prospects", "academy", "rebuilding", "upcoming"],
    "loyal": ["passionate", "dedicated", "support", "atmosphere"],
    "historic": ["legacy", "tradition", "trophies", "iconic"],
    "successful": ["winning", "dominant", "elite", "championship"],
    "defensive": ["defense", "physical", "disciplined", "tough"],
    "offensive": ["attacking", "creative", "fast", "scoring"],
}


def tokenize(text):
    """Lowercase and tokenize text into alphanumeric terms."""
    return TOKEN_PATTERN.findall((text or "").lower())


def is_good_term(term):
    """
    Basic term cleanup to reduce junk in the vocabulary.
    """
    if not isinstance(term, str):
        return False
    if len(term) < 3:
        return False
    if term.isdigit():
        return False
    if len(set(term)) == 1:
        return False
    if re.fullmatch(r"(ha)+", term):
        return False
    return True


def expand_query_tokens(tokens):
    """
    Add synonym style expansions for better matching.
    """
    expanded = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(QUERY_EXPANSIONS.get(token, []))
    return expanded


class InvertedIndexSearchEngine:
    """
    Search engine backed by a term -> {team: term_frequency} inverted index.

    Retrieval method:
      - Cosine similarity on team vectors weighted by TF-IDF.
      - Query uses TF-IDF weights.
      - Final rank adds a query-term coverage boost.
    """

    def __init__(self, index_path):
        self.index_path = index_path
        self.inverted_index = {}
        self.team_term_tf = defaultdict(dict)
        self.teams = []
        self.idf = {}
        self.doc_norm = {}
        self._load_index()

    @staticmethod
    def _tf_weight(tf):
        return 1.0 + math.log(tf) if tf > 0 else 0.0

    def _load_index(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"Inverted index file not found: {self.index_path}"
            )

        with open(self.index_path, "r", encoding="utf-8") as f:
            raw_index = json.load(f)

        normalized_index = {}
        for term, postings in raw_index.items():
            if not is_good_term(term):
                continue

            term = term.lower()
            term_postings = {}

            if isinstance(postings, dict):
                for team, tf in postings.items():
                    if not isinstance(team, str):
                        continue
                    if not isinstance(tf, (int, float)) or tf <= 0:
                        continue
                    term_postings[team] = int(tf)

            elif isinstance(postings, list):
                # Backward compatibility for previous binary format: term -> [team]
                for team in postings:
                    if isinstance(team, str):
                        term_postings[team] = 1
            else:
                continue

            if not term_postings:
                continue

            normalized_index[term] = term_postings
            for team, tf in term_postings.items():
                self.team_term_tf[team][term] = tf

        # Add team-name tokens so direct team queries (e.g. "lakers") are retrievable.
        all_teams = sorted(self.team_term_tf.keys())
        for team in all_teams:
            team_name_tf = Counter(tokenize(team))
            for token, tf in team_name_tf.items():
                if not is_good_term(token):
                    continue
                postings = normalized_index.setdefault(token, {})
                postings[team] = postings.get(team, 0) + tf
                self.team_term_tf[team][token] = (
                    self.team_term_tf[team].get(token, 0) + tf
                )

        self.inverted_index = normalized_index
        self.teams = all_teams

        num_teams = max(1, len(self.teams))
        for term, teams in self.inverted_index.items():
            df = len(teams)
            self.idf[term] = 1.0 + math.log((num_teams + 1.0) / (df + 1.0))

        for team, term_tf in self.team_term_tf.items():
            norm_sq = 0.0
            for term, tf in term_tf.items():
                weight = self._tf_weight(tf) * self.idf.get(term, 0.0)
                norm_sq += weight * weight
            self.doc_norm[team] = math.sqrt(norm_sq) if norm_sq > 0 else 1.0

    def search(self, query, top_k=20):
        query_tokens = tokenize(query)
        query_tokens = [stemmer.stem(t) for t in query_tokens]
        query_tokens = expand_query_tokens(query_tokens)

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
            query_weights[term] = self._tf_weight(tf) * idf

        if not query_weights:
            return []

        query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
        if query_norm == 0:
            return []

        candidate_teams = set()
        for term in query_weights:
            candidate_teams.update(self.inverted_index.get(term, {}).keys())

        scored = []
        for team in candidate_teams:
            dot = 0.0
            matched_terms = []
            term_contributions = []

            for term, q_weight in query_weights.items():
                tf = self.inverted_index.get(term, {}).get(team, 0)
                if tf > 0:
                    d_weight = self._tf_weight(tf) * self.idf[term]
                    contribution = q_weight * d_weight
                    dot += contribution
                    matched_terms.append(term)
                    term_contributions.append((term, contribution))

            denom = query_norm * self.doc_norm.get(team, 1.0)
            cosine_score = dot / denom if denom else 0.0
            coverage = len(matched_terms) / max(1, len(query_weights))
            score = cosine_score + (0.1 * coverage)

            if score <= 0:
                continue

            term_contributions.sort(key=lambda x: x[1], reverse=True)
            top_terms = [term for term, _ in term_contributions[:5]]

            scored.append(
                {
                    "title": team,
                    "descr": "Strongest matches: " + ", ".join(top_terms),
                    "imdb_rating": round(score, 4),
                    "score": round(score, 4),
                    "matched_terms": matched_terms,
                    "top_terms": top_terms,
                }
            )

        scored.sort(key=lambda x: (-x["score"], x["title"].lower()))
        return scored[:top_k]
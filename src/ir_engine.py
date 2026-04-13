import json
import math
import os
import re
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from helper import tokenize
from rapidfuzz import process, fuzz
from helper import normalize_text, MULTIWORDS, TEAM_TO_SPORT




SVD_EPSILON = 1e-10
QUERY_EXPANSION_WEIGHT = 0.35
MIN_EXACT_CANDIDATES = 8

QUERY_EXPANSIONS = {
    "exciting": ["energetic", "lethal", "attacking", "explosive"],
    "young": ["prospects", "academy", "rebuilding", "upcoming"],
    "loyal": ["passionate", "dedicated", "support", "atmosphere"],
    "historic": ["legacy", "tradition", "trophies", "iconic"],
    "successful": ["winning", "dominant", "elite", "championship"],
    "defensive": ["defense", "physical", "disciplined", "tough"],
    "offensive": ["attacking", "creative", "fast", "scoring"],
    # Domain-level expansions for broad sports queries.
    "basketball": ["nba", "wnba", "hoop", "fiba", "dunk", "lebron"],
    "nba": ["basketball", "wnba", "fiba", "dunk", "playoffs"],
}

QUERY_TERM_EXPANSION_WEIGHT = {
    "basketball": 0.85,
}


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
      - Build TF-IDF document vectors for teams.
      - Compute truncated SVD to create latent semantic vectors.
      - Rank with cosine similarity in SVD space.
      - Explain results with query-term and latent-component contributions.
    """

    def __init__(self, index_path, max_svd_components=20):
        self.index_path = index_path
        self.max_svd_components = max(1, int(max_svd_components))

        self.inverted_index = {}
        self.team_term_tf = defaultdict(dict)
        self.teams = []
        self.team_to_idx = {}
        self.team_name_tokens = {}
        self.idf = {}
        self.doc_norm = {}

        self._u_k = None
        self._s_k = None
        self._team_latent = None
        self._team_latent_norm = None

        self._load_index()
    
    def _fuzz(self, token):
        if token in self.inverted_index:
            return token

        fuzzMatch = process.extractOne(token, 
        self.inverted_index.keys(),
        scorer=fuzz.ratio, 
        score_cutoff=70)

        if fuzzMatch:
            return fuzzMatch[0]
        return token

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

        self.team_name_tokens = {}
        for team in self.teams:
            tokens = set(tokenize(team))
            normalized_name = normalize_text(team)
            for phrase, token in MULTIWORDS.items():
                if phrase in normalized_name:
                    tokens.add(token)
            self.team_name_tokens[team] = tokens
        for team, term_tf in self.team_term_tf.items():
            norm_sq = 0.0
            for term, tf in term_tf.items():
                weight = self._tf_weight(tf) * self.idf.get(term, 0.0)
                norm_sq += weight * weight
            self.doc_norm[team] = math.sqrt(norm_sq) if norm_sq > 0 else 1.0

        self._build_svd_model()

    def _build_svd_model(self):
        num_teams = len(self.teams)
        self.team_to_idx = {team: idx for idx, team in enumerate(self.teams)}

        if num_teams == 0:
            self._u_k = np.zeros((0, 1), dtype=np.float64)
            self._s_k = np.ones(1, dtype=np.float64)
            self._team_latent = np.zeros((0, 1), dtype=np.float64)
            self._team_latent_norm = np.ones(0, dtype=np.float64)
            return

        num_terms = len(self.inverted_index)
        if num_terms == 0:
            self._u_k = np.zeros((num_teams, 1), dtype=np.float64)
            self._s_k = np.ones(1, dtype=np.float64)
            self._team_latent = np.zeros((num_teams, 1), dtype=np.float64)
            self._team_latent_norm = np.ones(num_teams, dtype=np.float64)
            return

        max_components = min(self.max_svd_components, num_teams - 1, num_terms - 1)
        if max_components <= 0:
            self._u_k = np.zeros((num_teams, 1), dtype=np.float64)
            self._s_k = np.ones(1, dtype=np.float64)
            self._team_latent = np.zeros((num_teams, 1), dtype=np.float64)
            self._team_latent_norm = np.ones(num_teams, dtype=np.float64)
            return

        rows = []
        cols = []
        values = []
        for term_idx, (term, postings) in enumerate(self.inverted_index.items()):
            term_idf = self.idf.get(term, 0.0)
            if term_idf <= 0:
                continue
            for team, tf in postings.items():
                row_idx = self.team_to_idx.get(team)
                if row_idx is None:
                    continue
                weight = self._tf_weight(tf) * term_idf
                if weight <= 0:
                    continue
                rows.append(row_idx)
                cols.append(term_idx)
                values.append(weight)

        if not values:
            self._u_k = np.zeros((num_teams, 1), dtype=np.float64)
            self._s_k = np.ones(1, dtype=np.float64)
            self._team_latent = np.zeros((num_teams, 1), dtype=np.float64)
            self._team_latent_norm = np.ones(num_teams, dtype=np.float64)
            return

        tfidf_matrix = csr_matrix(
            (
                np.asarray(values, dtype=np.float32),
                (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
            ),
            shape=(num_teams, num_terms),
            dtype=np.float32,
        )

        svd = TruncatedSVD(
            n_components=max_components,
            algorithm="randomized",
            random_state=0,
        )
        team_latent = svd.fit_transform(tfidf_matrix).astype(np.float64, copy=False)
        singular_values = svd.singular_values_.astype(np.float64, copy=False)

        positive = singular_values > SVD_EPSILON
        if not np.any(positive):
            self._u_k = np.zeros((num_teams, 1), dtype=np.float64)
            self._s_k = np.ones(1, dtype=np.float64)
            self._team_latent = np.zeros((num_teams, 1), dtype=np.float64)
            self._team_latent_norm = np.ones(num_teams, dtype=np.float64)
            return

        self._s_k = singular_values[positive]
        self._team_latent = team_latent[:, positive]
        self._u_k = self._team_latent / self._s_k

        norms = np.linalg.norm(self._team_latent, axis=1)
        norms[norms == 0] = 1.0
        self._team_latent_norm = norms

    def _build_query_weights(self, query_tokens, include_expansions=True):
        query_tf = Counter(query_tokens)
        query_weights = defaultdict(float)

        for term, tf in query_tf.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            query_weights[term] += self._tf_weight(tf) * idf

        if include_expansions:
            for term in query_tf.keys():
                expansion_weight = QUERY_TERM_EXPANSION_WEIGHT.get(
                    term, QUERY_EXPANSION_WEIGHT
                )
                for expanded_term in QUERY_EXPANSIONS.get(term, []):
                    idf = self.idf.get(expanded_term)
                    if idf is None:
                        continue
                    query_weights[expanded_term] += expansion_weight * idf

        return dict(query_weights)

    def _project_query_to_latent(self, query_weights):
        doc_query_dots = np.zeros(len(self.teams), dtype=np.float64)

        for term, q_weight in query_weights.items():
            postings = self.inverted_index.get(term)
            if not postings:
                continue
            term_idf = self.idf.get(term, 0.0)
            for team, tf in postings.items():
                doc_idx = self.team_to_idx.get(team)
                if doc_idx is None:
                    continue
                d_weight = self._tf_weight(tf) * term_idf
                doc_query_dots[doc_idx] += d_weight * q_weight

        if self._u_k is None or self._s_k is None:
            return np.zeros(1, dtype=np.float64)

        return (doc_query_dots @ self._u_k) / self._s_k

    def _term_latent_vector(self, term):
        if self._u_k is None or self._s_k is None:
            return np.zeros(1, dtype=np.float64)

        postings = self.inverted_index.get(term)
        if not postings:
            return np.zeros(self._u_k.shape[1], dtype=np.float64)

        latent = np.zeros(self._u_k.shape[1], dtype=np.float64)
        term_idf = self.idf.get(term, 0.0)

        for team, tf in postings.items():
            doc_idx = self.team_to_idx.get(team)
            if doc_idx is None:
                continue
            latent += (self._tf_weight(tf) * term_idf) * self._u_k[doc_idx]

        return latent / self._s_k

    def _explain_match(self, team_idx, query_weights, q_unit):
        team_vec = self._team_latent[team_idx]
        team_norm = self._team_latent_norm[team_idx]
        team_unit = team_vec / team_norm if team_norm else team_vec

        component_scores = q_unit * team_unit
        component_order = np.argsort(np.abs(component_scores))[::-1][:3]
        svd_components = [
            {
                "component": int(comp_idx + 1),
                "contribution": round(float(component_scores[comp_idx]), 4),
            }
            for comp_idx in component_order
        ]

        term_scores = []
        for term, q_weight in query_weights.items():
            term_latent = self._term_latent_vector(term)
            contribution = q_weight * float(np.dot(term_latent, team_unit))
            term_scores.append((term, contribution))

        term_scores.sort(key=lambda item: item[1], reverse=True)
        top_terms = [term for term, score in term_scores if score > 0][:5]
        if not top_terms:
            top_terms = [term for term, _ in term_scores[:5]]
        matched_terms = [term for term, score in term_scores if score > 0]

        component_text = ", ".join(
            f"LS{item['component']} ({item['contribution']:+.3f})"
            for item in svd_components
        )
        terms_text = ", ".join(top_terms) if top_terms else "latent-semantic overlap"
        description = (
            f"SVD evidence terms: {terms_text}. "
            f"Top latent factors: {component_text}"
        )

        return {
            "descr": description,
            "top_terms": top_terms,
            "matched_terms": matched_terms,
            "svd_components": svd_components,
        }

    def search(self, query, top_k=20):
        query_tokens = tokenize(query)
        query_tokens = [self._fuzz(t) for t in query_tokens]

        if not query_tokens:
            return [
                {
                    "title": team,
                    "descr": "SVD baseline ranking (no query provided).",
                    "imdb_rating": 0.0,
                    "score": 0.0,
                    "matched_terms": [],
                    "sport": TEAM_TO_SPORT.get(team, "unknown"),
                }
                for team in self.teams[:top_k]
            ]

        exact_query_weights = self._build_query_weights(
            query_tokens, include_expansions=False
        )
        expanded_query_weights = self._build_query_weights(
            query_tokens, include_expansions=True
        )

        if not expanded_query_weights:
            return []

        has_domain_intent = any(
            term in QUERY_TERM_EXPANSION_WEIGHT for term in query_tokens
        )

        exact_candidate_teams = set()
        for term in exact_query_weights:
            exact_candidate_teams.update(self.inverted_index.get(term, {}).keys())

        expanded_candidate_teams = set()
        for term in expanded_query_weights:
            expanded_candidate_teams.update(self.inverted_index.get(term, {}).keys())

        use_expanded_lexical_reference = (
            has_domain_intent
            or not exact_query_weights
            or len(exact_candidate_teams) < min(top_k, MIN_EXACT_CANDIDATES)
        )
        lexical_reference_weights = (
            expanded_query_weights
            if use_expanded_lexical_reference
            else exact_query_weights
        )

        candidate_teams = set(exact_candidate_teams)
        if has_domain_intent or len(candidate_teams) < min(top_k, MIN_EXACT_CANDIDATES):
            candidate_teams.update(expanded_candidate_teams)
        if not candidate_teams:
            candidate_teams.update(expanded_candidate_teams)
        if not candidate_teams:
            return []

        q_latent = self._project_query_to_latent(expanded_query_weights)
        q_norm = float(np.linalg.norm(q_latent))
        if q_norm <= SVD_EPSILON:
            return []
        q_unit = q_latent / q_norm

        lexical_query_norm = math.sqrt(
            sum(weight * weight for weight in lexical_reference_weights.values())
        )
        lexical_scores = {}
        lexical_matches = {}

        for team in candidate_teams:
            dot = 0.0
            matched_terms = []

            for term, q_weight in lexical_reference_weights.items():
                tf = self.inverted_index.get(term, {}).get(team, 0)
                if tf <= 0:
                    continue
                d_weight = self._tf_weight(tf) * self.idf.get(term, 0.0)
                dot += q_weight * d_weight
                matched_terms.append(term)

            denom = lexical_query_norm * self.doc_norm.get(team, 1.0)
            lexical_scores[team] = (dot / denom) if denom else 0.0
            lexical_matches[team] = matched_terms

        max_lexical = max(lexical_scores.values()) if lexical_scores else 0.0
        num_ref_terms = max(1, len(lexical_reference_weights))
        exact_term_set = set(exact_query_weights.keys())
        exact_query_set = set(query_tokens)

        ranked = []
        for team in candidate_teams:
            team_idx = self.team_to_idx.get(team)
            if team_idx is None:
                continue
            denom = self._team_latent_norm[team_idx] * q_norm
            svd_score = float((self._team_latent[team_idx] @ q_latent) / denom) if denom else 0.0
            if svd_score <= 0:
                continue

            matched_ref_terms = lexical_matches.get(team, [])
            coverage = len(matched_ref_terms) / num_ref_terms
            lexical = lexical_scores.get(team, 0.0)
            lexical_norm = (lexical / max_lexical) if max_lexical > SVD_EPSILON else 1.0

            team_tokens = self.team_name_tokens.get(team, set())
            name_overlap = (
                len(team_tokens.intersection(exact_term_set)) / len(exact_term_set)
                if exact_term_set
                else 1.0
            )

            coverage_factor = 0.35 + (0.55 * coverage)
            lexical_factor = 0.30 + (0.60 * lexical_norm)
            name_factor = 0.60 + (0.40 * name_overlap)
            score = svd_score * coverage_factor * lexical_factor * name_factor

            if exact_query_set and exact_query_set == team_tokens:
                score += 0.50
            elif exact_query_set and exact_query_set.issubset(team_tokens):
                score += 0.35
            elif exact_query_set and team_tokens.issubset(exact_query_set):
                score += 0.30

            if score <= 0:
                continue

            explanation = self._explain_match(team_idx, expanded_query_weights, q_unit)
            strong_terms = []
            if matched_ref_terms:
                strong_terms.extend(matched_ref_terms)
            for term in explanation["top_terms"]:
                if term not in strong_terms:
                    strong_terms.append(term)
                if len(strong_terms) >= 5:
                    break
            if not strong_terms:
                strong_terms = explanation["top_terms"]

            explanation_descr = (
                f"{explanation['descr']} "
                f"(query overlap: {len(matched_ref_terms)}/{num_ref_terms})"
            )
            ranked.append(
                {
                    "title": team,
                    "descr": explanation_descr,
                    "imdb_rating": round(score, 4),
                    "score": round(score, 4),
                    "matched_terms": strong_terms,
                    "top_terms": strong_terms,
                    "svd_components": explanation["svd_components"],
                    "svd_score": round(svd_score, 4),
                    "lexical_score": round(lexical, 4),
                    "sport": TEAM_TO_SPORT.get(team, "unknown"),
                }
            )

        ranked.sort(key=lambda item: (-item["score"], item["title"].lower()))
        return ranked[:top_k]

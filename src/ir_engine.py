import json
import math
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
import numpy as np

UI_DIMENSION_COUNT = 8
TOP_TERMS_PER_DIM = 10
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from helper import tokenize
from rapidfuzz import process, fuzz
from helper import (
    normalize_text,
    MULTIWORDS,
    TEAM_TO_SPORT,
    TEAM_TO_LEAGUE,
    TEAM_TO_SUMMARY,
)
try:
    from gensim.models import Word2Vec
except ImportError:
    Word2Vec = None




SVD_EPSILON = 1e-10
MIN_EXACT_CANDIDATES = 8

EMBEDDING_VECTOR_SIZE = 80
EMBEDDING_WINDOW = 8
EMBEDDING_EPOCHS = 18
EMBEDDING_TOPN = 8
EMBEDDING_MAX_EXPANSIONS = 4
EMBEDDING_MIN_SIMILARITY = 0.45
EMBEDDING_EXPANSION_WEIGHT = 0.35
EMBEDDING_REPEAT_CAP = 6
EMBEDDING_TERMS_PER_TEAM = 1200
DEFAULT_EMBEDDING_WORKERS = max(1, (os.cpu_count() or 2) - 1)
EMBEDDING_WORKERS_ENV = "STF_EMBEDDING_WORKERS"


JUNK_TERMS = {
    "shit", "fuck", "fucking", "damn", "lol", "lmao", "rofl",
    "yeah", "yes", "gonna", "wanna", "gotta",
    "http", "https", "www", "com", "amp", "tweetposter",
    "deleted", "removed", "reddit", "subreddit", "upvote", "downvote",
    "gif", "jpg", "png", "mp4",
    "just", "like", "think", "really", "know", "good", "don",
    "going", "right", "better", "got", "need", "want", "great",
    "way", "make", "people", "say", "love", "bad", "didn",
    "sure", "lot", "guys", "pretty", "let", "doesn", "isn",
    "guy", "getting", "hope", "probably", "does", "look",
    "maybe", "thing", "come", "actually", "feel", "watch",
    "work", "thought", "hes", "edit", "honestly", "dont", "gets", "thats",
    "weird", "wow", "sorry", "mean", "saying", "looks", "makes", "needs", "agree",
    "doing", "things", "thanks", "bit", "looking",
    "wasn", "remember", "definitely", "believe", "wouldn",
    "god", "hate", "watching", "trying", "wrong",
    "guess", "happy", "true", "hes", "nice",
    "far", "today", "dude", "tonight", "didnt", "expect", "aren", "wtf", "thread", "kinda", "imagine", "understand",
    "happens", "obviously", "funny", "feels", "amazing",
"knows", "yea", "hasn", "absolutely", "stupid",
"hell", "hopefully", "wait", "thank", "awesome",
"talking", "ago", "exactly", "tell", "happen",
"kind", "rest", "especially", "looked", "stop", "nah", "ive", "doesnt", "shouldn", "garbage", "forgot",
"imo", "yesterday", "ass", "fault", "thought", "little",
"couple", "idea", "read", "happened", "worse", "course",
"life", "means", "okay", "ridiculous", "dumb", "downvoted", "bullshit",
"disagree", "sucks", "shitty", "worried", "wonder",
"quite", "gone", "matter", "sense", "solid",
"thinking", "wish", "forget", "fucked", "anymore", "isnt", "dad", "stuff",
"nbsp", "bitch", "crazy", "alright", "holy", "talk",
"seriously", "completely",
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
    if term in JUNK_TERMS:
        return False
    return True


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
        self._team_name_choices = []
        self._team_name_choice_strings = []
        self.idf = {}
        self.doc_norm = {}
        self._fuzzy_terms = ()
        self._protected_name_terms = set()
        self._term_team_weights = {}

        self._u_k = None
        self._s_k = None
        self._team_latent = None
        self._team_latent_norm = None
        self._embedding_vectors = None
        self._term_latent_cache = {}
        self._dim_ranks = {}
        self._top_terms_by_team_dim = {}

        self._load_index()
        dim_names_path = os.path.join(os.path.dirname(self.index_path), "svd_dimension_names.json")
        self.dimension_names = {}
        if os.path.exists(dim_names_path):
            with open(dim_names_path, "r", encoding="utf-8") as f:
                self.dimension_names = {int(k): v for k, v in json.load(f).items()}

    @staticmethod
    def _embedding_workers():
        configured = os.getenv(EMBEDDING_WORKERS_ENV)
        if configured is None:
            return DEFAULT_EMBEDDING_WORKERS
        try:
            return max(1, int(configured))
        except ValueError:
            return DEFAULT_EMBEDDING_WORKERS
    
    @staticmethod
    def _fuzzy_cutoff(token):
        token_len = len(token)
        if token_len <= 4:
            return 90
        if token_len <= 6:
            return 82
        return 76

    @lru_cache(maxsize=4096)
    def _fuzz(self, token):
        if token in self.inverted_index:
            return token
        if len(token) < 4 or token.isdigit():
            return token

        fuzzy_matches = process.extract(
            token,
            self._fuzzy_terms,
            scorer=fuzz.ratio,
            score_cutoff=self._fuzzy_cutoff(token),
            limit=5,
        )
        if fuzzy_matches:
            best_score = fuzzy_matches[0][1]
            near_best = [
                match for match in fuzzy_matches
                if match[1] >= (best_score - 2)
            ]
            protected_matches = [
                match for match in near_best
                if match[0] in self._protected_name_terms
            ]
            pool = protected_matches if protected_matches else near_best
            candidate = min(
                pool,
                key=lambda match: abs(len(match[0]) - len(token))
            )[0]
            if len(candidate) <= 3 and len(token) >= 6 and (len(candidate) + 2) < len(token):
                return token
            return candidate
        return token

    @staticmethod
    def _tf_weight(tf):
        return 1.0 + math.log(tf) if tf > 0 else 0.0

    @staticmethod
    def _normalize_for_fuzzy(text):
        return " ".join(re.findall(r"[a-z0-9]+", normalize_text(text or "")))

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
        protected_name_terms = set()
        for team in all_teams:
            team_name_tf = Counter(tokenize(team))
            normalized_name = normalize_text(team)
            for phrase, token in MULTIWORDS.items():
                if phrase in normalized_name:
                    team_name_tf[token] += 1
            for token, tf in team_name_tf.items():
                if not is_good_term(token):
                    continue
                protected_name_terms.add(token)
                postings = normalized_index.setdefault(token, {})
                postings[team] = postings.get(team, 0) + tf
                self.team_term_tf[team][token] = (
                    self.team_term_tf[team].get(token, 0) + tf
                )
        
        max_df = int(0.6 * len(all_teams))
        self._protected_name_terms = set(protected_name_terms)
        normalized_index = {
            term: postings for term, postings in normalized_index.items()
            if (
                term in protected_name_terms
                or (
                    3 <= len(postings) <= max_df
                    and not (sum(postings.values()) > 10000 and len(postings) < 50)
                )
            )
        }

        self.team_term_tf = defaultdict(dict)
        for term, postings in normalized_index.items():
            for team, tf in postings.items():
                self.team_term_tf[team][term] = tf
        

        self.inverted_index = normalized_index
        self.teams = all_teams
        self._fuzzy_terms = tuple(self.inverted_index.keys())

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
        self._team_name_choices = list(self.teams)
        self._team_name_choice_strings = [
            self._normalize_for_fuzzy(team)
            for team in self._team_name_choices
        ]
        for team, term_tf in self.team_term_tf.items():
            norm_sq = 0.0
            for term, tf in term_tf.items():
                weight = self._tf_weight(tf) * self.idf.get(term, 0.0)
                norm_sq += weight * weight
            self.doc_norm[team] = math.sqrt(norm_sq) if norm_sq > 0 else 1.0

        self._build_embedding_model()
        self._build_svd_model()

    def _build_embedding_model(self):
        """
        Train domain-specific word embeddings using gensim so query expansion is
        data-driven rather than hardcoded.
        """
        if Word2Vec is None:
            self._embedding_vectors = None
            return

        sentences = []
        for term_tf in self.team_term_tf.values():
            sentence = []
            top_terms = sorted(
                term_tf.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:EMBEDDING_TERMS_PER_TEAM]
            for term, tf in top_terms:
                if term not in self.inverted_index or tf <= 0:
                    continue
                repeats = int(round(self._tf_weight(tf)))
                repeats = max(1, min(EMBEDDING_REPEAT_CAP, repeats))
                sentence.extend([term] * repeats)
            if len(sentence) >= 2:
                sentences.append(sentence)

        if len(sentences) < 2:
            self._embedding_vectors = None
            return

        try:
            model = Word2Vec(
                sentences=sentences,
                vector_size=EMBEDDING_VECTOR_SIZE,
                window=EMBEDDING_WINDOW,
                min_count=1,
                workers=self._embedding_workers(),
                sg=1,
                epochs=EMBEDDING_EPOCHS,
                seed=0,
            )
            self._embedding_vectors = model.wv
        except Exception:
            self._embedding_vectors = None
    def print_latent_dimensions(self, n_dims=20, n_terms=10):
        if self._svd_components is None:
            print("No SVD components available.")
            return
        for i in range(min(n_dims, self._svd_components.shape[0])):
            component = self._svd_components[i]
            top_pos = component.argsort()[-n_terms:][::-1]
            top_neg = component.argsort()[:n_terms]
            print(f"Dimension {i+1}:")
            print("  Positive:", [self._term_list[j] for j in top_pos])
            print("  Negative:", [self._term_list[j] for j in top_neg])
    def _build_svd_model(self):
        num_teams = len(self.teams)
        self.team_to_idx = {team: idx for idx, team in enumerate(self.teams)}
        self._term_team_weights = {}
        self._term_latent_cache = {}

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
            term_rows = []
            term_values = []
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
                term_rows.append(row_idx)
                term_values.append(weight)
            if term_rows:
                self._term_team_weights[term] = (
                    np.asarray(term_rows, dtype=np.int32),
                    np.asarray(term_values, dtype=np.float64),
                )

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

        self._svd_components = svd.components_
        self._term_list = list(self.inverted_index.keys())

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
        self._precompute_dim_ranks()
        self._precompute_top_terms_by_team_dim()

    def _precompute_dim_ranks(self):
        """
        compute dimension ranks:
        - positive side ranked among positive values (descending)
        - negative side ranked among negative values (more negative ranks higher)
        """
        self._dim_ranks = {}
        if self._team_latent is None or not len(self.teams):
            return

        max_dims = min(UI_DIMENSION_COUNT, self._team_latent.shape[1])
        for dim_idx in range(max_dims):
            col = self._team_latent[:, dim_idx]

            pos_indices = [idx for idx, val in enumerate(col) if val >= 0]
            neg_indices = [idx for idx, val in enumerate(col) if val < 0]

            pos_sorted = sorted(pos_indices, key=lambda idx: col[idx], reverse=True)
            neg_sorted = sorted(neg_indices, key=lambda idx: col[idx])

            rank_by_team = {}
            for rank, idx in enumerate(pos_sorted, start=1):
                rank_by_team[self.teams[idx]] = rank
            for rank, idx in enumerate(neg_sorted, start=1):
                rank_by_team[self.teams[idx]] = rank

            self._dim_ranks[dim_idx + 1] = rank_by_team

    def _precompute_top_terms_by_team_dim(self):
        """
        Precompute signed per-team dimension terms:
        """
        self._top_terms_by_team_dim = {}
        if self._svd_components is None or not self._term_list:
            return

        max_dims = min(UI_DIMENSION_COUNT, self._svd_components.shape[0])
        term_to_idx = {term: idx for idx, term in enumerate(self._term_list)}

        for team in self.teams:
            team_tf = self.team_term_tf.get(team, {})
            if not team_tf:
                self._top_terms_by_team_dim[team] = {
                    dim_id: [] for dim_id in range(1, max_dims + 1)
                }
                continue

            weighted_terms = []
            for term, tf in team_tf.items():
                term_idx = term_to_idx.get(term)
                if term_idx is None:
                    continue
                tfidf = self._tf_weight(tf) * self.idf.get(term, 0.0)
                if tfidf <= 0:
                    continue
                weighted_terms.append((term, term_idx, tfidf))

            by_dim = {}
            for dim_id in range(1, max_dims + 1):
                comp = self._svd_components[dim_id - 1]
                contribs = [
                    (term, float(tfidf * comp[term_idx]))
                    for term, term_idx, tfidf in weighted_terms
                ]
                contribs.sort(key=lambda item: abs(item[1]), reverse=True)
                by_dim[dim_id] = [
                    {"t": term, "w": round(weight, 4)}
                    for term, weight in contribs[:TOP_TERMS_PER_DIM]
                ]
            self._top_terms_by_team_dim[team] = by_dim

    @staticmethod
    def _as_dict(value):
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _coerce_dim_terms(raw_terms):
        """
        Normalize arbitrary JSON-ish term entries into:
        [(term: str, weight: float), ...]
        """
        if not isinstance(raw_terms, list):
            return []
        out = []
        for item in raw_terms:
            if not isinstance(item, dict):
                continue
            term = item.get("t")
            if not isinstance(term, str) or not term:
                continue
            try:
                weight = float(item.get("w", 0.0))
            except (TypeError, ValueError):
                weight = 0.0
            out.append((term, weight))
        return out

    def _derive_pole_label(self, dim_entry, positive=True):
        key = "top_terms_pos" if positive else "top_terms_neg"
        terms = self._coerce_dim_terms(self._as_dict(dim_entry).get(key, []))
        names = [term.replace("_", " ") for term, _ in terms]
        if not names:
            return ""
        return " / ".join(names[:2])

    def _get_dimension_display(self, dim_id, dim_score):
        dim_entry = self._as_dict(self.dimension_names.get(dim_id, {}))
        label_pos = str(dim_entry.get("label_pos", "")).strip()
        label_neg = str(dim_entry.get("label_neg", "")).strip()
        blurb = str(dim_entry.get("blurb", "")).strip()

        if not label_pos:
            label_pos = self._derive_pole_label(dim_entry, positive=True)
        if not label_neg:
            label_neg = self._derive_pole_label(dim_entry, positive=False)

        if dim_score >= 0:
            label = label_pos or f"Dimension LS{dim_id}"
        else:
            label = label_neg or f"Dimension LS{dim_id}"

        if not blurb:
            if label_pos and label_neg:
                blurb = f"{label_pos} vs {label_neg}"
            else:
                blurb = f"Latent semantic axis LS{dim_id}."

        return label, blurb

    def _query_dim_overlap(self, query_weights, team_top_terms_by_dim, dim_scores):
        """
        For each dimension, return query terms that overlap with this team's
        top contributing terms on that dimension (pole-aware when possible).
        """
        query_term_set = set(query_weights.keys())
        overlap_by_dim = {}
        if not query_term_set:
            return overlap_by_dim

        for dim_id, terms in team_top_terms_by_dim.items():
            dim_score = dim_scores.get(dim_id, 0.0)
            sign = 1 if dim_score >= 0 else -1
            coerced = self._coerce_dim_terms(terms)
            aligned_terms = [term for term, weight in coerced if weight * sign > 0]
            if not aligned_terms:
                aligned_terms = [term for term, _ in coerced]
            overlap = sorted(query_term_set.intersection(set(aligned_terms)))
            if overlap:
                overlap_by_dim[dim_id] = overlap
        return overlap_by_dim


    def _embedding_expansions(self, term):
        if self._embedding_vectors is None:
            return []
        if term not in self._embedding_vectors.key_to_index:
            return []

        expansions = []
        try:
            candidates = self._embedding_vectors.most_similar(term, topn=EMBEDDING_TOPN)
        except KeyError:
            return []

        for expanded_term, similarity in candidates:
            if similarity < EMBEDDING_MIN_SIMILARITY:
                continue
            if expanded_term == term:
                continue
            if expanded_term not in self.inverted_index:
                continue
            if not is_good_term(expanded_term):
                continue
            expansions.append((expanded_term, float(similarity)))
            if len(expansions) >= EMBEDDING_MAX_EXPANSIONS:
                break

        return expansions

    def _build_query_weights(self, query_tokens, include_expansions=True):
        query_tf = Counter(query_tokens)
        query_weights = defaultdict(float)
        expanded_terms = set()

        for term, tf in query_tf.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            query_weights[term] += self._tf_weight(tf) * idf

        if include_expansions:
            for term in query_tf.keys():
                for expanded_term, similarity in self._embedding_expansions(term):
                    idf = self.idf.get(expanded_term)
                    if idf is None:
                        continue
                    query_weights[expanded_term] += (
                        EMBEDDING_EXPANSION_WEIGHT * similarity * idf
                    )
                    expanded_terms.add(expanded_term)

        return dict(query_weights), expanded_terms

    def _project_query_to_latent(self, query_weights):
        doc_query_dots = np.zeros(len(self.teams), dtype=np.float64)

        for term, q_weight in query_weights.items():
            team_weights = self._term_team_weights.get(term)
            if team_weights is None:
                continue
            doc_indices, doc_weights = team_weights
            doc_query_dots[doc_indices] += doc_weights * q_weight

        if self._u_k is None or self._s_k is None:
            return np.zeros(1, dtype=np.float64)

        return (doc_query_dots @ self._u_k) / self._s_k

    def _term_latent_vector(self, term):
        if self._u_k is None or self._s_k is None:
            return np.zeros(1, dtype=np.float64)

        if term in self._term_latent_cache:
            return self._term_latent_cache[term]

        team_weights = self._term_team_weights.get(term)
        if team_weights is None:
            return np.zeros(self._u_k.shape[1], dtype=np.float64)

        doc_indices, doc_weights = team_weights
        if doc_indices.size == 0:
            return np.zeros(self._u_k.shape[1], dtype=np.float64)

        latent = (doc_weights[:, None] * self._u_k[doc_indices]).sum(axis=0)
        latent = latent / self._s_k
        self._term_latent_cache[term] = latent
        return latent

    def _explain_match(self, team_idx, query_weights, q_unit):
        team_vec = self._team_latent[team_idx]
        team_norm = self._team_latent_norm[team_idx]
        team_unit = team_vec / team_norm if team_norm else team_vec

        component_scores = q_unit * team_unit
        component_order = np.argsort(np.abs(component_scores))[::-1][:6]
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

    def _team_name_fuzzy_boosts(self, query):
        if not self._team_name_choice_strings:
            return {}

        normalized_query = self._normalize_for_fuzzy(query)
        if len(normalized_query) < 3:
            return {}

        matches = process.extract(
            normalized_query,
            self._team_name_choice_strings,
            scorer=fuzz.WRatio,
            score_cutoff=80,
            limit=8,
        )
        boosts = {}
        for _, score, idx in matches:
            team = self._team_name_choices[idx]
            boost = 0.05 + ((float(score) - 80.0) / 20.0) * 0.35
            boosts[team] = max(boosts.get(team, 0.0), min(0.45, boost))
        return boosts

    def search(self, query, top_k=20):
        query_tokens = tokenize(query)
        query_tokens = [self._fuzz(t) for t in query_tokens]
        team_name_boosts = self._team_name_fuzzy_boosts(query)

        if not query_tokens:
            return []

        exact_query_weights, _ = self._build_query_weights(
            query_tokens, include_expansions=False
        )
        expanded_query_weights, expanded_terms = self._build_query_weights(
            query_tokens, include_expansions=True
        )

        if not expanded_query_weights:
            return []

        has_query_expansion = bool(expanded_terms)

        exact_candidate_teams = set()
        for term in exact_query_weights:
            exact_candidate_teams.update(self.inverted_index.get(term, {}).keys())

        expanded_candidate_teams = set()
        for term in expanded_query_weights:
            expanded_candidate_teams.update(self.inverted_index.get(term, {}).keys())

        use_expanded_lexical_reference = (
            has_query_expansion
            or not exact_query_weights
            or len(exact_candidate_teams) < min(top_k, MIN_EXACT_CANDIDATES)
        )
        lexical_reference_weights = (
            expanded_query_weights
            if use_expanded_lexical_reference
            else exact_query_weights
        )

        candidate_teams = set(exact_candidate_teams)
        if has_query_expansion or len(candidate_teams) < min(top_k, MIN_EXACT_CANDIDATES):
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
        lexical_matches = defaultdict(list)
        candidate_indices = {
            self.team_to_idx[team]
            for team in candidate_teams
            if team in self.team_to_idx
        }
        lexical_dots = np.zeros(len(self.teams), dtype=np.float64)

        for term, q_weight in lexical_reference_weights.items():
            team_weights = self._term_team_weights.get(term)
            if team_weights is None:
                continue
            doc_indices, doc_weights = team_weights
            for doc_idx, doc_weight in zip(doc_indices, doc_weights):
                if doc_idx not in candidate_indices:
                    continue
                lexical_dots[doc_idx] += q_weight * doc_weight
                lexical_matches[self.teams[doc_idx]].append(term)

        for team in candidate_teams:
            team_idx = self.team_to_idx.get(team)
            if team_idx is None:
                continue
            denom = lexical_query_norm * self.doc_norm.get(team, 1.0)
            lexical_scores[team] = (lexical_dots[team_idx] / denom) if denom else 0.0

        max_lexical = max(lexical_scores.values()) if lexical_scores else 0.0
        num_ref_terms = max(1, len(lexical_reference_weights))
        exact_term_set = set(exact_query_weights.keys())
        exact_query_set = set(query_tokens)

        ranked = []
        max_dims = min(UI_DIMENSION_COUNT, self._team_latent.shape[1])
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
            if exact_term_set and exact_term_set.issubset(team_tokens):
                score += 0.35
            score += team_name_boosts.get(team, 0.0)

            if score <= 0:
                continue

            explanation = self._explain_match(team_idx, expanded_query_weights, q_unit)
            team_vec = self._team_latent[team_idx]
            team_norm = self._team_latent_norm[team_idx]
            team_unit = team_vec / team_norm if team_norm else team_vec

            component_scores = q_unit * team_unit
            dim_scores = {
                dim_id: round(float(component_scores[dim_id - 1]), 4)
                for dim_id in range(1, max_dims + 1)
            }
            dim_ranks = {
                dim_id: self._dim_ranks.get(dim_id, {}).get(team)
                for dim_id in range(1, max_dims + 1)
            }
            top_terms_by_dim = self._top_terms_by_team_dim.get(
                team,
                {dim_id: [] for dim_id in range(1, max_dims + 1)},
            )
            dim_labels = {}
            dim_blurbs = {}
            for dim_id in range(1, max_dims + 1):
                label, blurb = self._get_dimension_display(dim_id, dim_scores[dim_id])
                dim_labels[dim_id] = label
                dim_blurbs[dim_id] = blurb
            query_overlap_by_dim = self._query_dim_overlap(
                expanded_query_weights,
                top_terms_by_dim,
                dim_scores,
            )

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
                    "expanded_terms": sorted(expanded_terms),
                    "dim_scores": dim_scores,
                    "dim_ranks": dim_ranks,
                    "top_terms_by_dim": top_terms_by_dim,
                    "query_overlap_by_dim": query_overlap_by_dim,
                    "dim_labels": dim_labels,
                    "dim_blurbs": dim_blurbs,
                    "top_terms": strong_terms,
                    "svd_components": explanation["svd_components"],
                    "svd_score": round(svd_score, 4),
                    "lexical_score": round(lexical, 4),
                    "sport": TEAM_TO_SPORT.get(team, "unknown"),
                    "league": TEAM_TO_LEAGUE.get(team, ""),
                    "summary": TEAM_TO_SUMMARY.get(team, ""),
                }
            )

        ranked.sort(key=lambda item: (-item["score"], item["title"].lower()))
        return ranked[:top_k]

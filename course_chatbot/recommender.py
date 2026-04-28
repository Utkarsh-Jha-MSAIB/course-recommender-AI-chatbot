import json
import math
import os
import pickle
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "course_artifacts"
load_dotenv()

EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_EMBEDDING_TIMEOUT_SECONDS = int(os.getenv("GEMINI_EMBEDDING_TIMEOUT_SECONDS", "30"))
GEMINI_EMBEDDING_DELAY_SECONDS = float(os.getenv("GEMINI_EMBEDDING_DELAY_SECONDS", "0"))


class GeminiEmbeddingClient:
    def __init__(
        self,
        model_name: str,
        timeout_seconds: int = GEMINI_EMBEDDING_TIMEOUT_SECONDS,
        delay_seconds: float = GEMINI_EMBEDDING_DELAY_SECONDS,
    ):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found. Add it to .env or set it in the environment.")

        self.model_name = model_name.removeprefix("models/")
        self.timeout_seconds = timeout_seconds
        self.delay_seconds = delay_seconds
        self.url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:embedContent?key={api_key}"
        )
        self.cache: Dict[str, np.ndarray] = {}

    def embed_query(self, text: str) -> np.ndarray:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if cleaned in self.cache:
            return self.cache[cleaned]

        payload = {
            "model": f"models/{self.model_name}",
            "content": {"parts": [{"text": cleaned}]},
            "taskType": "RETRIEVAL_QUERY",
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini embedding HTTP {exc.code}: {error_body}") from exc

        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

        values = body.get("embedding", {}).get("values")
        if not values:
            raise RuntimeError(f"Gemini embedding response did not include values: {body}")

        vector = np.array([values], dtype="float32")
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        vector = vector / norm
        self.cache[cleaned] = vector
        return vector


class CourseRecommender:
    def __init__(
        self,
        artifact_dir: Path = ARTIFACT_DIR,
        model_name: str = EMBEDDING_MODEL,
        hybrid_alpha: float = 0.6,
    ):
        self.artifact_dir = artifact_dir
        self.model_name = model_name.removeprefix("models/")
        self.hybrid_alpha = hybrid_alpha
        self.embedding_provider = "gemini" if self.model_name.startswith("gemini-embedding") else "local"

        self.index = faiss.read_index(str(self.resolve_index_path()))
        with open(self.artifact_dir / "course_metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        if self.embedding_provider == "gemini":
            self.model = GeminiEmbeddingClient(self.model_name)
        else:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is required when using a local embedding model.")
            self.model = SentenceTransformer(self.model_name)
        self.lexical_index, self.doc_lengths, self.avg_doc_length = self.build_lexical_index()

    def resolve_index_path(self) -> Path:
        if self.embedding_provider == "gemini":
            index_path = self.artifact_dir / f"course_faiss_{self.model_name}.index"
            if not index_path.exists():
                raise FileNotFoundError(
                    f"Missing Gemini FAISS index: {index_path}. "
                    "Build it from the matching Gemini course embeddings first."
                )
            return index_path
        return self.artifact_dir / "course_faiss.index"

    # =========================================================
    # LEXICAL SEARCH
    # =========================================================
    def course_key(self, course: Dict) -> str:
        return (
            str(course.get("Course ID") or "").strip()
            or str(course.get("Course URL") or "").strip()
            or str(course.get("Course Name") or "").strip().lower()
        )

    def tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
        return [token for token in tokens if len(token) > 1]

    def lexical_text(self, course: Dict) -> str:
        parts = [
            course.get("course_text_for_embedding", ""),
            course.get("Course Name", ""),
            course.get("University / Industry Partner Name", ""),
            course.get("Type of Content", ""),
            course.get("Difficulty Level_Clean", ""),
            course.get("Domain_Clean", ""),
            course.get("Sub-Domain_Clean", ""),
            course.get("Course Language_Clean", ""),
            course.get("Unified Skills Text", ""),
            course.get("Course Description_Clean", ""),
            course.get("Course Description", ""),
            course.get("Specialization", ""),
        ]
        return " ".join(str(part or "") for part in parts)

    def build_lexical_index(self) -> Tuple[Dict[str, List[Tuple[int, int]]], List[int], float]:
        inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        doc_lengths: List[int] = []

        for doc_idx, course in enumerate(self.metadata):
            token_counts = Counter(self.tokenize(self.lexical_text(course)))
            doc_length = sum(token_counts.values())
            doc_lengths.append(doc_length)

            for token, count in token_counts.items():
                inverted_index[token].append((doc_idx, count))

        avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
        return dict(inverted_index), doc_lengths, avg_doc_length

    def lexical_search(self, query: str, top_k: int = 20) -> List[Dict]:
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        scores = defaultdict(float)
        num_docs = len(self.metadata)
        k1 = 1.5
        b = 0.75

        for token, query_count in query_counts.items():
            postings = self.lexical_index.get(token)
            if not postings:
                continue

            doc_freq = len(postings)
            idf = math.log(1 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))

            for doc_idx, term_freq in postings:
                doc_length = self.doc_lengths[doc_idx] or 0
                length_norm = 1 - b
                if self.avg_doc_length:
                    length_norm += b * (doc_length / self.avg_doc_length)
                denom = term_freq + k1 * length_norm
                bm25 = idf * ((term_freq * (k1 + 1)) / denom)
                scores[doc_idx] += bm25 * query_count

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        results = []
        for doc_idx, score in ranked:
            row = dict(self.metadata[doc_idx])
            row["lexical_score"] = float(score)
            results.append(row)
        return results

    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)
        if max_score == min_score:
            return {key: 1.0 for key in scores}
        return {key: (score - min_score) / (max_score - min_score) for key, score in scores.items()}

    def hybrid_search(
        self,
        query: str,
        semantic_k: int = 20,
        lexical_k: int = 20,
        alpha: Optional[float] = None,
    ) -> List[Dict]:
        alpha = self.hybrid_alpha if alpha is None else alpha
        alpha = max(0.0, min(1.0, float(alpha)))

        semantic_hits = self.semantic_search(query, top_k=semantic_k)
        lexical_hits = self.lexical_search(query, top_k=lexical_k)

        by_key: Dict[str, Dict] = {}
        semantic_scores = {}
        lexical_scores = {}

        for course in semantic_hits:
            key = self.course_key(course)
            if not key:
                continue
            by_key[key] = dict(course)
            semantic_scores[key] = float(course.get("semantic_score", 0.0))

        for course in lexical_hits:
            key = self.course_key(course)
            if not key:
                continue
            merged = by_key.get(key, dict(course))
            merged["lexical_score"] = float(course.get("lexical_score", 0.0))
            by_key[key] = merged
            lexical_scores[key] = float(course.get("lexical_score", 0.0))

        semantic_norm = self.normalize_scores(semantic_scores)
        lexical_norm = self.normalize_scores(lexical_scores)

        results = []
        for key, course in by_key.items():
            sem = semantic_norm.get(key, 0.0)
            lex = lexical_norm.get(key, 0.0)
            course["semantic_score_norm"] = float(sem)
            course["lexical_score"] = float(course.get("lexical_score", 0.0))
            course["lexical_score_norm"] = float(lex)
            course["hybrid_score"] = float(alpha * sem + (1.0 - alpha) * lex)
            results.append(course)

        results.sort(
            key=lambda course: (
                course.get("hybrid_score", 0.0),
                course.get("semantic_score_norm", 0.0),
                course.get("lexical_score_norm", 0.0),
            ),
            reverse=True,
        )
        return results

    # =========================================================
    # SEMANTIC SEARCH
    # =========================================================
    def semantic_search(self, query: str, top_k: int = 20) -> List[Dict]:
        if self.embedding_provider == "gemini":
            query_vec = self.model.embed_query(query)
        else:
            query_vec = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = dict(self.metadata[idx])
            row["semantic_score"] = float(score)
            results.append(row)
        return results

    # =========================================================
    # FILTERING
    # =========================================================
    def passes_filters(
        self,
        course: Dict,
        difficulty: Optional[str] = None,
        allowed_difficulties: Optional[Set[str]] = None,
        max_hours: Optional[float] = None,
        min_rating: Optional[float] = None,
        require_rated: bool = False,
        require_enrollment: bool = False,
    ) -> bool:
        course_difficulty = str(course.get("Difficulty Level_Clean", "UNKNOWN") or "UNKNOWN")
        hours = course.get("Hours_Final", None)
        rating = course.get("Course Rating_Clean", None)
        has_rating = int(course.get("Has_Rating", 0) or 0)
        has_enrollment = int(course.get("Has_Enrollment", 0) or 0)

        if difficulty is not None and course_difficulty != difficulty:
            return False

        if allowed_difficulties is not None and course_difficulty not in allowed_difficulties:
            return False

        if max_hours is not None and hours is not None:
            try:
                if float(hours) > float(max_hours):
                    return False
            except Exception:
                return False

        if require_rated and not has_rating:
            return False

        if min_rating is not None:
            try:
                if float(rating) < float(min_rating):
                    return False
            except Exception:
                return False

        if require_enrollment and not has_enrollment:
            return False

        return True

    # =========================================================
    # RANKING COMPONENTS
    # =========================================================
    def rating_component(self, course: Dict) -> float:
        has_rating = int(course.get("Has_Rating", 0) or 0)
        rating = float(course.get("Course Rating_Clean", 0.0) or 0.0)

        if not has_rating:
            return 0.02

        return 0.15 * (rating / 5.0)

    def popularity_component(self, course: Dict) -> float:
        has_enrollment = int(course.get("Has_Enrollment", 0) or 0)
        log_enrollment = float(course.get("Enrollment_Count_Log1p", 0.0) or 0.0)

        if not has_enrollment:
            return 0.02

        normalized = min(log_enrollment / 14.0, 1.0)
        return 0.12 * normalized

    def hours_component(self, course: Dict, preferred_max_hours: Optional[float] = None) -> float:
        hours = course.get("Hours_Final", None)
        if hours is None:
            return 0.0

        try:
            hours = float(hours)
        except Exception:
            return 0.0

        if preferred_max_hours is not None:
            if hours <= preferred_max_hours:
                ratio = hours / max(preferred_max_hours, 1.0)
                return 0.08 * (1.0 - 0.5 * ratio)
            return 0.0

        if hours <= 5:
            return 0.05
        if hours <= 15:
            return 0.06
        if hours <= 30:
            return 0.04
        return 0.01

    # =========================================================
    # RERANK
    # =========================================================
    def rerank(self, results: List[Dict], preferred_max_hours: Optional[float] = None) -> List[Dict]:
        reranked = []

        for course in results:
            relevance_score = float(course.get("hybrid_score", course.get("semantic_score", 0.0)))

            final_score = (
                0.65 * relevance_score
                + self.rating_component(course)
                + self.popularity_component(course)
                + self.hours_component(course, preferred_max_hours=preferred_max_hours)
            )

            row = dict(course)
            row["final_score"] = float(final_score)
            reranked.append(row)

        reranked.sort(
            key=lambda x: (
                x["final_score"],
                x.get("semantic_score", 0.0),
                x.get("Course Rating_Clean", 0.0),
                x.get("Enrollment_Count_Log1p", 0.0),
            ),
            reverse=True,
        )
        return reranked

    # =========================================================
    # MAIN SEARCH
    # =========================================================
    def search_courses(
        self,
        query: str,
        top_k_retrieval: int = 20,
        top_k_final: int = 5,
        search_mode: str = "hybrid",
        hybrid_alpha: Optional[float] = None,
        difficulty: Optional[str] = None,
        allowed_difficulties: Optional[Set[str]] = None,
        max_hours: Optional[float] = None,
        min_rating: Optional[float] = None,
        require_rated: bool = False,
        require_enrollment: bool = False,
    ) -> List[Dict]:
        if search_mode == "semantic":
            raw_hits = self.semantic_search(query, top_k=top_k_retrieval)
        elif search_mode == "lexical":
            raw_hits = self.lexical_search(query, top_k=top_k_retrieval)
        elif search_mode == "hybrid":
            raw_hits = self.hybrid_search(
                query,
                semantic_k=top_k_retrieval,
                lexical_k=top_k_retrieval,
                alpha=hybrid_alpha,
            )
        else:
            raise ValueError("search_mode must be 'semantic', 'lexical', or 'hybrid'.")

        filtered = [
            course for course in raw_hits
            if self.passes_filters(
                course,
                difficulty=difficulty,
                allowed_difficulties=allowed_difficulties,
                max_hours=max_hours,
                min_rating=min_rating,
                require_rated=require_rated,
                require_enrollment=require_enrollment,
            )
        ]

        reranked = self.rerank(
            filtered,
            preferred_max_hours=max_hours,
        )

        results = []
        for course in reranked[:top_k_final]:
            unified_skills = (
                course.get("Unified Skills List")
                or course.get("Unified Skills Text")
                or course.get("skills")
                or ""
            )

            results.append({
                "course_name": course.get("Course Name", ""),
                "partner": course.get("University / Industry Partner Name", ""),
                "difficulty": course.get("Difficulty Level_Clean", "UNKNOWN"),
                "domain": course.get("Domain_Clean", ""),
                "sub_domain": course.get("Sub-Domain_Clean", ""),
                "hours": course.get("Hours_Final", None),
                "rating": course.get("Course Rating_Clean", None),
                "has_rating": int(course.get("Has_Rating", 0) or 0),
                "enrollment_count": course.get("Enrollment_Count_Clean", None),
                "has_enrollment": int(course.get("Has_Enrollment", 0) or 0),
                "enrollment_log1p": course.get("Enrollment_Count_Log1p", 0.0),
                "skills": unified_skills,
                "specialization": course.get("Specialization", ""),
                "url": course.get("Course URL", "") or "",
                "image_url": course.get("Course Image URL", "") or course.get("image_url", "") or "",
                "semantic_score": float(course.get("semantic_score", 0.0)),
                "lexical_score": float(course.get("lexical_score", 0.0)),
                "hybrid_score": float(course.get("hybrid_score", 0.0)),
                "final_score": float(course.get("final_score", 0.0)),
            })
        return results

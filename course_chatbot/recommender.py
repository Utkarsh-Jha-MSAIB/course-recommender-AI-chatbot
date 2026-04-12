import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set

import faiss
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "course_artifacts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class CourseRecommender:
    def __init__(self, artifact_dir: Path = ARTIFACT_DIR, model_name: str = EMBEDDING_MODEL):
        self.artifact_dir = artifact_dir
        self.model_name = model_name

        self.index = faiss.read_index(str(self.artifact_dir / "course_faiss.index"))
        with open(self.artifact_dir / "course_metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        self.model = SentenceTransformer(self.model_name)

    # =========================================================
    # SEMANTIC SEARCH
    # =========================================================
    def semantic_search(self, query: str, top_k: int = 20) -> List[Dict]:
        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
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
            semantic_score = float(course.get("semantic_score", 0.0))

            final_score = (
                0.65 * semantic_score
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
        difficulty: Optional[str] = None,
        allowed_difficulties: Optional[Set[str]] = None,
        max_hours: Optional[float] = None,
        min_rating: Optional[float] = None,
        require_rated: bool = False,
        require_enrollment: bool = False,
    ) -> List[Dict]:
        raw_hits = self.semantic_search(query, top_k=top_k_retrieval)

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
                "final_score": float(course.get("final_score", 0.0)),
            })
        return results
import pickle
from pathlib import Path
from typing import Optional

import faiss
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = Path("course_artifacts")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K_RETRIEVAL = 30
TOP_K_FINAL = 10

DEFAULT_MAX_HOURS = None
DEFAULT_MIN_RATING = 4.5
DEFAULT_ALLOWED_DIFFICULTIES = None
DEFAULT_REQUIRE_RATED = False
DEFAULT_REQUIRE_ENROLLMENT = False

model = SentenceTransformer(EMBEDDING_MODEL)
index = faiss.read_index(str(OUTPUT_DIR / "course_faiss.index"))

with open(OUTPUT_DIR / "course_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


# =========================================================
# SEMANTIC SEARCH
# =========================================================
def semantic_search(user_query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    query_vec = model.encode(
        [user_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        row = dict(metadata[idx])
        row["semantic_score"] = float(score)
        results.append(row)
    return results


# =========================================================
# FILTERING
# =========================================================
def passes_filters(
    course: dict,
    max_hours: Optional[float] = None,
    min_rating: Optional[float] = None,
    allowed_difficulties: Optional[set[str]] = None,
    require_rated: bool = False,
    require_enrollment: bool = False,
) -> bool:
    difficulty = str(course.get("Difficulty Level_Clean", "UNKNOWN") or "UNKNOWN")
    hours = course.get("Hours_Final", None)
    rating = course.get("Course Rating_Clean", None)
    has_rating = int(course.get("Has_Rating", 0) or 0)
    has_enrollment = int(course.get("Has_Enrollment", 0) or 0)

    if allowed_difficulties is not None and difficulty not in allowed_difficulties:
        return False

    if max_hours is not None and hours is not None:
        try:
            if float(hours) > max_hours:
                return False
        except Exception:
            return False

    if require_rated and not has_rating:
        return False

    if min_rating is not None:
        try:
            if float(rating) < min_rating:
                return False
        except Exception:
            return False

    if require_enrollment and not has_enrollment:
        return False

    return True


# =========================================================
# RANKING COMPONENTS
# =========================================================
def rating_component(course: dict) -> float:
    has_rating = int(course.get("Has_Rating", 0) or 0)
    rating = float(course.get("Course Rating_Clean", 0) or 0.0)

    if not has_rating:
        return 0.02

    return 0.15 * (rating / 5.0)


def popularity_component(course: dict) -> float:
    has_enrollment = int(course.get("Has_Enrollment", 0) or 0)
    log_enrollment = float(course.get("Enrollment_Count_Log1p", 0.0) or 0.0)

    if not has_enrollment:
        return 0.02

    normalized = min(log_enrollment / 14.0, 1.0)
    return 0.12 * normalized


def hours_component(course: dict, preferred_max_hours: Optional[float] = None) -> float:
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


def rerank_results(
    candidates: list[dict],
    preferred_max_hours: Optional[float] = None,
) -> list[dict]:
    reranked = []

    for course in candidates:
        semantic_score = float(course.get("semantic_score", 0.0))

        final_score = (
            0.65 * semantic_score
            + rating_component(course)
            + popularity_component(course)
            + hours_component(course, preferred_max_hours=preferred_max_hours)
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
# MAIN SEARCH FUNCTION
# =========================================================
def search_courses(
    user_query: str,
    top_k: int = TOP_K_FINAL,
    retrieval_k: int = TOP_K_RETRIEVAL,
    max_hours: Optional[float] = DEFAULT_MAX_HOURS,
    min_rating: Optional[float] = DEFAULT_MIN_RATING,
    allowed_difficulties: Optional[set[str]] = DEFAULT_ALLOWED_DIFFICULTIES,
    require_rated: bool = DEFAULT_REQUIRE_RATED,
    require_enrollment: bool = DEFAULT_REQUIRE_ENROLLMENT,
) -> list[dict]:
    raw_hits = semantic_search(user_query, top_k=retrieval_k)

    filtered_hits = [
        course for course in raw_hits
        if passes_filters(
            course,
            max_hours=max_hours,
            min_rating=min_rating,
            allowed_difficulties=allowed_difficulties,
            require_rated=require_rated,
            require_enrollment=require_enrollment,
        )
    ]

    reranked = rerank_results(
        filtered_hits,
        preferred_max_hours=max_hours,
    )

    results = []
    for course in reranked[:top_k]:
        results.append({
            "semantic_score": float(course["semantic_score"]),
            "final_score": float(course["final_score"]),
            "course_name": course.get("Course Name", ""),
            "partner": course.get("University / Industry Partner Name", ""),
            "difficulty": course.get("Difficulty Level_Clean", "UNKNOWN"),
            "domain": course.get("Domain_Clean", ""),
            "sub_domain": course.get("Sub-Domain_Clean", ""),
            "skills": course.get("Unified Skills Text", ""),
            "rating": course.get("Course Rating_Clean", 0.0),
            "hours": course.get("Hours_Final", None),
            "enrollment_count": course.get("Enrollment_Count_Clean", 0.0),
            "enrollment_log1p": course.get("Enrollment_Count_Log1p", 0.0),
            "url": course.get("Course URL", ""),
        })
    return results


# =========================================================
# TEST QUERY
# =========================================================
query = "advanced AI deployment with kubernetes"

results = search_courses(user_query=query)

print(f"\nQuery: {query}")
for i, r in enumerate(results, 1):
    print(f"\nRank {i}")
    print(f"Final Score: {r['final_score']:.4f}")
    print(f"Semantic Score: {r['semantic_score']:.4f}")
    print(f"Course: {r['course_name']}")
    print(f"Partner: {r['partner']}")
    print(f"Difficulty: {r['difficulty']}")
    print(f"Domain: {r['domain']} | Sub-domain: {r['sub_domain']}")
    print(f"Hours: {r['hours']} | Rating: {r['rating']}")
    print(f"Enrollment Count: {r['enrollment_count']} | Log1p: {r['enrollment_log1p']}")
    print(f"Skills: {r['skills']}")
    print(f"URL: {r['url']}")
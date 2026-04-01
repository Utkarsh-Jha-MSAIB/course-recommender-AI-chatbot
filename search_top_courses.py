import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIG
# =========================================================
OUTPUT_DIR = Path("course_artifacts")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

RUN_NAME = "generic_course_search"
EXCEL_OUTPUT = f"{RUN_NAME}_shortlist.xlsx"

SEARCH_QUERIES = [
    "artificial intelligence courses",
    "machine learning with python",
    "data analysis and visualization",
    "cloud computing and deployment",
    "generative AI tools",
]

TOP_K_PER_QUERY = 20
FINAL_SHORTLIST_SIZE = 25

# Optional metadata filters
ALLOWED_DIFFICULTIES = None
MAX_HOURS = None
MIN_RATING = 4.5
REQUIRE_RATED = False
REQUIRE_ENROLLMENT = False

# Optional mild domain preference
PREFERRED_DOMAINS = None


# =========================================================
# LOAD ARTIFACTS
# =========================================================
def load_artifacts(output_dir: Path, model_name: str):
    index = faiss.read_index(str(output_dir / "course_faiss.index"))
    with open(output_dir / "course_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    model = SentenceTransformer(model_name)
    return model, index, metadata


# =========================================================
# SEMANTIC SEARCH
# =========================================================
def semantic_search(model, index, metadata, query: str, top_k: int = 20) -> list[dict]:
    query_vec = model.encode(
        [query],
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
        row["matched_query"] = query
        results.append(row)
    return results


# =========================================================
# FILTERING
# =========================================================
def passes_filters(course: dict) -> bool:
    difficulty = str(course.get("Difficulty Level_Clean", "UNKNOWN") or "UNKNOWN")
    hours = course.get("Hours_Final", None)
    rating = course.get("Course Rating_Clean", None)
    has_rating = int(course.get("Has_Rating", 0) or 0)
    has_enrollment = int(course.get("Has_Enrollment", 0) or 0)

    if ALLOWED_DIFFICULTIES is not None and difficulty not in ALLOWED_DIFFICULTIES:
        return False

    if MAX_HOURS is not None and hours is not None:
        try:
            if float(hours) > MAX_HOURS:
                return False
        except Exception:
            return False

    if REQUIRE_RATED and not has_rating:
        return False

    if MIN_RATING is not None:
        try:
            if float(rating) < MIN_RATING:
                return False
        except Exception:
            return False

    if REQUIRE_ENROLLMENT and not has_enrollment:
        return False

    return True


# =========================================================
# RANKING COMPONENTS
# =========================================================
def domain_component(course: dict) -> float:
    if PREFERRED_DOMAINS is None:
        return 0.0
    domain = str(course.get("Domain_Clean", "") or "")
    return 0.04 if domain in PREFERRED_DOMAINS else 0.0


def rating_component(course: dict) -> float:
    has_rating = int(course.get("Has_Rating", 0) or 0)
    rating = float(course.get("Course Rating_Clean", 0.0) or 0.0)

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


def hours_component(course: dict) -> float:
    hours = course.get("Hours_Final", None)
    if hours is None:
        return 0.0

    try:
        hours = float(hours)
    except Exception:
        return 0.0

    if MAX_HOURS is not None:
        if hours <= MAX_HOURS:
            ratio = hours / max(MAX_HOURS, 1.0)
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
# AGGREGATION
# =========================================================
def aggregate_results(all_hits: list[dict]) -> pd.DataFrame:
    grouped = defaultdict(lambda: {
        "Course Name": None,
        "University / Industry Partner Name": None,
        "Type of Content": None,
        "Difficulty Level_Clean": None,
        "Hours_Final": None,
        "Course Rating_Clean": None,
        "Has_Rating": None,
        "Enrollment_Count_Clean": None,
        "Has_Enrollment": None,
        "Enrollment_Count_Log1p": None,
        "Domain_Clean": None,
        "Sub-Domain_Clean": None,
        "Course Language_Clean": None,
        "Unified Skills Text": None,
        "Course URL": None,
        "Course ID": None,
        "Specialization": None,
        "New Course Flag": None,
        "best_semantic_score": -1.0,
        "mean_semantic_score": 0.0,
        "semantic_score_sum": 0.0,
        "semantic_score_count": 0,
        "query_matches": 0,
        "matched_queries": set(),
    })

    for hit in all_hits:
        course_key = hit.get("Course ID") or hit.get("Course URL") or hit.get("Course Name")
        bucket = grouped[course_key]

        for key in list(bucket.keys()):
            if key in {
                "best_semantic_score",
                "mean_semantic_score",
                "semantic_score_sum",
                "semantic_score_count",
                "query_matches",
                "matched_queries",
            }:
                continue
            if bucket[key] is None and key in hit:
                bucket[key] = hit[key]

        score = float(hit.get("semantic_score", 0.0))
        if score > bucket["best_semantic_score"]:
            bucket["best_semantic_score"] = score

        bucket["semantic_score_sum"] += score
        bucket["semantic_score_count"] += 1

        matched_query = hit.get("matched_query", "")
        if matched_query and matched_query not in bucket["matched_queries"]:
            bucket["matched_queries"].add(matched_query)
            bucket["query_matches"] += 1

    rows = []
    for _, item in grouped.items():
        if item["semantic_score_count"] > 0:
            item["mean_semantic_score"] = (
                item["semantic_score_sum"] / item["semantic_score_count"]
            )
        else:
            item["mean_semantic_score"] = 0.0

        item["matched_queries"] = " | ".join(sorted(item["matched_queries"]))
        query_coverage_component = min(item["query_matches"], 5) * 0.06

        item["final_score"] = (
            0.50 * float(item["best_semantic_score"])
            + 0.10 * float(item["mean_semantic_score"])
            + rating_component(item)
            + popularity_component(item)
            + hours_component(item)
            + domain_component(item)
            + query_coverage_component
        )

        rows.append(item)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(
        by=[
            "final_score",
            "query_matches",
            "best_semantic_score",
            "mean_semantic_score",
            "Course Rating_Clean",
            "Enrollment_Count_Log1p",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)

    return df


# =========================================================
# EXPORT
# =========================================================
def export_shortlist_excel(
    shortlist_df: pd.DataFrame,
    all_hits_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    out_path: str,
):
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        shortlist_df.to_excel(writer, sheet_name="Shortlist", index=False)
        all_hits_df.to_excel(writer, sheet_name="All Query Hits", index=False)
        queries_df.to_excel(writer, sheet_name="Queries Used", index=False)


# =========================================================
# MAIN
# =========================================================
def main():
    print("Loading model, FAISS index, and metadata...")
    model, index, metadata = load_artifacts(OUTPUT_DIR, EMBEDDING_MODEL)

    print(f"Running {len(SEARCH_QUERIES)} search queries...")
    all_hits = []

    for query in SEARCH_QUERIES:
        hits = semantic_search(model, index, metadata, query, top_k=TOP_K_PER_QUERY)
        filtered_hits = [hit for hit in hits if passes_filters(hit)]
        all_hits.extend(filtered_hits)

        print(f"- Query: {query}")
        print(f"  Raw hits: {len(hits)} | After filters: {len(filtered_hits)}")

    if not all_hits:
        print("No courses passed the current filters.")
        return

    all_hits_df = pd.DataFrame(all_hits)
    aggregated_df = aggregate_results(all_hits)

    if aggregated_df.empty:
        print("Aggregation returned no rows.")
        return

    shortlist_df = aggregated_df.head(FINAL_SHORTLIST_SIZE).copy()

    preferred_cols = [
        "final_score",
        "best_semantic_score",
        "mean_semantic_score",
        "query_matches",
        "Enrollment_Count_Clean",
        "Enrollment_Count_Log1p",
        "Course Name",
        "University / Industry Partner Name",
        "Type of Content",
        "Difficulty Level_Clean",
        "Hours_Final",
        "Course Rating_Clean",
        "Has_Rating",
        "Has_Enrollment",
        "Domain_Clean",
        "Sub-Domain_Clean",
        "Course Language_Clean",
        "Specialization",
        "Unified Skills Text",
        "matched_queries",
        "Course URL",
        "Course ID",
        "New Course Flag",
    ]
    shortlist_df = shortlist_df[[c for c in preferred_cols if c in shortlist_df.columns]]

    queries_df = pd.DataFrame({
        "run_name": [RUN_NAME] * len(SEARCH_QUERIES),
        "query": SEARCH_QUERIES,
    })

    export_shortlist_excel(shortlist_df, all_hits_df, queries_df, EXCEL_OUTPUT)

    print("\nDone.")
    print(f"Saved shortlist workbook: {EXCEL_OUTPUT}")
    print(f"Final shortlist size: {len(shortlist_df)}")

    print("\nTop 10 shortlisted courses:")
    for i, row in shortlist_df.head(10).iterrows():
        print(
            f"{i+1}. {row['Course Name']} | "
            f"{row['Difficulty Level_Clean']} | "
            f"Score={row['final_score']:.4f}"
        )


if __name__ == "__main__":
    main()
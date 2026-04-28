import argparse
import csv
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from dotenv import load_dotenv

from course_chatbot.recommender import CourseRecommender


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_GOLD_PATH = PROJECT_ROOT / "eval_data" / "rag_gold_standard.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval_data" / "rag_metrics"
DEFAULT_K_VALUES = (5, 10)
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def clean_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_course_name(value) -> str:
    return clean_text(value).casefold()


def parse_k_values(value: str) -> List[int]:
    k_values = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k <= 0:
            raise ValueError("k values must be positive integers.")
        k_values.append(k)
    if not k_values:
        raise ValueError("At least one k value is required.")
    return sorted(set(k_values))


def load_gold_records(path: Path) -> List[Dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


class GeminiJudge:
    def __init__(self, model_name: str, timeout: int = 60):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found. Add it to .env or set it in the environment.")

        self.model_name = model_name
        self.timeout = timeout
        self.url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={api_key}"
        )

    def generate_json(self, prompt: str, retries: int = 2) -> Dict:
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "topP": 0.9,
                "topK": 32,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json",
            },
        }
        last_error = None
        for attempt in range(1, retries + 1):
            request = urllib.request.Request(
                self.url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Gemini HTTP {exc.code}: {error_body}") from exc

            try:
                parts = body["candidates"][0]["content"]["parts"]
                text = "".join(part.get("text", "") for part in parts)
                return json.loads(text)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                recovered = self.recover_score_json(text if "text" in locals() else "")
                if recovered:
                    return recovered
                last_error = exc
                payload["contents"][0]["parts"][0]["text"] = (
                    f"{prompt}\n\nYour previous response was invalid JSON. "
                    "Return compact JSON only. No explanations. No markdown. No trailing commas."
                )
                if attempt < retries:
                    time.sleep(0.5 * attempt)

        raise RuntimeError(f"Gemini judge did not return valid JSON after {retries} attempts: {last_error}")

    def recover_score_json(self, text: str) -> Optional[Dict]:
        matches = re.findall(
            r'"id"\s*:\s*(\d+).*?"relevance_grade"\s*:\s*([0-3])',
            text or "",
            flags=re.DOTALL,
        )
        if not matches:
            return None
        return {
            "scores": [
                {"id": int(candidate_id), "relevance_grade": int(grade)}
                for candidate_id, grade in matches
            ]
        }


def compact_course_for_judge(course: Dict, idx: int) -> Dict:
    result = {
        "id": idx,
        "course_name": clean_text(course.get("course_name") or course.get("Course Name")),
        "partner": clean_text(course.get("partner") or course.get("University / Industry Partner Name")),
        "difficulty": clean_text(course.get("difficulty") or course.get("Difficulty Level_Clean")),
        "domain": clean_text(course.get("domain") or course.get("Domain_Clean")),
        "sub_domain": clean_text(course.get("sub_domain") or course.get("Sub-Domain_Clean")),
        "hours": course.get("hours") or course.get("Hours_Final"),
        "rating": course.get("rating") or course.get("Course Rating_Clean"),
        "skills": clean_text(course.get("skills") or course.get("Unified Skills Text")),
        "specialization": clean_text(course.get("specialization") or course.get("Specialization")),
    }
    if len(result["skills"]) > 350:
        result["skills"] = result["skills"][:350]
    return result


def build_gemini_judge_prompt(record: Dict, query: str, candidates: List[Dict]) -> str:
    profile = record.get("synthetic_profile", {})
    compact_candidates = [compact_course_for_judge(course, idx) for idx, course in enumerate(candidates)]

    return f"""
You are a strict course-recommendation relevance judge.

Score each candidate course for how well it matches the learner query and profile.

Learner query:
{query}

Learner profile:
{json.dumps(profile, ensure_ascii=False)}

Candidate courses:
{json.dumps(compact_candidates, ensure_ascii=False, indent=2)}

Return only valid JSON with this schema:
{{
  "scores": [
    {{
      "id": 0,
      "relevance_grade": 0
    }}
  ]
}}

Rules:
- Score every candidate exactly once using its id.
- relevance_grade must be 0, 1, 2, or 3.
- 3 = excellent match for topic, skill goal, level/time, and career goal.
- 2 = relevant but has some mismatch.
- 1 = weakly related.
- 0 = not relevant.
- Be strict; do not reward popularity unless the course is relevant.
""".strip()


def gemini_rerank_candidates(
    judge: GeminiJudge,
    record: Dict,
    query: str,
    candidates: List[Dict],
) -> List[Dict]:
    if not candidates:
        return candidates

    prompt = build_gemini_judge_prompt(record, query, candidates)
    response = judge.generate_json(prompt)
    scores = response.get("scores", [])
    score_by_id = {}

    for item in scores:
        if not isinstance(item, dict):
            continue
        try:
            candidate_id = int(item.get("id"))
            grade = int(item.get("relevance_grade", 0))
        except Exception:
            continue
        score_by_id[candidate_id] = max(0, min(3, grade))

    reranked = []
    for idx, course in enumerate(candidates):
        row = dict(course)
        row["gemini_relevance_grade"] = score_by_id.get(idx, 0)
        reranked.append(row)

    reranked.sort(
        key=lambda course: (
            course.get("gemini_relevance_grade", 0),
            course.get("final_score", 0.0),
            course.get("hybrid_score", 0.0),
        ),
        reverse=True,
    )
    return reranked


def query_for_record(record: Dict, query_field: str) -> str:
    if query_field == "synthetic_question":
        question = record.get("synthetic_question", {}).get("question", "")
        return clean_text(question)
    if query_field == "ideal_search_query":
        return clean_text(record.get("ideal_search_query", ""))
    if query_field == "combined":
        question = record.get("synthetic_question", {}).get("question", "")
        ideal = record.get("ideal_search_query", "")
        return clean_text(f"{question} {ideal}")
    if query_field == "original_chatbot":
        return build_original_chatbot_query(record, include_resume=False)
    if query_field == "original_chatbot_resume":
        return build_original_chatbot_query(record, include_resume=True)
    raise ValueError(f"Unsupported query field: {query_field}")


def build_original_chatbot_query(record: Dict, include_resume: bool) -> str:
    profile = record.get("synthetic_profile", {})
    resume = record.get("synthetic_resume", {})

    pieces = []
    level = clean_text(profile.get("level"))
    topic = clean_text(profile.get("topic"))
    subskill = clean_text(profile.get("subskill"))
    provider = clean_text(profile.get("provider"))
    background = clean_text(profile.get("background"))
    career_goal = clean_text(profile.get("career_goal"))

    if level and "no preference" not in level.lower():
        pieces.append(level)
    if topic:
        pieces.append(topic)
    if subskill:
        pieces.append(subskill)
    if provider and "no preference" not in provider.lower():
        pieces.append(f"from {provider}")
    if background:
        pieces.append(f"for someone with background in {background}")
    if career_goal:
        pieces.append(f"targeting {career_goal}")

    if include_resume:
        education = clean_text(resume.get("education"))
        experience = clean_text(resume.get("experience"))
        skills = clean_text(resume.get("skills"))
        interests = clean_text(resume.get("interests"))
        resume_goal = clean_text(resume.get("goal"))

        if education:
            pieces.append(f"education {education}")
        if experience:
            pieces.append(f"experience {experience}")
        if skills:
            pieces.append(f"resume skills {skills}")
        if interests:
            pieces.append(f"interests {interests}")
        if resume_goal and resume_goal.lower() != career_goal.lower():
            pieces.append(f"resume goal {resume_goal}")

    return clean_text(" ".join(pieces)) or clean_text(record.get("ideal_search_query"))


def relevant_names_for_record(record: Dict) -> List[str]:
    synthetic_sources = record.get("synthetic_question", {}).get("source_course_names")
    if isinstance(synthetic_sources, list) and synthetic_sources:
        return [clean_text(name) for name in synthetic_sources]

    expected = record.get("gold_standard", {}).get("expected_course_names")
    if isinstance(expected, list) and expected:
        return [clean_text(name) for name in expected]

    return [clean_text(course.get("course_name")) for course in record.get("courses", [])]


def normalize_time_to_hours(time_answer: str) -> Optional[float]:
    text = clean_text(time_answer).lower()
    if not text or "no preference" in text or "any" in text or "flexible" in text:
        return None
    if "less than 5" in text:
        return 5.0
    if "5-15" in text or "5 to 15" in text:
        return 15.0
    if "more than 15" in text:
        return None

    range_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)", text)
    if range_match:
        return float(range_match.group(2))

    single_match = re.search(r"(\d+(?:\.\d+)?)", text)
    if single_match:
        return float(single_match.group(1))

    return None


def allowed_difficulties_for_level(level_answer: str) -> Optional[Set[str]]:
    text = clean_text(level_answer).lower()
    if not text or "no preference" in text or "any" in text:
        return None
    if "beginner" in text:
        return {"BEGINNER", "UNKNOWN"}
    if "intermediate" in text:
        return {"INTERMEDIATE"}
    if "advanced" in text:
        return {"ADVANCED"}
    return None


def profile_filters(record: Dict, enabled: bool) -> Dict:
    if not enabled:
        return {"allowed_difficulties": None, "max_hours": None}

    profile = record.get("synthetic_profile", {})
    return {
        "allowed_difficulties": allowed_difficulties_for_level(profile.get("level", "")),
        "max_hours": normalize_time_to_hours(profile.get("time", "")),
    }


def dcg(relevance_values: Sequence[float]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevance_values))


def ndcg_at_k(retrieved_names: List[str], grade_by_name: Dict[str, int], k: int) -> float:
    relevance = [grade_by_name.get(name, 0) for name in retrieved_names[:k]]
    ideal = sorted(grade_by_name.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(relevance) / ideal_dcg


def grade_lookup(record: Dict) -> Dict[str, int]:
    grades = {}
    for item in record.get("gold_standard", {}).get("relevance_judgments", []):
        if not isinstance(item, dict):
            continue
        name = normalize_course_name(item.get("course_name"))
        try:
            grade = int(item.get("relevance_grade", 0))
        except Exception:
            grade = 0
        if name:
            grades[name] = max(0, min(3, grade))
    return grades


def evaluate_records(
    records: Iterable[Dict],
    recommender: CourseRecommender,
    k_values: Sequence[int],
    query_field: str,
    top_k_retrieval: int,
    search_mode: str,
    hybrid_alpha: float,
    reranker: str,
    candidate_k: int,
    gemini_judge: Optional[GeminiJudge],
    gemini_delay_seconds: float,
    use_profile_filters: bool,
) -> List[Dict]:
    max_k = max(k_values)
    rows = []

    for record in records:
        record_id = record.get("record_id", "")
        query = query_for_record(record, query_field)
        relevant_names = {normalize_course_name(name) for name in relevant_names_for_record(record)}
        relevant_names.discard("")
        filters = profile_filters(record, enabled=use_profile_filters)

        retrieved = recommender.search_courses(
            query=query,
            top_k_retrieval=max(top_k_retrieval, max_k),
            top_k_final=max(max_k, candidate_k if reranker == "gemini" else max_k),
            search_mode=search_mode,
            hybrid_alpha=hybrid_alpha,
            allowed_difficulties=filters["allowed_difficulties"],
            max_hours=filters["max_hours"],
            min_rating=None,
            require_rated=False,
            require_enrollment=False,
        )

        if reranker == "gemini":
            if gemini_judge is None:
                raise ValueError("Gemini reranker selected without a Gemini judge.")
            retrieved = gemini_rerank_candidates(
                judge=gemini_judge,
                record=record,
                query=query,
                candidates=retrieved[:candidate_k],
            )
            if gemini_delay_seconds > 0:
                time.sleep(gemini_delay_seconds)

        retrieved_names = [normalize_course_name(course.get("course_name")) for course in retrieved]
        grades = grade_lookup(record)

        for k in k_values:
            names_at_k = retrieved_names[:k]
            hits = len(set(names_at_k) & relevant_names)
            precision = hits / k
            recall = hits / len(relevant_names) if relevant_names else 0.0
            ndcg = ndcg_at_k(names_at_k, grades, k)

            rows.append(
                {
                    "record_id": record_id,
                    "query_field": query_field,
                    "k": k,
                    "precision": precision,
                    "recall": recall,
                    "ndcg": ndcg,
                    "hits": hits,
                    "relevant_count": len(relevant_names),
                    "retrieved_count": len(names_at_k),
                    "query": query,
                    "expected_courses": " | ".join(relevant_names_for_record(record)),
                    "retrieved_courses": " | ".join(course.get("course_name", "") for course in retrieved[:k]),
                }
            )

    return rows


def summarize(rows: Iterable[Dict]) -> List[Dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[int(row["k"])].append(row)

    summary = []
    for k in sorted(buckets):
        items = buckets[k]
        summary.append(
            {
                "k": k,
                "num_queries": len(items),
                "precision": sum(item["precision"] for item in items) / len(items),
                "recall": sum(item["recall"] for item in items) / len(items),
                "ndcg": sum(item["ndcg"] for item in items) / len(items),
            }
        )
    return summary


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_rows: List[Dict]) -> None:
    print("\nRAG retrieval metrics")
    print("k  queries  precision  recall  ndcg")
    for row in summary_rows:
        print(
            f"{row['k']:>2} {row['num_queries']:>8} "
            f"{row['precision']:.4f}     {row['recall']:.4f}  {row['ndcg']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate course recommender retrieval against rag_gold_standard.jsonl."
    )
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument(
        "--query-field",
        choices=[
            "synthetic_question",
            "ideal_search_query",
            "combined",
            "original_chatbot",
            "original_chatbot_resume",
        ],
        default="original_chatbot_resume",
        help="Gold-record field to use as the retrieval query.",
    )
    parser.add_argument("--top-k-retrieval", type=int, default=50)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N records. Useful for Gemini reranker smoke tests.",
    )
    parser.add_argument(
        "--search-mode",
        choices=["semantic", "lexical", "hybrid"],
        default="hybrid",
        help="Retrieval strategy to evaluate.",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.3,
        help="Semantic weight for hybrid search. Lexical weight is 1 - alpha.",
    )
    parser.add_argument(
        "--reranker",
        choices=["none", "gemini"],
        default="none",
        help="Optional second-stage reranker for retrieved candidate courses.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Number of retrieved candidates to pass to the reranker.",
    )
    parser.add_argument("--gemini-model", default=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
    parser.add_argument("--gemini-timeout", type=int, default=60)
    parser.add_argument("--gemini-delay-seconds", type=float, default=0.0)
    parser.add_argument(
        "--use-profile-filters",
        action="store_true",
        help="Apply synthetic profile time and level filters, closer to the app flow.",
    )
    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)
    records = load_gold_records(args.gold)
    if args.limit is not None:
        records = records[:args.limit]
    recommender = CourseRecommender()
    gemini_judge = None
    if args.reranker == "gemini":
        gemini_judge = GeminiJudge(model_name=args.gemini_model, timeout=args.gemini_timeout)

    detail_rows = evaluate_records(
        records=records,
        recommender=recommender,
        k_values=k_values,
        query_field=args.query_field,
        top_k_retrieval=args.top_k_retrieval,
        search_mode=args.search_mode,
        hybrid_alpha=args.hybrid_alpha,
        reranker=args.reranker,
        candidate_k=args.candidate_k,
        gemini_judge=gemini_judge,
        gemini_delay_seconds=args.gemini_delay_seconds,
        use_profile_filters=args.use_profile_filters,
    )
    summary_rows = summarize(detail_rows)

    suffix = f"{args.query_field}_{args.search_mode}"
    if args.search_mode == "hybrid":
        alpha_label = str(args.hybrid_alpha).replace(".", "p")
        suffix += f"_alpha_{alpha_label}"
    suffix += f"_retrieval_{args.top_k_retrieval}"
    if args.reranker != "none":
        suffix += f"_{args.reranker}_rerank"
    if args.limit is not None:
        suffix += f"_limit_{args.limit}"
    if args.use_profile_filters:
        suffix += "_profile_filters"
    summary_path = args.output_dir / f"rag_metrics_summary_{suffix}.csv"
    write_csv(summary_path, summary_rows)

    print_summary(summary_rows)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()

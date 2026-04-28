import argparse
import csv
import hashlib
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
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "course_artifacts"
DEFAULT_GOLD_PATH = PROJECT_ROOT / "eval_data" / "rag_gold_standard.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval_data" / "rag_metrics"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "eval_data" / "embedding_cache"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_K_VALUES = (5, 10)


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


def load_metadata(artifact_dir: Path) -> List[Dict]:
    metadata_path = artifact_dir / "course_metadata.pkl"
    with metadata_path.open("rb") as f:
        return pickle.load(f)


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
        skills = clean_text(resume.get("skills"))
        interests = clean_text(resume.get("interests"))
        resume_goal = clean_text(resume.get("goal"))

        if skills:
            pieces.append(f"resume skills {skills}")
        if interests:
            pieces.append(f"interests {interests}")
        if resume_goal and resume_goal.lower() != career_goal.lower():
            pieces.append(f"resume goal {resume_goal}")

    return clean_text(" ".join(pieces)) or clean_text(record.get("ideal_search_query"))


def query_for_record(record: Dict, query_field: str) -> str:
    if query_field == "synthetic_question":
        return clean_text(record.get("synthetic_question", {}).get("question", ""))
    if query_field == "ideal_search_query":
        return clean_text(record.get("ideal_search_query", ""))
    if query_field == "combined":
        return clean_text(
            f"{record.get('synthetic_question', {}).get('question', '')} "
            f"{record.get('ideal_search_query', '')}"
        )
    if query_field == "original_chatbot":
        return build_original_chatbot_query(record, include_resume=False)
    if query_field == "original_chatbot_resume":
        return build_original_chatbot_query(record, include_resume=True)
    raise ValueError(f"Unsupported query field: {query_field}")


def relevant_names_for_record(record: Dict) -> List[str]:
    expected = record.get("gold_standard", {}).get("expected_course_names")
    if isinstance(expected, list) and expected:
        return [clean_text(name) for name in expected]
    return [clean_text(course.get("course_name")) for course in record.get("courses", [])]


def course_key(course: Dict) -> str:
    return (
        str(course.get("Course ID") or "").strip()
        or str(course.get("Course URL") or "").strip()
        or str(course.get("Course Name") or "").strip().lower()
    )


def course_text(course: Dict) -> str:
    if course.get("course_text_for_embedding"):
        return clean_text(course.get("course_text_for_embedding"))
    parts = [
        course.get("Course Name"),
        course.get("University / Industry Partner Name"),
        course.get("Type of Content"),
        course.get("Difficulty Level_Clean"),
        course.get("Domain_Clean"),
        course.get("Sub-Domain_Clean"),
        course.get("Course Language_Clean"),
        course.get("Unified Skills Text"),
        course.get("Course Description_Clean"),
        course.get("Specialization"),
    ]
    return clean_text(" | ".join(str(part or "") for part in parts))


class GeminiEmbeddingClient:
    def __init__(self, model_name: str, timeout: int, delay_seconds: float):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found. Add it to .env or set it in the environment.")
        self.model_name = model_name.removeprefix("models/")
        self.timeout = timeout
        self.delay_seconds = delay_seconds
        self.url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:embedContent?key={api_key}"
        )

    def embed(self, text: str, task_type: str) -> List[float]:
        payload = {
            "model": f"models/{self.model_name}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
        }
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
            raise RuntimeError(f"Gemini embedding HTTP {exc.code}: {error_body}") from exc

        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

        values = body.get("embedding", {}).get("values")
        if not values:
            raise RuntimeError(f"Gemini embedding response did not include values: {body}")
        return [float(value) for value in values]


class EmbeddingCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, model_name: str, task_type: str, text: str) -> Path:
        digest = hashlib.sha256(f"{model_name}\0{task_type}\0{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, model_name: str, task_type: str, text: str) -> Optional[List[float]]:
        path = self.path_for(model_name, task_type, text)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, model_name: str, task_type: str, text: str, embedding: List[float]) -> None:
        path = self.path_for(model_name, task_type, text)
        path.write_text(json.dumps(embedding), encoding="utf-8")


def cached_embed(
    client: GeminiEmbeddingClient,
    cache: EmbeddingCache,
    text: str,
    task_type: str,
) -> List[float]:
    cached = cache.get(client.model_name, task_type, text)
    if cached is not None:
        return cached
    embedding = client.embed(text, task_type=task_type)
    cache.set(client.model_name, task_type, text, embedding)
    return embedding


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def build_or_load_course_embeddings(
    metadata: List[Dict],
    client: GeminiEmbeddingClient,
    cache: EmbeddingCache,
    output_path: Path,
    rebuild: bool,
) -> np.ndarray:
    if output_path.exists() and not rebuild:
        return np.load(output_path)

    embeddings = []
    for idx, course in enumerate(metadata, start=1):
        text = course_text(course)
        embedding = cached_embed(client, cache, text, task_type="RETRIEVAL_DOCUMENT")
        embeddings.append(embedding)
        if idx % 25 == 0 or idx == len(metadata):
            print(f"Embedded {idx}/{len(metadata)} course documents", flush=True)

    matrix = normalize_matrix(np.array(embeddings, dtype=np.float32))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, matrix)
    return matrix


def tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(token) > 1]


def build_lexical_index(metadata: List[Dict]) -> Tuple[Dict[str, List[Tuple[int, int]]], List[int], float]:
    inverted_index = defaultdict(list)
    doc_lengths = []
    for doc_idx, course in enumerate(metadata):
        token_counts = Counter(tokenize(course_text(course)))
        doc_length = sum(token_counts.values())
        doc_lengths.append(doc_length)
        for token, count in token_counts.items():
            inverted_index[token].append((doc_idx, count))
    avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    return dict(inverted_index), doc_lengths, avg_doc_length


def lexical_scores(
    query: str,
    metadata: List[Dict],
    lexical_index: Dict[str, List[Tuple[int, int]]],
    doc_lengths: List[int],
    avg_doc_length: float,
) -> Dict[int, float]:
    query_counts = Counter(tokenize(query))
    scores = defaultdict(float)
    num_docs = len(metadata)
    k1 = 1.5
    b = 0.75
    for token, query_count in query_counts.items():
        postings = lexical_index.get(token)
        if not postings:
            continue
        doc_freq = len(postings)
        idf = math.log(1 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        for doc_idx, term_freq in postings:
            doc_length = doc_lengths[doc_idx] or 0
            length_norm = 1 - b
            if avg_doc_length:
                length_norm += b * (doc_length / avg_doc_length)
            denom = term_freq + k1 * length_norm
            score = idf * ((term_freq * (k1 + 1)) / denom)
            scores[doc_idx] += score * query_count
    return dict(scores)


def normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {key: 1.0 for key in scores}
    return {key: (score - min_score) / (max_score - min_score) for key, score in scores.items()}


def rating_component(course: Dict) -> float:
    has_rating = int(course.get("Has_Rating", 0) or 0)
    rating = float(course.get("Course Rating_Clean", 0.0) or 0.0)
    return 0.02 if not has_rating else 0.15 * (rating / 5.0)


def popularity_component(course: Dict) -> float:
    has_enrollment = int(course.get("Has_Enrollment", 0) or 0)
    log_enrollment = float(course.get("Enrollment_Count_Log1p", 0.0) or 0.0)
    if not has_enrollment:
        return 0.02
    return 0.12 * min(log_enrollment / 14.0, 1.0)


def hours_component(course: Dict) -> float:
    hours = course.get("Hours_Final")
    if hours is None:
        return 0.0
    try:
        hours = float(hours)
    except Exception:
        return 0.0
    if hours <= 5:
        return 0.05
    if hours <= 15:
        return 0.06
    if hours <= 30:
        return 0.04
    return 0.01


def retrieve_courses(
    query: str,
    metadata: List[Dict],
    course_embeddings: np.ndarray,
    client: GeminiEmbeddingClient,
    cache: EmbeddingCache,
    lexical_index: Dict[str, List[Tuple[int, int]]],
    doc_lengths: List[int],
    avg_doc_length: float,
    top_k_retrieval: int,
    top_k_final: int,
    hybrid_alpha: float,
) -> List[Dict]:
    query_embedding = np.array(
        cached_embed(client, cache, query, task_type="RETRIEVAL_QUERY"),
        dtype=np.float32,
    )
    query_embedding = query_embedding / max(np.linalg.norm(query_embedding), 1e-12)
    semantic_raw = course_embeddings @ query_embedding
    semantic_top_indices = np.argsort(-semantic_raw)[:top_k_retrieval]
    semantic_scores = {int(idx): float(semantic_raw[idx]) for idx in semantic_top_indices}

    lexical_raw = lexical_scores(query, metadata, lexical_index, doc_lengths, avg_doc_length)
    lexical_top_indices = sorted(lexical_raw, key=lexical_raw.get, reverse=True)[:top_k_retrieval]
    lexical_top_scores = {idx: lexical_raw[idx] for idx in lexical_top_indices}

    semantic_norm = normalize_scores(semantic_scores)
    lexical_norm = normalize_scores(lexical_top_scores)
    candidate_indices = set(semantic_norm) | set(lexical_norm)

    rows = []
    for idx in candidate_indices:
        course = dict(metadata[idx])
        hybrid_score = (
            hybrid_alpha * semantic_norm.get(idx, 0.0)
            + (1.0 - hybrid_alpha) * lexical_norm.get(idx, 0.0)
        )
        final_score = (
            0.65 * hybrid_score
            + rating_component(course)
            + popularity_component(course)
            + hours_component(course)
        )
        course["semantic_score"] = float(semantic_scores.get(idx, 0.0))
        course["lexical_score"] = float(lexical_top_scores.get(idx, 0.0))
        course["hybrid_score"] = float(hybrid_score)
        course["final_score"] = float(final_score)
        rows.append(course)

    rows.sort(
        key=lambda course: (
            course["final_score"],
            course["hybrid_score"],
            course["semantic_score"],
            course["lexical_score"],
        ),
        reverse=True,
    )
    return rows[:top_k_final]


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


def dcg(relevance_values: Sequence[float]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevance_values))


def ndcg_at_k(retrieved_names: List[str], grade_by_name: Dict[str, int], k: int) -> float:
    relevance = [grade_by_name.get(name, 0) for name in retrieved_names[:k]]
    ideal = sorted(grade_by_name.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(relevance) / ideal_dcg


def evaluate(
    records: Iterable[Dict],
    metadata: List[Dict],
    course_embeddings: np.ndarray,
    client: GeminiEmbeddingClient,
    cache: EmbeddingCache,
    k_values: Sequence[int],
    query_field: str,
    top_k_retrieval: int,
    hybrid_alpha: float,
) -> List[Dict]:
    lexical_index, doc_lengths, avg_doc_length = build_lexical_index(metadata)
    max_k = max(k_values)
    rows = []

    for record in records:
        query = query_for_record(record, query_field)
        relevant_names = {normalize_course_name(name) for name in relevant_names_for_record(record)}
        retrieved = retrieve_courses(
            query=query,
            metadata=metadata,
            course_embeddings=course_embeddings,
            client=client,
            cache=cache,
            lexical_index=lexical_index,
            doc_lengths=doc_lengths,
            avg_doc_length=avg_doc_length,
            top_k_retrieval=top_k_retrieval,
            top_k_final=max_k,
            hybrid_alpha=hybrid_alpha,
        )
        retrieved_names = [normalize_course_name(course.get("Course Name")) for course in retrieved]
        grades = grade_lookup(record)

        for k in k_values:
            names_at_k = retrieved_names[:k]
            hits = len(set(names_at_k) & relevant_names)
            rows.append(
                {
                    "k": k,
                    "precision": hits / k,
                    "recall": hits / len(relevant_names) if relevant_names else 0.0,
                    "ndcg": ndcg_at_k(names_at_k, grades, k),
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
    print("\nGemini embedding ablation metrics")
    print("k  queries  precision  recall  ndcg")
    for row in summary_rows:
        print(
            f"{row['k']:>2} {row['num_queries']:>8} "
            f"{row['precision']:.4f}     {row['recall']:.4f}  {row['ndcg']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation evaluation using Gemini embeddings instead of local SentenceTransformer embeddings."
    )
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
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
    )
    parser.add_argument("--top-k-retrieval", type=int, default=100)
    parser.add_argument("--hybrid-alpha", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--delay-seconds", type=float, default=0.05)
    parser.add_argument("--rebuild-embeddings", action="store_true")
    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)
    records = load_gold_records(args.gold)
    if args.limit is not None:
        records = records[: args.limit]
    metadata = load_metadata(args.artifact_dir)
    client = GeminiEmbeddingClient(
        model_name=args.embedding_model,
        timeout=args.timeout,
        delay_seconds=args.delay_seconds,
    )
    cache = EmbeddingCache(args.cache_dir)

    course_embedding_path = (
        args.cache_dir
        / f"course_embeddings_{args.embedding_model}_{len(metadata)}.npy"
    )
    course_embeddings = build_or_load_course_embeddings(
        metadata=metadata,
        client=client,
        cache=cache,
        output_path=course_embedding_path,
        rebuild=args.rebuild_embeddings,
    )

    rows = evaluate(
        records=records,
        metadata=metadata,
        course_embeddings=course_embeddings,
        client=client,
        cache=cache,
        k_values=k_values,
        query_field=args.query_field,
        top_k_retrieval=args.top_k_retrieval,
        hybrid_alpha=args.hybrid_alpha,
    )
    summary_rows = summarize(rows)

    alpha_label = str(args.hybrid_alpha).replace(".", "p")
    suffix = (
        f"{args.query_field}_gemini_embedding_{args.embedding_model}"
        f"_alpha_{alpha_label}_retrieval_{args.top_k_retrieval}"
    )
    if args.limit is not None:
        suffix += f"_limit_{args.limit}"
    summary_path = args.output_dir / f"rag_metrics_summary_{suffix}.csv"
    write_csv(summary_path, summary_rows)

    print_summary(summary_rows)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()

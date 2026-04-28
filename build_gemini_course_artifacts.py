import argparse
import json
import os
import pickle
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "course_artifacts"
CACHE_DIR = PROJECT_ROOT / "eval_data" / "embedding_cache"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"


def clean_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def course_text_for_embedding(course: Dict) -> str:
    parts = [
        course.get("course_text_for_embedding"),
        course.get("Course Name"),
        course.get("University / Industry Partner Name"),
        course.get("Type of Content"),
        course.get("Difficulty Level_Clean"),
        course.get("Domain_Clean"),
        course.get("Sub-Domain_Clean"),
        course.get("Course Language_Clean"),
        course.get("Unified Skills Text"),
        course.get("Course Description_Clean"),
        course.get("Course Description"),
        course.get("Specialization"),
    ]
    return clean_text(" | ".join(str(part or "") for part in parts))


class GeminiEmbeddingClient:
    def __init__(self, model_name: str, timeout_seconds: int, delay_seconds: float):
        load_dotenv()
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

    def embed_document(self, text: str) -> List[float]:
        payload = {
            "model": f"models/{self.model_name}",
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_DOCUMENT",
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
        return [float(value) for value in values]


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = embeddings.astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def load_full_cache(cache_dir: Path, model_name: str, count: int) -> Optional[np.ndarray]:
    path = cache_dir / f"course_embeddings_{model_name}_{count}.npy"
    if not path.exists():
        return None
    return normalize_embeddings(np.load(path))


def build_embeddings_from_api(
    metadata: List[Dict],
    model_name: str,
    timeout_seconds: int,
    delay_seconds: float,
) -> np.ndarray:
    client = GeminiEmbeddingClient(model_name, timeout_seconds, delay_seconds)
    vectors = []
    for idx, course in enumerate(metadata, start=1):
        text = course_text_for_embedding(course)
        vectors.append(client.embed_document(text))
        if idx % 25 == 0:
            print(f"Embedded {idx}/{len(metadata)} courses")
    return normalize_embeddings(np.array(vectors, dtype="float32"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gemini course embeddings and FAISS index.")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--force-api", action="store_true")
    args = parser.parse_args()

    metadata_path = args.artifact_dir / "course_metadata.pkl"
    with metadata_path.open("rb") as f:
        metadata = pickle.load(f)

    model_name = args.embedding_model.removeprefix("models/")
    embeddings = None if args.force_api else load_full_cache(args.cache_dir, model_name, len(metadata))
    if embeddings is None:
        embeddings = build_embeddings_from_api(
            metadata=metadata,
            model_name=model_name,
            timeout_seconds=args.timeout_seconds,
            delay_seconds=args.delay_seconds,
        )

    if len(embeddings) != len(metadata):
        raise ValueError(f"Embedding count {len(embeddings)} does not match metadata count {len(metadata)}.")

    embedding_path = args.artifact_dir / f"course_embeddings_{model_name}.npy"
    index_path = args.artifact_dir / f"course_faiss_{model_name}.index"

    np.save(embedding_path, embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_path))

    print(f"Wrote embeddings: {embedding_path} shape={embeddings.shape}")
    print(f"Wrote FAISS index: {index_path} vectors={index.ntotal} dim={embeddings.shape[1]}")


if __name__ == "__main__":
    main()

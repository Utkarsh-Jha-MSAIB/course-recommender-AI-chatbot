import argparse
import json
import os
import pickle
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "course_artifacts"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "eval_data" / "rag_gold_standard.jsonl"
DEFAULT_MODEL_NAME = "gemini-2.5-flash"
QUESTION_CONSTRAINTS = [
    "Ask about which courses best fit a specific career goal.",
    "Ask as a student with limited study time.",
    "Ask as an early-career learner trying to build a portfolio.",
    "Ask about balancing practical tools with conceptual foundations.",
    "Ask as someone switching into the target field from another background.",
]


def clean_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def split_skills(value, limit: int = 8) -> List[str]:
    if isinstance(value, list):
        parts = value
    else:
        parts = re.split(r"[,;|]", str(value or ""))
    cleaned = []
    seen = set()
    for part in parts:
        skill = clean_text(part)
        key = skill.lower()
        if skill and key not in seen:
            cleaned.append(skill)
            seen.add(key)
        if len(cleaned) >= limit:
            break
    return cleaned


def safe_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def load_course_metadata(artifact_dir: Path) -> List[Dict]:
    metadata_path = artifact_dir / "course_metadata.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Run build_course_catalog.py first to create course_artifacts."
        )

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    if not isinstance(metadata, list) or not metadata:
        raise ValueError(f"{metadata_path} did not contain a non-empty course metadata list.")

    return metadata


def course_to_public_record(course: Dict) -> Dict:
    return {
        "course_name": clean_text(course.get("Course Name")),
        "partner": clean_text(course.get("University / Industry Partner Name")),
        "difficulty": clean_text(course.get("Difficulty Level_Clean") or "UNKNOWN"),
        "domain": clean_text(course.get("Domain_Clean")),
        "sub_domain": clean_text(course.get("Sub-Domain_Clean")),
        "hours": safe_float(course.get("Hours_Final")),
        "rating": safe_float(course.get("Course Rating_Clean")),
        "enrollment_count": safe_float(course.get("Enrollment_Count_Clean")),
        "skills": split_skills(course.get("Unified Skills Text")),
        "specialization": clean_text(course.get("Specialization")),
        "url": clean_text(course.get("Course URL")),
        "course_id": clean_text(course.get("Course ID")),
    }


def group_courses(courses: List[Dict]) -> Dict[Tuple[str, str], List[Dict]]:
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for course in courses:
        domain = clean_text(course.get("Domain_Clean")) or "General"
        sub_domain = clean_text(course.get("Sub-Domain_Clean")) or "General"
        if clean_text(course.get("Course Name")):
            groups[(domain, sub_domain)].append(course)
    return groups


def sample_course_sets(
    courses: List[Dict],
    count: int,
    seed: int,
    sizes: Tuple[int, ...] = (3,),
) -> List[List[Dict]]:
    rng = random.Random(seed)
    groups = group_courses(courses)
    eligible_groups = [items for items in groups.values() if len(items) >= min(sizes)]
    if not eligible_groups:
        raise ValueError("No course groups have enough rows to create 3-course combinations.")

    all_courses = [course for group in groups.values() for course in group]
    sets: List[List[Dict]] = []
    seen_signatures = set()
    attempts = 0
    max_attempts = count * 80

    while len(sets) < count and attempts < max_attempts:
        attempts += 1
        size = rng.choice(sizes)
        possible_groups = [group for group in eligible_groups if len(group) >= size]

        if possible_groups and rng.random() < 0.8:
            pool = rng.choice(possible_groups)
        else:
            pool = all_courses

        selected = rng.sample(pool, k=min(size, len(pool)))
        signature = tuple(sorted(clean_text(c.get("Course Name")).lower() for c in selected))
        if len(signature) != size or signature in seen_signatures:
            continue

        seen_signatures.add(signature)
        sets.append(selected)

    if len(sets) < count:
        raise RuntimeError(
            f"Only created {len(sets)} unique sets after {attempts} attempts. "
            "Try lowering --count or using a larger course catalog."
        )

    return sets


def infer_profile_seed(course_set: List[Dict]) -> Dict:
    domains = [clean_text(c.get("Domain_Clean")) for c in course_set if clean_text(c.get("Domain_Clean"))]
    sub_domains = [
        clean_text(c.get("Sub-Domain_Clean")) for c in course_set if clean_text(c.get("Sub-Domain_Clean"))
    ]
    difficulties = [
        clean_text(c.get("Difficulty Level_Clean")) for c in course_set if clean_text(c.get("Difficulty Level_Clean"))
    ]
    skills = []
    for course in course_set:
        skills.extend(split_skills(course.get("Unified Skills Text"), limit=5))

    def most_common(values: Iterable[str], fallback: str) -> str:
        counts = defaultdict(int)
        for value in values:
            counts[value] += 1
        if not counts:
            return fallback
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    hours = [safe_float(c.get("Hours_Final")) for c in course_set]
    hours = [h for h in hours if h is not None]
    max_hours = max(hours) if hours else None

    if max_hours is None:
        time_pref = "Flexible"
    elif max_hours <= 5:
        time_pref = "Less than 5 hours"
    elif max_hours <= 15:
        time_pref = "5-15 hours"
    else:
        time_pref = "More than 15 hours"

    level = most_common(difficulties, "No preference").title()
    if level == "Not Calibrated":
        level = "No preference"

    return {
        "topic": most_common(domains, "Career development"),
        "subskill": most_common(sub_domains, "General skills"),
        "level": level,
        "time": time_pref,
        "skills": ", ".join(split_skills(", ".join(skills), limit=8)),
    }


def question_constraint_for_record(record_id: str) -> str:
    index = sum(ord(char) for char in record_id) % len(QUESTION_CONSTRAINTS)
    return QUESTION_CONSTRAINTS[index]


def build_fallback_synthetic_question(record_id: str, course_set: List[Dict]) -> Dict:
    seed = infer_profile_seed(course_set)
    courses = [course_to_public_record(course) for course in course_set[:3]]
    names = [course["course_name"] for course in courses]
    constraint = question_constraint_for_record(record_id)
    topic = seed["topic"]
    subskill = seed["subskill"]
    level = seed["level"].lower()
    article = "an" if level[:1] in {"a", "e", "i", "o", "u"} else "a"

    return {
        "question": (
            f"I am {article} {level} learner interested in {topic} and {subskill}. "
            "Which three courses would help me build practical skills?"
        ),
        "answer": (
            "A strong three-course set would be "
            + ", ".join(names[:-1])
            + f", and {names[-1]}."
        ),
        "source_course_names": names,
        "constraint": constraint,
    }


def build_gemini_prompt(record_id: str, course_set: List[Dict]) -> str:
    seed = infer_profile_seed(course_set)
    courses = [course_to_public_record(course) for course in course_set]
    question_constraint = question_constraint_for_record(record_id)

    return f"""
You are creating gold-standard evaluation data for a course-recommender RAG system.

Given this fixed set of courses, create a realistic synthetic user profile and synthetic resume data
that would make these courses good recommendations. Then create gold-standard labels that future
RAG tests can compare against. The evaluation compares only the top three recommended courses.
Also create one synthetic learner question following this method: generate a natural question from
the source courses, then use the provided course names as known retrieval ground truth.

Seed hints:
- Topic: {seed["topic"]}
- Skill focus: {seed["subskill"]}
- Preferred level: {seed["level"]}
- Time preference: {seed["time"]}
- Skills to consider: {seed["skills"]}

Courses:
{json.dumps(courses, indent=2, ensure_ascii=False)}

Return only valid JSON. Do not wrap it in Markdown.

Required schema:
{{
  "record_id": "{record_id}",
  "synthetic_profile": {{
    "background": "...",
    "topic": "...",
    "subskill": "...",
    "time": "...",
    "provider": "No preference or a provider from the set",
    "level": "Beginner, Intermediate, Advanced, or No preference",
    "career_goal": "..."
  }},
  "synthetic_resume": {{
    "education": "...",
    "experience": "...",
    "skills": "...",
    "interests": "...",
    "goal": "..."
  }},
  "ideal_search_query": "...",
  "synthetic_question": {{
    "question": "...",
    "answer": "...",
    "source_course_names": ["..."],
    "constraint": "{question_constraint}"
  }},
  "gold_standard": {{
    "expected_course_names": ["..."],
    "relevance_judgments": [
      {{
        "course_name": "...",
        "relevance_grade": 0,
        "fit_reason": "...",
        "possible_tradeoff": "..."
      }}
    ],
    "expected_response": "...",
    "verification_notes": "..."
  }}
}}

Rules:
- Use exactly the three provided courses in expected_course_names.
- Include exactly three relevance_judgments, one for each provided course.
- synthetic_question.question must be natural, specific, concise, and answerable from the three courses.
- synthetic_question.question must not mention "this record", "the data", or "the provided courses".
- synthetic_question.answer must answer the question using the three course names.
- synthetic_question.source_course_names must contain exactly the three provided course names.
- Synthetic question style constraint: {question_constraint}
- relevance_grade must be 0, 1, 2, or 3, where 3 means highly relevant.
- expected_response should be under 120 words and should mention each course once.
- Keep fit_reason, possible_tradeoff, and verification_notes concise.
- Avoid quotation marks inside string values unless they are part of a course name.
- Do not invent course URLs, ratings, or hours.
- Keep the profile realistic for a student or early-career learner.
""".strip()


def extract_json_object(text: str) -> Dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def fallback_gold_record(record_id: str, course_set: List[Dict], note: str) -> Dict:
    seed = infer_profile_seed(course_set)
    courses = [course_to_public_record(course) for course in course_set]
    names = [course["course_name"] for course in courses]
    profile = {
        "background": "Student or early-career learner",
        "topic": seed["topic"],
        "subskill": seed["subskill"],
        "time": seed["time"],
        "provider": "No preference",
        "level": seed["level"],
        "career_goal": f"Build practical skills in {seed['subskill']}",
    }

    return {
        "record_id": record_id,
        "synthetic_profile": profile,
        "synthetic_resume": {
            "education": "Undergraduate or recent graduate",
            "experience": "Course projects and early internship experience",
            "skills": seed["skills"],
            "interests": seed["topic"],
            "goal": profile["career_goal"],
        },
        "ideal_search_query": " ".join(
            [profile["level"], profile["topic"], profile["subskill"], f"for {profile['career_goal']}"]
        ).strip(),
        "synthetic_question": build_fallback_synthetic_question(record_id, course_set),
        "gold_standard": {
            "expected_course_names": names,
            "relevance_judgments": [
                {
                    "course_name": course["course_name"],
                    "relevance_grade": 2,
                    "fit_reason": f"Matches the target area {seed['topic']} and skill focus {seed['subskill']}.",
                    "possible_tradeoff": "May need more learner-specific evidence to rank above similar courses.",
                }
                for course in courses
            ],
            "expected_response": " ".join(
                f"- {name}: A relevant option for {seed['topic']} and {seed['subskill']}."
                for name in names
            ),
            "verification_notes": note,
        },
    }


def validate_gold_record(record: Dict, record_id: str, course_set: List[Dict]) -> Dict:
    record["record_id"] = record_id
    expected_names = [clean_text(c.get("Course Name")) for c in course_set[:3]]
    synthetic_question = record.get("synthetic_question")
    if not isinstance(synthetic_question, dict):
        synthetic_question = {}
    fallback_question = build_fallback_synthetic_question(record_id, course_set)
    record["synthetic_question"] = {
        "question": clean_text(synthetic_question.get("question")) or fallback_question["question"],
        "answer": clean_text(synthetic_question.get("answer")) or fallback_question["answer"],
        "source_course_names": expected_names,
        "constraint": question_constraint_for_record(record_id),
    }

    gold = record.setdefault("gold_standard", {})
    gold["expected_course_names"] = expected_names

    judgments = gold.get("relevance_judgments")
    if not isinstance(judgments, list):
        judgments = []

    by_name = {
        clean_text(item.get("course_name")): item
        for item in judgments
        if isinstance(item, dict)
    }
    normalized_judgments = []
    for name in expected_names:
        item = by_name.get(name, {})
        grade = item.get("relevance_grade", 2)
        try:
            grade = int(grade)
        except Exception:
            grade = 2
        normalized_judgments.append(
            {
                "course_name": name,
                "relevance_grade": max(0, min(3, grade)),
                "fit_reason": clean_text(item.get("fit_reason")) or "Relevant to the synthetic profile.",
                "possible_tradeoff": clean_text(item.get("possible_tradeoff")) or "No major tradeoff noted.",
            }
        )
    gold["relevance_judgments"] = normalized_judgments
    return record


def generate_gold_with_retries(
    model,
    prompt: str,
    record_id: str,
    course_set: List[Dict],
    retries: int,
    request_timeout: int,
) -> Dict:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            attempt_prompt = prompt
            if attempt > 1:
                attempt_prompt = (
                    f"{prompt}\n\n"
                    "Your previous response was not parseable JSON. Return compact JSON only. "
                    "Escape every quote inside string values. Do not include markdown, comments, or trailing commas."
                )
            response = model.generate_content(
                attempt_prompt,
                request_options={"timeout": request_timeout},
            )
            text = getattr(response, "text", "") or ""
            gold = extract_json_object(text)
            return validate_gold_record(gold, record_id, course_set)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.5 * attempt)

    raise RuntimeError(f"Gemini did not return usable JSON after {retries} attempts: {last_error}")


class GeminiRestModel:
    def __init__(self, model_name: str, api_key: str, temperature: float):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature

    def generate_content(self, prompt: str, request_options: Optional[Dict] = None):
        timeout = (request_options or {}).get("timeout", 45)
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": 0.9,
                "topK": 32,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
            },
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini HTTP {exc.code}: {error_body}") from exc

        try:
            parts = body["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Gemini response did not include text: {body}") from exc

        return SimpleNamespace(text=text)


def make_model(model_name: str, temperature: float):

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Add it to .env or set it in the environment.")

    return GeminiRestModel(model_name=model_name, api_key=api_key, temperature=temperature)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl_row(file_obj, row: Dict) -> None:
    file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
    file_obj.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic gold-standard data for future RAG verification."
    )
    parser.add_argument("--count", type=int, default=120, help="Number of gold records to create.")
    parser.add_argument(
        "--allow-small-count",
        action="store_true",
        help="Allow fewer than 100 records for Gemini smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable course sets.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", DEFAULT_MODEL_NAME))
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--gemini-retries", type=int, default=3, help="Retries for malformed Gemini JSON.")
    parser.add_argument("--request-timeout", type=int, default=45, help="Gemini request timeout in seconds.")
    parser.add_argument("--delay-seconds", type=float, default=1.0, help="Delay between Gemini calls.")
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Create fallback records without calling Gemini. Useful for checking the pipeline.",
    )
    args = parser.parse_args()

    if args.count < 100 and not args.allow_small_count:
        raise ValueError("Use --count 100 or higher so the dataset has 100+ course-combination sets.")

    courses = load_course_metadata(args.artifact_dir)
    course_sets = sample_course_sets(courses, count=args.count, seed=args.seed)
    model = None if args.no_gemini else make_model(args.model, args.temperature)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = args.output.with_name(f"{args.output.name}.partial")
    if partial_output.exists():
        partial_output.unlink()

    with open(partial_output, "w", encoding="utf-8") as output_file:
        for idx, course_set in enumerate(course_sets, start=1):
            record_id = f"gold_{idx:04d}"
            public_courses = [course_to_public_record(course) for course in course_set]
            prompt = build_gemini_prompt(record_id, course_set)
            source = "gemini"

            if model is None:
                gold = fallback_gold_record(record_id, course_set, "Generated without Gemini.")
                source = "fallback"
            else:
                try:
                    gold = generate_gold_with_retries(
                        model,
                        prompt,
                        record_id,
                        course_set,
                        retries=max(1, args.gemini_retries),
                        request_timeout=max(1, args.request_timeout),
                    )
                except Exception as exc:
                    raise RuntimeError(f"Failed to generate {record_id}: {exc}") from exc

                if args.delay_seconds > 0:
                    time.sleep(args.delay_seconds)

            generator = {
                "source": source,
                "model": None if model is None else args.model,
            }

            row = {
                "record_id": record_id,
                "generator": generator,
                "courses": public_courses[:3],
                **gold,
            }
            write_jsonl_row(output_file, row)

            print(f"[{idx}/{len(course_sets)}] wrote {record_id} ({source}, {len(course_set)} courses)", flush=True)

    partial_output.replace(args.output)
    print(f"\nSaved {len(course_sets)} gold-standard records to {args.output}")


if __name__ == "__main__":
    main()

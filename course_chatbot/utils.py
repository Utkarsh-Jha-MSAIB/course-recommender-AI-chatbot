import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_session_id() -> str:
    return f"session_{uuid.uuid4().hex[:12]}"


def is_greeting(text: str) -> bool:
    if not text:
        return False
    text = text.lower().strip()
    greetings = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening"
    ]
    return any(g in text for g in greetings)


def extract_simple_preferences(text: str) -> Dict:
    text = text.lower()

    prefs = {
        "difficulty": None,
        "max_hours": None,
        "min_rating": None,
    }

    if any(x in text for x in ["beginner", "intro", "basic", "foundation"]):
        prefs["difficulty"] = "BEGINNER"
    elif "intermediate" in text:
        prefs["difficulty"] = "INTERMEDIATE"
    elif any(x in text for x in ["advanced", "rigorous", "expert"]):
        prefs["difficulty"] = "ADVANCED"

    import re
    hours_match = re.search(r"under\s+(\d+)\s*hours?", text)
    if hours_match:
        prefs["max_hours"] = float(hours_match.group(1))

    rating_match = re.search(r"rating\s+(?:above|over|>=?)\s*(\d+(\.\d+)?)", text)
    if rating_match:
        prefs["min_rating"] = float(rating_match.group(1))

    return prefs


def format_course_block(course: Dict, idx: int) -> str:
    skills = course.get("skills", [])
    skills_text = ", ".join(skills[:8]) if skills else "N/A"

    return (
        f"Course {idx}\n"
        f"Name: {course.get('course_name', 'N/A')}\n"
        f"Partner: {course.get('partner', 'N/A')}\n"
        f"Difficulty: {course.get('difficulty', 'N/A')}\n"
        f"Domain: {course.get('domain', 'N/A')}\n"
        f"Sub-domain: {course.get('sub_domain', 'N/A')}\n"
        f"Hours: {course.get('hours', 'N/A')}\n"
        f"Rating: {course.get('rating', 'N/A')}\n"
        f"Enrollment Count: {course.get('enrollment_count', 'N/A')}\n"
        f"Popularity Percentile: {course.get('popularity_percentile', 'N/A')}\n"
        f"Skills: {skills_text}\n"
        f"URL: {course.get('url', '')}\n"
    )


def save_chat_log(log_file: Path, session_id: str, user_message: str, assistant_message: str) -> None:
    ensure_dir(log_file.parent)
    file_exists = log_file.exists()

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "session_id", "user_message", "assistant_message"]
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
            "user_message": user_message,
            "assistant_message": assistant_message,
        })
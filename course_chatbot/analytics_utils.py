import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "analytics_data"

SESSIONS_FILE = DATA_DIR / "sessions.csv"
MESSAGES_FILE = DATA_DIR / "chat_messages.csv"
RECOMMENDATION_EVENTS_FILE = DATA_DIR / "recommendation_events.csv"
RECOMMENDED_COURSES_FILE = DATA_DIR / "recommended_courses.csv"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_row(file_path: Path, fieldnames: List[str], row: Dict) -> None:
    ensure_dir(file_path.parent)
    file_exists = file_path.exists()

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def generate_session_id() -> str:
    return f"session_{uuid.uuid4().hex[:12]}"


def generate_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:12]}"


def generate_recommendation_event_id() -> str:
    return f"rec_{uuid.uuid4().hex[:12]}"


def normalize_background_cluster(text: str) -> str:
    if not text:
        return "Other"

    t = text.lower().strip()

    rules = {
        "Finance": [
            "finance", "investment", "banking", "risk", "asset", "portfolio",
            "equity", "financial", "accounting", "fintech"
        ],
        "Law": [
            "law", "legal", "attorney", "lawyer", "juris", "compliance",
            "contract", "litigation", "paralegal"
        ],
        "Artificial Intelligence": [
            "artificial intelligence", "ai", "machine learning", "ml",
            "deep learning", "nlp", "computer vision", "llm", "data science"
        ],
        "Marketing": [
            "marketing", "brand", "advertising", "seo", "growth",
            "consumer insights", "market research", "digital marketing"
        ],
        "Business": [
            "business", "strategy", "operations", "consulting", "management",
            "product", "analytics"
        ],
        "Software / Engineering": [
            "software", "developer", "engineer", "programming", "backend",
            "frontend", "full stack", "computer science"
        ],
        "Healthcare": [
            "healthcare", "medicine", "clinical", "doctor", "nurse",
            "public health", "hospital", "biomedical"
        ],
        "Education": [
            "education", "teacher", "teaching", "learning", "instruction",
            "curriculum", "professor"
        ],
    }

    for cluster, keywords in rules.items():
        if any(k in t for k in keywords):
            return cluster

    return "Other"


def save_session_start(session_id: str, user_name: Optional[str] = None) -> None:
    append_row(
        SESSIONS_FILE,
        [
            "session_id",
            "started_at",
            "ended_at",
            "user_name",
            "background_raw",
            "background_cluster",
            "topic",
            "subskill",
            "time_pref",
            "provider_pref",
            "level_pref",
            "questionnaire_completed",
            "recommendation_generated",
        ],
        {
            "session_id": session_id,
            "started_at": now_iso(),
            "ended_at": "",
            "user_name": user_name or "",
            "background_raw": "",
            "background_cluster": "",
            "topic": "",
            "subskill": "",
            "time_pref": "",
            "provider_pref": "",
            "level_pref": "",
            "questionnaire_completed": "false",
            "recommendation_generated": "false",
        },
    )


def save_chat_message(session_id: str, role: str, step: str, message_text: str) -> None:
    append_row(
        MESSAGES_FILE,
        ["message_id", "session_id", "timestamp", "role", "step", "message_text"],
        {
            "message_id": generate_message_id(),
            "session_id": session_id,
            "timestamp": now_iso(),
            "role": role,
            "step": step,
            "message_text": message_text,
        },
    )


def update_session_profile(
    session_id: str,
    profile: Dict[str, str],
    questionnaire_completed: bool = False,
    recommendation_generated: bool = False,
) -> None:
    if not SESSIONS_FILE.exists():
        return

    df = pd.read_csv(SESSIONS_FILE, dtype=str).fillna("")

    matches = df.index[df["session_id"] == session_id]
    if len(matches) == 0:
        return

    i = matches[0]
    background_raw = profile.get("background", "")

    df.at[i, "background_raw"] = background_raw
    df.at[i, "background_cluster"] = normalize_background_cluster(background_raw)
    df.at[i, "topic"] = profile.get("topic", "")
    df.at[i, "subskill"] = profile.get("subskill", "")
    df.at[i, "time_pref"] = profile.get("time", "")
    df.at[i, "provider_pref"] = profile.get("provider", "")
    df.at[i, "level_pref"] = profile.get("level", "")
    df.at[i, "questionnaire_completed"] = str(questionnaire_completed).lower()
    df.at[i, "recommendation_generated"] = str(recommendation_generated).lower()

    if recommendation_generated:
        df.at[i, "ended_at"] = now_iso()

    df.to_csv(SESSIONS_FILE, index=False)


def save_recommendation_event(session_id: str, query_used: str, courses: List[Dict]) -> str:
    recommendation_event_id = generate_recommendation_event_id()

    append_row(
        RECOMMENDATION_EVENTS_FILE,
        ["recommendation_event_id", "session_id", "timestamp", "query_used", "num_courses_shown"],
        {
            "recommendation_event_id": recommendation_event_id,
            "session_id": session_id,
            "timestamp": now_iso(),
            "query_used": query_used,
            "num_courses_shown": len(courses),
        },
    )

    return recommendation_event_id


def save_recommended_courses(session_id: str, courses: List[Dict]) -> None:
    recommendation_event_id = generate_recommendation_event_id()

    for idx, course in enumerate(courses, start=1):
        append_row(
            RECOMMENDED_COURSES_FILE,
            [
                "recommendation_event_id",
                "session_id",
                "rank",
                "course_name",
                "partner",
                "domain",
                "sub_domain",
                "difficulty",
                "hours",
                "rating",
                "semantic_score",
                "final_score",
                "url",
                "image_url",
            ],
            {
                "recommendation_event_id": recommendation_event_id,
                "session_id": session_id,
                "rank": idx,
                "course_name": course.get("course_name", ""),
                "partner": course.get("partner", ""),
                "domain": course.get("domain", ""),
                "sub_domain": course.get("sub_domain", ""),
                "difficulty": course.get("difficulty", ""),
                "hours": course.get("hours", ""),
                "rating": course.get("rating", ""),
                "semantic_score": course.get("semantic_score", ""),
                "final_score": course.get("final_score", ""),
                "url": course.get("url", ""),
                "image_url": course.get("image_url", ""),
            },
        )


def to_label_count_list(series: pd.Series, limit: int = 10) -> List[Dict]:
    if series.empty:
        return []

    counts = (
        series.fillna("Unknown")
        .astype(str)
        .replace("", "Unknown")
        .value_counts()
        .head(limit)
    )

    return [
        {"label": str(label), "count": int(count)}
        for label, count in counts.items()
    ]


def build_top_courses_overall(rec_courses: pd.DataFrame, limit: int = 3) -> List[Dict]:
    if rec_courses.empty or "course_name" not in rec_courses.columns:
        return []

    working = rec_courses.copy().fillna("")

    grouped = (
        working.groupby("course_name", dropna=False)
        .agg({
            "partner": "first",
            "domain": "first",
            "sub_domain": "first",
            "difficulty": "first",
            "hours": "first",
            "rating": "first",
            "url": "first",
            "image_url": "first" if "image_url" in working.columns else lambda x: "",
            "semantic_score": "first",
            "final_score": "first",
            "course_name": "count",
        })
        .rename(columns={"course_name": "count"})
        .reset_index()
        .sort_values(by=["count", "course_name"], ascending=[False, True])
        .head(limit)
    )

    top_courses = []
    for _, row in grouped.iterrows():
        partner = str(row.get("partner", "") or "").strip()
        domain = str(row.get("domain", "") or "").strip()

        if domain and partner:
            domain_partner = domain
        elif domain:
            domain_partner = domain
        elif partner:
            domain_partner = partner
        else:
            domain_partner = "General"

        top_courses.append({
            "course_name": str(row.get("course_name", "") or "").strip(),
            "partner": partner,
            "domain": domain,
            "sub_domain": str(row.get("sub_domain", "") or "").strip(),
            "difficulty": str(row.get("difficulty", "") or "").strip(),
            "hours": str(row.get("hours", "") or "").strip(),
            "rating": str(row.get("rating", "") or "").strip(),
            "semantic_score": str(row.get("semantic_score", "") or "").strip(),
            "final_score": str(row.get("final_score", "") or "").strip(),
            "url": str(row.get("url", "") or "").strip(),
            "image_url": str(row.get("image_url", "") or "").strip(),
            "count": int(row.get("count", 0) or 0),
            "domain_partner": domain_partner,
        })

    return top_courses


def load_analytics_summary() -> Dict:
    sessions = pd.read_csv(SESSIONS_FILE, dtype=str).fillna("") if SESSIONS_FILE.exists() else pd.DataFrame()
    messages = pd.read_csv(MESSAGES_FILE, dtype=str).fillna("") if MESSAGES_FILE.exists() else pd.DataFrame()
    rec_events = pd.read_csv(RECOMMENDATION_EVENTS_FILE, dtype=str).fillna("") if RECOMMENDATION_EVENTS_FILE.exists() else pd.DataFrame()
    rec_courses = pd.read_csv(
        RECOMMENDED_COURSES_FILE,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    ).fillna("") if RECOMMENDED_COURSES_FILE.exists() else pd.DataFrame()

    total_sessions = int(len(sessions)) if not sessions.empty else 0
    total_messages = int(len(messages)) if not messages.empty else 0
    total_recommendation_events = int(len(rec_events)) if not rec_events.empty else 0
    total_recommended_courses = int(len(rec_courses)) if not rec_courses.empty else 0

    top_backgrounds = []
    top_interests = []
    top_providers = []
    top_difficulties = []
    recent_sessions = []
    top_courses_overall = []

    if not sessions.empty:
        top_backgrounds = to_label_count_list(sessions["background_cluster"], limit=10)
        top_interests = to_label_count_list(sessions["topic"], limit=10)
        top_providers = to_label_count_list(sessions["provider_pref"], limit=10)
        top_difficulties = to_label_count_list(sessions["level_pref"], limit=10)

        recent = sessions.copy()

        for col in ["started_at", "background_cluster", "topic", "provider_pref", "level_pref"]:
            if col not in recent.columns:
                recent[col] = ""

        recent = recent.sort_values(by="started_at", ascending=False).head(10)

        recent_sessions = [
            {
                "session_id": row.get("session_id", ""),
                "started_at": row.get("started_at", ""),
                "background": row.get("background_cluster", ""),
                "interest": row.get("topic", ""),
                "provider": row.get("provider_pref", ""),
                "difficulty": row.get("level_pref", ""),
            }
            for _, row in recent.iterrows()
        ]

    if not rec_courses.empty:
        top_courses_overall = build_top_courses_overall(rec_courses, limit=3)

    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "total_recommendation_events": total_recommendation_events,
        "total_recommended_courses": total_recommended_courses,
        "top_backgrounds": top_backgrounds,
        "top_interests": top_interests,
        "top_providers": top_providers,
        "top_difficulties": top_difficulties,
        "recent_sessions": recent_sessions,
        "top_courses_overall": top_courses_overall,
    }
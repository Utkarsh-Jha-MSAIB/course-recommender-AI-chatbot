import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader_analyzer = SentimentIntensityAnalyzer()
except Exception:
    _vader_analyzer = None


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "analytics_data"

SESSIONS_FILE = DATA_DIR / "sessions.csv"
MESSAGES_FILE = DATA_DIR / "chat_messages.csv"
RECOMMENDATION_EVENTS_FILE = DATA_DIR / "recommendation_events.csv"
RECOMMENDED_COURSES_FILE = DATA_DIR / "recommended_courses.csv"
RECOMMENDATION_RATINGS_FILE = DATA_DIR / "recommendation_ratings.csv"
RECOMMENDATION_FEEDBACK_FILE = DATA_DIR / "recommendation_feedback.csv"
CONVERSATION_NLP_FILE = DATA_DIR / "conversation_nlp.csv"


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


def sentiment_label_from_score(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def compute_vader_sentiment(text: str) -> float:
    if not text or not str(text).strip():
        return 0.0

    if _vader_analyzer is None:
        return 0.0

    return float(_vader_analyzer.polarity_scores(str(text))["compound"])


def compute_sentiment_alignment(user_score: float, assistant_score: float) -> float:
    alignment = 1.0 - (abs(float(user_score) - float(assistant_score)) / 2.0)
    return max(0.0, min(1.0, alignment))


def save_conversation_nlp_pair(
    session_id: str,
    user_message: str,
    assistant_message: str,
    step: str = "",
) -> None:
    user_sentiment = compute_vader_sentiment(user_message)
    assistant_sentiment = compute_vader_sentiment(assistant_message)
    sentiment_alignment = compute_sentiment_alignment(user_sentiment, assistant_sentiment)

    append_row(
        CONVERSATION_NLP_FILE,
        [
            "timestamp",
            "session_id",
            "step",
            "user_message",
            "assistant_message",
            "user_sentiment",
            "user_sentiment_label",
            "assistant_sentiment",
            "assistant_sentiment_label",
            "sentiment_alignment",
        ],
        {
            "timestamp": now_iso(),
            "session_id": session_id,
            "step": step,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "user_sentiment": round(user_sentiment, 4),
            "user_sentiment_label": sentiment_label_from_score(user_sentiment),
            "assistant_sentiment": round(assistant_sentiment, 4),
            "assistant_sentiment_label": sentiment_label_from_score(assistant_sentiment),
            "sentiment_alignment": round(sentiment_alignment, 4),
        },
    )


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


def save_recommendation_rating(
    session_id: str,
    recommendation_tracking_id: str,
    course_name: str,
    course_rank,
    course_url: str,
    rating_value: int,
    rating_label: str,
) -> None:
    append_row(
        RECOMMENDATION_RATINGS_FILE,
        [
            "timestamp",
            "session_id",
            "recommendation_tracking_id",
            "course_name",
            "course_rank",
            "course_url",
            "rating_value",
            "rating_label",
        ],
        {
            "timestamp": now_iso(),
            "session_id": session_id,
            "recommendation_tracking_id": recommendation_tracking_id,
            "course_name": course_name,
            "course_rank": course_rank,
            "course_url": course_url,
            "rating_value": rating_value,
            "rating_label": rating_label,
        },
    )


def save_recommendation_feedback(
    session_id: str,
    recommendation_tracking_id: str,
    course_name: str,
    course_rank,
    course_url: str,
    feedback_text: str,
) -> None:
    append_row(
        RECOMMENDATION_FEEDBACK_FILE,
        [
            "timestamp",
            "session_id",
            "recommendation_tracking_id",
            "course_name",
            "course_rank",
            "course_url",
            "feedback_text",
        ],
        {
            "timestamp": now_iso(),
            "session_id": session_id,
            "recommendation_tracking_id": recommendation_tracking_id,
            "course_name": course_name,
            "course_rank": course_rank,
            "course_url": course_url,
            "feedback_text": feedback_text.strip(),
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


def build_rating_summary(ratings_df: pd.DataFrame, total_recommended_courses: int) -> Dict:
    if ratings_df.empty:
        return {
            "total_up_ratings": 0,
            "total_down_ratings": 0,
            "total_ignored_ratings": total_recommended_courses,
            "net_rating_score": 0,
            "feedback_rating_breakdown": [],
        }

    working = ratings_df.copy().fillna("")

    for col in ["session_id", "recommendation_tracking_id", "rating_value", "timestamp"]:
        if col not in working.columns:
            working[col] = ""

    working["rating_value_num"] = pd.to_numeric(
        working["rating_value"], errors="coerce"
    ).fillna(0).astype(int)

    working = working.sort_values(by="timestamp")
    latest = working.drop_duplicates(
        subset=["session_id", "recommendation_tracking_id"],
        keep="last"
    )

    total_up = int((latest["rating_value_num"] == 1).sum())
    total_down = int((latest["rating_value_num"] == -1).sum())
    explicit_ignored = int((latest["rating_value_num"] == 0).sum())

    rated_unique = int(len(latest))
    derived_unrated = max(total_recommended_courses - rated_unique, 0)
    total_ignored = explicit_ignored + derived_unrated

    return {
        "total_up_ratings": total_up,
        "total_down_ratings": total_down,
        "total_ignored_ratings": total_ignored,
        "net_rating_score": total_up - total_down,
        "feedback_rating_breakdown": [
            {"label": "Up", "count": total_up},
            {"label": "Down", "count": total_down},
            {"label": "Ignored", "count": total_ignored},
        ],
    }


def build_recent_feedback(feedback_df: pd.DataFrame, limit: int = 10) -> List[Dict]:
    if feedback_df.empty:
        return []

    working = feedback_df.copy().fillna("")

    for col in ["timestamp", "course_name", "feedback_text", "session_id"]:
        if col not in working.columns:
            working[col] = ""

    working = working.sort_values(by="timestamp", ascending=False).head(limit)

    return [
        {
            "timestamp": row.get("timestamp", ""),
            "course_name": row.get("course_name", ""),
            "feedback_text": row.get("feedback_text", ""),
            "session_id": row.get("session_id", ""),
        }
        for _, row in working.iterrows()
    ]


def build_nlp_summary(nlp_df: pd.DataFrame) -> Dict:
    if nlp_df.empty:
        return {
            "avg_user_sentiment": 0.0,
            "avg_assistant_sentiment": 0.0,
            "avg_sentiment_alignment": 0.0,
            "low_alignment_sessions": 0,
            "user_sentiment_distribution": [
                {"label": "Positive", "count": 0},
                {"label": "Neutral", "count": 0},
                {"label": "Negative", "count": 0},
            ],
            "assistant_sentiment_distribution": [
                {"label": "Positive", "count": 0},
                {"label": "Neutral", "count": 0},
                {"label": "Negative", "count": 0},
            ],
            "session_sentiment_rows": [],
            "session_turn_details": {},
        }

    working = nlp_df.copy().fillna("")

    for col in [
        "timestamp", "session_id", "step", "user_message", "assistant_message",
        "user_sentiment", "assistant_sentiment", "sentiment_alignment",
        "user_sentiment_label", "assistant_sentiment_label"
    ]:
        if col not in working.columns:
            working[col] = ""

    working["user_sentiment_num"] = pd.to_numeric(working["user_sentiment"], errors="coerce").fillna(0.0)
    working["assistant_sentiment_num"] = pd.to_numeric(working["assistant_sentiment"], errors="coerce").fillna(0.0)
    working["sentiment_alignment_num"] = pd.to_numeric(working["sentiment_alignment"], errors="coerce").fillna(0.0)

    avg_user_sentiment = round(float(working["user_sentiment_num"].mean()), 3)
    avg_assistant_sentiment = round(float(working["assistant_sentiment_num"].mean()), 3)
    avg_sentiment_alignment = round(float(working["sentiment_alignment_num"].mean()), 3)

    session_grouped = (
        working.groupby("session_id", dropna=False)
        .agg({
            "user_sentiment_num": "mean",
            "assistant_sentiment_num": "mean",
            "sentiment_alignment_num": "mean",
            "session_id": "count",
            "timestamp": "max",
        })
        .rename(columns={"session_id": "turn_count"})
        .reset_index()
        .sort_values(by="timestamp", ascending=False)
    )

    low_alignment_sessions = int((session_grouped["sentiment_alignment_num"] < 0.55).sum())

    session_sentiment_rows = [
        {
            "session_id": row["session_id"],
            "avg_user_sentiment": round(float(row["user_sentiment_num"]), 3),
            "avg_assistant_sentiment": round(float(row["assistant_sentiment_num"]), 3),
            "avg_sentiment_alignment": round(float(row["sentiment_alignment_num"]), 3),
            "turn_count": int(row["turn_count"]),
            "last_seen": row["timestamp"],
        }
        for _, row in session_grouped.iterrows()
    ]

    user_sentiment_distribution = [
        {"label": "Positive", "count": int((working["user_sentiment_num"] >= 0.05).sum())},
        {"label": "Neutral", "count": int(((working["user_sentiment_num"] > -0.05) & (working["user_sentiment_num"] < 0.05)).sum())},
        {"label": "Negative", "count": int((working["user_sentiment_num"] <= -0.05).sum())},
    ]

    assistant_sentiment_distribution = [
        {"label": "Positive", "count": int((working["assistant_sentiment_num"] >= 0.05).sum())},
        {"label": "Neutral", "count": int(((working["assistant_sentiment_num"] > -0.05) & (working["assistant_sentiment_num"] < 0.05)).sum())},
        {"label": "Negative", "count": int((working["assistant_sentiment_num"] <= -0.05).sum())},
    ]

    working = working.sort_values(by=["session_id", "timestamp"]).copy()
    working["turn_number"] = working.groupby("session_id").cumcount() + 1

    session_turn_details: Dict[str, List[Dict]] = {}
    for session_id, group in working.groupby("session_id", dropna=False):
        session_turn_details[str(session_id)] = [
            {
                "turn_number": int(row["turn_number"]),
                "timestamp": row.get("timestamp", ""),
                "step": row.get("step", ""),
                "user_message": row.get("user_message", ""),
                "assistant_message": row.get("assistant_message", ""),
                "user_sentiment": round(float(row.get("user_sentiment_num", 0.0)), 3),
                "assistant_sentiment": round(float(row.get("assistant_sentiment_num", 0.0)), 3),
                "sentiment_alignment": round(float(row.get("sentiment_alignment_num", 0.0)), 3),
                "user_sentiment_label": row.get("user_sentiment_label", ""),
                "assistant_sentiment_label": row.get("assistant_sentiment_label", ""),
            }
            for _, row in group.iterrows()
        ]

    return {
        "avg_user_sentiment": avg_user_sentiment,
        "avg_assistant_sentiment": avg_assistant_sentiment,
        "avg_sentiment_alignment": avg_sentiment_alignment,
        "low_alignment_sessions": low_alignment_sessions,
        "user_sentiment_distribution": user_sentiment_distribution,
        "assistant_sentiment_distribution": assistant_sentiment_distribution,
        "session_sentiment_rows": session_sentiment_rows,
        "session_turn_details": session_turn_details,
    }


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
    rec_ratings = pd.read_csv(
        RECOMMENDATION_RATINGS_FILE,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    ).fillna("") if RECOMMENDATION_RATINGS_FILE.exists() else pd.DataFrame()
    rec_feedback = pd.read_csv(
        RECOMMENDATION_FEEDBACK_FILE,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    ).fillna("") if RECOMMENDATION_FEEDBACK_FILE.exists() else pd.DataFrame()
    conversation_nlp = pd.read_csv(
        CONVERSATION_NLP_FILE,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    ).fillna("") if CONVERSATION_NLP_FILE.exists() else pd.DataFrame()

    total_sessions = int(len(sessions)) if not sessions.empty else 0
    total_messages = int(len(messages)) if not messages.empty else 0
    total_recommendation_events = int(len(rec_events)) if not rec_events.empty else 0
    total_recommended_courses = int(len(rec_courses)) if not rec_courses.empty else 0
    total_feedback_submissions = int(len(rec_feedback)) if not rec_feedback.empty else 0

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

    rating_summary = build_rating_summary(rec_ratings, total_recommended_courses)
    recent_feedback = build_recent_feedback(rec_feedback, limit=10)
    nlp_summary = build_nlp_summary(conversation_nlp)

    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "total_recommendation_events": total_recommendation_events,
        "total_recommended_courses": total_recommended_courses,
        "total_feedback_submissions": total_feedback_submissions,
        "total_up_ratings": rating_summary["total_up_ratings"],
        "total_down_ratings": rating_summary["total_down_ratings"],
        "total_ignored_ratings": rating_summary["total_ignored_ratings"],
        "net_rating_score": rating_summary["net_rating_score"],
        "feedback_rating_breakdown": rating_summary["feedback_rating_breakdown"],
        "top_backgrounds": top_backgrounds,
        "top_interests": top_interests,
        "top_providers": top_providers,
        "top_difficulties": top_difficulties,
        "recent_sessions": recent_sessions,
        "top_courses_overall": top_courses_overall,
        "recent_feedback": recent_feedback,
        "avg_user_sentiment": nlp_summary["avg_user_sentiment"],
        "avg_assistant_sentiment": nlp_summary["avg_assistant_sentiment"],
        "avg_sentiment_alignment": nlp_summary["avg_sentiment_alignment"],
        "low_alignment_sessions": nlp_summary["low_alignment_sessions"],
        "user_sentiment_distribution": nlp_summary["user_sentiment_distribution"],
        "assistant_sentiment_distribution": nlp_summary["assistant_sentiment_distribution"],
        "session_sentiment_rows": nlp_summary["session_sentiment_rows"],
        "session_turn_details": nlp_summary["session_turn_details"],
    }
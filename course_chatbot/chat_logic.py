import json
import os
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv
import google.generativeai as genai

from recommender import CourseRecommender
from utils import is_greeting, format_course_block
from analytics_utils import (
    update_session_profile,
    save_recommendation_event,
    save_recommended_courses,
)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
gemini_model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config={
        "temperature": 0.55,
        "top_p": 0.9,
        "top_k": 32,
        "max_output_tokens": 800,
    }
)

recommender = CourseRecommender()

MAX_RECOMMENDATION_COUNT = 5
MIN_RECOMMENDATION_COUNT = 3
MIN_TRANSITION_SECONDS = 1.1
MIN_RECOMMENDATION_SECONDS = 1.6

SYSTEM_CONTEXT = """
You are an AI Course Recommendation Assistant for Coursera-style course discovery.

Your job is to help users find the most relevant courses based on:
- learning goals
- background or major
- experience level
- time availability
- interests
- provider preference
- popularity and rating when useful

Rules:
- Be warm, professional, and concise.
- Use only the retrieved course metadata provided.
- Do not invent course details.
- Recommend 3 to 5 courses when enough relevant courses are available.
- Explain why each recommendation fits the user profile.
- Mention tradeoffs clearly when helpful.
- Keep responses readable and practical.
- Sound conversational, not robotic.
- Be slightly more explanatory than a search engine result.
"""

QUESTION_FLOW = [
    ("background", "What is your major or background?\nType your answer."),
    ("topic", "Which area are you most interested in right now?"),
    ("time", "How much time can you realistically spend on this course?"),
    ("provider", "Do you have any preference for the course provider?"),
    ("level", "What level are you looking for?"),
]

QUESTION_OPTIONS = {
    "topic": [
        "AI / Machine Learning",
        "Data Analysis / Visualization",
        "Cloud / Deployment",
        "Business / Analytics",
    ],
    "time": [
        "Up to 2 hours",
        "2 to 5 hours",
        "5 to 10 hours",
        "10 to 20 hours",
        "20+ hours",
    ],
    "provider": [
        "Coursera",
        "Google Cloud",
        "IBM",
        "Microsoft",
        "Packt",
        "No preference",
    ],
    "level": [
        "Beginner",
        "Intermediate",
        "Advanced",
        "No preference",
    ],
}

TOPIC_SUBSKILLS = {
    "AI / Machine Learning": [
        "Generative AI",
        "Prompt Engineering",
        "LLM Application",
        "Natural Language Processing",
        "Deep Learning",
        "Model Evaluation",
        "Retrieval-Augmented Generation",
        "Generative AI Agents",
        "Computer Vision",
        "MLOps (Machine Learning Operations)",
    ],
    "Data Analysis / Visualization": [
        "Data Analysis",
        "Data Visualization",
        "SQL",
        "Dashboard",
        "Microsoft Excel",
        "Data Storytelling",
        "Exploratory Data Analysis",
        "Predictive Analytics",
        "Business Intelligence",
        "Statistical Analysis",
    ],
    "Cloud / Deployment": [
        "Cloud Deployment",
        "MLOps (Machine Learning Operations)",
        "Application Programming Interface (API)",
        "CI/CD",
        "Kubernetes",
        "Amazon Web Services",
        "Microsoft Azure",
        "Google Cloud Platform",
        "Containerization",
        "System Monitoring",
    ],
    "Business / Analytics": [
        "Business Analytics",
        "AI Product Strategy",
        "Customer Insights",
        "Marketing Analytics",
        "Business Strategy",
        "Decision Making",
        "Forecasting",
        "Project Management",
        "Business Intelligence",
        "Strategic Thinking",
    ],
}


def ensure_min_delay(start_time: float, min_seconds: float) -> None:
    elapsed = time.perf_counter() - start_time
    remaining = min_seconds - elapsed
    if remaining > 0:
        time.sleep(remaining)


def attach_options(question_key: str, question_text: str) -> str:
    options = QUESTION_OPTIONS.get(question_key)
    if not options:
        return question_text
    return f"{question_text}\n[[OPTIONS:{'||'.join(options)}]]"


def attach_skill_dropdown(message_text: str, options: List[str]) -> str:
    safe = [x.strip() for x in options if x.strip()]
    return f"{message_text}\n[[SKILL_DROPDOWN:{'||'.join(safe)}]]"


def strip_markers(text: str) -> str:
    text = re.sub(r"\n?\[\[OPTIONS:.*?\]\]", "", text or "", flags=re.DOTALL)
    text = re.sub(r"\n?\[\[SKILL_DROPDOWN:.*?\]\]", "", text or "", flags=re.DOTALL)
    text = re.sub(r"\n?\[\[AWAIT_CUSTOM_SKILL\]\]", "", text or "", flags=re.DOTALL)
    text = re.sub(r"\n?\[\[STATE:.*?\]\]", "", text or "", flags=re.DOTALL)
    return text.strip()


def build_course_context(courses: List[Dict]) -> str:
    if not courses:
        return "No relevant courses retrieved."
    return "\n\n".join([format_course_block(course, idx + 1) for idx, course in enumerate(courses)])


def extract_name(text: str) -> Optional[str]:
    if not text:
        return None

    patterns = [
        r"\bmy name is ([A-Za-z][A-Za-z'\-]+)\b",
        r"\bi am ([A-Za-z][A-Za-z'\-]+)\b",
        r"\bi'm ([A-Za-z][A-Za-z'\-]+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            return name[:1].upper() + name[1:]
    return None


def extract_name_from_history(chat_history: List[Dict]) -> Optional[str]:
    for msg in reversed(chat_history):
        if msg.get("role") != "user":
            continue
        found = extract_name(msg.get("content", ""))
        if found:
            return found
    return None


def questionnaire_started(chat_history: List[Dict]) -> bool:
    for msg in chat_history:
        if msg.get("role") == "assistant":
            if "[[STATE:step=" in msg.get("content", ""):
                return True
    return False


def normalize_topic_answer(topic_answer: str) -> str:
    if not topic_answer:
        return ""

    text = topic_answer.strip().lower()

    mapping = {
        "1": "AI / Machine Learning",
        "ai": "AI / Machine Learning",
        "machine learning": "AI / Machine Learning",
        "ai / machine learning": "AI / Machine Learning",
        "2": "Data Analysis / Visualization",
        "data analysis / visualization": "Data Analysis / Visualization",
        "data analysis": "Data Analysis / Visualization",
        "visualization": "Data Analysis / Visualization",
        "3": "Cloud / Deployment",
        "cloud": "Cloud / Deployment",
        "deployment": "Cloud / Deployment",
        "cloud / deployment": "Cloud / Deployment",
        "4": "Business / Analytics",
        "business": "Business / Analytics",
        "analytics": "Business / Analytics",
        "business / analytics": "Business / Analytics",
    }

    return mapping.get(text, topic_answer.strip())


def normalize_time_answer(time_answer: str) -> str:
    if not time_answer:
        return ""

    text = time_answer.strip().lower()

    mapping = {
        "1": "Up to 2 hours",
        "2": "2 to 5 hours",
        "3": "5 to 10 hours",
        "4": "10 to 20 hours",
        "5": "20+ hours",
    }

    return mapping.get(text, time_answer.strip())


def normalize_provider_answer(provider_answer: str) -> str:
    if not provider_answer:
        return ""

    text = provider_answer.strip().lower()

    mapping = {
        "1": "Coursera",
        "2": "Google Cloud",
        "3": "IBM",
        "4": "Microsoft",
        "5": "Packt",
        "6": "No preference",
        "coursera": "Coursera",
        "google cloud": "Google Cloud",
        "ibm": "IBM",
        "microsoft": "Microsoft",
        "packt": "Packt",
        "no preference": "No preference",
    }

    return mapping.get(text, provider_answer.strip())


def normalize_level_answer(level_answer: str) -> str:
    if not level_answer:
        return ""

    text = level_answer.strip().lower()

    mapping = {
        "1": "Beginner",
        "2": "Intermediate",
        "3": "Advanced",
        "4": "No preference",
        "beginner": "Beginner",
        "intermediate": "Intermediate",
        "advanced": "Advanced",
        "no preference": "No preference",
    }

    return mapping.get(text, level_answer.strip())


def normalize_subskill_answer(subskill_answer: str) -> str:
    if not subskill_answer:
        return ""
    return subskill_answer.strip()


def normalize_time_to_hours(time_answer: str) -> Optional[float]:
    if not time_answer:
        return None

    text = normalize_time_answer(time_answer).lower().strip()

    if "no preference" in text or "any" in text:
        return None
    if "up to 2" in text:
        return 2.0
    if "2 to 5" in text:
        return 5.0
    if "5 to 10" in text:
        return 10.0
    if "10 to 20" in text:
        return 20.0
    if "20+" in text or "20 +" in text:
        return None

    range_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)", text)
    if range_match:
        return float(range_match.group(2))

    plus_match = re.search(r"(\d+(?:\.\d+)?)\s*\+", text)
    if plus_match:
        return None

    single_match = re.search(r"(\d+(?:\.\d+)?)", text)
    if single_match:
        return float(single_match.group(1))

    return None


def normalize_level_to_allowed_difficulties(level_answer: str) -> Optional[Set[str]]:
    if not level_answer:
        return None

    text = normalize_level_answer(level_answer).lower().strip()

    if "no preference" in text or "any" in text:
        return None
    if "beginner" in text:
        return {"BEGINNER", "UNKNOWN"}
    if "intermediate" in text:
        return {"INTERMEDIATE"}
    if "advanced" in text:
        return {"ADVANCED"}

    return None


def normalize_skill_text(skill: str) -> str:
    return re.sub(r"\s+", " ", str(skill or "")).strip()


def split_skills(skills_value) -> List[str]:
    if not skills_value:
        return []

    if isinstance(skills_value, list):
        return [normalize_skill_text(x) for x in skills_value if normalize_skill_text(x)]

    raw = str(skills_value).strip()
    if not raw:
        return []

    return [
        normalize_skill_text(x)
        for x in re.split(r"[|,;•\n]+", raw)
        if normalize_skill_text(x)
    ]


def parse_top_skills(skills_value, limit: int = 3) -> List[str]:
    cleaned = []
    seen = set()

    for item in split_skills(skills_value):
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(item)
        if len(cleaned) >= limit:
            break

    return cleaned


def topic_filter_match(row: Dict, normalized_topic: str) -> bool:
    domain = str(row.get("Domain_Clean", "") or "").lower()
    sub_domain = str(row.get("Sub-Domain_Clean", "") or "").lower()
    title = str(row.get("Course Name", "") or "").lower()
    partner = str(row.get("University / Industry Partner Name", "") or "").lower()
    skills_text = str(row.get("Unified Skills Text", "") or row.get("Unified Skills List", "") or "").lower()

    blob = " | ".join([domain, sub_domain, title, partner, skills_text])

    if normalized_topic == "Cloud / Deployment":
        keys = ["cloud", "deployment", "devops", "kubernetes", "docker", "mlops", "terraform", "ci/cd", "api", "aws", "azure", "gcp"]
    elif normalized_topic == "AI / Machine Learning":
        keys = ["machine learning", "artificial intelligence", "deep learning", "nlp", "llm", "neural", "vision", "prompt", "generative ai", "rag", "agent"]
    elif normalized_topic == "Data Analysis / Visualization":
        keys = ["data analysis", "visualization", "tableau", "power bi", "sql", "analytics", "dashboard", "excel", "statistics"]
    elif normalized_topic == "Business / Analytics":
        keys = ["business", "analytics", "strategy", "decision", "operations", "product", "finance", "marketing", "forecasting"]
    else:
        return False

    return any(k in blob for k in keys)


def get_catalog_subskills(topic_answer: str, limit: int = 12) -> List[str]:
    normalized_topic = normalize_topic_answer(topic_answer)
    if not normalized_topic:
        return []

    generic = {
        "artificial intelligence",
        "machine learning",
        "cloud computing",
        "cloud",
        "deployment",
        "data analysis",
        "visualization",
        "business analytics",
        "analytics",
        "business",
        "beginner",
        "intermediate",
        "advanced",
        "communication",
        "creativity",
        "problem solving",
        "leadership",
    }

    counter = Counter()

    for row in getattr(recommender, "metadata", []):
        if not topic_filter_match(row, normalized_topic):
            continue

        skills_value = row.get("Unified Skills List") or row.get("Unified Skills Text") or ""
        for skill in split_skills(skills_value):
            s = skill.strip()
            if not s:
                continue
            low = s.lower()
            if len(low) < 3:
                continue
            if low in generic:
                continue
            counter[s] += 1

    return [skill for skill, _ in counter.most_common(limit)]


def get_subskill_options(topic_answer: str, max_options: int = 8) -> List[str]:
    normalized_topic = normalize_topic_answer(topic_answer)
    curated = TOPIC_SUBSKILLS.get(normalized_topic, [])
    catalog_candidates = get_catalog_subskills(normalized_topic, limit=12)

    merged = []
    seen = set()

    for item in curated + catalog_candidates:
        norm = item.strip().lower()
        if not item.strip() or norm in seen:
            continue
        seen.add(norm)
        merged.append(item.strip())
        if len(merged) >= max_options:
            break

    merged.append("Type your own")
    return merged


def get_state(chat_history: List[Dict]) -> Dict[str, str]:
    state = {
        "background": "",
        "topic": "",
        "subskill": "",
        "time": "",
        "provider": "",
        "level": "",
        "awaiting_custom_skill": "false",
        "step": "resume",
        "resume_status": "pending",
        "resume_background": "",
        "resume_skills": "",
        "resume_level": "",
        "resume_interests": "",
        "resume_goal": "",
        "resume_education": "",
        "resume_experience": "",
        "career_dirs": "",
    }

    for msg in chat_history:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        state_matches = re.findall(r"\[\[STATE:([a-z_]+)=(.*?)\]\]", content)
        for key, value in state_matches:
            state[key] = value

    return state


def build_state_markers(state: Dict[str, str]) -> str:
    allowed_keys = [
        "background", "topic", "subskill", "time", "provider", "level",
        "awaiting_custom_skill", "step",
        "resume_status", "resume_background", "resume_skills",
        "resume_level", "resume_interests", "resume_goal",
        "resume_education", "resume_experience", "career_dirs",
    ]
    return "".join([f"\n[[STATE:{k}={state.get(k, '')}]]" for k in allowed_keys])


def build_profile_summary(profile: Dict[str, str]) -> str:
    lines = [
        f"- Background: {profile.get('background', 'Not provided')}",
        f"- Topic of interest: {normalize_topic_answer(profile.get('topic', 'Not provided'))}",
        f"- Specific skill focus: {normalize_subskill_answer(profile.get('subskill', 'Not provided'))}",
        f"- Time available: {normalize_time_answer(profile.get('time', 'Not provided'))}",
        f"- Provider preference: {normalize_provider_answer(profile.get('provider', 'Not provided'))}",
        f"- Preferred level: {normalize_level_answer(profile.get('level', 'Not provided'))}",
    ]

    if profile.get("resume_status") == "provided":
        resume_parts = []
        if profile.get("resume_skills"):
            resume_parts.append(f"  - Skills from resume: {profile['resume_skills']}")
        if profile.get("resume_interests"):
            resume_parts.append(f"  - Interests from resume: {profile['resume_interests']}")
        if profile.get("resume_goal"):
            resume_parts.append(f"  - Career goal from resume: {profile['resume_goal']}")
        if resume_parts:
            lines.append("- Resume context:\n" + "\n".join(resume_parts))

    return "\n".join(lines)


def generate_intro_message(user_name: Optional[str]) -> str:
    base_state = {
        "background": "", "topic": "", "subskill": "", "time": "", "provider": "", "level": "",
        "awaiting_custom_skill": "false", "step": "resume",
        "resume_status": "pending", "resume_background": "", "resume_skills": "",
        "resume_level": "", "resume_interests": "", "resume_goal": "",
        "resume_education": "", "resume_experience": "", "career_dirs": "",
    }

    greeting = f"Hi {user_name}!" if user_name else "Hi!"

    return (
        f"{greeting} I’m your AI Course Recommendation Assistant.\n\n"
        "To get you the best recommendations, you can **paste your resume below** — "
        "or type **Skip** to answer a few quick questions instead."
        f"{build_state_markers(base_state)}"
    )


resume_extraction_model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config={
        "temperature": 0.1,
        "max_output_tokens": 1024,
    }
)

resume_tips_model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config={
        "temperature": 0.5,
        "max_output_tokens": 2048,
    }
)


def extract_resume_info(resume_text: str) -> Dict[str, str]:
    prompt = f"""Extract information from the resume below. Return ONLY a compact JSON object on a SINGLE LINE.
Use an empty string "" for any field you cannot find.
Keep ALL values short (under 10 words each).
For background: just the main academic major or field (e.g. "Computer Science", "Artificial Intelligence").
For skills: top 5 technical skills as a comma-separated STRING (not a list).
For interests: 2-3 interest areas as a comma-separated STRING.
For level, return exactly one of: "Beginner", "Intermediate", or "Advanced".
For goal: one short career goal phrase (e.g. "AI Product Manager", "Software Engineer").
For education: highest degree only (e.g. "Master's", "Bachelor's", "PhD").
For experience: work experience summary (e.g. "Student", "Internship experience", "1-2 years").

Keys: background, skills, level, interests, goal, education, experience

Resume:
{resume_text[:2500]}

Return only compact JSON on one line. Example:
{{"background": "Computer Science", "skills": "Python, TensorFlow", "level": "Intermediate", "interests": "machine learning", "goal": "ML engineer", "education": "Master's", "experience": "Internship experience"}}"""

    try:
        response = resume_extraction_model.generate_content(prompt)
        text = (response.text or "").strip()
        # Strip markdown code blocks
        text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
        # Find the JSON object even if multi-line
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return {}
        data = json.loads(match.group(0))
        # Normalize list fields to comma-separated strings
        for key in ("skills", "interests"):
            if isinstance(data.get(key), list):
                data[key] = ", ".join(str(x) for x in data[key])
        # Normalize level to one of our three valid values
        raw_level = str(data.get("level", "")).lower()
        if any(w in raw_level for w in ("advanced", "senior", "expert", "lead")):
            data["level"] = "Advanced"
        elif any(w in raw_level for w in ("beginner", "student", "entry", "junior", "fresh")):
            data["level"] = "Beginner"
        else:
            data["level"] = "Intermediate"
        return data
    except Exception as e:
        err_str = str(e).lower()
        if "quota" in err_str or "resource_exhausted" in err_str or "429" in err_str:
            print(f"[RESUME EXTRACTION RATE LIMIT] {e}")
            return {"_rate_limited": True}
        print(f"[RESUME EXTRACTION ERROR] {e}")
        return {}


def suggest_career_directions(state: Dict[str, str]) -> List[Dict[str, str]]:
    """Return 3 career directions based on resume data, each mapping to a topic + subskill."""
    goal = state.get("resume_goal", "").lower()
    background = state.get("resume_background", "").lower()
    skills = state.get("resume_skills", "").lower()
    interests = state.get("resume_interests", "").lower()

    # All possible paths: (label, topic, subskill, keywords_that_boost_score)
    all_paths = [
        ("AI Product Manager",   "Business / Analytics",          "AI Product Strategy",                  ["product", "ai", "manager", "management", "strategy"]),
        ("ML Engineer",          "AI / Machine Learning",         "LLM Application",                      ["machine learning", "ml", "model", "nlp", "deep learning", "tensorflow", "pytorch"]),
        ("Data Scientist",       "AI / Machine Learning",         "Deep Learning",                        ["data science", "statistics", "python", "analysis", "machine learning"]),
        ("Data Analyst",         "Data Analysis / Visualization", "Data Analysis",                        ["data", "sql", "excel", "analytics", "visualization", "dashboard"]),
        ("Cloud Engineer",       "Cloud / Deployment",            "Amazon Web Services",                  ["cloud", "aws", "azure", "gcp", "devops", "deployment", "kubernetes"]),
        ("Business Analyst",     "Business / Analytics",          "Business Analytics",                   ["business", "analytics", "operations", "stakeholder", "requirements"]),
        ("Product Manager",      "Business / Analytics",          "Project Management",                   ["product", "project", "roadmap", "agile", "scrum", "management"]),
        ("AI Researcher",        "AI / Machine Learning",         "Natural Language Processing",          ["research", "nlp", "language", "ai", "neural", "academic"]),
        ("BI Developer",         "Data Analysis / Visualization", "Business Intelligence",                ["bi", "tableau", "power bi", "dashboard", "reporting", "sql"]),
        ("MLOps Engineer",       "Cloud / Deployment",            "MLOps (Machine Learning Operations)",  ["mlops", "deployment", "pipeline", "model", "cloud", "docker"]),
    ]

    blob = " ".join([goal, background, skills, interests])

    scored = []
    for label, topic, subskill, keywords in all_paths:
        score = 0
        for kw in keywords:
            if kw in blob:
                score += 1
        scored.append((score, label, topic, subskill))

    scored.sort(key=lambda x: -x[0])

    # Always ensure 3 distinct results; pad with defaults if needed
    seen_topics = set()
    result = []
    for score, label, topic, subskill in scored:
        if topic not in seen_topics:
            result.append({"label": label, "topic": topic, "subskill": subskill})
            seen_topics.add(topic)
        if len(result) == 3:
            break

    # If still < 3, fill remaining with any remaining paths
    for score, label, topic, subskill in scored:
        if len(result) >= 3:
            break
        if not any(d["label"] == label for d in result):
            result.append({"label": label, "topic": topic, "subskill": subskill})

    return result[:3]


def generate_resume_tips(state: Dict[str, str], career_label: str) -> str:
    """Generate 5 numbered resume improvement tips for the chosen career direction."""
    prompt = f"""You are a career coach. Based on this person's resume profile and their chosen career direction, give exactly 5 specific and actionable tips to strengthen their resume for that career.

Profile:
- Background: {state.get("resume_background", "")}
- Education: {state.get("resume_education", "")}
- Experience: {state.get("resume_experience", "")}
- Skills: {state.get("resume_skills", "")}
- Career goal: {state.get("resume_goal", "")}
- Chosen career direction: {career_label}

Rules:
- Each tip must be specific to their profile and the chosen career — no generic advice
- Keep each tip to 1-2 sentences, clear and direct
- Number them 1. through 5.
- No intro sentence, just the 5 numbered tips"""

    try:
        response = resume_tips_model.generate_content(prompt)
        return (response.text or "").strip()
    except Exception as e:
        err_str = str(e).lower()
        if "quota" in err_str or "resource_exhausted" in err_str or "429" in err_str:
            print(f"[RESUME TIPS RATE LIMIT] {e}")
        else:
            print(f"[RESUME TIPS ERROR] {e}")
        return ""


def encode_career_dirs(directions: List[Dict[str, str]]) -> str:
    return "|||".join(f"{d['label']};{d['topic']};{d['subskill']}" for d in directions)


def parse_career_dirs(encoded: str) -> List[Dict[str, str]]:
    if not encoded:
        return []
    result = []
    for part in encoded.split("|||"):
        parts = part.split(";")
        if len(parts) == 3:
            result.append({"label": parts[0], "topic": parts[1], "subskill": parts[2]})
    return result


def format_hours(hours_value) -> str:
    if hours_value is None or hours_value == "":
        return "Flexible"

    try:
        hours = float(hours_value)
        if hours.is_integer():
            return f"{int(hours)} hours"
        return f"{hours:.1f} hours"
    except Exception:
        return str(hours_value)


def format_rating(rating_value) -> str:
    if rating_value is None or rating_value == "":
        return "N/A"

    try:
        return f"{float(rating_value):.1f}"
    except Exception:
        return str(rating_value)


def format_score(score_value) -> str:
    if score_value is None or score_value == "":
        return "N/A"

    try:
        return f"{float(score_value):.2f}"
    except Exception:
        return str(score_value)


def response_looks_complete(text: str, expected_bullets: int) -> bool:
    if not text or len(text.strip()) < 60:
        return False

    cleaned = text.strip()

    bad_endings = (",", ":", ";", "-", "•", "and", "with", "for", "we", "the", "a")
    if cleaned.lower().endswith(bad_endings):
        return False

    bullet_count = cleaned.count("- ")
    if bullet_count < expected_bullets:
        return False

    if cleaned[-1] not in ".!?":
        return False

    return True


def build_combined_query(profile: Dict[str, str], extra_request: Optional[str] = None) -> str:
    background = profile.get("background", "").strip()
    topic = normalize_topic_answer(profile.get("topic", "")).strip()
    subskill = normalize_subskill_answer(profile.get("subskill", "")).strip()
    provider = normalize_provider_answer(profile.get("provider", "")).strip()
    level = normalize_level_answer(profile.get("level", "")).strip()
    resume_skills = profile.get("resume_skills", "").strip()
    resume_interests = profile.get("resume_interests", "").strip()
    resume_goal = profile.get("resume_goal", "").strip()

    pieces = []

    if level and "no preference" not in level.lower():
        pieces.append(level)
    if topic:
        pieces.append(topic)
    if subskill:
        pieces.append(subskill)
    elif resume_skills:
        pieces.append(f"with skills in {resume_skills}")
    if provider and "no preference" not in provider.lower():
        pieces.append(f"from {provider}")
    if background:
        pieces.append(f"for someone with background in {background}")
    if resume_interests and not topic:
        pieces.append(f"interested in {resume_interests}")
    if resume_goal:
        pieces.append(f"targeting a role as {resume_goal}")
    if extra_request:
        pieces.append(extra_request.strip())

    combined = " ".join([p for p in pieces if p]).strip()
    return combined if combined else (extra_request or topic or "recommended courses")


def build_course_cards(courses: List[Dict]) -> List[Dict]:
    cards: List[Dict] = []

    for course in courses:
        top_skills = parse_top_skills(course.get("skills", ""), limit=3)

        domain = (course.get("domain") or "").strip()
        partner = (course.get("partner") or "").strip()

        if domain and partner:
            domain_partner = domain
        elif domain:
            domain_partner = domain
        elif partner:
            domain_partner = partner
        else:
            domain_partner = "General"

        hours_value = format_hours(course.get("hours"))
        rating_value = format_rating(course.get("rating"))

        if hours_value != "Flexible" and rating_value != "N/A":
            hours_rating = hours_value
        elif hours_value != "Flexible":
            hours_rating = hours_value
        elif rating_value != "N/A":
            hours_rating = "Rated"
        else:
            hours_rating = "N/A"

        cards.append({
            "domain_partner": domain_partner,
            "hours_rating": hours_rating,
            "skills": top_skills,
            "semantic_score": format_score(course.get("semantic_score")),
            "final_score": format_score(course.get("final_score")),
            "url": (course.get("url") or "").strip(),
            "image_url": (course.get("image_url") or "").strip(),
        })

    return cards


def infer_tradeoff(course: Dict, profile: Dict[str, str]) -> str:
    hours = course.get("hours")
    rating = course.get("rating")
    difficulty = str(course.get("difficulty", "")).lower()
    preferred_level = normalize_level_answer(profile.get("level", "")).lower()
    max_hours = normalize_time_to_hours(profile.get("time", ""))

    try:
        hours_val = float(hours) if hours not in (None, "") else None
    except Exception:
        hours_val = None

    try:
        rating_val = float(rating) if rating not in (None, "") else None
    except Exception:
        rating_val = None

    subskill = normalize_subskill_answer(profile.get("subskill", "")).lower()
    course_blob = " ".join([
        str(course.get("course_name", "") or ""),
        str(course.get("domain", "") or ""),
        str(course.get("sub_domain", "") or ""),
        str(course.get("skills", "") or "")
    ]).lower()

    if hours_val is not None and max_hours is not None and hours_val > max_hours:
        return "It goes beyond your preferred time limit, but that may come with stronger depth."

    if hours_val is not None and max_hours is not None and hours_val <= max_hours and hours_val <= 3:
        return "It fits your schedule well, though it may work better as a quick practical introduction than a deep dive."

    if preferred_level == "beginner" and difficulty == "intermediate":
        return "It could stretch you a bit beyond beginner level, which is good for growth but may require more effort."

    if preferred_level == "beginner" and difficulty == "advanced":
        return "It may be too steep if you want something immediately approachable."

    if preferred_level == "advanced" and difficulty == "beginner":
        return "It may feel more foundational than advanced, but it could still be useful as a fast targeted refresher."

    if rating_val is not None and rating_val < 4.5:
        return "It looks relevant, though learner feedback appears slightly weaker than some alternatives."

    if subskill and subskill not in course_blob:
        return f"It supports your broader goal, though it may be less directly focused on {profile.get('subskill', '')} than the ideal match."

    return "It looks balanced overall, though the main tradeoff is breadth versus specialization."


def build_fallback_explanation(selected_courses: List[Dict], profile: Dict[str, str]) -> str:
    topic = normalize_topic_answer(profile.get("topic", "your selected area"))
    subskill = normalize_subskill_answer(profile.get("subskill", ""))
    time_pref = normalize_time_answer(profile.get("time", ""))
    level = normalize_level_answer(profile.get("level", ""))

    intro = f"I focused on courses that align with {topic}"
    if subskill:
        intro += f", especially {subskill}"
    if time_pref:
        intro += f", while staying realistic for {time_pref.lower()}"
    if level and "no preference" not in level.lower():
        intro += f", and leaning toward a {level.lower()} fit"
    intro += ".\n"

    bullets = []

    for idx, course in enumerate(selected_courses, start=1):
        name = course.get("course_name") or f"Option {idx}"
        skills = parse_top_skills(course.get("skills", ""), limit=2)
        difficulty = str(course.get("difficulty", "")).lower()
        hours = format_hours(course.get("hours"))

        fit_parts = []
        if skills:
            fit_parts.append(f"it touches on {', '.join(skills)}")
        if hours:
            fit_parts.append(f"it’s about {hours}")
        if difficulty and difficulty != "unknown":
            fit_parts.append(f"and it looks {difficulty}")

        fit_text = ", ".join(fit_parts) if fit_parts else "it looks relevant to your goal"
        tradeoff = infer_tradeoff(course, profile)

        bullets.append(
            f"- **{name}**: A solid option because {fit_text}. Tradeoff: {tradeoff}"
        )

    return intro + "\n".join(bullets)


def choose_dynamic_course_count(retrieved_courses: List[Dict]) -> int:
    if not retrieved_courses:
        return 0

    total = min(len(retrieved_courses), MAX_RECOMMENDATION_COUNT)

    if total <= MIN_RECOMMENDATION_COUNT:
        return total

    top_score = float(retrieved_courses[0].get("final_score", 0.0) or 0.0)
    chosen = total

    for idx, course in enumerate(retrieved_courses[:total]):
        score = float(course.get("final_score", 0.0) or 0.0)
        if idx >= MIN_RECOMMENDATION_COUNT and top_score > 0 and score < 0.78 * top_score:
            chosen = idx
            break

    return max(MIN_RECOMMENDATION_COUNT, min(chosen, total))


def recommend_from_profile(
    profile: Dict[str, str],
    extra_request: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict:
    start_time = time.perf_counter()

    combined_query = build_combined_query(profile, extra_request=extra_request)
    max_hours = normalize_time_to_hours(profile.get("time", ""))
    allowed_difficulties = normalize_level_to_allowed_difficulties(profile.get("level", ""))

    retrieved_courses = recommender.search_courses(
        query=combined_query,
        top_k_retrieval=20,
        top_k_final=MAX_RECOMMENDATION_COUNT,
        allowed_difficulties=allowed_difficulties,
        max_hours=max_hours,
        min_rating=None,
        require_rated=False,
        require_enrollment=False,
    )

    if not retrieved_courses:
        ensure_min_delay(start_time, MIN_RECOMMENDATION_SECONDS)
        return {
            "text": "I couldn’t find strong matches for that profile right now. Try adjusting the topic, skill focus, level, or time preference.",
            "courses": []
        }

    chosen_count = choose_dynamic_course_count(retrieved_courses)
    selected_courses = retrieved_courses[:chosen_count]

    if session_id:
        try:
            update_session_profile(
                session_id=session_id,
                profile=profile,
                questionnaire_completed=True,
                recommendation_generated=True,
            )
            save_recommendation_event(
                session_id=session_id,
                query_used=combined_query,
                courses=selected_courses,
            )
            save_recommended_courses(
                session_id=session_id,
                courses=selected_courses,
            )
        except Exception as e:
            print(f"[ANALYTICS ERROR] {e}")

    course_context = build_course_context(selected_courses)
    profile_summary = build_profile_summary(profile)

    prompt = f"""
{SYSTEM_CONTEXT}

User profile:
{profile_summary}

Combined search query:
{combined_query}

Retrieved courses:
{course_context}

Write a recommendation response for the user.

Requirements:
- Start with 2 short sentences summarizing the overall match.
- Then provide exactly {len(selected_courses)} bullets.
- Each bullet must include:
  1. why the course fits the user's selected topic/subskill,
  2. whether the duration fits the user's time preference,
  3. whether the level is a good fit,
  4. one concrete tradeoff.
- Tradeoffs should be varied. Do NOT repeat the same tradeoff wording across bullets.
- Use natural, advisor-like language.
- Keep the total response under 220 words.
- Do not invent missing details.
- Do not include URLs.
"""

    explanation = build_fallback_explanation(selected_courses, profile)

    try:
        response = gemini_model.generate_content(prompt)
        print("\n[DEBUG] Gemini raw response:", response)

        candidate = ""
        if hasattr(response, "text") and response.text:
            candidate = response.text.strip()

        finish_reason = None
        try:
            finish_reason = response.candidates[0].finish_reason
            print("[DEBUG] Gemini finish reason:", finish_reason)
        except Exception:
            pass

        if candidate and response_looks_complete(candidate, len(selected_courses)):
            explanation = candidate
        else:
            print("[DEBUG] Gemini response looked incomplete, using fallback.")

    except Exception as e:
        print(f"[GEMINI ERROR] {e}")

    cards = build_course_cards(selected_courses)

    ensure_min_delay(start_time, MIN_RECOMMENDATION_SECONDS)

    return {
        "text": explanation,
        "courses": cards
    }


def get_chatbot_response(message: str, chat_history: List[Dict], session_id: Optional[str] = None):
    start_time = time.perf_counter()
    stripped = message.strip()

    prior_history = chat_history[:-1] if chat_history and chat_history[-1].get("role") == "user" else chat_history
    state = get_state(prior_history)

    current_name = extract_name(stripped)
    known_name = current_name or extract_name_from_history(prior_history)

    started = questionnaire_started(prior_history)

    if not started:
        response = {
            "text": generate_intro_message(known_name),
            "courses": []
        }
        ensure_min_delay(start_time, MIN_TRANSITION_SECONDS)
        return response

    # --- Resume step ---
    if state.get("step") == "resume" or state.get("resume_status") == "pending":
        if stripped.lower() in ("skip", "skip resume", "no resume", "no", "no thanks"):
            state["resume_status"] = "skipped"
            state["step"] = "background"
            response_text = (
                "No problem — let's go through a few quick questions.\n\n"
                "What is your major or background?\nType your answer."
                + build_state_markers(state)
            )
            ensure_min_delay(start_time, MIN_TRANSITION_SECONDS)
            return {"text": response_text, "courses": []}

        extracted = extract_resume_info(stripped)

        if extracted.get("_rate_limited"):
            state["resume_status"] = "skipped"
            state["step"] = "background"
            response_text = (
                "The AI is temporarily unavailable due to API rate limits. "
                "Let's continue with a few quick questions instead.\n\n"
                "What is your major or background?\nType your answer."
                + build_state_markers(state)
            )
            ensure_min_delay(start_time, MIN_TRANSITION_SECONDS)
            return {"text": response_text, "courses": []}

        state["resume_status"] = "provided"
        state["resume_background"] = extracted.get("background", "")
        state["resume_skills"] = extracted.get("skills", "")
        state["resume_level"] = extracted.get("level", "")
        state["resume_interests"] = extracted.get("interests", "")
        state["resume_goal"] = extracted.get("goal", "")
        state["resume_education"] = extracted.get("education", "")
        state["resume_experience"] = extracted.get("experience", "")

        extracted_bg = state.get("resume_background", "").strip()

        if extracted_bg:
            state["step"] = "background_confirm"
            summary_lines = [f"**Background:** {extracted_bg}"]
            if state.get("resume_education"):
                summary_lines.append(f"**Education:** {state['resume_education']}")
            if state.get("resume_experience"):
                summary_lines.append(f"**Experience:** {state['resume_experience']}")
            if state.get("resume_skills"):
                summary_lines.append(f"**Skills:** {state['resume_skills']}")
            if state.get("resume_interests"):
                summary_lines.append(f"**Interests:** {state['resume_interests']}")
            if state.get("resume_goal"):
                summary_lines.append(f"**Career goal:** {state['resume_goal']}")
            summary = "\n".join(summary_lines)
            response_text = (
                "Thanks for sharing your resume! Here's what I found:\n\n"
                f"{summary}\n\n"
                "Does this look right? Type **Yes** to confirm, or type a correction."
                + build_state_markers(state)
            )
        else:
            state["step"] = "background"
            response_text = (
                "Thanks! I couldn't pull a clear background from that — let me ask you directly.\n\n"
                "What is your major or background?\nType your answer."
                + build_state_markers(state)
            )
        ensure_min_delay(start_time, MIN_TRANSITION_SECONDS)
        return {"text": response_text, "courses": []}

    # --- Background confirmation step ---
    if state.get("step") == "background_confirm":
        if stripped.lower() in ("yes", "yeah", "correct", "looks good", "that's right", "yep", "yup", "right", "ok", "okay"):
            state["background"] = state.get("resume_background", "")
        else:
            state["background"] = stripped

        directions = suggest_career_directions(state)
        if directions:
            state["career_dirs"] = encode_career_dirs(directions)
            state["step"] = "career_direction"
            labels = [d["label"] for d in directions]
            options_marker = "[[OPTIONS:" + "||".join(labels) + "]]"
            response_text = (
                "Based on your profile, here are 3 career paths that match you well:\n\n"
                + "\n".join(f"**{d['label']}**" for d in directions) + "\n\n"
                "Which direction are you most interested in?\n"
                + options_marker
                + build_state_markers(state)
            )
        else:
            state["step"] = "topic"
            response_text = (
                "Got it.\n\n"
                + attach_options("topic", "Which area are you most interested in right now?")
                + build_state_markers(state)
            )
        ensure_min_delay(start_time, MIN_TRANSITION_SECONDS)
        return {"text": response_text, "courses": []}

    # --- Career direction step ---
    if state.get("step") == "career_direction":
        directions = parse_career_dirs(state.get("career_dirs", ""))
        selected = None
        for d in directions:
            if stripped.lower() == d["label"].lower() or d["label"].lower() in stripped.lower():
                selected = d
                break
        if selected:
            state["topic"] = selected["topic"]
            state["subskill"] = selected["subskill"]
            state["step"] = "time"
            tips = generate_resume_tips(state, selected["label"])
            if tips:
                response_text = (
                    f"**{selected['label']}** — great direction!\n\n"
                    f"[[TIPS]]**Resume Feedback**\n\n{tips}[[/TIPS]]\n\n"
                    "Now, how much time can you spend on a course per week?\n"
                    + "[[OPTIONS:" + "||".join(QUESTION_OPTIONS["time"]) + "]]"
                    + build_state_markers(state)
                )
            else:
                response_text = (
                    f"Great choice! I'll find courses to help you grow as a **{selected['label']}**.\n\n"
                    + attach_options("time", "How much time can you realistically spend on this course per week?")
                    + build_state_markers(state)
                )
        else:
            state["step"] = "topic"
            response_text = (
                "Let me ask a bit more specifically.\n\n"
                + attach_options("topic", "Which area are you most interested in right now?")
                + build_state_markers(state)
            )
        ensure_min_delay(start_time, MIN_TRANSITION_SECONDS)
        return {"text": response_text, "courses": []}

    if state.get("awaiting_custom_skill") == "true":
        state["subskill"] = stripped
        state["awaiting_custom_skill"] = "false"
        state["step"] = "time"
        response_text = (
            f"Perfect — I’ll use {stripped} as your focus area.\n\n"
            + attach_options("time", "How much time can you realistically spend on this course?")
            + build_state_markers(state)
        )
        ensure_min_delay(start_time, MIN_TRANSITION_SECONDS)
        return {"text": response_text, "courses": []}

    if not state.get("background"):
        state["background"] = stripped
        state["step"] = "topic"
        response_text = (
            "Thanks — that gives me a good starting point.\n\n"
            + attach_options("topic", "Which area are you most interested in right now?")
            + build_state_markers(state)
        )
        ensure_min_delay(start_time, MIN_TRANSITION_SECONDS)
        return {"text": response_text, "courses": []}

    if not state.get("topic"):
        state["topic"] = stripped
        state["step"] = "subskill_select"
        options = get_subskill_options(stripped)

        natural_reply = (
            f"Great choice — {normalize_topic_answer(stripped)} can branch into a few useful directions.\n\n"
            "I’ve narrowed this into a curated set of focus areas for you. Pick one from the dropdown below, or choose Type your own if you want something more specific."
        )

        response_text = attach_skill_dropdown(natural_reply, options) + build_state_markers(state)
        ensure_min_delay(start_time, MIN_TRANSITION_SECONDS)
        return {"text": response_text, "courses": []}

    if not state.get("subskill"):
        if stripped.lower() == "type your own":
            state["awaiting_custom_skill"] = "true"
            state["step"] = "subskill_custom"
            response_text = (
                "Absolutely — type the specific skill or focus area you want, and I’ll use that before narrowing the recommendations."
                "\n[[AWAIT_CUSTOM_SKILL]]"
                + build_state_markers(state)
            )
            ensure_min_delay(start_time, MIN_TRANSITION_SECONDS)
            return {"text": response_text, "courses": []}

        state["subskill"] = stripped
        state["step"] = "time"
        response_text = (
            f"Perfect — I’ll use {stripped} as your main focus within this area.\n\n"
            + attach_options("time", "How much time can you realistically spend on this course?")
            + build_state_markers(state)
        )
        ensure_min_delay(start_time, MIN_TRANSITION_SECONDS)
        return {"text": response_text, "courses": []}

    if not state.get("time"):
        state["time"] = stripped
        state["step"] = "provider"
        response_text = (
            "Got it — that helps me balance speed versus depth.\n\n"
            + attach_options("provider", "Do you have any preference for the course provider?")
            + build_state_markers(state)
        )
        ensure_min_delay(start_time, MIN_TRANSITION_SECONDS)
        return {"text": response_text, "courses": []}

    if not state.get("provider"):
        state["provider"] = stripped
        state["step"] = "level"
        response_text = (
            "Nice — I’ll factor that into the ranking.\n\n"
            + attach_options("level", "What level are you looking for?")
            + build_state_markers(state)
        )
        ensure_min_delay(start_time, MIN_TRANSITION_SECONDS)
        return {"text": response_text, "courses": []}

    if not state.get("level"):
        state["level"] = stripped
        profile = {
            "background": state.get("background", ""),
            "topic": state.get("topic", ""),
            "subskill": state.get("subskill", ""),
            "time": state.get("time", ""),
            "provider": state.get("provider", ""),
            "level": state.get("level", ""),
            "resume_status": state.get("resume_status", ""),
            "resume_skills": state.get("resume_skills", ""),
            "resume_interests": state.get("resume_interests", ""),
            "resume_goal": state.get("resume_goal", ""),
            "resume_level": state.get("resume_level", ""),
        }
        return recommend_from_profile(profile, session_id=session_id)

    profile = {
        "background": state.get("background", ""),
        "topic": state.get("topic", ""),
        "subskill": state.get("subskill", ""),
        "time": state.get("time", ""),
        "provider": state.get("provider", ""),
        "level": state.get("level", ""),
        "resume_status": state.get("resume_status", ""),
        "resume_skills": state.get("resume_skills", ""),
        "resume_interests": state.get("resume_interests", ""),
        "resume_goal": state.get("resume_goal", ""),
        "resume_level": state.get("resume_level", ""),
    }
    return recommend_from_profile(profile, extra_request=message, session_id=session_id)

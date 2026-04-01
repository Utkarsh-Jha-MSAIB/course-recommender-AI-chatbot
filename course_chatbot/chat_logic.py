import os
import re
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv
import google.generativeai as genai

from recommender import CourseRecommender
from utils import is_greeting, format_course_block


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
gemini_model = genai.GenerativeModel(MODEL_NAME)

recommender = CourseRecommender()

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
- Mention tradeoffs briefly when helpful.
- Keep responses readable and practical.
"""

QUESTION_FLOW = [
    ("background", "Question 1 of 5: What is your major or background?\nType your answer."),
    ("topic", "Question 2 of 5: Which area are you most interested in right now?"),
    ("time", "Question 3 of 5: How much time can you realistically spend on this course?"),
    ("provider", "Question 4 of 5: Do you have any preference for the course provider?"),
    ("level", "Question 5 of 5: What level are you looking for?"),
]

QUESTION_OPTIONS = {
    "topic": [
        "AI / Machine Learning",
        "Data Analysis / Visualization",
        "Cloud / Deployment",
        "Business / Analytics",
    ],
    "time": [
        "Under 5 hours",
        "5 to 15 hours",
        "15 to 30 hours",
        "30+ hours",
    ],
    "provider": [
        "Coursera",
        "Google Cloud",
        "IBM",
        "Microsoft",
        "Packt",
    ],
    "level": [
        "Beginner",
        "Intermediate",
        "Advanced",
        "No preference",
    ],
}


def attach_options(question_key: str, question_text: str) -> str:
    options = QUESTION_OPTIONS.get(question_key)
    if not options:
        return question_text
    return f"{question_text}\n[[OPTIONS:{'||'.join(options)}]]"


def build_course_context(courses: List[Dict]) -> str:
    if not courses:
        return "No relevant courses retrieved."
    return "\n\n".join([format_course_block(course, idx + 1) for idx, course in enumerate(courses)])


def extract_question_number(text: str) -> Optional[int]:
    match = re.search(r"Question\s+(\d+)\s+of\s+5", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


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


def strip_option_markers(text: str) -> str:
    return re.sub(r"\n?\[\[OPTIONS:.*?\]\]", "", text, flags=re.DOTALL).strip()


def extract_profile_answers(chat_history: List[Dict]) -> Dict[str, str]:
    answers: Dict[str, str] = {}
    pending_key: Optional[str] = None

    for msg in chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            visible_content = strip_option_markers(content)
            q_num = extract_question_number(visible_content)
            if q_num is not None and 1 <= q_num <= len(QUESTION_FLOW):
                pending_key = QUESTION_FLOW[q_num - 1][0]

        elif role == "user" and pending_key:
            text = content.strip()
            if text:
                answers[pending_key] = text
                pending_key = None

    return answers


def questionnaire_started(chat_history: List[Dict]) -> bool:
    for msg in chat_history:
        if msg.get("role") == "assistant":
            visible_content = strip_option_markers(msg.get("content", ""))
            if extract_question_number(visible_content) is not None:
                return True
    return False


def get_pending_question_key(chat_history: List[Dict]) -> Optional[str]:
    for msg in reversed(chat_history):
        if msg.get("role") != "assistant":
            continue
        visible_content = strip_option_markers(msg.get("content", ""))
        q_num = extract_question_number(visible_content)
        if q_num is not None and 1 <= q_num <= len(QUESTION_FLOW):
            return QUESTION_FLOW[q_num - 1][0]
    return None


def next_question_text(answer_count: int) -> Optional[str]:
    if answer_count < len(QUESTION_FLOW):
        key, text = QUESTION_FLOW[answer_count]
        return attach_options(key, text)
    return None


def normalize_topic_answer(topic_answer: str) -> str:
    if not topic_answer:
        return ""

    text = topic_answer.strip().lower()

    mapping = {
        "1": "AI / Machine Learning",
        "ai": "AI / Machine Learning",
        "machine learning": "AI / Machine Learning",
        "ai / machine learning": "AI / Machine Learning",
        "data analysis / visualization": "Data Analysis / Visualization",
        "data analysis": "Data Analysis / Visualization",
        "visualization": "Data Analysis / Visualization",
        "2": "Data Analysis / Visualization",
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
        "1": "Under 5 hours",
        "2": "5 to 15 hours",
        "3": "15 to 30 hours",
        "4": "30+ hours",
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
        "coursera": "Coursera",
        "google cloud": "Google Cloud",
        "ibm": "IBM",
        "microsoft": "Microsoft",
        "packt": "Packt",
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


def normalize_time_to_hours(time_answer: str) -> Optional[float]:
    if not time_answer:
        return None

    text = normalize_time_answer(time_answer).lower().strip()

    if "no preference" in text or "any" in text:
        return None
    if "under 5" in text or "less than 5" in text:
        return 5.0
    if "under 10" in text or "less than 10" in text:
        return 10.0
    if "under 15" in text or "less than 15" in text:
        return 15.0
    if "5 to 15" in text:
        return 15.0
    if "under 20" in text or "less than 20" in text:
        return 20.0
    if "15 to 30" in text:
        return 30.0
    if "under 30" in text or "less than 30" in text:
        return 30.0
    if "30+" in text:
        return 30.0
    if "short" in text:
        return 10.0

    range_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)", text)
    if range_match:
        return float(range_match.group(2))

    plus_match = re.search(r"(\d+(?:\.\d+)?)\s*\+", text)
    if plus_match:
        return float(plus_match.group(1))

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


def build_combined_query(profile: Dict[str, str], extra_request: Optional[str] = None) -> str:
    background = profile.get("background", "").strip()
    topic = normalize_topic_answer(profile.get("topic", "")).strip()
    provider = normalize_provider_answer(profile.get("provider", "")).strip()
    level = normalize_level_answer(profile.get("level", "")).strip()

    pieces = []

    if level and "no preference" not in level.lower():
        pieces.append(level)
    if topic:
        pieces.append(topic)
    if provider:
        pieces.append(f"from {provider}")
    if background:
        pieces.append(f"for someone with background in {background}")
    if extra_request:
        pieces.append(extra_request.strip())

    combined = " ".join([p for p in pieces if p]).strip()
    return combined if combined else (extra_request or topic or "recommended courses")


def build_profile_summary(profile: Dict[str, str]) -> str:
    lines = [
        f"- Background: {profile.get('background', 'Not provided')}",
        f"- Topic of interest: {normalize_topic_answer(profile.get('topic', 'Not provided'))}",
        f"- Time available: {normalize_time_answer(profile.get('time', 'Not provided'))}",
        f"- Provider preference: {normalize_provider_answer(profile.get('provider', 'Not provided'))}",
        f"- Preferred level: {normalize_level_answer(profile.get('level', 'Not provided'))}",
    ]
    return "\n".join(lines)


def generate_intro_message(user_name: Optional[str]) -> str:
    first_question = attach_options(*QUESTION_FLOW[0])

    if user_name:
        return (
            f"Hi {user_name}! I’m your AI Course Recommendation Assistant.\n\n"
            "I’ll ask you 5 quick questions and then recommend courses that fit your goals.\n\n"
            f"{first_question}"
        )

    return (
        "Hi! I’m your AI Course Recommendation Assistant.\n\n"
        "I’ll ask you 5 quick questions and then recommend courses that fit your goals.\n\n"
        f"{first_question}"
    )


def generate_question_transition(next_question: str) -> str:
    return f"Got it.\n\n{next_question}"


def recommend_from_profile(profile: Dict[str, str], extra_request: Optional[str] = None) -> str:
    combined_query = build_combined_query(profile, extra_request=extra_request)
    max_hours = normalize_time_to_hours(profile.get("time", ""))
    allowed_difficulties = normalize_level_to_allowed_difficulties(profile.get("level", ""))

    retrieved_courses = recommender.search_courses(
        query=combined_query,
        top_k_retrieval=20,
        top_k_final=5,
        allowed_difficulties=allowed_difficulties,
        max_hours=max_hours,
        min_rating=None,
        require_rated=False,
        require_enrollment=False,
    )

    course_context = build_course_context(retrieved_courses)
    profile_summary = build_profile_summary(profile)

    prompt = f"""
{SYSTEM_CONTEXT}

User profile:
{profile_summary}

Combined search query:
{combined_query}

Retrieved courses:
{course_context}

Instructions:
- Recommend the top 3 to 5 courses from the retrieved list.
- For each course, explain why it fits the user's background, topic interest, provider preference, time preference, and level when relevant.
- Mention difficulty, duration, rating, and provider when useful.
- If there are tradeoffs, mention them briefly.
- Do not invent any missing details.
- Keep the answer concise, practical, and conversational.
"""

    response = gemini_model.generate_content(prompt)
    if hasattr(response, "text") and response.text:
        return response.text.strip()

    return "Sorry, I couldn’t generate recommendations right now. Please try again."


def get_chatbot_response(message: str, chat_history: List[Dict]) -> str:
    stripped = message.strip()

    prior_history = chat_history[:-1] if chat_history and chat_history[-1].get("role") == "user" else chat_history
    prior_answers = extract_profile_answers(prior_history)

    current_name = extract_name(stripped)
    known_name = current_name or extract_name_from_history(prior_history)

    started = questionnaire_started(prior_history)
    pending_key = get_pending_question_key(prior_history)

    if not started and (is_greeting(stripped) or current_name is not None):
        return generate_intro_message(known_name)

    if not started:
        updated_answers = {"background": stripped}
        next_q = next_question_text(len(updated_answers))
        return generate_question_transition(next_q or "Please continue.")

    if len(prior_answers) < len(QUESTION_FLOW):
        current_key = pending_key or QUESTION_FLOW[len(prior_answers)][0]
        updated_answers = dict(prior_answers)
        updated_answers[current_key] = stripped

        if len(updated_answers) < len(QUESTION_FLOW):
            next_q = next_question_text(len(updated_answers))
            return generate_question_transition(next_q or "Please continue.")

        return recommend_from_profile(updated_answers)

    return recommend_from_profile(prior_answers, extra_request=message)
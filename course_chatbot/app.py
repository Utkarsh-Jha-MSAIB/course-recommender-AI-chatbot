import io
from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from chat_logic import get_chatbot_response
from utils import ensure_dir, save_chat_log
from analytics_utils import (
    ensure_dir as ensure_analytics_dir,
    generate_session_id,
    save_session_start,
    save_chat_message,
    load_analytics_summary,
)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "chat_history.csv"
ANALYTICS_DIR = PROJECT_ROOT / "analytics_data"

ensure_dir(LOG_DIR)
ensure_analytics_dir(ANALYTICS_DIR)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)
CORS(app)

chat_histories: Dict[str, List[Dict]] = {}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analytics")
def analytics():
    summary = load_analytics_summary()
    return render_template("analytics.html", summary=summary)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    session_id = (data.get("session_id") or "").strip()

    if not message:
        return jsonify({"error": "No message provided."}), 400

    if not session_id:
        session_id = generate_session_id()

    if session_id not in chat_histories:
        chat_histories[session_id] = []
        save_session_start(session_id=session_id)

    chat_histories[session_id].append({
        "role": "user",
        "content": message
    })

    try:
        assistant_payload = get_chatbot_response(
            message,
            chat_histories[session_id],
            session_id=session_id
        )

        if isinstance(assistant_payload, dict):
            assistant_response = assistant_payload.get("text", "")
            recommended_courses = assistant_payload.get("courses", [])
        else:
            assistant_response = str(assistant_payload)
            recommended_courses = []

    except Exception as e:
        print(f"[CHAT ERROR] {e}")
        assistant_response = (
            "Sorry, something went wrong while generating recommendations. "
            "Please try again."
        )
        recommended_courses = []

    chat_histories[session_id].append({
        "role": "assistant",
        "content": assistant_response
    })

    if len(chat_histories[session_id]) > 30:
        chat_histories[session_id] = chat_histories[session_id][-30:]

    save_chat_log(
        log_file=LOG_FILE,
        session_id=session_id,
        user_message=message,
        assistant_message=assistant_response
    )

    save_chat_message(
        session_id=session_id,
        role="user",
        step="",
        message_text=message
    )

    save_chat_message(
        session_id=session_id,
        role="assistant",
        step="",
        message_text=assistant_response
    )

    return jsonify({
        "session_id": session_id,
        "response": assistant_response,
        "courses": recommended_courses
    })


@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    f = request.files["file"]
    filename = (f.filename or "").lower()

    raw = f.read(MAX_UPLOAD_BYTES)
    if not raw:
        return jsonify({"error": "File is empty."}), 400

    try:
        if filename.endswith(".pdf"):
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                text = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                ).strip()

        elif filename.endswith(".docx"):
            from docx import Document
            doc = Document(io.BytesIO(raw))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        elif filename.endswith(".txt"):
            text = raw.decode("utf-8", errors="ignore").strip()

        else:
            return jsonify({"error": "Unsupported file type. Please upload a PDF, DOCX, or TXT file."}), 400

    except Exception as e:
        print(f"[UPLOAD ERROR] {e}")
        return jsonify({"error": "Could not read the file. Please try a different format."}), 500

    if not text:
        return jsonify({"error": "No readable text found in the file."}), 400

    return jsonify({"text": text})


@app.route("/clear_history", methods=["POST"])
def clear_history():
    data = request.get_json(force=True)
    session_id = (data.get("session_id") or "").strip()

    if session_id and session_id in chat_histories:
        chat_histories.pop(session_id, None)

    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
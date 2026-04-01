from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from chat_logic import get_chatbot_response
from utils import ensure_dir, generate_session_id, save_chat_log


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "chat_history.csv"

ensure_dir(LOG_DIR)

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

    chat_histories[session_id].append({
        "role": "user",
        "content": message
    })

    try:
        assistant_response = get_chatbot_response(message, chat_histories[session_id])
    except Exception as e:
        print(f"[CHAT ERROR] {e}")
        assistant_response = (
            "Sorry, something went wrong while generating recommendations. "
            "Please try again."
        )

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

    return jsonify({
        "session_id": session_id,
        "response": assistant_response
    })


@app.route("/clear_history", methods=["POST"])
def clear_history():
    data = request.get_json(force=True)
    session_id = (data.get("session_id") or "").strip()

    if session_id and session_id in chat_histories:
        chat_histories.pop(session_id, None)

    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
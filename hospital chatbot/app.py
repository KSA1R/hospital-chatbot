"""
app.py

Flask web GUI for the hospital intake bot. This wraps the EXISTING
ConversationSession from state_machine.py in a ChatGPT-style chat
interface - it doesn't touch state_machine.py, llm_client.py,
symptom_kb.py, or appointments_db.py at all. All it does is:

    browser  <-- JSON -->  Flask (this file)  <-->  ConversationSession

Run:
    pip install flask
    python app.py
Then open http://127.0.0.1:5000 in your browser.

NOTE: this keeps ONE conversation in memory for the whole running app,
the same as running main.py once - it's built for local single-patient
testing, not multiple simultaneous patients talking to it at once. If
you need that later, sessions would need to be tracked per-browser
(e.g. via a cookie / session id) instead of the single global below.

TODO (voice, once this GUI is solid): the same STT -> handle_input ->
      TTS hook mentioned in main.py's docstring can plug in here too -
      e.g. a mic button in the GUI posts audio to a new endpoint that
      transcribes it, calls session.handle_input(), then returns audio.
"""

from flask import Flask, render_template, request, jsonify

from state_machine import ConversationSession, State

app = Flask(__name__)

session = ConversationSession()


def _status_payload(reply: str) -> dict:
    return {
        "reply": reply,
        "state": session.state.name,
        "is_emergency": session.state == State.EMERGENCY,
        "is_closing": session.state == State.CLOSING,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/greeting", methods=["GET"])
def greeting():
    """Called once when the page first loads, to get the bot's opening line - mirrors main.py's kickoff call."""
    reply = session.handle_input("")
    return jsonify(_status_payload(reply))


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    user_text = (data.get("message") or "").strip()
    if not user_text:
        return jsonify({"error": "empty message"}), 400
    reply = session.handle_input(user_text)
    return jsonify(_status_payload(reply))


@app.route("/api/reset", methods=["POST"])
def reset():
    """Starts a brand-new conversation from scratch, same as restarting main.py."""
    global session
    session = ConversationSession()
    reply = session.handle_input("")
    return jsonify(_status_payload(reply))


if __name__ == "__main__":
    app.run(debug=True, port=5000)

"""
fact_extractor.py

One LLM call per patient message. It captures every piece of information in the
message, whatever step the bot is on, so a name, age, or symptom mentioned early
is never lost. It also classifies two things the state machine needs:

  data_question:    does the patient ask about their OWN recorded data, or ask
                    for code, the database, or other people's data?
  symptom_consent:  when symptom questions were declined, does the patient now
                    agree (now or later), or still decline?

Built on _call_structured from llm_client.py. If the API fails, the result has
"_failed": True, and the state machine falls back to simple local patterns.
"""

from llm_client import _call_structured

CATEGORIES = [
    "symptom", "identity", "vital", "history", "medication",
    "allergy", "family_history", "social", "other",
]

_FACT_PROMPT = """You are the information-capture step of a hospital intake assistant.
Read the patient's latest message and list every piece of health or identity information in it.

IMPORTANT: capture information even when it was not what the bot just asked. A patient may
mention a name, age, symptom, medication, or vital in any message, at any step.

For each fact, return:
- category: one of symptom, identity, vital, history, medication, allergy, family_history, social, other.
    identity = name, age, or gender only.
    vital = blood pressure, sugar level, weight, height, temperature, or diabetes status.
    history = past conditions. family_history = relatives' conditions (subject = relation).
- subject: short lowercase name, e.g. "headache", "name", "age", "gender", "temperature", "diabetes".
- value: the value as stated, e.g. "Sharon", "51", "female", "101 F", "yes". Null if there is none.
- status: active (current), resolved (no longer present), historical (past), suspected (patient unsure),
  or declined (patient refused to share it). Use active when unclear.
- verbatim_quote: the exact words from the message that support this fact. Copy, do not paraphrase.
- episode: for symptoms only. "new" if this is a complaint not already in OPEN PROBLEMS,
  "existing" if it adds detail to an open problem, "none" for everything else.
- details: for symptoms, any of these keys when the patient states them: onset, location, duration,
  character, aggravating, relieving, radiation, timing, severity. Each as {key, value}.
  Empty list if none. Never guess a detail the patient did not state.

Also return:
- data_question: "own_data" if the patient asks what has been recorded or noted about THEMSELVES
  (e.g. "did you note my name?", "what is the duration of my headache?").
  "other_or_code" if they ask about source code, the database, system settings, other patients or
  users, a CSV file, or any other person's data. Otherwise "none". Never answer the question yourself.
- symptom_consent: only when SYMPTOM QUESTIONS CURRENTLY DECLINED is true:
  "agrees_now" = agrees to be asked about symptoms now (e.g. "I consent", "ok you can ask",
  "I can give my symptoms, its ok").
  "agrees_later" = agrees, but after finishing something else (e.g. "ok, I'll get there after I finish").
  "declines" = still does not want to share symptoms.
  Otherwise "none". Typos and informal wording count. Read the meaning.
- wants_to_stop: true only if the patient says they have nothing more to add.
- off_topic_remark: a short description (under 12 words) of any part that is not health or identity
  information, such as small talk or a question about the bot. Otherwise null.

Rules:
- Never invent information. A missing field is null, not a guess.
- A bare yes, no, or ok with no information gives an empty facts list.
- A negated symptom ("no fever") is not a fact.
- A relative's condition is family_history, e.g. subject "mother", value "diabetes".
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": CATEGORIES},
                    "subject": {"type": "string"},
                    "value": {"type": "string", "nullable": True},
                    "status": {
                        "type": "string",
                        "enum": ["active", "resolved", "historical", "suspected", "declined"],
                    },
                    "verbatim_quote": {"type": "string"},
                    "episode": {"type": "string", "enum": ["new", "existing", "none"]},
                    "details": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string"},
                                "value": {"type": "string"},
                            },
                            "required": ["key", "value"],
                        },
                    },
                },
                "required": ["category", "subject", "value", "status", "verbatim_quote", "episode", "details"],
            },
        },
        "data_question": {"type": "string", "enum": ["none", "own_data", "other_or_code"]},
        "symptom_consent": {
            "type": "string",
            "enum": ["none", "agrees_now", "agrees_later", "declines"],
        },
        "wants_to_stop": {"type": "boolean"},
        "off_topic_remark": {"type": "string", "nullable": True},
    },
    "required": ["facts", "data_question", "symptom_consent", "wants_to_stop", "off_topic_remark"],
}


def extract_facts(
    user_text: str,
    recent_context: list[str] | None,
    open_problems: list[str],
    step_name: str,
    symptoms_declined: bool,
) -> dict:
    """
    Returns the parsed extraction. On failure returns the same shape with
    "_failed": True and nothing extracted.
    """
    prompt = _FACT_PROMPT + (
        f"\nCURRENT STEP: {step_name}"
        f"\nSYMPTOM QUESTIONS CURRENTLY DECLINED: {'true' if symptoms_declined else 'false'}"
        f"\nOPEN PROBLEMS: {', '.join(open_problems) if open_problems else 'none'}"
    )
    if recent_context:
        prompt += (
            "\n\nRECENT CONVERSATION (context only, use it to understand the message; "
            "do not re-extract facts from earlier messages):\n" + "\n".join(recent_context)
        )
    fallback = {
        "facts": [],
        "data_question": "none",
        "symptom_consent": "none",
        "wants_to_stop": False,
        "off_topic_remark": None,
        "_failed": True,
    }
    result = _call_structured(user_text, prompt, _SCHEMA, fallback)
    if result is not fallback:
        result["_failed"] = False
    return result

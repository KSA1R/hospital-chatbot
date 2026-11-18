"""
llm_client.py

Wraps calls to the Gemini API. Kept in its own file so state_machine.py
never touches API details directly.

SETUP REQUIRED:
    1. Get an API key from https://aistudio.google.com/apikey
    2. Set it as an environment variable before running the bot:
         export GEMINI_API_KEY="your-key-here"      (Mac/Linux)
         setx GEMINI_API_KEY "your-key-here"         (Windows)
    3. Never hardcode the key in this file or commit it to git.

ARCHITECTURE:
    _call_structured() is the ONE place that actually talks to the API -
    it handles the client, retries, JSON parsing, and error fallback.
    Every specific need (symptom extraction, yes/no intent, vitals
    parsing) is a small wrapper function that just supplies its own
    prompt + schema + fallback value to _call_structured().

    This means adding a new kind of extraction later is just adding one
    short wrapper function, not copy-pasting the whole API-calling block
    again.

    SLOT-FILLING, NOT SEQUENCE-MATCHING: almost every extractor below
    now also returns "off_topic_remark" - a short, neutral note about
    anything in the message that wasn't an answer to what was asked
    (small talk, a random question, a tangent). state_machine.py uses
    this to acknowledge-and-redirect instead of either ignoring it or
    silently re-asking the same question, which is what used to happen.
    Similarly, extract_vitals_batch() deliberately extracts EVERY vital
    field on EVERY turn (not just "the one we just asked about"), so a
    patient who volunteers info out of order is captured immediately
    instead of being asked again once the bot "gets to" that field.

NOTE ON MODEL NAME: check https://ai.google.dev/gemini-api/docs/models
for the current recommended fast/cheap model name, since these change
over time - don't trust this hardcoded value indefinitely.
"""

import json
import os
import re
import time

from google import genai
from google.genai import types
from google.genai.errors import ServerError

MODEL_NAME = "gemini-3.8-flash"

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable not set. "
                "See the setup notes at the top of llm_client.py."
            )
        # attempts=1: disable the SDK's automatic retries. On a tight
        # free-tier quota, one failed call auto-retrying 5x can burn
        # through your whole per-minute allowance by itself.
        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
    return _client


def _call_structured(user_text: str, system_prompt: str, schema: dict, fallback: dict) -> dict:
    """
    Generic structured-extraction call. Every specific extraction need
    (symptoms, yes/no, vitals, etc.) goes through this one function.

    - user_text: the raw thing the patient typed
    - system_prompt: instructions specific to this extraction task
    - schema: the exact JSON shape we want back (Gemini enforces this)
    - fallback: what to return if anything goes wrong, so a failure
      never crashes the conversation - just degrades gracefully

    Retries ONCE, after a short pause, specifically for transient
    "server overloaded" (503) errors - this is different from the SDK's
    built-in retry (disabled above via attempts=1), which retried
    silently up to 5x and could burn through a whole free-tier quota
    window on a single failure. One deliberate, visible retry here is
    safe because it only fires for a specific, genuinely transient error.

    Returns parsed JSON as a dict, or `fallback` on any failure.
    """
    for attempt in (1, 2):
        try:
            client = _get_client()
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_text,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0,  # deterministic extraction, not creative writing
                ),
            )
            return json.loads(response.text)

        except ServerError as e:
            if attempt == 1:
                print(f"[llm_client] server busy (attempt 1), retrying once in 2s: {e}")
                time.sleep(2)
                continue
            print(f"[llm_client] server still unavailable after retry, falling back: {e}")
            return fallback

        except Exception as e:
            # TODO: log this properly instead of printing, once you have logging set up
            print(f"[llm_client] extraction failed, falling back to default: {e}")
            return fallback


from symptom_kb import SYMPTOM_TABLE

_DIALOGUE_LABEL_RE = re.compile(r'(?:^|\n)\s*(?:bot|assistant|patient|you|user)\s*:\s*', re.IGNORECASE)

# Shared instruction fragment appended to every extraction prompt below,
# so off-topic detection behaves consistently everywhere instead of
# being redefined slightly differently in each prompt.
_OFF_TOPIC_INSTRUCTION = """
- "off_topic_remark": if part of the message is unrelated to what's being
  asked here (small talk, a GREETING like "hi"/"hello"/"hii" - including a
  greeting that's the patient's ENTIRE message with no actual answer in
  it, a question about you/the bot, a random tangent, a joke, etc.), a
  short neutral description of it (under 12 words), phrased to make clear
  what kind of remark it is - e.g. "said hello" (greeting), "asked
  whether you are Claude" (question), "mentioned they hate hospitals"
  (statement). Otherwise null. A plain "I don't know" / "no" / "skip" is
  a normal, on-topic non-answer - NOT off-topic (it directly answers what
  was asked, unlike a greeting, which doesn't).
  CRITICAL: off_topic_remark is ADDITIONAL to every other field above, never
  a REPLACEMENT for them. A single message routinely does two things at
  once - answers the actual question AND tacks on an aside (e.g. "nope,
  but I have a doll", "50kg, by the way is this covered by insurance?").
  Extract every other field normally from the on-topic part FIRST, and
  ONLY THEN separately note the aside here - never let noticing an aside
  cause you to under-extract, skip, or soften a clear answer elsewhere in
  the same message.
"""

_OFFTOPIC_ANSWER_PROMPT = """You are a hospital intake assistant. A patient said something off-topic during
their intake conversation. Here is a short description of what they said, which also indicates whether it
was a QUESTION or a STATEMENT: "{off_topic}"

{snapshot_block}

Write ONE short, natural sentence handling it correctly based on which type it is:
- If it's a GREETING (e.g. "said hello", with no real answer to the actual question in the message):
  return the greeting briefly and warmly, the way a person would ("Hi there!", "Hey, good to hear from
  you!") - one short greeting, not a paragraph, and don't ask how THEY are doing (you're mid-intake, not
  making small talk).
- If it's a QUESTION ABOUT THE PATIENT'S OWN DATA (e.g. "what was my temperature", "what did I say my
  weight was", "what's that in Celsius"): answer it using ONLY the KNOWN PATIENT INFO given above - if a
  Celsius/Fahrenheit conversion is already provided there, use that exact figure rather than calculating
  your own. If the info needed isn't listed there, say plainly that it isn't recorded yet - never invent or
  guess a number that isn't actually in the KNOWN PATIENT INFO.
- If it's a QUESTION about you/the bot in general (e.g. "asked whether you are Claude"): answer it plainly
  and honestly, in one sentence.
- If it's a STATEMENT (e.g. "mentioned they hate hospitals", "said it's raining"): acknowledge it briefly
  and genuinely, the way a warm, attentive person would - not a bare log-entry restatement.

Do not give medical advice or engage with anything substantive beyond this one short, genuine sentence -
this is a quick aside on the way back to the real conversation, not the main point. One sentence only."""


def answer_offtopic_remark(off_topic: str, patient_snapshot: str | None = None) -> str | None:
    """
    Generates an actual short response to an off-topic remark, instead
    of a bare "Noted - X" acknowledgment or a guessed answer.

    patient_snapshot (optional): a compact summary of what's actually
    been collected so far (e.g. "temperature: 99.1 F (equivalent to
    about 37.3 C)"). Without this, a genuine question about the
    patient's OWN data - "what was my temperature?" - had no real facts
    to draw on and could only guess or recall loosely from conversation
    text, which is also why it couldn't handle a follow-up unit
    conversion request correctly. Always pass this when available.

    Returns None on failure, so the caller can fall back to a plain
    generic acknowledgment rather than leaving the remark unaddressed.
    """
    if not off_topic:
        return None
    snapshot_block = f"KNOWN PATIENT INFO SO FAR:\n{patient_snapshot}" if patient_snapshot else "KNOWN PATIENT INFO SO FAR: nothing collected yet."
    prompt = _OFFTOPIC_ANSWER_PROMPT.format(off_topic=off_topic, snapshot_block=snapshot_block)
    return _call_text(prompt, "Write the one-sentence response now.")


def _sanitize_composed_reply(text: str) -> str:
    """
    Defensive cleanup, independent of the prompt: strips out any
    hallucinated dialogue-turn labels ("Bot:", "Patient:", "You:", etc.)
    the model might generate, and cuts off anything after the first such
    label. The composed reply must always be exactly ONE line, in the
    assistant's own voice - never a fabricated back-and-forth. This
    exists because prompt instructions alone (see _COMPOSE_PROMPT_TEMPLATE)
    aren't enough to guarantee this on their own - this is a hard
    code-level guarantee that doesn't depend on the model cooperating.
    """
    match = _DIALOGUE_LABEL_RE.search(text)
    if match and match.start() == 0:
        # Text starts with a label, e.g. "Bot: hello there" - strip just
        # the label itself, then check if a SECOND label follows (a
        # hallucinated continuation) and cut there if so.
        text = text[match.end():]
        second = _DIALOGUE_LABEL_RE.search(text)
        if second:
            text = text[:second.start()]
    elif match:
        # A label shows up partway through - the model started writing a
        # new turn. Cut everything from that point onward.
        text = text[:match.start()]
    return text.strip()


def _call_text(system_prompt: str, user_message: str) -> str | None:
    """
    Like _call_structured, but for generating freeform reply TEXT instead
    of extracting structured data. Returns None on any failure (instead
    of a JSON fallback dict) so the caller can fall back to a plain
    hardcoded string - the conversation should NEVER go silent just
    because the empathy/phrasing call failed.
    """
    for attempt in (1, 2):
        try:
            client = _get_client()
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,  # some natural variation in phrasing is fine here, unlike extraction
                ),
            )
            return _sanitize_composed_reply(response.text.strip())

        except ServerError as e:
            if attempt == 1:
                print(f"[llm_client] server busy composing reply (attempt 1), retrying once in 2s: {e}")
                time.sleep(2)
                continue
            print(f"[llm_client] server still unavailable, falling back to plain text: {e}")
            return None

        except Exception as e:
            print(f"[llm_client] reply composition failed, falling back to plain text: {e}")
            return None


_COMPOSE_PROMPT_TEMPLATE = """You are the voice of a hospital intake assistant, speaking directly to a patient.

Speak the way a good, experienced clinician would - the way a skilled psychiatrist or nurse listens and responds: genuinely present and attentive, professional, and measured. Not performative, not robotic, not indifferent.

- Acknowledge what the patient shares in a way that shows you actually registered it - not a generic template stamp repeated every turn.
- Do NOT thank the patient for giving routine information (symptoms, vitals, medical history, personal details). Answering a question is not something to be thanked for. Reserve genuine thanks for something actually significant - like when they agree to proceed / grant consent - not for routine answers.
- Do NOT over-apologize, over-praise, or pile on reassurance ("that's totally understandable!", "great job telling me that", "I really appreciate you opening up"). This reads as fake and undermines trust, the same way an overly effusive therapist would.
- Do NOT be flat, clinical, or indifferent either - if a patient describes pain or distress, let that register naturally in tone, without performing sympathy or over-emoting.
- Be concise. A good clinician doesn't pad responses with filler or unnecessary softening.

TONE: {tone}
IS_FIRST_MESSAGE: {is_first}

RULES (never break these):
- CRITICAL - DO NOT INVENT THE PATIENT'S EMOTIONAL STATE OR REASONS: never tell the patient how they're
  feeling, or guess WHY they said what they said, unless they stated it themselves in plain words this
  conversation. This includes but is not limited to: "feeling overwhelmed", "feeling rushed", "in a hurry",
  "anxious", "stressed", "busy". A short, casual, off-topic, or unclear reply (e.g. "hi", "hello, how are
  you", "ok", "no") is NOT evidence of any emotional state or motive - it is just a short reply. Do not
  invent one for the sake of sounding empathetic or to explain why you're asking again. If you need to
  acknowledge a repeated/unclear reply, acknowledge concretely WHAT they said (or that it didn't answer the
  question) - never WHY you think they said it or how they must feel.
- If IS_FIRST_MESSAGE is True, you may greet the patient once (e.g. "Hi"). Otherwise, do NOT greet them again -
  the conversation is already underway; re-greeting every turn feels robotic and unsettling.
- If TONE is "serious": stay calm and steady, no exclamation points, no minimizing words ("just", "probably
  nothing"), no alarming words either ("dangerous", "critical", "could be fatal"). Speak like a calm,
  experienced clinician delivering something that needs prompt attention.
- If TONE is "normal": warm and professional, per the guidance above.
- CONTENT below is what must be conveyed IN SUBSTANCE - it is guidance on meaning, not a script to imitate.
  You have full freedom to compose your own natural wording for it from scratch. Do NOT copy its phrasing,
  including any pleasantries or fillers it happens to contain.
- CRITICAL - PERSPECTIVE: you are ALWAYS the assistant speaking TO the patient, using "you" to refer to the
  patient. You are NEVER the patient. If CONTENT is phrased as a question (e.g. "do you know your weight?"),
  your job is to ASK that question to the patient in your own words - NOT to answer it, and NOT to switch
  into the patient's voice (e.g. never say "I don't know my weight" - that is the patient's line, not yours).
  Do not let the recent conversation history pull you into writing the patient's next reply instead of your own.
- CRITICAL - DO NOT CHANGE WHAT IS BEING ASKED: you may reword CONTENT, but never change its underlying
  meaning or the TYPE of answer it's asking for. For example, if CONTENT asks whether the patient KNOWS a
  numeric reading (e.g. "do you know your blood pressure reading?"), do not rephrase it into a yes/no
  diagnosis question (e.g. "have you been diagnosed with high blood pressure?") - those expect completely
  different answers and will corrupt the data collected. Rewording must preserve both the topic AND the
  exact kind of answer being requested (a number vs. a yes/no fact vs. an open description, etc.).
- If CONTENT includes a short response to something off-topic the patient said, that response is a genuine,
  already-composed answer/acknowledgment - preserve its substance (don't strip it down to a bare "noted"), and
  transition smoothly from it into the real question right after, so it reads as one natural, flowing reply
  rather than two disconnected sentences bolted together.
- If LOCKED CONTENT is provided, those specific facts (numbers, conditions, urgency levels, reference IDs)
  must survive exactly as given, word for word. Nothing outside of explicitly locked content is fixed.
- RECENT CONVERSATION below is background ONLY, so your phrasing can flow naturally from what was just said.
  Do NOT re-answer, re-explain, or bring back up any topic, question, or aside that appears in RECENT
  CONVERSATION unless it is explicitly part of CONTENT (or the REPEAT NOTICE below) for THIS turn - even if
  that topic feels unresolved or interesting. Something the patient asked two turns ago is closed once you've
  answered it once; don't spontaneously revisit it.
- Keep it to 1-2 sentences. Do not add new medical claims or advice beyond what's given.
- Avoid using the patient's name more than once across the whole conversation, if possible.

CONTENT (what must be conveyed - compose fresh wording, do not mimic this phrasing):
{content}
{locked_block}
{repeat_block}
{history_block}

Now write ONLY the assistant's next line to the patient. Do not write the patient's response."""


def compose_reply(
    content: str,
    tone: str = "normal",
    locked_facts: str | None = None,
    is_first_message: bool = False,
    recent_history: list[str] | None = None,
    repeat_count: int = 0,
) -> str | None:
    """
    Generates a reply from scratch based on `content` (what needs to be
    conveyed) - `content` is guidance, never phrasing to preserve. Only
    `locked_facts`, when explicitly given, is treated as fixed wording
    (numbers, conditions, urgency, reference IDs). This is deliberate:
    earlier versions defaulted locked_facts to the full fallback string,
    which meant incidental phrasing (like a hardcoded "Thanks.") got
    treated as something to protect - this fix removes that entirely.

    `recent_history` is the last few conversation turns, so replies can
    reference what the patient actually just said. Adds tokens to this
    SAME call, not an extra API call - no added quota cost.

    `repeat_count` is how many consecutive unclear/unusable replies the
    patient has given to THIS SAME question before now (0 = first ask).
    Every re-prompt in the app now goes through this function instead of
    ever returning a raw hardcoded string a second time - a stuck patient
    should see the bot visibly react (acknowledge, rephrase, eventually
    give a concrete example), not the identical sentence bounced back at
    them, which reads as broken rather than attentive.

    Returns None on failure - callers MUST fall back to their own plain
    text in that case, never leave the user without a reply.
    """
    locked_block = f"\n\nLOCKED CONTENT (must survive exactly, word for word):\n{locked_facts}" if locked_facts else ""
    history_block = ""
    if recent_history:
        history_block = "\n\nRECENT CONVERSATION (background only, see rule above - do not revisit unprompted):\n" + "\n".join(recent_history)
    repeat_block = ""
    if repeat_count > 0:
        repeat_block = (
            f"\n\nREPEAT NOTICE: this is attempt #{repeat_count + 1} at this same question - the patient's "
            "previous reply(ies) didn't give a usable answer. Do NOT reuse the same wording you used before "
            "(see RECENT CONVERSATION for how you phrased it last time) - rephrase it noticeably differently. "
            "Do NOT quote the patient's words back at them (no quotation marks, no 'You mentioned/said X' "
            "template) - that reads as robotic and mechanical, especially if it happens on every retry. "
            "Just move naturally into asking again, briefly signaling that you still need an answer (e.g. "
            "'Just so I can move forward -', 'To make sure I've got it -'), the way a person would when "
            "gently repeating a question, without a formulaic acknowledgment stamp. Do NOT guess at or "
            "comment on their tone, mood, or reason for saying it."
        )
        if repeat_count >= 2:
            repeat_block += (
                " This has now happened multiple times - also give a concrete example of the kind of answer "
                "you're looking for (e.g. a sample number, or 'you can just say yes, no, or I don't know')."
            )
    prompt = _COMPOSE_PROMPT_TEMPLATE.format(
        tone=tone, content=content, is_first=is_first_message, locked_block=locked_block,
        history_block=history_block, repeat_block=repeat_block,
    )
    # IMPORTANT: do NOT pass `content` itself as the "user message" here.
    # `content` is already embedded in the system prompt's CONTENT block
    # above. If it's ALSO sent as the model's incoming message, Gemini
    # sees something like "Is that okay with you?" arrive as if the
    # PATIENT said it, and answers it ("Yes, that's okay.") instead of
    # realizing it's the line it's supposed to say TO the patient. A
    # short, neutral trigger avoids that confusion entirely.
    return _call_text(prompt, "Write the reply now, following the rules and CONTENT above.")


# ----------------------------------------------------------------------
# Wrapper 1: symptom extraction (used in SYMPTOM_COLLECTION state)
# ----------------------------------------------------------------------

def _canonical_symptom_terms() -> list[str]:
    """
    Pulls the exact vocabulary symptom_kb.py's lookup_condition() matches
    against. Built dynamically (not hardcoded here) so this can NEVER
    silently drift out of sync with the KB - if someone edits
    SYMPTOM_TABLE, this list updates automatically next call.
    """
    vocab = set()
    for row in SYMPTOM_TABLE:
        vocab.update(row["symptoms"])
    return sorted(vocab)


def _build_symptom_prompt() -> str:
    canonical = ", ".join(f'"{term}"' for term in _canonical_symptom_terms())
    return f"""You extract medical symptom information from a patient's message.

Rules:
- "symptoms": list any symptoms mentioned. Each symptom is an OBJECT, not
  just a name - fill in every field below:
    - "name": IMPORTANT for matching - if what the patient describes
      clearly corresponds to one of these known terms, use that EXACT
      term, even if they phrased it differently (e.g. "throwing up" or
      "puking" -> "vomiting", "tummy ache" -> "stomach pain"): {canonical}.
      If a symptom is real but doesn't match any known term, describe it
      in your own simple lowercase term instead.
    - "duration_value" + "duration_unit": how long the patient has had
      this symptom, if they said so (e.g. "3 continuous days" ->
      duration_value=3, duration_unit="days"). If not mentioned, both
      are null - NEVER guess a duration the patient didn't state.
    - "severity": "mild", "moderate", or "severe", ONLY if the patient's
      wording clearly implies one of these (e.g. "unbearable" -> severe,
      "a little" -> mild). If not stated or unclear, null - do not guess.
    - "onset": "sudden" or "gradual", ONLY if the patient's wording
      clearly implies one (e.g. "started suddenly" -> sudden, "been
      building up" -> gradual). If not stated, null.
  If the message negates a symptom (e.g. "no fever"), do NOT include it.
  If no symptoms are mentioned, return an empty list.
- "symptoms_to_remove": list the plain NAME (string, not object) of any
  symptoms the patient is explicitly RETRACTING or CORRECTING from
  earlier in the conversation (e.g. "actually, I don't have a fever, I
  was wrong", "wait, no nausea after all"). This is different from
  simply not mentioning a symptom again - it means they're taking back
  something they said before. If no correction is being made, return an
  empty list.
- "wants_to_stop": true if the patient is indicating they have nothing
  more to add (e.g. "no", "that's all", "nothing else"). Otherwise false.
  IMPORTANT: this is INDEPENDENT of off_topic_remark below - a message can
  be BOTH a "nothing else" answer AND contain an unrelated aside in the
  same breath (e.g. "nope, but I have a doll", "no, though my cat is
  sick") - a trailing aside does NOT cancel out a clear "no"/"nope" that
  came right before it. Extract BOTH: wants_to_stop should still be true,
  and the aside should still be captured in off_topic_remark. Never let
  noticing the aside distract you from also recognizing the stop signal.
- Never invent a symptom, duration, severity, or onset the patient
  didn't mention or clearly imply. Leaving a field null is always
  better than guessing.
{_OFF_TOPIC_INSTRUCTION}"""


_SYMPTOM_SCHEMA = {
    "type": "object",
    "properties": {
        "symptoms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "duration_value": {"type": "number", "nullable": True},
                    "duration_unit": {
                        "type": "string",
                        "enum": ["hours", "days", "weeks", "months", "years"],
                        "nullable": True,
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["mild", "moderate", "severe"],
                        "nullable": True,
                    },
                    "onset": {
                        "type": "string",
                        "enum": ["sudden", "gradual"],
                        "nullable": True,
                    },
                },
                "required": ["name", "duration_value", "duration_unit", "severity", "onset"],
            },
        },
        "symptoms_to_remove": {"type": "array", "items": {"type": "string"}},
        "wants_to_stop": {"type": "boolean"},
        "off_topic_remark": {"type": "string", "nullable": True},
    },
    "required": ["symptoms", "symptoms_to_remove", "wants_to_stop", "off_topic_remark"],
}


def extract_symptoms(user_text: str) -> dict:
    """
    Returns e.g.
    {
        "symptoms": [
            {"name": "fever", "duration_value": 3, "duration_unit": "days",
             "severity": "moderate", "onset": None},
        ],
        "symptoms_to_remove": [],
        "wants_to_stop": False,
        "off_topic_remark": None,
    }
    """
    fallback = {"symptoms": [], "symptoms_to_remove": [], "wants_to_stop": False, "off_topic_remark": None}
    prompt = _build_symptom_prompt()   # rebuilt each call so KB edits are always reflected
    return _call_structured(user_text, prompt, _SYMPTOM_SCHEMA, fallback)


# ----------------------------------------------------------------------
# Wrapper 1b: triage reasoning (used in TRIAGE state, AFTER the plain
# symptom_kb.py table lookup) - takes the full structured picture
# (symptoms with duration/severity/onset, history, vitals) plus the
# table's simple name-only-match hint, and asks Gemini to reason over
# all of it together. The table lookup stays deterministic and
# untouched; this step refines it, and is never treated as more
# trustworthy than a human doctor - see _TRIAGE_PROMPT below.
# ----------------------------------------------------------------------

def _format_symptoms_for_prompt(symptoms: list[dict]) -> str:
    """Turns the structured symptom list into a short readable line for the prompt."""
    if not symptoms:
        return "none reported"
    pieces = []
    for s in symptoms:
        bits = [s.get("name", "unknown")]
        if s.get("duration_value") is not None and s.get("duration_unit"):
            bits.append(f"for {s['duration_value']} {s['duration_unit']}")
        if s.get("severity"):
            bits.append(f"({s['severity']})")
        if s.get("onset"):
            bits.append(f"[{s['onset']} onset]")
        pieces.append(" ".join(bits))
    return "; ".join(pieces)


_TRIAGE_PROMPT = """You are a careful medical triage assistant helping a hospital intake bot.
You are NOT making a final diagnosis - you are giving a best-effort, cautious
triage suggestion based on the information given, the same way an experienced
triage nurse would form a first impression before a doctor sees the patient.

You will be given:
- A structured list of the patient's reported symptoms, including duration,
  severity, and onset where known.
- The patient's medical history (conditions, medications, allergies), if known.
- The patient's relevant vital signs, if known.
- A reference hint from a simple lookup table. That table ONLY matches on
  symptom NAMES - it has no idea about duration, severity, history, or
  vitals. Treat it as a rough starting point only, not a fixed answer -
  adjust or override it if the fuller picture (especially duration and
  severity) points somewhere more specific, or suggests higher or lower
  urgency than the table's guess.

Rules:
- "condition": your best-effort plain-language description of what this
  could be. Null if you genuinely cannot narrow it down at all - do not
  force a guess.
- "urgency": one of "low", "medium", "high", "critical", or "unknown".
  Base this on the FULL picture - a symptom that has lasted a long time,
  a severe symptom, or a concerning combination with history/vitals
  should raise urgency even if the table's hint alone was "low".
- "advice": one or two sentences of safe, general next-step advice (e.g.
  "monitor symptoms", "see a doctor within a day or two", "seek care
  promptly"). NEVER give a definitive diagnosis, a specific drug or
  dosage recommendation, or claim more certainty than you actually have.
- Be conservative: if genuinely in doubt, lean toward recommending the
  patient see a doctor rather than reassuring them everything is fine.
"""

_TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "condition": {"type": "string", "nullable": True},
        "urgency": {"type": "string", "enum": ["low", "medium", "high", "critical", "unknown"]},
        "advice": {"type": "string"},
    },
    "required": ["condition", "urgency", "advice"],
}


def triage_reasoning(
    symptoms: list[dict],
    history: dict | None,
    vitals: dict | None,
    kb_hint: dict | None,
) -> dict | None:
    """
    Reasons over the full picture (structured symptoms + history + vitals
    + the plain table's hint) to produce a refined condition/urgency/advice.

    Returns None on ANY failure (not a fallback dict) - deliberately, so
    the caller falls back to kb_hint (or no_match_response()) directly,
    since state_machine.py already has a perfectly good deterministic
    result available and shouldn't need a second fallback shape here.
    """
    user_text = (
        f"Symptoms: {_format_symptoms_for_prompt(symptoms)}\n"
        f"History: {history or 'none provided'}\n"
        f"Vitals: {vitals or 'none provided'}\n"
        f"Reference table hint: {kb_hint or 'no match found in table'}"
    )
    return _call_structured(user_text, _TRIAGE_PROMPT, _TRIAGE_SCHEMA, None)


# ----------------------------------------------------------------------
# Wrapper 2: yes/no intent (used in CONSENT, APPOINTMENT_BOOKING,
# ANYTHING_ELSE). Returns a dict now (not a bare string) so callers can
# also see off_topic_remark and acknowledge it instead of silently
# dropping it or just repeating the question.
# ----------------------------------------------------------------------

_YES_NO_PROMPT = f"""You classify whether a patient's reply is affirmative, negative, or unclear.

Rules:
- "intent": one of exactly "yes", "no", or "unclear".
  Treat casual affirmatives ("yeah", "sure", "yep", "okay", and a standalone
  "s" used as shorthand agreement) as "yes".
  Treat casual negatives ("nah", "not really", "no thanks") as "no".
- A bare GREETING ("hi", "hello", "hey", "good morning", etc.) with nothing
  else is NOT an answer to the question that was just asked - it's
  "unclear", never "yes". This applies even if the greeting sounds
  friendly or agreeable in tone; friendliness is not consent. Only
  classify "yes" when the patient has actually responded to what was asked.
- IMPORTANT: work out the actual logical meaning for indirect or
  double-negative phrasing - do NOT just pattern-match on the presence
  of the word "no" and assume negative. Phrases like "I don't see why
  not", "sure, why not", "no objections", "can't think of a reason not
  to", "why would I say no" all logically mean YES, even though they
  contain "no" or "not". Reason through what the patient is actually
  agreeing to or refusing, not which negation words appear.
- If the reply doesn't clearly answer yes or no (e.g. it changes the
  subject, or is a real answer to a different question), return "unclear".
{_OFF_TOPIC_INSTRUCTION}"""

_YES_NO_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["yes", "no", "unclear"]},
        "off_topic_remark": {"type": "string", "nullable": True},
    },
    "required": ["intent", "off_topic_remark"],
}


_CLOSING_INTENT_PROMPT = """You classify a patient's message that arrived AFTER the conversation was
already closed (the bot already said something like "Take care! Reach out
anytime you need help.").

Rules:
- "intent": one of exactly "new_request" or "closing_ack".
  "new_request" = the patient is asking for something new: a new symptom,
  a new appointment, a question, anything that needs the bot to actually
  do something. Examples: "book another appointment", "I have a headache
  now", "can you check my blood pressure again", "actually I need help
  with something else".
  "closing_ack" = the patient is just acknowledging the close, with
  nothing further needed: "ok", "thanks", "bye", "got it", "sounds good",
  a bare greeting, or an empty/meaningless message.
- When genuinely unsure, prefer "closing_ack" - reopening the whole
  intake flow for something that was really just a sign-off is more
  disruptive than occasionally asking "what would you like help with?"
  one extra time.
- "off_topic_remark": short note on anything present that doesn't fit
  either category cleanly (e.g. a random comment), else null.
"""

_CLOSING_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["new_request", "closing_ack"]},
        "off_topic_remark": {"type": "string", "nullable": True},
    },
    "required": ["intent", "off_topic_remark"],
}


def extract_closing_intent(user_text: str, recent_context: list[str] | None = None) -> dict:
    """
    Used ONLY in State.CLOSING (see _handle_closing). Distinguishes a
    patient typing something substantive after close ("book another
    appointment") from a plain sign-off ("ok", "thanks", "bye") -
    extract_yes_no can't do this, since it's built to interpret a reply
    to a yes/no QUESTION the bot just asked, and closing doesn't ask one.
    Without this, CLOSING previously fed every message through pure
    LLM composition with no real intent check behind it - which is how a
    patient asking to book another appointment got a reply that SOUNDED
    like it agreed, but never actually re-entered the booking flow, so
    nothing was ever really booked (no reference number, nothing written
    to the DB).
    """
    prompt = _CLOSING_INTENT_PROMPT
    if recent_context:
        prompt += (
            "\n\nRECENT CONVERSATION (background only):\n" + "\n".join(recent_context)
        )
    fallback = {"intent": "closing_ack", "off_topic_remark": None}
    return _call_structured(user_text, prompt, _CLOSING_INTENT_SCHEMA, fallback)


def extract_yes_no(user_text: str, recent_context: list[str] | None = None) -> dict:
    """
    Returns e.g. {"intent": "yes", "off_topic_remark": None}. Fallback
    intent is "unclear".

    recent_context (optional): the last few conversation turns, e.g.
    ["Bot: Do you have diabetes?", "Patient: I just said yes"]. Without
    this, a reply like "I just said yes" or a bare "yeah" is genuinely
    ambiguous to the model in isolation - it has no idea WHAT they're
    saying yes to. Pass this whenever the reply might be short or
    backward-referencing, which is most of the time for yes/no answers.
    """
    prompt = _YES_NO_PROMPT
    if recent_context:
        prompt += (
            "\n\nRECENT CONVERSATION (use this ONLY to work out what the "
            "patient's reply is actually answering - e.g. a bare 'yeah' or "
            "'I already said yes' only makes sense in light of what was "
            "just asked):\n" + "\n".join(recent_context)
        )
    fallback = {"intent": "unclear", "off_topic_remark": None}
    return _call_structured(user_text, prompt, _YES_NO_SCHEMA, fallback)


# ----------------------------------------------------------------------
# Wrapper 3: batched vitals extraction (used in the single
# VITALS_COLLECTION state, covering diabetes + every numeric vital in
# ONE call). This deliberately does NOT scope itself to "the field we
# just asked about" - it looks for every field, every turn, so the
# patient can answer in any order, volunteer several values in one
# message, or bring up something unrelated, and the bot handles all of
# it correctly instead of only understanding the one thing it expected.
# ----------------------------------------------------------------------

_VITALS_FIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["known", "unclear", "will_provide", "doesnt_know", "declined", "not_mentioned"],
        },
        "value_text": {"type": "string", "nullable": True},
        "numeric_value": {"type": "number", "nullable": True},
        "secondary_numeric_value": {"type": "number", "nullable": True},
        "unit": {"type": "string", "nullable": True},
        "clarifying_question": {"type": "string", "nullable": True},
    },
    "required": ["status", "value_text", "numeric_value", "secondary_numeric_value", "unit", "clarifying_question"],
}

_VITALS_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "diabetes": {
            "type": "object",
            "properties": {
                "stated": {"type": "boolean"},
                "has_diabetes": {"type": "boolean", "nullable": True},
                "declined": {"type": "boolean"},
            },
            "required": ["stated", "has_diabetes", "declined"],
        },
        "sugar_level": _VITALS_FIELD_SCHEMA,
        "blood_pressure": _VITALS_FIELD_SCHEMA,
        "weight": _VITALS_FIELD_SCHEMA,
        "height": _VITALS_FIELD_SCHEMA,
        "temperature": _VITALS_FIELD_SCHEMA,
        "off_topic_remark": {"type": "string", "nullable": True},
    },
    "required": [
        "diabetes", "sugar_level", "blood_pressure", "weight", "height",
        "temperature", "off_topic_remark",
    ],
}

_VITALS_BATCH_PROMPT = """You extract medical vitals from a patient's message, for a hospital intake bot.

IMPORTANT: the patient may state ANY of these fields, in ANY order, in a single
message - even fields that weren't just asked about. Extract every field they
clearly stated, not just the most recently requested one. Never invent a value
for something they didn't mention - leave it unstated/unknown instead.

- diabetes: "stated" true if they said whether they have diabetes (either
  way). "has_diabetes": true/false if stated, else null. "declined" true only
  if they explicitly refuse to answer this specific question.
- sugar_level, blood_pressure, weight, height, temperature: each is a numeric
  vital reading. For each, return:
    - "status": one of:
        "known" - a clear, usable reading was given (e.g. "50 kg", "120/80",
          or a single number for weight/height/temperature/sugar_level -
          those are naturally single numbers, so one number IS complete).
        "unclear" - the patient gave SOME numeric info, but it's genuinely
          ambiguous or incomplete in a way that matters medically. The
          main case: a single bare number for blood_pressure specifically
          (e.g. "my blood pressure is 100") - a real reading needs both a
          systolic and diastolic number, so a lone number is ambiguous
          (could be just the top number, or a misunderstanding of the
          question). Do NOT guess which one they meant, and do NOT treat
          it as a complete reading - flag it instead.
        "will_provide" - the patient CONFIRMS they know this reading, but
          hasn't actually stated the number/value itself yet (e.g. a bare
          "yes", "yeah", "yup", "mhm" in answer to "do you know your X
          reading?"). This is different from "known" (no usable value is
          present yet) and different from "doesnt_know" (they're saying
          they DO know it) - it means the bot must still ask them what the
          actual value is, not silently move on to a different vital as
          if this one were resolved.
        "doesnt_know" - they indicate they don't know/haven't measured it,
          including a bare "no" given in answer to "do you know your X".
          This is a normal, complete answer - not a failure to extract.
          IMPORTANT: this still applies even if the SAME message also
          contains something else - a question, an off-topic remark, a
          different vital, etc. (e.g. "no and what was my temp" is
          doesnt_know for the vital being asked about - the "no" is not
          ambiguous just because something else follows it in the same
          message). Do not downgrade a clear "no" to "unclear" just
          because the message has multiple parts - only use "unclear" for
          genuine ambiguity in the VALUE itself, like the blood-pressure
          case above.
        "declined" - they explicitly refuse to share a value they might
          actually know (e.g. "I'd rather not say").
        "not_mentioned" - this vital wasn't addressed in this message at all.
    - "value_text": the raw value they gave, whenever status is "known" OR
      "unclear" - never lose an ambiguous value, it still needs to be shown
      back to the patient for confirmation. Null for doesnt_know/declined/
      not_mentioned.
    - "numeric_value": the value as an actual number (not a string), when
      status is "known" or "unclear" and a number is present. For
      blood_pressure, this is the SYSTOLIC (top) number. Null if no number
      is present (e.g. status is doesnt_know/declined/not_mentioned).
    - "secondary_numeric_value": ONLY for blood_pressure - the DIASTOLIC
      (bottom) number, if given (e.g. "120/80" -> numeric_value=120,
      secondary_numeric_value=80). Null for every other vital, and null
      for blood_pressure if only one number was given.
    - "unit": the unit as stated, normalized to one of "F", "C" (for
      temperature), "kg", "lb" (for weight), "cm", "in" (for height), if
      the patient stated or implied one. Null if not stated - callers will
      assume a sensible default (F / kg / cm) when unit is null.
    - "clarifying_question": ONLY when status is "unclear" - write a short,
      specific question that would resolve the ambiguity, referencing what
      they actually said (e.g. "You mentioned 100 for your blood pressure -
      is that the full reading, or just the top number?"). Null otherwise.
  Never silently guess or auto-complete an ambiguous reading as if it were
  a full, confirmed value - and never silently drop it either.
- "off_topic_remark": a short neutral description (under 12 words) of
  anything in the message that ISN'T the patient stating a value/status
  for one of these fields. This includes:
    - small talk, a question about you/the bot, a random tangent
    - CRITICALLY: the patient ASKING what one of these vitals' values IS
      (e.g. "what is my sugar level", "what was my temperature again?",
      "did you note my weight?") - this is a QUESTION about their own
      data, not an ANSWER supplying a value, even though it names one of
      the tracked fields. Flag it as off-topic (e.g. "asked what their
      sugar level is") so it gets answered, rather than leaving that
      field's status as not_mentioned and silently dropping the question.
  Otherwise null. A plain "I don't know" / "no" is a normal non-answer
  for a vital being asked about - NOT off-topic.
  CRITICAL: off_topic_remark is ADDITIONAL, never a replacement - extract
  every vital field normally from the on-topic part of the message FIRST
  (e.g. a stated value, a "no", a bare "yes"/status), and only then
  separately note any aside here. Never let an aside cause you to
  under-extract or drop a clear answer to one of the fields above.
"""


def _vitals_field_fallback() -> dict:
    """Shared fallback shape for one numeric vitals field - kept as a function since it's now used in two places (batch + resolver) and has several fields."""
    return {
        "status": "not_mentioned", "value_text": None, "numeric_value": None,
        "secondary_numeric_value": None, "unit": None, "clarifying_question": None,
    }


def extract_vitals_batch(user_text: str, recent_context: list[str] | None = None, pending_field: str | None = None) -> dict:
    """
    One combined call covering diabetes + every numeric vital + an
    off-topic flag, run on EVERY turn of vitals collection - not scoped
    to "the field we just asked about". This is what lets the patient
    answer out of order, or give several values in one message, without
    the bot losing track or re-asking something already given.

    recent_context (optional): the last few conversation turns. Without
    this, a short reply like "yeah", "yesss", or "I just said yes" is
    genuinely ambiguous in isolation - the model has no way to know
    WHICH vital (if any) it's answering, so it can correctly end up not
    attributing it to anything. This was the actual cause of diabetes
    never resolving even after being answered clearly multiple times:
    each call only ever saw the bare reply, never the question it was
    replying to. Always pass this - it costs nothing extra structurally,
    just more input tokens on the same call.

    pending_field (optional): the ONE field the bot's last message
    actually asked about (state_machine.py knows this exactly - it's
    whatever _vitals_needed()[0] was before this turn). A short reply
    like a bare "no" is still meant to be inferred against ALL fields in
    principle, but making the model re-derive "what was just asked" from
    raw history text alone is fragile and was the actual cause of a
    plain "no"/"I don't know" occasionally not resolving anything at
    all. Passing this explicitly doesn't remove the multi-field,
    out-of-order behavior above the fallback: the model can still assign
    the reply to a DIFFERENT field if the patient's words clearly
    support that instead. It just gives it a reliable default target
    when the reply is too short to carry its own field reference.
    """
    prompt = _VITALS_BATCH_PROMPT
    if pending_field:
        prompt += (
            f"\n\nThe question the bot's last message actually asked about is: {pending_field.replace('_', ' ')}. "
            "If the patient's reply is short and doesn't clearly reference a different field (e.g. a bare "
            "'no', 'I don't know', 'idk', 'yes'/'yeah'/'yup' with no number, or a single unlabeled number), "
            "treat it as answering THIS field, not as not_mentioned for everything - including using status "
            "'will_provide' for a bare confirmation like 'yes' that doesn't include the actual number. Still "
            "assign it to a different field instead if the patient's actual words clearly point there."
        )
    if recent_context:
        prompt += (
            "\n\nRECENT CONVERSATION (use this ONLY to work out what the "
            "patient's reply is actually answering - e.g. a bare 'yeah' or "
            "'I already told you' only makes sense in light of what was "
            "just asked; do NOT invent a vital value that isn't actually "
            "supported by the patient's own words):\n" + "\n".join(recent_context)
        )
    fallback = {
        "diabetes": {"stated": False, "has_diabetes": None, "declined": False},
        "sugar_level": _vitals_field_fallback(),
        "blood_pressure": _vitals_field_fallback(),
        "weight": _vitals_field_fallback(),
        "height": _vitals_field_fallback(),
        "temperature": _vitals_field_fallback(),
        "off_topic_remark": None,
    }
    return _call_structured(user_text, prompt, _VITALS_BATCH_SCHEMA, fallback)


_RESOLVE_CLARIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "outcome": {"type": "string", "enum": ["value", "doesnt_know", "declined", "still_unclear"]},
        "value_text": {"type": "string", "nullable": True},
        "numeric_value": {"type": "number", "nullable": True},
        "secondary_numeric_value": {"type": "number", "nullable": True},
        "unit": {"type": "string", "nullable": True},
        "off_topic_remark": {"type": "string", "nullable": True},
    },
    "required": ["outcome", "value_text", "numeric_value", "secondary_numeric_value", "unit", "off_topic_remark"],
}


def resolve_vitals_clarification(field: str, original_value: str, clarifying_question: str, user_text: str) -> dict:
    """
    The clarifying question asked here is open-ended (e.g. "is that the
    full reading, or just the top number?") - NOT yes/no - so the reply
    needs to be interpreted freely, not forced through a yes/no filter.

    Critically, the reply to a clarifying question ISN'T always a value -
    the patient might reveal here that they actually don't know/can't
    share it at all (e.g. asked "is that your full reading or just the
    top number?", they might reply "actually I don't know my blood
    pressure at all"). Forcing every reply through "did we get a value
    or not" caused an infinite loop in exactly this situation, so
    "outcome" is a proper three-way (really four-way) split instead of
    a resolved/not-resolved boolean:

    - "value": they gave/confirmed an actual value -> use value_text etc.
    - "doesnt_know": they clarified they don't actually know/have it.
    - "declined": they clarified they don't want to share it.
    - "still_unclear": their reply doesn't settle anything yet - ask again.
    """
    prompt = f"""You're helping resolve an ambiguous medical vital reading for a hospital intake bot.

The patient was asked this clarifying question about their "{field.replace('_', ' ')}":
"{clarifying_question}"
(Their original, ambiguous value was: "{original_value}")

Given their reply, determine the outcome:
- "outcome": one of:
    "value" - their reply gives enough info to settle on a final value -
      confirming the original value as-is, explaining what part of it
      means (e.g. "just the top number"), or giving a full corrected
      value instead (e.g. "120/80").
    "doesnt_know" - their reply reveals they actually don't know/haven't
      measured this vital at all (e.g. "actually I don't know", "never
      checked it"). This is a normal, complete answer.
    "declined" - their reply reveals they don't want to share it.
    "still_unclear" - their reply doesn't address the question at all
      (changes the subject, still vague, doesn't resolve the ambiguity).
- "value_text": the final value to store, ONLY when outcome is "value" -
  incorporating their clarification in plain words (e.g. original "100"
  + reply "top" -> "100 (top/systolic number only, diastolic not
  given)"). Null for every other outcome.
- "numeric_value": the final value as an actual number (not a string),
  matching value_text, ONLY when outcome is "value". For blood pressure,
  this is the SYSTOLIC (top) number. Null otherwise.
- "secondary_numeric_value": ONLY for blood_pressure when outcome is
  "value" - the DIASTOLIC (bottom) number, if now known. Null otherwise.
- "unit": the unit, normalized to "F"/"C"/"kg"/"lb"/"cm"/"in", if
  relevant and stated. Null otherwise.
{_OFF_TOPIC_INSTRUCTION}"""
    fallback = {
        "outcome": "still_unclear", "value_text": None, "numeric_value": None,
        "secondary_numeric_value": None, "unit": None, "off_topic_remark": None,
    }
    return _call_structured(user_text, prompt, _RESOLVE_CLARIFICATION_SCHEMA, fallback)


# ----------------------------------------------------------------------
# Wrapper 3b: correction-vs-pending-answer disambiguation (used in
# VITALS_COLLECTION when a patient states a new value for an already-
# resolved vital, in the same turn the bot is still waiting on a
# DIFFERENT (pending) vital). This is genuinely ambiguous - e.g. weight
# was already recorded as 50kg, the bot then asks about temperature, and
# the patient replies "90kg" - is that a real correction to weight, or
# did they mean to answer the temperature question and just wrote the
# wrong unit? Guessing either way risks silently corrupting a vital, so
# state_machine.py asks the patient to clarify instead of assuming.
# ----------------------------------------------------------------------

_CORRECTION_AMBIGUITY_SCHEMA = {
    "type": "object",
    "properties": {
        "outcome": {
            "type": "string",
            "enum": ["correction_confirmed", "was_pending_field", "both", "still_unclear"],
        },
        "correction_value_text": {"type": "string", "nullable": True},
        "correction_numeric_value": {"type": "number", "nullable": True},
        "correction_unit": {"type": "string", "nullable": True},
        "pending_status": {
            "type": "string",
            "enum": ["known", "doesnt_know", "declined", "not_mentioned"],
        },
        "pending_value_text": {"type": "string", "nullable": True},
        "pending_numeric_value": {"type": "number", "nullable": True},
        "pending_unit": {"type": "string", "nullable": True},
        "off_topic_remark": {"type": "string", "nullable": True},
    },
    "required": [
        "outcome", "correction_value_text", "correction_numeric_value", "correction_unit",
        "pending_status", "pending_value_text", "pending_numeric_value", "pending_unit",
        "off_topic_remark",
    ],
}


def resolve_correction_ambiguity(
    field: str, pending_field: str, value_text: str, question: str,
    user_text: str, recent_context: list[str] | None = None,
) -> dict:
    """
    field: the already-known vital the patient's new value COULD be
    correcting (e.g. "weight").
    pending_field: the vital the bot had actually just asked about and
    still hasn't gotten an answer for (e.g. "temperature").
    value_text: the ambiguous value as originally given (e.g. "90kg").
    question: the exact clarifying question the bot already asked.
    """
    field_label, pending_label = field.replace("_", " "), pending_field.replace("_", " ")
    prompt = f"""You are resolving an ambiguity during a hospital intake bot's vitals collection.

The patient had already given a {field_label} earlier. The bot had just asked about their
{pending_label}, which is still unanswered. Then the patient's message contained "{value_text}",
which is ambiguous: it could be read as an UPDATED {field_label} (a genuine correction), OR the
patient may have actually meant to answer the {pending_label} question with that number (e.g. they
mixed up units or misspoke).

The bot asked the patient to clarify: "{question}"
Now interpret the patient's reply below.

Rules:
- "outcome":
    "correction_confirmed" - the patient confirms the number really was an update to their
      {field_label}. The {pending_label} question is STILL unanswered - don't invent a value for it.
    "was_pending_field" - the patient clarifies the number was actually meant as their answer to the
      {pending_label} question, NOT a correction to {field_label} at all - the original {field_label}
      value should be left unchanged. IMPORTANT: a bare, unambiguous naming of ONE of the two fields
      (e.g. just replying "{pending_label}" alone) IS a confident resolution to whichever field they
      named - do not treat it as "still_unclear" just because no new number came with it. Naming the
      field IS the answer; the number is already on record from the message that triggered this
      clarification in the first place.
    "both" - the patient confirms the {field_label} update AND ALSO now separately gives a value for
      {pending_label} in the same reply.
    "still_unclear" - the reply doesn't name either field and doesn't restate/confirm a number for
      either - genuinely ambiguous, not just terse.
- "correction_value_text" / "correction_numeric_value" / "correction_unit": the confirmed value for
  {field_label}, only when outcome is "correction_confirmed" or "both". Use "{value_text}" (parsed
  into a number/unit) unless the patient states a different value now. Null otherwise.
- "pending_status": "known" if the patient confirms this reading belongs to {pending_label} - this
  includes a bare confirmation of the FIELD alone (e.g. just replying "{pending_label}" or "yes, that
  one") with no new number restated. In that case still use "known" - the number "{value_text}" was
  already given moments ago as part of the very ambiguity being resolved, so confirming which field it
  belongs to is a complete answer, NOT a request for yet another number. Use "doesnt_know" or "declined"
  if they indicate that instead, otherwise "not_mentioned" (only when the reply doesn't address
  {pending_label} at all).
- "pending_value_text" / "pending_numeric_value" / "pending_unit": the value for {pending_label},
  whenever pending_status is "known". If the patient restated a different number now, use that instead.
  Otherwise, since no new number was given, use "{value_text}" itself (parsed into a number/unit) - do
  NOT leave these null just because the patient didn't repeat the number. Null only when pending_status
  isn't "known".
{_OFF_TOPIC_INSTRUCTION}"""
    if recent_context:
        prompt += "\n\nRECENT CONVERSATION (context only):\n" + "\n".join(recent_context)
    fallback = {
        "outcome": "still_unclear", "correction_value_text": None, "correction_numeric_value": None,
        "correction_unit": None, "pending_status": "not_mentioned", "pending_value_text": None,
        "pending_numeric_value": None, "pending_unit": None, "off_topic_remark": None,
    }
    return _call_structured(user_text, prompt, _CORRECTION_AMBIGUITY_SCHEMA, fallback)


# ----------------------------------------------------------------------
# Wrapper 3c: booking-review action extraction (used in BOOKING_REVIEW,
# after every collected field is shown back to the patient in a table
# but BEFORE the appointment is actually booked)
# ----------------------------------------------------------------------

_REVIEW_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["confirm", "edit", "unclear"]},
        "edit_field": {
            "type": "string",
            "enum": [
                "name", "age", "gender", "diabetes", "sugar_level", "blood_pressure",
                "weight", "height", "temperature", "symptoms", "history",
            ],
            "nullable": True,
        },
        "edit_value_text": {"type": "string", "nullable": True},
        "off_topic_remark": {"type": "string", "nullable": True},
    },
    "required": ["action", "edit_field", "edit_value_text", "off_topic_remark"],
}

_REVIEW_ACTION_PROMPT = f"""You are reviewing a hospital intake summary with a patient before their
appointment is actually booked. The bot just showed them a table of everything collected so far and
asked whether it's correct or needs any changes.

Rules:
- "action":
    "confirm" - the patient says everything looks right / to go ahead and book (e.g. "yes", "looks
      good", "book it", "that's all correct", "confirm").
    "edit" - the patient wants to correct or change one specific piece of information (e.g. "change my
      weight to 80kg", "actually my name is Priya", "my temperature is wrong, it's 101", "I don't
      actually have diabetes").
    "unclear" - anything else, including a message that mentions wanting to change something without
      making clear WHICH single field, or WHAT the new value is.
- "edit_field": which ONE field the patient is correcting, from: name, age, gender, diabetes,
  sugar_level, blood_pressure, weight, height, temperature, symptoms, history. If more than one field
  is referenced, or it isn't clear which single field they mean, set this to null and use action
  "unclear" instead (so the bot asks them to specify one at a time) - never guess.
- "edit_value_text": the new value/correction the patient stated, in their own words, verbatim-ish.
  Only set when action is "edit". Null otherwise.
{_OFF_TOPIC_INSTRUCTION}"""


def extract_review_action(user_text: str, recent_context: list[str] | None = None) -> dict:
    """
    Used in BOOKING_REVIEW: the patient is looking at a full summary table
    and deciding whether to confirm as-is or correct something before the
    appointment is actually booked. recent_context helps interpret a short
    reply ("yes", "that one") against what was actually just shown.
    """
    prompt = _REVIEW_ACTION_PROMPT
    if recent_context:
        prompt += "\n\nRECENT CONVERSATION (context only):\n" + "\n".join(recent_context)
    fallback = {"action": "unclear", "edit_field": None, "edit_value_text": None, "off_topic_remark": None}
    return _call_structured(user_text, prompt, _REVIEW_ACTION_SCHEMA, fallback)


# ----------------------------------------------------------------------
# Wrapper 4: age extraction (used in AGE_COLLECTION). Kept as its own
# dedicated prompt (rather than folded into extract_vitals_batch) because
# it needs explicit current-year grounding for birth-year arithmetic, and
# it only ever runs once, late in the flow (during booking) - not worth
# bundling into the vitals batch call.
# ----------------------------------------------------------------------

_AGE_SCHEMA = {
    "type": "object",
    "properties": {
        "value_known": {"type": "boolean"},
        "value_text": {"type": "string", "nullable": True},
        "numeric_value": {"type": "number", "nullable": True},
        "declined": {"type": "boolean"},
        "off_topic_remark": {"type": "string", "nullable": True},
    },
    "required": ["value_known", "value_text", "numeric_value", "declined", "off_topic_remark"],
}


def extract_age(user_text: str) -> dict:
    """
    Age gets its own dedicated prompt because it needs explicit
    current-year grounding: the model isn't reliably aware of "today's
    date" on its own, so if a patient gives a birth year instead of a
    direct age (e.g. "I was born in 2001"), it can get the arithmetic
    wrong without being told what year it currently is.
    """
    from datetime import datetime
    current_year = datetime.now().year

    prompt = f"""You extract a person's age from their message.

Rules:
- "value_known": true if they gave an age directly, OR gave a birth
  year/date from which age can be calculated. False if they don't know
  or explicitly refuse (see "declined" below).
- "value_text": the age in years as a short string (e.g. "34 years").
- "numeric_value": the age as an actual number (not a string), matching
  value_text. Null if value_known is false.
  IMPORTANT: today's year is {current_year}. If the patient gives a
  birth year instead of a direct age (e.g. "I was born in 2001", "born
  in '98"), calculate their age as ({current_year} - birth year).
  Double-check this subtraction carefully - it must be exactly correct.
- "declined": true if they explicitly refuse to answer, even if they
  might know their own age.
{_OFF_TOPIC_INSTRUCTION}"""
    fallback = {"value_known": False, "value_text": None, "numeric_value": None, "declined": False, "off_topic_remark": None}
    return _call_structured(user_text, prompt, _AGE_SCHEMA, fallback)


# ----------------------------------------------------------------------
# Wrapper 5: medical history extraction (used in HISTORY_COLLECTION)
# ----------------------------------------------------------------------

_HISTORY_PROMPT = f"""You extract structured medical history from a patient's message.

Rules:
- "conditions": list any past or current medical conditions mentioned
  (e.g. "asthma", "diabetes", "high blood pressure"). Lowercase, simple terms.
- "medications": list any medications mentioned. Lowercase, simple terms.
- "allergies": list any allergies mentioned. Lowercase, simple terms.
- If the patient says they have none of something (e.g. "no allergies"),
  return an empty list for that field - don't guess or invent entries.
- If a message doesn't mention a category at all, return an empty list
  for it rather than leaving it out.
{_OFF_TOPIC_INSTRUCTION}"""

_HISTORY_SCHEMA = {
    "type": "object",
    "properties": {
        "conditions": {"type": "array", "items": {"type": "string"}},
        "medications": {"type": "array", "items": {"type": "string"}},
        "allergies": {"type": "array", "items": {"type": "string"}},
        "off_topic_remark": {"type": "string", "nullable": True},
    },
    "required": ["conditions", "medications", "allergies", "off_topic_remark"],
}


def extract_history(user_text: str) -> dict:
    """Returns e.g. {"conditions": ["asthma"], "medications": ["inhaler"], "allergies": [], "off_topic_remark": None}"""
    fallback = {"conditions": [], "medications": [], "allergies": [], "off_topic_remark": None}
    return _call_structured(user_text, _HISTORY_PROMPT, _HISTORY_SCHEMA, fallback)


# ----------------------------------------------------------------------
# Wrapper 6: name extraction (used in NAME_COLLECTION)
# ----------------------------------------------------------------------

_NAME_PROMPT = f"""You extract a person's name from their reply to "what's your name?"

Rules:
- "name_given": true if they stated an actual name, in any phrasing
  (e.g. "it's Sarah", "I'm Raj Patel", "you can call me Alex").
- "name": the extracted name only, properly capitalized, with no
  surrounding words (e.g. "Sarah", "Raj Patel"), otherwise null.
- "declined": true if they explicitly don't want to give their name
  (e.g. "I'd rather not say", "skip", "prefer not to answer").
  This is a valid, complete answer - not a failure to extract.
- If neither a name nor a decline is clearly present, set both
  "name_given" and "declined" to false.
{_OFF_TOPIC_INSTRUCTION}"""

_NAME_SCHEMA = {
    "type": "object",
    "properties": {
        "name_given": {"type": "boolean"},
        "name": {"type": "string", "nullable": True},
        "declined": {"type": "boolean"},
        "off_topic_remark": {"type": "string", "nullable": True},
    },
    "required": ["name_given", "name", "declined", "off_topic_remark"],
}


def extract_name(user_text: str) -> dict:
    """Returns e.g. {"name_given": True, "name": "Sarah", "declined": False, "off_topic_remark": None}"""
    fallback = {"name_given": False, "name": None, "declined": False, "off_topic_remark": None}
    return _call_structured(user_text, _NAME_PROMPT, _NAME_SCHEMA, fallback)


# ----------------------------------------------------------------------
# Wrapper 7: gender extraction (used in GENDER_COLLECTION)
# ----------------------------------------------------------------------

_GENDER_PROMPT = f"""You extract a person's gender from their reply to "what's your gender?"

Rules:
- "gender_given": true if they stated a gender in any phrasing
  (e.g. "male", "I'm a woman", "non-binary", "F", "M", "guy", "lady").
- "gender": NORMALIZE the answer to one of these standard lowercase
  terms: "male", "female", "non-binary", or "other". For example,
  "m", "man", "guy", "boy" -> "male"; "f", "woman", "lady", "girl" ->
  "female". If they give a term that doesn't fit these categories,
  use "other". Never store the patient's raw abbreviation or slang
  directly - always map it to one of these four terms.
- "declined": true if they explicitly don't want to share
  (e.g. "prefer not to say", "skip", "rather not answer").
  This is a valid, complete answer - not a failure to extract.
- If neither a clear answer nor a decline is present, set both
  "gender_given" and "declined" to false.
{_OFF_TOPIC_INSTRUCTION}"""

_GENDER_SCHEMA = {
    "type": "object",
    "properties": {
        "gender_given": {"type": "boolean"},
        "gender": {"type": "string", "enum": ["male", "female", "non-binary", "other"], "nullable": True},
        "declined": {"type": "boolean"},
        "off_topic_remark": {"type": "string", "nullable": True},
    },
    "required": ["gender_given", "gender", "declined", "off_topic_remark"],
}


def extract_gender(user_text: str) -> dict:
    """Returns e.g. {"gender_given": True, "gender": "female", "declined": False, "off_topic_remark": None}"""
    fallback = {"gender_given": False, "gender": None, "declined": False, "off_topic_remark": None}
    return _call_structured(user_text, _GENDER_PROMPT, _GENDER_SCHEMA, fallback)


# ----------------------------------------------------------------------
# Wrapper 8: next-topic proposal (used by state_machine.py after the
# symptom round and between topics). The LLM only SUGGESTS; state_machine.py
# validates the answer against the allowed list and falls back to the
# default order if this returns None.
# ----------------------------------------------------------------------

_NEXT_STATE_PROMPT = """You decide which intake topic should come next in a hospital intake
conversation, based on what the patient has actually said.

Rules:
- Pick "next_state" from the ALLOWED values listed below. Never invent a value.
- Prefer the topic that fits the patient's own words best right now (for example, if the
  patient already mentioned a past illness, a history topic may come sooner).
- "reason": one short plain sentence explaining the choice.
- Return JSON only."""

_NEXT_STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "next_state": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["next_state", "reason"],
}


def propose_next_state(recent_history: list[str], status: dict, allowed: list[str]) -> dict | None:
    """
    Returns {"next_state": "<one of allowed>", "reason": "..."} or None.
    None means the caller should use its default order.
    """
    if not allowed:
        return None
    prompt = (
        _NEXT_STATE_PROMPT
        + f"\n\nALLOWED next_state values: {', '.join(allowed)}"
        + f"\n\nSTATUS: {json.dumps(status)}"
        + "\n\nRECENT CONVERSATION:\n"
        + "\n".join(recent_history or [])
    )
    fallback = {"next_state": None, "reason": "fallback"}
    result = _call_structured("Propose the next state now.", prompt, _NEXT_STATE_SCHEMA, fallback)
    if not isinstance(result, dict) or result.get("next_state") not in allowed:
        return None
    return result

"""
state_machine.py

The conversation "brain". Text in, text out.

Every patient message, in every state:
  1. Red flags -> EMERGENCY (saved, sticky until "new patient")
  2. "new patient" -> saves the current patient, starts a fresh record
  3. Emergency lock
  4. "back" / "skip" navigation
  5. ONE fact extraction (fact_extractor.py). Every fact in the message is saved
     to the patient's record (intake_store.py), whatever step the bot is on.
     Privacy: a question about code or other people is refused. A question about
     the patient's OWN data is answered from that patient's record only.
  6. Consent reversal (only after symptom questions were declined)
  7. Symptom priority: if the message contains symptoms, they are saved and the
     next question is about symptoms, even during name, age, gender, or booking.
     Other facts in the same message are acknowledged ("I've noted your ...").
  8. Exit intent -> CLOSING with a summary
  9. The current step's handler

Consent:
  - First refusal: explain why the questions help, then ask once more.
  - Second refusal: symptom questions are skipped. Ask whether to book.
      yes -> name, age, gender, review, booking
      no  -> closing summary
  - Reversal at any step ("I consent", "ok, I'll get there after I finish") reopens
    symptoms now, or after the current step.

Records:
  - Facts live in intake.db. Each appointment is one patients row.

The LLM never changes state directly. Only book_appointment() returning a
reference number counts as a booking.
"""

import re
import uuid
import json
import os
import tempfile
from datetime import datetime
from enum import Enum, auto

from red_flags import check_red_flags, emergency_message
from symptom_kb import SYMPTOM_TABLE, lookup_condition, no_match_response
from appointments_db import init_db, book_appointment
from intake_store import (
    init_store, create_patient, set_patient_outcome, add_fact, retire_fact,
    active_facts, has_facts, open_episodes, get_or_open_episode,
)
from fact_extractor import extract_facts

from llm_client import (
    compose_reply,
    answer_offtopic_remark,
    extract_yes_no,
    extract_vitals_batch,
    resolve_vitals_clarification,
    triage_reasoning,
    extract_review_action,
    extract_closing_intent,
)

# If propose_next_state is missing, the default order is used and the app still runs.
try:
    from llm_client import propose_next_state
except ImportError:
    propose_next_state = None


class State(Enum):
    GREETING = auto()
    CONSENT = auto()
    SYMPTOM_COLLECTION = auto()
    VITALS_COLLECTION = auto()
    HISTORY_COLLECTION = auto()
    APPOINTMENT_OFFER = auto()
    NAME_COLLECTION = auto()
    AGE_COLLECTION = auto()
    GENDER_COLLECTION = auto()
    BOOKING_REVIEW = auto()
    ANYTHING_ELSE = auto()
    CLOSING = auto()
    EMERGENCY = auto()


# Steps that a symptom detour can interrupt, and that resume afterwards.
RESUMABLE_STATES = {
    State.VITALS_COLLECTION,
    State.HISTORY_COLLECTION,
    State.APPOINTMENT_OFFER,
    State.NAME_COLLECTION,
    State.AGE_COLLECTION,
    State.GENDER_COLLECTION,
    State.BOOKING_REVIEW,
    State.ANYTHING_ELSE,
}

# Topics the flow can visit after symptoms, in default order.
NAV_ORDER = [
    State.VITALS_COLLECTION,
    State.HISTORY_COLLECTION,
    State.APPOINTMENT_OFFER,
]

VITAL_ORDER = ["diabetes", "sugar_level", "blood_pressure", "weight", "height", "temperature"]

VITAL_LABELS = {
    "diabetes": "diabetes status",
    "sugar_level": "sugar level",
    "blood_pressure": "blood pressure",
    "weight": "weight",
    "height": "height",
    "temperature": "temperature",
}

VITAL_QUESTIONS = {
    "diabetes": "Do you have diabetes?",
    "sugar_level": "Do you know your current blood sugar level? If so, what is it?",
    "blood_pressure": "Do you know your latest blood pressure reading? If so, what is it?",
    "weight": "What is your weight?",
    "height": "What is your height?",
    "temperature": "Do you know your current temperature? If so, what is it?",
}

MAX_SYMPTOM_FOLLOWUPS = 4
MAX_REPEATS = 3

CONSENT_FALLBACK = (
    "Hi, I'm here to help you check your symptoms and book a hospital appointment if needed. "
    "I'm not a doctor, so this isn't a diagnosis - just a starting point. Is that okay with you?"
)

OTHER_DATA_REFUSAL = (
    "I'm sorry, I can only share the information you've given me in this conversation. "
    "I can't show system details, code, or other people's records."
)

EXIT_PATTERN = re.compile(
    r"\b("
    r"bye|goodbye|good bye|"
    r"that'?s all|that is all|"
    r"i'?m done|im done|i am done|"
    r"end (the )?(conversation|chat)|end it|"
    r"stop (the )?(conversation|chat)|"
    r"quit|exit|"
    r"i want to (stop|leave)"
    r")\b"
)

# Used only when the fact extractor failed, so consent is never lost to an API error.
CONSENT_PHRASE_PATTERN = re.compile(
    r"\b(i (give|gave|consent)|you can ask|go ahead and ask|i changed my mind|change my mind)\b",
    re.IGNORECASE,
)

# Words that mean "not now, after the current step".
DEFER_PATTERN = re.compile(r"\b(after|later|once|when i'?m done|when i finish|finish|first)\b", re.IGNORECASE)

# Whole-message commands. Exact matches only, so "my back hurts" is never a command.
BACK_PATTERN = re.compile(
    r"^\s*(go back|back|previous|undo|change (my )?(last )?answer)\W*$", re.IGNORECASE
)
SKIP_PATTERN = re.compile(r"^\s*(skip|skip (this|it)|next|move on)\W*$", re.IGNORECASE)
NEW_PATIENT_PATTERN = re.compile(r"^\s*(start (a )?)?(new|next) patient\W*$", re.IGNORECASE)

# Short answers handled in Python, including common typos.
QUICK_NO = re.compile(
    r"^\s*(no|nope|nop|nah|naa|na|n|no thanks|no thank you|not really|no way)\W*$",
    re.IGNORECASE,
)
QUICK_YES = re.compile(
    r"^\s*(yes|yeah|yea|yeh|yep|yup|yah|ya|yaa|sure|ok|okay|k|y|s)\W*$",
    re.IGNORECASE,
)

# Local backup patterns, used only when the fact extractor fails.
NAME_FALLBACK = re.compile(r"(?:my name is|name is)\s+([A-Za-z][A-Za-z'\-]+)", re.IGNORECASE)
AGE_FALLBACK = re.compile(
    r"\b(\d{1,3})\s*(?:years?|yrs?)\s*old\b|\bage\s*(?:is|:)?\s*(\d{1,3})\b", re.IGNORECASE
)

EMPTY_EXTRACTION = {
    "facts": [],
    "data_question": "none",
    "symptom_consent": "none",
    "wants_to_stop": False,
    "off_topic_remark": None,
    "_failed": False,
}


def is_exit_intent(user_text: str) -> bool:
    return EXIT_PATTERN.search(user_text.lower()) is not None


def quick_yes_no(user_text: str) -> str | None:
    """Handles obvious one-word answers in Python. Returns 'yes', 'no', or None."""
    if QUICK_NO.match(user_text):
        return "no"
    if QUICK_YES.match(user_text):
        return "yes"
    return None


# Symptom words from the knowledge base, used as a backup when extraction fails.
_LOCAL_SYMPTOM_TERMS = sorted(
    {term.lower() for row in SYMPTOM_TABLE for term in row["symptoms"]},
    key=len,
    reverse=True,
)


def local_symptom_match(user_text: str) -> list:
    text = user_text.lower()
    found = []
    for term in _LOCAL_SYMPTOM_TERMS:
        if re.search(rf"\b{re.escape(term)}s?\b", text):
            if not any(term in f or f in term for f in found):
                found.append(term)
    return found


def _local_facts(user_text: str) -> list:
    """Backup extraction (name, age, symptoms) used only when the LLM fails."""
    facts = []
    m = NAME_FALLBACK.search(user_text)
    if m:
        facts.append({"category": "identity", "subject": "name", "value": m.group(1),
                      "status": "active", "verbatim_quote": m.group(0), "episode": "none", "details": []})
    m = AGE_FALLBACK.search(user_text)
    if m:
        age = m.group(1) or m.group(2)
        facts.append({"category": "identity", "subject": "age", "value": age,
                      "status": "active", "verbatim_quote": m.group(0), "episode": "none", "details": []})
    for term in local_symptom_match(user_text):
        facts.append({"category": "symptom", "subject": term, "value": None,
                      "status": "active", "verbatim_quote": term, "episode": "new", "details": []})
    return facts


def _normalize_gender(value):
    if not value:
        return None
    s = value.strip().lower()
    if s in ("m", "male", "man", "boy", "guy"):
        return "male"
    if s in ("f", "female", "woman", "girl", "lady"):
        return "female"
    if "non" in s:
        return "non-binary"
    return "other"


def _details_dict(details) -> dict:
    """Accepts the list-of-{key,value} format from the LLM, or a plain dict."""
    if isinstance(details, dict):
        return {k: v for k, v in details.items() if v not in (None, "")}
    out = {}
    for item in details or []:
        if isinstance(item, dict) and item.get("key") and item.get("value") not in (None, ""):
            out[item["key"]] = item["value"]
    return out

BOOKINGS_FILE = "bookings.json"


def _append_booking_json(reference: str, details: dict) -> None:
    """Appends one booking copy to bookings.json. The database remains the source of truth."""
    bookings = []
    if os.path.exists(BOOKINGS_FILE):
        try:
            with open(BOOKINGS_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            bookings = loaded if isinstance(loaded, list) else []
        except (json.JSONDecodeError, OSError):
            os.replace(BOOKINGS_FILE, f"bookings_unreadable_{datetime.now():%Y%m%d%H%M%S}.json")

    bookings.append({
        "reference": reference,
        "booked_at": datetime.now().isoformat(timespec="seconds"),
        **details,
    })

    folder = os.path.dirname(os.path.abspath(BOOKINGS_FILE))
    fd, tmp_path = tempfile.mkstemp(dir=folder, suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(bookings, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, BOOKINGS_FILE)
    print(f"    [debug] booking copy saved: {reference} -> {BOOKINGS_FILE}")
    
class ConversationSession:
    def _start_identity(self, aside=None) -> str:
        """Asks only for identity fields that are not already on record."""
        for subject, state, ask in (
            ("name", State.NAME_COLLECTION, self._name_question),
            ("age", State.AGE_COLLECTION, self._age_question),
            ("gender", State.GENDER_COLLECTION, self._gender_question),
        ):
            value, status = self._identity(subject)
            if value is None and status != "declined":
                self._move(state)
                return ask(aside)
        self.repeat_count = 0
        return self._show_review(aside)
    def __init__(self):
        init_db()
        init_store()

        self.session_id = str(uuid.uuid4())
        self.patient_no = 1
        self.record_id = create_patient(self.session_id, self.patient_no)
        self.record_start = 0
        self.patient_started = False
        self.closed = False

        self.state = State.GREETING
        self.transcript = []
        self.visited_states = set()
        self.back_stack = []

        # Per-appointment state that is not stored as facts
        self.symptoms_declined = False
        self.symptom_followups = 0
        self.vitals = {f: None for f in VITAL_ORDER}
        self.vitals_order = []
        self.pending_clarification = None
        self.triage_result = None
        self.booking_reference = None

        # Consent reversal and deferral
        self.symptoms_deferred = False
        self.defer_from = None
        self.after_deferred = None
        self.consent_refusals = 0

        self.repeat_count = 0

        # Values for the current turn only
        self._turn_aside = None
        self._turn_wants_stop = False
        self._turn_consent = "none"
        self._turn_question = "none"
        self._turn_granted = False

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def handle_input(self, user_text: str) -> str:
        reply = self._route(user_text)
        if user_text:
            self.transcript.append(("patient", user_text))
        self.transcript.append(("bot", reply))
        return reply

    def _reset_turn(self) -> None:
        self._turn_aside = None
        self._turn_wants_stop = False
        self._turn_consent = "none"
        self._turn_question = "none"
        self._turn_granted = False

    def _route(self, user_text: str) -> str:
        self._reset_turn()

        if check_red_flags(user_text):
            self.closed = True
            self.state = State.EMERGENCY
            set_patient_outcome(self.record_id, "emergency")
            return emergency_message()

        if NEW_PATIENT_PATTERN.match(user_text):
            self._start_new_record(previous_outcome="abandoned")
            return self._consent_prompt()

        if self.state == State.EMERGENCY:
            return emergency_message()

        if self.state not in (State.CLOSING, State.GREETING):
            if BACK_PATTERN.match(user_text):
                return self._go_back()
            if SKIP_PATTERN.match(user_text):
                return self._skip_current()

        quick = quick_yes_no(user_text) if user_text.strip() else None
        if quick == "no" and self.state == State.SYMPTOM_COLLECTION:
            self._turn_wants_stop = True

        extraction = self._extract(user_text, quick)

        # Privacy: refuse requests for code, system details, or other people's data.
        if extraction.get("data_question") == "other_or_code":
            return OTHER_DATA_REFUSAL + " " + self._ask_current()

        verdict = extraction.get("symptom_consent", "none")
        if extraction.get("_failed") and self.symptoms_declined and CONSENT_PHRASE_PATTERN.search(user_text):
            verdict = "agrees_now"
        self._turn_consent = verdict
        self._turn_question = extraction.get("data_question", "none")
        self._turn_aside = extraction.get("off_topic_remark")
        if extraction.get("wants_to_stop") and self.state == State.SYMPTOM_COLLECTION:
            self._turn_wants_stop = True
        self._turn_granted = (
            self.symptoms_declined
            and verdict in ("agrees_now", "agrees_later")
            and self.state in RESUMABLE_STATES
        )

        # Save every fact in the message, whatever the current step is.
        saved = self._absorb(extraction, user_text)

        if self._turn_granted:
            return self._handle_consent_grant(deferred=(verdict == "agrees_later"))

        if self.state != State.CLOSING and is_exit_intent(user_text):
            return self._close_with_summary()

        # Symptom priority: symptoms mentioned at any resumable step start a symptom
        # round now. The step is saved and resumes once symptoms are finished.
        symptom_saved = [s for s in saved if s["category"] == "symptom"]
        other_saved = [s for s in saved if s["category"] != "symptom"]
        if symptom_saved and self.state in RESUMABLE_STATES:
            note = self._noted_text(other_saved)
            self.symptoms_declined = False
            self.after_deferred = self.state
            self.repeat_count = 0
            self._move(State.SYMPTOM_COLLECTION)
            return note + self._symptom_question()

        prefix = self._noted_text(other_saved) if (other_saved and self.state == State.SYMPTOM_COLLECTION) else ""

        handler = getattr(self, f"_handle_{self.state.name.lower()}")
        reply = self._after_turn(handler(user_text))
        if prefix:
            reply = prefix + reply
        if self._turn_question == "own_data":
            reply = self._own_data_answer(user_text) + " " + reply
        return reply

    def _extract(self, user_text: str, quick):
        """One fact extraction per message. Plain yes/no answers skip it to save quota."""
        if not user_text.strip() or quick is not None:
            return dict(EMPTY_EXTRACTION)
        open_labels = [e["label"] for e in open_episodes(self.record_id)]
        result = extract_facts(
            user_text, self._recent(), open_labels, self.state.name, self.symptoms_declined
        )
        print(f"    [debug] facts: {result}")
        return result

    def _absorb(self, extraction, user_text: str) -> list:
        """Saves every fact in the extraction to this patient's record. Returns the new ones."""
        facts = extraction.get("facts", [])
        if extraction.get("_failed"):
            facts = _local_facts(user_text)

        saved = []
        for f in facts:
            category = (f.get("category") or "other").lower()
            subject = (f.get("subject") or "").strip().lower()
            if not subject:
                continue
            # While symptoms are declined, do not store symptoms unless the patient just agreed.
            if category == "symptom" and self.symptoms_declined and not self._turn_granted:
                continue

            value = f.get("value")
            status = f.get("status") or "active"
            if category == "identity":
                if subject == "gender":
                    value = _normalize_gender(value)
                elif subject == "age" and value:
                    value = re.sub(r"\D", "", value) or None
                if status == "declined":
                    value = None

            episode_id = None
            if category == "symptom":
                episode_id = self._episode_for(f.get("episode", "none"), subject)

            _, changed = add_fact(
                self.record_id, episode_id, category, subject, value,
                _details_dict(f.get("details")), status, f.get("verbatim_quote") or "",
            )
            if changed:
                saved.append({"category": category, "subject": subject, "value": value})
        return saved

    def _episode_for(self, hint: str, subject: str):
        open_eps = open_episodes(self.record_id)
        if hint == "new" or not open_eps:
            return get_or_open_episode(self.record_id, subject)
        return open_eps[-1]["id"]

    def _after_turn(self, reply: str) -> str:
        """Switches to the deferred symptom round once the current step has moved on."""
        if self.symptoms_deferred and self.state not in (self.defer_from, State.CLOSING, State.EMERGENCY):
            self.symptoms_deferred = False
            self.symptoms_declined = False
            self.consent_refusals = 0
            self.repeat_count = 0
            self.after_deferred = self.state
            self._move(State.SYMPTOM_COLLECTION)
            return "Thanks. Now, you mentioned going over your symptoms. " + self._symptom_question()
        return reply

    def _noted_text(self, saved: list) -> str:
        labels = []
        for s in saved:
            if s["category"] == "identity":
                labels.append(f"{s['subject']} ({s['value'] or 'declined'})")
            elif s["category"] == "vital":
                labels.append(f"{s['subject'].replace('_', ' ')} ({s['value']})")
            else:
                labels.append(f"{s['subject']} ({s['category'].replace('_', ' ')})")
        return ("I've noted your " + ", ".join(labels) + ". ") if labels else ""

    # ------------------------------------------------------------------
    # Reading this patient's record
    # ------------------------------------------------------------------

    def _facts(self, category: str) -> list:
        return [f for f in active_facts(self.record_id) if f["category"] == category]

    def _identity(self, subject: str):
        for f in self._facts("identity"):
            if f["subject"] == subject:
                return f["value"], f["status"]
        return None, None

    def _symptom_rows(self) -> list:
        return self._facts("symptom")

    def _history_summary(self):
        parts = []
        for cat in ("history", "medication", "allergy", "family_history"):
            items = [f"{f['subject']}{': ' + f['value'] if f['value'] else ''}" for f in self._facts(cat)]
            if items:
                parts.append(f"{cat.replace('_', ' ')}: " + ", ".join(items))
        return "; ".join(parts) or None

    def _own_snapshot(self):
        """This patient's own active facts, as text. Never includes anyone else's data."""
        lines = []
        for f in active_facts(self.record_id):
            cat = f["category"]
            if cat == "identity":
                lines.append(f"{f['subject']}: {f['value'] or 'declined to give'}")
            elif cat == "symptom":
                detail = ", ".join(f"{k} {v}" for k, v in f["details"].items())
                lines.append(f"symptom {f['subject']}" + (f" ({detail})" if detail else "") + f" [{f['status']}]")
            else:
                lines.append(f"{cat.replace('_', ' ')} {f['subject']}" + (f": {f['value']}" if f["value"] else ""))
        return "\n".join(lines) or None

    def _own_data_answer(self, user_text: str) -> str:
        snap = self._own_snapshot()
        if not snap:
            return "I don't have any information recorded about you yet."
        answer = answer_offtopic_remark(
            f'asked about their own recorded information: "{user_text}"', snap
        )
        return answer or ("So far I have recorded: " + snap.replace("\n", "; ") + ".")

    def _has_facts(self) -> bool:
        return has_facts(self.record_id) or any(self.vitals.values())

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _move(self, state: State) -> None:
        if self.state != state:
            self.back_stack.append(self.state)
        self.state = state
        self.visited_states.add(state)

    def _recent(self, n: int = 6) -> list:
        """Recent turns from the current patient only, so no other patient's text is sent."""
        lines = []
        for speaker, text in self.transcript[self.record_start:][-n:]:
            label = "Patient" if speaker == "patient" else "Bot"
            lines.append(f"{label}: {text}")
        return lines

    @staticmethod
    def _pick(options, index: int) -> str:
        if isinstance(options, str):
            return options
        return options[min(max(index, 0), len(options) - 1)]

    def _say(self, content, fallback, aside=None, locked=None, is_first=False) -> str:
        aside = aside or self._turn_aside
        self._turn_aside = None
        fallback = self._pick(fallback, self.repeat_count - 1 if self.repeat_count else 0)
        if aside:
            aside_text = answer_offtopic_remark(aside, self._own_snapshot())
            if aside_text:
                content = (
                    f"First, respond to the patient's aside with this one short sentence: "
                    f"{aside_text} Then continue: {content}"
                )
                fallback = f"{aside_text} {fallback}"
        reply = compose_reply(
            content,
            locked_facts=locked,
            is_first_message=is_first,
            recent_history=self._recent(),
            repeat_count=self.repeat_count,
        )
        if reply is None:
            print("    [debug] LLM composer unavailable - using fallback wording")
        return reply or fallback

    def _retry(self, content, fallback, aside=None) -> str:
        self.repeat_count += 1
        return self._say(content, fallback, aside)

    def _yes_no(self, user_text: str) -> str:
        quick = quick_yes_no(user_text)
        if quick:
            print(f"    [debug] quick yes/no: {quick}")
            return quick
        r = extract_yes_no(user_text, self._recent())
        print(f"    [debug] LLM yes/no: {r}")
        return r["intent"]

    def _needed_vitals(self) -> list:
        return [f for f in VITAL_ORDER if self.vitals[f] is None]

    def _resolved_count(self) -> int:
        return sum(1 for v in self.vitals.values() if v is not None)

    def _format_value(self, src: dict) -> str:
        value = src.get("value_text") or ""
        if not value and src.get("numeric_value") is not None:
            value = str(src["numeric_value"])
        unit = src.get("unit")
        if unit and unit.lower() not in value.lower():
            value = f"{value} {unit}".strip()
        return value or "given"

    # ------------------------------------------------------------------
    # Records
    # ------------------------------------------------------------------

    def _start_new_record(self, previous_outcome) -> None:
        """Starts a new appointment record. The previous record is saved first, if given."""
        if previous_outcome and not self.closed and (self.patient_started or self._has_facts()):
            set_patient_outcome(self.record_id, previous_outcome)
        self.patient_no += 1
        self.record_id = create_patient(self.session_id, self.patient_no)
        self.record_start = len(self.transcript)
        self.patient_started = False
        self.closed = False
        self.consent_refusals = 0
        self._clear_intake_state()

    def _clear_intake_state(self) -> None:
        self.symptoms_declined = False
        self.symptom_followups = 0
        self.vitals = {f: None for f in VITAL_ORDER}
        self.vitals_order = []
        self.pending_clarification = None
        self.triage_result = None
        self.booking_reference = None
        self.symptoms_deferred = False
        self.defer_from = None
        self.after_deferred = None
        self.visited_states = set()
        self.back_stack = []
        self.repeat_count = 0

    def _consent_prompt(self) -> str:
        self.state = State.CONSENT
        return self._say(
            "Introduce yourself as a hospital intake assistant. Say you help check symptoms and book "
            "an appointment if needed, that you are not a doctor, and that this is not a diagnosis. "
            "Ask whether that is okay.",
            CONSENT_FALLBACK,
        )

    # ------------------------------------------------------------------
    # Consent reversal
    # ------------------------------------------------------------------

    def _handle_consent_grant(self, deferred: bool) -> str:
        if deferred:
            self.symptoms_deferred = True
            self.defer_from = self.state
            return "Sure, we'll finish this first and then go over your symptoms. " + self._ask_current()
        return self._reopen_symptoms()

    def _reopen_symptoms(self) -> str:
        self.after_deferred = self.state if self.state in RESUMABLE_STATES else None
        self.symptoms_declined = False
        self.consent_refusals = 0
        self.repeat_count = 0
        self._move(State.SYMPTOM_COLLECTION)
        return self._symptom_question()

    # ------------------------------------------------------------------
    # Back and skip navigation
    # ------------------------------------------------------------------

    def _go_back(self) -> str:
        self.repeat_count = 0

        if self.state == State.VITALS_COLLECTION:
            if self.pending_clarification:
                self.pending_clarification = None
                return self._vitals_step()
            if self.vitals_order:
                field = self.vitals_order.pop()
                self.vitals[field] = None
                retire_fact(self.record_id, "vital", field)
                return self._say(
                    f"Acknowledge that you're going back, and ask about their {VITAL_LABELS[field]} again.",
                    f"Okay, let's go back. {VITAL_QUESTIONS[field]}",
                )

        if self.back_stack:
            previous = self.back_stack.pop()
            self.visited_states.discard(self.state)
            self.visited_states.discard(previous)
            self.state = previous
            return self._ask_current()

        return "There's no earlier question to go back to. " + self._ask_current()

    def _skip_current(self) -> str:
        s = self.state
        self.repeat_count = 0

        if s == State.SYMPTOM_COLLECTION:
            if not self._symptom_rows():
                return self._say(
                    "Explain that at least one symptom is needed to continue, and ask again.",
                    "I do need at least one symptom to continue. What's been bothering you?",
                )
            return self._go_next()

        if s == State.VITALS_COLLECTION:
            if self.pending_clarification:
                field = self.pending_clarification["field"]
                self.pending_clarification = None
            else:
                needed = self._needed_vitals()
                if not needed:
                    return self._go_next()
                field = needed[0]
            self._set_vital(field, {"status": "skipped", "display": "skipped by patient"})
            return self._vitals_step()

        if s == State.HISTORY_COLLECTION:
            return self._go_next()

        if s == State.APPOINTMENT_OFFER:
            return self._close_with_summary()

        if s == State.NAME_COLLECTION:
            self._move(State.AGE_COLLECTION)
            return self._age_question()

        if s == State.AGE_COLLECTION:
            self._move(State.GENDER_COLLECTION)
            return self._gender_question()

        if s == State.GENDER_COLLECTION:
            return self._show_review()

        if s == State.BOOKING_REVIEW:
            return self._book()

        if s == State.ANYTHING_ELSE:
            return self._close_with_summary()

        return self._ask_current()

    def _ask_current(self) -> str:
        """Asks the question for the current step again (used by 'back' and resumes)."""
        s = self.state
        if s == State.CONSENT:
            return self._say(
                "Ask whether it is okay to ask about their symptoms.",
                "Is it okay if I ask you about your symptoms?",
            )
        if s == State.SYMPTOM_COLLECTION:
            return self._symptom_question()
        if s == State.VITALS_COLLECTION:
            return self._vitals_step()
        if s == State.HISTORY_COLLECTION:
            return self._history_question()
        if s == State.APPOINTMENT_OFFER:
            return self._offer_question()
        if s == State.NAME_COLLECTION:
            return self._name_question()
        if s == State.AGE_COLLECTION:
            return self._age_question()
        if s == State.GENDER_COLLECTION:
            return self._gender_question()
        if s == State.BOOKING_REVIEW:
            return self._show_review()
        if s == State.ANYTHING_ELSE:
            return self._say(
                "Ask whether there is anything else they need today.",
                "Is there anything else you'd like help with?",
            )
        return self._say("Ask what they need help with today.", "What can I help you with today?")

    # ------------------------------------------------------------------
    # Choosing the next topic
    # ------------------------------------------------------------------

    def _status(self) -> dict:
        return {
            "symptoms": [r["subject"] for r in self._symptom_rows()],
            "vitals_known": [f for f in VITAL_ORDER if self.vitals[f] is not None],
            "history_collected": bool(self._history_summary()),
            "visited": sorted(s.name for s in self.visited_states),
        }

    def _choose_next_state(self, candidates: list) -> State:
        default = candidates[0]
        if propose_next_state is None:
            return default
        allowed = [s.name for s in candidates]
        try:
            proposal = propose_next_state(self._recent(8), self._status(), allowed)
        except Exception as err:
            print(f"    [debug] propose_next_state failed: {err}")
            return default
        if not isinstance(proposal, dict):
            return default
        name = proposal.get("next_state")
        print(f"    [debug] LLM proposed {name}: {proposal.get('reason')}")
        for s in candidates:
            if s.name == name:
                return s
        return default

    def _go_next(self, aside=None) -> str:
        # A step was interrupted by symptoms. Symptoms are done, so return to it.
        if self.after_deferred is not None:
            resume = self.after_deferred
            self.after_deferred = None
            self.repeat_count = 0
            if resume == State.APPOINTMENT_OFFER:
                return "Thanks, I have your symptoms now. " + self._offer_booking(aside)
            if resume in (State.NAME_COLLECTION, State.AGE_COLLECTION, State.GENDER_COLLECTION):
                return "Thanks, I have your symptoms now. " + self._start_identity()
            self._move(resume)
            return "Thanks, I have your symptoms now. " + self._ask_current()

        candidates = [s for s in NAV_ORDER if s not in self.visited_states]
        if not candidates:
            return self._close_with_summary(aside)

        chosen = self._choose_next_state(candidates)
        self.repeat_count = 0

        if chosen == State.VITALS_COLLECTION:
            self._move(chosen)
            return self._vitals_step(aside)
        if chosen == State.HISTORY_COLLECTION:
            self._move(chosen)
            return self._history_question(aside)
        return self._offer_booking(aside)

    # ------------------------------------------------------------------
    # Symptoms
    # ------------------------------------------------------------------

    def _symptom_question(self, aside=None) -> str:
        rows = self._symptom_rows()
        if not rows:
            return self._say("Ask what has been bothering them today.", "What's been bothering you today?", aside)

        parts, missing = [], []
        for r in rows:
            d = r["details"]
            bits = [r["subject"]]
            if d.get("duration"):
                bits.append(f"for {d['duration']}")
            else:
                missing.append(f"how long the {r['subject']} has lasted")
            if d.get("severity"):
                bits.append(f"(severity {d['severity']})")
            else:
                missing.append(f"how severe the {r['subject']} is")
            parts.append(" ".join(bits))

        summary = "; ".join(parts)
        content = (
            f"The patient has reported: {summary}. Still unknown: "
            f"{'; '.join(missing) if missing else 'nothing important'}. "
            "Ask ONE focused follow-up question a doctor would ask next, about the most useful "
            "unknown detail, referring to the patient's own words. If nothing important is unknown, "
            "ask whether anything else is bothering them."
        )
        fallback = (
            f"Okay. Can you tell me {missing[0]}?" if missing
            else "Okay. Anything else you're feeling alongside that?"
        )
        return self._say(content, fallback, aside)

    def _handle_symptom_collection(self, user_text: str) -> str:
        if self._turn_wants_stop and self._symptom_rows():
            self.repeat_count = 0
            return self._go_next()

        if not self._symptom_rows():
            if self.repeat_count >= MAX_REPEATS:
                self.repeat_count = 0
                return self._close_with_summary()
            return self._retry(
                "Ask the patient what symptom is bothering them, gently.",
                [
                    "Could you tell me what symptom is bothering you?",
                    "What's the main thing you're feeling today? For example, pain, fever, or a cough.",
                    "Can you describe the one symptom that made you come in today?",
                ],
            )

        self.symptom_followups += 1
        if self.symptom_followups > MAX_SYMPTOM_FOLLOWUPS:
            self.repeat_count = 0
            return self._go_next()

        self.repeat_count = 0
        return self._symptom_question()

    # ------------------------------------------------------------------
    # Vitals
    # ------------------------------------------------------------------

    def _set_vital(self, field: str, entry: dict) -> None:
        if self.vitals[field] is None and field not in self.vitals_order:
            self.vitals_order.append(field)
        self.vitals[field] = entry
        if entry["status"] == "known":
            add_fact(self.record_id, None, "vital", field, entry["display"], {}, "active", "")
        elif entry["status"] == "declined":
            add_fact(self.record_id, None, "vital", field, entry["display"], {}, "declined", "")

    def _apply_vitals_batch(self, batch: dict) -> None:
        d = batch.get("diabetes") or {}
        if self.vitals["diabetes"] is None:
            if d.get("declined"):
                self._set_vital("diabetes", {"status": "declined", "display": "declined to say"})
            elif d.get("stated") and d.get("has_diabetes") is not None:
                self._set_vital("diabetes", {
                    "status": "known",
                    "display": "yes" if d["has_diabetes"] else "no",
                })

        for field in VITAL_ORDER[1:]:
            if self.vitals[field] is not None:
                continue
            f = batch.get(field) or {}
            status = f.get("status")
            if status == "known":
                self._set_vital(field, {"status": "known", "display": self._format_value(f)})
            elif status == "doesnt_know":
                self._set_vital(field, {"status": "doesnt_know", "display": "doesn't know"})
            elif status == "declined":
                self._set_vital(field, {"status": "declined", "display": "declined to say"})
            elif status == "unclear" and self.pending_clarification is None:
                self.pending_clarification = {
                    "field": field,
                    "value_text": f.get("value_text") or "",
                    "question": f.get("clarifying_question") or f"Could you clarify your {VITAL_LABELS[field]}?",
                }

    def _vitals_step(self, aside=None) -> str:
        if self.pending_clarification:
            q = self.pending_clarification["question"]
            return self._say(f"Ask this clarifying question, keeping its meaning: {q}", q, aside)

        needed = self._needed_vitals()
        if not needed:
            return self._go_next(aside)

        field = needed[0]
        q = VITAL_QUESTIONS[field]
        return self._say(
            f"Ask the patient about their {VITAL_LABELS[field]}. Keep the same kind of answer: {q}",
            q,
            aside,
        )

    def _handle_vitals_collection(self, user_text: str) -> str:
        if self.pending_clarification:
            return self._handle_clarification_reply(user_text)

        needed = self._needed_vitals()
        if not needed:
            return self._go_next()

        before = self._resolved_count()
        batch = extract_vitals_batch(user_text, recent_context=self._recent(), pending_field=needed[0])
        print(f"    [debug] LLM extracted vitals: {batch}")
        self._apply_vitals_batch(batch)

        progressed = self._resolved_count() > before or self.pending_clarification is not None
        if progressed:
            self.repeat_count = 0
        else:
            self.repeat_count += 1
            if self.repeat_count >= MAX_REPEATS:
                self._set_vital(needed[0], {"status": "skipped", "display": "not provided"})
                self.repeat_count = 0

        return self._vitals_step(batch.get("off_topic_remark"))

    def _handle_clarification_reply(self, user_text: str) -> str:
        pc = self.pending_clarification
        res = resolve_vitals_clarification(pc["field"], pc["value_text"], pc["question"], user_text)
        outcome = res.get("outcome")
        aside = res.get("off_topic_remark")

        if outcome == "value":
            self._set_vital(pc["field"], {"status": "known", "display": self._format_value(res)})
            self.pending_clarification = None
            self.repeat_count = 0
        elif outcome in ("doesnt_know", "declined"):
            self._set_vital(pc["field"], {"status": outcome, "display": outcome.replace("_", " ")})
            self.pending_clarification = None
            self.repeat_count = 0
        else:
            self.repeat_count += 1
            if self.repeat_count >= MAX_REPEATS:
                self._set_vital(pc["field"], {"status": "skipped", "display": "not provided"})
                self.pending_clarification = None
                self.repeat_count = 0

        return self._vitals_step(aside)

    # ------------------------------------------------------------------
    # History (facts were saved by _absorb)
    # ------------------------------------------------------------------

    def _history_question(self, aside=None) -> str:
        return self._say(
            "Ask one natural question about any past medical conditions, regular medication, or allergies.",
            "Do you have any past medical conditions, regular medication, or allergies?",
            aside,
        )

    def _handle_history_collection(self, user_text: str) -> str:
        self.repeat_count = 0
        return self._go_next()

    # ------------------------------------------------------------------
    # Triage and booking offer
    # ------------------------------------------------------------------

    def _offer_booking(self, aside=None) -> str:
        self._move(State.APPOINTMENT_OFFER)

        rows = self._symptom_rows()
        names = {r["subject"] for r in rows}
        kb_result = (lookup_condition(names) if names else None) or no_match_response()
        symptom_list = [
            {
                "name": r["subject"] + (
                    " (" + ", ".join(f"{k} {v}" for k, v in r["details"].items()) + ")"
                    if r["details"] else ""
                ),
                "duration_value": None, "duration_unit": None, "severity": None, "onset": None,
            }
            for r in rows
        ]
        vitals_hint = {f: v["display"] for f, v in self.vitals.items() if v and v["status"] == "known"}
        refined = triage_reasoning(symptom_list, self._history_summary(), vitals_hint, kb_result)
        result = refined or kb_result
        self.triage_result = result

        condition = result.get("condition") or "something I cannot confidently identify"
        advice = result["advice"]
        content = (
            f"Tell the patient, calmly, that this could be related to: {condition}. "
            f"Then give this advice: {advice} "
            "Then ask whether they would like you to help book an appointment."
        )
        fallback = (
            f"Based on what you've told me, this could be related to: {condition}. "
            f"{advice} Would you like me to help book an appointment?"
        )
        self.repeat_count = 0
        return self._say(content, fallback, aside, locked=f"{condition}\n{advice}")

    def _offer_question(self, aside=None) -> str:
        if self.symptoms_declined:
            return self._say(
                "Ask whether they would like to book an appointment. A yes or no is enough.",
                "Would you like to book an appointment? A yes or no is fine.",
                aside,
            )
        return self._say(
            "Ask again whether they'd like help booking an appointment. A yes or no is enough.",
            "Would you like me to help book an appointment? A yes or no is fine.",
            aside,
        )

    def _handle_appointment_offer(self, user_text: str) -> str:
        intent = self._yes_no(user_text)
        
        if intent == "unclear":
            if self.repeat_count >= MAX_REPEATS:
                self.repeat_count = 0
                if self.symptoms_declined:
                    return self._close_with_summary()
                return self._go_next()
            return self._retry(
                "Ask again whether they'd like help booking an appointment. A yes or no is enough.",
                [
                    "Sorry, would you like me to book an appointment? A yes or no is fine.",
                    "Would you like me to set up an appointment for you? Just yes or no.",
                ],
            )

        self.repeat_count = 0
        if intent == "no":
            if self.symptoms_declined:
                return self._close_with_summary()
            return self._go_next()

        return self._start_identity()

    # ------------------------------------------------------------------
    # Identity (facts were saved by _absorb)
    # ------------------------------------------------------------------

    def _name_question(self, aside=None) -> str:
        return self._say(
            "Ask for the patient's name for the booking, and mention they can skip it if they prefer.",
            "Before I book, may I have your name? You can skip this if you prefer.",
            aside,
        )

    def _age_question(self, aside=None) -> str:
        return self._say("Briefly acknowledge, then ask for their age.", "Thanks. How old are you?", aside)

    def _gender_question(self, aside=None) -> str:
        return self._say(
            "Briefly acknowledge, then ask for their gender. They can skip it if they prefer.",
            "Thanks. What is your gender? You can skip this if you prefer.",
            aside,
        )

    def _handle_name_collection(self, user_text: str) -> str:
        value, status = self._identity("name")
        if value:
            self.repeat_count = 0
            return self._start_identity()
        if status == "declined" or self.repeat_count >= MAX_REPEATS:
            self.repeat_count = 0
            return self._start_identity()
        return self._retry(
            "Politely ask again for their name, and mention they can skip it.",
            [
                "Sorry, could I get your name? You can skip this if you prefer.",
                "What name should I put on the booking? You can also skip this.",
            ],
        )

    def _handle_age_collection(self, user_text: str) -> str:
        value, status = self._identity("age")
        if value:
            self.repeat_count = 0
            return self._start_identity()
        if status == "declined" or self.repeat_count >= MAX_REPEATS:
            self.repeat_count = 0
            return self._start_identity()
        return self._retry(
            "Politely ask again for their age in years.",
            ["Sorry, how old are you in years?", "What is your age, in years?"],
        )

    def _handle_gender_collection(self, user_text: str) -> str:
        value, status = self._identity("gender")
        if value:
            self.repeat_count = 0
            return self._start_identity()
        if status == "declined" or self.repeat_count >= MAX_REPEATS:
            self.repeat_count = 0
            return self._start_identity()
        return self._retry(
            "Politely ask again for their gender, and mention they can skip it.",
            [
                "Sorry, what is your gender? You can skip this if you prefer.",
                "What gender should I record? You're welcome to skip it.",
            ],
        )

    # ------------------------------------------------------------------
    # Booking review and booking
    # ------------------------------------------------------------------

    def _review_table(self) -> str:
        name, _ = self._identity("name")
        age, _ = self._identity("age")
        gender, _ = self._identity("gender")
        rows = self._symptom_rows()
        if self.symptoms_declined:
            symptoms = "declined to share"
        elif rows:
            symptoms = ", ".join(
                r["subject"] + (f" ({', '.join(f'{k} {v}' for k, v in r['details'].items())})" if r["details"] else "")
                for r in rows
            )
        else:
            symptoms = "none"
        vitals = [f"{VITAL_LABELS[f]}: {self.vitals[f]['display']}" for f in VITAL_ORDER if self.vitals[f]]
        condition = (self.triage_result or {}).get("condition") or "not identified"
        lines = [
            f"Name: {name or 'not given'}",
            f"Age: {age or 'not given'}",
            f"Gender: {gender or 'not given'}",
            f"Symptoms: {symptoms}",
            f"Vitals: {'; '.join(vitals) or 'none given'}",
            f"History: {self._history_summary() or 'none given'}",
            f"Possible condition: {condition}",
        ]
        return "\n".join(lines)

    def _show_review(self, aside=None) -> str:
        self._move(State.BOOKING_REVIEW)
        table = self._review_table()
        return self._say(
            "Show the patient this summary of what was collected, then ask whether it looks right "
            "or if they want to change their name, age, or gender before booking.",
            f"Here is what I have:\n{table}\nDoes this look right, or would you like to change anything?",
            aside,
            locked=table,
        )

    def _handle_booking_review(self, user_text: str) -> str:
        r = extract_review_action(user_text, self._recent())
        aside = r.get("off_topic_remark")

        if r["action"] == "confirm":
            self.repeat_count = 0
            return self._book(aside)

        if r["action"] == "edit" and r.get("edit_field") in ("name", "age", "gender"):
            field = r["edit_field"]
            self.repeat_count = 0
            target = {
                "name": State.NAME_COLLECTION,
                "age": State.AGE_COLLECTION,
                "gender": State.GENDER_COLLECTION,
            }[field]
            self._move(target)
            questions = {
                "name": "What name should I use for the booking?",
                "age": "What is the correct age?",
                "gender": "What is the correct gender?",
            }
            return self._say(f"Ask for the corrected {field}: {questions[field]}", questions[field], aside)

        if r["action"] == "edit":
            return self._retry(
                "Explain that for now only name, age, or gender can be changed here, and ask whether to "
                "change one of those or go ahead and book.",
                "I can only change your name, age, or gender at this stage. Would you like to change one of those, or shall I book?",
                aside,
            )

        if self.repeat_count >= MAX_REPEATS:
            self.repeat_count = 0
            return self._book(aside)

        return self._retry(
            "Ask again whether the summary looks right or whether they want to change name, age, or gender.",
            [
                "Sorry, does this look right, or would you like to change your name, age, or gender?",
                "Shall I go ahead and book this, or is something here wrong?",
            ],
            aside,
        )

    def _book(self, aside=None) -> str:
        name, _ = self._identity("name")
        age, _ = self._identity("age")
        gender, _ = self._identity("gender")
        triage = self.triage_result or {}
        symptoms = [r["subject"] for r in self._symptom_rows()]

        # Vitals as plain text. Skipped vitals become None.
        vitals = {
            f: (self.vitals[f]["display"]
                if self.vitals[f] and self.vitals[f]["status"] in ("known", "declined") else None)
            for f in VITAL_ORDER
        }

        try:
            row_id = book_appointment(
                patient_name=name or "Unknown",
                age=age,
                gender=gender,
                symptoms=symptoms,
                diabetes=vitals["diabetes"],
                sugar_level=vitals["sugar_level"],
                blood_pressure=vitals["blood_pressure"],
                weight=vitals["weight"],
                height=vitals["height"],
                temperature=vitals["temperature"],
                history=self._history_dict(),
                condition=triage.get("condition"),
                urgency=triage.get("urgency", "unknown"),
                requested_time="TBD - placeholder slot",   # TODO: real slot selection
            )
        except Exception as err:
            print(f"    [debug] booking failed: {err}")
            return self._say(
                "Tell the patient the booking could not be completed right now, and that they should "
                "contact the hospital directly.",
                "Sorry, I couldn't complete the booking right now. Please contact the hospital directly.",
                aside,
            )

        reference = str(row_id)

        # JSON copy for inspection. The database is the source of truth.
        try:
            _append_booking_json(reference, {
                "patient_record_id": self.record_id,
                "name": name or "Unknown",
                "age": age,
                "gender": gender,
                "symptoms": [
                    {"name": r["subject"], "details": r["details"], "patient_words": r["verbatim_quote"]}
                    for r in self._symptom_rows()
                ],
                "vitals": {f: v for f, v in vitals.items() if v},
                "history": self._history_dict(),
                "possible_condition": triage.get("condition"),
                "urgency": triage.get("urgency", "unknown"),
                "requested_time": "TBD - placeholder slot",
            })
        except Exception as err:
            print(f"    [debug] JSON copy failed (booking itself was saved): {err}")

        self.booking_reference = reference
        self._move(State.ANYTHING_ELSE)
        return self._say(
            f"Confirm that the appointment has been requested with reference number {reference}, "
            "then ask if there is anything else before wrapping up.",
            f"Your appointment has been requested (reference #{reference}). Anything else before we wrap up?",
            aside,
            locked=f"reference #{reference}",
        )

    # ------------------------------------------------------------------
    # Ending
    # ------------------------------------------------------------------

    def _close_with_summary(self, aside=None) -> str:
        self.state = State.CLOSING
        self.closed = True
        if self.patient_started or self._has_facts() or self.booking_reference:
            set_patient_outcome(self.record_id, "booked" if self.booking_reference else "no_booking")

        symptoms = ", ".join(r["subject"] for r in self._symptom_rows())
        parts = [f"Symptoms noted: {symptoms}." if symptoms else "No symptoms were noted."]
        if self.booking_reference:
            parts.append(f"Appointment requested (reference #{self.booking_reference}).")
        else:
            parts.append("No appointment was booked.")
        summary = " ".join(parts)
        return self._say(
            f"Give the patient this brief summary, keeping its facts exactly: {summary} "
            "Then end with a warm, brief goodbye.",
            f"{summary} Take care, and see a doctor if things get worse.",
            aside,
            locked=summary,
        )

    def _handle_anything_else(self, user_text: str) -> str:
        intent = self._yes_no(user_text)

        if intent == "unclear":
            if self.repeat_count >= MAX_REPEATS:
                return self._close_with_summary()
            return self._retry(
                "Ask again whether there is anything else they need today. A yes or no is enough.",
                [
                    "Sorry, is there anything else you'd like help with? A yes or no is fine.",
                    "Is there anything else I can help you with today? Just yes or no.",
                ],
            )

        self.repeat_count = 0
        if intent == "no":
            return self._close_with_summary()

        self._start_new_record(previous_outcome="booked" if self.booking_reference else "no_booking")
        self._move(State.SYMPTOM_COLLECTION)
        return self._say(
            "Ask what else they would like help with today, such as new symptoms or a new appointment.",
            "Sure - what's bothering you now?",
        )

    def _handle_closing(self, user_text: str) -> str:
        r = extract_closing_intent(user_text, self._recent())
        aside = r.get("off_topic_remark")

        if r["intent"] == "new_request":
            self._start_new_record(previous_outcome=None)
            return self._consent_prompt()

        return self._say(
            "Say a short, warm goodbye.",
            "Take care! Reach out anytime you need help.",
            aside,
        )

    # ------------------------------------------------------------------
    # Opening and consent
    # ------------------------------------------------------------------

    def _handle_greeting(self, user_text: str) -> str:
        self.state = State.CONSENT
        return self._say(
            "Introduce yourself as a hospital intake assistant. Say you help check symptoms and book "
            "an appointment if needed, that you are not a doctor, and that this is not a diagnosis. "
            "Ask whether that is okay.",
            CONSENT_FALLBACK,
            is_first=True,
        )

    def _handle_consent(self, user_text: str) -> str:
        intent = self._yes_no(user_text)

        if intent == "no":
            self.consent_refusals += 1
            self.patient_started = True
            self.repeat_count = 0

            if self.consent_refusals >= 2:
                self.symptoms_declined = True
                self._move(State.APPOINTMENT_OFFER)
                return self._say(
                    "The patient has declined the symptom questions twice. Respect that without pressure. "
                    "Say you will skip the symptom questions, then ask whether they would like to book an "
                    "appointment. A yes or no is enough.",
                    "That's okay, we'll skip the symptom questions. Would you like to book an appointment? "
                    "A yes or no is fine.",
                )

            return self._say(
                "The patient said no to the intake questions. Gently explain why it helps: the questions "
                "are only about their symptoms, and the answers let the hospital give them the right "
                "guidance and a proper appointment. Then ask simply whether it is okay to ask about "
                "their symptoms. No pressure.",
                "I understand. I only ask about your symptoms, so the hospital can point you to the right "
                "care. Would you be okay with a few questions about what's bothering you?",
            )

        if intent == "unclear":
            return self._retry(
                "Ask again, simply, whether it is okay to ask about their symptoms.",
                [
                    "Sorry, just to confirm - is it okay if I ask about your symptoms?",
                    "Are you okay with me asking a few questions about your symptoms? A yes or no works.",
                ],
            )

        self.repeat_count = 0
        self.patient_started = True
        self._move(State.SYMPTOM_COLLECTION)
        if self._symptom_rows():
            return self._symptom_question()
        return self._say(
            "Briefly acknowledge their agreement, then ask what has been bothering them today.",
            "Thank you. Let's start - what's been bothering you today?",
        )

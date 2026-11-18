"""
intake_store.py

Three-layer storage for the intake bot. Uses SQLite now. The same tables map
directly to PostgreSQL (with a JSONB column for details) when you move to it.

Layer 1 - core (patients): the fixed fields the code always needs.
Layer 2 - facts (patient_facts): one row per piece of information. details is
          JSON, so new kinds of information need no new column. Facts are
          append-only: a changed fact gets a new row, and the old row is marked
          superseded. verbatim_quote keeps the patient's own words. provenance
          records where a fact came from.
Layer 3 - episodes: groups symptom facts into problems, so follow-up answers
          attach to the right problem.

Every query filters by patient_id, so one patient's data is never returned
for another patient.

Semantic (embedding) memory is not implemented yet. It needs an embedding model,
which uses API quota.
"""

import json
import sqlite3
from datetime import datetime

DB_PATH = "intake.db"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_fact(row) -> dict:
    d = dict(row)
    d["details"] = json.loads(d["details"]) if d.get("details") else {}
    return d


def init_store() -> None:
    """Creates the tables if they do not exist. Safe to call many times."""
    conn = _conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            patient_no INTEGER NOT NULL,
            outcome TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            opened_at TEXT
        );
        CREATE TABLE IF NOT EXISTS patient_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            episode_id INTEGER,
            category TEXT NOT NULL,
            subject TEXT NOT NULL,
            value TEXT,
            details TEXT,
            status TEXT,
            verbatim_quote TEXT,
            provenance TEXT NOT NULL DEFAULT 'user_reported',
            active INTEGER NOT NULL DEFAULT 1,
            superseded_by INTEGER,
            created_at TEXT
        );
        """
    )
    conn.commit()
    conn.close()


# ----------------------------------------------------------------------
# Layer 1: core
# ----------------------------------------------------------------------

def create_patient(session_id: str, patient_no: int) -> int:
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO patients (session_id, patient_no, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (session_id, patient_no, _now(), _now()),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def set_patient_outcome(patient_id: int, outcome: str) -> None:
    conn = _conn()
    conn.execute(
        "UPDATE patients SET outcome = ?, updated_at = ? WHERE id = ?",
        (outcome, _now(), patient_id),
    )
    conn.commit()
    conn.close()


# ----------------------------------------------------------------------
# Layer 3: episodes (problems)
# ----------------------------------------------------------------------

def open_episodes(patient_id: int) -> list:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM episodes WHERE patient_id = ? AND status = 'open' ORDER BY id",
        (patient_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_or_open_episode(patient_id: int, label: str) -> int:
    for ep in open_episodes(patient_id):
        if ep["label"] == label:
            return ep["id"]
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO episodes (patient_id, label, status, opened_at) VALUES (?, ?, 'open', ?)",
        (patient_id, label, _now()),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


# ----------------------------------------------------------------------
# Layer 2: facts
# ----------------------------------------------------------------------

def add_fact(patient_id, episode_id, category, subject, value, details, status, quote):
    """
    Saves one fact for one patient. Returns (fact_id, changed).

    Same category + subject as an active fact: the values are merged. If nothing
    changed, the existing row is returned and changed is False. Otherwise a new row
    is written, and the old row is marked superseded_by the new row. Nothing is deleted.
    """
    details = details or {}
    conn = _conn()
    old_row = conn.execute(
        "SELECT * FROM patient_facts WHERE patient_id = ? AND category = ? AND subject = ? "
        "AND active = 1 ORDER BY id DESC LIMIT 1",
        (patient_id, category, subject),
    ).fetchone()

    if old_row:
        old = _row_to_fact(old_row)
        merged_details = {**old["details"], **details}
        merged_value = value if value is not None else old["value"]
        merged_status = status or old["status"]
        merged_episode = episode_id if episode_id is not None else old["episode_id"]
        merged_quote = quote or old["verbatim_quote"]
        unchanged = (
            merged_details == old["details"]
            and merged_value == old["value"]
            and merged_status == old["status"]
            and merged_quote == old["verbatim_quote"]
        )
        if unchanged:
            conn.close()
            return old["id"], False
    else:
        old = None
        merged_details, merged_value = details, value
        merged_status, merged_episode, merged_quote = status, episode_id, quote

    cur = conn.execute(
        "INSERT INTO patient_facts (patient_id, episode_id, category, subject, value, details, "
        "status, verbatim_quote, provenance, active, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'user_reported', 1, ?)",
        (
            patient_id, merged_episode, category, subject, merged_value,
            json.dumps(merged_details, ensure_ascii=False), merged_status, merged_quote, _now(),
        ),
    )
    new_id = cur.lastrowid
    if old:
        conn.execute(
            "UPDATE patient_facts SET active = 0, superseded_by = ? WHERE id = ?",
            (new_id, old["id"]),
        )
    conn.commit()
    conn.close()
    return new_id, True


def retire_fact(patient_id: int, category: str, subject: str) -> None:
    """Used when the patient goes back and undoes an answer. The row is kept, but inactive."""
    conn = _conn()
    conn.execute(
        "UPDATE patient_facts SET active = 0, status = 'retracted' "
        "WHERE patient_id = ? AND category = ? AND subject = ? AND active = 1",
        (patient_id, category, subject),
    )
    conn.commit()
    conn.close()


def active_facts(patient_id: int) -> list:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM patient_facts WHERE patient_id = ? AND active = 1 ORDER BY id",
        (patient_id,),
    ).fetchall()
    conn.close()
    return [_row_to_fact(r) for r in rows]


def has_facts(patient_id: int) -> bool:
    conn = _conn()
    row = conn.execute(
        "SELECT 1 FROM patient_facts WHERE patient_id = ? AND active = 1 LIMIT 1",
        (patient_id,),
    ).fetchone()
    conn.close()
    return row is not None

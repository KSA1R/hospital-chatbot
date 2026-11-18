"""
appointments_db.py

Persistent storage for appointments AND emergency events.

SCHEMA NOTE: this version adds diabetes/sugar_level/blood_pressure/height
columns to support the restructured vitals flow. If you have an existing
hospital_bot.db from before this change, delete it - CREATE TABLE IF NOT
EXISTS won't retroactively add columns to an already-existing table.

Two tables:
- appointments: the full patient record collected during the conversation.
- emergency_log: every time a red flag fires, for later follow-up/auditing.
  This never affects the immediate emergency response itself.

TODO: Add doctor availability table so booking checks real free slots
      instead of accepting any requested time.
TODO: Add basic encryption or at least avoid storing raw symptom/history
      text here long-term - see privacy notes from earlier in the project.
"""

import json
import sqlite3
from datetime import datetime

DB_PATH = "hospital_bot.db"

# Every column each table SHOULD have, kept in one place. init_db() uses
# this both to create a brand-new table AND to patch an existing one that
# is missing columns - see _migrate_missing_columns() below. When you add
# a new field in the future, add it here and it will self-heal on the
# next run instead of throwing "table X has no column named Y".
_APPOINTMENTS_COLUMNS = {
    "patient_name": "TEXT",
    "age": "TEXT",
    "gender": "TEXT",
    "symptoms": "TEXT",
    "diabetes": "TEXT",
    "sugar_level": "TEXT",
    "blood_pressure": "TEXT",
    "weight": "TEXT",
    "height": "TEXT",
    "temperature": "TEXT",
    "history": "TEXT",
    "condition": "TEXT",
    "urgency": "TEXT",
    "requested_time": "TEXT",
    "created_at": "TEXT",
}

_EMERGENCY_LOG_COLUMNS = {
    "patient_name": "TEXT",
    "triggered_phrase": "TEXT",
    "user_message": "TEXT",
    "created_at": "TEXT",
}


def _migrate_missing_columns(conn: sqlite3.Connection, table: str, expected_columns: dict) -> None:
    """
    Adds any columns that are missing from an ALREADY-EXISTING table,
    without touching existing rows. This is what stops the
    "table X has no column named Y" error from coming back every time a
    new field gets added - CREATE TABLE IF NOT EXISTS only creates a
    table when it's missing entirely; it never updates an existing
    table's columns on its own, so without this, every future schema
    change would need a manual `rm hospital_bot.db` or ALTER TABLE.
    """
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    existing_columns = {row[1] for row in cur.fetchall()}   # row[1] is the column name
    for column_name, column_type in expected_columns.items():
        if column_name not in existing_columns:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column_name} {column_type}")
            print(f"    [db migration] added missing column '{column_name}' to '{table}'")


def init_db(path: str = DB_PATH) -> None:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            age TEXT,
            gender TEXT,
            symptoms TEXT,
            diabetes TEXT,
            sugar_level TEXT,
            blood_pressure TEXT,
            weight TEXT,
            height TEXT,
            temperature TEXT,
            history TEXT,
            condition TEXT,
            urgency TEXT,
            requested_time TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS emergency_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            triggered_phrase TEXT,
            user_message TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()

    # Self-heal: if this is an EXISTING database from before some column
    # was added, patch it now instead of failing later on insert.
    _migrate_missing_columns(conn, "appointments", _APPOINTMENTS_COLUMNS)
    _migrate_missing_columns(conn, "emergency_log", _EMERGENCY_LOG_COLUMNS)

    conn.close()


def book_appointment(
    patient_name: str,
    age: str | None,
    gender: str | None,
    symptoms: list[str],
    diabetes: str | None,
    sugar_level: str | None,
    blood_pressure: str | None,
    weight: str | None,
    height: str | None,
    temperature: str | None,
    history: dict | None,
    condition: str | None,
    urgency: str,
    requested_time: str,
    path: str = DB_PATH,
) -> int:
    """Insert a new appointment record with the full patient record. Returns the new row id."""
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO appointments
            (patient_name, age, gender, symptoms, diabetes, sugar_level,
             blood_pressure, weight, height, temperature, history,
             condition, urgency, requested_time, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_name,
            age,
            gender,
            ", ".join(symptoms),
            diabetes,
            sugar_level,
            blood_pressure,
            weight,
            height,
            temperature,
            json.dumps(history) if history else None,
            condition,
            urgency,
            requested_time,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def log_emergency(patient_name: str | None, triggered_phrase: str, user_message: str, path: str = DB_PATH) -> int:
    """Record that a red flag fired - for follow-up/auditing, not for altering the emergency response itself."""
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO emergency_log (patient_name, triggered_phrase, user_message, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (patient_name, triggered_phrase, user_message, datetime.now().isoformat()),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_appointment(appointment_id: int, path: str = DB_PATH) -> tuple | None:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM appointments WHERE id = ?", (appointment_id,))
    row = cur.fetchone()
    conn.close()
    return row

"""
symptom_kb.py

Static symptom -> possible condition lookup. This is NOT an LLM call -
it's a plain table lookup, so results are predictable and checkable.

TODO: Replace SYMPTOM_TABLE with real curated data, e.g. derived from
      SNOMED CT / ICD-10 mappings or a public dataset like DDXPlus.
      Keep the same shape (list of dicts) so lookup_condition() doesn't
      need to change.

TODO (future extension, not implemented yet): use the patient's
      structured medical history (see llm_client.extract_history()) to
      escalate urgency for genuinely relevant combinations - e.g. a
      respiratory infection is riskier for someone with asthma or COPD.
      This would mean adding a "risk_modifiers" field per row and
      passing history_conditions into lookup_condition(). Deliberately
      left out for now to keep the KB simple while other pieces
      (structured history extraction) get built first.
"""

# Placeholder data - NOT medically validated. Fill this in properly later.
SYMPTOM_TABLE = [
    {
        "symptoms": {"fever", "cough", "fatigue"},
        "condition": "Possible viral infection (e.g. flu)",
        "urgency": "low",
        "advice": "Rest, fluids, and monitor. See a doctor if it doesn't improve in a few days.",
    },
    {
        "symptoms": {"headache", "fever"},
        "condition": "Possible viral infection or migraine",
        "urgency": "low",
        "advice": "Monitor symptoms. See a doctor if headache is severe or persistent.",
    },
    {
        "symptoms": {"stomach pain", "nausea", "vomiting"},
        "condition": "Possible gastrointestinal infection",
        "urgency": "medium",
        "advice": "Stay hydrated. See a doctor within a day or two if symptoms persist.",
    },
]


def lookup_condition(reported_symptoms: set[str]) -> dict | None:
    """
    Very basic matching: find the KB row with the most symptom overlap.
    Real version should probably score/rank multiple matches instead of
    just returning the single best guess.
    """
    best_match = None
    best_overlap = 0

    for row in SYMPTOM_TABLE:
        overlap = len(reported_symptoms & row["symptoms"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = row

    return best_match


def no_match_response() -> dict:
    """Fallback when nothing in the KB matches well enough."""
    return {
        "condition": None,
        "urgency": "unknown",
        "advice": "I'm not confident about what might be causing this. "
        "Please see a doctor for a proper evaluation.",
    }

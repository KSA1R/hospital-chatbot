"""
red_flags.py
 
Emergency detection. This runs on EVERY user message, before anything else.
Deliberately dumb and fast (plain substring matching) - no LLM, no KB lookup.
False positives are fine. False negatives are not.
 
TODO: Replace RED_FLAG_PHRASES with a clinically sourced list
      (e.g. adapted from WHO emergency triage guidelines or the
      Emergency Severity Index framework). Group by category if it
      helps you keep track of coverage while building it out.
 
TODO (future extension, not implemented yet): this file only scans
      TEXT for danger phrases - it has no concept of a dangerous VITALS
      VALUE (e.g. a 109F temperature, or an extreme heart rate) being
      an emergency on its own, even if the patient's wording sounds
      calm. Would need a separate numeric check (e.g. check_vitals_red_flags(
      temperature=...)) called from state_machine.py's vitals-collection
      step, with real clinical thresholds. Deliberately deferred until
      the skeleton with real DB/KB data is in place.

"""

# Placeholder list - NOT clinically validated. Fill this in properly later.
# Grouped by category per the TODO above, to make coverage gaps easier to
# spot while this grows. Deliberately includes casual/typo'd phrasing real
# patients use ("blood every where", "im dying") - not just clinical
# terminology - since a scared patient describing a traumatic injury does
# not talk like a textbook. Overlap between categories is fine; the same
# real-world statement often trips more than one entry, which is a feature
# here, not redundancy to clean up.
RED_FLAG_PHRASES = [
    # --- Cardiac / respiratory ---
    "chest pain",
    "can't breathe",
    "cant breathe",
    "difficulty breathing",
    "shortness of breath",
    "throat closing",

    # --- Bleeding / hemorrhage ---
    "severe bleeding",
    "won't stop bleeding",
    "wont stop bleeding",
    "can't stop the bleeding",
    "cant stop the bleeding",
    "bleeding a lot",
    "bleeding heavily",
    "heavy bleeding",
    "blood everywhere",
    "blood every where",
    "blood all over",
    "so much blood",
    "lots of blood",
    "losing a lot of blood",
    "gushing blood",

    # --- Traumatic injury / amputation / limb loss ---
    "hand is gone",
    "arm is gone",
    "leg is gone",
    "foot is gone",
    "no hand",
    "lost my hand",
    "lost my arm",
    "lost my leg",
    "lost a finger",
    "missing my hand",
    "cut off my hand",
    "amputat",   # catches "amputation" / "amputated" / "amputate"

    # --- Neuro / stroke ---
    "unconscious",
    "passed out",
    "not responding",
    "face drooping",
    "slurred speech",
    "one side numb",
    "seizure",

    # --- Severe distress / dying ---
    "i am dying",
    "i'm dying",
    "im dying",
    "i think i'm dying",
    "i think im dying",
    "going to die",
    "gonna die",

    # --- Allergic reaction / overdose ---
    "severe allergic reaction",
    "overdose",

    # --- Mental health crisis ---
    "suicidal",
    "want to die",
]


def check_red_flags(user_text: str) -> str | None:
    """
    Check a single user message for emergency red-flag phrases.

    Returns the matched phrase if found, else None.
    Kept separate from symptom_kb.py on purpose - see project notes on
    why emergency detection must not depend on the KB lookup.
    """
    text = user_text.lower()
    for phrase in RED_FLAG_PHRASES:
        if phrase in text:
            return phrase
    return None


def emergency_message() -> str:
    """
    Fixed, non-negotiable response shown when a red flag fires.
    Do not let the LLM/agent layer rephrase this dynamically -
    keep it deterministic so it can't be "softened" by mistake.
    """
    return (
        "This sounds like it could be a medical emergency. "
        "Please call your local emergency number right now, or go to "
        "the nearest emergency room. I'm not able to help further with "
        "this in chat - please seek immediate in-person help."
        # TODO: swap in the correct local emergency number for your deployment region
    )
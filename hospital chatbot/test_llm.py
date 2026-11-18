"""
test_llm.py

Standalone test for ALL llm_client extraction functions - isolated
from the full conversation flow. Each function is tested with exactly
ONE real API call, so this is the lightest way to confirm Gemini is
actually responding, without running main.py end to end.

Usage:
    export GEMINI_API_KEY="your-key-here"
    python3 test_llm.py
"""

import time
from llm_client import (
    extract_symptoms,
    extract_yes_no,
    extract_temperature,
    extract_weight,
    extract_history,
    extract_name,
)

tests = [
    ("extract_symptoms", extract_symptoms, "I have a fever and a really bad headache"),
    ("extract_yes_no", extract_yes_no, "why"),
    ("extract_temperature", extract_temperature, "it's about 100.4"),
    ("extract_weight", extract_weight, "around 65 kg"),
    ("extract_history", extract_history, "I have asthma and take an inhaler"),
    ("extract_name", extract_name, "you can call me Alex"),
]

for name, func, sample_input in tests:
    print(f"\n{name}({sample_input!r})")
    result = func(sample_input)
    print(f"  -> {result}")
    time.sleep(3)  # small pause to avoid tripping per-minute rate limits

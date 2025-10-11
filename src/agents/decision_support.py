# src/agents/decision_support.py

from typing import Dict

# Edit/extend this map with your dataset’s labels.
# Leaving it minimal is fine; unknown labels will use DEFAULT_ADVICE.
ADVICE_MAP: Dict[str, str] = {
    "Healthy": "No immediate concerns from CBC. Maintain routine follow-ups.",
    "Anemia": "Consider iron studies and retesting hemoglobin; evaluate symptoms and diet.",
    "Leukocytosis": "Assess for infection/inflammation; consider differential and follow-up labs.",
    "Thrombocytopenia": "Recheck platelets; review meds and bleeding history; consider hematology consult.",
}

DEFAULT_ADVICE = (
    "No specific guidance available for this label. "
    "Consider clinician review and confirm with follow-up testing."
)

def suggest(label: str) -> str:
    """Return human-readable guidance for a predicted diagnosis label."""
    if label is None:
        return DEFAULT_ADVICE
    return ADVICE_MAP.get(str(label), DEFAULT_ADVICE)

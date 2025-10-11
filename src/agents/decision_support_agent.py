# src/agents/decision_support_agent.py

from dataclasses import dataclass
from .decision_support import suggest as _suggest_text

@dataclass
class DecisionSupportAgent:
    """Thin wrapper so the rest of the system calls a consistent API."""
    def suggest(self, diagnosis_label: str) -> str:
        return _suggest_text(diagnosis_label)

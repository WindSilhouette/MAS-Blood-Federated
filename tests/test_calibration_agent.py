import numpy as np
from src.agents.calibration_agent import CalibrationAgent
from src.utils.metrics import ece

def test_calibration_reduces_ece():
    y = np.array([0, 1, 0, 1] * 25)
    probs = np.array([0.2, 0.8, 0.4, 0.9] * 25)
    cal = CalibrationAgent()
    cal.fit(y, probs)
    e1 = ece(probs, y)
    e2 = ece(cal.predict_proba(probs), y)
    assert e2 <= e1 + 0.05

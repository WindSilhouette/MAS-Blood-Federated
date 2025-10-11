# tests/test_agents.py

import sys, os
import pandas as pd
import pytest
from pathlib import Path

# Add the project root (one level above tests/) to sys.path so `import src...` works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.data_processing_agent import DataProcessingAgent
from src.agents.feature_extraction_agent import FeatureExtractionAgent

@pytest.fixture
def tiny_csv(tmp_path):
    # Create a small CSV for testing
    data = pd.DataFrame({
        "A": [1.0, None, 3.0],
        "B": [4.0, 5.0, None],
        "Diagnosis": ["X", "Y", "X"]
    })
    file = tmp_path / "mini.csv"
    data.to_csv(file, index=False)
    return file

def test_data_processing_agent_imputes_missing(tiny_csv):
    agent = DataProcessingAgent(data_path=tiny_csv)
    df = agent.run()
    # No NaNs remain
    assert not df.isnull().values.any()
    # Means are imputed: A’s missing becomes (1+3)/2 = 2.0; B’s missing becomes (4+5)/2 = 4.5
    assert pytest.approx(df.loc[1, "A"]) == 2.0
    assert pytest.approx(df.loc[2, "B"]) == 4.5

def test_feature_extraction_splits(tiny_csv):
    # First clean then split
    df = DataProcessingAgent(data_path=tiny_csv).run()
    fe = FeatureExtractionAgent(target_col="Diagnosis")
    X, y = fe.run(df)
    # X should have only A and B; y has Diagnosis
    assert list(X.columns) == ["A", "B"]
    assert list(y) == ["X", "Y", "X"]
    assert X.shape == (3, 2)
    assert y.shape == (3,)

if __name__ == "__main__":
    pytest.main()

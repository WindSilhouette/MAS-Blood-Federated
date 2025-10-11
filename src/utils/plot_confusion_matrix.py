#!/usr/bin/env python
import sys
from pathlib import Path

# 0) Ensure the project root (two levels above utils/) is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.agents.data_processing_agent import DataProcessingAgent
from src.agents.feature_extraction_agent import FeatureExtractionAgent
from src.agents.federated_coordinator_agent import FederatedCoordinatorAgent

def main():
    # 1) Path to your CSV at project_root/data/...
    data_path = PROJECT_ROOT / "data" / "processed_diagnosed_cbc_data.csv"

    # 2) Run federated training to get the final model & encoder
    fed = FederatedCoordinatorAgent(data_path, num_clients=3, rounds=5)
    model, le, _ = fed.run()

    # 3) Prepare a global test set
    df = DataProcessingAgent(data_path).run()
    X, y = FeatureExtractionAgent().run(df)
    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, le.transform(y),
        test_size=0.2,
        random_state=42,
        stratify=le.transform(y)
    )

    # 4) Compute confusion matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test_enc, y_pred)

    # 5) Plot heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=le.classes_, yticklabels=le.classes_,
        cmap='Blues'
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

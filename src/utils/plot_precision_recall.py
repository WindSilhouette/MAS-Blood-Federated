#!/usr/bin/env python
# src/utils/plot_precision_recall.py

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

# 0) Ensure project root (two levels above utils/) on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.data_processing_agent import DataProcessingAgent
from src.agents.feature_extraction_agent import FeatureExtractionAgent
from src.agents.federated_coordinator_agent import FederatedCoordinatorAgent

def main():
    # 1) Path to your CSV at project_root/data/…
    data_path = PROJECT_ROOT / "data" / "processed_diagnosed_cbc_data.csv"

    # 2) Run federated training to get the global model and label encoder
    fed = FederatedCoordinatorAgent(data_path, num_clients=3, rounds=5)
    model, le, _ = fed.run()

    # 3) Prepare full dataset and split off test set
    df     = DataProcessingAgent(data_path).run()
    X, y    = FeatureExtractionAgent().run(df)
    y_enc   = le.transform(y)
    _, X_test, _, y_test_enc = train_test_split(
        X, y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc
    )

    # 4) Compute predicted probabilities and binarize true labels
    y_score  = model.predict_proba(X_test)
    y_test_b = label_binarize(y_test_enc, classes=list(range(len(le.classes_))))

    # 5) Plot precision–recall for each class
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(le.classes_):
        precision, recall, _ = precision_recall_curve(y_test_b[:,i], y_score[:,i])
        ap = average_precision_score(y_test_b[:,i], y_score[:,i])
        plt.plot(recall, precision, label=f"{cls} (AP={ap:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves by Class")
    plt.legend(loc="lower left", bbox_to_anchor=(1, 0))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# src/utils/plot_feature_importance.py

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# 0) Ensure project root on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explore_and_prepare import load_and_clean, prepare_data
from xgboost import Booster

def main():
    # 1) Load & clean the data
    csv_path = PROJECT_ROOT / "data" / "processed_diagnosed_cbc_data.csv"
    df = load_and_clean(csv_path)
    X, _ = prepare_data(df, target_col="Diagnosis")

    # 2) Load the trained Booster
    model_path = PROJECT_ROOT / "src" / "models" / "xgboost_cbc_model.json"
    booster = Booster()
    booster.load_model(str(model_path))

    # 3) Get feature scores (by weight)
    # returns dict {feature_name: score}
    score_dict = booster.get_score(importance_type='weight')
    # map to DataFrame columns (missing ones get 0)
    fi = [score_dict.get(col, 0) for col in X.columns]

    # 4) Sort and plot
    idx = sorted(range(len(fi)), key=lambda i: fi[i], reverse=True)
    features = X.columns

    plt.figure(figsize=(8, 6))
    plt.barh([features[i] for i in idx], [fi[i] for i in idx])
    plt.xlabel("Feature Importance (weight)")
    plt.title("XGBoost Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()



# src/run_mas.py

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import os
import random

from src.agents.federated_coordinator_agent import FederatedCoordinatorAgent
from src.agents.data_processing_agent import DataProcessingAgent
from src.agents.feature_extraction_agent import FeatureExtractionAgent
from src.agents.decision_support_agent import DecisionSupportAgent


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed_diagnosed_cbc_data.csv"
DEFAULT_METRICS_CSV = PROJECT_ROOT / "output" / "federated_metrics_calib.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH),
                        help="Path to the processed CSV with features/labels.")
    parser.add_argument("--metrics_csv", type=str, default=str(DEFAULT_METRICS_CSV),
                        help="Where to append per-round metrics CSV.")
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Fraction of TRAIN used for calibration (temperature scaling).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration", action="store_true",
                        help="Enable temperature scaling per round.")
    parser.add_argument("--ece_bins", type=int, default=15)
    args = parser.parse_args()

    # Ensure output directory exists
    out_dir = Path(args.metrics_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)

    # Determinism toggles
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)

    # 1) Train the federated system (GlobalEnsemble + optional Calibration + logs CSV)
    fed = FederatedCoordinatorAgent(
        data_path=args.data,
        num_clients=args.clients,
        rounds=args.rounds,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
        metrics_csv=args.metrics_csv,
        enable_calibration=args.calibration,
        ece_bins=args.ece_bins
    )
    model, le, history = fed.run()

    # 2) Pretty-print round-by-round metrics (Week-2 column names)
    df_hist = pd.DataFrame(history, columns=[
        "round", "accuracy", "macro_f1", "macro_auroc", "macro_auprc", "ece", "brier", "temp_T"
    ])
    print("\n=== Federated Metrics per Round ===")
    print(df_hist.to_string(index=False))

    # 3) One-sample decision-support demo on a hard case (highest uncertainty)
    df_full = DataProcessingAgent(args.data).run()
    X, y = FeatureExtractionAgent().run(df_full)
    proba = model.predict_proba(X)

    # pick the hardest case by entropy of predicted probabilities
    eps = 1e-12
    ent = -np.sum(proba * np.log(proba + eps), axis=1)
    hard_idx = int(np.argmax(ent))

    pred_enc = int(np.argmax(proba[hard_idx]))
    pred_label = le.inverse_transform([pred_enc])[0]
    pred_conf = float(np.max(proba[hard_idx]))

    advisor = DecisionSupportAgent()
    advice = advisor.suggest(pred_label)

    print("\n=== Decision Support Demo (Hardest Case by Uncertainty) ===")
    print(f"Index: {hard_idx}")
    print(f"Predicted diagnosis: {pred_label}")
    print(f"Confidence (max prob): {pred_conf:.3f}")
    print(f"Recommendation: {advice}")

    # 4) Confirm where artifacts were written
    print(f"\nMetrics CSV: {Path(args.metrics_csv).resolve()}")
    print(f"Test labels: {(DEFAULT_OUTPUT_DIR / 'test_labels.npy').resolve()}")
    last_probs = DEFAULT_OUTPUT_DIR / f'probs_round_{args.rounds}.npy'
    print(f"Last-round probs: {last_probs.resolve()}")

if __name__ == "__main__":
    main()

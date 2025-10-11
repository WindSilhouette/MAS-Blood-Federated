# src/utils/plot_federated_metrics.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_CSV = PROJECT_ROOT / "federated_metrics.csv"

def main():
    df = pd.read_csv(METRICS_CSV)

    # Expect these columns from the coordinator
    required = {"round","accuracy","f1_macro","auroc_macro","auprc_macro"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {METRICS_CSV}: {missing}")

    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df["accuracy"], marker="o", label="Accuracy")
    plt.plot(df["round"], df["f1_macro"], marker="o", label="F1 (macro)")
    plt.plot(df["round"], df["auroc_macro"], marker="o", label="AUROC (macro)")
    plt.plot(df["round"], df["auprc_macro"], marker="o", label="AUPRC (macro)")

    plt.xlabel("Federation Round")
    plt.ylabel("Score")
    plt.title("Federated Convergence — Multiple Metrics")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # To save instead of show:
    # out = PROJECT_ROOT / "reports" / "figures" / "federated_convergence_multimetric.png"
    # out.parent.mkdir(parents=True, exist_ok=True)
    # plt.savefig(out, dpi=200)

if __name__ == "__main__":
    main()

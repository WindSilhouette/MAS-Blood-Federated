import numpy as np, pandas as pd, matplotlib.pyplot as plt

def reliability_diagram(y_true, probs, n_bins=15, out_png="output/reliability.png"):
    if probs.ndim > 1:
        conf = probs.max(axis=1)
        preds = probs.argmax(axis=1)
    else:
        conf = probs
        preds = (probs >= 0.5).astype(int)

    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        m, M = bins[i], bins[i + 1]
        mask = (conf > m) & (conf <= M) if i > 0 else (conf >= m) & (conf <= M)
        if np.any(mask):
            xs.append(np.mean(conf[mask]))
            ys.append(np.mean(preds[mask] == y_true[mask]))
    plt.figure()
    plt.plot([0, 1], [0, 1], "k--")
    plt.scatter(xs, ys, color="blue")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.savefig(out_png, bbox_inches="tight")

def plot_ece_trend(csv_path="output/federated_metrics_calib.csv",
                   out_png="output/ece_trend.png"):
    df = pd.read_csv(csv_path)
    if "ece" not in df.columns:
        print("No ECE column found.")
        return
    plt.figure()
    plt.plot(df["round"], df["ece"], marker="o")
    plt.xlabel("Round")
    plt.ylabel("ECE")
    plt.title("ECE vs Rounds")
    plt.savefig(out_png, bbox_inches="tight")

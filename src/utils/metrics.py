import numpy as np
from sklearn.metrics import brier_score_loss

def ece(probs, y_true, n_bins=15):
    if probs.ndim > 1:
        conf = probs.max(axis=1)
        preds = probs.argmax(axis=1)
    else:
        conf = probs
        preds = (probs >= 0.5).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val, N = 0.0, len(y_true)
    for i in range(n_bins):
        m, M = bins[i], bins[i + 1]
        mask = (conf > m) & (conf <= M) if i > 0 else (conf >= m) & (conf <= M)
        if not np.any(mask):
            continue
        acc = np.mean(preds[mask] == y_true[mask])
        avg_conf = np.mean(conf[mask])
        ece_val += (np.sum(mask) / N) * abs(acc - avg_conf)
    return float(ece_val)

def brier(probs, y_true):
    if probs.ndim == 1:
        return float(brier_score_loss(y_true, probs))
    C = probs.shape[1]
    y_oh = np.eye(C)[y_true]
    return float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)))

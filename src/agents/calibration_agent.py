import numpy as np
from dataclasses import dataclass

@dataclass
class CalibrationState:
    T: float = 1.0

class CalibrationAgent:
    """
    Temperature scaling calibration.
    Fits a scalar T using validation data, then rescales logits/probs.
    """

    def __init__(self, init_T=1.0, max_iter=200, lr=0.01, tol=1e-5):
        self.state = CalibrationState(T=init_T)
        self.max_iter, self.lr, self.tol = max_iter, lr, tol
        self.is_fitted = False
        self.num_classes = None

    @staticmethod
    def _to_logits(probs, eps=1e-12):
        probs = np.clip(probs, eps, 1 - eps)
        if probs.ndim == 1 or probs.shape[1] == 1:
            p = probs.ravel()
            return np.log(p) - np.log1p(-p)
        return np.log(probs)

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, y_val, proba_val):
        logits = self._to_logits(proba_val)
        self.num_classes = 1 if logits.ndim == 1 else logits.shape[1]

        def nll(T):
            if self.num_classes == 1:
                p = self._sigmoid(logits / T)
                p = np.clip(p, 1e-12, 1 - 1e-12)
                return -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))
            else:
                p = self._softmax(logits / T)
                p = np.clip(p, 1e-12, 1 - 1e-12)
                return -np.mean(np.log(p[np.arange(len(y_val)), y_val]))

        s = np.log(max(self.state.T, 1e-3))
        last = nll(np.exp(s))
        for _ in range(self.max_iter):
            eps = 1e-4
            g = (nll(np.exp(s + eps)) - nll(np.exp(s - eps))) / (2 * eps)
            s_new = s - self.lr * g
            cur = nll(np.exp(s_new))
            if abs(cur - last) < self.tol:
                s = s_new
                break
            s, last = s_new, cur

        self.state.T = float(np.exp(s))
        self.is_fitted = True
        return self.state.T

    def predict_proba(self, proba_uncal):
        assert self.is_fitted, "Call fit() first."
        logits = self._to_logits(proba_uncal)
        T = self.state.T
        if self.num_classes == 1:
            return self._sigmoid(logits / T)
        return self._softmax(logits / T)


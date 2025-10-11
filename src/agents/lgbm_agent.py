from pathlib import Path
from lightgbm import Booster
import numpy as np
import pickle

class LGBMAgent:
    def __init__(self, model_path: Path, encoder_path: Path):
        # Load Booster from file
        self.booster = Booster(model_file=str(model_path))
        # Load label encoder
        with open(encoder_path, "rb") as f:
            self.le = pickle.load(f)

    def run(self, X):
        """
        Predict probabilities, take argmax, and return decoded labels.
        """
        proba = self.booster.predict(X)  # shape (n_samples, n_classes)
        preds_enc = np.argmax(proba, axis=1)
        return self.le.inverse_transform(preds_enc)

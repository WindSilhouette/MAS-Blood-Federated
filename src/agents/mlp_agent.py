from pathlib import Path
from tensorflow.keras.models import load_model
import pickle

class MLPAagent:
    def __init__(self,
                 model_path: Path,
                 encoder_path: Path,
                 scaler_path: Path):
        # Load trained Keras model
        self.model = load_model(str(model_path))
        # Load label encoder
        with open(encoder_path, "rb") as f:
            self.le = pickle.load(f)
        # Load scaler
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def run(self, X):
        """
        Scale X, predict probabilities, argmax, and decode labels.
        """
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict(X_scaled)
        preds_enc = proba.argmax(axis=1)
        return self.le.inverse_transform(preds_enc)

from pathlib import Path
from xgboost import XGBClassifier
import pickle

class XGBAgent:
    def __init__(self, model_path: Path, encoder_path: Path):
        # Load trained model
        self.model = XGBClassifier()
        self.model.load_model(str(model_path))
        # Load label encoder
        with open(encoder_path, "rb") as f:
            self.le = pickle.load(f)

    def run(self, X):
        """
        Predict and return decoded labels for X.
        """
        preds_enc = self.model.predict(X)
        return self.le.inverse_transform(preds_enc)

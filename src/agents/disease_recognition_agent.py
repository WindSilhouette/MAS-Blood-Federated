from xgboost import XGBClassifier
import pickle

class DiseaseRecognitionAgent:
    def __init__(self,
                 model_path,
                 encoder_path,
                 model_type: str = "xgb"):
        """
        model_type: 'xgb', 'lgbm', 'rf', or 'mlp'
        """
        self.model_type = model_type
        if model_type == "xgb":
            self.model = XGBClassifier(use_label_encoder=False)
            self.model.load_model(str(model_path))
        else:
            # you can extend to other types here...
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        with open(encoder_path, "rb") as f:
            self.le = pickle.load(f)

    def run(self, X):
        preds_enc = self.model.predict(X)
        return self.le.inverse_transform(preds_enc)

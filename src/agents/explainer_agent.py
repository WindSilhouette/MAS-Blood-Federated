import shap
import numpy as np
import pandas as pd
import os

class ExplainerAgent:
    def __init__(self, model, model_type="mlp", background_data=None, output_dir="outputs/shap"):
        os.makedirs(output_dir, exist_ok=True)
        self.model = model
        self.model_type = model_type
        self.output_dir = output_dir

        if background_data is not None:
            self.background = background_data
        else:
            self.background = None

        self.explainer = self._init_explainer()

    def _init_explainer(self):
        if self.model_type in ["xgboost", "rf", "tree"]:
            return shap.TreeExplainer(self.model)
        elif self.model_type == "mlp":
            return shap.Explainer(self.model, self.background)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def explain_global(self, X, feature_names=None, top_k=10):
        shap_values = self.explainer(X)
        importance = np.abs(shap_values.values).mean(axis=0)
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame({"feature": feature_names, "importance": importance})
        df = df.sort_values("importance", ascending=False).head(top_k)
        df.to_csv(os.path.join(self.output_dir, "global_importance.csv"), index=False)
        return df

    def explain_local(self, X, idx, feature_names=None):
        shap_values = self.explainer(X)
        sv = shap_values[idx]
        shap.plots.waterfall(sv, show=False)
        file_path = os.path.join(self.output_dir, f"local_case_{idx}.png")
        shap.plots.waterfall(sv, show=False)
        import matplotlib.pyplot as plt
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        return file_path

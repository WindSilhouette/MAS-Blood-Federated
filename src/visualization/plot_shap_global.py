import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_shap_global(csv_path="outputs/shap/global_importance.csv", save_path=None):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8,6))
    plt.barh(df["feature"], df["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Mean(|SHAP value|)")
    plt.title("Global Feature Importance")
    if save_path is None:
        save_path = os.path.join(os.path.dirname(csv_path), "shap_global.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved global SHAP plot to {save_path}")

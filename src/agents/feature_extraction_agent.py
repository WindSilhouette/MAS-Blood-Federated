import pandas as pd

class FeatureExtractionAgent:
    def __init__(self, target_col: str = "Diagnosis"):
        self.target_col = target_col

    def run(self, df: pd.DataFrame):
        """
        Split DataFrame into feature matrix X and target series y.
        """
        feature_cols = [c for c in df.columns if c != self.target_col]
        X = df[feature_cols]
        y = df[self.target_col]
        return X, y

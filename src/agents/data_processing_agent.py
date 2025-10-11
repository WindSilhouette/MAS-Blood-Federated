import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer

class DataProcessingAgent:
    def __init__(self, data_path: Path):
        self.data_path = data_path

    def run(self) -> pd.DataFrame:
        """
        Load CSV, impute numeric columns by mean, and return cleaned DataFrame.
        """
        df = pd.read_csv(self.data_path)
        num_cols = df.select_dtypes(include=['int64','float64']).columns
        imputer = SimpleImputer(strategy='mean')
        df[num_cols] = imputer.fit_transform(df[num_cols])
        return df

# src/agents/explore_and_prepare.py

import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Compute project root and data file path
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed_diagnosed_cbc_data.csv"

def load_and_clean(path):
    """
    Load a CSV from `path` into a DataFrame and impute missing numeric values by column mean.
    """
    df = pd.read_csv(path)
    # Identify all numeric columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df

def prepare_data(df, target_col='Diagnosis'):
    """
    Split DataFrame into feature matrix X and target vector y.
    All columns except `target_col` are treated as features.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]
    return X, y

def main():
    # 1) Load and clean
    print(f"Loading data from: {DATA_FILE}")
    df = load_and_clean(DATA_FILE)
    print(f"Data shape after cleaning: {df.shape}")

    # 2) Prepare features and target
    X, y = prepare_data(df)  # uses default target_col='Diagnosis'
    print(f"Feature matrix X: {X.shape}; Target vector y: {y.shape}")

    # 3) Quick train/test split check
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train/Test sizes: {X_train.shape[0]} / {X_test.shape[0]} samples")

if __name__ == '__main__':
    main()

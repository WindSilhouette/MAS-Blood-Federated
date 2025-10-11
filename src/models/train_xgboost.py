# src/models/train_xgboost.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path
import pickle

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Correct import path
from src.explore_and_prepare import DATA_FILE, load_and_clean, prepare_data

def main():
    # 1) Load & clean
    print(f"Loading and cleaning data from {DATA_FILE}")
    df = load_and_clean(DATA_FILE)

    # 2) Prepare features X and raw target y (strings)
    X, y = prepare_data(df, target_col='Diagnosis')
    print(f"Data shapes — X: {X.shape}, y: {y.shape}")

    # 3) Encode string labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes found: {le.classes_.tolist()}")

    # Save the encoder for later
    le_path = Path(__file__).resolve().parent / "label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    print(f"LabelEncoder saved to: {le_path}")

    # 4) Split into train/test sets (preserve class balance)
    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    print(f"Train/Test sizes: {X_train.shape[0]} / {X_test.shape[0]} samples")

    # 5) Initialize XGBoost with multi-class settings
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )

    # 6) Train
    print("Training XGBoost model...")
    model.fit(X_train, y_train_enc)

    # 7) Predict & decode back to string labels
    y_pred_enc = model.predict(X_test)
    y_test = le.inverse_transform(y_test_enc)
    y_pred = le.inverse_transform(y_pred_enc)

    # 8) Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 9) Save model
    model_path = Path(__file__).resolve().parent / "xgboost_cbc_model.json"
    model.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()


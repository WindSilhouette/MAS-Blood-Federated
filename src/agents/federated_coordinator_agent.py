# src/agents/federated_coordinator_agent.py

import os
import csv
import numpy as np
from dataclasses import dataclass
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from .data_processing_agent import DataProcessingAgent
from .feature_extraction_agent import FeatureExtractionAgent
from .calibration_agent import CalibrationAgent
from src.utils.metrics import ece, brier


@dataclass
class GlobalEnsemble:
    """Averages class probabilities from client models (soft voting)."""
    models: list          # fitted client models (must expose predict_proba / predict)
    classes_: np.ndarray  # label encoder classes (for consistency)

    def predict_proba(self, X):
        probs = [m.predict_proba(X) for m in self.models]
        return np.mean(np.stack(probs, axis=0), axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class FederatedCoordinatorAgent:
    def __init__(self,
                 data_path,
                 num_clients: int = 3,
                 rounds: int = 5,
                 test_size: float = 0.2,
                 val_size: float = 0.2,
                 random_state: int = 42,
                 metrics_csv: str = "federated_metrics.csv",
                 enable_calibration: bool = True,
                 ece_bins: int = 15):
        """
        data_path: path to CSV
        num_clients: how many simulated clients
        rounds: federated aggregation rounds
        test_size: fraction held out as global test set
        val_size: fraction of TRAIN reserved for validation (for calibration)
        random_state: seed for reproducibility
        metrics_csv: where to append per-round metrics
        enable_calibration: if True, fit temperature per round on val set
        ece_bins: number of bins for ECE computation
        """
        self.data_path        = data_path
        self.num_clients      = num_clients
        self.rounds           = rounds
        self.test_size        = test_size
        self.val_size         = val_size
        self.random_state     = random_state
        self.metrics_csv_path = metrics_csv
        self.enable_calib     = enable_calibration
        self.ece_bins         = ece_bins

    @staticmethod
    def _multiclass_metrics(y_true_enc, proba, n_classes):
        y_pred_enc = np.argmax(proba, axis=1)
        acc  = accuracy_score(y_true_enc, y_pred_enc)
        f1m  = f1_score(y_true_enc, y_pred_enc, average='macro')

        y_bin = label_binarize(y_true_enc, classes=list(range(n_classes)))
        try:
            auroc = roc_auc_score(y_bin, proba, average='macro', multi_class='ovr')
        except ValueError:
            auroc = np.nan
        auprc = average_precision_score(y_bin, proba, average='macro')
        return acc, f1m, auroc, auprc

    def run(self):
        rng = np.random.RandomState(self.random_state)

        # 1) Load & preprocess full dataset
        df = DataProcessingAgent(self.data_path).run()
        X, y = FeatureExtractionAgent().run(df)

        # 2) Encode labels
        le    = LabelEncoder()
        y_enc = le.fit_transform(y)
        n_classes = len(le.classes_)

        # 3) Split into global train/test
        X_train_all, X_test, y_train_all, y_test = train_test_split(
            X, y_enc,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_enc
        )

        # 3b) Split TRAIN into train/val (val used ONLY for calibration)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_all, y_train_all,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_train_all
        )

        # 4) Split training data into client shards
        idx = np.arange(len(y_train))
        rng.shuffle(idx)
        shards = np.array_split(idx, self.num_clients)

        # 5) Initialize one XGBoost per client
        client_models = [
            XGBClassifier(
                objective='multi:softprob',
                num_class=n_classes,
                eval_metric='mlogloss',
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state
            )
            for _ in range(self.num_clients)
        ]

        # Prepare metrics CSV (Week 2 columns)
        os.makedirs(os.path.dirname(self.metrics_csv_path) or ".", exist_ok=True)
        csv_exists = os.path.exists(self.metrics_csv_path)
        with open(self.metrics_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "round",
                    "accuracy",
                    "macro_f1",
                    "macro_auroc",
                    "macro_auprc",
                    "ece",
                    "brier",
                    "temp_T"
                ]
            )
            if not csv_exists:
                writer.writeheader()

        # 6) Federated rounds with ensemble aggregation, calibration, and per-round evaluation
        history = []
        global_model = None

        for r in range(self.rounds):
            print(f"--- Round {r+1}/{self.rounds} ---")

            # Local training on each client's shard
            for i, model in enumerate(client_models):
                xi = X_train.iloc[shards[i]]
                yi = y_train[shards[i]]
                model.fit(xi, yi)

            # Global aggregation via probability averaging (soft voting)
            global_model = GlobalEnsemble(models=client_models, classes_=le.classes_)

            # --- Uncalibrated predictions ---
            proba_val_uncal  = global_model.predict_proba(X_val)
            proba_test_uncal = global_model.predict_proba(X_test)

            # --- Optional temperature scaling (fit on VAL, apply to TEST) ---
            if self.enable_calib:
                calib = CalibrationAgent()
                T = calib.fit(y_val, proba_val_uncal)
                proba_val  = calib.predict_proba(proba_val_uncal)
                proba_test = calib.predict_proba(proba_test_uncal)
            else:
                T = 1.0
                proba_val  = proba_val_uncal
                proba_test = proba_test_uncal

            # Evaluate on the global test set (after calibration if enabled)
            acc, f1m, auroc, auprc = self._multiclass_metrics(
                y_true_enc=y_test,
                proba=proba_test,
                n_classes=n_classes
            )

            # Reliability metrics
            ece_val   = ece(proba_test, y_test, n_bins=self.ece_bins)
            brier_val = brier(proba_test, y_test)

            row = {
                "round": r + 1,
                "accuracy": acc,
                "macro_f1": f1m,
                "macro_auroc": auroc,
                "macro_auprc": auprc,
                "ece": ece_val,
                "brier": brier_val,
                "temp_T": float(T),
            }
            history.append(row)

            # Append to CSV
            with open(self.metrics_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "round",
                        "accuracy",
                        "macro_f1",
                        "macro_auroc",
                        "macro_auprc",
                        "ece",
                        "brier",
                        "temp_T"
                    ]
                )
                writer.writerow(row)

        return global_model, le, history



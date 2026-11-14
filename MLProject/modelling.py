"""
modelling.py (Workflow-CI)
Re-training model Credit Scoring melalui MLflow Project.
Dijalankan dengan:  mlflow run MLProject --env-manager=local
Tracking URI diambil dari environment variable MLFLOW_TRACKING_URI
(diatur oleh GitHub Actions), default ke folder lokal ./mlruns.
"""

import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

TARGET = "default"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="credit_preprocessing")
    return parser.parse_args()


def main():
    args = parse_args()
    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

    # autolog untuk parameter & metrik training; model di-log manual
    # agar memiliki signature + input example (dibutuhkan saat serving).
    mlflow.sklearn.autolog(log_models=False)

    # Saat dijalankan via `mlflow run`, start_run() melanjutkan run milik project
    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mlflow.log_metrics(
            {
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_f1_weighted": f1_score(y_test, y_pred, average="weighted"),
                "test_roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
            }
        )

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.head(5),
        )
        print(f"run_id: {run.info.run_id}")
        print(f"test accuracy: {accuracy_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "datasets"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "performance"

DATASET_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


DATASET_URLS = {
    "diabetes": "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    "heart": "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv",
    "kidney": "https://raw.githubusercontent.com/Mohamed2821/Chronic-Kidney-Disease-project/main/kidney_disease.csv",
}


def download_csv(name: str, url: str) -> Path:
    out = DATASET_DIR / f"{name}.csv"
    df = pd.read_csv(url)
    df.to_csv(out, index=False)
    return out


def preprocess_diabetes(df: pd.DataFrame):
    target = "Outcome"
    invalid_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in invalid_zeros:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    return X, y


def preprocess_heart(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    target = "target" if "target" in df.columns else "output"
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    return X, y


def preprocess_kidney(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    target_candidates = ["classification", "class", "target", "Outcome"]
    target = None
    for cand in target_candidates:
        if cand in df.columns:
            target = cand
            break
    if target is None:
        raise ValueError("Kidney target column not found.")

    y = (
        df[target]
        .astype(str)
        .str.lower()
        .replace({"ckd": 1, "notckd": 0, "yes": 1, "no": 0, "1": 1, "0": 0})
    )
    y = pd.to_numeric(y, errors="coerce")

    X = df.drop(columns=[target])
    keep_cols = []
    for col in X.columns:
        numeric_col = pd.to_numeric(X[col], errors="coerce")
        if numeric_col.notna().mean() >= 0.4:
            X[col] = numeric_col
            keep_cols.append(col)
    X = X[keep_cols]

    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].astype(int).copy()
    return X, y


def build_preprocess(numeric_columns):
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        ]
    )


def train_select_calibrate(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocess = build_preprocess(X.columns.tolist())
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best_name, best_auc = None, -1.0
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["roc_auc"])
        auc = float(scores["test_roc_auc"].mean())
        if auc > best_auc:
            best_auc = auc
            best_name = name

    best_pipe = Pipeline([("preprocess", preprocess), ("model", models[best_name])])
    calibrated = CalibratedClassifierCV(estimator=best_pipe, cv=5, method="sigmoid")
    calibrated.fit(X_train, y_train)

    test_proba = calibrated.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
    metrics = {
        "model": best_name,
        "roc_auc": float(roc_auc_score(y_test, test_proba)),
        "pr_auc": float(average_precision_score(y_test, test_proba)),
        "f1": float(f1_score(y_test, test_pred)),
        "precision": float(precision_score(y_test, test_pred)),
        "recall": float(recall_score(y_test, test_pred)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": 0.50,
        "n_test": int(len(y_test)),
    }
    return calibrated, metrics


def main():
    downloaded_paths = {}
    for name, url in DATASET_URLS.items():
        try:
            downloaded_paths[name] = download_csv(name, url)
            print(f"[OK] Downloaded {name}: {downloaded_paths[name]}")
        except Exception as exc:
            print(f"[WARN] Could not download {name}: {exc}")

    preprocessors = {
        "diabetes": preprocess_diabetes,
        "heart": preprocess_heart,
        "kidney": preprocess_kidney,
    }

    performance_rows = []
    model_index = {}

    for disease, file_path in downloaded_paths.items():
        try:
            df = pd.read_csv(file_path)
            X, y = preprocessors[disease](df)
            model, metrics = train_select_calibrate(X, y)

            model_file = MODEL_DIR / f"{disease}_model.joblib"
            joblib.dump(model, model_file)

            metrics["disease"] = disease
            metrics["dataset_file"] = str(file_path)
            metrics["model_file"] = str(model_file)
            performance_rows.append(metrics)
            model_index[disease] = {
                "dataset_file": str(file_path),
                "model_file": str(model_file),
                "selected_model": metrics["model"],
                "threshold": metrics["threshold"],
            }
            print(f"[OK] Trained + saved model for {disease}")
        except Exception as exc:
            print(f"[WARN] Failed pipeline for {disease}: {exc}")

    if performance_rows:
        perf_df = pd.DataFrame(performance_rows)[
            [
                "disease",
                "model",
                "roc_auc",
                "pr_auc",
                "f1",
                "precision",
                "recall",
                "threshold",
                "tp",
                "tn",
                "fp",
                "fn",
                "n_test",
                "dataset_file",
                "model_file",
            ]
        ]
        perf_csv = REPORT_DIR / "performance_index.csv"
        perf_json = REPORT_DIR / "performance_index.json"
        perf_df.to_csv(perf_csv, index=False)
        perf_json.write_text(json.dumps(performance_rows, indent=2), encoding="utf-8")
        print(f"[OK] Wrote performance CSV: {perf_csv}")
        print(f"[OK] Wrote performance JSON: {perf_json}")

    model_index_file = REPORT_DIR / "model_index.json"
    model_index_file.write_text(json.dumps(model_index, indent=2), encoding="utf-8")
    print(f"[OK] Wrote model index: {model_index_file}")


if __name__ == "__main__":
    main()

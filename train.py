"""One-off training script: train_collab.ipynb sections 4-10, ported to data/loans.csv.

Run once to create models/model.joblib and models/results.joblib. The Streamlit app only
loads these files and never retrains.

Leakage controls:
  * stratified train/test split happens BEFORE imputation, encoding and SMOTE
  * imputer values and OneHotEncoder are fitted on the training split only
  * SMOTE is applied to the training split only (and inside each CV training fold)
  * the test set is never resampled
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from pipeline import (
    CAT_COLS, ID_COLS, RANDOM_STATE, RAW_FEATURE_COLS, TARGET_COL, TEST_SIZE,
    clean, encode_and_merge, engineer_features, fit_imputer, impute, make_explainer, shap_explain,
)

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "loans.csv"
MODEL_DIR = ROOT / "models"


def build_models():
    # Same model set and hyperparameters as the notebook. LogisticRegression gets a StandardScaler:
    # unscaled (income in the millions) it fails to converge after 1000 lbfgs iterations.
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "xgboost": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
    }


def fit_preprocess(X_train):
    fill = fit_imputer(X_train)
    Xt = engineer_features(impute(X_train, fill))
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(Xt[CAT_COLS])
    return fill, enc


def transform(X, fill, enc):
    return encode_and_merge(engineer_features(impute(X, fill)), enc)


def score(y_true, pred, prob):
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "avg_precision": average_precision_score(y_true, prob),
    }


def cross_validate(X_train, y_train, n_splits=5):
    """Leak-free CV on the training split: preprocessing + SMOTE refit inside every fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (tr, va) in enumerate(skf.split(X_train, y_train), 1):
        Xtr, Xva = X_train.iloc[tr], X_train.iloc[va]
        ytr, yva = y_train.iloc[tr], y_train.iloc[va]
        fill, enc = fit_preprocess(Xtr)
        Xtr_e, Xva_e = transform(Xtr, fill, enc), transform(Xva, fill, enc)
        Xtr_b, ytr_b = SMOTE(random_state=RANDOM_STATE).fit_resample(Xtr_e, ytr)
        for name, m in build_models().items():
            m = clone(m).fit(Xtr_b, ytr_b)
            prob = m.predict_proba(Xva_e)[:, 1]
            rows.append({"model": name, "fold": fold, **score(yva, (prob >= 0.5).astype(int), prob)})
    return pd.DataFrame(rows)


def main():
    raw = pd.read_csv(DATA_PATH)
    df = clean(raw)
    print(f"Loaded {raw.shape}, after cleaning {df.shape}")

    X = df[RAW_FEATURE_COLS]
    y = df[TARGET_COL].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train {X_train.shape}, Test {X_test.shape}")

    fill_values, encoder = fit_preprocess(X_train)
    X_train_enc = transform(X_train, fill_values, encoder)
    X_test_enc = transform(X_test, fill_values, encoder)
    feature_names = X_train_enc.columns.tolist()

    X_train_bal, y_train_bal = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_enc, y_train)
    print(f"SMOTE (train only): {X_train_enc.shape} -> {X_train_bal.shape}")

    models = build_models()
    metrics, eval_detail = {}, {}
    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        prob = model.predict_proba(X_test_enc)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics[name] = score(y_test, pred, prob)
        fpr, tpr, _ = roc_curve(y_test, prob)
        prec, rec, _ = precision_recall_curve(y_test, prob)
        eval_detail[name] = {
            "y_prob": prob, "y_pred": pred,
            "confusion_matrix": confusion_matrix(y_test, pred),
            "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
            "roc": (fpr, tpr), "pr": (prec, rec),
        }
        print(f"\n=== {name} ===\n{metrics[name]}")

    best_name = max(metrics, key=lambda k: metrics[k]["roc_auc"])
    best_model = models[best_name]
    print(f"\nBest model (by test ROC-AUC, as in notebook): {best_name}")

    cv = cross_validate(X_train, y_train)
    print("\n5-fold CV (train split) mean:\n", cv.groupby("model").mean(numeric_only=True).drop(columns="fold"))

    # SHAP (same explainer choice as notebook)
    shap_background = X_train_bal.sample(min(100, len(X_train_bal)), random_state=RANDOM_STATE)
    explainer = make_explainer(best_model, shap_background)
    values, base = shap_explain(explainer, best_model, X_test_enc)
    model_type = type(best_model).__name__
    shap_results = {
        "values": values, "base_values": base, "data": X_test_enc.reset_index(drop=True),
        "feature_names": feature_names, "model_type": model_type,
    }

    artifact = {
        "model": best_model,
        "models": models,
        "encoder": encoder,
        "fill_values": fill_values,
        "feature_names": feature_names,
        "cat_cols": CAT_COLS,
        "raw_feature_cols": RAW_FEATURE_COLS,
        "categories": {c: list(cats) for c, cats in zip(CAT_COLS, encoder.categories_)},
        "metrics": metrics,
        "best_model_name": best_name,
        "shap_background": shap_background,
        "threshold": 0.5,
    }
    results = {
        "eval": eval_detail,
        "cv": cv,
        "shap": shap_results,
        "y_test": y_test.to_numpy(),
        "X_test_raw": X_test.reset_index(drop=True),
        "test_ids": df.loc[X_test.index, ID_COLS].reset_index(drop=True),
        "split": {
            "n_raw": len(raw), "n_clean": len(df),
            "n_train": len(X_train), "n_test": len(X_test),
            "train_dist": y_train.value_counts().sort_index().to_dict(),
            "test_dist": y_test.value_counts().sort_index().to_dict(),
            "train_bal_dist": pd.Series(y_train_bal).value_counts().sort_index().to_dict(),
        },
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_DIR / "model.joblib")
    joblib.dump(results, MODEL_DIR / "results.joblib")
    print(f"\nSaved {MODEL_DIR / 'model.joblib'} and {MODEL_DIR / 'results.joblib'}")


if __name__ == "__main__":
    main()

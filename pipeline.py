"""Shared preprocessing for training (train.py) and inference (app.py).

Mirrors train_collab.ipynb, adapted to the loans.csv schema:
clean -> engineer features -> one-hot encode (encoder fitted on train only).
SMOTE is applied in train.py to the training split only and never at inference.
"""
import numpy as np
import pandas as pd

TARGET_COL = "Defaulted"
ID_COLS = ["Loan_ID", "Customer"]
NUM_INPUT_COLS = ["Loan_Amount", "Tenure_Months", "Interest_Rate", "Annual_Income", "Credit_Score"]
CAT_COLS = ["Has_Collateral", "Sector"]
RAW_FEATURE_COLS = NUM_INPUT_COLS + CAT_COLS
ENGINEERED_COLS = ["loan_to_income", "payment_to_income"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Dedup + outlier guards (training data only; inference rows are never dropped)."""
    df = df.drop_duplicates().copy()
    df = df[(df["Annual_Income"] > 0) & (df["Loan_Amount"] > 0) & (df["Tenure_Months"] > 0)]
    df = df[df["Credit_Score"].between(300, 900)]
    return df.reset_index(drop=True)


def fit_imputer(X_train: pd.DataFrame) -> dict:
    """Median/mode fill values learned from the training split only."""
    fill = {c: float(X_train[c].median()) for c in NUM_INPUT_COLS}
    fill.update({c: X_train[c].mode().iloc[0] for c in CAT_COLS})
    return fill


def impute(df: pd.DataFrame, fill_values: dict) -> pd.DataFrame:
    df = df.copy()
    for c in NUM_INPUT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.fillna(fill_values)


def monthly_payment(principal, annual_rate_pct, months):
    r = np.asarray(annual_rate_pct, dtype=float) / 1200.0
    n = np.asarray(months, dtype=float)
    p = np.asarray(principal, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pay = np.where(r > 0, p * r / (1 - (1 + r) ** (-n)), p / n)
    return pay


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features, analogous to the notebook's loan_to_income."""
    df = df.copy()
    income = df["Annual_Income"].replace(0, 1)
    df["loan_to_income"] = df["Loan_Amount"] / income
    df["payment_to_income"] = (
        monthly_payment(df["Loan_Amount"], df["Interest_Rate"], df["Tenure_Months"]) * 12 / income
    )
    return df


def encode_and_merge(df: pd.DataFrame, encoder, cat_cols=CAT_COLS) -> pd.DataFrame:
    """One-hot encode categoricals and merge with numeric features (same as notebook)."""
    encoded = encoder.transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
    return pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)


def prepare_features(raw: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Raw applicant rows -> model-ready matrix, using objects fitted at training time."""
    X = raw[RAW_FEATURE_COLS].copy()
    for c in CAT_COLS:
        s = X[c].astype("string").str.strip().str.title()
        s = s.mask(s.isin(["", "Nan", "None"]))
        X[c] = s.astype(object).where(s.notna(), np.nan)
    X = impute(X, artifact["fill_values"])
    X = engineer_features(X)
    X = encode_and_merge(X, artifact["encoder"])
    return X[artifact["feature_names"]]


def make_explainer(model, background: pd.DataFrame):
    """Explainer choice follows the notebook: Tree for RF/XGB, Linear for logistic regression."""
    import shap

    if hasattr(model, "named_steps"):  # Pipeline(StandardScaler, LogisticRegression)
        scaler, est = model.named_steps["scaler"], model.named_steps["clf"]
        bg = pd.DataFrame(scaler.transform(background), columns=background.columns)
        return shap.LinearExplainer(est, bg)
    if type(model).__name__ in ("RandomForestClassifier", "XGBClassifier"):
        return shap.TreeExplainer(model)
    return shap.LinearExplainer(model, background)


def shap_explain(explainer, model, X: pd.DataFrame):
    """SHAP values for the default class (1) -> (values[n, f], base_values[n]). Units: model margin."""
    Xin = X
    if hasattr(model, "named_steps"):
        Xin = pd.DataFrame(model.named_steps["scaler"].transform(X), columns=X.columns, index=X.index)
    sv = explainer(Xin)
    values, base = sv.values, np.asarray(sv.base_values)
    if values.ndim == 3:
        values, base = values[:, :, 1], base[:, 1]
    base = np.broadcast_to(base, (len(X),)).astype(float)
    return values, base


def score_batch(batch: pd.DataFrame, artifact: dict, model_name: str):
    """Validate + score an uploaded batch. Returns (scored_df, warnings). Raises ValueError on bad schema."""
    batch = batch.copy()
    batch.columns = batch.columns.str.strip()
    missing = [c for c in RAW_FEATURE_COLS if c not in batch.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if batch.empty:
        raise ValueError("The file has no rows.")

    warnings = []
    check = batch[RAW_FEATURE_COLS].copy()
    for c in NUM_INPUT_COLS:
        check[c] = pd.to_numeric(check[c], errors="coerce")
    n_imputed = int(check.isna().any(axis=1).sum())
    if n_imputed:
        warnings.append(f"{n_imputed} row(s) had missing or non-numeric values. They were filled with "
                        "the training-set median/mode.")
    for c in CAT_COLS:
        seen = set(batch[c].dropna().astype(str).str.strip().str.title()) - {"", "Nan", "None"}
        unknown = sorted(seen - set(artifact["categories"][c]))
        if unknown:
            warnings.append(f"Unseen {c} values {unknown}: encoded as all-zero (no category effect).")

    X = prepare_features(batch, artifact)
    prob = artifact["models"][model_name].predict_proba(X)[:, 1]
    pred = (prob >= artifact["threshold"]).astype(int)
    out = batch.copy()
    out["Default_Probability"] = prob.round(4)
    out["Prediction"] = pred
    out["Prediction_Label"] = np.where(pred == 1, "Defaulted", "Repaid")
    return out, warnings

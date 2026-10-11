"""Tests: preprocessing parity with training, real (non-hardcoded) predictions, and every app page.

Run: .venv/bin/python -m pytest -q
"""
import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import RAW_FEATURE_COLS, prepare_features  # noqa: E402

ART = joblib.load(ROOT / "models" / "model.joblib")
RES = joblib.load(ROOT / "models" / "results.joblib")
DF = pd.read_csv(ROOT / "data" / "loans.csv")


def test_inference_matches_training_probabilities():
    """App-side preprocessing on raw test rows reproduces the probabilities stored at training time."""
    X = prepare_features(RES["X_test_raw"], ART)
    assert list(X.columns) == ART["feature_names"]
    for name, model in ART["models"].items():
        np.testing.assert_allclose(model.predict_proba(X)[:, 1], RES["eval"][name]["y_prob"], atol=1e-9)


def test_predictions_respond_to_inputs():
    base = DF[RAW_FEATURE_COLS].iloc[[0]].copy()
    good, bad = base.copy(), base.copy()
    good[["Credit_Score", "Annual_Income", "Has_Collateral"]] = [840, 2_400_000, "Yes"]
    bad[["Credit_Score", "Annual_Income", "Has_Collateral"]] = [310, 180_000, "No"]
    p = ART["model"].predict_proba(prepare_features(pd.concat([good, bad]), ART))[:, 1]
    assert p[0] < p[1]
    assert len(np.unique(ART["model"].predict_proba(prepare_features(DF, ART))[:, 1].round(6))) > 50


def test_messy_batch_input_is_handled():
    raw = DF[RAW_FEATURE_COLS].head(4).copy().astype(object)
    raw.loc[0, "Sector"] = "  trade "
    raw.loc[1, "Credit_Score"] = "abc"
    raw.loc[2, "Has_Collateral"] = None
    raw.loc[3, "Sector"] = "Mining"  # unseen category
    X = prepare_features(raw, ART)
    assert not X.isna().any().any()
    assert X.loc[0, "Sector_Trade"] == 1
    assert X.loc[1, "Credit_Score"] == ART["fill_values"]["Credit_Score"]
    assert X.filter(like="Sector_").iloc[3].sum() == 0


def test_smote_only_on_training_split():
    sp = RES["split"]
    assert sp["n_train"] + sp["n_test"] == sp["n_clean"]
    assert sum(sp["test_dist"].values()) == sp["n_test"]  # test set not resampled
    assert sp["train_bal_dist"][0] == sp["train_bal_dist"][1]


# ------------------------------------------------------------- Streamlit UI (AppTest)
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ["Dataset Dashboard", "EDA & Feature Engineering", "Model Evaluation",
         "SHAP Explainability", "Loan Prediction", "Batch Prediction"]


def _app(page):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = _app(page)
    assert at.title[0].value == page


def test_single_prediction_form():
    at = _app("Loan Prediction")
    at.number_input[0].set_value(5_000_000)  # loan amount
    at.slider[0].set_value(320)  # credit score
    at.button[0].click().run()
    assert not at.exception, at.exception
    txt = " ".join(e.value for e in at.error) + " ".join(s.value for s in at.success)
    assert "Probability of default" in txt


def test_model_choice_changes_prediction():
    probs = []
    for model in ART["models"]:
        at = _app("Loan Prediction")
        sb = [s for s in at.selectbox if s.label == "Model"][0]
        sb.set_value(model)
        at.button[0].click().run()
        txt = " ".join(e.value for e in at.error) + " ".join(s.value for s in at.success)
        probs.append(txt.split("**")[1])
    assert len(set(probs)) > 1


def test_batch_scoring_end_to_end():
    from pipeline import score_batch

    csv = DF.drop(columns="Defaulted").head(20).to_csv(index=False)
    out, notes = score_batch(pd.read_csv(io.StringIO(csv)), ART, ART["best_model_name"])
    assert len(out) == 20 and not notes
    assert out["Default_Probability"].between(0, 1).all()
    ref = ART["model"].predict_proba(prepare_features(DF.head(20), ART))[:, 1].round(4)
    np.testing.assert_allclose(out["Default_Probability"], ref)
    assert {"Loan_ID", "Prediction", "Prediction_Label"} <= set(out.columns)
    out.to_csv(io.StringIO(), index=False)  # downloadable

    with pytest.raises(ValueError, match="Missing required columns: Credit_Score"):
        score_batch(DF.drop(columns="Credit_Score"), ART, "logreg")
    bad = DF.head(3).copy().astype(object)
    bad.loc[0, "Interest_Rate"] = "n/a"
    bad.loc[1, "Sector"] = "Mining"
    _, notes = score_batch(bad, ART, "xgboost")
    assert len(notes) == 2

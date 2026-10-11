# Loan Default Prediction: Streamlit app

Streamlit dashboard for `data/loans.csv`, built on the pipeline from `train_collab.ipynb`
(clean → feature engineering → stratified split → train-only one-hot encoding → train-only SMOTE →
Logistic Regression / Random Forest / XGBoost → SHAP).

| File | Purpose |
|---|---|
| `pipeline.py` | Shared preprocessing used by both training and the app (guarantees identical transforms) |
| `train.py` | One-off training; writes `models/model.joblib` + `models/results.joblib` |
| `app.py` | Streamlit app; loads the saved model, never retrains |
| `tests/test_app.py` | Parity, leakage, prediction and UI tests |

## Run

```bash
cd ~/Downloads/loan_prediction_app
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python train.py          # only if models/ is missing or the data changes
.venv/bin/streamlit run app.py
.venv/bin/pip install pytest && .venv/bin/python -m pytest -q   # optional tests
```

Batch CSV columns: `Loan_Amount, Tenure_Months, Interest_Rate, Annual_Income, Credit_Score,
Has_Collateral, Sector` (optional: `Loan_ID`, `Customer`, `Defaulted`).

## Differences from the notebook
- Column names adapted to `loans.csv` (`Defaulted` target; ID columns dropped).
- Engineered features: `loan_to_income` (as in the notebook) and `payment_to_income`.
- Logistic Regression wrapped with `StandardScaler`: unscaled it failed to converge.
- Imputation values learned from the training split only (the notebook imputed before splitting).
- Added leak-free 5-fold CV on the training split for more stable estimates than the 60-row test set.

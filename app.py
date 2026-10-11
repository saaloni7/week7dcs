"""Loan Default Prediction dashboard.

Loads the model and evaluation results produced once by train.py; never retrains.
All inference goes through pipeline.prepare_features, the same transforms used in training.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from pipeline import (
    CAT_COLS, ENGINEERED_COLS, ID_COLS, NUM_INPUT_COLS, RAW_FEATURE_COLS, TARGET_COL,
    clean, engineer_features, make_explainer, prepare_features, score_batch, shap_explain,
)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "loans.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
RESULTS_PATH = ROOT / "models" / "results.joblib"

# Reference data-viz palette (validated categorical slots 1-3, blue sequential, blue<->red diverging)
BLUE, ORANGE, AQUA, RED, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#f0efec"
CLASS_LABELS = {0: "Repaid", 1: "Defaulted"}
CLASS_COLORS = {"Repaid": BLUE, "Defaulted": ORANGE}
MODEL_LABELS = {"logreg": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}
MODEL_COLORS = {"Logistic Regression": BLUE, "Random Forest": ORANGE, "XGBoost": AQUA}
SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0, BLUE], [0.5, NEUTRAL], [1, RED]]
METRIC_LABELS = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "f1": "F1-score", "roc_auc": "ROC-AUC", "avg_precision": "Avg. precision (PR-AUC)",
}

st.set_page_config(page_title="Loan Default Prediction", page_icon="🏦", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(RESULTS_PATH)


@st.cache_resource
def get_explainer(model_name):
    art, _ = load_artifacts()
    return make_explainer(art["models"][model_name], art["shap_background"])


def style(fig, height=360, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=80 if legend else 50, b=10), showlegend=legend,
        title=dict(y=0.98, yanchor="top", x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title=None),
        bargap=0.25, hoverlabel=dict(namelength=-1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=0.5)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch")


def with_label(df):
    df = df.copy()
    df["Outcome"] = pd.Categorical(df[TARGET_COL].map(CLASS_LABELS), categories=["Repaid", "Defaulted"])
    return df


def margin_unit(model_type):
    return "probability" if model_type == "RandomForestClassifier" else "log-odds"


def local_contrib_fig(shap_row, X_row, raw_row, unit, title, top=10):
    """Per-applicant SHAP bars; one-hot columns are summed back into their source feature (SHAP is additive)."""
    contrib = {}
    for f, v in zip(X_row.index, shap_row):
        src = next((c for c in CAT_COLS if f.startswith(c + "_")), None)
        if src:
            key = f"{src} = {raw_row[src]}"
            contrib[key] = contrib.get(key, 0.0) + v
        else:
            val = X_row[f]
            contrib[f"{f} = {val:,.2f}" if f in ENGINEERED_COLS or f == "Interest_Rate" else f"{f} = {val:,.0f}"] = v
    c = pd.Series(contrib)
    c = c[c.abs() > 1e-9].sort_values(key=abs).tail(top)
    d = pd.DataFrame({"feature": c.index, "shap": c.values})
    d["Effect"] = np.where(d["shap"] > 0, "Raises risk", "Lowers risk")
    fig = px.bar(d, x="shap", y="feature", orientation="h", color="Effect",
                 color_discrete_map={"Raises risk": RED, "Lowers risk": BLUE},
                 labels={"shap": f"SHAP ({unit})", "feature": ""}, title=title)
    fig.update_yaxes(categoryorder="array", categoryarray=d["feature"].tolist())
    fig.add_vline(x=0, line_width=1, line_color="gray")
    return style(fig, height=400)


def predict(raw, art, model_name):
    X = prepare_features(raw, art)
    model = art["models"][model_name]
    prob = model.predict_proba(X)[:, 1]
    return X, prob, (prob >= art["threshold"]).astype(int)


if not (MODEL_PATH.exists() and RESULTS_PATH.exists()):
    st.error("Model files not found. Run `python train.py` once to create models/model.joblib.")
    st.stop()

df = load_data()
art, res = load_artifacts()
best = art["best_model_name"]

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🏦 Loan Default Prediction")
page = st.sidebar.radio(
    "Navigate",
    ["Dataset Dashboard", "EDA & Feature Engineering", "Model Evaluation",
     "SHAP Explainability", "Loan Prediction", "Batch Prediction"],
)
st.sidebar.divider()
st.sidebar.markdown(f"**Production model:** {MODEL_LABELS[best]}")
m = art["metrics"][best]
st.sidebar.markdown(f"Test ROC-AUC **{m['roc_auc']:.3f}** · F1 **{m['f1']:.3f}**")
st.sidebar.caption(
    f"Trained on {res['split']['n_train']} loans (SMOTE on the training split only), "
    f"evaluated on {res['split']['n_test']} held-out loans. Decision threshold {art['threshold']}."
)


# ================================================================ 1. DATASET
if page == "Dataset Dashboard":
    st.title("Dataset Dashboard")
    st.caption(f"Source: `{DATA_PATH.relative_to(ROOT)}`")

    c = st.columns(5)
    c[0].metric("Rows", f"{len(df):,}")
    c[1].metric("Columns", df.shape[1])
    c[2].metric("Missing cells", int(df.isna().sum().sum()))
    c[3].metric("Duplicate rows", int(df.duplicated().sum()))
    c[4].metric("Default rate", f"{df[TARGET_COL].mean():.1%}")

    tab_data, tab_types, tab_stats = st.tabs(["Data preview", "Types & missing values", "Summary statistics"])
    with tab_data:
        f1, f2 = st.columns(2)
        sectors = f1.multiselect("Sector", sorted(df["Sector"].unique()), default=sorted(df["Sector"].unique()))
        coll = f2.multiselect("Has collateral", sorted(df["Has_Collateral"].unique()),
                              default=sorted(df["Has_Collateral"].unique()))
        view = df[df["Sector"].isin(sectors) & df["Has_Collateral"].isin(coll)]
        st.caption(f"{len(view)} of {len(df)} rows")
        st.dataframe(view, width="stretch", height=380, hide_index=True)
    with tab_types:
        info = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "non-null": df.notna().sum(),
            "missing": df.isna().sum(),
            "missing %": (df.isna().mean() * 100).round(2),
            "unique values": df.nunique(),
        })
        st.dataframe(info, width="stretch")
        st.caption(f"Duplicate rows: {int(df.duplicated().sum())} · duplicate Loan_IDs: "
                   f"{int(df['Loan_ID'].duplicated().sum())}")
    with tab_stats:
        st.dataframe(df.describe().T.round(2), width="stretch")
        st.dataframe(df.select_dtypes(exclude="number").describe().T, width="stretch")

    st.subheader("Interactive exploration")
    num_options = NUM_INPUT_COLS
    a, b = st.columns(2)
    with a:
        col = st.selectbox("Numeric feature", num_options, index=4)
        fig = px.histogram(with_label(df), x=col, color="Outcome", barmode="group",
                           color_discrete_map=CLASS_COLORS, title=f"{col} by outcome")
        show(style(fig))
    with b:
        cat = st.selectbox("Categorical feature", CAT_COLS + ["Tenure_Months"], index=1)
        rate = df.groupby(cat)[TARGET_COL].agg(["mean", "count"]).reset_index()
        rate["mean"] *= 100
        fig = px.bar(rate, x=cat, y="mean", text=rate["mean"].map("{:.0f}%".format),
                     hover_data={"count": True}, color_discrete_sequence=[ORANGE],
                     labels={"mean": "Default rate (%)"}, title=f"Default rate by {cat}")
        fig.update_xaxes(type="category")
        show(style(fig, legend=False))

    x1, x2 = st.columns(2)
    xs = x1.selectbox("X axis", num_options, index=4)
    ys = x2.selectbox("Y axis", num_options, index=2)
    fig = px.scatter(with_label(df), x=xs, y=ys, color="Outcome", color_discrete_map=CLASS_COLORS,
                     hover_data=["Loan_ID", "Sector", "Has_Collateral"], title=f"{ys} vs {xs}")
    fig.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=1, color="white")))
    show(style(fig, height=420))


# ================================================================ 2. EDA & FE
elif page == "EDA & Feature Engineering":
    st.title("EDA & Feature Engineering")
    dfl = with_label(df)

    st.subheader("Target distribution")
    a, b = st.columns([1, 2])
    counts = dfl["Outcome"].value_counts().reindex(["Repaid", "Defaulted"]).reset_index()
    fig = px.bar(counts, x="Outcome", y="count", color="Outcome", color_discrete_map=CLASS_COLORS,
                 text="count", title="Loans by outcome")
    a.plotly_chart(style(fig, legend=False), width="stretch")
    with b:
        st.markdown(
            f"- **{counts.loc[0, 'count']}** repaid vs **{counts.loc[1, 'count']}** defaulted "
            f"({df[TARGET_COL].mean():.1%} default rate): moderately imbalanced.\n"
            "- The notebook handles imbalance with **SMOTE applied to the training split only**, "
            "after the train/test split, so no synthetic rows leak into evaluation."
        )
        sp = res["split"]
        rows = [("Train (original)", sp["train_dist"]), ("Train (after SMOTE)", sp["train_bal_dist"]),
                ("Test (untouched)", sp["test_dist"])]
        smote = pd.DataFrame([{"Set": s, "Outcome": CLASS_LABELS[int(k)], "Count": v}
                              for s, d in rows for k, v in d.items()])
        fig = px.bar(smote, x="Set", y="Count", color="Outcome", barmode="group", text="Count",
                     color_discrete_map=CLASS_COLORS, title="Class balance: split and SMOTE")
        show(style(fig, height=300))

    st.subheader("Numerical distributions")
    col = st.selectbox("Feature", NUM_INPUT_COLS, index=4, key="eda_num")
    a, b = st.columns(2)
    fig = px.box(dfl, x="Outcome", y=col, color="Outcome", color_discrete_map=CLASS_COLORS,
                 points="all", title=f"{col} by outcome")
    a.plotly_chart(style(fig, legend=False), width="stretch")
    rate = df.groupby(pd.qcut(df[col], 5, duplicates="drop"), observed=True)[TARGET_COL].agg(["mean", "count"])
    rate.index = rate.index.astype(str)
    rate = rate.reset_index().rename(columns={col: "Bin"})
    fig = px.bar(rate, x="Bin", y=rate["mean"] * 100, hover_data={"count": True},
                 color_discrete_sequence=[ORANGE], labels={"y": "Default rate (%)"},
                 title=f"Default rate by {col} quantile bin")
    b.plotly_chart(style(fig, legend=False), width="stretch")

    st.subheader("Categorical features vs outcome")
    a, b = st.columns(2)
    for slot, cat in zip((a, b), CAT_COLS):
        ct = dfl.groupby([cat, "Outcome"]).size().reset_index(name="count")
        fig = px.bar(ct, x=cat, y="count", color="Outcome", barmode="group",
                     color_discrete_map=CLASS_COLORS, title=f"{cat} by outcome")
        slot.plotly_chart(style(fig), width="stretch")

    st.subheader("Feature engineering")
    st.markdown(
        "Engineered features, computed by `pipeline.engineer_features()` and used identically at "
        "training and prediction time:\n\n"
        "| Feature | Formula | Meaning |\n|---|---|---|\n"
        "| `loan_to_income` | Loan_Amount / Annual_Income | Debt size relative to income (as in the notebook) |\n"
        "| `payment_to_income` | 12 × amortised monthly payment / Annual_Income | Share of yearly income spent on repayments |\n\n"
        f"Categoricals `{CAT_COLS[0]}` and `{CAT_COLS[1]}` are one-hot encoded with an encoder fitted on the "
        f"training split only. Identifier columns ({', '.join(ID_COLS)}) are dropped. "
        f"Final model input: **{len(art['feature_names'])} features**."
    )
    fe = with_label(engineer_features(df))
    a, b = st.columns(2)
    for slot, feat in zip((a, b), ENGINEERED_COLS):
        fig = px.box(fe, x="Outcome", y=feat, color="Outcome", color_discrete_map=CLASS_COLORS,
                     points="all", log_y=True, title=f"{feat} by outcome (log scale)")
        slot.plotly_chart(style(fig, legend=False), width="stretch")

    st.subheader("Correlation matrix")
    corr = fe[NUM_INPUT_COLS + ENGINEERED_COLS + [TARGET_COL]].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=DIVERGING, zmin=-1, zmax=1, aspect="auto")
    show(style(fig, height=480, legend=False))
    tc = corr[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
    st.caption("Strongest linear association with default: " +
               ", ".join(f"{k} ({v:+.2f})" for k, v in tc.head(3).items()))

    st.subheader("Data cleaning summary")
    cleaned = clean(df)
    st.markdown(
        f"- Rows: {len(df)} → {len(cleaned)} after removing duplicates and invalid values "
        "(non-positive income/amount/tenure, credit score outside 300–900).\n"
        f"- Missing values in source: {int(df.isna().sum().sum())}. Median/mode fill values are learned "
        "from the training split and reused for incomplete prediction inputs."
    )


# ================================================================ 3. EVALUATION
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.caption(f"Held-out test set: {res['split']['n_test']} loans, never resampled. "
               "Models trained on the SMOTE-balanced training split. Best model chosen by test ROC-AUC, as in the notebook.")

    mt = pd.DataFrame(art["metrics"]).T.rename(index=MODEL_LABELS, columns=METRIC_LABELS)
    st.subheader("Test-set metrics")
    st.dataframe(mt.style.format("{:.3f}").highlight_max(axis=0, props="font-weight:bold;"), width="stretch")

    long = mt.reset_index(names="Model").melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group",
                 color_discrete_map=MODEL_COLORS, range_y=[0, 1], title="Model comparison")
    show(style(fig, height=380))

    st.subheader("5-fold cross-validation (training split)")
    st.caption("Imputation, encoding and SMOTE are refit inside every fold, so the scores are leak-free. "
               "They're steadier than a single 60-row test set.")
    cv = res["cv"].copy()
    cv["model"] = cv["model"].map(MODEL_LABELS)
    agg = cv.drop(columns="fold").groupby("model").agg(["mean", "std"])
    tbl = pd.DataFrame({METRIC_LABELS[k]: agg[k]["mean"].map("{:.3f}".format) + " ± " + agg[k]["std"].map("{:.3f}".format)
                        for k in METRIC_LABELS})
    st.dataframe(tbl, width="stretch")

    st.subheader("Per-model diagnostics")
    name = st.selectbox("Model", list(art["models"]), index=list(art["models"]).index(best),
                        format_func=lambda k: MODEL_LABELS[k] + (" (best)" if k == best else ""))
    ev = res["eval"][name]
    y_test = res["y_test"]
    a, b = st.columns(2)
    thr = a.slider("Decision threshold", 0.05, 0.95, float(art["threshold"]), 0.05,
                   help="Metrics are recomputed from the model's stored test-set probabilities.")
    pred = (ev["y_prob"] >= thr).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    k = a.columns(4)
    k[0].metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")
    k[1].metric("Precision", f"{precision_score(y_test, pred, zero_division=0):.3f}")
    k[2].metric("Recall", f"{recall_score(y_test, pred, zero_division=0):.3f}")
    k[3].metric("F1", f"{f1_score(y_test, pred, zero_division=0):.3f}")
    fig = px.imshow(cm, text_auto=True, color_continuous_scale=SEQ_BLUE,
                    x=["Pred: Repaid", "Pred: Defaulted"], y=["Actual: Repaid", "Actual: Defaulted"],
                    title=f"Confusion matrix @ threshold {thr:.2f}")
    fig.update_coloraxes(showscale=False)
    a.plotly_chart(style(fig, height=340, legend=False), width="stretch")

    roc = go.Figure()
    for n, e in res["eval"].items():
        fpr, tpr = e["roc"]
        lbl = MODEL_LABELS[n]
        roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(width=2, color=MODEL_COLORS[lbl]),
                                 name=f"{lbl} (AUC {art['metrics'][n]['roc_auc']:.3f})"))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", width=1, color="gray"),
                             name="Chance", hoverinfo="skip"))
    roc.update_layout(title="ROC curves (test set)", xaxis_title="False positive rate", yaxis_title="True positive rate")
    b.plotly_chart(style(roc, height=340), width="stretch")

    pr = go.Figure()
    for n, e in res["eval"].items():
        prec, rec = e["pr"]
        lbl = MODEL_LABELS[n]
        pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", line=dict(width=2, color=MODEL_COLORS[lbl]),
                                name=f"{lbl} (AP {art['metrics'][n]['avg_precision']:.3f})"))
    pr.update_layout(title="Precision-recall curves (test set)", xaxis_title="Recall", yaxis_title="Precision")
    b.plotly_chart(style(pr, height=340), width="stretch")

    st.markdown("**Classification report** (threshold 0.5)")
    rep = pd.DataFrame(ev["classification_report"]).T.rename(index={"0": "Repaid (0)", "1": "Defaulted (1)"})
    st.dataframe(rep.style.format("{:.3f}"), width="stretch")


# ================================================================ 4. SHAP
elif page == "SHAP Explainability":
    sh = res["shap"]
    unit = margin_unit(sh["model_type"])
    st.title("SHAP Explainability")
    st.caption(f"{MODEL_LABELS[best]} on the {len(sh['data'])} held-out test loans. "
               f"SHAP values are contributions to the predicted default {unit}: positive pushes toward default.")
    vals = pd.DataFrame(sh["values"], columns=sh["feature_names"])
    data = sh["data"]
    imp = vals.abs().mean().sort_values()

    # Direction: does a higher feature value raise or lower risk?
    direction = {}
    for f in vals.columns:
        x = data[f].astype(float)
        direction[f] = np.sign(np.corrcoef(x, vals[f])[0, 1]) if x.std() > 0 and vals[f].std() > 0 else 0.0

    a, b = st.columns(2)
    fig = px.bar(x=imp.values, y=imp.index, orientation="h", color_discrete_sequence=[BLUE],
                 labels={"x": f"mean |SHAP| ({unit})", "y": ""}, title="Global feature importance")
    a.plotly_chart(style(fig, height=460, legend=False), width="stretch")

    order = imp.index.tolist()
    rows = []
    rng = np.random.default_rng(0)
    for f in order:
        x = data[f].astype(float)
        norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0
        rows.append(pd.DataFrame({"feature": f, "shap": vals[f], "value": data[f],
                                  "norm": norm, "y": order.index(f) + rng.uniform(-0.25, 0.25, len(x))}))
    bee = pd.concat(rows)
    fig = go.Figure(go.Scatter(
        x=bee["shap"], y=bee["y"], mode="markers",
        marker=dict(size=8, color=bee["norm"], colorscale=SEQ_BLUE, line=dict(width=1, color="white"),
                    colorbar=dict(title="Feature value", tickvals=[0, 1], ticktext=["Low", "High"])),
        customdata=np.c_[bee["feature"], bee["value"]],
        hovertemplate="%{customdata[0]} = %{customdata[1]}<br>SHAP %{x:.3f}<extra></extra>",
    ))
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=order)
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_layout(title="SHAP summary (beeswarm)", xaxis_title=f"SHAP value ({unit})")
    b.plotly_chart(style(fig, height=460, legend=False), width="stretch")

    st.subheader("What drives predictions")
    lines = []
    for f in imp.index[::-1][:6]:
        d = direction[f]
        if f.startswith(tuple(c + "_" for c in CAT_COLS)):
            effect = "raises" if d > 0 else "lowers"
            lines.append(f"- **{f}**: being in this category {effect} predicted default risk "
                         f"(mean |SHAP| {imp[f]:.3f}).")
        else:
            effect = "higher risk" if d > 0 else "lower risk"
            lines.append(f"- **{f}**: higher values → {effect} (mean |SHAP| {imp[f]:.3f}).")
    st.markdown("Computed from the SHAP values above, ranked by importance:\n" + "\n".join(lines))

    st.subheader("Dependence plot")
    feat = st.selectbox("Feature", order[::-1])
    color_by = st.selectbox("Colour by", ["Actual outcome"] + [c for c in order[::-1] if c != feat])
    dep = pd.DataFrame({feat: data[feat], "SHAP": vals[feat]})
    if color_by == "Actual outcome":
        dep["Outcome"] = pd.Series(res["y_test"]).map(CLASS_LABELS)
        fig = px.scatter(dep, x=feat, y="SHAP", color="Outcome", color_discrete_map=CLASS_COLORS)
    else:
        dep[color_by] = data[color_by]
        fig = px.scatter(dep, x=feat, y="SHAP", color=color_by, color_continuous_scale=SEQ_BLUE)
    fig.update_traces(marker=dict(size=9, line=dict(width=1, color="white")))
    fig.add_hline(y=0, line_width=1, line_color="gray")
    show(style(fig, height=380))

    st.subheader("Explain a single test-set loan")
    ids = res["test_ids"]["Loan_ID"].tolist()
    i = ids.index(st.selectbox("Loan_ID", ids))
    prob = res["eval"][best]["y_prob"][i]
    c = st.columns(3)
    c[0].metric("Predicted default probability", f"{prob:.1%}")
    c[1].metric("Actual outcome", CLASS_LABELS[int(res["y_test"][i])])
    c[2].metric(f"Base value ({unit})", f"{sh['base_values'][i]:.3f}")
    raw_row = res["X_test_raw"].iloc[i].copy()
    for c_ in CAT_COLS:
        raw_row[c_] = str(raw_row[c_]).strip().title()
    show(local_contrib_fig(sh["values"][i], data.iloc[i], raw_row, unit, "Top contributions"))
    st.dataframe(res["X_test_raw"].iloc[[i]], hide_index=True, width="stretch")


# ================================================================ 5. SINGLE PREDICTION
elif page == "Loan Prediction":
    st.title("Loan Prediction")
    st.caption("Enter applicant details. The input goes through the same cleaning, feature engineering and "
               "encoding as training, then the saved model scores it.")
    rng_ = {c: (df[c].min(), df[c].max()) for c in NUM_INPUT_COLS}
    with st.form("applicant"):
        a, b, c = st.columns(3)
        loan = a.number_input("Loan amount", min_value=1_000, value=int(df["Loan_Amount"].median()), step=50_000)
        tenure = a.selectbox("Tenure (months)", sorted(df["Tenure_Months"].unique()), index=2)
        rate = b.number_input("Interest rate (%)", min_value=0.0, max_value=40.0,
                              value=float(round(df["Interest_Rate"].median(), 2)), step=0.25)
        income = b.number_input("Annual income", min_value=1_000, value=int(df["Annual_Income"].median()), step=60_000)
        score = c.slider("Credit score", 300, 900, int(df["Credit_Score"].median()))
        collateral = c.radio("Has collateral", art["categories"]["Has_Collateral"], horizontal=True)
        sector = c.selectbox("Sector", art["categories"]["Sector"])
        model_name = st.selectbox("Model", list(art["models"]), index=list(art["models"]).index(best),
                                  format_func=lambda k: MODEL_LABELS[k] + (" (best)" if k == best else ""))
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        raw = pd.DataFrame([{
            "Loan_Amount": loan, "Tenure_Months": tenure, "Interest_Rate": rate, "Annual_Income": income,
            "Credit_Score": score, "Has_Collateral": collateral, "Sector": sector,
        }])
        out = [k for k, v in raw.iloc[0][NUM_INPUT_COLS].items() if not rng_[k][0] <= v <= rng_[k][1]]
        if out:
            st.warning("Outside the training data range (the prediction is an extrapolation): " +
                       ", ".join(f"{k} (seen {rng_[k][0]:,}–{rng_[k][1]:,})" for k in out))
        X, prob, pred = predict(raw, art, model_name)
        p = float(prob[0])
        st.divider()
        a, b = st.columns([1, 2])
        with a:
            if pred[0] == 1:
                st.error(f"### ⚠️ Likely to DEFAULT\nProbability of default: **{p:.1%}**")
            else:
                st.success(f"### ✅ Likely to REPAY\nProbability of default: **{p:.1%}**")
            st.progress(p, text=f"Default probability {p:.1%} (threshold {art['threshold']:.0%})")
            fe = engineer_features(raw)
            st.markdown(f"- Loan-to-income: **{fe['loan_to_income'][0]:.2f}**\n"
                        f"- Repayments / income: **{fe['payment_to_income'][0]:.1%}**\n"
                        f"- Model: {MODEL_LABELS[model_name]}")
        with b:
            vals, base = shap_explain(get_explainer(model_name), art["models"][model_name], X)
            unit = margin_unit(type(art["models"][model_name]).__name__)
            show(local_contrib_fig(vals[0], X.iloc[0], raw.iloc[0], unit, "Why this prediction (SHAP contributions)"))
        with st.expander("Model input after preprocessing"):
            st.dataframe(X, hide_index=True, width="stretch")


# ================================================================ 6. BATCH
elif page == "Batch Prediction":
    st.title("Batch Prediction")
    st.markdown(f"Upload a CSV with columns: `{'`, `'.join(RAW_FEATURE_COLS)}`. "
                f"Optional: `Loan_ID`, `Customer`, and `{TARGET_COL}` (if present, accuracy against it is reported).")
    template = df[ID_COLS + RAW_FEATURE_COLS].head(5)
    st.download_button("Download template CSV", template.to_csv(index=False), "loan_template.csv", "text/csv")

    model_name = st.selectbox("Model", list(art["models"]), index=list(art["models"]).index(best),
                              format_func=lambda k: MODEL_LABELS[k] + (" (best)" if k == best else ""))
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            batch = pd.read_csv(up)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read CSV: {e}")
            st.stop()
        try:
            out, notes = score_batch(batch, art, model_name)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        for n in notes:
            st.warning(n)
        out["Model"] = MODEL_LABELS[model_name]
        prob, pred = out["Default_Probability"].to_numpy(), out["Prediction"].to_numpy()

        c = st.columns(4)
        c[0].metric("Applications", len(out))
        c[1].metric("Predicted defaults", int(pred.sum()))
        c[2].metric("Predicted default rate", f"{pred.mean():.1%}")
        c[3].metric("Mean default probability", f"{prob.mean():.1%}")
        if TARGET_COL in batch.columns and batch[TARGET_COL].notna().all():
            yt = batch[TARGET_COL].astype(int)
            st.info(f"Against the provided `{TARGET_COL}` column: accuracy {accuracy_score(yt, pred):.3f}, "
                    f"F1 {f1_score(yt, pred, zero_division=0):.3f}. Rows from loans.csv were partly used "
                    "for training, so this can be optimistic.")

        a, b = st.columns(2)
        fig = px.histogram(out, x="Default_Probability", nbins=20, color="Prediction_Label",
                           color_discrete_map=CLASS_COLORS, range_x=[0, 1], title="Default probability distribution",
                           category_orders={"Prediction_Label": ["Repaid", "Defaulted"]})
        fig.add_vline(x=art["threshold"], line_dash="dot", line_color="gray")
        a.plotly_chart(style(fig), width="stretch")
        if "Sector" in out:
            s = out.groupby("Sector")["Prediction"].mean().mul(100).reset_index()
            fig = px.bar(s, x="Sector", y="Prediction", color_discrete_sequence=[ORANGE],
                         labels={"Prediction": "Predicted default rate (%)"}, title="Predicted default rate by sector")
            b.plotly_chart(style(fig, legend=False), width="stretch")

        st.dataframe(out, width="stretch", hide_index=True, height=400)
        st.download_button("Download predictions CSV", out.to_csv(index=False), "loan_predictions.csv",
                           "text/csv", type="primary")

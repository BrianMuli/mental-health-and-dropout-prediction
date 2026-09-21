"""
Student Dropout Risk — prediction app
Loads models/xgb_dropout_relevant_cols.pkl + the matching metadata json
(the two files your training notebook already writes) and serves live
predictions through a custom-styled interface.

Run with:
    streamlit run streamlit_app.py
from the same working directory that contains your `models/` folder.
"""

import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Student Dropout Risk",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = "models"
MODEL_STEM = "xgb_dropout_relevant_cols"
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_STEM}.pkl")
META_PATH = os.path.join(MODEL_DIR, f"{MODEL_STEM}.json")

# ----------------------------------------------------------------------
# Styling — custom theme, not the default Streamlit look
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --paper: #0E2A1E;
        --ink: #F5F7F5;
        --ink-soft: #A9C4B4;
        --line: #2E5943;
        --risk-high: #FF8F6B;
        --risk-high-bg: #4A2A20;
        --risk-medium: #E8C066;
        --risk-medium-bg: #453C1E;
        --risk-low: #8FD9AE;
        --risk-low-bg: #1F4A34;
        --card: #153A29;
    }

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        color: var(--ink) !important;
    }

    .stApp {
        background: var(--paper) !important;
    }

    /* Force readable colors on widget internals regardless of the
       browser/OS light-dark setting Streamlit auto-detects */
    p, span, label, div, li {
        color: var(--ink) !important;
    }
    .stMarkdown, .stMarkdown p {
        color: var(--ink) !important;
    }
    input, textarea, select {
        background: #1F4634 !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
    }
    [data-baseweb="input"], [data-baseweb="select"], [data-baseweb="base-input"] {
        background: #1F4634 !important;
    }
    [data-baseweb="input"] input, [data-baseweb="select"] div {
        color: var(--ink) !important;
    }
    [data-testid="stForm"] {
        background: transparent !important;
    }
    [data-testid="stNumberInput"] label, [data-testid="stSelectbox"] label {
        color: var(--ink-soft) !important;
    }

    h1, h2, h3 {
        font-family: 'Source Serif 4', serif;
        color: var(--ink);
        letter-spacing: -0.01em;
    }

    .app-header {
        padding: 2.2rem 0 1.4rem 0;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1.8rem;
    }
    .app-header h1 {
        font-size: 2.1rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .app-header p {
        color: var(--ink-soft);
        font-size: 1.02rem;
        margin: 0;
    }

    .card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 1.6rem 1.8rem;
    }

    .verdict-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .verdict-high {
        background: var(--risk-high-bg);
        color: var(--risk-high);
    }
    .verdict-medium {
        background: var(--risk-medium-bg);
        color: var(--risk-medium);
    }
    .verdict-low {
        background: var(--risk-low-bg);
        color: var(--risk-low);
    }

    .metric-label {
        font-size: 0.82rem;
        color: var(--ink-soft);
        margin-bottom: 0.1rem;
    }
    .metric-value {
        font-family: 'Source Serif 4', serif;
        font-size: 1.6rem;
        font-weight: 600;
    }

    .stButton>button {
        background: #8FD9AE;
        color: #0E2A1E !important;
        border-radius: 7px;
        border: none;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: #A9E6C1;
        color: #0E2A1E !important;
    }
    .stButton>button p {
        color: #0E2A1E !important;
    }

    section[data-testid="stSidebar"] {
        background: #0A2018;
        border-right: 1px solid var(--line);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Load model + metadata
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta


model, meta = load_artifacts()

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>Student Dropout Risk</h1>
        <p>Enter a student's academic indicators to estimate dropout risk from the trained XGBoost model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        f"Couldn't find `{MODEL_PATH}` and/or `{META_PATH}`. "
        "Run this app from the same directory your notebook saved the "
        "`models/` folder into, or retrain and re-save first."
    )
    st.stop()

features = meta.get("features", [])
best_threshold = float(meta.get("best_threshold", 0.5))
LOW_MEDIUM_CUTOFF = 0.50
MEDIUM_HIGH_CUTOFF = 0.80

# ----------------------------------------------------------------------
# Sidebar — model info
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Model details")
    st.markdown(f"**Algorithm:** {meta.get('algorithm', '—')}")
    st.markdown(f"**Target:** `{meta.get('target', '—')}`")
    st.markdown(f"**Model's own threshold:** {best_threshold:.3f}")
    st.markdown(f"**Risk bands:** 0–{LOW_MEDIUM_CUTOFF:.0%} low · {LOW_MEDIUM_CUTOFF:.0%}–{MEDIUM_HIGH_CUTOFF:.0%} medium · {MEDIUM_HIGH_CUTOFF:.0%}–100% high")
    st.markdown(f"**Test AUC:** {meta.get('test_auc', float('nan')):.3f}")
    st.markdown(f"**Train AUC:** {meta.get('train_auc', float('nan')):.3f}")
    trained = meta.get("date_trained", "—")
    st.markdown(f"**Trained:** {trained}")

# ----------------------------------------------------------------------
# Input form — built dynamically from the model's own feature list,
# so it always matches whatever columns the model was actually trained on
# ----------------------------------------------------------------------
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown("#### Student indicators")
    with st.form("prediction_form"):
        raw_inputs = {}
        cols = st.columns(2)
        for i, feat in enumerate(features):
            with cols[i % 2]:
                raw_inputs[feat] = st.text_input(
                    feat.replace("_", " ").title(),
                    value="0",
                    key=f"input_{feat}",
                )
        submitted = st.form_submit_button("Predict risk")

    # Parse whatever was typed into floats, exactly as entered
    # (no forced decimal padding — "2.9" stays "2.9").
    inputs = {}
    parse_error = None
    if submitted:
        for feat, raw in raw_inputs.items():
            try:
                inputs[feat] = float(raw.strip())
            except ValueError:
                parse_error = feat
                break

with right:
    st.markdown("#### Result")
    result_slot = st.container()

    if not submitted:
        with result_slot:
            st.markdown(
                '<div class="card"><span class="metric-label">'
                "Fill in the form and click Predict risk to see a result here."
                "</span></div>",
                unsafe_allow_html=True,
            )
    elif parse_error:
        with result_slot:
            st.error(
                f"'{raw_inputs[parse_error]}' isn't a valid number for "
                f"**{parse_error.replace('_', ' ').title()}**. Use digits and "
                "at most one decimal point, e.g. 2.9."
            )
    else:
        row = pd.DataFrame([inputs])[features]
        proba = float(model.predict_proba(row)[:, 1][0])

        if proba >= MEDIUM_HIGH_CUTOFF:
            verdict_class, verdict_text = "verdict-high", "High risk"
        elif proba >= LOW_MEDIUM_CUTOFF:
            verdict_class, verdict_text = "verdict-medium", "Medium risk"
        else:
            verdict_class, verdict_text = "verdict-low", "Low risk"

        with result_slot:
            st.markdown(
                f"""
                <div class="card">
                    <span class="verdict-badge {verdict_class}">{verdict_text}</span>
                    <div style="margin-top:1.1rem;">
                        <div class="metric-label">Predicted dropout probability</div>
                        <div class="metric-value">{proba:.1%}</div>
                    </div>
                    <div style="margin-top:0.9rem; display:flex; gap:1.8rem;">
                        <div>
                            <div class="metric-label">Medium/high boundary</div>
                            <div class="metric-value" style="font-size:1.1rem;">{MEDIUM_HIGH_CUTOFF:.0%}</div>
                        </div>
                        <div>
                            <div class="metric-label">Low/medium boundary</div>
                            <div class="metric-value" style="font-size:1.1rem;">{LOW_MEDIUM_CUTOFF:.0%}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(proba, 0.0), 1.0))
            st.caption(
                f"Prediction made {datetime.now().strftime('%Y-%m-%d %H:%M')} "
                f"using `{meta.get('model_file', MODEL_STEM)}`."
            )

st.markdown("---")
st.caption(
    "This tool supports early identification and does not replace advisor judgment. "
    "Treat borderline probabilities as a prompt for follow-up, not a final decision."
)

"""
AI-Based Behavioral Ransomware Detection Dashboard
CIC-MalMem2022 | Decision Tree | 52 Behavioral Features
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_dataframe, predict_sample

# ─── paths ───────────────────────────────────────────────────────────────────
METADATA_PATH    = PROJECT_ROOT / "models" / "best_model_metadata.json"
DATASET_PATH     = PROJECT_ROOT / "dataset" / "Obfuscated-MalMem2022.csv"
SAMPLE_CSV_PATH  = PROJECT_ROOT / "dataset" / "sample_prediction.csv"
IMG_PATH         = PROJECT_ROOT / "images"
FEAT_IMP_PATH    = PROJECT_ROOT / "reports" / "final_feature_importance.csv"
MODEL_RESULTS    = PROJECT_ROOT / "reports" / "model_results.csv"
CV_RESULTS       = PROJECT_ROOT / "reports" / "cross_validation_results.csv"
ALT_RESULTS      = PROJECT_ROOT / "reports" / "alternative_split_results.csv"
EXPERIMENT       = PROJECT_ROOT / "reports" / "experiment_summary.json"

# ─── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Behavioral Ransomware Detection",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117 !important;
    color: #c9d1d9 !important;
}
[data-testid="stSidebar"] {
    background: #010409 !important;
    border-right: 1px solid #21262d !important;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* ── hide default streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.4rem !important; max-width: 1340px; }

/* ── dark header bar (removes white strip) ── */
header[data-testid="stHeader"] {
    background-color: #0d1117 !important;
    border-bottom: 1px solid #21262d !important;
}
/* hide deploy button only — keep sidebar toggle visible */
[data-testid="stToolbar"] {
    background-color: #0d1117 !important;
}
[data-testid="stToolbarActionButtonIcon"],
[data-testid="stDecoration"],
.stDeployButton,
button[title="Deploy"],
a[href*="share.streamlit.io"] {
    display: none !important;
}
/* keep the sidebar collapse/expand arrow visible and themed */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    z-index: 9999 !important;
    background-color: #161b22 !important;
    border: 1px solid #3fb950 !important;
    border-radius: 0 6px 6px 0 !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg {
    fill: #3fb950 !important;
}

/* ── force sidebar toggle button always visible ── */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    z-index: 9999 !important;
    background-color: #161b22 !important;
    border: 1px solid #3fb950 !important;
    border-radius: 0 6px 6px 0 !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg {
    fill: #3fb950 !important;
    color: #3fb950 !important;
}

/* ── typography ── */
h1,h2,h3 { color: #e6edf3 !important; }
.stMarkdown p { color: #8b949e; }

/* ── nav pill ── */
.nav-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #3fb950 !important;
    margin-bottom: 0.35rem; padding-left: 4px;
}

/* ── cards ── */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 1.1rem 1.2rem;
}
.card-danger  { border-color: #f85149 !important; }
.card-success { border-color: #3fb950 !important; }
.card-warn    { border-color: #d29922 !important; }

/* ── metric tiles ── */
.mtile {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 0.85rem 1rem;
    text-align: center;
}
.mtile-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #8b949e;
}
.mtile-value {
    font-size: 1.75rem; font-weight: 700; color: #e6edf3;
    margin-top: 0.2rem; font-family: 'Courier New', monospace;
}
.mtile-sub { font-size: 0.72rem; color: #3fb950; margin-top: 0.15rem; }

/* ── status badge ── */
.badge-green {
    display:inline-block; background:#0d2818; border:1px solid #3fb950;
    color:#3fb950; border-radius:4px; padding:2px 10px;
    font-size:0.72rem; font-weight:700; letter-spacing:0.08em;
}
.badge-red {
    display:inline-block; background:#2d0f0f; border:1px solid #f85149;
    color:#f85149; border-radius:4px; padding:2px 10px;
    font-size:0.72rem; font-weight:700; letter-spacing:0.08em;
}
.badge-yellow {
    display:inline-block; background:#2d2008; border:1px solid #d29922;
    color:#d29922; border-radius:4px; padding:2px 10px;
    font-size:0.72rem; font-weight:700; letter-spacing:0.08em;
}
.badge-blue {
    display:inline-block; background:#0c1a2e; border:1px solid #58a6ff;
    color:#58a6ff; border-radius:4px; padding:2px 10px;
    font-size:0.72rem; font-weight:700; letter-spacing:0.08em;
}

/* ── section header ── */
.sec-head {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #3fb950;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.4rem; margin-bottom: 0.9rem; margin-top: 0.3rem;
}

/* ── result banner ── */
.result-malware {
    background: linear-gradient(135deg,#2d0f0f,#1a0a0a);
    border: 1px solid #f85149; border-radius: 8px;
    padding: 1.6rem 2rem; text-align: center;
}
.result-benign {
    background: linear-gradient(135deg,#0d2818,#091a10);
    border: 1px solid #3fb950; border-radius: 8px;
    padding: 1.6rem 2rem; text-align: center;
}
.result-title { font-size: 1.6rem; font-weight: 800; letter-spacing: 0.06em; }
.result-sub   { font-size: 0.9rem; color: #8b949e; margin-top: 0.4rem; }

/* ── pipeline steps ── */
.pipe-step {
    background: #161b22; border: 1px solid #21262d; border-radius: 4px;
    padding: 0.5rem 0.9rem; text-align: center;
    font-size: 0.82rem; color: #8b949e; font-family: monospace;
}
.pipe-arrow { text-align:center; color: #3fb950; font-size: 1.1rem; margin: 2px 0; }

/* ── table ── */
[data-testid="stDataFrame"] { border: 1px solid #21262d !important; border-radius: 6px; }

/* ── sidebar nav radio ── */
[data-testid="stRadio"] label {
    font-size: 0.88rem !important;
    color: #c9d1d9 !important;
}
[data-testid="stRadio"] { gap: 2px !important; }
div[role="radiogroup"] > label > div { border-radius: 4px !important; }

/* ── progress / spinner ── */
.stProgress > div > div { background-color: #3fb950 !important; }

/* ── divider ── */
hr { border-color: #21262d !important; }

/* ── terminal mono text ── */
.mono { font-family: 'Courier New', monospace; font-size: 0.82rem; color: #3fb950; }

/* ── confidence bar ── */
.conf-bar-bg {
    background:#21262d; border-radius:4px; height:10px; width:100%; margin-top:6px;
}
.conf-bar-fill {
    height:10px; border-radius:4px;
    background: linear-gradient(90deg,#1f6feb,#3fb950);
}
</style>
""", unsafe_allow_html=True)


# ─── helpers ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_experiment() -> dict:
    return json.loads(EXPERIMENT.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_dataset_sample(n_benign: int = 5, n_malware: int = 5) -> pd.DataFrame:
    """Load a small real slice of the dataset for demo purposes.
    Uses parquet for speed; falls back to CSV nrows trick if needed."""
    features = load_metadata()["feature_names"]
    cols = features + ["Class"]
    parquet_path = PROJECT_ROOT / "dataset" / "Obfuscated-MalMem2022.parquet"
    try:
        # parquet is ~10x faster than CSV for this
        df = pd.read_parquet(parquet_path, columns=cols)
    except Exception:
        # fallback: read only first 2000 rows of CSV (enough to find both classes)
        df = pd.read_csv(DATASET_PATH, nrows=2000, usecols=lambda c: c in cols)
    benign  = df[df["Class"] == "Benign"].head(n_benign)
    malware = df[df["Class"] == "Malware"].head(n_malware)
    return pd.concat([benign, malware], ignore_index=True)


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(FEAT_IMP_PATH)


@st.cache_data(show_spinner=False)
def load_model_results() -> pd.DataFrame:
    return pd.read_csv(MODEL_RESULTS)


@st.cache_data(show_spinner=False)
def load_cv_results() -> pd.DataFrame:
    return pd.read_csv(CV_RESULTS)


@st.cache_data(show_spinner=False)
def load_alt_results() -> pd.DataFrame:
    return pd.read_csv(ALT_RESULTS)


def confidence_to_risk(conf: float, label: str) -> tuple[str, str]:
    """Return (risk_level, badge_html) based on prediction and confidence."""
    if label == "Benign":
        if conf >= 0.98:
            return "LOW", '<span class="badge-green">● LOW RISK</span>'
        return "MEDIUM", '<span class="badge-yellow">● MEDIUM RISK</span>'
    else:  # Malware
        if conf >= 0.98:
            return "CRITICAL", '<span class="badge-red">● CRITICAL</span>'
        if conf >= 0.90:
            return "HIGH", '<span class="badge-red">● HIGH RISK</span>'
        return "MEDIUM", '<span class="badge-yellow">● MEDIUM RISK</span>'


def mtile(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="mtile-sub">{sub}</div>' if sub else ""
    return (f'<div class="mtile">'
            f'<div class="mtile-label">{label}</div>'
            f'<div class="mtile-value">{value}</div>'
            f'{sub_html}</div>')


def sec(title: str) -> None:
    st.markdown(f'<div class="sec-head">{title}</div>', unsafe_allow_html=True)


def card(html: str, extra_class: str = "") -> None:
    st.markdown(f'<div class="card {extra_class}">{html}</div>', unsafe_allow_html=True)


# ─── load core data ──────────────────────────────────────────────────────────
try:
    metadata   = load_metadata()
    experiment = load_experiment()
except Exception as e:
    st.error(f"Failed to load model configuration: {e}")
    st.stop()

FEATURE_NAMES: list[str] = metadata["feature_names"]

# ─── session state init ───────────────────────────────────────────────────────
for key in ("live_result", "batch_results", "history", "presentation_mode"):
    if key not in st.session_state:
        st.session_state[key] = None if key != "history" else []
        if key == "presentation_mode":
            st.session_state[key] = False

# ─── sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.6rem 0 0.8rem 0">
      <div style="font-size:1.05rem;font-weight:800;color:#e6edf3;letter-spacing:0.04em">
        🛡 RANSOMWARE DETECTION
      </div>
      <div style="font-size:0.7rem;color:#3fb950;letter-spacing:0.1em;margin-top:2px">
        CIC-MalMem2022 · BEHAVIORAL ANALYSIS
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "nav",
        ["Overview", "Live Detection", "Batch Analysis",
         "Model Intelligence", "Behavioral Analysis", "Validation", "About"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="nav-label">System Status</div>', unsafe_allow_html=True)
    st.markdown('<span class="badge-green">● MODEL READY</span>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-top:0.7rem;font-size:0.78rem;color:#8b949e;line-height:2">
      <span class="mono">MODEL</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Decision Tree<br>
      <span class="mono">FEATURES</span>&nbsp;&nbsp;&nbsp;52 behavioral<br>
      <span class="mono">DATASET</span>&nbsp;&nbsp;&nbsp;&nbsp;CIC-MalMem2022<br>
      <span class="mono">SAMPLES</span>&nbsp;&nbsp;&nbsp;&nbsp;58,058 rows
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    pmode = st.toggle("Presentation Mode", value=st.session_state.presentation_mode)
    st.session_state.presentation_mode = pmode

    if st.session_state.history:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="nav-label">Recent Detections</div>', unsafe_allow_html=True)
        for entry in reversed(st.session_state.history[-4:]):
            color = "#f85149" if entry["label"] == "Malware" else "#3fb950"
            st.markdown(
                f'<div style="font-size:0.75rem;color:{color};margin-bottom:3px">'
                f'● {"Ransomware" if entry["label"] == "Malware" else entry["label"]} &nbsp;<span style="color:#8b949e">{entry["time"]}</span></div>',
                unsafe_allow_html=True,
            )

# ─── presentation mode overlay ───────────────────────────────────────────────
if st.session_state.presentation_mode:
    st.markdown("""
    <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;
                padding:1rem 1.5rem;margin-bottom:1.2rem;display:flex;
                align-items:center;gap:2rem;flex-wrap:wrap">
      <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.12em;
                  text-transform:uppercase;color:#3fb950">PRESENTATION MODE ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    # header
    st.markdown("""
    <div style="margin-bottom:1.2rem">
      <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.16em;
                  text-transform:uppercase;color:#3fb950;margin-bottom:4px">
        CIC-MalMem2022 · BEHAVIORAL MEMORY ANALYSIS
      </div>
      <h1 style="font-size:2.1rem;font-weight:800;color:#e6edf3;margin:0 0 6px">
        AI BEHAVIORAL RANSOMWARE DETECTION
      </h1>
      <div style="font-size:1rem;color:#8b949e">
        Machine Learning-Based Memory Behavioral Analysis &nbsp;·&nbsp;
        Ransomware-Focused Detection
      </div>
    </div>
    """, unsafe_allow_html=True)

    # status row
    sec("SYSTEM STATUS")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(mtile("MODEL STATUS", "READY", "● ONLINE"), unsafe_allow_html=True)
    with c2:
        st.markdown(mtile("MODEL", "DT", "Decision Tree"), unsafe_allow_html=True)
    with c3:
        st.markdown(mtile("INPUT FEATURES", "52", "Behavioral"), unsafe_allow_html=True)
    with c4:
        st.markdown(mtile("DATASET", "CIC-MM22", "MalMem2022"), unsafe_allow_html=True)
    with c5:
        st.markdown(mtile("ACCURACY", "100.00%", "Held-out test"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # dataset metrics + pipeline
    left, right = st.columns([2, 1])
    with left:
        sec("DATASET METRICS")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(mtile("TOTAL SAMPLES", "58,058", "CIC-MalMem2022"), unsafe_allow_html=True)
        with m2:
            st.markdown(mtile("BENIGN", "29,290", "50.4 %"), unsafe_allow_html=True)
        with m3:
            st.markdown(mtile("RANSOMWARE", "28,768", "49.6 %"), unsafe_allow_html=True)
        with m4:
            st.markdown(mtile("FEATURES", "52", "After preprocessing"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        sec("MODEL PERFORMANCE SUMMARY")
        perf_cols = st.columns(4)
        for col, (metric, value) in zip(perf_cols, metadata["test_metrics"].items()):
            with col:
                st.markdown(mtile(metric.upper(), f"{value:.4f}", "Held-out 20%"), unsafe_allow_html=True)

    with right:
        sec("DETECTION PIPELINE")
        steps = [
            "Memory Behavioral Data",
            "Feature Validation",
            "Preprocessing",
            "ML Classification",
            "Risk Assessment",
            "Detection Result",
        ]
        for i, step in enumerate(steps):
            st.markdown(f'<div class="pipe-step">{step}</div>', unsafe_allow_html=True)
            if i < len(steps) - 1:
                st.markdown('<div class="pipe-arrow">↓</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sec("RECENT ANALYSIS ACTIVITY")
    if not st.session_state.history:
        card('<span style="color:#8b949e;font-size:0.88rem">⏳ &nbsp;Awaiting analysis — run Live Detection or Batch Analysis to see results here.</span>')
    else:
        cols = st.columns(len(st.session_state.history[-5:]))
        for col, entry in zip(cols, reversed(st.session_state.history[-5:])):
            color = "#f85149" if entry["label"] == "Malware" else "#3fb950"
            with col:
                st.markdown(
                    f'<div class="mtile">'
                    f'<div class="mtile-label">{entry["time"]}</div>'
                    f'<div style="font-size:1rem;font-weight:700;color:{color};margin-top:4px">'
                    f'{"⚠ " if entry["label"] == "Malware" else "✓ "}'
                    f'{"Ransomware" if entry["label"] == "Malware" else entry["label"]}</div>'
                    f'<div class="mtile-sub">{entry["conf"]:.2%} confidence</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    sec("ABOUT THIS PROJECT")
    card("""
    <div style="font-size:0.85rem;color:#8b949e;line-height:1.8">
      This system applies a <span style="color:#e6edf3">Decision Tree classifier</span>
      trained on the <span style="color:#e6edf3">CIC-MalMem2022</span> dataset to classify
      memory-behavioral feature snapshots as <span style="color:#3fb950">Benign</span> or
      <span style="color:#f85149">Ransomware</span>. The dataset contains a broad range of malware
      categories including ransomware, making this a
      <em>behavioral ransomware detection system with memory-based analysis</em>.<br><br>
      Performance values reflect the held-out CIC-MalMem2022 test split.
      External or temporal validation is required before real-world deployment.
    </div>
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Live Detection":
    st.markdown("""
    <div style="margin-bottom:1.2rem">
      <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.16em;
                  text-transform:uppercase;color:#3fb950;margin-bottom:4px">SINGLE SAMPLE ANALYSIS</div>
      <h1 style="font-size:1.9rem;font-weight:800;color:#e6edf3;margin:0">Live Detection</h1>
      <div style="font-size:0.9rem;color:#8b949e">Analyze one memory behavioral sample for ransomware in real time</div>
    </div>
    """, unsafe_allow_html=True)

    engine_col, result_col = st.columns([1, 1])

    with engine_col:
        sec("ANALYSIS ENGINE")
        st.markdown("""
        <div class="card" style="margin-bottom:0.8rem">
          <div style="font-size:0.7rem;color:#8b949e;letter-spacing:0.1em;text-transform:uppercase">STATUS</div>
          <div style="margin-top:4px"><span class="badge-green">● READY</span></div>
          <div style="margin-top:0.7rem;font-size:0.78rem;color:#8b949e;font-family:monospace;line-height:1.8">
            MODEL &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Decision Tree<br>
            FEATURES &nbsp;&nbsp;52 behavioral<br>
            CLASSES &nbsp;&nbsp;&nbsp;Benign / Ransomware
          </div>
        </div>
        """, unsafe_allow_html=True)

        sec("INPUT SOURCE")
        source = st.radio(
            "input_source",
            ["Demo Samples", "Upload CSV Row"],
            label_visibility="collapsed",
            horizontal=True,
        )

        sample_row = None
        true_label = None

        if source == "Demo Samples":
            sec("DEMO SAMPLES")
            demo_col1, demo_col2 = st.columns(2)
            try:
                demo_df = load_dataset_sample(n_benign=3, n_malware=3)
                benign_rows  = demo_df[demo_df["Class"] == "Benign"]
                malware_rows = demo_df[demo_df["Class"] == "Malware"]

                with demo_col1:
                    if st.button("Load Benign Sample", use_container_width=True):
                        row = benign_rows.iloc[0]
                        true_label = "Benign"
                        st.session_state["demo_row"]        = row[FEATURE_NAMES].to_dict()
                        st.session_state["demo_true_label"] = true_label

                with demo_col2:
                    if st.button("Load Ransomware Sample", use_container_width=True):
                        row = malware_rows.iloc[0]
                        true_label = "Malware"
                        st.session_state["demo_row"]        = row[FEATURE_NAMES].to_dict()
                        st.session_state["demo_true_label"] = true_label

            except Exception as e:
                st.error(f"Could not load demo samples: {e}")

            if "demo_row" in st.session_state and st.session_state["demo_row"]:
                tl = st.session_state.get("demo_true_label", "")
                badge = (f'<span class="badge-green">BENIGN</span>'
                         if tl == "Benign"
                         else f'<span class="badge-red">RANSOMWARE</span>')
                st.markdown(
                    f'<div style="margin:0.5rem 0;font-size:0.8rem;color:#8b949e">'
                    f'Sample loaded &nbsp;·&nbsp; True label: {badge}</div>',
                    unsafe_allow_html=True,
                )
                sample_row = st.session_state["demo_row"]

        else:
            uploaded = st.file_uploader(
                "Upload single-row CSV (52 features, no Class/Category columns)",
                type=["csv"],
                key="single_upload",
            )
            if uploaded:
                try:
                    df_up = pd.read_csv(uploaded)
                    if len(df_up) > 1:
                        st.warning("Multiple rows detected — using the first row only. Use Batch Analysis for multiple rows.")
                        df_up = df_up.head(1)
                    sample_row = df_up[FEATURE_NAMES].iloc[0].to_dict()
                    st.success("CSV loaded — ready to analyze.")
                except Exception as e:
                    st.error(f"Could not parse CSV: {e}")

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶  Analyze Sample", type="primary", use_container_width=True,
                            disabled=(sample_row is None))

    with result_col:
        sec("DETECTION RESULT")

        if run_btn and sample_row is not None:
            # show progress
            progress_placeholder = st.empty()
            steps = ["VALIDATING INPUT", "PREPROCESSING", "RUNNING CLASSIFIER", "GENERATING RESULT"]
            progress_bar = progress_placeholder.progress(0)
            for i, step in enumerate(steps):
                progress_bar.progress((i + 1) / len(steps), text=f"⚙ {step}...")
                time.sleep(0.05)
            progress_placeholder.empty()

            try:
                result = predict_sample(sample_row)
                result["timestamp"] = datetime.now().strftime("%H:%M:%S")
                st.session_state["live_result"] = result
                # add to history
                st.session_state.history.append({
                    "label": result["prediction"],
                    "conf":  result.get("probability", 1.0),
                    "time":  result["timestamp"],
                })
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        live = st.session_state.get("live_result")
        if live:
            label    = live["prediction"]
            conf     = live.get("probability", 1.0)
            ts       = live.get("timestamp", "")
            risk, risk_badge = confidence_to_risk(conf, label)
            conf_pct = conf * 100

            if label == "Malware":
                st.markdown(f"""
                <div class="result-malware">
                  <div style="font-size:2rem">⚠</div>
                  <div class="result-title" style="color:#f85149">THREAT DETECTED</div>
                  <div class="result-sub">Classification: RANSOMWARE</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-benign">
                  <div style="font-size:2rem">✓</div>
                  <div class="result-title" style="color:#3fb950">NO MALICIOUS CLASSIFICATION</div>
                  <div class="result-sub">Classification: BENIGN</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            d1, d2 = st.columns(2)
            with d1:
                display_label = "Ransomware" if label == "Malware" else label
                st.markdown(mtile("PREDICTION", display_label,
                                  "RANSOMWARE" if label == "Malware" else "BENIGN"),
                            unsafe_allow_html=True)
            with d2:
                st.markdown(mtile("CONFIDENCE", f"{conf_pct:.2f}%", "Model probability"),
                            unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            d3, d4 = st.columns(2)
            with d3:
                st.markdown(mtile("RISK LEVEL", risk, "App-level interpretation"),
                            unsafe_allow_html=True)
            with d4:
                st.markdown(mtile("TIMESTAMP", ts, "Analysis time"),
                            unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="card">
              <div class="mtile-label" style="margin-bottom:0.5rem">CONFIDENCE LEVEL</div>
              <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#8b949e">
                <span>0%</span><span>{conf_pct:.1f}%</span><span>100%</span>
              </div>
              <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{conf_pct:.1f}%"></div>
              </div>
              <div style="margin-top:0.8rem;font-size:0.78rem;color:#8b949e;line-height:1.6">
                <span class="mono">MODEL</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Decision Tree<br>
                <span class="mono">FEATURES</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;52 behavioral<br>
                <span class="mono">STATUS</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;COMPLETE<br>
                <span class="mono">RISK INDICATOR</span> &nbsp;{risk_badge}
              </div>
            </div>
            """, unsafe_allow_html=True)

            if live.get("class_probabilities"):
                st.markdown("<br>", unsafe_allow_html=True)
                sec("CLASS PROBABILITIES")
                for cls, prob in live["class_probabilities"].items():
                    color = "#f85149" if cls == "Malware" else "#3fb950"
                    display_cls = "Ransomware" if cls == "Malware" else cls
                    pct   = prob * 100
                    st.markdown(f"""
                    <div style="margin-bottom:0.5rem">
                      <div style="display:flex;justify-content:space-between;
                                  font-size:0.8rem;color:#8b949e;margin-bottom:3px">
                        <span style="color:{color};font-weight:600">{display_cls}</span>
                        <span>{pct:.2f}%</span>
                      </div>
                      <div class="conf-bar-bg">
                        <div style="height:8px;border-radius:4px;background:{color};width:{pct:.1f}%"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            card('<span style="color:#8b949e;font-size:0.88rem">⏳ &nbsp;Select a sample and click Analyze Sample to see the result.</span>')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Batch Analysis":
    st.markdown("""
    <div style="margin-bottom:1.2rem">
      <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.16em;
                  text-transform:uppercase;color:#3fb950;margin-bottom:4px">MULTI-SAMPLE ANALYSIS</div>
      <h1 style="font-size:1.9rem;font-weight:800;color:#e6edf3;margin:0">Batch Analysis</h1>
      <div style="font-size:0.9rem;color:#8b949e">
        Upload a CSV with multiple samples — all 52 features required, no Class/Category column
      </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 2])
    with left:
        sec("UPLOAD")
        uploaded_batch = st.file_uploader(
            "CSV file (52 feature columns)",
            type=["csv"],
            key="batch_upload",
            help="Must contain the 52 behavioral feature columns. Class and Category columns will be ignored if present.",
        )

        if uploaded_batch:
            if st.button("▶  Run Batch Prediction", type="primary", use_container_width=True):
                progress = st.progress(0, text="⚙ Validating schema...")
                try:
                    raw_df = pd.read_csv(uploaded_batch)
                    progress.progress(25, text="⚙ Preprocessing features...")

                    # strip target/category if user accidentally included them
                    input_df = raw_df.drop(columns=["Class", "Category"], errors="ignore")

                    progress.progress(50, text="⚙ Running classifier...")
                    results = predict_dataframe(input_df)
                    progress.progress(90, text="⚙ Generating report...")
                    progress.progress(100, text="✓ Complete")
                    st.session_state["batch_results"] = results
                    progress.empty()
                except ValueError as ve:
                    progress.empty()
                    st.markdown(f"""
                    <div class="card card-danger">
                      <div style="color:#f85149;font-weight:700;margin-bottom:0.4rem">
                        ⛔ INVALID INPUT FILE
                      </div>
                      <div style="font-size:0.82rem;color:#8b949e">{ve}</div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as ex:
                    progress.empty()
                    st.error(f"Batch prediction error: {ex}")

        # download sample template
        st.markdown("<br>", unsafe_allow_html=True)
        sec("SAMPLE TEMPLATE")
        try:
            sample_bytes = SAMPLE_CSV_PATH.read_bytes()
            st.download_button(
                "↓ Download Sample CSV",
                data=sample_bytes,
                file_name="sample_batch_input.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown(
                '<div style="font-size:0.75rem;color:#8b949e;margin-top:0.4rem">'
                'Real samples from the CIC-MalMem2022 dataset.</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass

    with right:
        batch = st.session_state.get("batch_results")
        if batch is not None:
            n_total   = len(batch)
            n_benign  = int((batch["Prediction"] == "Benign").sum())
            n_malware = int((batch["Prediction"] == "Malware").sum())
            pct_mal   = n_malware / n_total * 100 if n_total > 0 else 0

            sec("SUMMARY METRICS")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(mtile("TOTAL", str(n_total), "samples"), unsafe_allow_html=True)
            with c2:
                st.markdown(mtile("BENIGN", str(n_benign), f"{100-pct_mal:.1f}%"), unsafe_allow_html=True)
            with c3:
                st.markdown(mtile("RANSOMWARE", str(n_malware), f"{pct_mal:.1f}%"), unsafe_allow_html=True)
            with c4:
                threat_level = "HIGH" if pct_mal > 50 else ("MEDIUM" if pct_mal > 10 else "LOW")
                st.markdown(mtile("THREAT LEVEL", threat_level, "App-level"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            sec("DETECTION DISTRIBUTION")
            try:
                import plotly.graph_objects as go
                chart_c1, chart_c2 = st.columns(2)
                with chart_c1:
                    fig_pie = go.Figure(go.Pie(
                        labels=["Benign", "Ransomware"],
                        values=[n_benign, n_malware],
                        marker_colors=["#3fb950", "#f85149"],
                        hole=0.55,
                        textfont_size=12,
                    ))
                    fig_pie.update_layout(
                        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                        font_color="#8b949e", margin=dict(t=20,b=20,l=20,r=20),
                        showlegend=True, legend=dict(font_color="#c9d1d9"),
                        height=240,
                    )
                    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

                with chart_c2:
                    if "Confidence" in batch.columns:
                        fig_hist = go.Figure(go.Histogram(
                            x=batch["Confidence"],
                            nbinsx=20,
                            marker_color="#1f6feb",
                            marker_line_color="#21262d",
                            marker_line_width=1,
                        ))
                        fig_hist.update_layout(
                            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                            font_color="#8b949e", xaxis_title="Confidence",
                            yaxis_title="Count", margin=dict(t=20,b=30,l=30,r=20),
                            height=240,
                            xaxis=dict(gridcolor="#21262d"),
                            yaxis=dict(gridcolor="#21262d"),
                        )
                        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
            except ImportError:
                st.info("Install plotly for interactive charts: pip install plotly")

            st.markdown("<br>", unsafe_allow_html=True)
            sec("PREDICTION RESULTS")

            display = batch[["Prediction", "Confidence"]].copy() if "Confidence" in batch.columns \
                      else batch[["Prediction"]].copy()
            display.insert(0, "Sample ID", range(1, len(display) + 1))

            # remap "Malware" -> "Ransomware" for display only (CSV download keeps original)
            display["Prediction"] = display["Prediction"].replace("Malware", "Ransomware")

            if "Confidence" in display.columns:
                display["Risk"] = display.apply(
                    lambda r: confidence_to_risk(r["Confidence"],
                              "Malware" if r["Prediction"] == "Ransomware" else r["Prediction"])[0], axis=1
                )
                display["Confidence"] = display["Confidence"].map(lambda x: f"{x:.4f}")

            st.dataframe(
                display.style.applymap(
                    lambda v: "color:#f85149;font-weight:700" if v == "Ransomware"
                              else "color:#3fb950;font-weight:700" if v == "Benign" else "",
                    subset=["Prediction"],
                ),
                use_container_width=True,
                hide_index=True,
                height=280,
            )

            st.download_button(
                "↓ Download Results CSV",
                data=batch.to_csv(index=False).encode("utf-8"),
                file_name=f"ransomware_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            sec("AWAITING INPUT")
            card('<span style="color:#8b949e;font-size:0.88rem">⏳ &nbsp;Upload a CSV file and click Run Batch Prediction.</span>')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Intelligence":
    st.markdown("""
    <div style="margin-bottom:1.2rem">
      <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.16em;
                  text-transform:uppercase;color:#3fb950;margin-bottom:4px">ML MODEL ANALYTICS</div>
      <h1 style="font-size:1.9rem;font-weight:800;color:#e6edf3;margin:0">Model Intelligence</h1>
      <div style="font-size:0.9rem;color:#8b949e">Comparative performance of all evaluated classifiers</div>
    </div>
    """, unsafe_allow_html=True)

    model_df = load_model_results()
    cv_df    = load_cv_results()

    # selected model callout
    sec("SELECTED MODEL")
    st.markdown("""
    <div class="card card-success" style="display:flex;align-items:center;gap:1.5rem;padding:1rem 1.4rem">
      <div style="font-size:2rem">🏆</div>
      <div>
        <div style="font-size:1.2rem;font-weight:800;color:#3fb950">Decision Tree</div>
        <div style="font-size:0.8rem;color:#8b949e;margin-top:2px">
          Selected model &nbsp;·&nbsp; Accuracy 1.0000 &nbsp;·&nbsp; F1-Score 1.0000
          &nbsp;·&nbsp; 52 behavioral features
        </div>
      </div>
      <div style="margin-left:auto"><span class="badge-green">● ACTIVE</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sec("HELD-OUT TEST PERFORMANCE (80/20 STRATIFIED SPLIT)")

    # metric cards per model
    for _, row in model_df.iterrows():
        is_selected = row["Model"] == "Decision Tree"
        border = "card-success" if is_selected else ""
        badge  = '<span class="badge-green" style="margin-left:0.5rem">SELECTED</span>' if is_selected else ""
        st.markdown(f"""
        <div class="card {border}" style="margin-bottom:0.5rem">
          <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.5rem">
            <div style="font-size:0.95rem;font-weight:700;color:#e6edf3">
              {row['Model']}{badge}
            </div>
            <div style="display:flex;gap:1.5rem;flex-wrap:wrap">
              <div style="text-align:center">
                <div class="mtile-label">Accuracy</div>
                <div style="font-size:1.1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['Accuracy']:.4f}
                </div>
              </div>
              <div style="text-align:center">
                <div class="mtile-label">Precision</div>
                <div style="font-size:1.1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['Precision']:.4f}
                </div>
              </div>
              <div style="text-align:center">
                <div class="mtile-label">Recall</div>
                <div style="font-size:1.1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['Recall']:.4f}
                </div>
              </div>
              <div style="text-align:center">
                <div class="mtile-label">F1-Score</div>
                <div style="font-size:1.1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['F1-Score']:.4f}
                </div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # comparison chart
    st.markdown("<br>", unsafe_allow_html=True)
    sec("ACCURACY COMPARISON CHART")
    try:
        import plotly.graph_objects as go
        colors = ["#3fb950" if m == "Decision Tree" else "#1f6feb" for m in model_df["Model"]]
        fig = go.Figure(go.Bar(
            y=model_df["Model"],
            x=model_df["Accuracy"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.4f}" for v in model_df["Accuracy"]],
            textposition="outside",
            textfont=dict(color="#c9d1d9", size=12),
        ))
        fig.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            font_color="#8b949e", xaxis_title="Accuracy",
            xaxis=dict(range=[0.998, 1.0005], gridcolor="#21262d", color="#8b949e"),
            yaxis=dict(gridcolor="#21262d", color="#c9d1d9"),
            margin=dict(t=10, b=30, l=10, r=60),
            height=260,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except ImportError:
        st.image(str(IMG_PATH / "model_accuracy_comparison.png"))

    # cross-validation
    st.markdown("<br>", unsafe_allow_html=True)
    sec("5-FOLD CROSS-VALIDATION")
    card("""
    <div style="font-size:0.82rem;color:#8b949e">
      Cross-validation evaluates model stability across multiple training-validation folds,
      providing a more robust performance estimate than a single train/test split.
    </div>
    """)
    st.markdown("<br>", unsafe_allow_html=True)

    for _, row in cv_df.iterrows():
        is_sel = row["Model"] == "Decision Tree"
        border = "card-success" if is_sel else ""
        badge  = '<span class="badge-green" style="margin-left:0.5rem">SELECTED</span>' if is_sel else ""
        st.markdown(f"""
        <div class="card {border}" style="margin-bottom:0.5rem">
          <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.5rem">
            <div style="font-size:0.95rem;font-weight:700;color:#e6edf3">
              {row['Model']}{badge}
            </div>
            <div style="display:flex;gap:1.5rem;flex-wrap:wrap">
              <div style="text-align:center">
                <div class="mtile-label">Mean Accuracy</div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['Mean Accuracy']:.6f}
                </div>
              </div>
              <div style="text-align:center">
                <div class="mtile-label">Std Accuracy</div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['Std Accuracy']:.6f}
                </div>
              </div>
              <div style="text-align:center">
                <div class="mtile-label">Mean F1-Score</div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['Mean F1-Score']:.6f}
                </div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    try:
        st.image(str(IMG_PATH / "cross_validation_f1.png"),
                 caption="5-Fold Cross-Validation Mean F1-Score", use_container_width=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — BEHAVIORAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Behavioral Analysis":
    st.markdown("""
    <div style="margin-bottom:1.2rem">
      <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.16em;
                  text-transform:uppercase;color:#3fb950;margin-bottom:4px">FEATURE INTELLIGENCE</div>
      <h1 style="font-size:1.9rem;font-weight:800;color:#e6edf3;margin:0">Behavioral Analysis</h1>
      <div style="font-size:0.9rem;color:#8b949e">
        Memory behavioral indicators extracted from the CIC-MalMem2022 dataset
      </div>
    </div>
    """, unsafe_allow_html=True)

    feat_df = load_feature_importance()

    sec("FEATURE CONTEXT")
    card("""
    <div style="font-size:0.82rem;color:#8b949e;line-height:1.8">
      These features represent observable behavioral characteristics extracted from memory analysis data
      (Volatility framework). Feature importance indicates which attributes contributed most to the
      Decision Tree's classification decisions on the CIC-MalMem2022 dataset.
      <br><br>
      Feature importance does not by itself prove a causal link to ransomware.
      It reflects the statistical separability of these features within this dataset.
    </div>
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    sec("TOP BEHAVIORAL INDICATORS")

    # horizontal bar chart
    top_n = feat_df.head(15)
    try:
        import plotly.graph_objects as go
        colors = ["#3fb950" if i == 0 else "#1f6feb" if i < 4 else "#58a6ff"
                  for i in range(len(top_n))]
        fig = go.Figure(go.Bar(
            y=top_n["Feature"][::-1],
            x=top_n["Importance"][::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:.6f}" for v in top_n["Importance"][::-1]],
            textposition="outside",
            textfont=dict(color="#c9d1d9", size=10),
        ))
        fig.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            font_color="#8b949e",
            xaxis_title="Feature Importance",
            xaxis=dict(gridcolor="#21262d", color="#8b949e"),
            yaxis=dict(gridcolor="#21262d", color="#c9d1d9", tickfont=dict(size=11)),
            margin=dict(t=10, b=30, l=10, r=80),
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except ImportError:
        try:
            st.image(str(IMG_PATH / "final_feature_importance.png"),
                     caption="Top 15 Feature Importances", use_container_width=True)
        except Exception:
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ranked cards
    st.markdown("<br>", unsafe_allow_html=True)
    sec("RANKED FEATURE DETAIL")
    for _, row in feat_df.iterrows():
        pct = row["Importance"] * 100
        bar_w = min(pct, 100)
        rank  = int(row["Rank"])
        rank_color = "#3fb950" if rank == 1 else "#1f6feb" if rank <= 4 else "#58a6ff"
        st.markdown(f"""
        <div class="card" style="margin-bottom:0.4rem;padding:0.6rem 1rem">
          <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap">
            <div style="font-size:0.75rem;font-weight:700;color:{rank_color};
                        font-family:monospace;min-width:24px">#{rank}</div>
            <div style="font-size:0.85rem;color:#e6edf3;font-family:monospace;min-width:300px">
              {row['Feature']}
            </div>
            <div style="flex:1;min-width:150px">
              <div class="conf-bar-bg">
                <div style="height:8px;border-radius:4px;background:{rank_color};
                            width:{bar_w:.2f}%"></div>
              </div>
            </div>
            <div style="font-size:0.85rem;color:#8b949e;font-family:monospace;min-width:90px;text-align:right">
              {pct:.4f}%
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # feature groups
    st.markdown("<br>", unsafe_allow_html=True)
    sec("FEATURE GROUPS")
    groups = {
        "SERVICE": ["svcscan.nservices", "svcscan.process_services", "svcscan.kernel_drivers",
                    "svcscan.fs_drivers", "svcscan.shared_process_services", "svcscan.nactive"],
        "HANDLES": ["handles.nhandles", "handles.avg_handles_per_proc", "handles.nfile",
                    "handles.nevent", "handles.nmutant", "handles.ndesktop",
                    "handles.nkey", "handles.nthread"],
        "PROCESS": ["pslist.nproc", "pslist.nppid", "pslist.avg_threads", "pslist.avg_handlers"],
        "DLL":     ["dlllist.ndlls", "dlllist.avg_dlls_per_proc",
                    "ldrmodules.not_in_load", "ldrmodules.not_in_init", "ldrmodules.not_in_mem"],
        "MEMORY":  ["malfind.ninjections", "malfind.commitCharge", "malfind.protection",
                    "malfind.uniqueInjections", "callbacks.ncallbacks", "modules.nmodules"],
    }
    gcols = st.columns(len(groups))
    for col, (group_name, features) in zip(gcols, groups.items()):
        with col:
            st.markdown(f"""
            <div class="card" style="height:100%">
              <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.12em;
                          color:#3fb950;margin-bottom:0.5rem">{group_name}</div>
              {''.join(f'<div style="font-size:0.72rem;color:#8b949e;font-family:monospace;'
                       f'padding:2px 0">{f}</div>' for f in features)}
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Validation":
    st.markdown("""
    <div style="margin-bottom:1.2rem">
      <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.16em;
                  text-transform:uppercase;color:#3fb950;margin-bottom:4px">MODEL VALIDATION</div>
      <h1 style="font-size:1.9rem;font-weight:800;color:#e6edf3;margin:0">Validation</h1>
      <div style="font-size:0.9rem;color:#8b949e">
        Held-out test performance, confusion matrix, cross-validation, and alternative split results
      </div>
    </div>
    """, unsafe_allow_html=True)

    alt_df = load_alt_results()
    cv_df  = load_cv_results()

    # A — held-out test
    sec("A · HELD-OUT TEST PERFORMANCE  (80/20 STRATIFIED SPLIT, random_state=42)")
    hc = st.columns(4)
    for col, (metric, value) in zip(hc, metadata["test_metrics"].items()):
        with col:
            st.markdown(mtile(metric.upper(), f"{value:.4f}", "Decision Tree"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # B — confusion matrix
    sec("B · CONFUSION MATRIX  (Decision Tree)")
    cm_col, img_col = st.columns([1, 1])
    with cm_col:
        # TN FP / FN TP
        st.markdown("""
        <div class="card" style="padding:1.2rem">
          <div style="font-family:monospace;font-size:0.82rem;color:#8b949e;margin-bottom:0.6rem">
            Predicted → &nbsp;&nbsp; BENIGN &nbsp;&nbsp; RANSOMWARE
          </div>
          <table style="width:100%;border-collapse:collapse">
            <tr>
              <td style="color:#8b949e;font-family:monospace;font-size:0.8rem;padding:4px 8px">Actual BENIGN</td>
              <td style="text-align:center;background:#0d2818;border:1px solid #3fb950;
                         border-radius:4px;padding:12px 16px;font-size:1.3rem;font-weight:800;
                         color:#3fb950;font-family:monospace">5846<br>
                <span style="font-size:0.65rem;font-weight:400;color:#8b949e">TRUE NEGATIVE</span></td>
              <td style="text-align:center;background:#1a0a0a;border:1px solid #21262d;
                         border-radius:4px;padding:12px 16px;font-size:1.3rem;font-weight:800;
                         color:#8b949e;font-family:monospace">0<br>
                <span style="font-size:0.65rem;font-weight:400;color:#8b949e">FALSE POSITIVE</span></td>
            </tr>
            <tr>
              <td style="color:#8b949e;font-family:monospace;font-size:0.8rem;padding:4px 8px">Actual RANSOMWARE</td>
              <td style="text-align:center;background:#1a0a0a;border:1px solid #21262d;
                         border-radius:4px;padding:12px 16px;font-size:1.3rem;font-weight:800;
                         color:#8b949e;font-family:monospace">0<br>
                <span style="font-size:0.65rem;font-weight:400;color:#8b949e">FALSE NEGATIVE</span></td>
              <td style="text-align:center;background:#2d0f0f;border:1px solid #f85149;
                         border-radius:4px;padding:12px 16px;font-size:1.3rem;font-weight:800;
                         color:#f85149;font-family:monospace">5766<br>
                <span style="font-size:0.65rem;font-weight:400;color:#8b949e">TRUE POSITIVE</span></td>
            </tr>
          </table>
          <div style="margin-top:0.8rem;font-size:0.75rem;color:#8b949e">
            Test set: 11,612 samples &nbsp;·&nbsp; 5,846 Benign &nbsp;·&nbsp; 5,766 Ransomware
          </div>
        </div>
        """, unsafe_allow_html=True)

    with img_col:
        try:
            st.image(str(IMG_PATH / "final_confusion_matrix.png"),
                     caption="Decision Tree Confusion Matrix", use_container_width=True)
        except Exception:
            pass

    st.markdown("<br>", unsafe_allow_html=True)

    # C — cross-validation
    sec("C · 5-FOLD CROSS-VALIDATION")
    card("""
    <div style="font-size:0.82rem;color:#8b949e">
      Cross-validation evaluates model stability across multiple training-validation folds,
      providing a more robust estimate than a single split. Random state = 42.
    </div>
    """)
    st.markdown("<br>", unsafe_allow_html=True)
    for _, row in cv_df.iterrows():
        is_sel = row["Model"] == "Decision Tree"
        border = "card-success" if is_sel else ""
        badge  = '<span class="badge-green" style="margin-left:0.5rem">SELECTED</span>' if is_sel else ""
        st.markdown(f"""
        <div class="card {border}" style="margin-bottom:0.5rem">
          <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.8rem">
            <div style="font-size:0.95rem;font-weight:700;color:#e6edf3">{row['Model']}{badge}</div>
            <div style="display:flex;gap:1.4rem;flex-wrap:wrap">
              <div>
                <div class="mtile-label">Mean Accuracy</div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['Mean Accuracy']:.6f}
                </div>
              </div>
              <div>
                <div class="mtile-label">Std Accuracy</div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['Std Accuracy']:.6f}
                </div>
              </div>
              <div>
                <div class="mtile-label">Mean F1-Score</div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['Mean F1-Score']:.6f}
                </div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    try:
        st.image(str(IMG_PATH / "cross_validation_f1.png"),
                 caption="5-Fold Cross-Validation Mean F1-Score", use_container_width=True)
    except Exception:
        pass

    st.markdown("<br>", unsafe_allow_html=True)

    # D — alternative split
    sec("D · ALTERNATIVE SPLIT VALIDATION  (70/30)")
    card("""
    <div style="font-size:0.82rem;color:#8b949e">
      Performance re-evaluated on a 70/30 stratified split to confirm results are not
      overly sensitive to the specific train/test division.
    </div>
    """)
    st.markdown("<br>", unsafe_allow_html=True)
    for _, row in alt_df.iterrows():
        is_sel = row["Model"] == "Decision Tree"
        border = "card-success" if is_sel else ""
        badge  = '<span class="badge-green" style="margin-left:0.5rem">SELECTED</span>' if is_sel else ""
        st.markdown(f"""
        <div class="card {border}" style="margin-bottom:0.5rem">
          <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.8rem">
            <div style="font-size:0.95rem;font-weight:700;color:#e6edf3">{row['Model']}{badge}</div>
            <div style="display:flex;gap:1.4rem;flex-wrap:wrap">
              <div>
                <div class="mtile-label">Accuracy</div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['Accuracy']:.6f}
                </div>
              </div>
              <div>
                <div class="mtile-label">Precision</div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['Precision']:.6f}
                </div>
              </div>
              <div>
                <div class="mtile-label">Recall</div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['Recall']:.6f}
                </div>
              </div>
              <div>
                <div class="mtile-label">F1-Score</div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['F1-Score']:.6f}
                </div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # E — dataset quality
    sec("E · DATASET QUALITY")
    dc1, dc2, dc3, dc4, dc5 = st.columns(5)
    with dc1:
        st.markdown(mtile("TOTAL ROWS", "58,058", "CIC-MalMem2022"), unsafe_allow_html=True)
    with dc2:
        st.markdown(mtile("ORIGINAL COLS", "57", "Before preprocessing"), unsafe_allow_html=True)
    with dc3:
        st.markdown(mtile("MODEL FEATURES", "52", "After removal"), unsafe_allow_html=True)
    with dc4:
        st.markdown(mtile("MISSING VALUES", "0", "After imputation"), unsafe_allow_html=True)
    with dc5:
        st.markdown(mtile("TARGET CLASSES", "2", "Benign / Ransomware"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    card("""
    <div style="font-size:0.82rem;color:#8b949e;line-height:1.8">
      <strong style="color:#e6edf3">Removed columns:</strong><br>
      &bull; <span style="font-family:monospace;color:#58a6ff">Category</span>
        — high cardinality text, excluded from ML features<br>
      &bull; <span style="font-family:monospace;color:#58a6ff">pslist.nprocs64bit</span>
        — constant/zero-variance feature<br>
      &bull; <span style="font-family:monospace;color:#58a6ff">handles.nport</span>
        — constant/zero-variance feature<br>
      &bull; <span style="font-family:monospace;color:#58a6ff">svcscan.interactive_process_services</span>
        — constant/zero-variance feature<br><br>
      <strong style="color:#e6edf3">Validation caveat:</strong><br>
      Performance is measured on CIC-MalMem2022 and may reflect dataset-specific behavioral feature
      separability. External or temporal validation is required before real-world deployment claims.
    </div>
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # C — cross-validation
    sec("C · 5-FOLD CROSS-VALIDATION")
    card("""
    <div style="font-size:0.82rem;color:#8b949e">
      Cross-validation evaluates model stability across multiple training-validation folds,
      providing a more robust estimate than a single split. Random state = 42.
    </div>
    """)
    st.markdown("<br>", unsafe_allow_html=True)
    for _, row in cv_df.iterrows():
        is_sel = row["Model"] == "Decision Tree"
        border = "card-success" if is_sel else ""
        badge  = '<span class="badge-green" style="margin-left:0.5rem">SELECTED</span>' if is_sel else ""
        st.markdown(f"""
        <div class="card {border}" style="margin-bottom:0.5rem">
          <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.8rem">
            <div style="font-size:0.95rem;font-weight:700;color:#e6edf3">{row['Model']}{badge}</div>
            <div style="display:flex;gap:1.4rem;flex-wrap:wrap">
              <div>
                <div class="mtile-label">Mean Accuracy</div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['Mean Accuracy']:.6f}
                </div>
              </div>
              <div>
                <div class="mtile-label">Std Accuracy</div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['Std Accuracy']:.6f}
                </div>
              </div>
              <div>
                <div class="mtile-label">Mean F1-Score</div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['Mean F1-Score']:.6f}
                </div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # D — alternative split
    sec("D · ALTERNATIVE SPLIT VALIDATION  (70/30)")
    card("""
    <div style="font-size:0.82rem;color:#8b949e">
      Performance re-evaluated on a 70/30 stratified split to confirm results are not
      overly sensitive to the specific train/test division.
    </div>
    """)
    st.markdown("<br>", unsafe_allow_html=True)
    for _, row in alt_df.iterrows():
        is_sel = row["Model"] == "Decision Tree"
        border = "card-success" if is_sel else ""
        badge  = '<span class="badge-green" style="margin-left:0.5rem">SELECTED</span>' if is_sel else ""
        st.markdown(f"""
        <div class="card {border}" style="margin-bottom:0.5rem">
          <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.8rem">
            <div style="font-size:0.95rem;font-weight:700;color:#e6edf3">{row['Model']}{badge}</div>
            <div style="display:flex;gap:1.4rem;flex-wrap:wrap">
              <div>
                <div class="mtile-label">Accuracy</div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['Accuracy']:.6f}
                </div>
              </div>
              <div>
                <div class="mtile-label">Precision</div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['Precision']:.6f}
                </div>
              </div>
              <div>
                <div class="mtile-label">Recall</div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['Recall']:.6f}
                </div>
              </div>
              <div>
                <div class="mtile-label">F1-Score</div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;font-family:monospace">
                  {row['F1-Score']:.6f}
                </div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # E — dataset quality
    sec("E · DATASET QUALITY")
    dc1, dc2, dc3, dc4, dc5 = st.columns(5)
    with dc1:
        st.markdown(mtile("TOTAL ROWS", "58,058", "CIC-MalMem2022"), unsafe_allow_html=True)
    with dc2:
        st.markdown(mtile("ORIGINAL COLS", "57", "Before preprocessing"), unsafe_allow_html=True)
    with dc3:
        st.markdown(mtile("MODEL FEATURES", "52", "After removal"), unsafe_allow_html=True)
    with dc4:
        st.markdown(mtile("MISSING VALUES", "0", "After imputation"), unsafe_allow_html=True)
    with dc5:
        st.markdown(mtile("TARGET CLASSES", "2", "Benign / Ransomware"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    card("""
    <div style="font-size:0.82rem;color:#8b949e;line-height:1.8">
      <strong style="color:#e6edf3">Removed columns:</strong><br>
      • <span style="font-family:monospace;color:#58a6ff">Category</span> — high cardinality text, excluded from ML features<br>
      • <span style="font-family:monospace;color:#58a6ff">pslist.nprocs64bit</span> — constant/zero-variance feature<br>
      • <span style="font-family:monospace;color:#58a6ff">handles.nport</span> — constant/zero-variance feature<br>
      • <span style="font-family:monospace;color:#58a6ff">svcscan.interactive_process_services</span> — constant/zero-variance feature<br><br>
      <strong style="color:#e6edf3">Validation caveat:</strong><br>
      Performance is measured on CIC-MalMem2022 and may reflect dataset-specific behavioral feature
      separability. External or temporal validation is required before real-world deployment claims.
    </div>
    """)



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown(
        '<div style="margin-bottom:1.2rem">'
        '<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.16em;'
        'text-transform:uppercase;color:#3fb950;margin-bottom:4px">PROJECT INFORMATION</div>'
        '<h1 style="font-size:1.9rem;font-weight:800;color:#e6edf3;margin:0">About</h1>'
        '<div style="font-size:0.9rem;color:#8b949e">Academic PBL project documentation</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2])

    with left:
        sec("PROJECT")
        card(
            '<div style="font-size:0.95rem;font-weight:700;color:#e6edf3;margin-bottom:0.4rem">'
            'AI-Based Behavioral Ransomware Detection Using Machine Learning'
            '</div>'
            '<div style="font-size:0.82rem;color:#8b949e;line-height:1.8">'
            'A reproducible academic demonstration of AI-based behavioural ransomware detection '
            'using memory-forensics features from the CIC-MalMem2022 dataset.'
            '</div>'
        )

        st.markdown("<br>", unsafe_allow_html=True)
        sec("OBJECTIVE")
        card(
            '<div style="font-size:0.82rem;color:#8b949e;line-height:1.8">'
            'To develop a machine-learning-based application that analyzes memory behavioral features '
            'and classifies samples as <span style="color:#3fb950">Benign</span> or '
            '<span style="color:#f85149">Ransomware</span>.<br><br>'
            'This project — <em style="color:#e6edf3">AI-Based Behavioural Ransomware Detection '
            'Using Machine Learning</em> — applies behavioral memory analysis to detect '
            'ransomware-related activity using the CIC-MalMem2022 dataset.'
            '</div>'
        )

        st.markdown("<br>", unsafe_allow_html=True)
        sec("DATASET")
        card(
            '<div style="font-size:0.82rem;color:#8b949e;line-height:1.8">'
            '<strong style="color:#e6edf3">CIC-MalMem2022</strong> (Obfuscated-MalMem2022)<br>'
            'Canadian Institute for Cybersecurity &nbsp;&middot;&nbsp; Memory behavioral feature dataset<br><br>'
            '&bull; 58,058 samples &nbsp;&nbsp; &bull; 57 original columns &nbsp;&nbsp; &bull; 52 model features<br>'
            '&bull; Classes: Benign, Ransomware &nbsp;&nbsp; &bull; Includes ransomware and broader malware categories<br>'
            '&bull; Features extracted via the Volatility memory forensics framework'
            '</div>'
        )

        st.markdown("<br>", unsafe_allow_html=True)
        sec("WORKFLOW")
        workflow = [
            ("Dataset",            "CIC-MalMem2022 &middot; 58,058 rows"),
            ("Preprocessing",      "Remove constants, median impute"),
            ("Feature Engineering","52 behavioral features selected"),
            ("Model Training",     "5 classifiers &middot; 80/20 stratified split"),
            ("Evaluation",         "Accuracy, Precision, Recall, F1 &middot; 5-fold CV"),
            ("Detection",          "Streamlit dashboard &middot; live &amp; batch prediction"),
        ]
        for i, (step, detail) in enumerate(workflow):
            arrow = '<div style="text-align:left;padding-left:1rem;color:#3fb950;font-size:1rem">&darr;</div>' \
                    if i < len(workflow) - 1 else ""
            st.markdown(
                f'<div class="card" style="margin-bottom:0;padding:0.6rem 1rem">'
                f'<span style="font-size:0.85rem;font-weight:700;color:#e6edf3">{step}</span>'
                f'<span style="font-size:0.78rem;color:#8b949e;margin-left:0.8rem">{detail}</span>'
                f'</div>{arrow}',
                unsafe_allow_html=True,
            )

    with right:
        sec("TECHNOLOGY STACK")
        tech = [
            ("Python",       "3.12",              "#3fb950"),
            ("Pandas",       "data processing",   "#1f6feb"),
            ("NumPy",        "numerical ops",     "#1f6feb"),
            ("Scikit-learn", "1.9.1 ML pipeline", "#d29922"),
            ("XGBoost",      "gradient boosting", "#d29922"),
            ("Streamlit",    "dashboard UI",      "#f85149"),
            ("Joblib",       "model serialization","#58a6ff"),
        ]
        for name, detail, color in tech:
            st.markdown(
                f'<div class="card" style="margin-bottom:0.4rem;padding:0.55rem 1rem;'
                f'display:flex;align-items:center;gap:0.8rem">'
                f'<div style="width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0"></div>'
                f'<div><span style="font-size:0.88rem;font-weight:700;color:#e6edf3">{name}</span>'
                f'<span style="font-size:0.75rem;color:#8b949e;margin-left:0.5rem">{detail}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        sec("LIMITATIONS")
        card(
            '<div style="font-size:0.82rem;color:#8b949e;line-height:1.9">'
            '&bull; Performance metrics reflect CIC-MalMem2022 test split only<br>'
            '&bull; Dataset-specific behavioral feature separability may inflate scores<br>'
            '&bull; CIC-MalMem2022 is a broad memory-based dataset; not exclusively ransomware samples<br>'
            '&bull; External or temporal validation required before real-world deployment<br>'
            '&bull; Real-world performance may differ due to obfuscation and novel variants<br>'
            '&bull; The system does not claim to detect every real-world ransomware sample'
            '</div>'
        )

        st.markdown("<br>", unsafe_allow_html=True)
        sec("HOW TO RUN")
        st.markdown(
            '<div class="card">'
            '<div style="font-family:monospace;font-size:0.82rem;color:#3fb950;line-height:2">'
            'python -m streamlit run app/app.py'
            '</div>'
            '<div style="font-size:0.75rem;color:#8b949e;margin-top:0.5rem">'
            'Requires: streamlit, pandas, numpy, scikit-learn==1.9.1, xgboost, joblib, plotly (optional)'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        try:
            st.image(
                str(IMG_PATH / "class_distribution.png"),
                caption="CIC-MalMem2022 class distribution",
                use_container_width=True,
            )
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# PRESENTATION MODE PANEL  (shown at bottom of any page when toggled)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.presentation_mode:
    st.markdown("<hr>", unsafe_allow_html=True)
    sec("PRESENTATION SUMMARY")

    live  = st.session_state.get("live_result")
    batch = st.session_state.get("batch_results")

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown(mtile("SYSTEM STATUS", "READY", "ONLINE"), unsafe_allow_html=True)
    with p2:
        st.markdown(mtile("MODEL", "Decision Tree", "52 features"), unsafe_allow_html=True)
    with p3:
        st.markdown(mtile("DATASET", "CIC-MalMem2022", "58,058 samples"), unsafe_allow_html=True)
    with p4:
        analyzed = len(st.session_state.history)
        st.markdown(mtile("ANALYZED", str(analyzed), "this session"), unsafe_allow_html=True)

    if live:
        st.markdown("<br>", unsafe_allow_html=True)
        label = live["prediction"]
        conf  = live.get("probability", 1.0)
        color = "#f85149" if label == "Malware" else "#3fb950"
        p5, p6 = st.columns(2)
        with p5:
            st.markdown(
                f'<div class="mtile"><div class="mtile-label">LAST PREDICTION</div>'
                f'<div style="font-size:1.5rem;font-weight:800;color:{color};margin-top:4px">'
                f'{label}</div></div>',
                unsafe_allow_html=True,
            )
        with p6:
            st.markdown(
                mtile("CONFIDENCE", f"{conf:.2%}", live.get("timestamp", "")),
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    pm1, pm2 = st.columns(2)

    with pm1:
        sec("MODEL PERFORMANCE")
        for metric, value in metadata["test_metrics"].items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;font-size:0.85rem;'
                f'padding:4px 0;border-bottom:1px solid #21262d">'
                f'<span style="color:#8b949e">{metric}</span>'
                f'<span style="color:#3fb950;font-family:monospace;font-weight:700">{value:.4f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with pm2:
        sec("TOP BEHAVIORAL FEATURES")
        try:
            fi = load_feature_importance().head(5)
            for _, row in fi.iterrows():
                pct = row["Importance"] * 100
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;font-size:0.78rem;'
                    f'padding:4px 0;border-bottom:1px solid #21262d">'
                    f'<span style="color:#8b949e;font-family:monospace">{row["Feature"]}</span>'
                    f'<span style="color:#1f6feb;font-family:monospace">{pct:.2f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        except Exception:
            pass
# ══════════════════════════════════════════════════════════════════════════════
# PRESENTATION MODE OVERLAY (appended to any active page when toggled)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.presentation_mode:
    st.markdown("<hr>", unsafe_allow_html=True)
    sec("PRESENTATION SUMMARY")
    live = st.session_state.get("live_result")
    batch = st.session_state.get("batch_results")

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown(mtile("SYSTEM STATUS", "READY", "● ONLINE"), unsafe_allow_html=True)
    with p2:
        st.markdown(mtile("MODEL", "Decision Tree", "52 features"), unsafe_allow_html=True)
    with p3:
        st.markdown(mtile("DATASET", "CIC-MalMem2022", "58,058 samples"), unsafe_allow_html=True)
    with p4:
        analyzed = len(st.session_state.history)
        st.markdown(mtile("ANALYZED", str(analyzed), "this session"), unsafe_allow_html=True)

    if live:
        st.markdown("<br>", unsafe_allow_html=True)
        p5, p6 = st.columns(2)
        label = live["prediction"]
        conf  = live.get("probability", 1.0)
        color = "#f85149" if label == "Malware" else "#3fb950"
        with p5:
            st.markdown(f'<div class="mtile"><div class="mtile-label">LAST PREDICTION</div>'
                        f'<div style="font-size:1.5rem;font-weight:800;color:{color};margin-top:4px">'
                        f'{label}</div></div>', unsafe_allow_html=True)
        with p6:
            st.markdown(mtile("CONFIDENCE", f"{conf:.2%}", live.get("timestamp", "")),
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    pm1, pm2 = st.columns(2)
    with pm1:
        sec("MODEL PERFORMANCE")
        for metric, value in metadata["test_metrics"].items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.85rem;padding:3px 0;border-bottom:1px solid #21262d">'
                f'<span style="color:#8b949e">{metric}</span>'
                f'<span style="color:#3fb950;font-family:monospace;font-weight:700">{value:.4f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with pm2:
        sec("TOP BEHAVIORAL FEATURES")
        try:
            fi = load_feature_importance().head(5)
            for _, row in fi.iterrows():
                pct = row["Importance"] * 100
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.78rem;padding:3px 0;border-bottom:1px solid #21262d">'
                    f'<span style="color:#8b949e;font-family:monospace">{row["Feature"]}</span>'
                    f'<span style="color:#1f6feb;font-family:monospace">{pct:.2f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

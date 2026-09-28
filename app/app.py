import os
import sys
import hashlib
import time
import streamlit as st

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import agentops
import matplotlib
matplotlib.use("Agg")   # non-interactive backend – prevents GUI pop-ups
from dotenv import load_dotenv

load_dotenv()

agentops.init(auto_start_session=False, instrument_llm_calls=False)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Scientist",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Base ────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0 }
html, body, [data-testid="stApp"],
[data-testid="stAppViewContainer"], .main { background: #070810 !important }
[data-testid="stAppViewContainer"] > .main { padding-top: 0 !important }
#MainMenu, footer, header,
[data-testid="stDeployButton"] { display: none !important }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #dde3f0 }

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#0c0e1a 0%,#080b16 100%) !important;
  border-right: 1px solid rgba(99,102,241,.15) !important;
}
.sb-brand {
  padding: 28px 20px 20px;
  border-bottom: 1px solid rgba(99,102,241,.12);
  margin-bottom: 4px;
}
.sb-brand-title {
  font-size: 1.2rem; font-weight: 800; letter-spacing: -.025em;
  background: linear-gradient(135deg,#818cf8 0%,#a78bfa 55%,#38bdf8 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.sb-brand-sub {
  font-size: .68rem; color: #4b5680; letter-spacing: .07em;
  text-transform: uppercase; margin-top: 3px;
}
.sb-section {
  font-size: .63rem; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase; color: #374168;
  padding: 18px 20px 6px;
}

/* ── Sidebar buttons (nav tabs) ──────────────────── */
.stButton > button {
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: 9px !important;
  color: #7283a8 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: .83rem !important; font-weight: 500 !important;
  padding: 9px 14px !important;
  text-align: left !important;
  transition: all .18s ease !important;
  box-shadow: none !important;
}
.stButton > button:hover {
  background: rgba(99,102,241,.1) !important;
  border-color: rgba(99,102,241,.22) !important;
  color: #c4b5fd !important;
  transform: none !important;
  box-shadow: none !important;
}

/* Primary run button – override narrow specificity */
button[kind="primary"] {
  background: linear-gradient(135deg,#6366f1 0%,#8b5cf6 60%,#06b6d4 100%) !important;
  border: none !important;
  border-radius: 12px !important;
  color: #fff !important;
  font-weight: 700 !important; font-size: .9rem !important;
  padding: 13px 28px !important;
  box-shadow: 0 4px 24px rgba(99,102,241,.35) !important;
  transition: all .22s !important;
  letter-spacing: .015em !important;
}
button[kind="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(99,102,241,.5) !important;
  filter: brightness(1.08) !important;
}

/* ── File uploader ───────────────────────────────── */
[data-testid="stFileUploader"] {
  background: rgba(99,102,241,.04) !important;
  border: 1.5px dashed rgba(99,102,241,.28) !important;
  border-radius: 12px !important; padding: 6px !important;
  transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: rgba(99,102,241,.5) !important;
}

/* ── Metric cards ────────────────────────────────── */
.kpi-card {
  background: linear-gradient(135deg,#0e1022 0%,#131628 100%);
  border: 1px solid rgba(99,102,241,.14);
  border-radius: 16px; padding: 20px 22px;
  position: relative; overflow: hidden;
  transition: transform .22s, box-shadow .22s, border-color .22s;
  margin-bottom: 4px;
}
.kpi-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 12px 36px rgba(0,0,0,.4);
  border-color: rgba(99,102,241,.3);
}
.kpi-card::before {
  content: ''; position: absolute; top: 0; left: 0;
  width: 100%; height: 3px; border-radius: 16px 16px 0 0;
}
.kpi-card.p::before { background: linear-gradient(90deg,#6366f1,#8b5cf6) }
.kpi-card.b::before { background: linear-gradient(90deg,#06b6d4,#818cf8) }
.kpi-card.g::before { background: linear-gradient(90deg,#10b981,#34d399) }
.kpi-card.a::before { background: linear-gradient(90deg,#f59e0b,#fbbf24) }
.kpi-card .k-icon { position: absolute; right: 18px; top: 16px; font-size: 1.7rem; opacity: .12 }
.kpi-card .k-label {
  font-size: .7rem; color: #4b5680; text-transform: uppercase;
  letter-spacing: .09em; font-weight: 600;
}
.kpi-card .k-value {
  font-size: 2.1rem; font-weight: 800; color: #eef2ff;
  letter-spacing: -.03em; margin-top: 5px; line-height: 1;
}
.kpi-card .k-sub { font-size: .75rem; color: #4b5680; margin-top: 5px }

/* ── Section headers ─────────────────────────────── */
.sec-hdr {
  display: flex; align-items: center; gap: 10px;
  margin: 30px 0 16px;
}
.sec-hdr-icon {
  padding: 7px; border-radius: 9px;
  background: rgba(99,102,241,.1);
  font-size: 1rem; line-height: 1;
}
.sec-hdr h2 {
  font-size: 1rem; font-weight: 700; color: #dde3f0;
  white-space: nowrap;
}
.sec-hdr .sec-line {
  flex: 1; height: 1px;
  background: linear-gradient(90deg,rgba(99,102,241,.2),transparent);
}

/* ── Hero ────────────────────────────────────────── */
.hero-row {
  display: flex; align-items: flex-start;
  justify-content: space-between; gap: 20px;
  margin-bottom: 28px;
}
.hero-title {
  font-size: 2rem; font-weight: 900; letter-spacing: -.04em;
  background: linear-gradient(135deg,#fff 0%,#c4b5fd 45%,#67e8f9 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; line-height: 1.15;
}
.hero-sub { font-size: .88rem; color: #4b5680; margin-top: 5px }

/* ── Status pill ─────────────────────────────────── */
.pill {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 6px 14px; border-radius: 999px;
  font-size: .76rem; font-weight: 600; letter-spacing: .025em;
}
.pill.idle    { background: rgba(71,85,105,.18); color: #7283a8; border: 1px solid rgba(71,85,105,.25) }
.pill.running { background: rgba(99,102,241,.14); color: #a5b4fc; border: 1px solid rgba(99,102,241,.3) }
.pill.done    { background: rgba(16,185,129,.1);  color: #34d399; border: 1px solid rgba(16,185,129,.25) }
.pill.error   { background: rgba(239,68,68,.1);   color: #f87171; border: 1px solid rgba(239,68,68,.25) }
.pill-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor }
.pill.running .pill-dot { animation: blink 1.4s infinite }
@keyframes blink { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(1.4)} }

/* ── Pipeline stepper ────────────────────────────── */
.stepper {
  display: flex; align-items: flex-start; gap: 0;
  margin: 20px 0 8px; overflow-x: auto; padding-bottom: 4px;
}
.stepper::-webkit-scrollbar { height: 4px }
.stepper::-webkit-scrollbar-track { background: transparent }
.stepper::-webkit-scrollbar-thumb { background: rgba(99,102,241,.3); border-radius: 2px }

.step-wrap { display: flex; align-items: center; flex-shrink: 0 }
.step-node {
  display: flex; flex-direction: column; align-items: center; gap: 6px;
  min-width: 80px;
}
.step-circle {
  width: 34px; height: 34px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: .78rem; font-weight: 700;
  border: 2px solid rgba(99,102,241,.2);
  background: rgba(99,102,241,.06);
  color: #4b5680; transition: all .3s;
}
.step-node.done  .step-circle {
  background: rgba(16,185,129,.15); border-color: #10b981;
  color: #34d399; box-shadow: 0 0 10px rgba(16,185,129,.3);
}
.step-node.active .step-circle {
  background: rgba(99,102,241,.2); border-color: #818cf8;
  color: #a5b4fc; box-shadow: 0 0 14px rgba(99,102,241,.5);
  animation: pulse-ring 1.8s infinite;
}
@keyframes pulse-ring {
  0%,100%{box-shadow:0 0 0 0 rgba(99,102,241,.4)}
  50%{box-shadow:0 0 0 8px rgba(99,102,241,0)}
}
.step-label {
  font-size: .65rem; font-weight: 600; letter-spacing: .03em;
  color: #374168; text-align: center; max-width: 72px;
  line-height: 1.3; text-transform: uppercase;
}
.step-node.done  .step-label { color: #34d399 }
.step-node.active .step-label { color: #a5b4fc }
.step-connector {
  width: 28px; height: 2px; background: rgba(99,102,241,.15);
  flex-shrink: 0; margin-bottom: 22px;
}
.step-connector.done { background: #10b981 }

/* ── Progress bar ────────────────────────────────── */
.prog-wrap {
  background: rgba(99,102,241,.06);
  border: 1px solid rgba(99,102,241,.14);
  border-radius: 14px; padding: 18px 22px; margin-bottom: 16px;
}
.prog-top {
  display: flex; justify-content: space-between;
  margin-bottom: 10px;
}
.prog-label { font-size: .8rem; color: #7283a8; font-weight: 500 }
.prog-pct   { font-size: .8rem; color: #818cf8; font-weight: 700 }
.prog-track {
  background: rgba(99,102,241,.1); border-radius: 8px;
  height: 8px; overflow: hidden;
}
.prog-fill {
  height: 100%; border-radius: 8px;
  background: linear-gradient(90deg,#6366f1,#8b5cf6,#06b6d4);
  transition: width .5s cubic-bezier(.4,0,.2,1);
  position: relative; overflow: hidden;
}
.prog-fill::after {
  content: ''; position: absolute;
  top: 0; left: -100%; width: 100%; height: 100%;
  background: linear-gradient(90deg,transparent,rgba(255,255,255,.35),transparent);
  animation: shimmer 2.2s infinite;
}
@keyframes shimmer { to { left: 200% } }
.prog-status {
  font-size: .78rem; color: #a5b4fc; margin-top: 8px;
  font-style: italic;
}

/* ── Log terminal ────────────────────────────────── */
.terminal {
  background: #040509; border: 1px solid rgba(99,102,241,.15);
  border-radius: 12px; padding: 14px 18px;
  font-family: 'JetBrains Mono', monospace; font-size: .76rem;
  max-height: 200px; overflow-y: auto; margin-top: 12px;
}
.terminal::-webkit-scrollbar { width: 4px }
.terminal::-webkit-scrollbar-thumb { background: rgba(99,102,241,.3); border-radius: 2px }
.lg { padding: 1.5px 0; line-height: 1.7 }
.lg .ts { color: #2d3248; margin-right: 8px }
.lg.info  .msg { color: #64748b }
.lg.step  .msg { color: #818cf8 }
.lg.done  .msg { color: #10b981 }
.lg.warn  .msg { color: #f59e0b }
.lg.error .msg { color: #f87171 }

/* ── Viz gallery ─────────────────────────────────── */
.viz-card {
  background: #0d0f1c; border: 1px solid rgba(99,102,241,.14);
  border-radius: 14px; overflow: hidden;
  transition: transform .22s, border-color .22s; margin-bottom: 18px;
}
.viz-card:hover { transform: translateY(-4px); border-color: rgba(99,102,241,.35) }
.viz-hdr {
  padding: 10px 16px; border-bottom: 1px solid rgba(99,102,241,.1);
  font-size: .78rem; font-weight: 600; color: #5b6a9a;
  letter-spacing: .04em; text-transform: uppercase;
}

/* ── Tag pills ───────────────────────────────────── */
.tag {
  display: inline-block;
  background: rgba(99,102,241,.1); color: #818cf8;
  border: 1px solid rgba(99,102,241,.22);
  border-radius: 6px; padding: 2px 9px;
  font-size: .73rem; font-weight: 500;
  margin: 3px 2px; font-family: 'JetBrains Mono', monospace;
}

/* ── Finding items ───────────────────────────────── */
.fi {
  display: flex; align-items: flex-start; gap: 10px;
  padding: 10px 14px; border-radius: 10px; margin-bottom: 7px;
  font-size: .83rem; background: rgba(99,102,241,.05);
  border: 1px solid rgba(99,102,241,.1);
}
.fi.warn { background: rgba(245,158,11,.05); border-color: rgba(245,158,11,.2); color: #fbbf24 }
.fi.ok   { background: rgba(16,185,129,.05); border-color: rgba(16,185,129,.18) }
.fi.info-c{ background: rgba(6,182,212,.05); border-color: rgba(6,182,212,.2) }
.fi-dot { flex-shrink: 0; margin-top: 1px }

/* ── Download card ───────────────────────────────── */
.dl-card {
  background: linear-gradient(135deg,#0d0f1c,#121528);
  border: 1px solid rgba(99,102,241,.15);
  border-radius: 14px; padding: 20px 22px;
  display: flex; align-items: center; gap: 18px;
  transition: all .22s; margin-bottom: 14px;
}
.dl-card:hover { border-color: rgba(99,102,241,.4); transform: translateY(-2px) }
.dl-ico {
  width: 50px; height: 50px; border-radius: 12px;
  background: rgba(99,102,241,.12);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.7rem; flex-shrink: 0;
}
.dl-title { font-weight: 700; font-size: .9rem; color: #dde3f0 }
.dl-desc  { font-size: .75rem; color: #4b5680; margin-top: 3px }

/* ── Data frames ─────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden }

/* ── Expanders ───────────────────────────────────── */
[data-testid="stExpander"] {
  background: #0d0f1c !important;
  border: 1px solid rgba(99,102,241,.14) !important;
  border-radius: 12px !important;
}

/* ── Alerts ──────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: 10px !important;
  border: 1px solid !important;
  font-size: .84rem !important;
}

/* ── Scrollbar global ────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px }
::-webkit-scrollbar-track { background: #070810 }
::-webkit-scrollbar-thumb { background: #1c2040; border-radius: 3px }
::-webkit-scrollbar-thumb:hover { background: #272d52 }

/* ── Divider ─────────────────────────────────────── */
hr { border-color: rgba(99,102,241,.1) !important; margin: 20px 0 !important }

/* ── Metric override ─────────────────────────────── */
[data-testid="stMetric"] {
  background: #0d0f1c !important;
  border: 1px solid rgba(99,102,241,.14) !important;
  border-radius: 12px !important; padding: 16px !important;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════
def _s(k, d=None):  return st.session_state.get(k, d)
def _ss(k, v):      st.session_state[k] = v

def kpi(color, icon, label, value, sub=""):
    return (
        f'<div class="kpi-card {color}">'
        f'<span class="k-icon">{icon}</span>'
        f'<div class="k-label">{label}</div>'
        f'<div class="k-value">{value}</div>'
        f'{"<div class=k-sub>"+sub+"</div>" if sub else ""}'
        f'</div>'
    )

def sh(icon, title):
    return (
        f'<div class="sec-hdr"><span class="sec-hdr-icon">{icon}</span>'
        f'<h2>{title}</h2><div class="sec-line"></div></div>'
    )


# ═════════════════════════════════════════════════════════════════
# Sidebar  (brand + upload + nav only – no pipeline status, no run btn)
# ═════════════════════════════════════════════════════════════════
TABS = [
    ("🏠", "Overview",       "overview"),
    ("🛡️", "Data Quality",   "quality"),
    ("📈", "EDA",            "eda"),
    ("📊", "Visualizations", "visualizations"),
    ("⚙️", "Preprocessing",  "preprocessing"),
    ("🤖", "ML Models",      "ml"),
    ("🔎", "Validation",     "validation"),
    ("📄", "Report",         "report"),
]

STAGES = [
    ("profiler",      "Profiling",   "🔍"),
    ("quality",       "Quality",     "🛡️"),
    ("cleaning",      "Cleaning",    "🧹"),
    ("eda",           "EDA",         "📈"),
    ("visualization", "Visualize",   "📊"),
    ("preprocessing", "Preprocess",  "⚙️"),
    ("ml",            "ML",          "🤖"),
    ("critic",        "Validation",  "🔎"),
    ("reporter",      "Report",      "📝"),
]

STAGE_MATCH = {
    "profiler":      ("profil",),
    "quality":       ("quality",),
    "cleaning":      ("clean",),
    "eda":           ("eda",),
    "visualization": ("visual",),
    "preprocessing": ("preprocess",),
    "ml":            ("ml", "machine learning"),
    "critic":        ("critic", "validation"),
    "reporter":      ("report",),
}

PROG_MAP = {
    "profiler":      ( 5, 14, "🔍 Profiling dataset..."),
    "quality":       (14, 25, "🛡️ Analyzing data quality..."),
    "cleaning":      (25, 36, "🧹 Cleaning dataset..."),
    "eda":           (36, 50, "📈 Running exploratory analysis..."),
    "visualization": (50, 62, "📊 Generating visualizations..."),
    "preprocessing": (62, 73, "⚙️ Preprocessing features..."),
    "ml":            (73, 87, "🤖 Training & evaluating models..."),
    "critic":        (87, 94, "🔎 Validating results..."),
    "reporter":      (94,100, "📝 Generating final report..."),
}

with st.sidebar:
    st.markdown(
        '<div class="sb-brand">'
        '<div class="sb-brand-title">🧠 AI Data Scientist</div>'
        '<div class="sb-brand-sub">Autonomous Analysis Pipeline</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-section">Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "dataset", type=["csv", "xlsx", "xls", "pdf"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sb-section">Navigate</div>', unsafe_allow_html=True)

    if "active_tab" not in st.session_state:
        _ss("active_tab", "overview")

    for icon, label, key in TABS:
        if st.button(f"{icon}  {label}", key=f"_nav_{key}", use_container_width=True):
            _ss("active_tab", key)
            st.rerun()


# ═════════════════════════════════════════════════════════════════
# File upload / state management
# ═════════════════════════════════════════════════════════════════
dataset_path = None
preview_df   = None

if uploaded_file is not None:
    from tools.data_profiler import load_dataset

    ext = os.path.splitext(uploaded_file.name)[1].lower()
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, f"uploaded_dataset{ext}")

    raw_bytes = uploaded_file.getvalue()
    uhash = hashlib.sha256(uploaded_file.name.encode() + raw_bytes).hexdigest()

    if _s("uploaded_hash") != uhash:
        with open(raw_path, "wb") as fh:
            fh.write(raw_bytes)
        try:
            _ss("uploaded_preview_df", load_dataset(raw_path))
            _ss("uploaded_hash",       uhash)
            _ss("uploaded_path",       raw_path)
            st.session_state.pop("analysis_result", None)
        except Exception:
            st.session_state.pop("uploaded_preview_df", None)
            st.session_state.pop("uploaded_hash",       None)

    dataset_path = _s("uploaded_path", raw_path)
    preview_df   = _s("uploaded_preview_df")


# ═════════════════════════════════════════════════════════════════
# Helper: build animated pipeline stepper HTML
# ═════════════════════════════════════════════════════════════════
def _stepper_html(completed_keys: set, active_key: str) -> str:
    parts = []
    for i, (key, label, icon) in enumerate(STAGES):
        prefixes = STAGE_MATCH.get(key, (key,))
        is_done   = any(any(p in c.lower() for p in prefixes) for c in completed_keys)
        is_active = (key == (active_key or "").lower()) and not is_done
        cls = "done" if is_done else ("active" if is_active else "")
        circle = "✓" if is_done else f"{i+1}"
        conn_cls = "done" if is_done else ""
        parts.append(
            f'<div class="step-wrap">'
            f'<div class="step-node {cls}">'
            f'<div class="step-circle">{circle}</div>'
            f'<div class="step-label">{icon} {label}</div>'
            f'</div>'
            + (f'<div class="step-connector {conn_cls}"></div>' if i < len(STAGES)-1 else "")
            + '</div>'
        )
    return f'<div class="stepper">{"".join(parts)}</div>'


# ═════════════════════════════════════════════════════════════════
# Pipeline execution
# ═════════════════════════════════════════════════════════════════
result     = _s("analysis_result")
has_result = result is not None
active_tab = _s("active_tab", "overview")


# ─── Render the run button + pipeline status in Overview ──────────
def _render_run_area(placeholder_progress, placeholder_stepper,
                     placeholder_log, placeholder_run):
    """Render the Run button + live progress – called only from Overview."""
    completed_keys = set()
    active_node    = _s("running_step", "")

    if has_result:
        completed_keys = {s.lower() for s in (result.get("completed_steps") or [])}
        active_node    = ""

    # Stepper
    placeholder_stepper.markdown(
        _stepper_html(completed_keys, active_node),
        unsafe_allow_html=True,
    )

    # Progress (only while running)
    pct   = _s("pipeline_progress", 0)
    plbl  = _s("pipeline_label",    "Idle")
    if _s("running_step") and not has_result:
        placeholder_progress.markdown(
            f'<div class="prog-wrap">'
            f'<div class="prog-top"><span class="prog-label">Pipeline Progress</span>'
            f'<span class="prog-pct">{pct}%</span></div>'
            f'<div class="prog-track"><div class="prog-fill" style="width:{pct}%"></div></div>'
            f'<div class="prog-status">{plbl}</div></div>',
            unsafe_allow_html=True,
        )
        logs = _s("pipeline_logs", [])
        html = "".join(
            f'<div class="lg {e["k"]}">'
            f'<span class="ts">[{e["ts"]}]</span>'
            f'<span class="msg">{e["m"]}</span></div>'
            for e in logs[-35:]
        )
        placeholder_log.markdown(
            f'<div class="terminal">{html}</div>',
            unsafe_allow_html=True,
        )
    else:
        placeholder_progress.empty()
        placeholder_log.empty()

    # Run button
    sc = "idle"
    sl = "Ready" if uploaded_file else "Upload a dataset first"
    if has_result:   sc, sl = "done",    "Analysis complete"
    if _s("running_step"): sc, sl = "running", "Running..."

    with placeholder_run:
        col_lbl, col_btn = st.columns([3, 1])
        with col_lbl:
            st.markdown(
                f'<div class="pill {sc}" style="margin-top:4px">'
                f'<span class="pill-dot"></span>{sl}</div>',
                unsafe_allow_html=True,
            )
        with col_btn:
            return st.button(
                "🚀 Run Analysis",
                type="primary",
                disabled=(uploaded_file is None or bool(_s("running_step"))),
                use_container_width=True,
            )


# ═════════════════════════════════════════════════════════════════
# ── TAB: OVERVIEW ────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════
if active_tab == "overview":

    st.markdown(
        '<div class="hero-title">AI Data Scientist</div>'
        '<div class="hero-sub">Autonomous end-to-end pipeline · 9 stages · PDF report generation</div>'
        '<div style="margin-bottom:20px"></div>',
        unsafe_allow_html=True,
    )

    # ── Run controls (top-right of Overview) ─────────────────────
    ph_progress = st.empty()
    ph_stepper  = st.empty()
    ph_log      = st.empty()
    ph_run      = st.empty()

    run_clicked = _render_run_area(ph_progress, ph_stepper, ph_log, ph_run)

    # ── Dataset preview ───────────────────────────────────────────
    if preview_df is not None:
        st.markdown(sh("📋", "Dataset Preview"), unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(kpi("p","📋","Rows", f"{preview_df.shape[0]:,}"), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi("b","🗂","Columns", preview_df.shape[1]), unsafe_allow_html=True)
        with c3:
            mv = int(preview_df.isnull().sum().sum())
            st.markdown(kpi("a","⚠️","Missing", f"{mv:,}"), unsafe_allow_html=True)
        with c4:
            mb = round(preview_df.memory_usage(deep=True).sum() / 1e6, 2)
            st.markdown(kpi("g","💾","Memory", f"{mb} MB"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(preview_df.head(10), use_container_width=True)

    elif uploaded_file is None:
        st.markdown(
            '<div style="text-align:center;padding:70px 20px;color:#374168">'
            '<div style="font-size:3.5rem;margin-bottom:14px;opacity:.7">📂</div>'
            '<div style="font-size:1rem;font-weight:600;color:#4b5680">Upload a dataset from the sidebar to get started</div>'
            '<div style="font-size:.82rem;margin-top:6px;color:#2d3248">Supports CSV, Excel, and PDF</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Completed steps ───────────────────────────────────────────
    if has_result:
        done = [s for s in (result.get("completed_steps") or [])
                if not s.lower().startswith("supervisor")]
        st.markdown(sh("⚙️", "Pipeline Completed"), unsafe_allow_html=True)
        cols = st.columns(3)
        for i, step in enumerate(done):
            with cols[i % 3]:
                st.success(f"✓ {step}")

    # ─── EXECUTE pipeline when button clicked ─────────────────────
    if run_clicked and uploaded_file is not None:
        _ss("pipeline_logs",     [])
        _ss("pipeline_progress", 5)
        _ss("pipeline_label",    "Compiling pipeline graph...")
        _ss("running_step",      "profiler")
        _ss("analysis_result",   None)

        logs = [{"m": "Pipeline initialized", "k": "info", "ts": time.strftime("%H:%M:%S")}]
        completed_keys = set()

        def _update_ui(pct, label, active_node, log_items):
            ph_stepper.markdown(_stepper_html(completed_keys, active_node), unsafe_allow_html=True)
            ph_progress.markdown(
                f'<div class="prog-wrap">'
                f'<div class="prog-top"><span class="prog-label">Pipeline Progress</span>'
                f'<span class="prog-pct">{pct}%</span></div>'
                f'<div class="prog-track"><div class="prog-fill" style="width:{pct}%"></div></div>'
                f'<div class="prog-status">{label}</div></div>',
                unsafe_allow_html=True,
            )
            html = "".join(
                f'<div class="lg {e["k"]}">'
                f'<span class="ts">[{e["ts"]}]</span>'
                f'<span class="msg">{e["m"]}</span></div>'
                for e in log_items[-35:]
            )
            ph_log.markdown(f'<div class="terminal">{html}</div>', unsafe_allow_html=True)

        _update_ui(5, "Compiling workflow graph...", "profiler", logs)

        try:
            from workflows.graph import build_graph
            from gemini_guard import reset_gemini_circuit_breaker
            reset_gemini_circuit_breaker()

            _compiled_graph = build_graph()
            logs.append({"m": "LangGraph workflow compiled successfully", "k": "done", "ts": time.strftime("%H:%M:%S")})
            _update_ui(8, "Starting stage 1: Profiler...", "profiler", logs)

            _init_state = {
                "dataset_path":    dataset_path,
                "dataframe":       preview_df,
                "completed_steps": [],
                "messages":        [],
            }

            accumulated_state = dict(_init_state)

            NODE_ORDER = [
                "profiler", "quality", "cleaning", "eda", "visualization",
                "preprocessing", "ml", "critic", "reporter"
            ]

            for chunk in _compiled_graph.stream(_init_state):
                node_name = list(chunk.keys())[0]
                node_data = chunk[node_name]
                accumulated_state.update(node_data)

                if "completed_steps" in node_data:
                    accumulated_state["completed_steps"] = node_data["completed_steps"]
                completed_keys = {s.lower() for s in accumulated_state.get("completed_steps", [])}

                p0, p1, plbl = PROG_MAP.get(node_name, (50, 60, f"{node_name}..."))
                logs.append({"m": f"✓  {node_name.upper()} completed", "k": "done", "ts": time.strftime("%H:%M:%S")})

                curr_idx = NODE_ORDER.index(node_name) if node_name in NODE_ORDER else -1
                next_node = NODE_ORDER[curr_idx + 1] if curr_idx >= 0 and curr_idx + 1 < len(NODE_ORDER) else ""

                if next_node:
                    _, _, next_lbl = PROG_MAP.get(next_node, (p1, p1, f"Running {next_node}..."))
                    logs.append({"m": f"▶  Starting {next_node.upper()}...", "k": "step", "ts": time.strftime("%H:%M:%S")})
                    _update_ui(p1, next_lbl, next_node, logs)
                else:
                    _update_ui(100, "Analysis complete! Finalizing report...", "", logs)

                time.sleep(0.05)

            _ss("analysis_result",   accumulated_state)
            _ss("pipeline_progress", 100)
            _ss("pipeline_label",    "✅ Analysis complete!")
            _ss("running_step",      "")
            _ss("pipeline_logs",     logs)
            st.rerun()

        except Exception as exc:
            logs.append({"m": f"✗ Error: {exc}", "k": "error", "ts": time.strftime("%H:%M:%S")})
            _ss("running_step",  "")
            _ss("pipeline_logs", logs)
            st.error(f"Pipeline execution error: {exc}")


# ═════════════════════════════════════════════════════════════════
# ── TAB: DATA QUALITY ────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════
elif active_tab == "quality":
    st.markdown(
        '<div class="hero-title" style="font-size:1.65rem">🛡️ Data Quality</div>'
        '<div class="hero-sub">Quality scoring · missing value analysis · anomaly detection</div>'
        '<div style="margin-bottom:22px"></div>',
        unsafe_allow_html=True,
    )

    if not has_result:
        st.info("Run the analysis pipeline to see data quality results.")
    else:
        quality  = result.get("quality_report") or {}
        cleaning = result.get("cleaning_report") or {}
        qs       = quality.get("quality_score", 0)
        qc       = "#10b981" if qs >= 75 else ("#f59e0b" if qs >= 50 else "#f87171")
        id_cols  = quality.get("id_columns") or []

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(kpi("p","🏆","Quality Score",
                f'<span style="color:{qc}">{qs}</span>'
                '<span style="font-size:.9rem;color:#374168">/100</span>'), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi("b","🔁","Duplicate Rows", quality.get("duplicate_rows",0)), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi("a","🔑","ID Columns Detected", len(id_cols)), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        l, r = st.columns(2)

        with l:
            st.markdown(sh("❓","Missing Values"), unsafe_allow_html=True)
            mv = quality.get("missing_values") or {}
            if mv:
                import pandas as pd
                df_mv = pd.DataFrame([(k,v) for k,v in mv.items()],
                                     columns=["Column","Missing"]).sort_values("Missing",ascending=False)
                st.dataframe(df_mv, use_container_width=True, hide_index=True)
            else:
                st.markdown('<div class="fi ok"><span class="fi-dot">✅</span>'
                            '<span>No missing values detected</span></div>', unsafe_allow_html=True)

        with r:
            st.markdown(sh("🧹","After Cleaning"), unsafe_allow_html=True)
            if cleaning:
                rb = cleaning.get("rows_before","N/A");    ra = cleaning.get("rows_after","N/A")
                cb = cleaning.get("columns_before","N/A"); ca = cleaning.get("columns_after","N/A")
                ma = cleaning.get("missing_values_after") or {}
                rem = sum(ma.values()) if ma else 0
                st.markdown(
                    f'<div style="display:grid;gap:8px">'
                    f'<div class="fi"><span class="fi-dot">📊</span>'
                    f'<span>Rows: <b>{rb}</b> → <b>{ra}</b></span></div>'
                    f'<div class="fi"><span class="fi-dot">🗂</span>'
                    f'<span>Columns: <b>{cb}</b> → <b>{ca}</b></span></div>'
                    f'<div class="fi {"ok" if rem==0 else "warn"}"><span class="fi-dot">'
                    f'{"✅" if rem==0 else "⚠️"}</span>'
                    f'<span>Remaining missing: <b>{rem}</b></span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Cleaning report unavailable.")

        if id_cols:
            st.markdown(sh("🔑","Detected ID Columns"), unsafe_allow_html=True)
            st.markdown("".join(f'<span class="tag">{c}</span>' for c in id_cols),
                        unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# ── TAB: EDA ─────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════
elif active_tab == "eda":
    st.markdown(
        '<div class="hero-title" style="font-size:1.65rem">📈 Exploratory Data Analysis</div>'
        '<div class="hero-sub">Statistical insights · distributions · correlation analysis</div>'
        '<div style="margin-bottom:22px"></div>',
        unsafe_allow_html=True,
    )

    if not has_result:
        st.info("Run the analysis pipeline to see EDA results.")
    else:
        eda = result.get("eda_report") or {}
        ds  = eda.get("dataset_shape") or {}
        if ds:
            c1, c2 = st.columns(2)
            with c1:
                rv = ds.get("rows","N/A")
                st.markdown(kpi("p","📋","Rows", f"{rv:,}" if isinstance(rv,int) else rv), unsafe_allow_html=True)
            with c2:
                st.markdown(kpi("b","🗂","Columns", ds.get("columns","N/A")), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        corrs = eda.get("strong_correlations") or []
        st.markdown(sh("🔗","Strong Correlations"), unsafe_allow_html=True)
        if corrs:
            import pandas as pd
            cdf = pd.DataFrame(corrs) if isinstance(corrs[0], dict) \
                  else pd.DataFrame(corrs, columns=["Feature A","Feature B","Correlation"])
            st.dataframe(cdf, use_container_width=True, hide_index=True)
            st.caption("⚠️ Correlation ≠ causation.")
        else:
            st.markdown('<div class="fi"><span class="fi-dot">ℹ️</span>'
                        '<span>No strong correlations found</span></div>', unsafe_allow_html=True)

        desc = eda.get("descriptive_stats") or eda.get("summary_stats") or {}
        if desc:
            st.markdown(sh("📐","Descriptive Statistics"), unsafe_allow_html=True)
            import pandas as pd
            st.dataframe(pd.DataFrame(desc), use_container_width=True)

        ols = eda.get("outlier_columns") or eda.get("outliers") or []
        if ols:
            st.markdown(sh("📍","Outlier Columns"), unsafe_allow_html=True)
            st.markdown("".join(f'<span class="tag">{c}</span>' for c in ols), unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# ── TAB: VISUALIZATIONS ──────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════
elif active_tab == "visualizations":
    st.markdown(
        '<div class="hero-title" style="font-size:1.65rem">📊 Visualizations</div>'
        '<div class="hero-sub">Auto-generated distribution plots · feature charts · correlation heatmaps</div>'
        '<div style="margin-bottom:22px"></div>',
        unsafe_allow_html=True,
    )

    if not has_result:
        st.info("Run the analysis pipeline to generate visualizations.")
    else:
        vdir = os.path.join(PROJECT_ROOT, "reports", "visualizations")
        if os.path.exists(vdir):
            imgs = sorted([f for f in os.listdir(vdir)
                           if f.lower().endswith((".png",".jpg",".jpeg"))])
            if imgs:
                st.markdown(
                    f'<div class="pill done" style="margin-bottom:18px">'
                    f'<span class="pill-dot"></span>{len(imgs)} charts generated</div>',
                    unsafe_allow_html=True,
                )
                for i in range(0, len(imgs), 2):
                    c1, c2 = st.columns(2)
                    for j, col in [(0,c1),(1,c2)]:
                        idx = i + j
                        if idx >= len(imgs): break
                        fname = imgs[idx]
                        title = os.path.splitext(fname)[0].replace("_"," ").title()
                        with col:
                            st.markdown(
                                f'<div class="viz-card">'
                                f'<div class="viz-hdr">📈 {title}</div>',
                                unsafe_allow_html=True,
                            )
                            st.image(os.path.join(vdir, fname), use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No visualization images were generated.")
        else:
            st.info("Visualization directory not found.")


# ═════════════════════════════════════════════════════════════════
# ── TAB: PREPROCESSING ───────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════
elif active_tab == "preprocessing":
    st.markdown(
        '<div class="hero-title" style="font-size:1.65rem">⚙️ Preprocessing</div>'
        '<div class="hero-sub">Feature engineering decisions by the Preprocessing Agent</div>'
        '<div style="margin-bottom:22px"></div>',
        unsafe_allow_html=True,
    )

    if not has_result:
        st.info("Run the analysis pipeline to see preprocessing results.")
    else:
        plan = result.get("preprocessing_plan") or {}
        if plan:
            def _plan_card(col, icon, title, key):
                with col:
                    item   = plan.get(key) or {}
                    req    = item.get("required", False)
                    method = item.get("method", "N/A")
                    reason = item.get("reason", "")
                    st.markdown(sh(icon, title), unsafe_allow_html=True)
                    cls = "ok" if req else ""
                    ico = "✅" if req else "⬜"
                    body = f"Required · <b>{method}</b>" if req else "Not required"
                    st.markdown(
                        f'<div class="fi {cls}"><span class="fi-dot">{ico}</span>'
                        f'<div><div>{body}</div>'
                        + (f'<div style="font-size:.75rem;color:#374168;margin-top:3px">{reason}</div>' if reason else '')
                        + '</div></div>',
                        unsafe_allow_html=True,
                    )

            c1, c2 = st.columns(2)
            _plan_card(c1, "💉", "Imputation",    "imputation")
            _plan_card(c2, "📏", "Scaling",        "scaling")
            c3, c4 = st.columns(2)
            _plan_card(c3, "🏷️", "Encoding",       "encoding")
            _plan_card(c4, "🔀", "Transformation", "transformation")
            st.caption("Automatically determined by the Preprocessing Agent.")
        else:
            st.info("Preprocessing information unavailable.")


# ═════════════════════════════════════════════════════════════════
# ── TAB: ML MODELS ───────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════
elif active_tab == "ml":
    st.markdown(
        '<div class="hero-title" style="font-size:1.65rem">🤖 Machine Learning</div>'
        '<div class="hero-sub">Model evaluation · performance metrics · best model selection</div>'
        '<div style="margin-bottom:22px"></div>',
        unsafe_allow_html=True,
    )

    if not has_result:
        st.info("Run the analysis pipeline to see ML results.")
    else:
        ml = result.get("ml_report") or {}
        if ml:
            pt = ml.get("problem_type","N/A")
            tc = ml.get("target_column","N/A")
            bm = ml.get("best_model","N/A")
            tr = ml.get("train_samples","N/A")
            te = ml.get("test_samples","N/A")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(kpi("p","🎯","Problem Type",
                    f'<span style="font-size:.95rem;display:block;margin-top:6px">{pt}</span>'), unsafe_allow_html=True)
            with c2:
                st.markdown(kpi("b","🏹","Target Column",
                    f'<span style="font-size:.95rem;display:block;margin-top:6px">{tc}</span>'), unsafe_allow_html=True)
            with c3:
                st.markdown(kpi("g","📚","Train Samples",
                    f"{tr:,}" if isinstance(tr,int) else tr), unsafe_allow_html=True)
            with c4:
                st.markdown(kpi("a","🧪","Test Samples",
                    f"{te:,}" if isinstance(te,int) else te), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            feats = ml.get("features") or {}
            nf = feats.get("numerical") or []
            cf = feats.get("categorical") or []
            fl, fr = st.columns(2)
            with fl:
                st.markdown(sh("🔢","Numerical Features"), unsafe_allow_html=True)
                st.markdown(
                    "".join(f'<span class="tag">{f}</span>' for f in nf)
                    or '<div class="fi"><span>None</span></div>',
                    unsafe_allow_html=True,
                )
            with fr:
                st.markdown(sh("🏷️","Categorical Features"), unsafe_allow_html=True)
                st.markdown(
                    "".join(f'<span class="tag">{f}</span>' for f in cf)
                    or '<div class="fi"><span>None</span></div>',
                    unsafe_allow_html=True,
                )

            models = ml.get("models") or {}
            if models:
                st.markdown(sh("📊","Model Performance"), unsafe_allow_html=True)
                import pandas as pd
                mkeys = ["accuracy","precision","recall","f1_score",
                         "roc_auc","mae","mse","rmse","r2_score"]
                rows = []
                for mn, mm in models.items():
                    row = {"Model": mn}
                    for mk in mkeys:
                        if mk in mm:
                            v = mm[mk]
                            row[mk] = round(v,4) if isinstance(v,float) else v
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown(sh("🏆","Best Model"), unsafe_allow_html=True)
            st.markdown(
                f'<div class="fi ok" style="background:rgba(16,185,129,.06);border-color:rgba(16,185,129,.22)">'
                f'<span style="font-size:1.4rem">🏆</span>'
                f'<span style="color:#34d399;font-weight:700;font-size:1.05rem">{bm}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Machine learning report unavailable.")


# ═════════════════════════════════════════════════════════════════
# ── TAB: VALIDATION ──────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════
elif active_tab == "validation":
    st.markdown(
        '<div class="hero-title" style="font-size:1.65rem">🔎 Critic & Validation</div>'
        '<div class="hero-sub">Independent AI review of pipeline decisions & model performance</div>'
        '<div style="margin-bottom:22px"></div>',
        unsafe_allow_html=True,
    )

    if not has_result:
        st.info("Run the analysis pipeline to see validation results.")
    else:
        critic   = result.get("critic_report") or {}
        findings = critic.get("findings") or []
        warnings = critic.get("warnings") or []
        rec      = critic.get("recommendation","")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(kpi("b","📋","Findings", len(findings)), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi("a","⚠️","Warnings", len(warnings)), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if findings:
            st.markdown(sh("📋","Findings"), unsafe_allow_html=True)
            for f in findings:
                st.markdown(
                    f'<div class="fi"><span class="fi-dot">•</span><span>{f}</span></div>',
                    unsafe_allow_html=True,
                )

        if warnings:
            st.markdown(sh("⚠️","Warnings"), unsafe_allow_html=True)
            for w in warnings:
                st.markdown(
                    f'<div class="fi warn"><span class="fi-dot">⚠️</span><span>{w}</span></div>',
                    unsafe_allow_html=True,
                )

        if rec:
            st.markdown(sh("💡","Recommendation"), unsafe_allow_html=True)
            st.markdown(
                f'<div class="fi info-c">'
                f'<span style="font-size:1.2rem">💡</span>'
                f'<span style="color:#22d3ee">{rec}</span></div>',
                unsafe_allow_html=True,
            )

        rs = result.get("report_source","")
        st.markdown(sh("🤖","Report Source"), unsafe_allow_html=True)
        if rs == "gemini":
            st.success("🤖 Report generated using Gemini.")
        elif rs == "deterministic_fallback":
            st.warning("⚠️ Gemini was unavailable — deterministic fallback used.")
        else:
            st.info("Report source information unavailable.")


# ═════════════════════════════════════════════════════════════════
# ── TAB: REPORT  (PDF only download) ─────────────────────────────
# ═════════════════════════════════════════════════════════════════
elif active_tab == "report":
    st.markdown(
        '<div class="hero-title" style="font-size:1.65rem">📄 Final Report</div>'
        '<div class="hero-sub">Download your comprehensive data science report as PDF</div>'
        '<div style="margin-bottom:22px"></div>',
        unsafe_allow_html=True,
    )

    if not has_result:
        st.info("Run the analysis pipeline to generate the report.")
    else:
        pdf_path = result.get(
            "pdf_report_path",
            os.path.join(PROJECT_ROOT, "reports", "final_report.pdf")
        )

        # ── PDF (primary) ──────────────────────────────────────
        st.markdown(sh("📥","Download Report"), unsafe_allow_html=True)

        if os.path.exists(pdf_path):
            sz = round(os.path.getsize(pdf_path) / 1024, 1)
            st.markdown(
                f'<div class="dl-card">'
                f'<div class="dl-ico">📕</div>'
                f'<div><div class="dl-title">AI Data Science Report — PDF</div>'
                f'<div class="dl-desc">Full analysis: insights, metrics, visualizations · {sz} KB</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            with open(pdf_path, "rb") as fh:
                st.download_button(
                    "⬇️  Download PDF Report",
                    data=fh,
                    file_name="ai_data_science_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                )
        else:
            st.warning("📕 PDF report not yet generated. Run the pipeline first.")

        # ── Dataset outputs ──────────────────────────────────────
        st.markdown(sh("📦","Dataset Outputs"), unsafe_allow_html=True)

        cp = result.get("cleaned_dataset_path",
                        os.path.join(PROJECT_ROOT,"data","cleaned_dataset.csv"))
        pp = result.get("preprocessed_dataset_path",
                        os.path.join(PROJECT_ROOT,"data","preprocessed_dataset.csv"))

        d1, d2 = st.columns(2)
        with d1:
            st.markdown(
                '<div class="dl-card">'
                '<div class="dl-ico" style="background:rgba(16,185,129,.12)">🧹</div>'
                '<div><div class="dl-title">Cleaned Dataset</div>'
                '<div class="dl-desc">Deduplicated, imputed CSV</div></div></div>',
                unsafe_allow_html=True,
            )
            if os.path.exists(cp):
                with open(cp,"rb") as fh:
                    st.download_button("⬇️  Cleaned CSV", data=fh,
                                       file_name="cleaned_dataset.csv",
                                       mime="text/csv", use_container_width=True)
            else:
                st.warning("Cleaned dataset not available.")

        with d2:
            st.markdown(
                '<div class="dl-card">'
                '<div class="dl-ico" style="background:rgba(6,182,212,.12)">⚙️</div>'
                '<div><div class="dl-title">Preprocessed Dataset</div>'
                '<div class="dl-desc">Encoded, scaled ML-ready CSV</div></div></div>',
                unsafe_allow_html=True,
            )
            if os.path.exists(pp):
                with open(pp,"rb") as fh:
                    st.download_button("⬇️  Preprocessed CSV", data=fh,
                                       file_name="preprocessed_dataset.csv",
                                       mime="text/csv", use_container_width=True)
            else:
                st.warning("Preprocessed dataset not available.")

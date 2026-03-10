"""
SpermAI — Clinical Morphology Intelligence Platform
Streamlit UI · v2.0

Run: streamlit run app.py
Requires: streamlit, requests, plotly, pillow, pandas
"""

import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64
import time

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG — must be first Streamlit call
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SpermAI — Clinical Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# ══════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Medical Dark / Precision Aesthetic
# Inspired by Siemens Healthineers & high-end lab equipment UIs
# Font: DM Sans (body) + DM Mono (data) + Instrument Serif (headings)
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&family=Instrument+Serif:ital@0;1&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #080E1A;
    color: #C8D6E8;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0C1525 !important;
    border-right: 1px solid #1C2D45;
}
[data-testid="stSidebar"] * { color: #8BA3BF !important; }
[data-testid="stSidebar"] .stRadio label { color: #A8C0D6 !important; }

/* ── Top navigation bar ── */
.topbar {
    background: linear-gradient(135deg, #0C1525 0%, #0F1E35 100%);
    border-bottom: 1px solid #1C3455;
    padding: 1rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2rem;
    border-radius: 0 0 12px 12px;
}
.topbar-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}
.topbar-logo {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #0070CC, #00A8FF);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}
.topbar-title {
    font-family: 'Instrument Serif', serif;
    font-size: 1.5rem;
    color: #E8F1FA;
    letter-spacing: -0.3px;
}
.topbar-sub {
    font-size: 0.72rem;
    color: #4A6B8A;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 1px;
}
.topbar-right {
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.topbar-badge {
    background: #0A1E38;
    border: 1px solid #1C3A5C;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.72rem;
    color: #4A9FD4;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.5px;
}
.status-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    background: #041525;
    border: 1px solid #1C3A5C;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
}
.dot-live {
    width: 7px; height: 7px;
    background: #00D68F;
    border-radius: 50%;
    box-shadow: 0 0 6px #00D68F;
    animation: pulse 2s infinite;
}
.dot-dead {
    width: 7px; height: 7px;
    background: #FF4757;
    border-radius: 50%;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Section headings ── */
.section-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #2D5C8A;
    margin-bottom: 0.6rem;
}

/* ── Cards / Panels ── */
.panel {
    background: #0C1525;
    border: 1px solid #1A2E48;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.panel-sm {
    background: #0C1525;
    border: 1px solid #1A2E48;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
}

/* ── KPI tiles ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem;
    margin-bottom: 1.2rem;
}
.kpi {
    background: #0C1525;
    border: 1px solid #1A2E48;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
}
.kpi::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-normal::before  { background: linear-gradient(90deg, #00D68F, transparent); }
.kpi-abnormal::before { background: linear-gradient(90deg, #FF9F43, transparent); }
.kpi-nonsperm::before { background: linear-gradient(90deg, #778CA3, transparent); }
.kpi-flagged::before { background: linear-gradient(90deg, #FF4757, transparent); }

.kpi-label {
    font-size: 0.65rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4A6B8A;
    font-weight: 600;
}
.kpi-value {
    font-family: 'Instrument Serif', serif;
    font-size: 2.2rem;
    color: #D8EAF8;
    line-height: 1.1;
    margin: 0.3rem 0 0.1rem;
}
.kpi-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #3A5F80;
}

/* ── Result display ── */
.result-main {
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.result-main::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(255,255,255,0.03), transparent 70%);
    pointer-events: none;
}
.result-normal-bg   { background: #041A10; border-color: #00D68F33; }
.result-abnormal-bg { background: #1A0F04; border-color: #FF9F4333; }
.result-nonsperm-bg { background: #0D1420; border-color: #778CA333; }

.result-class {
    font-family: 'Instrument Serif', serif;
    font-size: 2.4rem;
    font-weight: 400;
    letter-spacing: -0.5px;
    line-height: 1;
    margin: 0.4rem 0;
}
.result-class-normal   { color: #00D68F; }
.result-class-abnormal { color: #FF9F43; }
.result-class-nonsperm { color: #A0B4C8; }

.result-conf {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #4A7A9B;
    margin-top: 0.3rem;
}

/* ── Flag bar ── */
.flag-alert {
    background: #1A0A00;
    border: 1px solid #FF9F4355;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.78rem;
    color: #FF9F43;
    margin-top: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.flag-clear {
    background: #001A0D;
    border: 1px solid #00D68F33;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.78rem;
    color: #00D68F;
    margin-top: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Probability bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.7rem;
}
.prob-name {
    font-size: 0.75rem;
    font-weight: 600;
    color: #6A8FAF;
    width: 80px;
    text-align: right;
}
.prob-track {
    flex: 1;
    height: 6px;
    background: #0F1E33;
    border-radius: 10px;
    overflow: hidden;
}
.prob-fill-n { height: 100%; border-radius: 10px; background: linear-gradient(90deg, #00B377, #00D68F); }
.prob-fill-a { height: 100%; border-radius: 10px; background: linear-gradient(90deg, #CC7F30, #FF9F43); }
.prob-fill-s { height: 100%; border-radius: 10px; background: linear-gradient(90deg, #5C6E80, #778CA3); }
.prob-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #5A8AB0;
    width: 42px;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #060F1C;
    border: 1px dashed #1C3A5C;
    border-radius: 10px;
    padding: 0.5rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: #2A6090;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #005EA3, #0070CC);
    color: white;
    border: none;
    border-radius: 7px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    letter-spacing: 0.3px;
    padding: 0.55rem 1.5rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #006EBD, #0090F0);
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0, 112, 204, 0.3);
}

/* ── Table ── */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    border: 1px solid #1A2E48;
    overflow: hidden;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: #060F1C;
    border-radius: 8px;
    gap: 0;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #4A7A9B;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: #0C1E35 !important;
    color: #A8D4F5 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0C1525;
    border: 1px solid #1A2E48;
    border-radius: 7px;
    color: #5A8AB0 !important;
    font-size: 0.8rem;
}

/* ── Sidebar nav ── */
.sidebar-brand {
    padding: 1.2rem 0 1rem;
    border-bottom: 1px solid #1A2E48;
    margin-bottom: 1.2rem;
}
.sidebar-heading {
    font-size: 0.6rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #2D4D6A !important;
    font-weight: 700;
    margin-bottom: 0.5rem;
    padding-top: 0.5rem;
}
.sidebar-info-item {
    display: flex;
    justify-content: space-between;
    padding: 0.35rem 0;
    font-size: 0.78rem;
    border-bottom: 1px solid #0F1E30;
}
.sidebar-info-key { color: #3D607A !important; }
.sidebar-info-val {
    font-family: 'DM Mono', monospace;
    color: #5A8AB0 !important;
    font-size: 0.72rem;
}

/* ── WHO reference ── */
.who-box {
    background: #060F1C;
    border: 1px solid #1A2E48;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 0.5rem;
}
.who-row {
    display: flex;
    justify-content: space-between;
    padding: 0.3rem 0;
    font-size: 0.77rem;
    border-bottom: 1px solid #0F1E30;
}
.who-param { color: #4A7A9B; }
.who-val { font-family: 'DM Mono', monospace; color: #2A5C80; }

/* ── Disclaimer ── */
.disclaimer-bar {
    background: #060F1C;
    border: 1px solid #1A2E48;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.7rem;
    color: #2D4D6A;
    text-align: center;
    margin-top: 1.5rem;
    letter-spacing: 0.3px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #080E1A; }
::-webkit-scrollbar-thumb { background: #1A2E48; border-radius: 10px; }

/* ── Misc text overrides ── */
p, li, div { color: #8BA3BF; }
h1, h2, h3 { color: #C8D6E8; }
.stMarkdown { color: #8BA3BF; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# API HELPERS
# ══════════════════════════════════════════════════════════════
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, {}


def predict_single(image_bytes, filename):
    files = {"file": (filename, image_bytes, "image/jpeg")}
    r = requests.post(f"{API_URL}/predict", files=files, timeout=30)
    r.raise_for_status()
    return r.json()


def predict_batch(file_list):
    files = [("files", (f["name"], f["bytes"], "image/jpeg")) for f in file_list]
    r = requests.post(f"{API_URL}/batch-predict", files=files, timeout=120)
    r.raise_for_status()
    return r.json()


def get_sessions():
    try:
        r = requests.get(f"{API_URL}/sessions", timeout=5)
        return r.json().get("sessions", [])
    except Exception:
        return []


def get_stats():
    try:
        r = requests.get(f"{API_URL}/stats", timeout=5)
        return r.json()
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════
CHART_THEME = {
    "paper_bgcolor": "#0C1525",
    "plot_bgcolor": "#0C1525",
    "font": {"family": "DM Sans", "color": "#4A7A9B"},
    "margin": {"t": 20, "b": 20, "l": 20, "r": 20},
}

CLASS_COLORS = {
    "Normal": "#00D68F",
    "Abnormal": "#FF9F43",
    "Non-Sperm": "#778CA3",
}


def gauge_chart(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number={"suffix": "%", "font": {"size": 32, "color": "#D8EAF8", "family": "DM Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "#1A3050",
                     "tickfont": {"size": 9, "color": "#2A4A6A"}},
            "bar": {"color": "#0070CC", "thickness": 0.22},
            "bgcolor": "#060F1C",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 70],  "color": "#0F0A05"},
                {"range": [70, 85], "color": "#0F0D05"},
                {"range": [85, 100], "color": "#05100A"},
            ],
            "threshold": {
                "line": {"color": "#FF4757", "width": 2},
                "thickness": 0.7, "value": 70,
            },
        },
    ))
    fig.update_layout(height=200, **CHART_THEME)
    return fig


def donut_chart(summary):
    labels = list(summary.keys())
    values = list(summary.values())
    colors = [CLASS_COLORS.get(l, "#778CA3") for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#080E1A", width=2)),
        textinfo="percent",
        textfont={"size": 11, "family": "DM Mono", "color": "#080E1A"},
        hovertemplate="<b>%{label}</b><br>%{value} samples<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        height=260,
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1.05, y=0.5,
            font={"size": 11, "color": "#5A8AB0"},
        ),
        **CHART_THEME,
    )
    return fig


def confidence_histogram(predictions):
    confs = [p.get("confidence", 0) for p in predictions if "confidence" in p]
    if not confs:
        return go.Figure()
    fig = go.Figure(go.Histogram(
        x=confs, nbinsx=20,
        marker_color="#0070CC",
        marker_line_color="#0050AA",
        marker_line_width=0.5,
        opacity=0.85,
    ))
    fig.add_vline(x=0.70, line_dash="dash", line_color="#FF4757",
                  annotation_text="Threshold", annotation_font_size=10,
                  annotation_font_color="#FF4757")
    fig.update_layout(
        height=200,
        xaxis=dict(title="Confidence", range=[0, 1], gridcolor="#0F1E33",
                   tickformat=".0%", tickfont={"size": 9}),
        yaxis=dict(title="Count", gridcolor="#0F1E33", tickfont={"size": 9}),
        bargap=0.05,
        **CHART_THEME,
    )
    return fig


def timeline_chart(predictions):
    if not predictions:
        return go.Figure()
    rows = []
    for i, p in enumerate(predictions):
        if "confidence" in p:
            rows.append({"idx": i + 1, "conf": p["confidence"],
                         "cls": p.get("prediction", ""), "fn": p.get("filename", f"#{i+1}")})
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows)
    colors = [CLASS_COLORS.get(c, "#778CA3") for c in df["cls"]]
    fig = go.Figure()
    for cls, color in CLASS_COLORS.items():
        mask = df["cls"] == cls
        if mask.any():
            fig.add_trace(go.Scatter(
                x=df[mask]["idx"], y=df[mask]["conf"],
                mode="markers", name=cls,
                marker=dict(color=color, size=7, opacity=0.85),
                hovertemplate="<b>%{text}</b><br>Confidence: %{y:.1%}<extra></extra>",
                text=df[mask]["fn"].tolist(),
            ))
    fig.add_hline(y=0.70, line_dash="dash", line_color="#FF475755",
                  annotation_text="Review threshold", annotation_font_size=9,
                  annotation_font_color="#FF4757")
    fig.update_layout(
        height=220,
        xaxis=dict(title="Sample Index", gridcolor="#0F1E33", tickfont={"size": 9}),
        yaxis=dict(title="Confidence", range=[0, 1.05], tickformat=".0%",
                   gridcolor="#0F1E33", tickfont={"size": 9}),
        legend=dict(font={"size": 10, "color": "#5A8AB0"}, bgcolor="rgba(0,0,0,0)"),
        **CHART_THEME,
    )
    return fig


# ══════════════════════════════════════════════════════════════
# RESULT RENDERING HELPERS
# ══════════════════════════════════════════════════════════════
def render_result_card(prediction, confidence, flagged, probs):
    cls_map = {
        "Normal":    ("result-normal-bg",   "result-class-normal",   "✦"),
        "Abnormal":  ("result-abnormal-bg", "result-class-abnormal", "⬡"),
        "Non-Sperm": ("result-nonsperm-bg", "result-class-nonsperm", "◈"),
    }
    bg, text_cls, icon = cls_map.get(prediction, ("result-nonsperm-bg", "result-class-nonsperm", "◇"))
    flag_html = (
        '<div class="flag-alert">⚠ Below confidence threshold — manual andrologist review recommended</div>'
        if flagged else
        '<div class="flag-clear">✓ High-confidence classification</div>'
    )
    st.markdown(f"""
    <div class="result-main {bg}">
        <div style="font-size:0.65rem;letter-spacing:2.5px;text-transform:uppercase;color:#2D5C8A;margin-bottom:0.3rem">CLASSIFICATION RESULT</div>
        <div style="font-size:2rem;color:#2D5C8A">{icon}</div>
        <div class="result-class {text_cls}">{prediction}</div>
        <div class="result-conf">CONFIDENCE &nbsp;·&nbsp; {confidence:.1%}</div>
    </div>
    {flag_html}
    """, unsafe_allow_html=True)

    # Probability bars
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Class Probabilities</div>', unsafe_allow_html=True)
    bar_map = {"Normal": "prob-fill-n", "Abnormal": "prob-fill-a", "Non-Sperm": "prob-fill-s"}
    for cls, val in probs.items():
        fill_cls = bar_map.get(cls, "prob-fill-s")
        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-name">{cls}</div>
            <div class="prob-track"><div class="{fill_cls}" style="width:{val*100:.1f}%"></div></div>
            <div class="prob-val">{val:.3f}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    api_ok, api_info = check_api()

    st.markdown("""
    <div class="sidebar-brand">
        <div style="font-family:'Instrument Serif',serif;font-size:1.25rem;color:#A8C8E8 !important">SpermAI</div>
        <div style="font-size:0.65rem;letter-spacing:1.8px;text-transform:uppercase;color:#2D4D6A !important;margin-top:2px">CLINICAL PLATFORM v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    # API Status
    st.markdown('<div class="sidebar-heading">SYSTEM</div>', unsafe_allow_html=True)
    if api_ok:
        device = api_info.get("device", "cpu").upper()
        st.markdown(f"""
        <div class="panel-sm" style="display:flex;align-items:center;justify-content:space-between">
            <span style="display:flex;align-items:center;gap:7px">
                <span class="dot-live"></span>
                <span style="font-size:0.78rem;color:#00D68F !important">API ONLINE</span>
            </span>
            <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#2A6090 !important">{device}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="panel-sm">
            <span class="dot-dead"></span>
            <span style="font-size:0.78rem;color:#FF4757 !important"> API OFFLINE</span>
        </div>
        """, unsafe_allow_html=True)
        st.code("uvicorn api.main:app --port 8000", language="bash")

    # Model info
    st.markdown('<div class="sidebar-heading" style="margin-top:1.2rem">MODEL INFO</div>', unsafe_allow_html=True)
    model_info = [
        ("Architecture", api_info.get("model_type", "EfficientNet-B0")),
        ("Classes", str(api_info.get("num_classes", 3))),
        ("Threshold", f"{api_info.get('threshold', 0.70):.0%}"),
        ("Device", api_info.get("device", "—").upper()),
    ]
    info_html = '<div class="who-box">'
    for k, v in model_info:
        info_html += f'<div class="who-row"><span class="who-param">{k}</span><span class="who-val">{v}</span></div>'
    info_html += "</div>"
    st.markdown(info_html, unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="sidebar-heading" style="margin-top:1.2rem">WORKSPACE</div>', unsafe_allow_html=True)
    app_mode = st.radio(
        "",
        ["🔬  Single Analysis", "📦  Batch Processing", "📊  Session Log"],
        label_visibility="collapsed",
    )

    # WHO reference
    st.markdown('<div class="sidebar-heading" style="margin-top:1.2rem">WHO 2021 REFERENCE</div>', unsafe_allow_html=True)
    who_params = [
        ("Normal morphology", "≥ 4%"),
        ("Total motility", "≥ 42%"),
        ("Progressive motility", "≥ 30%"),
        ("Sperm concentration", "≥ 16×10⁶/mL"),
        ("Total sperm count", "≥ 39×10⁶"),
    ]
    who_html = '<div class="who-box">'
    for p, v in who_params:
        who_html += f'<div class="who-row"><span class="who-param">{p}</span><span class="who-val">{v}</span></div>'
    who_html += "</div>"
    st.markdown(who_html, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-bar" style="margin-top:1.5rem">
        🏥 Research Use Only<br>Not validated for clinical diagnosis
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════════════
now_str = datetime.now().strftime("%d %b %Y  ·  %H:%M")
status_dot = '<span class="dot-live"></span><span style="font-size:0.72rem;color:#00D68F">ONLINE</span>' if api_ok else '<span class="dot-dead"></span><span style="font-size:0.72rem;color:#FF4757">OFFLINE</span>'

st.markdown(f"""
<div class="topbar">
    <div class="topbar-brand">
        <div class="topbar-logo">🔬</div>
        <div>
            <div class="topbar-title">SpermAI</div>
            <div class="topbar-sub">Morphology Intelligence Platform</div>
        </div>
    </div>
    <div class="topbar-right">
        <div class="topbar-badge">WHO 2021 · 3-Class Model</div>
        <div class="status-pill">{status_dot}</div>
        <div class="topbar-badge">🕐 {now_str}</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not api_ok:
    st.error("⚠️ Backend API is unreachable. Start the server: `uvicorn api.main:app --port 8000`")
    st.stop()


# ══════════════════════════════════════════════════════════════
# MODE: SINGLE ANALYSIS
# ══════════════════════════════════════════════════════════════
if "Single" in app_mode:

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="section-label">Upload Microscopy Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag & drop or click to browse — TIFF, PNG, JPEG, BMP",
            type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
            label_visibility="collapsed",
        )

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, use_column_width=True, caption=None)
            st.markdown(f"""
            <div style="font-size:0.72rem;color:#2D5C8A;margin-top:0.4rem;font-family:'DM Mono',monospace;
                        display:flex;gap:1.5rem;padding:0.5rem 0">
                <span>FILE &nbsp;{uploaded.name}</span>
                <span>RES &nbsp;{img.size[0]}×{img.size[1]}px</span>
                <span>SIZE &nbsp;{len(uploaded.getvalue())//1024} KB</span>
                <span>MODE &nbsp;{img.mode}</span>
            </div>
            """, unsafe_allow_html=True)

            if st.button("▶  Run Morphology Analysis", type="primary", use_container_width=True):
                with st.spinner("Running inference…"):
                    try:
                        uploaded.seek(0)
                        result = predict_single(uploaded.read(), uploaded.name)
                        st.session_state.single_result = result
                        st.session_state.single_image = img.copy()
                    except Exception as e:
                        st.error(f"Inference error: {e}")
        else:
            st.markdown("""
            <div style="text-align:center;padding:5rem 2rem;border:1px dashed #1A2E48;border-radius:10px">
                <div style="font-size:2.5rem;opacity:0.3">🔬</div>
                <div style="font-size:0.82rem;color:#2D4D6A;margin-top:0.5rem">
                    Upload a sperm microscopy image to begin
                </div>
                <div style="font-size:0.7rem;color:#1A3050;margin-top:0.3rem">
                    TIFF · PNG · JPEG · BMP
                </div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-label">Analysis Result</div>', unsafe_allow_html=True)

        if "single_result" in st.session_state:
            r = st.session_state.single_result
            prediction = r.get("prediction", "Unknown")
            confidence = r.get("confidence", 0.0)
            flagged    = r.get("flagged_for_review", False)
            probs      = r.get("probabilities", {})
            proc_ms    = r.get("processing_time_ms")
            morphology_idx = r.get("morphology_index", 0)

            render_result_card(prediction, confidence, flagged, probs)

            # Gauge + morphology index
            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            g1, g2 = st.columns(2)
            with g1:
                st.markdown('<div class="section-label">Model Confidence</div>', unsafe_allow_html=True)
                st.plotly_chart(gauge_chart(confidence), use_container_width=True)
            with g2:
                st.markdown('<div class="section-label">Report Summary</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="who-box" style="margin-top:0.2rem">
                    <div class="who-row"><span class="who-param">Classification</span><span class="who-val">{prediction}</span></div>
                    <div class="who-row"><span class="who-param">Confidence</span><span class="who-val">{confidence:.4f}</span></div>
                    <div class="who-row"><span class="who-param">Morphology Index</span><span class="who-val">{morphology_idx:.1f}%</span></div>
                    <div class="who-row"><span class="who-param">Inference</span><span class="who-val">{f"{proc_ms:.0f} ms" if proc_ms else "—"}</span></div>
                    <div class="who-row"><span class="who-param">Flagged</span><span class="who-val" style="color:{'#FF9F43' if flagged else '#00D68F'} !important">{"YES ⚠" if flagged else "NO ✓"}</span></div>
                    <div class="who-row"><span class="who-param">Timestamp</span><span class="who-val">{datetime.now().strftime("%H:%M:%S")}</span></div>
                </div>
                """, unsafe_allow_html=True)

            # Full report expander
            with st.expander("📋  Full Diagnostic Report"):
                report_md = f"""
| Field | Value |
|:---|:---|
| **Prediction** | {prediction} |
| **Confidence Score** | {confidence:.6f} |
| **Flagged for Review** | {"Yes — below 70% threshold" if flagged else "No"} |
| **Normal Probability** | {probs.get("Normal", 0):.6f} |
| **Abnormal Probability** | {probs.get("Abnormal", 0):.6f} |
| **Non-Sperm Probability** | {probs.get("Non-Sperm", 0):.6f} |
| **Morphology Index** | {morphology_idx:.1f}% |
| **Inference Time** | {f"{proc_ms:.1f} ms" if proc_ms else "N/A"} |
| **Image File** | {r.get("filename", "—")} |
| **Image Resolution** | {r.get("image_size", ["—","—"])[1]}×{r.get("image_size", ["—","—"])[0]} px |
| **Session ID** | `{r.get("session_id", "—")}` |
| **Analysis Time** | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
| **Model** | EfficientNet-B0 / ResNet50 |
| **WHO Criteria** | 2021 |
                """
                st.markdown(report_md)
                st.markdown("""
                > ⚕️ **Clinical Note:** This result is generated by an AI classification model and must be reviewed by a 
                > qualified andrologist before any clinical decision is made. Normal morphology threshold per WHO 2021 is ≥ 4%.
                """)
        else:
            st.markdown("""
            <div style="text-align:center;padding:5rem 2rem;border:1px dashed #1A2E48;border-radius:10px">
                <div style="font-size:0.82rem;color:#2D4D6A">
                    Results will appear here after analysis
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MODE: BATCH PROCESSING
# ══════════════════════════════════════════════════════════════
elif "Batch" in app_mode:

    st.markdown('<div class="section-label">Batch Morphology Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel-sm" style="margin-bottom:1.2rem">
        <span style="font-size:0.8rem;color:#4A7A9B">
        Upload up to <b style="color:#A8D4F5">50 images</b> for simultaneous analysis. 
        Results include per-sample predictions, session analytics, and exportable reports.
        </span>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop multiple microscopy images",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        n = len(uploaded_files)
        c1, c2, c3 = st.columns(3)
        c1.metric("Images Loaded", n)
        c2.metric("Est. Processing", f"~{n * 0.3:.0f}s")
        c3.metric("Max Batch", "50")

        # Thumbnail preview
        with st.expander(f"🖼  Preview ({min(n, 12)} of {n} images)"):
            tcols = st.columns(6)
            for i, f in enumerate(uploaded_files[:12]):
                with tcols[i % 6]:
                    st.image(Image.open(f), caption=f.name[:12], use_column_width=True)

        if st.button(f"▶  Analyse All {n} Images", type="primary", use_container_width=True):
            progress = st.progress(0, text="Preparing batch…")
            with st.spinner(""):
                try:
                    file_data = []
                    for i, f in enumerate(uploaded_files[:50]):
                        f.seek(0)
                        file_data.append({"name": f.name, "bytes": f.read()})
                        progress.progress((i + 1) / min(len(uploaded_files), 50),
                                          text=f"Loading image {i+1}/{min(n,50)}…")

                    progress.progress(0.95, text="Running inference…")
                    result = predict_batch(file_data)
                    st.session_state.batch_result = result
                    progress.progress(1.0, text="Complete ✓")
                    time.sleep(0.5)
                    progress.empty()
                    st.success(f"✓ {result.get('total_processed', 0)} images analysed — Session `{result.get('session_id')}`")
                except Exception as e:
                    progress.empty()
                    st.error(f"Batch error: {e}")

    # ── Show batch results ──
    if "batch_result" in st.session_state:
        res = st.session_state.batch_result
        predictions_list = res.get("predictions", [])
        summary = res.get("summary", {})
        analytics = res.get("analytics", {})

        if not predictions_list:
            st.warning("No predictions returned.")
            st.stop()

        st.markdown("---")
        st.markdown('<div class="section-label">Session Analytics</div>', unsafe_allow_html=True)

        # KPI row
        total  = res.get("total_processed", 0)
        avg_c  = analytics.get("average_confidence", 0)
        norm_r = analytics.get("normal_rate_percent", 0)
        flagged_n = analytics.get("flagged_count", 0)

        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi kpi-normal">
                <div class="kpi-label">Normal Rate</div>
                <div class="kpi-value">{norm_r:.0f}%</div>
                <div class="kpi-sub">WHO threshold ≥ 4%</div>
            </div>
            <div class="kpi kpi-abnormal">
                <div class="kpi-label">Abnormal</div>
                <div class="kpi-value">{analytics.get("abnormal_rate_percent", 0):.0f}%</div>
                <div class="kpi-sub">morphological defects</div>
            </div>
            <div class="kpi kpi-nonsperm">
                <div class="kpi-label">Avg Confidence</div>
                <div class="kpi-value">{avg_c:.0%}</div>
                <div class="kpi-sub">across {total} samples</div>
            </div>
            <div class="kpi kpi-flagged">
                <div class="kpi-label">Flagged</div>
                <div class="kpi-value">{flagged_n}</div>
                <div class="kpi-sub">require expert review</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Charts row
        ch1, ch2, ch3 = st.columns([1, 1, 1])

        with ch1:
            st.markdown('<div class="section-label">Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(donut_chart(summary), use_container_width=True)

        with ch2:
            st.markdown('<div class="section-label">Confidence Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(confidence_histogram(predictions_list), use_container_width=True)

        with ch3:
            st.markdown('<div class="section-label">Sample Confidence Timeline</div>', unsafe_allow_html=True)
            st.plotly_chart(timeline_chart(predictions_list), use_container_width=True)

        # Results table
        st.markdown('<div class="section-label" style="margin-top:0.5rem">Per-Sample Results</div>', unsafe_allow_html=True)
        rows = []
        for p in predictions_list:
            if "error" in p:
                rows.append({"#": p.get("index", "—"), "File": p.get("filename", "—"),
                             "Prediction": "ERROR", "Confidence": "—",
                             "Normal %": "—", "Abnormal %": "—", "Flagged": "—"})
            else:
                probs = p.get("probabilities", {})
                rows.append({
                    "#":          p.get("index", "—"),
                    "File":       p.get("filename", "—"),
                    "Prediction": p.get("prediction", "—"),
                    "Confidence": f"{p.get('confidence', 0):.1%}",
                    "Normal %":   f"{probs.get('Normal', 0):.1%}",
                    "Abnormal %": f"{probs.get('Abnormal', 0):.1%}",
                    "Flagged":    "⚠ Yes" if p.get("flagged_for_review") else "✓ No",
                    "Time (ms)":  f"{p.get('processing_time_ms', 0):.0f}",
                })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=340, hide_index=True)

        # Export
        col_e1, col_e2 = st.columns([1, 3])
        with col_e1:
            csv = df.to_csv(index=False)
            st.download_button(
                "📥  Export CSV Report",
                data=csv,
                file_name=f"spermai_batch_{res.get('session_id')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════
# MODE: SESSION LOG
# ══════════════════════════════════════════════════════════════
elif "Session" in app_mode:

    st.markdown('<div class="section-label">Session History & Platform Statistics</div>', unsafe_allow_html=True)

    # Overall stats
    stats = get_stats()
    if stats and "total_images_analyzed" in stats:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Sessions",   stats.get("total_sessions", 0))
        s2.metric("Images Analysed",  stats.get("total_images_analyzed", 0))
        s3.metric("Normal Rate",       f"{stats.get('normal_rate', 0):.1f}%")
        dist = stats.get("class_distribution", {})
        s4.metric("Abnormal Detected", dist.get("Abnormal", 0))

    # Session list
    sessions = get_sessions()
    if sessions:
        st.markdown('<div class="section-label" style="margin-top:1rem">Recent Sessions</div>', unsafe_allow_html=True)
        for s in sessions[:10]:
            entries = s.get("entries", [])
            valid = [e for e in entries if "prediction" in e]
            if not valid:
                continue
            norm_count = sum(1 for e in valid if e.get("prediction") == "Normal")
            avg_c = sum(e.get("confidence", 0) for e in valid) / max(len(valid), 1)
            ts = s.get("timestamp", "")[:19].replace("T", "  ")
            with st.expander(f"🗂  Session `{s.get('session_id')}` · {ts} · {len(valid)} images"):
                m1, m2, m3 = st.columns(3)
                m1.metric("Images", len(valid))
                m2.metric("Normal", norm_count)
                m3.metric("Avg Confidence", f"{avg_c:.1%}")
                rows = [{
                    "Prediction": e.get("prediction"),
                    "Confidence": f"{e.get('confidence', 0):.1%}",
                    "Flagged": "⚠ Yes" if e.get("flagged_for_review") else "✓ No",
                    "File": e.get("filename", "—"),
                } for e in valid[:20]]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem;border:1px dashed #1A2E48;border-radius:10px">
            <div style="font-size:2rem;opacity:0.2">📊</div>
            <div style="font-size:0.82rem;color:#2D4D6A;margin-top:0.5rem">No sessions logged yet</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="disclaimer-bar">
    SpermAI v2.0 &nbsp;·&nbsp; EfficientNet-B0 + ResNet50 &nbsp;·&nbsp;
    WHO 2021 Reference Criteria &nbsp;·&nbsp;
    <b>Research Use Only — Not validated for clinical diagnosis</b> &nbsp;·&nbsp;
    Results must be reviewed by a qualified andrologist
</div>
""", unsafe_allow_html=True)
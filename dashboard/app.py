"""FlightXAI — Streamlit Dashboard.

Interactive flight delay prediction dashboard with three explainability
views: Prediction result, SHAP + LIME explanations ("Why?"), and DiCE
counterfactual scenarios ("What if?").

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Project root on sys.path ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DICE_FEATURES_TO_VARY,
    FEATURE_COLUMNS,
    LIME_NUM_FEATURES,
    OUTPUT_DIR,
    PREDICTION_THRESHOLD,
    TARGET_COLUMN,
    TRAIN_DATA_PATH,
)

# ── Page config (MUST be first Streamlit call) ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ FlightXAI — Explainable Flight Delay Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f953c6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
        margin-bottom: 0;
    }
    .sub-title {
        color: #8892a4;
        font-size: 1rem;
        margin-top: -0.3rem;
        margin-bottom: 1.8rem;
    }

    /* Metric card */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        border-radius: 20px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        border: 1px solid rgba(102,126,234,0.25);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .metric-card .prob-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.4rem;
    }
    .metric-card .prob-value {
        font-size: 3.6rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0.2rem 0 0.6rem;
    }
    .metric-card .threshold-note {
        color: #5a6478;
        font-size: 0.75rem;
        margin-top: 0.8rem;
    }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.45rem 1.4rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
    }
    .badge-ontime  { background: rgba(0,200,83,0.12);  color: #00e676; border: 1.5px solid rgba(0,230,118,0.3); }
    .badge-delayed { background: rgba(255,82,82,0.12); color: #ff5252; border: 1.5px solid rgba(255,82,82,0.3); }

    /* Explanation box */
    .explanation-box {
        background: rgba(102,126,234,0.07);
        border-radius: 14px;
        padding: 1.3rem 1.6rem;
        border: 1px solid rgba(102,126,234,0.2);
        margin-top: 1rem;
        line-height: 1.85;
        font-size: 0.95rem;
        color: #c8d0de;
        white-space: pre-line;
    }

    /* CF scenario cards */
    .cf-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.06) 0%, rgba(118,75,162,0.06) 100%);
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        border-top: 1px solid rgba(102,126,234,0.15);
        border-right: 1px solid rgba(102,126,234,0.1);
        border-bottom: 1px solid rgba(102,126,234,0.1);
    }

    /* Sidebar button */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-size: 1.05rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px rgba(102,126,234,0.35);
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.5);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; font-weight: 600; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

    /* Divider */
    .fancy-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 1.4rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  Constants & Lookups
# ════════════════════════════════════════════════════════════════════════════

PEAK_HOURS = {7, 8, 17, 18, 19}

CARRIER_NAMES = {
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",
    "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines",
}

FEATURE_LABELS = {
    "departure_hour":           "Departure Hour",
    "day_of_week":              "Day of Week",
    "month":                    "Month",
    "is_weekend":               "Is Weekend",
    "distance":                 "Distance",
    "route_frequency":          "Route Frequency",
    "carrier_avg_delay":        "Carrier Avg Delay",
    "origin_avg_delay":         "Origin Avg Delay",
    "dest_avg_delay":           "Dest Avg Delay",
    "departure_peak_hour_flag": "Peak Hour",
    "holiday_flag":             "Holiday Period",
    "carrier_encoded":          "Carrier Delay Risk",
}

MONTH_NAMES = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# ════════════════════════════════════════════════════════════════════════════
#  Resource Loading  (cached — runs once per session)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading models and explainers...")
def load_resources():
    """Load models, feature artifacts, and all three explainers.

    Returns:
        Tuple of (xgb_model, lr_model, artifacts, shap_exp, lime_exp,
                  dice_exp, X_train, missing_files).
        If any required file is absent, missing_files is non-empty and
        all model values are None.
    """
    from src.explain_dice import build_dice_explainer
    from src.explain_lime import build_lime_explainer
    from src.explain_shap import build_shap_explainer

    xgb_path       = os.path.join(OUTPUT_DIR, "xgb_model.joblib")
    lr_path        = os.path.join(OUTPUT_DIR, "lr_model.joblib")
    artifacts_path = os.path.join(OUTPUT_DIR, "feature_artifacts.joblib")

    missing = [p for p in (xgb_path, lr_path, artifacts_path) if not os.path.exists(p)]
    if missing:
        return None, None, None, None, None, None, None, missing

    xgb_model = joblib.load(xgb_path)
    lr_model   = joblib.load(lr_path)
    artifacts  = joblib.load(artifacts_path)

    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    X_train  = train_df[FEATURE_COLUMNS]
    y_train  = train_df[TARGET_COLUMN]

    shap_exp = build_shap_explainer(xgb_model, X_train)
    lime_exp = build_lime_explainer(X_train)
    dice_exp = build_dice_explainer(xgb_model, X_train, y_train)

    return xgb_model, lr_model, artifacts, shap_exp, lime_exp, dice_exp, X_train, []


# ════════════════════════════════════════════════════════════════════════════
#  Feature Engineering  (mirrors preprocess.py logic exactly)
# ════════════════════════════════════════════════════════════════════════════

def build_feature_row(
    airline: str,
    origin: str,
    destination: str,
    month: int,
    departure_hour: int,
    distance_miles: float,
    day_of_week: int,
    artifacts: dict,
) -> pd.DataFrame:
    """Construct a single-row feature DataFrame matching FEATURE_COLUMNS.

    Applies the same scaler and lookup tables saved during preprocessing so
    the dashboard produces identical feature vectors to the training pipeline.
    """
    scaler = artifacts["scaler"]
    route_key = f"{origin}_{destination}"

    global_mean_delay = artifacts["global_mean_delay"]
    global_delay_rate = artifacts["global_delay_rate"]

    is_weekend        = 1 if day_of_week >= 6 else 0
    distance_scaled   = float(
        scaler.transform(np.array([[distance_miles]], dtype="float32")).ravel()[0]
    )
    route_frequency   = artifacts["route_frequency"].get(
        route_key, artifacts["route_frequency_median"]
    )
    carrier_avg_delay = artifacts["carrier_avg_delay"].get(airline, global_mean_delay)
    origin_avg_delay  = artifacts["origin_avg_delay"].get(origin, global_mean_delay)
    dest_avg_delay    = artifacts["dest_avg_delay"].get(destination, global_mean_delay)
    departure_peak    = 1 if departure_hour in PEAK_HOURS else 0

    today        = date.today()
    holiday_flag = 1 if today in artifacts.get("holiday_window", set()) else 0

    carrier_encoded = artifacts["carrier_encoded"].get(airline, global_delay_rate)

    row = pd.DataFrame([{
        "departure_hour":           departure_hour,
        "day_of_week":              day_of_week,
        "month":                    month,
        "is_weekend":               is_weekend,
        "distance":                 distance_scaled,
        "route_frequency":          route_frequency,
        "carrier_avg_delay":        carrier_avg_delay,
        "origin_avg_delay":         origin_avg_delay,
        "dest_avg_delay":           dest_avg_delay,
        "departure_peak_hour_flag": departure_peak,
        "holiday_flag":             holiday_flag,
        "carrier_encoded":          carrier_encoded,
    }])
    return row[FEATURE_COLUMNS]


# ════════════════════════════════════════════════════════════════════════════
#  Plotly Chart Builders
# ════════════════════════════════════════════════════════════════════════════

def _shap_waterfall_plotly(shap_dict: dict, base_value: float) -> go.Figure:
    """Horizontal waterfall-style bar chart for SHAP feature contributions."""
    sorted_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]))
    features = [FEATURE_LABELS.get(f, f) for f, _ in sorted_items]
    values   = [v for _, v in sorted_items]
    colors   = ["#ff5252" if v > 0 else "#00e676" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=features,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=10.5, color="#ccc"),
        hovertemplate="<b>%{y}</b><br>SHAP value: %{x:+.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="SHAP Feature Contributions", font=dict(size=15, color="#dde"), x=0),
        xaxis=dict(
            title="SHAP value (impact on delay probability)",
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=True, zerolinecolor="rgba(255,255,255,0.25)", zerolinewidth=1.5,
            tickfont=dict(color="#aaa"), titlefont=dict(color="#aaa", size=11),
        ),
        yaxis=dict(tickfont=dict(color="#ddd", size=11)),
        height=max(380, len(features) * 36),
        margin=dict(l=10, r=55, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ddd"), bargap=0.25,
    )
    return fig


def _lime_bar_plotly(lime_dict: dict) -> go.Figure:
    """Horizontal bar chart for LIME local feature weights."""
    top_items    = list(lime_dict.items())[:LIME_NUM_FEATURES]
    sorted_items = sorted(top_items, key=lambda x: abs(x[1]))
    features     = [FEATURE_LABELS.get(f, f) for f, _ in sorted_items]
    values       = [v for _, v in sorted_items]
    colors       = ["#ff5252" if v > 0 else "#00e676" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=features,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=10.5, color="#ccc"),
        hovertemplate="<b>%{y}</b><br>LIME weight: %{x:+.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"LIME Feature Weights (top {LIME_NUM_FEATURES})",
            font=dict(size=15, color="#dde"), x=0,
        ),
        xaxis=dict(
            title="Weight (local impact on delay class)",
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=True, zerolinecolor="rgba(255,255,255,0.25)", zerolinewidth=1.5,
            tickfont=dict(color="#aaa"), titlefont=dict(color="#aaa", size=11),
        ),
        yaxis=dict(tickfont=dict(color="#ddd", size=11)),
        height=max(340, len(features) * 38),
        margin=dict(l=10, r=55, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ddd"), bargap=0.25,
    )
    return fig


def _model_comparison_gauge(xgb_prob: float, lr_prob: float) -> go.Figure:
    """Side-by-side gauge indicators comparing XGBoost vs Logistic Regression."""
    thr = PREDICTION_THRESHOLD * 100

    def _gauge(val_pct, color, dom_x):
        return go.Indicator(
            mode="gauge+number",
            value=val_pct,
            number=dict(suffix="%", font=dict(size=30, color="#eee")),
            gauge=dict(
                axis=dict(
                    range=[0, 100], ticksuffix="%",
                    tickfont=dict(color="#888", size=10), nticks=6,
                ),
                bar=dict(color=color, thickness=0.28),
                bgcolor="rgba(255,255,255,0.03)",
                bordercolor="rgba(255,255,255,0.08)",
                steps=[
                    dict(range=[0, thr],     color="rgba(0,230,118,0.08)"),
                    dict(range=[thr, 100],   color="rgba(255,82,82,0.08)"),
                ],
                threshold=dict(
                    line=dict(color="#ffab40", width=3), thickness=0.85, value=thr,
                ),
            ),
            domain=dict(x=dom_x, y=[0, 1]),
        )

    fig = go.Figure()
    fig.add_trace(_gauge(xgb_prob * 100, "#667eea", [0.00, 0.44]))
    fig.add_trace(_gauge(lr_prob  * 100, "#764ba2", [0.56, 1.00]))
    fig.add_annotation(x=0.22, y=0.06, xref="paper", yref="paper",
                       text="<b>XGBoost</b>", showarrow=False,
                       font=dict(size=13, color="#aaa"))
    fig.add_annotation(x=0.78, y=0.06, xref="paper", yref="paper",
                       text="<b>Logistic Regression</b>", showarrow=False,
                       font=dict(size=13, color="#aaa"))
    fig.add_annotation(
        x=0.5, y=1.02, xref="paper", yref="paper",
        text=f"Threshold = {PREDICTION_THRESHOLD:.0%}", showarrow=False,
        font=dict(size=11, color="#ffab40"),
    )
    fig.update_layout(
        height=260, margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ddd"),
    )
    return fig


def _confidence_interval_plotly(prob: float) -> go.Figure:
    """Horizontal range bar showing the 90% heuristic confidence interval."""
    n_eff = 500
    sigma = 0.5 * np.sqrt(prob * (1 - prob) / n_eff)
    lo    = max(0.0, prob - 1.645 * sigma)
    hi    = min(1.0, prob + 1.645 * sigma)

    fig = go.Figure()
    # background track
    fig.add_trace(go.Bar(
        x=[1], y=[""], orientation="h",
        marker_color="rgba(255,255,255,0.06)",
        width=0.3, showlegend=False, hoverinfo="skip",
    ))
    # CI band
    fig.add_trace(go.Bar(
        x=[hi - lo], y=[""], base=[lo], orientation="h",
        marker_color="rgba(102,126,234,0.45)", width=0.3,
        name=f"90% CI [{lo:.1%}, {hi:.1%}]",
        hovertemplate=f"90% CI: {lo:.1%} – {hi:.1%}<extra></extra>",
    ))
    # point estimate dot
    fig.add_trace(go.Scatter(
        x=[prob], y=[""], mode="markers",
        marker=dict(size=14, color="#667eea", line=dict(width=2.5, color="white")),
        name=f"Estimate {prob:.1%}",
        hovertemplate=f"Estimate: {prob:.1%}<extra></extra>",
    ))
    fig.add_vline(
        x=PREDICTION_THRESHOLD, line_dash="dash",
        line_color="#ffab40", line_width=1.5,
        annotation_text=f"Threshold {PREDICTION_THRESHOLD:.0%}",
        annotation_position="top right",
        annotation_font=dict(color="#ffab40", size=10),
    )
    fig.update_layout(
        barmode="overlay",
        title=dict(text="Estimated Confidence Interval (90%)",
                   font=dict(size=13, color="#aaa"), x=0),
        xaxis=dict(
            range=[0, 1], tickformat=".0%",
            tickfont=dict(color="#888", size=10),
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(showticklabels=False, showgrid=False),
        height=115,
        margin=dict(l=0, r=10, t=38, b=25),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ddd"), showlegend=True,
        legend=dict(
            orientation="h", x=0, y=-0.45,
            font=dict(size=10, color="#999"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def _model_table(xgb_prob: float, lr_prob: float) -> pd.DataFrame:
    """Small DataFrame comparing LR vs XGBoost predictions."""
    thr = PREDICTION_THRESHOLD
    return pd.DataFrame({
        "Model": ["XGBoost ⭐", "Logistic Regression"],
        "Delay Probability": [f"{xgb_prob:.2%}", f"{lr_prob:.2%}"],
        "Prediction": [
            "🔴 Delayed" if xgb_prob >= thr else "🟢 On Time",
            "🔴 Delayed" if lr_prob  >= thr else "🟢 On Time",
        ],
    })


# ════════════════════════════════════════════════════════════════════════════
#  Main App
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Render the FlightXAI dashboard."""

    # Header
    st.markdown('<p class="main-title">✈️ FlightXAI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Explainable Flight Delay Prediction — '        'powered by XGBoost · SHAP · LIME · DiCE</p>',
        unsafe_allow_html=True,
    )

    # Load resources
    (
        xgb_model, lr_model, artifacts,
        shap_exp, lime_exp, dice_exp,
        X_train, missing,
    ) = load_resources()

    if missing:
        st.error(
            "**🚨 Missing required model files:**\n\n"
            + "\n".join(f"- `{p}`" for p in missing)
            + "\n\n**To fix:** run the training pipeline from the project root:\n"
            "```bash\npython src/preprocess.py\npython src/train.py\n```"
        )
        return

    # Sidebar — Flight Input
    st.sidebar.header("Enter Flight Details")

    airline = st.sidebar.selectbox(
        "Airline",
        options=list(CARRIER_NAMES.keys()),
        format_func=lambda c: f"{c} — {CARRIER_NAMES[c]}",
        index=0,
    )
    origin = st.sidebar.text_input(
        "Origin Airport (e.g. JFK)", value="JFK", max_chars=4
    ).upper().strip()
    destination = st.sidebar.text_input(
        "Destination Airport (e.g. LAX)", value="LAX", max_chars=4
    ).upper().strip()
    month          = st.sidebar.slider("Month", 1, 12, value=6)
    departure_hour = st.sidebar.slider("Departure Hour", 0, 23, value=14)
    distance_miles = st.sidebar.number_input(
        "Distance (miles)", min_value=50, max_value=5000, value=1000, step=50,
    )
    day_names  = ["Monday", "Tuesday", "Wednesday", "Thursday",
                  "Friday", "Saturday", "Sunday"]
    day_name   = st.sidebar.selectbox("Day of Week", day_names, index=0)
    day_of_week = day_names.index(day_name) + 1   # 1=Mon … 7=Sun

    st.sidebar.markdown("---")
    predict_btn = st.sidebar.button("🔍  Predict", use_container_width=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📊  Prediction",
        "🔬  Why?",
        "🔄  What if?",
    ])

    if not predict_btn:
        with tab1:
            st.info(
                "👈 Enter flight details in the sidebar and click "
                "**Predict** to begin."
            )
        return

    # Build feature row
    feature_row = build_feature_row(
        airline=airline, origin=origin, destination=destination,
        month=month, departure_hour=departure_hour,
        distance_miles=float(distance_miles),
        day_of_week=day_of_week, artifacts=artifacts,
    )

    # Run predictions
    xgb_prob   = float(xgb_model.predict_proba(feature_row)[:, 1][0])
    lr_prob    = float(lr_model.predict_proba(feature_row)[:, 1][0])
    is_delayed = xgb_prob >= PREDICTION_THRESHOLD

    # ══════════════════════════════════════════════════════════════════
    # TAB 1 — Prediction
    # ══════════════════════════════════════════════════════════════════
    with tab1:
        col_metric, col_gauge = st.columns([1, 2], gap="large")

        with col_metric:
            prob_color  = "#ff5252" if is_delayed else "#00e676"
            badge_class = "badge-delayed" if is_delayed else "badge-ontime"
            badge_text  = "🔴 Likely Delayed" if is_delayed else "🟢 On Time"

            # Large HTML metric card
            st.markdown(f"""
            <div class="metric-card">
                <div class="prob-label">Delay Probability</div>
                <div class="prob-value" style="color:{prob_color};">{xgb_prob:.1%}</div>
                <div><span class="status-badge {badge_class}">{badge_text}</span></div>
                <div class="threshold-note">Decision threshold: {PREDICTION_THRESHOLD:.0%}</div>
            </div>
            """, unsafe_allow_html=True)

            # Native st.metric (accessibility + delta indicator)
            st.metric(
                label="Delay Probability",
                value=f"{xgb_prob:.1%}",
                delta=f"{xgb_prob - PREDICTION_THRESHOLD:+.1%} vs threshold",
                delta_color="inverse",
                label_visibility="collapsed",
            )

        with col_gauge:
            st.plotly_chart(
                _model_comparison_gauge(xgb_prob, lr_prob),
                use_container_width=True,
            )

        # Confidence interval bar
        st.plotly_chart(
            _confidence_interval_plotly(xgb_prob), use_container_width=True,
        )

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

        cmp_left, cmp_right = st.columns(2)
        with cmp_left:
            st.markdown("#### 🤖 Model Comparison")
            st.dataframe(
                _model_table(xgb_prob, lr_prob),
                use_container_width=True, hide_index=True,
            )
        with cmp_right:
            st.markdown("#### 🛫 Flight Summary")
            r1 = st.columns(4)
            r1[0].metric("Airline",   airline)
            r1[1].metric("Route",     f"{origin} → {destination}")
            r1[2].metric("Month",     MONTH_NAMES[month])
            r1[3].metric("Hour",      f"{departure_hour:02d}:00")
            r2 = st.columns(3)
            r2[0].metric("Distance",  f"{distance_miles:,.0f} mi")
            r2[1].metric("Day",       day_name)
            r2[2].metric("Peak Hour", "Yes ⚠️" if departure_hour in PEAK_HOURS else "No ✅")

    # ══════════════════════════════════════════════════════════════════
    # TAB 2 — Why? (SHAP + LIME)
    # ══════════════════════════════════════════════════════════════════
    with tab2:
        from src.explain_lime import explain_single_lime
        from src.explain_shap import explain_single_shap, get_plain_english_shap

        with st.spinner("Computing SHAP and LIME explanations…"):
            shap_dict = explain_single_shap(shap_exp, feature_row)
            lime_dict = explain_single_lime(lime_exp, xgb_model, feature_row)
            shap_expl = shap_exp(feature_row)
            base_val  = (
                float(shap_expl.base_values[0])
                if hasattr(shap_expl, "base_values") else 0.0
            )

        col_shap, col_lime = st.columns(2, gap="medium")

        with col_shap:
            st.markdown("#### SHAP Waterfall")
            st.plotly_chart(
                _shap_waterfall_plotly(shap_dict, base_val),
                use_container_width=True,
            )

        with col_lime:
            st.markdown("#### LIME Local Weights")
            st.plotly_chart(
                _lime_bar_plotly(lime_dict), use_container_width=True,
            )

        plain_english = get_plain_english_shap(
            shap_dict, base_value=base_val, prediction=xgb_prob,
        )
        st.markdown(
            f'<div class="explanation-box">📝 <strong>Plain-English Summary</strong>'            f'<br><br>{plain_english}</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "ℹ️  **SHAP values** show how each feature pushed the prediction above or "
            "below the average delay rate.  "
            "🔴 Red bars **increase** delay probability · "
            "🟢 Green bars **decrease** it.  "
            "**LIME** produces independent local approximations via a surrogate linear model."
        )

    # ══════════════════════════════════════════════════════════════════
    # TAB 3 — What if? (DiCE Counterfactuals)
    # ══════════════════════════════════════════════════════════════════
    with tab3:
        if not is_delayed:
            st.info(
                "✅ **Flight predicted on-time.**  "
                "Counterfactual scenarios are only shown for predicted delays "
                f"(probability ≥ {PREDICTION_THRESHOLD:.0%}).  "
                "Try increasing the departure hour or selecting a busier route."
            )
        else:
            st.markdown(
                f"**This flight has a {xgb_prob:.1%} probability of delay.**  "
                "Here are alternative scenarios that could reduce it:"
            )

            with st.spinner("Generating counterfactual scenarios with DiCE…"):
                from src.explain_dice import generate_counterfactuals
                try:
                    cf_table = generate_counterfactuals(
                        dice_exp, feature_row, desired_class=0,
                    )
                except Exception as exc:
                    st.error(f"DiCE could not generate counterfactuals: {exc}")
                    cf_table = pd.DataFrame()

            if cf_table.empty:
                st.warning(
                    "⚠️ DiCE could not find actionable counterfactuals for this flight.  "
                    "Try a different set of inputs."
                )
            else:
                cf_cols = [c for c in cf_table.columns if c.startswith("cf")]

                for i, cf_col in enumerate(cf_cols, start=1):
                    cf_vector = feature_row.iloc[0].copy()
                    changes = []
                    for _, r in cf_table.iterrows():
                        feat, orig, new_val = r["feature"], r["original_value"], r[cf_col]
                        if not np.isclose(orig, new_val, atol=1e-6):
                            cf_vector[feat] = new_val
                            label    = FEATURE_LABELS.get(feat, feat)
                            orig_fmt = int(orig) if float(orig).is_integer() else f"{orig:.2f}"
                            new_fmt  = int(new_val) if float(new_val).is_integer() else f"{new_val:.2f}"
                            changes.append(f"<b>{label}</b>: {orig_fmt} → {new_fmt}")

                    if not changes:
                        continue

                    cf_row   = pd.DataFrame([cf_vector])[FEATURE_COLUMNS]
                    new_prob = float(xgb_model.predict_proba(cf_row)[:, 1][0])
                    delta    = xgb_prob - new_prob

                    if new_prob < PREDICTION_THRESHOLD:
                        status_text  = "🟢 On Time"
                        status_color = "#00e676"
                    else:
                        status_text  = "🟡 Reduced Risk"
                        status_color = "#ffd740"

                    changes_html = "<br>".join(f"&bull; {c}" for c in changes)

                    st.markdown(f"""
                    <div class="cf-card">
                        <div style="display:flex;justify-content:space-between;
                                    align-items:flex-start;margin-bottom:0.7rem;">
                            <span style="font-weight:700;font-size:1.05rem;">Scenario {i}</span>
                            <span style="text-align:right;">
                                <span style="color:{status_color};font-weight:700;">
                                    {status_text} &nbsp; {new_prob:.1%}
                                </span><br>
                                <span style="color:#00c853;font-size:0.85rem;">
                                    ↓ {delta:.1%} reduction
                                </span>
                            </span>
                        </div>
                        <div style="font-size:0.92rem;line-height:1.85;color:#c8d0de;">
                            {changes_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.caption(
                    "ℹ️  DiCE generates diverse actionable scenarios by varying: "
                    f"**{', '.join(FEATURE_LABELS.get(f, f) for f in DICE_FEATURES_TO_VARY)}**.  "
                    "All other features are held constant at their original values."
                )


# Entry point
main()

"""
SkyGuard PK - Streamlit Dashboard
Bilingual (English/Urdu) flood & heatwave risk dashboard for Pakistani cities,
powered by a Random Forest model trained on NASA POWER satellite-derived data.
"""

import streamlit as st
import pandas as pd
import joblib
from explain import explain_flood_risk, explain_heatwave_risk

st.set_page_config(page_title="SkyGuard PK", page_icon="🌦️", layout="centered")

RISK_COLORS = {"LOW": "#2ECC71", "MEDIUM": "#F39C12", "HIGH": "#E74C3C"}
FEATURES = ["precipitation", "temperature", "humidity", "soil_wetness", "rain_7day"]

# Default thresholds — must match label_risk.py constants
DEFAULT_FLOOD_RAIN_MM   = 100
DEFAULT_FLOOD_SOIL      = 0.6
DEFAULT_HEAT_TEMP_C     = 40
DEFAULT_HEAT_CONSEC     = 3

# Approximate city centroids (lat/lon)
CITY_COORDS = {
    "Karachi":    {"lat": 24.86, "lon": 67.01},
    "Lahore":     {"lat": 31.55, "lon": 74.35},
    "Faisalabad": {"lat": 31.42, "lon": 73.08},
    "Peshawar":   {"lat": 34.01, "lon": 71.57},
    "Multan":     {"lat": 30.19, "lon": 71.47},
    "Hyderabad":  {"lat": 25.37, "lon": 68.37},
    "Quetta":     {"lat": 30.18, "lon": 67.01},
    "Sukkur":     {"lat": 27.71, "lon": 68.85},
}


@st.cache_data
def load_data():
    return pd.read_csv("data/labeled_weather_data.csv", parse_dates=["date"])


@st.cache_resource
def load_models():
    flood_model = joblib.load("data/flood_model.pkl")
    heatwave_model = joblib.load("data/heatwave_model.pkl")
    return flood_model, heatwave_model


# Icons per risk level for the pill badges
_RISK_ICONS = {
    "LOW":    "✓",
    "MEDIUM": "▲",
    "HIGH":   "⚠",
}


def risk_badge(risk_level):
    color = RISK_COLORS.get(risk_level, "#95A5A6")
    icon  = _RISK_ICONS.get(risk_level, "")
    # Pill card: rounded, drop-shadow, icon prefix, larger bold text
    st.markdown(
        f"<div style='"
        f"background-color:{color};"
        f"color:#fff;"
        f"padding:12px 22px;"
        f"border-radius:999px;"
        f"display:inline-flex;"
        f"align-items:center;"
        f"gap:8px;"
        f"font-weight:700;"
        f"font-size:20px;"
        f"letter-spacing:0.04em;"
        f"box-shadow:0 4px 14px {color}55;"
        f"'>"
        f"<span style='font-size:18px;'>{icon}</span>{risk_level}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Threshold-based reclassification (mirrors label_risk.py logic) ────────────

def reclassify_flood(row, flood_rain_mm, flood_soil):
    if row["rain_7day"] >= flood_rain_mm and row["soil_wetness"] >= flood_soil:
        return "HIGH"
    if row["rain_7day"] >= flood_rain_mm * 0.5 or row["soil_wetness"] >= flood_soil * 0.8:
        return "MEDIUM"
    return "LOW"


def reclassify_heat(row, heat_temp_c, heat_consec):
    if row["consecutive_hot"] >= heat_consec:
        return "HIGH"
    if row["temperature"] >= heat_temp_c:
        return "MEDIUM"
    return "LOW"


def apply_flood_thresholds(df, flood_rain_mm, flood_soil):
    """Return df with flood_risk recalculated from slider values."""
    df = df.copy()
    df["flood_risk"] = df.apply(
        reclassify_flood, axis=1, flood_rain_mm=flood_rain_mm, flood_soil=flood_soil
    )
    return df


def apply_heat_thresholds(df, heat_temp_c, heat_consec):
    """Return df with heatwave_risk recalculated from slider values."""
    df = df.copy()
    df["heatwave_risk"] = df.apply(
        reclassify_heat, axis=1, heat_temp_c=heat_temp_c, heat_consec=heat_consec
    )
    return df


ADJUSTED_LABEL = ":material/tune: Adjusted view — not model prediction"

# ── Global CSS ────────────────────────────────────────────────────────────────
_CSS = """
<style>
/* Section spacing */
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

/* Hero header */
.sg-hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.15;
    color: #E2E8F0;
}
.sg-hero-sub {
    font-size: 1rem;
    color: #94A3B8;
    margin-top: 0.2rem;
    margin-bottom: 0.5rem;
}
.sg-hero-rule {
    height: 3px;
    background: linear-gradient(90deg, #38BDF8 0%, #0EA5E9 40%, transparent 100%);
    border: none;
    border-radius: 2px;
    margin: 0.8rem 0 1.4rem 0;
}

/* Risk cards */
.sg-risk-card {
    background: #152032;
    border: 1px solid #1E3A5F;
    border-radius: 16px;
    padding: 1.25rem 1.4rem 1.1rem 1.4rem;
    min-height: 160px;
}
.sg-risk-card h3 {
    margin: 0 0 0.75rem 0;
    font-size: 1rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Sidebar credit */
.sg-sidebar-credit {
    font-size: 0.75rem;
    color: #475569;
    text-align: center;
    margin-top: 1.5rem;
    padding-top: 0.75rem;
    border-top: 1px solid #1E3A5F;
    line-height: 1.6;
}
</style>
"""


def pakistan_map(latest_df, risk_type, adjusted=False):
    """Render a Vega-Lite scatter map of Pakistan colored by risk level."""
    if adjusted:
        st.caption(ADJUSTED_LABEL)
    map_df = latest_df[["city", "flood_risk", "heatwave_risk"]].copy()
    map_df["lat"] = map_df["city"].map(lambda c: CITY_COORDS[c]["lat"])
    map_df["lon"] = map_df["city"].map(lambda c: CITY_COORDS[c]["lon"])
    map_df["risk"] = map_df[risk_type]
    map_df["color"] = map_df["risk"].map(RISK_COLORS)

    spec = {
        "width": "container",
        "height": 340,
        "layer": [
            # Pakistan boundary — simple bounding-box rectangle as background
            {
                "mark": {
                    "type": "rect",
                    "color": "#eef2f7",
                    "stroke": "#c8d4e3",
                    "strokeWidth": 1,
                    "cornerRadius": 4,
                },
                "encoding": {
                    "x": {"value": 0},
                    "y": {"value": 0},
                    "x2": {"field": "__w"},
                    "y2": {"field": "__h"},
                },
            },
            # City dots
            {
                "mark": {
                    "type": "point",
                    "filled": True,
                    "size": 220,
                    "stroke": "white",
                    "strokeWidth": 1.5,
                },
                "encoding": {
                    "longitude": {"field": "lon", "type": "quantitative"},
                    "latitude": {"field": "lat", "type": "quantitative"},
                    "color": {
                        "field": "risk",
                        "type": "nominal",
                        "scale": {
                            "domain": ["LOW", "MEDIUM", "HIGH"],
                            "range": [RISK_COLORS["LOW"], RISK_COLORS["MEDIUM"], RISK_COLORS["HIGH"]],
                        },
                        "legend": {"title": "Risk level"},
                    },
                    "tooltip": [
                        {"field": "city", "title": "City"},
                        {"field": "risk", "title": "Risk"},
                    ],
                },
            },
            # City name labels
            {
                "mark": {
                    "type": "text",
                    "dy": -12,
                    "fontSize": 11,
                    "fontWeight": "bold",
                    "color": "#1f2328",
                },
                "encoding": {
                    "longitude": {"field": "lon", "type": "quantitative"},
                    "latitude": {"field": "lat", "type": "quantitative"},
                    "text": {"field": "city"},
                },
            },
        ],
        "projection": {"type": "mercator"},
        "data": {"values": map_df.to_dict(orient="records")},
    }
    st.vega_lite_chart(spec)


def main():
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Hero header ───────────────────────────────────────────────────────
    st.markdown(
        "<div class='sg-hero-title'>🌦️ SkyGuard PK</div>"
        "<div class='sg-hero-sub'>"
        "NASA satellite data → plain-language flood &amp; heatwave risk alerts for Pakistan"
        "</div>"
        "<hr class='sg-hero-rule'>",
        unsafe_allow_html=True,
    )

    lang = st.radio("Language / زبان", ["English", "اردو"], horizontal=True)

    df = load_data()
    flood_model, heatwave_model = load_models()

    cities = sorted(df["city"].unique())

    # ── Sidebar: alert sensitivity sliders ────────────────────────────────
    with st.sidebar:
        st.header(
            "تنبیہ کی حساسیت" if lang == "اردو" else "Alert sensitivity"
        )
        with st.expander(
            "سیلاب کی دہلیز" if lang == "اردو" else "Flood thresholds", expanded=True
        ):
            flood_rain_mm = st.slider(
                "7-day rainfall HIGH threshold (mm)",
                min_value=40, max_value=250,
                value=DEFAULT_FLOOD_RAIN_MM, step=5,
                help="Rain total over 7 days that triggers HIGH flood risk. "
                     f"Default: {DEFAULT_FLOOD_RAIN_MM} mm",
            )
            flood_soil = st.slider(
                "Soil wetness HIGH threshold (0–1)",
                min_value=0.3, max_value=1.0,
                value=DEFAULT_FLOOD_SOIL, step=0.05,
                format="%.2f",
                help="Soil saturation fraction that triggers HIGH flood risk. "
                     f"Default: {DEFAULT_FLOOD_SOIL}",
            )

        with st.expander(
            "گرمی کی لہر کی دہلیز" if lang == "اردو" else "Heatwave thresholds", expanded=True
        ):
            heat_temp_c = st.slider(
                "Extreme heat threshold (°C)",
                min_value=35, max_value=48,
                value=DEFAULT_HEAT_TEMP_C, step=1,
                help="Daily temperature that counts as an extreme-heat day. "
                     f"Default: {DEFAULT_HEAT_TEMP_C} °C",
            )
            heat_consec = st.slider(
                "Consecutive hot days for HIGH risk",
                min_value=1, max_value=7,
                value=DEFAULT_HEAT_CONSEC, step=1,
                help="Number of back-to-back extreme-heat days before HIGH is raised. "
                     f"Default: {DEFAULT_HEAT_CONSEC} days",
            )

        # Show a reset hint only when any slider has been moved
        if (
            flood_rain_mm != DEFAULT_FLOOD_RAIN_MM
            or flood_soil    != DEFAULT_FLOOD_SOIL
            or heat_temp_c   != DEFAULT_HEAT_TEMP_C
            or heat_consec   != DEFAULT_HEAT_CONSEC
        ):
            st.caption(":material/info: Thresholds differ from training defaults")

        st.markdown(
            "<div class='sg-sidebar-credit'>"
            "Powered by<br><strong>NASA POWER API</strong><br>+ IBM Bob"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Decide per-hazard whether to use the model or adjusted thresholds ─
    latest_raw = df.sort_values("date").groupby("city").tail(1).reset_index(drop=True)

    flood_adjusted = (
        flood_rain_mm != DEFAULT_FLOOD_RAIN_MM or flood_soil != DEFAULT_FLOOD_SOIL
    )
    heat_adjusted = (
        heat_temp_c != DEFAULT_HEAT_TEMP_C or heat_consec != DEFAULT_HEAT_CONSEC
    )

    # Map DataFrames: model columns (from CSV) used when not adjusted
    flood_map_df = (
        apply_flood_thresholds(latest_raw, flood_rain_mm, flood_soil)
        if flood_adjusted
        else latest_raw
    )
    heat_map_df = (
        apply_heat_thresholds(latest_raw, heat_temp_c, heat_consec)
        if heat_adjusted
        else latest_raw
    )

    # ── Pakistan map overview ──────────────────────────────────────────────
    st.subheader(
        "Pakistan Risk Map / پاکستان رسک نقشہ" if lang == "اردو" else "Pakistan risk map"
    )

    map_tab_flood, map_tab_heat = st.tabs(
        ["🌊 Flood risk", ":material/thermostat: Heatwave risk"]
    )
    with map_tab_flood:
        pakistan_map(flood_map_df, "flood_risk", adjusted=flood_adjusted)
    with map_tab_heat:
        pakistan_map(heat_map_df, "heatwave_risk", adjusted=heat_adjusted)

    st.divider()

    # ── Per-city detail ────────────────────────────────────────────────────
    city = st.selectbox(
        "Select City / شہر منتخب کریں" if lang == "اردو" else "Select City",
        cities,
    )

    city_df = df[df["city"] == city].sort_values("date")
    latest_row = city_df.iloc[-1]

    # Badge source: model prediction (CSV label) unless that hazard's sliders are moved
    flood_pred = (
        reclassify_flood(latest_row, flood_rain_mm, flood_soil)
        if flood_adjusted
        else flood_model.predict(latest_row[FEATURES].to_frame().T)[0]
    )
    heat_pred = (
        reclassify_heat(latest_row, heat_temp_c, heat_consec)
        if heat_adjusted
        else heatwave_model.predict(latest_row[FEATURES].to_frame().T)[0]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='sg-risk-card'><h3>🌊 Flood risk</h3>", unsafe_allow_html=True)
        risk_badge(flood_pred)
        if flood_adjusted:
            st.caption(ADJUSTED_LABEL)
        top_feature = pd.Series(
            flood_model.feature_importances_, index=FEATURES
        ).idxmax()
        explanation = explain_flood_risk(latest_row, flood_pred, top_feature)
        st.write(explanation["urdu"] if lang == "اردو" else explanation["english"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='sg-risk-card'><h3>🌡️ Heatwave risk</h3>", unsafe_allow_html=True)
        risk_badge(heat_pred)
        if heat_adjusted:
            st.caption(ADJUSTED_LABEL)
        explanation = explain_heatwave_risk(latest_row, heat_pred)
        st.write(explanation["urdu"] if lang == "اردو" else explanation["english"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("7-day trend" if lang == "English" else "7 Din Ka Trend")
    st.caption(
        f"Last 30 days of precipitation (mm), temperature (°C), and soil wetness (fraction) for {city}."
        if lang == "English"
        else f"{city} کے لیے آخری 30 دن کی بارش، درجہ حرارت اور مٹی کی نمی۔"
    )
    trend_df = city_df.tail(30).set_index("date")[["precipitation", "temperature", "soil_wetness"]]
    st.line_chart(
        trend_df,
        color=["#38BDF8", "#FB923C", "#34D399"],
    )

    with st.expander("About this data / Data ke baray mein"):
        st.write(
            "Data source: NASA POWER API (satellite-derived meteorological data). "
            "Risk thresholds are self-defined approximations for demonstration purposes, "
            "not official disaster management classifications. "
            "Model: Random Forest classifier, trained on 5 years of historical data per city."
        )


if __name__ == "__main__":
    main()

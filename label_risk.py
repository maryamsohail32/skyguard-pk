"""
SkyGuard PK - Risk Labeling Script
Defines flood/heatwave risk labels based on rolling weather thresholds.

IMPORTANT (honesty note for README): these thresholds are self-defined
approximations — not official government or NDMA disaster classifications.
They are calibrated to function as early-warning indicators: they are
intentionally set below known disaster-level totals so that rising risk
is flagged while there is still time to act. Document this clearly in
any submission or deployment context.

Flood threshold rationale — FLOOD_RAIN_7DAY_MM = 100 mm
--------------------------------------------------------
Pakistan's 30-year national average annual rainfall is approximately
250 mm, implying a rough monthly benchmark of ~130.8 mm (Rehman et al.,
Pakistan Meteorological Department historical records). During the 2022
catastrophic floods — the worst in Pakistan's recorded history — single-
day rainfall extremes reached 142 mm in Naushahro Feroze (Sindh,
August 2022), and Lahore recorded sustained urban flooding above 200 mm
over multi-day events. These figures represent disaster-scale totals at
which significant infrastructure damage and displacement were already
occurring.

The 100 mm / 7-day threshold is deliberately set below these known
disaster levels. Its purpose is early warning: 100 mm accumulated over
a week indicates persistently saturated conditions and elevated runoff
risk, giving emergency managers and communities time to prepare before
totals reach the 130–200 mm range where historical flooding has
materialised. It is not intended to match or replicate the exact figures
from any single event.

The MEDIUM threshold (50 mm / 7 days, i.e. 50 % of HIGH) and the soil-
wetness conditions further stratify risk at earlier stages of accumulation.

Heatwave threshold rationale — HEATWAVE_TEMP_C = 40 °C
-------------------------------------------------------
40 °C is widely cited as the threshold for extreme heat stress in South
Asian public health literature and aligns with Pakistan Meteorological
Department heat alert criteria. Three consecutive days (HEATWAVE_CONSECUTIVE_DAYS)
follows the WMO-adjacent convention used in regional early-warning systems.

Wet-bulb / humidity factor — HEATWAVE_HUMIDITY_THRESHOLD = 60 %
----------------------------------------------------------------
Physiological heat stress is determined not by air temperature alone but
by wet-bulb temperature, which combines heat and humidity. At 40 °C and
≥ 60 % relative humidity the wet-bulb temperature approaches or exceeds
35 °C — the level at which the human body cannot cool itself through
sweating even at rest, posing a direct risk of heat stroke within hours
(Sherwood & Huber, 2010; Raymond et al., 2020). Pakistan's pre-monsoon
and monsoon periods routinely combine extreme temperatures with high
humidity, making this the most dangerous heat scenario for the region.

classify_heat() therefore applies a two-path model:
  1. Wet-bulb path: temperature >= 40 °C AND humidity >= 60 % → HIGH
     immediately, regardless of how many consecutive days have elapsed.
  2. Dry-heat path (original logic): consecutive days at >= 40 °C
     determine MEDIUM (day 1–2) or HIGH (day 3+) as before.

The wet-bulb path takes priority. A single day of extreme humid heat
is treated as more dangerous than two days of equally extreme dry heat.
"""

import pandas as pd

# ── Flood thresholds ──────────────────────────────────────────────────────────
#
# FLOOD_RAIN_7DAY_MM = 100 mm (7-day rolling sum, per city)
#   Early-warning level — set below known disaster benchmarks:
#     - Pakistan 30-yr monthly avg rainfall:  ~130.8 mm
#     - 2022 single-day extreme (Naushahro Feroze, Aug 2022): 142 mm
#     - Lahore urban flooding threshold (multi-day):          200 mm+
#   100 mm / 7 days flags rising accumulation before these levels are reached.
#
# FLOOD_SOIL_WETNESS = 0.6 (fraction, 0–1)
#   Saturated-soil proxy: above 0.6 the ground has limited additional
#   absorption capacity, increasing surface runoff risk independently of
#   rainfall rate.
FLOOD_RAIN_7DAY_MM = 100       # early-warning 7-day total (see rationale above)
FLOOD_SOIL_WETNESS = 0.6       # soil saturation fraction above which runoff risk rises sharply

# ── Heatwave thresholds ───────────────────────────────────────────────────────
#
# HEATWAVE_TEMP_C = 40 °C
#   Pakistan Meteorological Department extreme-heat alert level; broadly
#   consistent with South Asian public-health heat-stress literature.
#
# HEATWAVE_CONSECUTIVE_DAYS = 3
#   Minimum run of extreme-heat days before a heatwave is declared HIGH
#   under the dry-heat path (no qualifying humidity).
#   Aligns with WMO-adjacent regional early-warning conventions.
#
# HEATWAVE_HUMIDITY_THRESHOLD = 60 %
#   Relative humidity level above which, combined with HEATWAVE_TEMP_C,
#   the wet-bulb temperature approaches the ~35 °C survivability limit.
#   A single day meeting both conditions is classified HIGH immediately
#   (wet-bulb path), bypassing the consecutive-day requirement.
HEATWAVE_TEMP_C            = 40   # daily max temperature for extreme-heat classification (°C)
HEATWAVE_CONSECUTIVE_DAYS  = 3    # consecutive dry-heat days required for HIGH heatwave risk
HEATWAVE_HUMIDITY_THRESHOLD = 60  # relative humidity (%) that activates the wet-bulb HIGH path


def label_flood_risk(df):
    """Add a flood_risk column: LOW / MEDIUM / HIGH per city-day."""
    df = df.sort_values(["city", "date"]).copy()

    df["rain_7day"] = (
        df.groupby("city")["precipitation"]
        .transform(lambda x: x.rolling(window=7, min_periods=1).sum())
    )

    def classify_flood(row):
        if row["rain_7day"] >= FLOOD_RAIN_7DAY_MM and row["soil_wetness"] >= FLOOD_SOIL_WETNESS:
            return "HIGH"
        elif row["rain_7day"] >= FLOOD_RAIN_7DAY_MM * 0.5 or row["soil_wetness"] >= FLOOD_SOIL_WETNESS * 0.8:
            return "MEDIUM"
        else:
            return "LOW"

    df["flood_risk"] = df.apply(classify_flood, axis=1)
    return df


def label_heatwave_risk(df):
    """Add a heatwave_risk column: LOW / MEDIUM / HIGH per city-day."""
    df = df.sort_values(["city", "date"]).copy()

    df["is_hot_day"] = df["temperature"] >= HEATWAVE_TEMP_C

    def consecutive_hot_days(series):
        # counts how many consecutive True values end at each row
        groups = (~series).cumsum()
        return series.groupby(groups).cumsum()

    df["consecutive_hot"] = (
        df.groupby("city")["is_hot_day"]
        .transform(consecutive_hot_days)
    )

    def classify_heat(row):
        # Wet-bulb path: extreme heat + high humidity → HIGH on day 1
        # (physiologically dangerous regardless of duration)
        if (
            row["temperature"] >= HEATWAVE_TEMP_C
            and row["humidity"] >= HEATWAVE_HUMIDITY_THRESHOLD
        ):
            return "HIGH"
        # Dry-heat path: consecutive-day counter (original logic)
        if row["consecutive_hot"] >= HEATWAVE_CONSECUTIVE_DAYS:
            return "HIGH"
        if row["consecutive_hot"] >= 1:
            return "MEDIUM"
        return "LOW"

    df["heatwave_risk"] = df.apply(classify_heat, axis=1)
    return df


def main():
    df = pd.read_csv("data/nasa_weather_data.csv", parse_dates=["date"])
    df = df.dropna(subset=["precipitation", "temperature", "humidity", "soil_wetness"])

    df = label_flood_risk(df)
    df = label_heatwave_risk(df)

    output_path = "data/labeled_weather_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Labeled {len(df)} rows and saved to {output_path}\n")
    print("Flood risk distribution:")
    print(df["flood_risk"].value_counts())
    print("\nHeatwave risk distribution:")
    print(df["heatwave_risk"].value_counts())


if __name__ == "__main__":
    main()

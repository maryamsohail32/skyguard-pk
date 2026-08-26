"""
SkyGuard PK - Risk Labeling Script
Defines flood/heatwave risk labels based on rolling weather thresholds.

IMPORTANT (honesty note for README): these thresholds are self-defined,
reasonable approximations based on commonly cited flood/heatwave triggers
for the region - not official government disaster classifications.
Document this clearly in your submission.
"""

import pandas as pd

# Thresholds - adjust these based on domain research if you have time
FLOOD_RAIN_7DAY_MM = 100       # 7-day rolling rainfall total considered high
FLOOD_SOIL_WETNESS = 0.6       # soil wetness fraction considered saturated
HEATWAVE_TEMP_C = 40           # daily temp considered extreme
HEATWAVE_CONSECUTIVE_DAYS = 3  # consecutive extreme-heat days to count as heatwave


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
        if row["consecutive_hot"] >= HEATWAVE_CONSECUTIVE_DAYS:
            return "HIGH"
        elif row["consecutive_hot"] >= 1:
            return "MEDIUM"
        else:
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

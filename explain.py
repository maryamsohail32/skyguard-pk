"""
SkyGuard PK - Explanation Layer
Converts a risk prediction + feature values into a plain-language
bilingual (English/Urdu) explanation.
"""

FEATURE_LABELS_EN = {
    "rain_7day": "7-day rainfall total",
    "soil_wetness": "soil saturation",
    "precipitation": "today's rainfall",
    "temperature": "temperature",
    "humidity": "humidity",
}

FEATURE_LABELS_UR = {
    "rain_7day": "pichle 7 din ki total barish",
    "soil_wetness": "mitti ki nami",
    "precipitation": "aaj ki barish",
    "temperature": "temperature",
    "humidity": "namee (humidity)",
}


def explain_flood_risk(row, risk_level, top_feature):
    en = (
        f"{row['city']}: {risk_level} flood risk. "
        f"Main factor: {FEATURE_LABELS_EN.get(top_feature, top_feature)} "
        f"is currently elevated ({row['rain_7day']:.1f}mm over 7 days, "
        f"soil saturation at {row['soil_wetness']*100:.0f}%)."
    )
    ur = (
        f"{row['city']}: {risk_level} flood risk hai. "
        f"Wajah: {FEATURE_LABELS_UR.get(top_feature, top_feature)} zyada hai "
        f"({row['rain_7day']:.1f}mm 7 din mein, mitti {row['soil_wetness']*100:.0f}% saturated hai)."
    )
    return {"english": en, "urdu": ur}


def explain_heatwave_risk(row, risk_level):
    en = (
        f"{row['city']}: {risk_level} heatwave risk. "
        f"Temperature at {row['temperature']:.1f}°C."
    )
    ur = (
        f"{row['city']}: {risk_level} garmi ka risk hai. "
        f"Temperature {row['temperature']:.1f}°C hai."
    )
    return {"english": en, "urdu": ur}


if __name__ == "__main__":
    # quick manual test
    import pandas as pd
    test_row = pd.Series({
        "city": "Karachi", "rain_7day": 120.5, "soil_wetness": 0.72,
        "precipitation": 15.0, "temperature": 32.0, "humidity": 70.0,
    })
    print(explain_flood_risk(test_row, "HIGH", "rain_7day"))

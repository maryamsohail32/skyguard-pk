"""
SkyGuard PK - Data Fetching Script
Pulls historical weather/soil data from NASA POWER API for major Pakistani cities.
"""

import requests
import pandas as pd
import time
import os

# 8 major Pakistani cities with coordinates
CITIES = {
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5497, 74.3436),
    "Multan": (30.1575, 71.5249),
    "Sukkur": (27.7052, 68.8574),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
    "Hyderabad": (25.3960, 68.3578),
    "Faisalabad": (31.4504, 73.1350),
}

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = "PRECTOTCORR,T2M,RH2M,GWETROOT"
START_DATE = "20200101"
END_DATE = "20241231"


def fetch_city_data(city_name, lat, lon):
    """Fetch daily weather data for one city from NASA POWER API."""
    params = {
        "parameters": PARAMETERS,
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON",
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        parameter_data = data["properties"]["parameter"]
        dates = list(parameter_data["PRECTOTCORR"].keys())

        rows = []
        for date in dates:
            rows.append({
                "city": city_name,
                "date": date,
                "precipitation": parameter_data["PRECTOTCORR"].get(date),
                "temperature": parameter_data["T2M"].get(date),
                "humidity": parameter_data["RH2M"].get(date),
                "soil_wetness": parameter_data["GWETROOT"].get(date),
            })
        return pd.DataFrame(rows)

    except requests.exceptions.RequestException as e:
        print(f"  ERROR fetching {city_name}: {e}")
        return None


def main():
    os.makedirs("data", exist_ok=True)
    all_dataframes = []

    print(f"Fetching NASA POWER data for {len(CITIES)} Pakistani cities...")
    print(f"Date range: {START_DATE} to {END_DATE}\n")

    for city_name, (lat, lon) in CITIES.items():
        print(f"Fetching {city_name}...")
        df = fetch_city_data(city_name, lat, lon)
        if df is not None:
            all_dataframes.append(df)
            print(f"  -> Got {len(df)} days of data")
        time.sleep(1)  # be polite to the API, avoid rate limits

    if not all_dataframes:
        print("\nNo data was fetched. Check your internet connection and try again.")
        return

    combined = pd.concat(all_dataframes, ignore_index=True)

    # NASA POWER uses -999 as a "no data" sentinel value - clean those out
    combined = combined.replace(-999, pd.NA)
    combined["date"] = pd.to_datetime(combined["date"], format="%Y%m%d")

    output_path = "data/nasa_weather_data.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nDone. Saved {len(combined)} total rows to {output_path}")
    print(f"Cities covered: {combined['city'].nunique()}")
    print(f"Missing values per column:\n{combined.isnull().sum()}")


if __name__ == "__main__":
    main()

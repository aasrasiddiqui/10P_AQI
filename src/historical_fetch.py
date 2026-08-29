import requests
import pandas as pd
from pathlib import Path

LATITUDE = 24.8607
LONGITUDE = 67.0011

START_DATE = "2026-02-01"
END_DATE = "2026-08-27"

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_historical_air_quality():
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "us_aqi"
        ],
        "timezone": "Asia/Karachi",
        "start_date": START_DATE,
        "end_date": END_DATE
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()

    return pd.DataFrame(data["hourly"])


def fetch_historical_weather():
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m"
        ],
        "timezone": "Asia/Karachi"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()

    return pd.DataFrame(data["hourly"])


def merge_historical_data(air_df, weather_df):
    return pd.merge(
        air_df,
        weather_df,
        on="time",
        how="inner"
    )


if __name__ == "__main__":

    print("Fetching historical air-quality data...")
    air_df = fetch_historical_air_quality()

    print("Fetching historical weather data...")
    weather_df = fetch_historical_weather()

    print("Merging historical datasets...")
    combined_df = merge_historical_data(
        air_df,
        weather_df
    )

    output_path = DATA_DIR / "karachi_historical.csv"

    combined_df.to_csv(
        output_path,
        index=False
    )

    print("\nHistorical dataset created successfully.")
    print("Shape:", combined_df.shape)

    print("\nColumns:")
    print(combined_df.columns.tolist())

    print("\nFirst rows:")
    print(combined_df.head())

    print("\nMissing values:")
    print(combined_df.isnull().sum())

    print(f"\nSaved to: {output_path}")
import requests
import pandas as pd
from pathlib import Path


# Karachi coordinates
LATITUDE = 24.8607
LONGITUDE = 67.0011

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_air_quality():
    """
    Fetch current + forecast air-quality data from Open-Meteo.
    """

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
        "forecast_days": 3
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data["hourly"])

    return df

def fetch_weather():
    """
    Fetch hourly weather forecast from Open-Meteo.
    """

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m"
        ],
        "timezone": "Asia/Karachi",
        "forecast_days": 3
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data["hourly"])

    return df

def merge_data(air_df, weather_df):

    df = pd.merge(
        air_df,
        weather_df,
        on="time",
        how="inner"
    )

    return df

if __name__ == "__main__":

    print("Fetching air quality data...")
    air_df = fetch_air_quality()

    print("Fetching weather data...")
    weather_df = fetch_weather()

    print("Merging datasets...")
    combined_df = merge_data(
        air_df,
        weather_df
    )

    output_path = DATA_DIR / "karachi_combined.csv"

    combined_df.to_csv(
        output_path,
        index=False
    )

    print("\nCombined Data:")
    print(combined_df.head())

    print("\nShape:", combined_df.shape)

    print(f"\nSaved to: {output_path}")
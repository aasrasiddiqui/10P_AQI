import os
import tempfile
import hopsworks
from dotenv import load_dotenv
import requests
import pandas as pd
import joblib
from pathlib import Path

load_dotenv()

if os.name == "nt":
    TEMP_DIR = r"C:\tmp"

    os.makedirs(
        TEMP_DIR,
        exist_ok=True
    )

    os.environ["TEMP"] = TEMP_DIR
    os.environ["TMP"] = TEMP_DIR
    os.environ["TMPDIR"] = TEMP_DIR
    tempfile.tempdir = TEMP_DIR
# --------------------------------------------------
# Configuration
# --------------------------------------------------

LATITUDE = 24.8607
LONGITUDE = 67.0011
TIMEZONE = "Asia/Karachi"

MODEL_DIR = Path("models")


# These MUST match the features used during training
FEATURES = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",

    "us_aqi",

    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",

    "hour",
    "day",
    "day_of_week",
    "month",

    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_24",

    "aqi_rolling_mean_3",
    "aqi_rolling_mean_6",
    "aqi_rolling_mean_24",

    "aqi_change"
]


# --------------------------------------------------
# Fetch recent AQI data
# --------------------------------------------------

def fetch_recent_air_quality():

    url = (
        "https://air-quality-api.open-meteo.com/"
        "v1/air-quality"
    )

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

        "timezone": TIMEZONE,

        # Needed for lag/rolling features
        "past_days": 4,

        # Current day is enough for prediction input
        "forecast_days": 1
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    return pd.DataFrame(
        data["hourly"]
    )


# --------------------------------------------------
# Fetch recent weather data
# --------------------------------------------------

def fetch_recent_weather():

    url = (
        "https://api.open-meteo.com/"
        "v1/forecast"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,

        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m"
        ],

        "timezone": TIMEZONE,

        "past_days": 4,
        "forecast_days": 1
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    return pd.DataFrame(
        data["hourly"]
    )


# --------------------------------------------------
# Merge AQI + weather
# --------------------------------------------------

def fetch_recent_data():

    print("Fetching recent air-quality data...")
    air_df = fetch_recent_air_quality()

    print("Fetching recent weather data...")
    weather_df = fetch_recent_weather()

    df = pd.merge(
        air_df,
        weather_df,
        on="time",
        how="inner"
    )

    df["time"] = pd.to_datetime(
        df["time"]
    )

    df = (
        df
        .sort_values("time")
        .reset_index(drop=True)
    )

    return df


# --------------------------------------------------
# Remove future API rows
# --------------------------------------------------

def keep_observed_hours(df):

    # Current Karachi time
    current_time = (
        pd.Timestamp.now(
            tz=TIMEZONE
        )
        .tz_localize(None)
        .floor("h")
    )

    # The API may contain forecasted future hours.
    # For model input, keep only data up to now.
    df = df[
        df["time"] <= current_time
    ].copy()

    if df.empty:
        raise ValueError(
            "No current or historical rows "
            "available for prediction."
        )

    return df


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

def create_prediction_features(df):

    df = df.copy()

    # --------------------------
    # Time features
    # --------------------------

    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["day_of_week"] = (
        df["time"].dt.dayofweek
    )
    df["month"] = df["time"].dt.month


    # --------------------------
    # AQI lag features
    # --------------------------

    df["aqi_lag_1"] = (
        df["us_aqi"].shift(1)
    )

    df["aqi_lag_3"] = (
        df["us_aqi"].shift(3)
    )

    df["aqi_lag_6"] = (
        df["us_aqi"].shift(6)
    )

    df["aqi_lag_24"] = (
        df["us_aqi"].shift(24)
    )


    # --------------------------
    # AQI rolling means
    # --------------------------

    df["aqi_rolling_mean_3"] = (
        df["us_aqi"]
        .rolling(window=3)
        .mean()
    )

    df["aqi_rolling_mean_6"] = (
        df["us_aqi"]
        .rolling(window=6)
        .mean()
    )

    df["aqi_rolling_mean_24"] = (
        df["us_aqi"]
        .rolling(window=24)
        .mean()
    )


    # --------------------------
    # AQI change
    # --------------------------

    df["aqi_change"] = (
        df["us_aqi"].diff()
    )


    # Fill any small missing numeric gaps
    numeric_cols = df.select_dtypes(
        include="number"
    ).columns

    df[numeric_cols] = (
        df[numeric_cols]
        .interpolate(
            method="linear"
        )
    )

    return df


# --------------------------------------------------
# Load models
# --------------------------------------------------

def load_models_from_hopsworks():

    print("Connecting to Hopsworks Model Registry...")

    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )

    mr = project.get_model_registry()

    model_names = {
        "24h": "aqi_predictor_24h",
        "48h": "aqi_predictor_48h",
        "72h": "aqi_predictor_72h"
    }

    models = {}

    for horizon, model_name in model_names.items():

        print(f"Loading {model_name}...")

        all_versions = mr.get_models(
            model_name
        )

        if not all_versions:
            raise ValueError(
                f"No registered models found "
                f"for {model_name}"
            )

        registered_model = max(
            all_versions,
            key=lambda model: model.version
        )

        model_dir = registered_model.download()

        model_path = (
            Path(model_dir)
            / f"aqi_model_{horizon}.joblib"
        )

        models[horizon] = joblib.load(
            model_path
        )

    return models


# --------------------------------------------------
# AQI category
# --------------------------------------------------

def get_aqi_status(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Moderate"

    elif aqi <= 150:
        return (
            "Unhealthy for Sensitive Groups"
        )

    elif aqi <= 200:
        return "Unhealthy"

    elif aqi <= 300:
        return "Very Unhealthy"

    else:
        return "Hazardous"


# --------------------------------------------------
# Make predictions
# --------------------------------------------------

def predict_aqi():

    df = fetch_recent_data()

    df = keep_observed_hours(df)

    df = create_prediction_features(df)

    # Keep rows that contain every feature
    valid_df = df.dropna(
        subset=FEATURES
    )

    if valid_df.empty:

        raise ValueError(
            "Not enough recent data to "
            "create prediction features."
        )

    latest_row = valid_df.iloc[-1]

    X = pd.DataFrame(
        [latest_row[FEATURES]]
    )

    models = load_models_from_hopsworks()
    predictions = {}

    for horizon, model in models.items():

        prediction = model.predict(
            X
        )[0]

        # AQI cannot be negative
        prediction = max(
            0,
            float(prediction)
        )

        predictions[horizon] = round(
            prediction,
            1
        )

    current_aqi = float(
        latest_row["us_aqi"]
    )

    result = {
        "time": latest_row["time"],

        "current_aqi": round(
            current_aqi,
            1
        ),

        "current_status": (
            get_aqi_status(current_aqi)
        ),

        "forecast_24h": (
            predictions["24h"]
        ),

        "forecast_24h_status": (
            get_aqi_status(
                predictions["24h"]
            )
        ),

        "forecast_48h": (
            predictions["48h"]
        ),

        "forecast_48h_status": (
            get_aqi_status(
                predictions["48h"]
            )
        ),

        "forecast_72h": (
            predictions["72h"]
        ),

        "forecast_72h_status": (
            get_aqi_status(
                predictions["72h"]
            )
        ),

        "pm2_5": float(
            latest_row["pm2_5"]
        ),

        "pm10": float(
            latest_row["pm10"]
        ),

        "temperature": float(
            latest_row["temperature_2m"]
        ),

        "humidity": float(
            latest_row[
                "relative_humidity_2m"
            ]
        ),

        "wind_speed": float(
            latest_row["wind_speed_10m"]
        )
    }

    return result


# --------------------------------------------------
# Run standalone prediction
# --------------------------------------------------

if __name__ == "__main__":

    print("\nAQI Prediction Pipeline")
    print("=" * 50)

    result = predict_aqi()

    print(
        f"\nLatest data time: "
        f"{result['time']}"
    )

    print(
        f"\nCurrent AQI: "
        f"{result['current_aqi']}"
    )

    print(
        f"Status: "
        f"{result['current_status']}"
    )

    print("\n3-Day Forecast")
    print("-" * 50)

    print(
        f"24 hours: "
        f"{result['forecast_24h']} "
        f"({result['forecast_24h_status']})"
    )

    print(
        f"48 hours: "
        f"{result['forecast_48h']} "
        f"({result['forecast_48h_status']})"
    )

    print(
        f"72 hours: "
        f"{result['forecast_72h']} "
        f"({result['forecast_72h_status']})"
    )

    print("\nCurrent Conditions")
    print("-" * 50)

    print(
        f"PM2.5: {result['pm2_5']}"
    )

    print(
        f"PM10: {result['pm10']}"
    )

    print(
        f"Temperature: "
        f"{result['temperature']} °C"
    )

    print(
        f"Humidity: "
        f"{result['humidity']} %"
    )

    print(
        f"Wind Speed: "
        f"{result['wind_speed']} km/h"
    )
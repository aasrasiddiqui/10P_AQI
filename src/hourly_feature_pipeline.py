import os
import tempfile

import hopsworks
import pandas as pd
from dotenv import load_dotenv

from src.predict import (
    FEATURES,
    fetch_recent_data,
    keep_observed_hours,
    create_prediction_features,
)


# --------------------------------------------------
# Environment
# --------------------------------------------------

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
# Full feature group schema
# --------------------------------------------------

FEATURE_GROUP_COLUMNS = [
    "time",

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

    "aqi_change",

]


def prepare_latest_features():

    print(
        "Fetching latest AQI and weather data..."
    )

    df = fetch_recent_data()

    df = keep_observed_hours(df)

    df = create_prediction_features(df)

    df = df.dropna(
        subset=FEATURES
    )

    if df.empty:
        raise ValueError(
            "No valid feature row available."
        )

    latest = df.iloc[[-1]].copy()

    # Future targets are unknown for live rows
    int64_cols = [
        "hour",
        "day",
        "day_of_week",
        "month",
    ]

    latest[int64_cols] = latest[int64_cols].astype("int64")
    latest = latest[
        FEATURE_GROUP_COLUMNS
    ]

    print(
        "Latest feature timestamp:",
        latest["time"].iloc[0]
    )

    return latest


def upload_latest_features(df):

    api_key = os.getenv(
        "HOPSWORKS_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "HOPSWORKS_API_KEY is missing."
        )

    print(
        "Connecting to Hopsworks..."
    )

    project = hopsworks.login(
        api_key_value=api_key
    )

    fs = project.get_feature_store()
    print("\nUploading dataframe columns:")
    print(df.columns.tolist())

    print("\nUploading dataframe dtypes:")
    print(df.dtypes)
    fg = fs.get_or_create_feature_group(
        name="karachi_aqi_live_features",
        version=2,
        description=(
            "Latest hourly AQI, pollutant, weather "
            "and engineered features for live inference"
        ),
        primary_key=["time"],
        event_time="time",
        online_enabled=False,
        time_travel_format="HUDI"
    )

    print(
        "Uploading latest feature row..."
    )

    fg.insert(
        df,
        write_options={
            "wait_for_job": True
        }
    )

    print(
        "Hourly feature pipeline completed."
    )


if __name__ == "__main__":

    latest_df = (
        prepare_latest_features()
    )

    upload_latest_features(
        latest_df
    )
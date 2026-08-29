import os
import pandas as pd
import hopsworks

from dotenv import load_dotenv
from pathlib import Path


load_dotenv()


DATA_PATH = Path(
    "data/processed/karachi_features.csv"
)


def connect_to_hopsworks():

    api_key = os.getenv(
        "HOPSWORKS_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "HOPSWORKS_API_KEY not found in .env"
        )

    print(
        "Connecting to Hopsworks..."
    )

    project = hopsworks.login(
        api_key_value=api_key
    )

    print(
        "Connected successfully."
    )

    return project


def upload_features():

    df = pd.read_csv(
        DATA_PATH
    )

    df["time"] = pd.to_datetime(
        df["time"]
    )

    print(
        "Feature dataset shape:",
        df.shape
    )

    project = connect_to_hopsworks()

    fs = project.get_feature_store()

    feature_group = fs.get_or_create_feature_group(
        name="karachi_aqi_features",
        version=2,
        description=(
            "Hourly AQI, pollution, weather "
            "and engineered features for Karachi"
        ),
        primary_key=["time"],
        event_time="time",
        online_enabled=False,
        time_travel_format="HUDI"
    )

    print(
        "Uploading features..."
    )

    feature_group.insert(
        df,
        write_options={
            "wait_for_job": True
        }
    )

    print(
        "Features uploaded successfully."
    )


if __name__ == "__main__":

    upload_features()
import pandas as pd
from pathlib import Path


INPUT_PATH = Path("data/raw/karachi_historical.csv")
OUTPUT_PATH = Path("data/processed/karachi_features.csv")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(INPUT_PATH)

    print("Raw shape:", df.shape)

    return df


def clean_data(df):
    # Convert time column
    df["time"] = pd.to_datetime(df["time"])

    # Sort chronologically
    df = df.sort_values("time").reset_index(drop=True)

    # Replace missing numeric values using interpolation
    numeric_cols = df.select_dtypes(include="number").columns

    df[numeric_cols] = df[numeric_cols].interpolate(
        method="linear"
    )

    # Remove any remaining missing values
    df = df.dropna()

    return df


def add_time_features(df):
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month

    return df


def add_aqi_features(df):
    # Previous AQI values
    df["aqi_lag_1"] = df["us_aqi"].shift(1)
    df["aqi_lag_3"] = df["us_aqi"].shift(3)
    df["aqi_lag_6"] = df["us_aqi"].shift(6)
    df["aqi_lag_24"] = df["us_aqi"].shift(24)

    # Rolling averages
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

    # AQI change rate
    df["aqi_change"] = df["us_aqi"].diff()

    return df


def add_targets(df):
    # Direct forecasting targets

    df["target_24h"] = df["us_aqi"].shift(-24)
    df["target_48h"] = df["us_aqi"].shift(-48)
    df["target_72h"] = df["us_aqi"].shift(-72)

    return df


def prepare_features():
    df = load_data()

    df = clean_data(df)
    df = add_time_features(df)
    df = add_aqi_features(df)
    df = add_targets(df)

    # Remove rows affected by shift/rolling operations
    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = prepare_features()

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nFeature engineering completed.")
    print("Final shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print(f"\nSaved to: {OUTPUT_PATH}")
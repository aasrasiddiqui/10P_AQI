import os
import tempfile
import hopsworks
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

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


from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


DATA_PATH = Path("data/processed/karachi_features.csv")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# Features used by our models
# -------------------------------------------------

FEATURES = [
    # Current pollutant information
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",

    # Current AQI
    "us_aqi",

    # Weather
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",

    # Time information
    "hour",
    "day",
    "day_of_week",
    "month",

    # Historical AQI
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_24",

    # Rolling AQI
    "aqi_rolling_mean_3",
    "aqi_rolling_mean_6",
    "aqi_rolling_mean_24",

    # Change
    "aqi_change"
]


TARGETS = {
    "24h": "target_24h",
    "48h": "target_48h",
    "72h": "target_72h"
}


def load_data():

    print("Connecting to Hopsworks Feature Store...")

    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )

    fs = project.get_feature_store()

    fg = fs.get_feature_group(
        name="karachi_aqi_features",
        version=2
    )

    print("Reading training data from Feature Store...")

    df = fg.read()

    df["time"] = pd.to_datetime(df["time"])

    # Hopsworks does not necessarily return rows chronologically
    df = df.sort_values("time").reset_index(drop=True)
    df = df.dropna(
    subset=[
        "target_24h",
        "target_48h",
        "target_72h"
    ]
    ).reset_index(drop=True)
    print("Dataset loaded:", df.shape)

    return df, project


def time_split(df):

    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    print("\nTraining rows:", len(train_df))
    print("Testing rows:", len(test_df))

    print(
        "Training period:",
        train_df["time"].min(),
        "to",
        train_df["time"].max()
    )

    print(
        "Testing period:",
        test_df["time"].min(),
        "to",
        test_df["time"].max()
    )

    return train_df, test_df


def get_models():

    models = {

        "Ridge": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                Ridge(alpha=1.0)
            )
        ]),

        "RandomForest": RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }

    return models


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return mae, rmse, r2


def train_models(df):

    train_df, test_df = time_split(df)

    results = []

    best_models = {}

    for horizon, target in TARGETS.items():

        print("\n" + "=" * 50)
        print(f"Training models for {horizon} forecast")
        print("=" * 50)

        X_train = train_df[FEATURES]
        y_train = train_df[target]

        X_test = test_df[FEATURES]
        y_test = test_df[target]

        models = get_models()

        best_rmse = float("inf")
        best_model = None
        best_model_name = None

        for model_name, model in models.items():

            print(
                f"\nTraining {model_name}..."
            )

            model.fit(
                X_train,
                y_train
            )

            mae, rmse, r2 = evaluate_model(
                model,
                X_test,
                y_test
            )

            print(
                f"MAE:  {mae:.3f}"
            )

            print(
                f"RMSE: {rmse:.3f}"
            )

            print(
                f"R²:   {r2:.3f}"
            )

            results.append({
                "forecast_horizon": horizon,
                "model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

            if rmse < best_rmse:

                best_rmse = rmse
                best_model = model
                best_model_name = model_name

        best_models[horizon] = best_model

        print(
            f"\nBest {horizon} model:",
            best_model_name
        )

        print(
            f"Best RMSE: {best_rmse:.3f}"
        )

        model_path = (
            MODEL_DIR /
            f"aqi_model_{horizon}.joblib"
        )

        joblib.dump(
            best_model,
            model_path
        )

        print(
            f"Saved model to: {model_path}"
        )

    return results, best_models


def register_models(project, results_df):

    print("\nConnecting to Hopsworks Model Registry...")

    mr = project.get_model_registry()

    for horizon in ["24h", "48h", "72h"]:

        model_path = MODEL_DIR / f"aqi_model_{horizon}.joblib"

        horizon_results = results_df[
            results_df["forecast_horizon"] == horizon
        ]

        best_row = horizon_results.loc[
            horizon_results["RMSE"].idxmin()
        ]

        model_name = f"aqi_predictor_{horizon}"

        print(
            f"\nRegistering {model_name}..."
        )

        model = mr.python.create_model(
            name=model_name,
            metrics={
                "mae": float(best_row["MAE"]),
                "rmse": float(best_row["RMSE"]),
                "r2": float(best_row["R2"])
            },
            description=(
                f"Best AQI forecasting model "
                f"for {horizon} prediction horizon"
            )
        )

        model.save(
            str(model_path)
        )

        print(
            f"{model_name} registered successfully."
        )

if __name__ == "__main__":

    df, project = load_data()

    results, best_models = train_models(df)

    results_df = pd.DataFrame(results)

    results_path = (
        MODEL_DIR /
        "model_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\n\nMODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to: {results_path}"
    )

    register_models(
        project,
        results_df
    )
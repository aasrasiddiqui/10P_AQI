import os
import tempfile
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import hopsworks

from dotenv import load_dotenv

from src.predict import (
    FEATURES,
    fetch_recent_data,
    keep_observed_hours,
    create_prediction_features,
    get_aqi_status,
)


# --------------------------------------------------
# Environment setup
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
# Streamlit page config
# --------------------------------------------------

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide"
)

def get_hopsworks_api_key():

    # Streamlit Cloud
    try:
        return st.secrets["HOPSWORKS_API_KEY"]
    except Exception:
        pass

    # Local development
    return os.getenv("HOPSWORKS_API_KEY")

def get_shap_explainer(model):
    return shap.Explainer(model)


def calculate_shap_values(model, X):

    explainer = get_shap_explainer(model)

    shap_values = explainer(X)

    return shap_values
# --------------------------------------------------
# Custom styling
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .aqi-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #f8f9fa;
        border: 1px solid #e5e7eb;
        text-align: center;
    }

    .aqi-number {
        font-size: 38px;
        font-weight: 700;
    }

    .aqi-label {
        font-size: 15px;
        color: #6b7280;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Load models from Hopsworks
# --------------------------------------------------

@st.cache_resource
def load_models_from_hopsworks():

    api_key = get_hopsworks_api_key()
    if not api_key:
        raise ValueError(
            "HOPSWORKS_API_KEY is missing."
        )

    project = hopsworks.login(
        api_key_value=api_key
    )

    mr = project.get_model_registry()

    model_names = {
        "24h": "aqi_predictor_24h",
        "48h": "aqi_predictor_48h",
        "72h": "aqi_predictor_72h",
    }

    models = {}

    for horizon, model_name in model_names.items():

        all_versions = mr.get_models(
            model_name)

        if not all_versions:
            raise ValueError(
                f"No registered models found "
                f"for {model_name}"
            )

        registered_model = max(
            all_versions,
            key=lambda model: model.version
        )
        downloaded_path = Path(
            registered_model.download()
        )

        # Find the saved joblib file
        joblib_files = list(
            downloaded_path.rglob("*.joblib")
        )

        if not joblib_files:
            raise FileNotFoundError(
                f"No .joblib file found "
                f"for {model_name}"
            )

        models[horizon] = joblib.load(
            joblib_files[0]
        )

    return models


# --------------------------------------------------
# Generate current prediction
# --------------------------------------------------

@st.cache_data(ttl=1800)
def get_prediction_data():

    df = fetch_recent_data()

    df = keep_observed_hours(df)

    df = create_prediction_features(df)

    valid_df = df.dropna(
        subset=FEATURES
    )

    if valid_df.empty:
        raise ValueError(
            "Not enough recent data "
            "for prediction."
        )

    latest_row = valid_df.iloc[-1]

    X = pd.DataFrame(
        [latest_row[FEATURES]]
    )

    models = load_models_from_hopsworks()

    predictions = {}

    for horizon, model in models.items():

        prediction = float(
            model.predict(X)[0]
        )

        prediction = max(
            0,
            prediction
        )

        predictions[horizon] = round(
            prediction,
            1
        )

    current_aqi = round(
        float(latest_row["us_aqi"]),
        1
    )

    return {
        "time": latest_row["time"],

        "current_aqi": current_aqi,
        "current_status": (
            get_aqi_status(
                current_aqi
            )
        ),

        "forecast_24h": (
            predictions["24h"]
        ),

        "forecast_48h": (
            predictions["48h"]
        ),

        "forecast_72h": (
            predictions["72h"]
        ),

        "forecast_24h_status": (
            get_aqi_status(
                predictions["24h"]
            )
        ),

        "forecast_48h_status": (
            get_aqi_status(
                predictions["48h"]
            )
        ),

        "forecast_72h_status": (
            get_aqi_status(
                predictions["72h"]
            )
        ),

        "pm2_5": round(
            float(latest_row["pm2_5"]),
            1
        ),

        "pm10": round(
            float(latest_row["pm10"]),
            1
        ),

        "temperature": round(
            float(
                latest_row[
                    "temperature_2m"
                ]
            ),
            1
        ),

        "humidity": round(
            float(
                latest_row[
                    "relative_humidity_2m"
                ]
            ),
            1
        ),

        "wind_speed": round(
            float(
                latest_row[
                    "wind_speed_10m"
                ]
            ),
            1
        ),  "input_features": X, "model_24h": models["24h"]
    }


# --------------------------------------------------
# AQI alert
# --------------------------------------------------

def show_aqi_alert(aqi):

    if aqi > 300:

        st.error(
            "🚨 Hazardous AQI predicted. "
            "Avoid outdoor exposure where possible."
        )

    elif aqi > 200:

        st.error(
            "⚠️ Very unhealthy AQI predicted."
        )

    elif aqi > 150:

        st.warning(
            "⚠️ Unhealthy AQI predicted. "
            "Consider reducing prolonged "
            "outdoor activity."
        )

    elif aqi > 100:

        st.warning(
            "Sensitive groups may experience "
            "health effects."
        )


# --------------------------------------------------
# Main application
# --------------------------------------------------

st.markdown(
    '<div class="main-title">'
    '🌍 Pearls AQI Predictor'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Karachi Air Quality Monitoring '
    'and 3-Day Forecast'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Refresh control
# --------------------------------------------------

col_refresh, col_space = st.columns(
    [1, 5]
)

with col_refresh:

    if st.button(
        "🔄 Refresh Data"
    ):

        st.cache_data.clear()

        st.rerun()


# --------------------------------------------------
# Load data
# --------------------------------------------------

try:

    with st.spinner(
        "Loading latest AQI data and forecasts..."
    ):

        result = get_prediction_data()


except Exception as e:

    st.error(
        "Unable to generate AQI forecast."
    )

    st.exception(e)

    st.stop()


# --------------------------------------------------
# Current AQI
# --------------------------------------------------

st.subheader(
    "Current Air Quality"
)

current_col1, current_col2 = st.columns(
    [1, 3]
)

with current_col1:

    st.metric(
        label="Current AQI",
        value=result["current_aqi"]
    )

with current_col2:

    st.markdown(
        f"### {result['current_status']}"
    )

    st.caption(
        f"Latest data: "
        f"{result['time']}"
    )


show_aqi_alert(
    result["current_aqi"]
)


st.divider()


# --------------------------------------------------
# 3-day forecast cards
# --------------------------------------------------

st.subheader(
    "3-Day AQI Forecast"
)

forecast_col1, forecast_col2, forecast_col3 = (
    st.columns(3)
)


with forecast_col1:

    st.metric(
        label="Next 24 Hours",
        value=result["forecast_24h"],
        delta=round(
            result["forecast_24h"]
            - result["current_aqi"],
            1
        )
    )

    st.caption(
        result["forecast_24h_status"]
    )


with forecast_col2:

    st.metric(
        label="Next 48 Hours",
        value=result["forecast_48h"],
        delta=round(
            result["forecast_48h"]
            - result["current_aqi"],
            1
        )
    )

    st.caption(
        result["forecast_48h_status"]
    )


with forecast_col3:

    st.metric(
        label="Next 72 Hours",
        value=result["forecast_72h"],
        delta=round(
            result["forecast_72h"]
            - result["current_aqi"],
            1
        )
    )

    st.caption(
        result["forecast_72h_status"]
    )


# Highest forecast alert
highest_forecast = max(
    result["forecast_24h"],
    result["forecast_48h"],
    result["forecast_72h"]
)

show_aqi_alert(
    highest_forecast
)


st.divider()


# --------------------------------------------------
# Forecast chart
# --------------------------------------------------

st.subheader(
    "AQI Forecast Trend"
)

forecast_df = pd.DataFrame(
    {
        "Period": [
            "Current",
            "+24 Hours",
            "+48 Hours",
            "+72 Hours"
        ],

        "AQI": [
            result["current_aqi"],
            result["forecast_24h"],
            result["forecast_48h"],
            result["forecast_72h"]
        ]
    }
)


fig = px.line(
    forecast_df,
    x="Period",
    y="AQI",
    markers=True,
    title="Current and Forecasted AQI"
)

fig.update_layout(
    yaxis_title="US AQI",
    xaxis_title=""
)

st.plotly_chart(
    fig,
    width="stretch"
)


st.divider()

#--------------------------------------------
st.divider()

st.subheader(
    "Why is the model predicting this AQI?"
)

st.write(
    "The chart below shows which features "
    "had the strongest influence on the "
    "24-hour AQI forecast."
)

shap_importance = None

try:

    X_explain = result["input_features"]

    model_24h = result["model_24h"]

    shap_values = calculate_shap_values(
        model_24h,
        X_explain
    )

    shap_importance = pd.DataFrame({
        "Feature": X_explain.columns,
        "SHAP Value": shap_values.values[0]
    })

    shap_importance["Importance"] = (
        shap_importance["SHAP Value"].abs()
    )

    shap_importance = (
        shap_importance
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(10)
    )

    fig_shap = px.bar(
        shap_importance,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        title="Top Factors Influencing the 24-Hour Forecast"
    )

    fig_shap.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        }
    )

    st.plotly_chart(
        fig_shap,
        width="stretch"
    )

except Exception as e:

    st.warning(
        "SHAP explanation could not be generated."
    )

    st.caption(str(e))


# Plain-English explanation
if shap_importance is not None and not shap_importance.empty:

    top_feature = shap_importance.iloc[0]

    feature_name = top_feature["Feature"]
    shap_effect = top_feature["SHAP Value"]

    if shap_effect > 0:
        direction = "increased the predicted AQI"
    else:
        direction = "reduced the predicted AQI"

    st.info(
        f"The most influential feature for this forecast was "
        f"**{feature_name}**, which {direction}."
    )
# --------------------------------------------------
# Pollutants
# --------------------------------------------------

st.subheader(
    "Current Pollutant Levels"
)

pollutant_col1, pollutant_col2 = (
    st.columns(2)
)

with pollutant_col1:

    st.metric(
        label="PM2.5",
        value=f"{result['pm2_5']} μg/m³"
    )

with pollutant_col2:

    st.metric(
        label="PM10",
        value=f"{result['pm10']} μg/m³"
    )


st.divider()


# --------------------------------------------------
# Weather
# --------------------------------------------------

st.subheader(
    "Current Weather Conditions"
)

weather_col1, weather_col2, weather_col3 = (
    st.columns(3)
)


with weather_col1:

    st.metric(
        label="🌡 Temperature",
        value=(
            f"{result['temperature']} °C"
        )
    )


with weather_col2:

    st.metric(
        label="💧 Humidity",
        value=(
            f"{result['humidity']} %"
        )
    )


with weather_col3:

    st.metric(
        label="💨 Wind Speed",
        value=(
            f"{result['wind_speed']} km/h"
        )
    )


st.divider()


# --------------------------------------------------
# System information
# --------------------------------------------------

with st.expander(
    "ℹ️ About the prediction system"
):

    st.write(
        """
        The AQI forecasting system combines
        pollutant measurements, weather conditions,
        time-based variables, AQI lag features and
        rolling statistics.

        Three separate machine-learning models are
        used for direct multi-horizon forecasting:

        - 24-hour AQI forecast
        - 48-hour AQI forecast
        - 72-hour AQI forecast

        The trained models are retrieved from the
        Hopsworks Model Registry.
        """
    )


st.caption(
    "AQI predictions are machine-learning estimates "
    "and may become less accurate at longer "
    "forecast horizons."
)
# Pearls AQI Predictor

A serverless machine-learning system for predicting
Karachi Air Quality Index up to 72 hours ahead.

## Features

- Real-time AQI and weather data retrieval
- Historical data backfill
- Automated feature engineering
- Hopsworks Feature Store
- 24h, 48h and 72h AQI forecasting
- Multiple ML model comparison
- Hopsworks Model Registry
- SHAP explainability
- Streamlit dashboard
- Hourly feature pipeline
- Daily model retraining
- GitHub Actions automation

## Architecture

Open-Meteo
→ Feature Engineering
→ Hopsworks Feature Store
→ Model Training
→ Hopsworks Model Registry
→ Prediction Pipeline
→ Streamlit Dashboard

## Data

The system uses hourly:

### Air-quality features
- PM2.5
- PM10
- CO
- NO2
- SO2
- O3
- US AQI

### Weather features
- Temperature
- Relative humidity
- Surface pressure
- Wind speed

### Engineered features
- Hour
- Day
- Day of week
- Month
- AQI lag 1h
- AQI lag 3h
- AQI lag 6h
- AQI lag 24h
- 3h rolling AQI
- 6h rolling AQI
- 24h rolling AQI
- AQI change

## Forecasting Strategy

Direct multi-horizon forecasting is used.

Separate models predict:

- AQI +24 hours
- AQI +48 hours
- AQI +72 hours

## Models

The following models were evaluated:

- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Models were evaluated using:

- MAE
- RMSE
- R²

## Results

| Horizon | Best Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| 24h | Gradient Boosting | 4.65 | 5.87 | 0.387 |
| 48h | Gradient Boosting | 7.52 | 9.02 | -0.291 |
| 72h | Gradient Boosting | 8.35 | 9.78 | -0.481 |

Forecast accuracy decreases as the forecasting
horizon increases.

## MLOps Pipeline

### Hourly

Open-Meteo
→ latest feature generation
→ Hopsworks live feature group

### Daily

Hopsworks historical Feature Store
→ model training
→ model evaluation
→ Hopsworks Model Registry

Both pipelines are automated using GitHub Actions.

## Explainability

SHAP is used to explain the 24-hour AQI prediction
and identify the features contributing most strongly
to each forecast.

## Dashboard

The Streamlit application displays:

- Current AQI
- AQI category
- 24h forecast
- 48h forecast
- 72h forecast
- AQI trend
- Pollutant concentrations
- Weather conditions
- SHAP explanations

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Open-Meteo
- Hopsworks
- SHAP
- Streamlit
- Plotly
- GitHub Actions

## Limitations

Longer-horizon forecasts currently show lower
predictive performance.

The 48-hour and 72-hour models have negative R²
scores on the chronological test set, indicating
that further feature engineering and model
optimization are required.

## Future Work

- XGBoost / LightGBM
- Hyperparameter optimization
- Cyclical time features
- Better long-horizon forecasting
- Multiple-city support
- Additional meteorological variables
- Prediction uncertainty intervals

- ## Live Demo

https://10p-internship-aqi.streamlit.app/

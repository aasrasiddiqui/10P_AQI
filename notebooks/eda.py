import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


DATA_PATH = Path("data/processed/karachi_features.csv")
OUTPUT_DIR = Path("data/processed/eda_plots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(DATA_PATH)

df["time"] = pd.to_datetime(df["time"])


print("Dataset shape:", df.shape)

print("\nAQI Statistics:")
print(df["us_aqi"].describe())

print("\nCorrelation with AQI:")
print(
    df.corr(numeric_only=True)["us_aqi"]
    .sort_values(ascending=False)
)


# --------------------------------
# 1. AQI over time
# --------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["time"],
    df["us_aqi"]
)

plt.title("AQI Over Time")
plt.xlabel("Time")
plt.ylabel("US AQI")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "aqi_over_time.png"
)

plt.close()


# --------------------------------
# 2. Average AQI by hour
# --------------------------------

hourly_aqi = (
    df.groupby("hour")["us_aqi"]
    .mean()
)

plt.figure(figsize=(8, 5))

plt.plot(
    hourly_aqi.index,
    hourly_aqi.values,
    marker="o"
)

plt.title("Average AQI by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average AQI")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "aqi_by_hour.png"
)

plt.close()


# --------------------------------
# 3. PM2.5 vs AQI
# --------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["pm2_5"],
    df["us_aqi"],
    alpha=0.4
)

plt.title("PM2.5 vs AQI")
plt.xlabel("PM2.5")
plt.ylabel("US AQI")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pm25_vs_aqi.png"
)

plt.close()


# --------------------------------
# 4. Average AQI by month
# --------------------------------

monthly_aqi = (
    df.groupby("month")["us_aqi"]
    .mean()
)

plt.figure(figsize=(8, 5))

plt.bar(
    monthly_aqi.index,
    monthly_aqi.values
)

plt.title("Average AQI by Month")
plt.xlabel("Month")
plt.ylabel("Average AQI")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "aqi_by_month.png"
)

plt.close()


print("\nEDA completed.")
print(f"Plots saved to: {OUTPUT_DIR}")
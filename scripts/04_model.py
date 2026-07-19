"""Phase 3: train forecasting models, compare them, and forecast the next 14 days.

Usage:
    python 04_model.py

Models compared on the last 28 days (held out):
    - Seasonal naive (same weekday last week)  <- baseline to beat
    - SARIMAX with weekly seasonality
    - XGBoost on calendar/lag/weather features

Outputs (in outputs/):
    model_comparison.csv, 05_model_comparison.png, 06_forecast.png,
    07_feature_importance.png, forecast_next14.csv, staffing_recommendations.csv
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

import viz_style
from config import OUTPUT_DIR, PROCESSED_DIR

TEST_DAYS = 28
HORIZON = 14

FEATURES = ["day_of_week", "is_weekend", "month", "week_of_year", "is_holiday",
            "orders_lag_1", "orders_lag_7", "orders_roll_7"]
WEATHER_FEATURES = ["temp_max_f", "temp_min_f", "precip_in"]


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - np.asarray(y_pred)[mask]) / y_true[mask])) * 100)


def main() -> None:
    viz_style.apply()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_DIR / "daily_features.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    features = FEATURES + [c for c in WEATHER_FEATURES if c in df.columns
                           and df[c].notna().mean() > 0.9]
    model_df = df.dropna(subset=["orders_lag_7", "orders_roll_7"]).reset_index(drop=True)
    train, test = model_df.iloc[:-TEST_DAYS], model_df.iloc[-TEST_DAYS:]
    print(f"Training on {len(train)} days, testing on last {len(test)} days")

    results = {}

    # 1. Seasonal naive baseline: predict same weekday last week
    naive_pred = test["orders_lag_7"].values
    results["Seasonal naive"] = naive_pred

    # 2. SARIMAX with weekly seasonality
    sarima = SARIMAX(train["orders"], order=(1, 0, 1),
                     seasonal_order=(1, 1, 1, 7),
                     enforce_stationarity=False).fit(disp=False)
    results["SARIMA"] = sarima.forecast(steps=len(test)).values

    # 3. XGBoost on engineered features
    xgb = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                       subsample=0.9, random_state=42)
    xgb.fit(train[features], train["orders"])
    results["XGBoost"] = xgb.predict(test[features])

    # Score
    rows = []
    for name, pred in results.items():
        rows.append({"model": name,
                     "MAE": round(mean_absolute_error(test["orders"], pred), 2),
                     "MAPE_%": round(mape(test["orders"], pred), 1)})
    scores = pd.DataFrame(rows).sort_values("MAE")
    scores.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\n", scores.to_string(index=False))
    best_name = scores.iloc[0]["model"]
    print(f"\nBest model on holdout: {best_name}")

    # Chart: actual vs predictions on the test window
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(test["date"], test["orders"], color=viz_style.INK, linewidth=2.2,
            label="Actual")
    for (name, pred), color in zip(results.items(), viz_style.SERIES):
        ax.plot(test["date"], pred, color=color, linewidth=1.6, alpha=0.9,
                label=name)
    ax.set_title(f"Holdout test: last {TEST_DAYS} days")
    ax.set_ylabel("Orders")
    ax.legend(frameon=False, ncols=2)
    viz_style.format_dates(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_model_comparison.png")

    # Feature importance from XGBoost
    imp = (pd.Series(xgb.feature_importances_, index=features)
           .sort_values())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(imp.index, imp.values, color=viz_style.BLUE, height=0.6)
    ax.set_title("XGBoost feature importance")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_feature_importance.png")

    # Refit XGBoost on all data and forecast the next 14 days iteratively
    xgb.fit(model_df[features], model_df["orders"])
    history = model_df[["date", "orders"]].copy()
    weather_means = {c: model_df[c].tail(28).mean()
                     for c in features if c in WEATHER_FEATURES}
    future_rows = []
    for i in range(1, HORIZON + 1):
        d = model_df["date"].max() + pd.Timedelta(days=i)
        recent = history["orders"].values
        row = {
            "day_of_week": d.dayofweek,
            "is_weekend": int(d.dayofweek >= 5),
            "month": d.month,
            "week_of_year": int(d.isocalendar().week),
            "is_holiday": 0,
            "orders_lag_1": recent[-1],
            "orders_lag_7": recent[-7],
            "orders_roll_7": recent[-7:].mean(),
            **weather_means,
        }
        pred = float(xgb.predict(pd.DataFrame([row])[features])[0])
        pred = max(pred, 0.0)
        future_rows.append({"date": d, "forecast_orders": round(pred, 1),
                            "day_name": d.day_name()})
        history = pd.concat([history, pd.DataFrame([{"date": d, "orders": pred}])],
                            ignore_index=True)

    forecast = pd.DataFrame(future_rows)
    forecast.to_csv(OUTPUT_DIR / "forecast_next14.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    recent = model_df.tail(60)
    ax.plot(recent["date"], recent["orders"], color=viz_style.INK,
            linewidth=1.8, label="Actual (last 60 days)")
    ax.plot(forecast["date"], forecast["forecast_orders"], color=viz_style.BLUE,
            linewidth=2.2, linestyle="--", label="14-day forecast")
    ax.set_title("Order forecast: next 14 days")
    ax.set_ylabel("Orders")
    ax.legend(frameon=False)
    viz_style.format_dates(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_forecast.png")

    # Staffing recommendation: classify each forecast day vs typical volume
    q33, q67 = model_df["orders"].quantile([0.33, 0.67])
    def tier(v):
        if v >= q67:
            return "Busy - add staff"
        if v <= q33:
            return "Slow - minimum staff"
        return "Normal"
    forecast["staffing"] = forecast["forecast_orders"].map(tier)
    forecast.to_csv(OUTPUT_DIR / "staffing_recommendations.csv", index=False)
    print("\nNext 14 days:\n", forecast.to_string(index=False))
    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


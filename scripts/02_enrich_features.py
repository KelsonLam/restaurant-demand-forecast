"""Phase 2: add weather, holiday, and calendar features to the daily sales table.

Usage:
    python 02_enrich_features.py

Weather comes from the free Open-Meteo archive API (no key needed) for the
location set in config.py. If the request fails (offline), the pipeline
continues without weather and warns.

Output: data/processed/daily_features.csv (platform == 'all' series only)
"""

import holidays as holidays_lib
import pandas as pd
import requests

from config import HOLIDAY_STATE, LATITUDE, LONGITUDE, PROCESSED_DIR, TIMEZONE


def fetch_weather(start, end) -> pd.DataFrame | None:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&start_date={start}&end_date={end}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&temperature_unit=fahrenheit&precipitation_unit=inch&timezone={TIMEZONE}"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        d = r.json()["daily"]
    except Exception as e:  # noqa: BLE001 - any failure means "no weather"
        print(f"! Weather fetch failed ({e}); continuing without weather features")
        return None
    return pd.DataFrame({
        "date": pd.to_datetime(d["time"]).date,
        "temp_max_f": d["temperature_2m_max"],
        "temp_min_f": d["temperature_2m_min"],
        "precip_in": d["precipitation_sum"],
    })


def main() -> None:
    daily = pd.read_csv(PROCESSED_DIR / "daily_sales.csv", parse_dates=["date"])
    df = daily[daily["platform"] == "all"].copy()
    df["date"] = df["date"].dt.date
    df = df.sort_values("date").reset_index(drop=True)

    # Calendar features
    dates = pd.to_datetime(df["date"])
    df["day_of_week"] = dates.dt.dayofweek
    df["day_name"] = dates.dt.day_name()
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = dates.dt.month
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)

    # Holidays
    us_hols = holidays_lib.US(state=HOLIDAY_STATE,
                              years=sorted(dates.dt.year.unique()))
    df["is_holiday"] = df["date"].map(lambda d: int(d in us_hols))
    df["holiday_name"] = df["date"].map(lambda d: us_hols.get(d, ""))

    # Weather
    weather = fetch_weather(df["date"].min(), df["date"].max())
    if weather is not None:
        df = df.merge(weather, on="date", how="left")
        print(f"Merged weather for {weather.shape[0]} days "
              f"(lat {LATITUDE}, lon {LONGITUDE})")

    # Lag / rolling features for the models
    df["orders_lag_1"] = df["orders"].shift(1)
    df["orders_lag_7"] = df["orders"].shift(7)
    df["orders_roll_7"] = df["orders"].shift(1).rolling(7).mean()
    df["revenue_lag_7"] = df["revenue"].shift(7)

    out = PROCESSED_DIR / "daily_features.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} columns -> {out}")


if __name__ == "__main__":
    main()

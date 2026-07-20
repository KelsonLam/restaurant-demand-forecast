"""Build a calibrated SIMULATED full-house dataset for a Knoxville steakhouse.

Covers every channel, not just delivery: dine-in checks, direct takeout,
Uber Eats, and DoorDash. No real steakhouse publishes its complete POS
records, so this stands in until real exports are available.

Base: the public Maven Analytics "Pizza Place Sales" dataset (21k real
full-service restaurant orders over one year) supplies the hour-of-day and
month-of-year demand shape. Each channel is rescaled to steakhouse norms and
modulated by REAL Knoxville weather (Open-Meteo) and US holidays:
  - rain nudges delivery UP (~+9%) and walk-in dine-in slightly DOWN (~-4%)
  - occasion nights (Valentine's, Mother's/Father's Day, NYE) spike the
    dining room hardest
  - dinner dominates everywhere; dine-in keeps a real lunch trade

Usage: python make_calibrated_data.py
"""

import numpy as np
import pandas as pd
import requests
import holidays as holidays_lib

from config import HOLIDAY_STATE, LATITUDE, LONGITUDE, PROJECT_ROOT, RAW_DIR, TIMEZONE

PUBLIC = PROJECT_ROOT / "data" / "public"
START, END = "2025-07-01", "2026-07-15"

CLOSED_HOLIDAYS = {"Thanksgiving", "Christmas Day"}

# Steakhouse weekly rhythm (mean = 1.0): Fri/Sat peak, Monday trough.
DOW_PROFILE = {0: 0.78, 1: 0.84, 2: 0.90, 3: 0.98, 4: 1.22, 5: 1.32, 6: 0.96}

# Per-channel targets and behavior.
CHANNELS = {
    "dine_in": dict(daily=66, ticket=118.0, rain=0.96, occasion=1.0,
                    lunch_wd=0.55, lunch_we=0.90, dinner=1.5,
                    fname="pos_dinein_checks_simulated.csv"),
    "takeout": dict(daily=9, ticket=72.0, rain=1.00, occasion=0.5,
                    lunch_wd=0.40, lunch_we=0.60, dinner=1.6,
                    fname="pos_takeout_orders_simulated.csv"),
    "ubereats": dict(daily=12, ticket=74.0, rain=1.09, occasion=0.5,
                     lunch_wd=0.30, lunch_we=0.60, dinner=1.6,
                     fname="ubereats_orders_simulated.csv"),
    "doordash": dict(daily=10, ticket=74.0, rain=1.09, occasion=0.5,
                     lunch_wd=0.30, lunch_we=0.60, dinner=1.6,
                     fname="doordash_orders_simulated.csv"),
}

rng = np.random.default_rng(7)


def occasion_boost(day) -> float:
    """Occasion spikes a steakhouse actually sees (not federal holidays)."""
    if (day.month, day.day) == (2, 14):                      # Valentine's Day
        return 1.9
    if day.month == 5 and day.dayofweek == 6 and 8 <= day.day <= 14:
        return 1.7                                           # Mother's Day
    if day.month == 6 and day.dayofweek == 6 and 15 <= day.day <= 21:
        return 1.5                                           # Father's Day
    if (day.month, day.day) == (12, 31):                     # New Year's Eve
        return 1.35
    return 1.0


def load_base_patterns():
    orders = pd.read_csv(PUBLIC / "orders.csv")
    details = pd.read_csv(PUBLIC / "order_details.csv")
    pizzas = pd.read_csv(PUBLIC / "pizzas.csv")

    line = details.merge(pizzas[["pizza_id", "price"]], on="pizza_id")
    line["value"] = line["price"] * line["quantity"]
    totals = line.groupby("order_id")["value"].sum()

    orders["ts"] = pd.to_datetime(orders["date"] + " " + orders["time"])
    orders["dow"] = orders["ts"].dt.dayofweek
    orders["hour"] = orders["ts"].dt.hour
    orders = orders.merge(totals.rename("total"), left_on="order_id",
                          right_index=True)

    daily = orders.groupby(orders["ts"].dt.date).size()
    idx = pd.to_datetime(pd.Series(daily.index))
    month_index = daily.groupby(idx.dt.month.values).mean()
    month_index /= month_index.mean()

    # Raw hour histogram per weekday, delivery/service hours 11:00-21:59.
    hour_hist = {}
    for d in range(7):
        h = orders.loc[(orders["dow"] == d) & orders["hour"].between(11, 21),
                       "hour"].value_counts().reindex(range(11, 22), fill_value=0)
        hour_hist[d] = h.astype(float)
    return month_index, hour_hist, orders["total"].values


def hour_probs_for(cfg, hour_hist):
    """Reweight the base hour shape for one channel."""
    out = {}
    for d in range(7):
        h = hour_hist[d].copy()
        for hr in h.index:
            if hr <= 13:
                h[hr] *= cfg["lunch_we"] if d >= 5 else cfg["lunch_wd"]
            elif hr <= 16:
                h[hr] *= 0.5
            else:
                h[hr] *= cfg["dinner"]
        out[d] = (h / h.sum()).values
    return out


def fetch_weather():
    cache = PUBLIC / f"weather_{LATITUDE}_{LONGITUDE}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={LATITUDE}&longitude={LONGITUDE}"
           f"&start_date={START}&end_date={END}"
           "&daily=precipitation_sum,temperature_2m_max"
           f"&precipitation_unit=inch&temperature_unit=fahrenheit&timezone={TIMEZONE}")
    d = requests.get(url, timeout=30).json()["daily"]
    w = pd.DataFrame({"date": pd.to_datetime(d["time"]),
                      "precip_in": d["precipitation_sum"],
                      "temp_max_f": d["temperature_2m_max"]})
    w.to_csv(cache, index=False)
    return w


def simulate_channel(name, cfg, month_index, hour_hist, base_totals, weather,
                     us_hols):
    hour_probs = hour_probs_for(cfg, hour_hist)
    ticket_scale = cfg["ticket"] / base_totals.mean()
    days = pd.date_range(START, END, freq="D")
    rows = []
    for day in days:
        hol = us_hols.get(day.date())
        if hol in CLOSED_HOLIDAYS:
            continue
        mu = cfg["daily"] * DOW_PROFILE[day.dayofweek] * month_index[day.month]
        w = weather.loc[day] if day in weather.index else None
        if w is not None and pd.notna(w["precip_in"]) and w["precip_in"] > 0.1:
            mu *= cfg["rain"]
        if hol:
            mu *= 0.9
        mu *= 1 + (occasion_boost(day) - 1) * cfg["occasion"]
        mu *= rng.gamma(30, 1 / 30)      # day-level noise beyond Poisson
        n = rng.poisson(mu)
        hours = rng.choice(np.arange(11, 22), size=n,
                           p=hour_probs[day.dayofweek])
        base = rng.choice(base_totals, size=n)
        totals = np.round(base * ticket_scale * rng.normal(1, 0.14, n), 2)
        totals = np.clip(totals, 25.0, None)
        for h, t in zip(hours, totals):
            rows.append((day.replace(hour=int(h),
                                     minute=int(rng.integers(0, 60)),
                                     second=int(rng.integers(0, 60))),
                         float(t)))
    return pd.DataFrame(rows, columns=["ts", "total"]).sort_values("ts")


def write_export(name, df):
    if name == "ubereats":
        out = pd.DataFrame({
            "Order ID": [f"UE-{i:06d}" for i in range(len(df))],
            "Time Customer Ordered": df["ts"].dt.strftime("%m/%d/%Y %I:%M %p"),
            "Order Status": "Completed",
            "Food Sales": df["total"].values,
        })
    elif name == "doordash":
        out = pd.DataFrame({
            "Order ID": [f"DD-{i:06d}" for i in range(len(df))],
            "Timestamp Local Date": df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Order Status": "Delivered",
            "Subtotal": df["total"].values,
        })
    else:  # POS-style export (dine-in / takeout)
        out = pd.DataFrame({
            "Check Number": [f"{i + 1:06d}" for i in range(len(df))],
            "Opened At": df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Order Type": "Dine-In" if name == "dine_in" else "Takeout",
            "Check Total": df["total"].values,
        })
    out.to_csv(RAW_DIR / CHANNELS[name]["fname"], index=False)


def main() -> None:
    month_index, hour_hist, base_totals = load_base_patterns()
    weather = fetch_weather().set_index("date")
    us_hols = holidays_lib.US(state=HOLIDAY_STATE, years=[2025, 2026])

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, cfg in CHANNELS.items():
        df = simulate_channel(name, cfg, month_index, hour_hist, base_totals,
                              weather, us_hols)
        write_export(name, df)
        summary.append(f"  {name:9s} {len(df):6d} orders, "
                       f"avg ${df['total'].mean():.2f}")
        print(summary[-1])

    (RAW_DIR / "DATA_README.txt").write_text(
        "SIMULATED DATA - calibrated, not actual restaurant records.\n"
        "Full-house steakhouse profile: dine-in checks, takeout, Uber Eats,\n"
        "DoorDash. Demand shape from the public Maven Analytics 'Pizza Place\n"
        "Sales' dataset (21k real full-service orders); rescaled to steakhouse\n"
        "norms (dinner-dominant, Fri/Sat peaks, occasion spikes on Valentine's/\n"
        "Mother's/Father's Day and NYE) and modulated by real Knoxville, TN\n"
        "weather (rain lifts delivery, dents walk-ins) and TN holidays.\n"
        "Replace these files with real POS / Uber Eats Manager / DoorDash\n"
        "Merchant Portal exports; the pipeline runs unchanged.\n",
        encoding="utf-8")
    print(f"Wrote exports to {RAW_DIR}")


if __name__ == "__main__":
    main()

"""Build a calibrated SIMULATED dataset for an upscale Knoxville steakhouse.

Base: the public Maven Analytics "Pizza Place Sales" dataset (21k real
restaurant orders over one year, data/public/) supplies the demand *shape* --
hour-of-day by weekday, day-of-week mix, and month-of-year seasonality.
That shape is rescaled to a delivery-only upscale steakhouse profile
(dinner-dominant, Fri/Sat peaks, occasion spikes) and modulated by REAL
Knoxville weather (Open-Meteo) and US holidays, then written out in
Uber Eats / DoorDash export formats to data/raw/.

This is simulated data. It stands in until the real Uber Eats Manager /
DoorDash Merchant Portal exports are available.

Usage: python make_calibrated_data.py
"""

import numpy as np
import pandas as pd
import requests
import holidays as holidays_lib

from config import HOLIDAY_STATE, LATITUDE, LONGITUDE, PROJECT_ROOT, RAW_DIR, TIMEZONE

PUBLIC = PROJECT_ROOT / "data" / "public"
START, END = "2025-07-01", "2026-07-15"

# Target scale: upscale steakhouse, delivery channels only. Lower volume,
# much higher ticket than casual concepts.
TARGET_DAILY_ORDERS = 22          # combined Uber Eats + DoorDash average
UBEREATS_SHARE = 0.55
TARGET_AVG_TICKET = 74.0          # steak entrees + sides skew checks high
CLOSED_HOLIDAYS = {"Thanksgiving", "Christmas Day"}

# The pizza base data is dine-in/carryout shaped (Friday peak, big weekday
# lunch). A steakhouse's delivery demand is strongly dinner- and
# weekend-skewed, so these profiles replace the base day-of-week mix and
# reweight the hourly shape. Mean of DOW = 1.0.
DOW_DELIVERY = {0: 0.78, 1: 0.84, 2: 0.90, 3: 0.98, 4: 1.22, 5: 1.32, 6: 0.96}


def hour_weight(hour: int, dow: int) -> float:
    if hour <= 13:                      # steakhouse lunch delivery is thin
        return 0.6 if dow >= 5 else 0.3
    if hour <= 16:
        return 0.5
    return 1.6                          # dinner overwhelmingly dominates


def occasion_boost(day) -> float:
    """Steakhouse occasion spikes not in the federal holiday calendar."""
    if (day.month, day.day) == (2, 14):                      # Valentine's Day
        return 1.7
    if day.month == 5 and day.dayofweek == 6 and 8 <= day.day <= 14:
        return 1.6                                           # Mother's Day
    if day.month == 6 and day.dayofweek == 6 and 15 <= day.day <= 21:
        return 1.45                                          # Father's Day
    if (day.month, day.day) == (12, 31):                     # New Year's Eve
        return 1.3
    return 1.0

rng = np.random.default_rng(7)


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
    orders["month"] = orders["ts"].dt.month
    orders = orders.merge(totals.rename("total"), left_on="order_id",
                          right_index=True)

    daily = orders.groupby(orders["ts"].dt.date).size()
    idx = pd.to_datetime(pd.Series(daily.index))
    month_index = daily.groupby(idx.dt.month.values).mean()
    month_index /= month_index.mean()
    dow_index = pd.Series(DOW_DELIVERY)

    # Hour histogram per weekday from the base data, clipped to delivery
    # hours 11:00-21:59, then reweighted toward a dinner-heavy profile.
    hour_probs = {}
    for d in range(7):
        h = orders.loc[(orders["dow"] == d) & orders["hour"].between(11, 21),
                       "hour"].value_counts().reindex(range(11, 22), fill_value=0)
        w = h.astype(float) * [hour_weight(hr, d) for hr in h.index]
        hour_probs[d] = (w / w.sum()).values

    ticket_scale = TARGET_AVG_TICKET / orders["total"].mean()
    return dow_index, month_index, hour_probs, orders["total"].values, ticket_scale


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


def main() -> None:
    dow_index, month_index, hour_probs, base_totals, ticket_scale = load_base_patterns()
    weather = fetch_weather().set_index("date")
    us_hols = holidays_lib.US(state=HOLIDAY_STATE, years=[2025, 2026])

    days = pd.date_range(START, END, freq="D")
    all_orders = []
    for day in days:
        hol = us_hols.get(day.date())
        if hol in CLOSED_HOLIDAYS:
            continue
        mu = (TARGET_DAILY_ORDERS
              * dow_index[day.dayofweek]
              * month_index[day.month])
        w = weather.loc[day] if day in weather.index else None
        if w is not None and pd.notna(w["precip_in"]) and w["precip_in"] > 0.1:
            mu *= 1.09  # rain nudges people toward delivery
        if hol:  # open but slower on most federal holidays
            mu *= 0.9
        mu *= occasion_boost(day)
        # day-level noise beyond Poisson (supply, promos, randomness)
        mu *= rng.gamma(30, 1 / 30)
        n = rng.poisson(mu)
        hours = rng.choice(np.arange(11, 22), size=n, p=hour_probs[day.dayofweek])
        base = rng.choice(base_totals, size=n)
        totals = np.round(base * ticket_scale * rng.normal(1, 0.14, n), 2)
        totals = np.clip(totals, 25.0, None)
        for h, t in zip(hours, totals):
            ts = day.replace(hour=int(h), minute=int(rng.integers(0, 60)),
                             second=int(rng.integers(0, 60)))
            all_orders.append((ts, float(t)))

    df = pd.DataFrame(all_orders, columns=["ts", "total"]).sort_values("ts")
    df["platform"] = np.where(rng.random(len(df)) < UBEREATS_SHARE,
                              "ubereats", "doordash")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ue = df[df["platform"] == "ubereats"].reset_index(drop=True)
    pd.DataFrame({
        "Order ID": [f"UE-{i:06d}" for i in range(len(ue))],
        "Time Customer Ordered": ue["ts"].dt.strftime("%m/%d/%Y %I:%M %p"),
        "Order Status": "Completed",
        "Food Sales": ue["total"],
    }).to_csv(RAW_DIR / "ubereats_orders_simulated.csv", index=False)

    dd = df[df["platform"] == "doordash"].reset_index(drop=True)
    pd.DataFrame({
        "Order ID": [f"DD-{i:06d}" for i in range(len(dd))],
        "Timestamp Local Date": dd["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Order Status": "Delivered",
        "Subtotal": dd["total"],
    }).to_csv(RAW_DIR / "doordash_orders_simulated.csv", index=False)

    (RAW_DIR / "DATA_README.txt").write_text(
        "SIMULATED DATA - calibrated, not actual restaurant records.\n"
        "Demand shape: Maven Analytics 'Pizza Place Sales' public dataset\n"
        "(hour-of-day, day-of-week, month-of-year patterns from 21k real orders).\n"
        "Rescaled to a delivery-only upscale steakhouse profile (~22 orders/day,\n"
        "~$74 avg ticket, dinner-dominant, Fri/Sat peaks, occasion spikes on\n"
        "Valentine's/Mother's/Father's Day and NYE) and modulated by real\n"
        "Knoxville, TN weather (Open-Meteo) and TN holidays.\n"
        "Replace these files with real Uber Eats Manager / DoorDash Merchant\n"
        "Portal exports when available; the pipeline runs unchanged.\n",
        encoding="utf-8")

    print(f"{len(ue)} Uber Eats + {len(dd)} DoorDash simulated orders "
          f"({df['ts'].min().date()} to {df['ts'].max().date()}) -> {RAW_DIR}")
    print(f"Avg ticket: ${df['total'].mean():.2f}   "
          f"Avg orders/day: {len(df) / df['ts'].dt.date.nunique():.1f}")


if __name__ == "__main__":
    main()

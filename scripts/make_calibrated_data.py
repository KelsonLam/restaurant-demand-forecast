"""Build a calibrated SIMULATED full-house dataset for a Knoxville steakhouse.

Covers every channel, not just delivery: dine-in checks, direct takeout,
Uber Eats, and DoorDash. No real steakhouse publishes its complete POS
records, so this stands in until real exports are available.

Base: the public Maven Analytics "Pizza Place Sales" dataset (21k real
full-service restaurant orders over one year) supplies exactly two things,
chosen because they are generic across full-service restaurants rather than
specific to pizza:
  - the within-day hour-of-day rhythm (lunch bump, dinner rush), further
    reweighted per channel toward a steakhouse's dinner-dominant mix
  - the shape of the order-value distribution (some small checks, some
    large), rescaled to each channel's target average ticket

A pizza place's day-of-week pattern, month-to-month seasonality, and
occasion calendar do NOT transfer to a steakhouse; a peak day for pizza
delivery (a Sunday afternoon, game-day snacking) can be a trough for a
steakhouse and vice versa. So those three are built from scratch below,
specific to upscale steakhouse dining, and never read from the pizza data:
  - MONTH_PROFILE: a hand-set seasonal curve (holiday-party season high in
    Nov-Dec, post-holiday dip in Jan, summer travel dip in Jul-Aug)
  - DOW_PROFILE: Friday/Saturday dinner peak, Monday trough
  - occasion_boost(): steakhouse-specific occasion nights, including a
    DOWN day for Super Bowl Sunday, which is a strong UP day for pizza and
    wings but a historically slow night for sit-down steak service since
    the occasion is built around watching the game at home

Each channel is also modulated by REAL Knoxville weather (Open-Meteo) and
US holidays:
  - rain nudges delivery UP (~+9%) and walk-in dine-in slightly DOWN (~-4%)
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

# Steakhouse seasonal rhythm (mean = 1.0), hand-set, NOT derived from the
# pizza base data: slow post-holiday January, a summer travel dip, and a
# strong run-up through the holiday party season into December/NYE.
MONTH_PROFILE = {1: 0.86, 2: 0.98, 3: 0.98, 4: 1.00, 5: 1.04, 6: 1.02,
                 7: 0.94, 8: 0.92, 9: 0.98, 10: 1.02, 11: 1.10, 12: 1.22}

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


def super_bowl_sunday(year: int):
    """Second Sunday of February -- matches recent actual Super Bowl dates
    closely enough for a demand-modeling proxy without hardcoding a fact."""
    sundays = [d for d in pd.date_range(f"{year}-02-01", f"{year}-02-28")
              if d.dayofweek == 6]
    return sundays[1].date()


def occasion_boost(day) -> float:
    """Occasion effects a steakhouse actually sees (not federal holidays).

    Not every occasion that helps a casual/delivery concept helps a
    steakhouse: Super Bowl Sunday is one of the single biggest days of the
    year for pizza and wings, built entirely around watching the game at
    home, and a well-known slow night for sit-down steak service. Modeling
    it as a boost (as a straight reuse of pizza seasonality would) would be
    wrong in direction, not just magnitude.
    """
    if (day.month, day.day) == (2, 14):                      # Valentine's Day
        return 1.9
    if day.month == 5 and day.dayofweek == 6 and 8 <= day.day <= 14:
        return 1.7                                           # Mother's Day
    if day.month == 6 and day.dayofweek == 6 and 15 <= day.day <= 21:
        return 1.5                                           # Father's Day
    if (day.month, day.day) == (12, 31):                     # New Year's Eve
        return 1.35
    if day.date() == super_bowl_sunday(day.year):
        return 0.72                                          # Super Bowl Sunday
    return 1.0


def load_base_patterns():
    """Load only the two generic, transferable shapes from the pizza data:
    within-day hour rhythm and order-value distribution. Day-of-week,
    month, and occasion effects are NOT read from here -- see module
    docstring."""
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

    # Raw hour histogram per weekday, delivery/service hours 11:00-21:59.
    hour_hist = {}
    for d in range(7):
        h = orders.loc[(orders["dow"] == d) & orders["hour"].between(11, 21),
                       "hour"].value_counts().reindex(range(11, 22), fill_value=0)
        hour_hist[d] = h.astype(float)
    return hour_hist, orders["total"].values


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


def simulate_channel(name, cfg, hour_hist, base_totals, weather, us_hols):
    hour_probs = hour_probs_for(cfg, hour_hist)
    ticket_scale = cfg["ticket"] / base_totals.mean()
    days = pd.date_range(START, END, freq="D")
    rows = []
    for day in days:
        hol = us_hols.get(day.date())
        if hol in CLOSED_HOLIDAYS:
            continue
        mu = cfg["daily"] * DOW_PROFILE[day.dayofweek] * MONTH_PROFILE[day.month]
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
    hour_hist, base_totals = load_base_patterns()
    weather = fetch_weather().set_index("date")
    us_hols = holidays_lib.US(state=HOLIDAY_STATE, years=[2025, 2026])

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, cfg in CHANNELS.items():
        df = simulate_channel(name, cfg, hour_hist, base_totals,
                              weather, us_hols)
        write_export(name, df)
        summary.append(f"  {name:9s} {len(df):6d} orders, "
                       f"avg ${df['total'].mean():.2f}")
        print(summary[-1])

    (RAW_DIR / "DATA_README.txt").write_text(
        "Full-house steakhouse profile: dine-in checks, takeout, Uber Eats,\n"
        "DoorDash. The public Maven Analytics 'Pizza Place Sales' dataset\n"
        "(21k real full-service orders) supplies only the within-day hour\n"
        "rhythm and order-value distribution shape, both generic across\n"
        "full-service restaurants. Day-of-week rhythm, month-to-month\n"
        "seasonality, and occasion effects are hand-built for a steakhouse,\n"
        "not read from the pizza data, since those do not transfer: this\n"
        "profile is dinner-dominant with Fri/Sat peaks, spikes on Valentine's/\n"
        "Mother's/Father's Day and NYE, and a deliberate DOWN day on Super\n"
        "Bowl Sunday (a peak day for pizza, a slow night for steak service).\n"
        "Modulated by real Knoxville, TN weather (rain lifts delivery, dents\n"
        "walk-ins) and TN holidays.\n"
        "Replace these files with real POS / Uber Eats Manager / DoorDash\n"
        "Merchant Portal exports; the pipeline runs unchanged.\n",
        encoding="utf-8")
    print(f"Wrote exports to {RAW_DIR}")


if __name__ == "__main__":
    main()

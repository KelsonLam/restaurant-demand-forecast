"""Generate SYNTHETIC Uber Eats / DoorDash style order exports.

Only for testing that the pipeline runs end to end. The real project must
use the actual exports from Uber Eats Manager and the DoorDash Merchant
Portal, dropped into data/raw/.
"""

import numpy as np
import pandas as pd

from config import SAMPLE_DIR

rng = np.random.default_rng(42)

START = "2025-07-01"
END = "2026-07-15"


def simulate_orders(platform: str, base_daily: float) -> pd.DataFrame:
    days = pd.date_range(START, END, freq="D")
    rows = []
    for day in days:
        dow_boost = {4: 1.35, 5: 1.5, 6: 1.2}.get(day.dayofweek, 1.0)
        season = 1.0 + 0.15 * np.sin(2 * np.pi * (day.dayofyear - 200) / 365)
        n_orders = rng.poisson(base_daily * dow_boost * season)
        for _ in range(n_orders):
            hour = rng.choice([11, 12, 13, 17, 18, 19, 20, 21],
                              p=[.08, .14, .10, .12, .18, .18, .13, .07])
            minute = rng.integers(0, 60)
            subtotal = round(float(rng.gamma(4, 8) + 12), 2)
            rows.append({
                "timestamp": day.replace(hour=int(hour), minute=int(minute)),
                "subtotal": subtotal,
            })
    return pd.DataFrame(rows)


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    ue = simulate_orders("ubereats", base_daily=22)
    ue_out = pd.DataFrame({
        "Order ID": [f"UE-{i:06d}" for i in range(len(ue))],
        "Time Customer Ordered": ue["timestamp"].dt.strftime("%m/%d/%Y %I:%M %p"),
        "Order Status": "Completed",
        "Food Sales": ue["subtotal"],
    })
    ue_out.to_csv(SAMPLE_DIR / "ubereats_orders_SAMPLE.csv", index=False)

    dd = simulate_orders("doordash", base_daily=17)
    dd_out = pd.DataFrame({
        "Order ID": [f"DD-{i:06d}" for i in range(len(dd))],
        "Timestamp Local Date": dd["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Order Status": "Delivered",
        "Subtotal": dd["subtotal"],
    })
    dd_out.to_csv(SAMPLE_DIR / "doordash_orders_SAMPLE.csv", index=False)

    print(f"Wrote {len(ue_out)} Uber Eats and {len(dd_out)} DoorDash synthetic orders to {SAMPLE_DIR}")


if __name__ == "__main__":
    main()

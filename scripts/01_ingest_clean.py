"""Phase 1: read Uber Eats / DoorDash exports, clean, and build a daily sales table.

Usage:
    python 01_ingest_clean.py            # reads real exports from data/raw/
    python 01_ingest_clean.py --sample   # reads synthetic data from data/sample/

Drop every CSV export into data/raw/. Platform is inferred from the filename
("uber" or "doordash" anywhere in the name). Column names are auto-detected,
so exports from different report types usually work without edits.

Outputs:
    data/processed/orders_clean.csv   one row per order
    data/processed/daily_sales.csv    one row per day per platform + combined
"""

import argparse
import sys

import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, SAMPLE_DIR

DATE_CANDIDATES = [
    "time customer ordered", "timestamp local date", "order date", "local order date",
    "date ordered", "order placed at", "timestamp", "date", "created at", "order time",
]
MONEY_CANDIDATES = [
    "food sales", "subtotal", "order subtotal", "sales (incl. tax)", "sales excl. tax",
    "order total", "total", "gross sales", "net sales",
]
STATUS_CANDIDATES = ["order status", "status", "final order status"]
CANCEL_WORDS = ("cancel", "refund", "fail", "unfulfilled", "rejected")


def detect(columns, candidates, kind, fname):
    lowered = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    for low, orig in lowered.items():
        if any(cand.split()[0] in low for cand in candidates):
            return orig
    raise SystemExit(
        f"Could not find a {kind} column in {fname}. Columns were: {list(columns)}\n"
        f"Rename the right column to '{candidates[0]}' and rerun."
    )


def load_export(path) -> pd.DataFrame:
    name = path.name.lower()
    if "uber" in name:
        platform = "ubereats"
    elif "doordash" in name or "dd_" in name:
        platform = "doordash"
    else:
        print(f"  ! Skipping {path.name}: filename must contain 'uber' or 'doordash'")
        return pd.DataFrame()

    df = pd.read_csv(path)
    date_col = detect(df.columns, DATE_CANDIDATES, "order date/time", path.name)
    money_col = detect(df.columns, MONEY_CANDIDATES, "sales amount", path.name)

    out = pd.DataFrame({
        "ordered_at": pd.to_datetime(df[date_col], errors="coerce"),
        "sales": pd.to_numeric(
            df[money_col].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        ),
        "platform": platform,
    })

    status_col = next(
        (c for c in df.columns if c.lower().strip() in STATUS_CANDIDATES), None)
    if status_col is not None:
        cancelled = df[status_col].astype(str).str.lower().str.contains(
            "|".join(CANCEL_WORDS), na=False)
        out = out[~cancelled.values]
        print(f"  - {path.name}: dropped {int(cancelled.sum())} cancelled/refunded orders")

    before = len(out)
    out = out.dropna(subset=["ordered_at", "sales"])
    out = out[out["sales"] > 0]
    print(f"  - {path.name}: {len(out)} valid orders ({before - len(out)} rows dropped)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true",
                        help="use synthetic sample data instead of data/raw")
    args = parser.parse_args()

    src = SAMPLE_DIR if args.sample else RAW_DIR
    if args.sample:
        print("=" * 60)
        print("WARNING: running on SYNTHETIC sample data, not real exports.")
        print("=" * 60)

    files = sorted(src.glob("*.csv"))
    if not files:
        sys.exit(
            f"No CSV files found in {src}.\n"
            "Export your order history from Uber Eats Manager and the DoorDash "
            "Merchant Portal and save the CSVs there (filenames must contain "
            "'uber' or 'doordash')."
        )

    print(f"Reading {len(files)} file(s) from {src}:")
    orders = pd.concat([load_export(f) for f in files], ignore_index=True)
    if orders.empty:
        sys.exit("No usable orders found.")

    orders = orders.sort_values("ordered_at").drop_duplicates()
    orders["date"] = orders["ordered_at"].dt.date
    orders["hour"] = orders["ordered_at"].dt.hour

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    orders.to_csv(PROCESSED_DIR / "orders_clean.csv", index=False)

    daily = (orders.groupby(["date", "platform"])
             .agg(orders=("sales", "size"), revenue=("sales", "sum"))
             .reset_index())
    combined = (orders.groupby("date")
                .agg(orders=("sales", "size"), revenue=("sales", "sum"))
                .reset_index())
    combined["platform"] = "all"
    daily = pd.concat([daily, combined], ignore_index=True)
    daily["revenue"] = daily["revenue"].round(2)

    # Fill missing calendar days with zeros so the time series is continuous.
    full_days = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D").date
    frames = []
    for plat, grp in daily.groupby("platform"):
        grp = grp.set_index("date").reindex(full_days)
        grp["platform"] = plat
        grp[["orders", "revenue"]] = grp[["orders", "revenue"]].fillna(0)
        frames.append(grp.rename_axis("date").reset_index())
    daily = pd.concat(frames, ignore_index=True)

    daily.to_csv(PROCESSED_DIR / "daily_sales.csv", index=False)
    span = f"{combined['date'].min()} to {combined['date'].max()}"
    print(f"\nSaved {len(orders)} orders across {combined.shape[0]} days ({span})")
    print(f"-> {PROCESSED_DIR / 'orders_clean.csv'}")
    print(f"-> {PROCESSED_DIR / 'daily_sales.csv'}")


if __name__ == "__main__":
    main()

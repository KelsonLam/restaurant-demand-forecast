"""Phase 2: exploratory analysis of the daily sales series.

Usage:
    python 03_eda.py

Reads data/processed/daily_features.csv and writes charts + a summary to outputs/.
"""

import matplotlib.pyplot as plt
import pandas as pd

import viz_style
from config import OUTPUT_DIR, PROCESSED_DIR

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]


def main() -> None:
    viz_style.apply()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_DIR / "daily_features.csv", parse_dates=["date"])

    # 1. Daily orders over time with a 7-day rolling mean
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["date"], df["orders"], color=viz_style.BLUE, linewidth=0.9,
            alpha=0.45, label="Daily orders")
    ax.plot(df["date"], df["orders"].rolling(7).mean(), color=viz_style.BLUE,
            linewidth=2.2, label="7-day average")
    ax.set_title("Delivery orders per day")
    ax.set_ylabel("Orders")
    ax.legend(frameon=False, loc="upper left")
    viz_style.format_dates(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_orders_over_time.png")

    # 2. Day-of-week pattern
    dow = (df.groupby("day_name")["orders"].mean()
           .reindex(DAY_ORDER))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dow.index, dow.values, color=viz_style.BLUE, width=0.62)
    peak = dow.idxmax()
    ax.bar([peak], [dow[peak]], color=viz_style.GREEN, width=0.62)
    ax.set_title("Average orders by day of week")
    ax.set_ylabel("Avg orders")
    ax.annotate(f"Peak: {peak}", xy=(peak, dow[peak]),
                xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=9, color=viz_style.INK)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_day_of_week.png")

    # 3. Hourly demand heatmap (needs order-level data)
    orders = pd.read_csv(PROCESSED_DIR / "orders_clean.csv",
                         parse_dates=["ordered_at"])
    orders["day_name"] = orders["ordered_at"].dt.day_name()
    hourly = (orders.groupby(["day_name", "hour"]).size().unstack(fill_value=0)
              .reindex(DAY_ORDER))
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(hourly.values, aspect="auto",
                   cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
                       "seq_blue", ["#fcfcfb"] + viz_style.SEQ_BLUE))
    ax.set_xticks(range(len(hourly.columns)), hourly.columns)
    ax.set_yticks(range(7), DAY_ORDER)
    ax.set_title("Orders by hour and day of week")
    ax.set_xlabel("Hour of day")
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, label="Total orders", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_hourly_heatmap.png")

    # 4. Weather effect, if weather columns exist
    if "precip_in" in df.columns and df["precip_in"].notna().any():
        rainy = df["precip_in"] > 0.1
        comp = pd.Series({
            "Dry days": df.loc[~rainy, "orders"].mean(),
            "Rainy days (>0.1 in)": df.loc[rainy, "orders"].mean(),
        })
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(comp.index, comp.values, color=viz_style.BLUE, width=0.5)
        ax.set_title("Average orders: dry vs rainy days")
        ax.set_ylabel("Avg orders")
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "04_weather_effect.png")

    # Text summary for the writeup
    lines = [
        f"Days covered: {df['date'].min().date()} to {df['date'].max().date()} ({len(df)} days)",
        f"Total orders: {int(df['orders'].sum()):,}   Total revenue: ${df['revenue'].sum():,.0f}",
        f"Average orders/day: {df['orders'].mean():.1f}   Average revenue/day: ${df['revenue'].mean():,.0f}",
        f"Busiest weekday on average: {dow.idxmax()} ({dow.max():.1f} orders)",
        f"Slowest weekday on average: {dow.idxmin()} ({dow.min():.1f} orders)",
        f"Holiday-day average: {df.loc[df['is_holiday'] == 1, 'orders'].mean():.1f} "
        f"vs non-holiday {df.loc[df['is_holiday'] == 0, 'orders'].mean():.1f}",
    ]
    if "precip_in" in df.columns and df["precip_in"].notna().any():
        lines.append(f"Rainy-day average: {df.loc[rainy, 'orders'].mean():.1f} "
                     f"vs dry {df.loc[~rainy, 'orders'].mean():.1f}")
    summary = "\n".join(lines)
    (OUTPUT_DIR / "eda_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nCharts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


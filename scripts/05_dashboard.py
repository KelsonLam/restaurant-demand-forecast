"""Phase 4: generate the interactive HTML dashboard from processed data.

Usage:
    python 05_dashboard.py

Reads data/processed + outputs CSVs and writes outputs/dashboard.html --
a single self-contained file, no external libraries, light + dark themes.
"""

import json

import pandas as pd

from config import OUTPUT_DIR, PROCESSED_DIR

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]


def build_payload() -> dict:
    df = pd.read_csv(PROCESSED_DIR / "daily_features.csv", parse_dates=["date"])
    orders = pd.read_csv(PROCESSED_DIR / "orders_clean.csv",
                         parse_dates=["ordered_at"])
    forecast = pd.read_csv(OUTPUT_DIR / "staffing_recommendations.csv",
                           parse_dates=["date"])
    scores = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")

    df = df.sort_values("date").reset_index(drop=True)
    df["roll7"] = df["orders"].rolling(7).mean().round(1)

    daily = [{"d": d.strftime("%Y-%m-%d"), "o": int(o),
              "a": None if pd.isna(a) else float(a)}
             for d, o, a in zip(df["date"], df["orders"], df["roll7"])]

    dow = (df.groupby("day_name")["orders"].mean().reindex(DAY_ORDER)
           .round(1))

    orders["day_name"] = orders["ordered_at"].dt.day_name()
    hours = list(range(11, 22))
    heat = (orders[orders["hour"].isin(hours)]
            .groupby(["day_name", "hour"]).size().unstack(fill_value=0)
            .reindex(DAY_ORDER).reindex(columns=hours, fill_value=0))

    recent = df.tail(60)
    fc = [{"d": d.strftime("%Y-%m-%d"), "f": float(f), "day": day, "tier": t}
          for d, f, day, t in zip(forecast["date"], forecast["forecast_orders"],
                                  forecast["day_name"], forecast["staffing"])]

    rainy = df["precip_in"] > 0.1 if "precip_in" in df else pd.Series(False, index=df.index)
    return {
        "range": f"{df['date'].min():%b %d, %Y} – {df['date'].max():%b %d, %Y}",
        "stats": {
            "avgOrders": round(df["orders"].mean(), 1),
            "avgRevenue": round(df["revenue"].mean()),
            "totalOrders": int(df["orders"].sum()),
            "busiestDay": dow.idxmax(),
            "busiestAvg": float(dow.max()),
            "rainLift": round((df.loc[rainy, "orders"].mean()
                               / df.loc[~rainy, "orders"].mean() - 1) * 100, 1),
        },
        "daily": daily,
        "dow": [{"day": d, "avg": float(v)} for d, v in dow.items()],
        "heat": {"days": DAY_ORDER, "hours": hours,
                 "values": heat.values.tolist()},
        "recent": [{"d": d.strftime("%Y-%m-%d"), "o": int(o)}
                   for d, o in zip(recent["date"], recent["orders"])],
        "forecast": fc,
        "models": scores.to_dict(orient="records"),
        "bestModel": scores.iloc[0]["model"],
    }


TEMPLATE = r"""<meta charset="utf-8">
<title>Asian Eatery — Demand Forecast</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --baseline: #c3c2b7; --ring: rgba(11,11,11,.10);
  --accent: #2a78d6; --accent-soft: rgba(42,120,214,.10);
  --green: #008300; --green-soft: rgba(0,131,0,.10);
  --seq0:#eef4fd; --seq1:#cde2fb; --seq2:#9ec5f4; --seq3:#6da7ec;
  --seq4:#3987e5; --seq5:#256abf; --seq6:#184f95; --seq7:#0d366b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --ring: rgba(255,255,255,.10);
    --accent: #3987e5; --accent-soft: rgba(57,135,229,.16);
    --green: #0ca30c; --green-soft: rgba(12,163,12,.16);
    --seq0:#20293a; --seq1:#0d366b; --seq2:#184f95; --seq3:#256abf;
    --seq4:#3987e5; --seq5:#6da7ec; --seq6:#9ec5f4; --seq7:#cde2fb;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19;
  --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --baseline: #383835; --ring: rgba(255,255,255,.10);
  --accent: #3987e5; --accent-soft: rgba(57,135,229,.16);
  --green: #0ca30c; --green-soft: rgba(12,163,12,.16);
  --seq0:#20293a; --seq1:#0d366b; --seq2:#184f95; --seq3:#256abf;
  --seq4:#3987e5; --seq5:#6da7ec; --seq6:#9ec5f4; --seq7:#cde2fb;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1060px; margin: 0 auto; padding: 32px 20px 56px; }
header h1 { margin: 0 0 2px; font-size: 24px; letter-spacing: -.01em; }
.sub { color: var(--ink-2); margin: 0; }
.eyebrow {
  font-size: 11px; font-weight: 600; letter-spacing: .09em;
  text-transform: uppercase; color: var(--muted); margin: 0 0 10px;
}
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px,1fr));
  gap: 14px; margin: 26px 0; }
.tile { background: var(--surface); border: 1px solid var(--ring);
  border-radius: 10px; padding: 14px 16px; }
.tile .v { font-size: 26px; font-weight: 700; font-variant-numeric: tabular-nums; }
.tile .l { font-size: 13px; color: var(--ink-2); }
.tile .n { font-size: 12px; color: var(--muted); }
.grid2 { display: grid; grid-template-columns: 1fr 1.25fr; gap: 14px; }
@media (max-width: 820px) { .grid2 { grid-template-columns: 1fr; } }
.card { background: var(--surface); border: 1px solid var(--ring);
  border-radius: 10px; padding: 18px 18px 12px; margin: 0 0 14px; }
.card h2 { margin: 0 0 2px; font-size: 15px; }
.card .hint { font-size: 12.5px; color: var(--muted); margin: 0 0 8px; }
svg { display: block; width: 100%; height: auto; }
svg text { font: 11px system-ui, "Segoe UI", sans-serif; fill: var(--muted); }
.legend { display: flex; gap: 16px; font-size: 12.5px; color: var(--ink-2);
  margin: 4px 0 2px; flex-wrap: wrap; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
.tt { position: fixed; pointer-events: none; z-index: 9; background: var(--surface);
  border: 1px solid var(--ring); border-radius: 8px; padding: 7px 10px;
  font-size: 12.5px; box-shadow: 0 4px 14px rgba(0,0,0,.14); display: none;
  color: var(--ink-2); }
.tt b { color: var(--ink); font-variant-numeric: tabular-nums; }
.tblwrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th { text-align: left; color: var(--muted); font-weight: 600; font-size: 12px;
  letter-spacing: .05em; text-transform: uppercase; padding: 7px 10px;
  border-bottom: 1px solid var(--grid); }
td { padding: 7px 10px; border-bottom: 1px solid var(--grid);
  font-variant-numeric: tabular-nums; }
tr:last-child td { border-bottom: none; }
.chip { display: inline-block; padding: 2px 9px; border-radius: 99px;
  font-size: 12px; font-weight: 600; }
.chip.busy { background: var(--accent-soft); color: var(--accent); }
.chip.slow { background: var(--grid); color: var(--ink-2); }
.chip.normal { background: var(--green-soft); color: var(--green); }
footer { color: var(--muted); font-size: 12.5px; margin-top: 26px;
  border-top: 1px solid var(--grid); padding-top: 14px; }
</style>
<div class="wrap">
<header>
  <p class="eyebrow">Demand forecasting · Germantown, TN</p>
  <h1>Asian Eatery — Delivery Demand Dashboard</h1>
  <p class="sub">Uber Eats + DoorDash orders, <span id="range"></span> · simulated data calibrated from public restaurant sales</p>
</header>

<div class="tiles" id="tiles"></div>

<div class="card">
  <h2>Orders per day</h2>
  <p class="hint">Daily delivery orders with 7-day average — hover for values</p>
  <div class="legend">
    <span><i class="sw" style="background:var(--accent);opacity:.4"></i>Daily orders</span>
    <span><i class="sw" style="background:var(--accent)"></i>7-day average</span>
  </div>
  <div id="daily"></div>
</div>

<div class="grid2">
  <div class="card">
    <h2>Average orders by weekday</h2>
    <p class="hint">Peak day highlighted</p>
    <div id="dow"></div>
  </div>
  <div class="card">
    <h2>When orders arrive</h2>
    <p class="hint">Total orders by hour and weekday — darker is busier</p>
    <div id="heat"></div>
  </div>
</div>

<div class="card">
  <h2>Next 14 days — forecast</h2>
  <p class="hint">Best model: <b id="bestm"></b>, chosen on a 28-day holdout</p>
  <div class="legend">
    <span><i class="sw" style="background:var(--ink)"></i>Actual (last 60 days)</span>
    <span><i class="sw" style="background:var(--accent)"></i>Forecast</span>
  </div>
  <div id="fc"></div>
</div>

<div class="grid2">
  <div class="card">
    <h2>Model comparison</h2>
    <p class="hint">Error on the last 28 days (held out)</p>
    <div class="tblwrap"><table id="models"></table></div>
  </div>
  <div class="card">
    <h2>Staffing plan</h2>
    <p class="hint">Forecast day tiered against historical volume terciles</p>
    <div class="tblwrap"><table id="staff"></table></div>
  </div>
</div>

<footer>
  Data: <b>simulated</b> — demand shape calibrated from the public Maven Analytics
  “Pizza Place Sales” dataset (21k orders), rescaled to a delivery-only Asian
  restaurant profile and modulated by real Germantown, TN weather (Open-Meteo)
  and TN holidays. Models: seasonal naive, SARIMA, XGBoost. Built July 2026;
  pipeline accepts real Uber Eats / DoorDash exports unchanged.
</footer>
</div>
<div class="tt" id="tt"></div>
<script>
const DATA = __DATA__;
const tt = document.getElementById('tt');
const fmt = new Intl.NumberFormat('en-US');
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function showTT(html, x, y) {
  tt.innerHTML = html; tt.style.display = 'block';
  const w = tt.offsetWidth;
  tt.style.left = Math.min(x + 14, innerWidth - w - 8) + 'px';
  tt.style.top = (y + 14) + 'px';
}
function hideTT() { tt.style.display = 'none'; }
function niceTicks(max, n) {
  const step = Math.pow(10, Math.floor(Math.log10(max / n)));
  const mult = [1, 2, 5, 10].find(m => max / (m * step) <= n) || 10;
  const s = mult * step, out = [];
  for (let v = 0; v <= max; v += s) out.push(v);
  return out;
}
const MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
function dlabel(ds) { const d = new Date(ds + 'T12:00');
  return MON[d.getMonth()] + ' ' + d.getDate() + ', ' + d.getFullYear(); }

// ---- stat tiles
const S = DATA.stats;
document.getElementById('range').textContent = DATA.range;
document.getElementById('bestm').textContent = DATA.bestModel;
document.getElementById('tiles').innerHTML = [
  ['Avg orders / day', S.avgOrders, fmt.format(S.totalOrders) + ' orders total'],
  ['Avg revenue / day', '$' + fmt.format(S.avgRevenue), 'delivery channels only'],
  ['Busiest weekday', S.busiestDay, S.busiestAvg.toFixed(1) + ' orders on average'],
  ['Rainy-day lift', (S.rainLift >= 0 ? '+' : '') + S.rainLift + '%', 'orders vs dry days'],
].map(([l, v, n]) =>
  `<div class="tile"><div class="l">${l}</div><div class="v">${v}</div><div class="n">${n}</div></div>`
).join('');

// ---- generic line chart with hover
function lineChart(mount, W, H, seriesList, xDates, tipFn) {
  const P = {l: 34, r: 8, t: 8, b: 22};
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`}, document.getElementById(mount));
  const ymax = Math.max(...seriesList.flatMap(s => s.pts.filter(v => v != null))) * 1.08;
  const x = i => P.l + i / (xDates.length - 1) * (W - P.l - P.r);
  const y = v => H - P.b - v / ymax * (H - P.t - P.b);
  for (const v of niceTicks(ymax, 5)) {
    el('line', {x1: P.l, x2: W - P.r, y1: y(v), y2: y(v),
      stroke: 'var(--grid)', 'stroke-width': .7}, svg);
    el('text', {x: P.l - 6, y: y(v) + 3, 'text-anchor': 'end'}, svg)
      .textContent = v;
  }
  const monthSeen = {};
  xDates.forEach((ds, i) => {
    const d = new Date(ds + 'T12:00'), key = d.getFullYear() + '-' + d.getMonth();
    if (!monthSeen[key] && d.getDate() <= 7) { monthSeen[key] = 1;
      el('text', {x: x(i), y: H - 6, 'text-anchor': 'middle'}, svg)
        .textContent = MON[d.getMonth()] + (d.getMonth() === 0 ? ' ' + d.getFullYear() : '');
    }});
  for (const s of seriesList) {
    let dd = '', pen = false;
    s.pts.forEach((v, i) => {
      if (v == null) { pen = false; return; }
      dd += (pen ? 'L' : 'M') + x(i).toFixed(1) + ' ' + y(v).toFixed(1); pen = true;
    });
    el('path', {d: dd, fill: 'none', stroke: s.color,
      'stroke-width': s.w, opacity: s.op || 1,
      'stroke-dasharray': s.dash || 'none', 'stroke-linejoin': 'round'}, svg);
  }
  const cursor = el('line', {y1: P.t, y2: H - P.b, stroke: 'var(--baseline)',
    'stroke-width': 1, opacity: 0}, svg);
  const dots = seriesList.map(s => el('circle', {r: 3.5, fill: s.color,
    stroke: 'var(--surface)', 'stroke-width': 2, opacity: 0}, svg));
  svg.addEventListener('mousemove', ev => {
    const r = svg.getBoundingClientRect();
    const px = (ev.clientX - r.left) / r.width * W;
    const i = Math.max(0, Math.min(xDates.length - 1,
      Math.round((px - P.l) / (W - P.l - P.r) * (xDates.length - 1))));
    cursor.setAttribute('x1', x(i)); cursor.setAttribute('x2', x(i));
    cursor.setAttribute('opacity', 1);
    seriesList.forEach((s, k) => {
      const v = s.pts[i];
      dots[k].setAttribute('opacity', v == null ? 0 : 1);
      if (v != null) { dots[k].setAttribute('cx', x(i)); dots[k].setAttribute('cy', y(v)); }
    });
    showTT(tipFn(i), ev.clientX, ev.clientY);
  });
  svg.addEventListener('mouseleave', () => { hideTT();
    cursor.setAttribute('opacity', 0); dots.forEach(d => d.setAttribute('opacity', 0)); });
}

// ---- daily trend
lineChart('daily', 1000, 300, [
  {pts: DATA.daily.map(d => d.o), color: 'var(--accent)', w: 1, op: .38},
  {pts: DATA.daily.map(d => d.a), color: 'var(--accent)', w: 2.4},
], DATA.daily.map(d => d.d),
  i => `<b>${dlabel(DATA.daily[i].d)}</b><br>Orders: <b>${DATA.daily[i].o}</b>` +
       (DATA.daily[i].a != null ? `<br>7-day avg: <b>${DATA.daily[i].a}</b>` : ''));

// ---- weekday bars
(function () {
  const W = 440, H = 260, P = {l: 34, r: 8, t: 18, b: 24};
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`}, document.getElementById('dow'));
  const max = Math.max(...DATA.dow.map(d => d.avg)) * 1.12;
  const bw = (W - P.l - P.r) / 7;
  const y = v => H - P.b - v / max * (H - P.t - P.b);
  for (const v of niceTicks(max, 4)) {
    el('line', {x1: P.l, x2: W - P.r, y1: y(v), y2: y(v),
      stroke: 'var(--grid)', 'stroke-width': .7}, svg);
    el('text', {x: P.l - 6, y: y(v) + 3, 'text-anchor': 'end'}, svg).textContent = v;
  }
  const peak = DATA.dow.reduce((a, b) => b.avg > a.avg ? b : a);
  DATA.dow.forEach((d, i) => {
    const bx = P.l + i * bw + bw * .18, w = bw * .64;
    const col = d.day === peak.day ? 'var(--green)' : 'var(--accent)';
    const bar = el('path', {d:
      `M${bx} ${H - P.b} V${y(d.avg) + 4} q0 -4 4 -4 h${w - 8} q4 0 4 4 V${H - P.b} Z`,
      fill: col}, svg);
    el('text', {x: bx + w / 2, y: H - 8, 'text-anchor': 'middle'}, svg)
      .textContent = d.day.slice(0, 3);
    if (d.day === peak.day)
      el('text', {x: bx + w / 2, y: y(d.avg) - 6, 'text-anchor': 'middle',
        fill: 'var(--ink)', 'font-weight': 600}, svg).textContent = d.avg;
    bar.addEventListener('mousemove', ev =>
      showTT(`<b>${d.day}</b><br>Avg orders: <b>${d.avg}</b>`, ev.clientX, ev.clientY));
    bar.addEventListener('mouseleave', hideTT);
  });
})();

// ---- heatmap
(function () {
  const {days, hours, values} = DATA.heat;
  const W = 560, H = 260, P = {l: 40, r: 8, t: 8, b: 34};
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`}, document.getElementById('heat'));
  const cw = (W - P.l - P.r) / hours.length, ch = (H - P.t - P.b) / 7;
  const max = Math.max(...values.flat());
  days.forEach((day, r) => {
    el('text', {x: P.l - 6, y: P.t + r * ch + ch / 2 + 3, 'text-anchor': 'end'},
      svg).textContent = day.slice(0, 3);
    hours.forEach((h, c) => {
      const v = values[r][c];
      const bin = v === 0 ? 0 : Math.min(7, 1 + Math.floor(v / max * 6.999));
      const rect = el('rect', {x: P.l + c * cw + 1, y: P.t + r * ch + 1,
        width: cw - 2, height: ch - 2, rx: 2.5,
        fill: `var(--seq${bin})`}, svg);
      rect.addEventListener('mousemove', ev => showTT(
        `<b>${day} ${h > 12 ? h - 12 : h}${h >= 12 ? 'pm' : 'am'}</b><br>Orders: <b>${v}</b>`,
        ev.clientX, ev.clientY));
      rect.addEventListener('mouseleave', hideTT);
    });
  });
  hours.forEach((h, c) => { if (c % 2 === 0)
    el('text', {x: P.l + c * cw + cw / 2, y: H - 18, 'text-anchor': 'middle'}, svg)
      .textContent = h > 12 ? (h - 12) + 'p' : h + (h === 12 ? 'p' : 'a'); });
  el('text', {x: P.l, y: H - 2}, svg).textContent = 'fewer';
  el('text', {x: W - P.r, y: H - 2, 'text-anchor': 'end'}, svg).textContent = 'more';
  for (let b = 0; b < 8; b++)
    el('rect', {x: P.l + 40 + b * 14, y: H - 10, width: 12, height: 7, rx: 2,
      fill: `var(--seq${b})`}, svg);
})();

// ---- forecast chart
(function () {
  const xs = DATA.recent.map(d => d.d).concat(DATA.forecast.map(d => d.d));
  const actual = DATA.recent.map(d => d.o).concat(DATA.forecast.map(() => null));
  const n = DATA.recent.length;
  const fc = xs.map((_, i) => i >= n - 1
    ? (i === n - 1 ? DATA.recent[n - 1].o : DATA.forecast[i - n].f) : null);
  lineChart('fc', 1000, 300, [
    {pts: actual, color: 'var(--ink)', w: 1.8},
    {pts: fc, color: 'var(--accent)', w: 2.4, dash: '6 5'},
  ], xs, i => {
    if (i < n) return `<b>${dlabel(xs[i])}</b><br>Orders: <b>${actual[i]}</b>`;
    const f = DATA.forecast[i - n];
    return `<b>${dlabel(f.d)}</b><br>Forecast: <b>${f.f}</b><br>${f.tier}`;
  });
})();

// ---- tables
document.getElementById('models').innerHTML =
  '<tr><th>Model</th><th>MAE</th><th>MAPE</th></tr>' +
  DATA.models.map((m, i) =>
    `<tr><td>${m.model}${i === 0 ? ' ★' : ''}</td><td>${m.MAE}</td><td>${m['MAPE_%']}%</td></tr>`
  ).join('');

const chipClass = t => t.startsWith('Busy') ? 'busy'
  : t.startsWith('Slow') ? 'slow' : 'normal';
document.getElementById('staff').innerHTML =
  '<tr><th>Date</th><th>Day</th><th>Forecast</th><th>Staffing</th></tr>' +
  DATA.forecast.map(f =>
    `<tr><td>${dlabel(f.d)}</td><td>${f.day}</td><td>${f.f}</td>` +
    `<td><span class="chip ${chipClass(f.tier)}">${f.tier}</span></td></tr>`
  ).join('');
</script>
"""


def main() -> None:
    payload = build_payload()
    html = TEMPLATE.replace("__DATA__", json.dumps(payload))
    out = OUTPUT_DIR / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

"""Phase 4: generate the interactive HTML dashboard from processed data.

Usage:
    python 05_dashboard.py

Reads data/processed + outputs CSVs and writes outputs/dashboard.html --
a single self-contained "demand ledger" page (fonts embedded as data URIs,
no external requests). Deliberately single-theme: a candlelit steakhouse
look, per the design brief.
"""

import base64
import json

import pandas as pd

from config import OUTPUT_DIR, PROCESSED_DIR, PROJECT_ROOT

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
HOURS = list(range(11, 22))
CHANNELS = ["all", "dine-in", "takeout", "ubereats", "doordash"]
LABELS = {"all": "All house", "dine-in": "Dine-in", "takeout": "Takeout",
          "ubereats": "Uber Eats", "doordash": "DoorDash"}


def channel_block(daily, orders, precip):
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["a"] = daily["orders"].rolling(7).mean().round(1)
    rows = [{"d": d.strftime("%Y-%m-%d"), "o": int(o), "r": round(r),
             "a": None if pd.isna(a) else float(a)}
            for d, o, r, a in zip(daily["date"], daily["orders"],
                                  daily["revenue"], daily["a"])]

    dow = (daily.assign(day=daily["date"].dt.day_name())
           .groupby("day")["orders"].mean().reindex(DAY_ORDER).round(1))

    heat = (orders[orders["hour"].isin(HOURS)]
            .groupby(["day_name", "hour"]).size()
            .unstack(fill_value=0)
            .reindex(DAY_ORDER).reindex(columns=HOURS, fill_value=0))

    hour_totals = heat.sum(axis=0)
    peak_hour = int(hour_totals.idxmax())
    ph = f"{peak_hour - 12 if peak_hour > 12 else peak_hour}–" \
         f"{peak_hour - 11 if peak_hour >= 12 else peak_hour + 1} pm"

    merged = daily.merge(precip, on="date", how="left")
    rainy = merged["precip_in"] > 0.1
    rain_lift = (merged.loc[rainy, "orders"].mean()
                 / merged.loc[~rainy, "orders"].mean() - 1) * 100

    return {
        "daily": rows,
        "dow": [float(v) for v in dow.values],
        "heat": heat.values.tolist(),
        "stats": {
            "avgOrders": round(daily["orders"].mean(), 1),
            "avgRev": round(daily["revenue"].mean()),
            "avgTicket": round(daily["revenue"].sum()
                               / max(daily["orders"].sum(), 1), 2),
            "totalRev": round(daily["revenue"].sum()),
            "peakDay": dow.idxmax(),
            "peakHour": ph,
            "rainLift": round(rain_lift, 1),
        },
    }


def build_payload():
    sales = pd.read_csv(PROCESSED_DIR / "daily_sales.csv", parse_dates=["date"])
    orders = pd.read_csv(PROCESSED_DIR / "orders_clean.csv",
                         parse_dates=["ordered_at"])
    orders["day_name"] = orders["ordered_at"].dt.day_name()
    feats = pd.read_csv(PROCESSED_DIR / "daily_features.csv",
                        parse_dates=["date"])
    precip = feats[["date", "precip_in"]]
    forecast = pd.read_csv(OUTPUT_DIR / "staffing_recommendations.csv",
                           parse_dates=["date"])
    scores = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")

    channels = {}
    for ch in CHANNELS:
        d = sales[sales["platform"] == ch]
        o = orders if ch == "all" else orders[orders["platform"] == ch]
        channels[ch] = channel_block(d.copy(), o, precip)

    total_rev = channels["all"]["stats"]["totalRev"]
    for ch in CHANNELS:
        channels[ch]["stats"]["share"] = round(
            channels[ch]["stats"]["totalRev"] / total_rev * 100, 1)

    tiers = {"Busy - add staff": "BUSY", "Normal": "STEADY",
             "Slow - minimum staff": "LIGHT"}
    fc = [{"d": d.strftime("%Y-%m-%d"), "f": round(float(f)),
           "day": day, "tier": tiers.get(t, "STEADY")}
          for d, f, day, t in zip(forecast["date"],
                                  forecast["forecast_orders"],
                                  forecast["day_name"], forecast["staffing"])]

    span = f"{feats['date'].min():%B %Y} – {feats['date'].max():%B %Y}"
    return {"range": span, "hours": HOURS, "order": CHANNELS,
            "labels": LABELS, "channels": channels, "forecast": fc,
            "models": scores.to_dict(orient="records"),
            "bestModel": scores.iloc[0]["model"]}


def font_b64(name):
    p = PROJECT_ROOT / "assets" / "fonts" / name
    return base64.b64encode(p.read_bytes()).decode()


TEMPLATE = r"""<meta charset="utf-8">
<title>Knoxville Steakhouse — Demand Ledger</title>
<style>
@font-face {
  font-family: 'Fraunces'; font-weight: 600; font-display: swap;
  src: url(data:font/woff2;base64,__F_FRAUNCES__) format('woff2');
}
@font-face {
  font-family: 'Archivo'; font-weight: 100 900; font-display: swap;
  src: url(data:font/woff2;base64,__F_ARCHIVO__) format('woff2');
}
:root {
  color-scheme: dark;
  --espresso: #201A16; --walnut: #2A231E; --walnut-2: #322A23;
  --cream: #F3EAD9; --cream-2: #CBBFA9; --taupe: #9C8E7B;
  --brass: #C9A24B; --brass-pale: #E9C77E; --copper: #C96F4A;
  --wine: #B25B57;
  --rule: rgba(243,234,217,.16); --rule-soft: rgba(243,234,217,.08);
  --grid: rgba(243,234,217,.06);
}
* { box-sizing: border-box; }
html { background: var(--espresso); }
body {
  margin: 0; background: var(--espresso); color: var(--cream);
  font: 400 15px/1.5 'Archivo', system-ui, sans-serif;
}
.page { max-width: 1040px; margin: 0 auto; padding: 44px 22px 64px; }
.display { font-family: 'Fraunces', Georgia, serif; font-weight: 600; }
.num { font-variant-numeric: tabular-nums; }

/* masthead: set like a menu cover */
.masthead { text-align: center; padding-bottom: 22px; }
.masthead .over { font-size: 11px; letter-spacing: .28em; color: var(--brass);
  text-transform: uppercase; margin: 0 0 10px; }
.masthead h1 { font-size: clamp(28px, 5vw, 38px); margin: 0 0 8px;
  letter-spacing: .01em; }
.masthead .sub { color: var(--taupe); font-size: 13.5px; margin: 0; }
.dbl { border: none; border-top: 1px solid var(--rule); position: relative;
  margin: 26px 0; }
.dbl::after { content: ""; position: absolute; left: 0; right: 0; top: 3px;
  border-top: 1px solid var(--rule-soft); }

/* controls */
.controls { display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; }
.seg { display: inline-flex; border: 1px solid var(--rule); border-radius: 3px;
  overflow: hidden; }
.seg button { background: none; border: none; color: var(--cream-2);
  font: 500 12.5px 'Archivo', sans-serif; padding: 7px 13px; cursor: pointer;
  display: inline-flex; align-items: center; gap: 7px;
  border-left: 1px solid var(--rule-soft); }
.seg button:first-child { border-left: none; }
.seg button:hover { background: var(--walnut); }
.seg button[aria-pressed="true"] { background: var(--walnut-2);
  color: var(--cream); }
.seg .dot { width: 8px; height: 8px; border-radius: 50%; }
.seg button:focus-visible, .chips button:focus-visible {
  outline: 2px solid var(--brass); outline-offset: -2px; }

/* section heads */
.sect { display: flex; align-items: baseline; gap: 14px; margin: 6px 0 4px; }
.sect h2 { font-size: 20px; margin: 0; }
.sect .note { color: var(--taupe); font-size: 12.5px; }
.sect .spacer { flex: 1; }

/* ledger stats: menu dot-leader lines */
.ledger { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 56px;
  margin: 18px 0 6px; }
@media (max-width: 760px) { .ledger { grid-template-columns: 1fr; } }
.ledger h3 { grid-row: 1; font-size: 11px; letter-spacing: .22em;
  text-transform: uppercase; color: var(--brass); margin: 0 0 2px;
  font-weight: 600; }
.lrow { display: flex; align-items: baseline; opacity: 0; }
.lrow .k { color: var(--cream-2); font-size: 14px; }
.lrow .dots { flex: 1; border-bottom: 2px dotted rgba(243,234,217,.22);
  margin: 0 10px 5px; min-width: 24px; }
.lrow .v { font-size: 17px; font-weight: 600; }
.lrow .v small { color: var(--taupe); font-weight: 400; font-size: 12px; }

/* charts */
svg { display: block; width: 100%; height: auto; }
svg text { font: 11px 'Archivo', sans-serif; fill: var(--taupe); }
.chartbox { margin: 10px 0 0; }
.chips { display: flex; gap: 6px; }
.chips button { background: none; border: 1px solid var(--rule);
  border-radius: 3px; color: var(--cream-2); font: 500 12px 'Archivo';
  padding: 4px 10px; cursor: pointer; }
.chips button[aria-pressed="true"] { background: var(--walnut-2);
  color: var(--cream); border-color: var(--rule); }
.legend { display: flex; gap: 16px; font-size: 12.5px; color: var(--cream-2);
  margin: 6px 0 0; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
.rhythm { display: grid; grid-template-columns: 1fr 1.3fr; gap: 34px;
  margin-top: 8px; }
@media (max-width: 820px) { .rhythm { grid-template-columns: 1fr; } }

/* tooltip */
.tt { position: fixed; pointer-events: none; z-index: 9;
  background: var(--walnut-2); border: 1px solid var(--rule);
  border-radius: 3px; padding: 7px 10px; font-size: 12.5px; display: none;
  color: var(--cream-2); box-shadow: 0 6px 18px rgba(0,0,0,.4); }
.tt b { color: var(--cream); font-variant-numeric: tabular-nums; }

/* ticket rail */
.rail { display: flex; gap: 14px; overflow-x: auto; padding: 26px 4px 18px;
  scroll-snap-type: x proximity; }
.rail::-webkit-scrollbar { height: 8px; }
.rail::-webkit-scrollbar-thumb { background: var(--walnut-2);
  border-radius: 4px; }
.ticket { flex: 0 0 128px; scroll-snap-align: start; background: var(--cream);
  color: #2A2019; border-radius: 2px; padding: 12px 12px 14px;
  position: relative; transform: rotate(var(--rot));
  box-shadow: 0 8px 16px rgba(0,0,0,.35); opacity: 0; }
.ticket::before { content: ""; position: absolute; left: 0; right: 0; top: 0;
  border-top: 2px dashed rgba(42,32,25,.35); }
.ticket::after { content: ""; position: absolute; top: -7px; left: 50%;
  width: 34px; height: 9px; margin-left: -17px; background: var(--walnut-2);
  border-radius: 2px; box-shadow: 0 1px 2px rgba(0,0,0,.5); }
.ticket .t-day { font-size: 11px; letter-spacing: .18em; font-weight: 600;
  color: #7A6A55; margin-top: 4px; }
.ticket .t-date { font-size: 13.5px; font-weight: 600; margin: 1px 0 8px; }
.ticket .t-num { font-size: 34px; font-weight: 700; line-height: 1;
  font-variant-numeric: tabular-nums; }
.ticket .t-unit { font-size: 10.5px; color: #7A6A55; margin: 2px 0 10px; }
.stamp { display: inline-block; font-size: 11px; font-weight: 700;
  letter-spacing: .14em; padding: 2px 7px; border: 2px solid;
  border-radius: 2px; transform: rotate(-4deg); }
.stamp.BUSY { color: #9E3B36; border-color: #9E3B36; }
.stamp.STEADY { color: #7A6A55; border-color: #7A6A55; }
.stamp.LIGHT { color: #A5988A; border-color: #A5988A;
  border-style: dashed; }

/* models table */
.models { margin-top: 10px; font-size: 13px; color: var(--cream-2); }
.models b { color: var(--cream); }

footer { color: var(--taupe); font-size: 12.5px; margin-top: 8px;
  line-height: 1.6; }

/* motion */
@keyframes rise { from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: none; } }
@keyframes print { from { opacity: 0; transform: translateY(-16px)
  rotate(var(--rot)); } to { opacity: 1; transform: rotate(var(--rot)); } }
.lrow { animation: rise .5s calc(var(--d) * 70ms) both; }
.ticket { animation: print .45s calc(var(--d) * 70ms + .2s) both; }
.fade { animation: rise .6s .1s both; }
@media (prefers-reduced-motion: reduce) {
  .lrow, .ticket, .fade { animation: none; opacity: 1; }
}
</style>
<div class="page">

<header class="masthead">
  <p class="over">Est. 2026 · Knoxville, Tennessee</p>
  <h1 class="display">The Demand Ledger</h1>
  <p class="sub">A steakhouse's full house — dine-in, takeout &amp; delivery ·
    <span id="range" class="num"></span> · simulated data calibrated from
    public restaurant sales</p>
</header>

<div class="controls" id="chctl"></div>

<hr class="dbl">

<section>
  <div class="sect">
    <h2 class="display">The house, by the numbers</h2>
    <span class="note" id="statnote"></span>
  </div>
  <div class="ledger" id="ledger"></div>
</section>

<hr class="dbl">

<section>
  <div class="sect">
    <h2 class="display">The ledger</h2>
    <span class="note">orders per day, with 7-day average</span>
    <span class="spacer"></span>
    <div class="chips" id="rangechips"></div>
  </div>
  <div class="legend">
    <span><i class="sw" style="background:rgba(243,234,217,.25)"></i>Daily</span>
    <span><i class="sw" id="avgsw" style="background:var(--brass)"></i>7-day average</span>
  </div>
  <div class="chartbox fade" id="trend"></div>
  <div class="chartbox" id="overview" title="Drag to zoom a date range"></div>
</section>

<hr class="dbl">

<section>
  <div class="sect">
    <h2 class="display">Service rhythm</h2>
    <span class="note">when the house fills</span>
    <span class="spacer"></span>
    <div class="chips" id="dayview"></div>
  </div>
  <div class="rhythm">
    <div class="fade" id="dow"></div>
    <div class="fade" id="heat"></div>
  </div>
</section>

<hr class="dbl">

<section>
  <div class="sect">
    <h2 class="display">Tomorrow's tickets</h2>
    <span class="note">14-day forecast for the whole house, tiered for staffing</span>
  </div>
  <div class="rail" id="rail"></div>
  <p class="models" id="models"></p>
</section>

<hr class="dbl">

<footer>
  Data: <b>simulated</b> — no steakhouse publishes its complete POS records, so
  demand shape is calibrated from the public Maven Analytics “Pizza Place Sales”
  dataset (21k real full-service orders) and rescaled to steakhouse norms:
  dinner-dominant service, Friday–Saturday peaks, occasion spikes on
  Valentine’s, Mother’s and Father’s Day, real Knoxville weather (rain lifts
  delivery, dents walk-ins) and Tennessee holidays. Forecasts: seasonal naive
  vs SARIMA vs XGBoost on a 28-day holdout. The pipeline accepts real POS,
  Uber Eats Manager and DoorDash exports unchanged.
</footer>
</div>
<div class="tt" id="tt"></div>
<script>
const DATA = __DATA__;
const CH_COLOR = {all: '#C9A24B', 'dine-in': '#E9C77E', takeout: '#9C8E7B',
                  ubereats: '#C96F4A', doordash: '#B25B57'};
const RAMP = ['#241E19', '#3A2F22', '#54422A', '#6E5530', '#8A6A38',
              '#A98440', '#C9A24B', '#E9C77E'];
const MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const DAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'];
const N = DATA.channels.all.daily.length;
const state = {ch: 'all', win: [0, N - 1], dayView: 'all'};
const fmt = new Intl.NumberFormat('en-US');
const tt = document.getElementById('tt');
const NS = 'http://www.w3.org/2000/svg';
let gradSeq = 0;

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function showTT(html, x, y) {
  tt.innerHTML = html; tt.style.display = 'block';
  tt.style.left = Math.min(x + 14, innerWidth - tt.offsetWidth - 8) + 'px';
  tt.style.top = (y + 14) + 'px';
}
function hideTT() { tt.style.display = 'none'; }
function dlabel(ds) { const d = new Date(ds + 'T12:00');
  return MON[d.getMonth()] + ' ' + d.getDate() + ', ' + d.getFullYear(); }
function niceTicks(max, n) {
  const step = Math.pow(10, Math.floor(Math.log10(max / n)));
  const mult = [1, 2, 5, 10].find(m => max / (m * step) <= n) || 10;
  const out = []; for (let v = 0; v <= max; v += mult * step) out.push(v);
  return out;
}
const cur = () => DATA.channels[state.ch];

// ---------- channel control
(function () {
  const seg = document.createElement('div'); seg.className = 'seg';
  seg.setAttribute('role', 'group'); seg.setAttribute('aria-label', 'Channel');
  for (const ch of DATA.order) {
    const b = document.createElement('button');
    b.innerHTML = `<span class="dot" style="background:${CH_COLOR[ch]}"></span>` +
      DATA.labels[ch];
    b.onclick = () => { state.ch = ch; renderAll(); };
    b.dataset.ch = ch;
    seg.appendChild(b);
  }
  document.getElementById('chctl').appendChild(seg);
})();

// ---------- range chips
const PRESETS = [['30 d', 30], ['90 d', 90], ['6 mo', 183], ['Full', N]];
(function () {
  const box = document.getElementById('rangechips');
  for (const [label, days] of PRESETS) {
    const b = document.createElement('button');
    b.textContent = label; b.dataset.days = days;
    b.onclick = () => { state.win = [Math.max(0, N - days), N - 1]; renderAll(); };
    box.appendChild(b);
  }
})();

// ---------- day-view chips
(function () {
  const box = document.getElementById('dayview');
  for (const [label, key] of [['Every day', 'all'], ['Weekdays', 'wd'],
                              ['Weekends', 'we']]) {
    const b = document.createElement('button');
    b.textContent = label; b.dataset.view = key;
    b.onclick = () => { state.dayView = key; renderRhythm(); syncPressed(); };
    box.appendChild(b);
  }
})();

function syncPressed() {
  document.querySelectorAll('#chctl button').forEach(b =>
    b.setAttribute('aria-pressed', b.dataset.ch === state.ch));
  const span = state.win[1] - state.win[0] + 1;
  document.querySelectorAll('#rangechips button').forEach(b =>
    b.setAttribute('aria-pressed',
      Math.abs(span - Math.min(+b.dataset.days, N)) < 2));
  document.querySelectorAll('#dayview button').forEach(b =>
    b.setAttribute('aria-pressed', b.dataset.view === state.dayView));
}

// ---------- ledger stats
function renderLedger() {
  const s = cur().stats;
  document.getElementById('statnote').textContent =
    DATA.labels[state.ch] + (state.ch === 'all' ? '' :
      ` · ${s.share}% of house revenue`);
  const rows = [
    ['HOW MUCH', [
      [state.ch === 'dine-in' ? 'Checks per day' : 'Orders per day',
       `<span class="num">${s.avgOrders}</span>`],
      ['Revenue per day', `<span class="num">$${fmt.format(s.avgRev)}</span>`],
      ['Average check', `<span class="num">$${fmt.format(Math.round(s.avgTicket))}</span>`],
      ['Share of house', `<span class="num">${s.share}%</span>`],
    ]],
    ['WHEN', [
      ['Peak day', s.peakDay],
      ['Peak hour', s.peakHour],
      ['Rainy days', `<span class="num">${s.rainLift >= 0 ? '+' : '−'}${Math.abs(s.rainLift)}%</span> <small>vs dry</small>`],
      ['Season covered', `<span class="num">${DATA.range}</span>`],
    ]],
  ];
  let html = '', d = 0;
  for (const [title, items] of rows) {
    html += `<div><h3>${title}</h3>` + items.map(([k, v]) =>
      `<div class="lrow" style="--d:${d++}"><span class="k">${k}</span>` +
      `<span class="dots"></span><span class="v">${v}</span></div>`
    ).join('') + '</div>';
  }
  document.getElementById('ledger').innerHTML = html;
}

// ---------- trend chart with crosshair
function renderTrend() {
  const box = document.getElementById('trend'); box.innerHTML = '';
  const daily = cur().daily.slice(state.win[0], state.win[1] + 1);
  const color = CH_COLOR[state.ch];
  document.getElementById('avgsw').style.background = color;
  const W = 1000, H = 280, P = {l: 40, r: 10, t: 10, b: 24};
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`}, box);
  const ymax = Math.max(...daily.map(d => d.o)) * 1.08 || 1;
  const x = i => P.l + i / Math.max(daily.length - 1, 1) * (W - P.l - P.r);
  const y = v => H - P.b - v / ymax * (H - P.t - P.b);
  for (const v of niceTicks(ymax, 5)) {
    el('line', {x1: P.l, x2: W - P.r, y1: y(v), y2: y(v),
      stroke: 'var(--grid)', 'stroke-width': 1}, svg);
    el('text', {x: P.l - 7, y: y(v) + 3, 'text-anchor': 'end',
      class: 'num'}, svg).textContent = v;
  }
  // x labels: months when wide, days when narrow
  const span = daily.length;
  if (span > 130) {
    const seen = {};
    daily.forEach((r, i) => {
      const d = new Date(r.d + 'T12:00'),
            k = d.getFullYear() + '-' + d.getMonth();
      if (!seen[k] && d.getDate() <= 7) { seen[k] = 1;
        el('text', {x: x(i), y: H - 6, 'text-anchor': 'middle'}, svg)
          .textContent = MON[d.getMonth()] +
            (d.getMonth() === 0 ? ' ' + d.getFullYear() : '');
      }});
  } else {
    const every = Math.ceil(span / 7);
    daily.forEach((r, i) => { if (i % every === 0) {
      const d = new Date(r.d + 'T12:00');
      el('text', {x: x(i), y: H - 6, 'text-anchor': 'middle'}, svg)
        .textContent = MON[d.getMonth()] + ' ' + d.getDate();
    }});
  }
  // raw daily line
  let dd = '';
  daily.forEach((r, i) => { dd += (i ? 'L' : 'M') + x(i).toFixed(1) + ' '
    + y(r.o).toFixed(1); });
  el('path', {d: dd, fill: 'none', stroke: 'rgba(243,234,217,.20)',
    'stroke-width': .9, 'stroke-linejoin': 'round'}, svg);
  // 7-day average + soft gradient area
  let ad = '', pen = false, fx = null, lx = null;
  daily.forEach((r, i) => {
    if (r.a == null) { pen = false; return; }
    ad += (pen ? 'L' : 'M') + x(i).toFixed(1) + ' ' + y(r.a).toFixed(1);
    pen = true; if (fx == null) fx = x(i); lx = x(i);
  });
  if (fx != null) {
    const gid = 'g' + (++gradSeq);
    const g = el('linearGradient', {id: gid, x1: 0, y1: 0, x2: 0, y2: 1}, svg);
    el('stop', {offset: '0%', style: `stop-color:${color};stop-opacity:.18`}, g);
    el('stop', {offset: '100%', style: `stop-color:${color};stop-opacity:0`}, g);
    el('path', {d: ad + `L${lx} ${H - P.b} L${fx} ${H - P.b} Z`,
      fill: `url(#${gid})`}, svg);
    el('path', {d: ad, fill: 'none', stroke: color, 'stroke-width': 2.4,
      'stroke-linejoin': 'round'}, svg);
  }
  // crosshair
  const cursor = el('line', {y1: P.t, y2: H - P.b,
    stroke: 'rgba(243,234,217,.3)', 'stroke-width': 1, opacity: 0}, svg);
  const dot = el('circle', {r: 3.5, fill: color, stroke: 'var(--espresso)',
    'stroke-width': 2, opacity: 0}, svg);
  svg.addEventListener('mousemove', ev => {
    const rc = svg.getBoundingClientRect();
    const px = (ev.clientX - rc.left) / rc.width * W;
    const i = Math.max(0, Math.min(daily.length - 1,
      Math.round((px - P.l) / (W - P.l - P.r) * (daily.length - 1))));
    const r = daily[i];
    cursor.setAttribute('x1', x(i)); cursor.setAttribute('x2', x(i));
    cursor.setAttribute('opacity', 1);
    dot.setAttribute('cx', x(i)); dot.setAttribute('cy', y(r.o));
    dot.setAttribute('opacity', 1);
    showTT(`<b>${dlabel(r.d)}</b><br>Orders: <b>${r.o}</b>` +
      `<br>Revenue: <b>$${fmt.format(r.r)}</b>` +
      (r.a != null ? `<br>7-day avg: <b>${r.a}</b>` : ''),
      ev.clientX, ev.clientY);
  });
  svg.addEventListener('mouseleave', () => { hideTT();
    cursor.setAttribute('opacity', 0); dot.setAttribute('opacity', 0); });
}

// ---------- overview brush (drag to zoom)
function renderOverview() {
  const box = document.getElementById('overview'); box.innerHTML = '';
  const daily = cur().daily;
  const W = 1000, H = 46, P = {l: 40, r: 10};
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`,
    style: 'cursor:crosshair'}, box);
  const ymax = Math.max(...daily.map(d => d.o)) || 1;
  const x = i => P.l + i / (N - 1) * (W - P.l - P.r);
  const xi = px => Math.max(0, Math.min(N - 1,
    Math.round((px - P.l) / (W - P.l - P.r) * (N - 1))));
  let ad = `M${P.l} ${H - 4}`;
  daily.forEach((r, i) => { ad += `L${x(i).toFixed(1)} `
    + (H - 4 - r.o / ymax * (H - 10)).toFixed(1); });
  ad += `L${W - P.r} ${H - 4} Z`;
  el('path', {d: ad, fill: 'rgba(243,234,217,.12)'}, svg);
  const win = el('rect', {y: 2, height: H - 6, fill: 'rgba(201,162,75,.14)',
    stroke: 'var(--brass)', 'stroke-width': 1, rx: 2}, svg);
  const setWin = () => { win.setAttribute('x', x(state.win[0]));
    win.setAttribute('width', Math.max(x(state.win[1]) - x(state.win[0]), 3)); };
  setWin();
  let anchor = null;
  const pxOf = ev => { const rc = svg.getBoundingClientRect();
    return (ev.clientX - rc.left) / rc.width * W; };
  svg.addEventListener('pointerdown', ev => {
    anchor = xi(pxOf(ev)); svg.setPointerCapture(ev.pointerId);
  });
  svg.addEventListener('pointermove', ev => {
    if (anchor == null) return;
    const b = xi(pxOf(ev));
    state.win = [Math.min(anchor, b), Math.max(anchor, b)]; setWin();
  });
  svg.addEventListener('pointerup', () => {
    if (anchor == null) return;
    if (state.win[1] - state.win[0] < 13)
      state.win = [Math.max(0, state.win[0] - 7),
                   Math.min(N - 1, state.win[0] + 7)];
    anchor = null; renderTrend(); syncPressed(); setWin();
  });
}

// ---------- weekday bars
function dayIdxs() {
  return state.dayView === 'wd' ? [0, 1, 2, 3, 4]
       : state.dayView === 'we' ? [5, 6] : [0, 1, 2, 3, 4, 5, 6];
}
function renderDow() {
  const box = document.getElementById('dow'); box.innerHTML = '';
  const dow = cur().dow, color = CH_COLOR[state.ch], active = dayIdxs();
  const W = 420, H = 240, P = {l: 36, r: 6, t: 20, b: 24};
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`}, box);
  const max = Math.max(...dow) * 1.12;
  const bw = (W - P.l - P.r) / 7;
  const y = v => H - P.b - v / max * (H - P.t - P.b);
  for (const v of niceTicks(max, 4)) {
    el('line', {x1: P.l, x2: W - P.r, y1: y(v), y2: y(v),
      stroke: 'var(--grid)', 'stroke-width': 1}, svg);
    el('text', {x: P.l - 6, y: y(v) + 3, 'text-anchor': 'end',
      class: 'num'}, svg).textContent = v;
  }
  const peak = active.reduce((a, b) => dow[b] > dow[a] ? b : a, active[0]);
  DAYS.forEach((day, i) => {
    const bx = P.l + i * bw + bw * .2, w = bw * .6;
    const on = active.includes(i);
    const bar = el('rect', {x: bx, y: y(dow[i]), width: w,
      height: H - P.b - y(dow[i]),
      fill: i === peak && on ? color : 'rgba(243,234,217,.16)',
      opacity: on ? 1 : .28}, svg);
    el('text', {x: bx + w / 2, y: H - 8, 'text-anchor': 'middle',
      opacity: on ? 1 : .4}, svg).textContent = day.slice(0, 3);
    if (i === peak && on)
      el('text', {x: bx + w / 2, y: y(dow[i]) - 6, 'text-anchor': 'middle',
        fill: 'var(--cream)', 'font-weight': 600, class: 'num'}, svg)
        .textContent = dow[i];
    bar.addEventListener('mousemove', ev =>
      showTT(`<b>${day}</b><br>Avg: <b>${dow[i]}</b>`, ev.clientX, ev.clientY));
    bar.addEventListener('mouseleave', hideTT);
  });
}

// ---------- hour x day service grid
function renderHeat() {
  const box = document.getElementById('heat'); box.innerHTML = '';
  const heat = cur().heat, hours = DATA.hours, rows = dayIdxs();
  const W = 560, H = 30 + rows.length * 30 + 34;
  const P = {l: 42, r: 8, t: 8, b: 34};
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`}, box);
  const cw = (W - P.l - P.r) / hours.length, ch = 30;
  const max = Math.max(...rows.flatMap(r => heat[r]));
  rows.forEach((r, ri) => {
    el('text', {x: P.l - 6, y: P.t + ri * ch + ch / 2 + 3,
      'text-anchor': 'end'}, svg).textContent = DAYS[r].slice(0, 3);
    hours.forEach((h, c) => {
      const v = heat[r][c];
      const bin = v === 0 ? 0 : Math.min(7, 1 + Math.floor(v / max * 6.999));
      const rect = el('rect', {x: P.l + c * cw + 1, y: P.t + ri * ch + 1,
        width: cw - 2, height: ch - 2, rx: 1.5, fill: RAMP[bin]}, svg);
      rect.addEventListener('mousemove', ev => showTT(
        `<b>${DAYS[r]} ${h > 12 ? h - 12 : h}${h >= 12 ? 'pm' : 'am'}</b>` +
        `<br>Orders: <b>${v}</b>`, ev.clientX, ev.clientY));
      rect.addEventListener('mouseleave', hideTT);
    });
  });
  hours.forEach((h, c) => { if (c % 2 === 0)
    el('text', {x: P.l + c * cw + cw / 2, y: P.t + rows.length * ch + 16,
      'text-anchor': 'middle'}, svg)
      .textContent = h > 12 ? (h - 12) + 'p' : h + (h === 12 ? 'p' : 'a'); });
  el('text', {x: P.l, y: H - 4}, svg).textContent = 'quiet';
  el('text', {x: W - P.r, y: H - 4, 'text-anchor': 'end'}, svg)
    .textContent = 'packed';
  for (let b = 0; b < 8; b++)
    el('rect', {x: P.l + 44 + b * 14, y: H - 12, width: 12, height: 7,
      rx: 1.5, fill: RAMP[b]}, svg);
}

// ---------- ticket rail
(function renderTickets() {
  const rail = document.getElementById('rail');
  DATA.forecast.forEach((f, i) => {
    const d = new Date(f.d + 'T12:00');
    const t = document.createElement('div');
    t.className = 'ticket';
    t.style.setProperty('--d', i);
    t.style.setProperty('--rot', ((i % 3) - 1) * 1.1 + 'deg');
    t.innerHTML =
      `<div class="t-day">${f.day.slice(0, 3).toUpperCase()}</div>` +
      `<div class="t-date num">${MON[d.getMonth()]} ${d.getDate()}</div>` +
      `<div class="t-num">${f.f}</div>` +
      `<div class="t-unit">orders, whole house</div>` +
      `<span class="stamp ${f.tier}">${f.tier}</span>`;
    t.addEventListener('mousemove', ev => showTT(
      `<b>${dlabel(f.d)}</b><br>Forecast: <b>${f.f}</b> orders<br>` +
      {BUSY: 'Add staff', STEADY: 'Normal staffing',
       LIGHT: 'Minimum staff'}[f.tier], ev.clientX, ev.clientY));
    t.addEventListener('mouseleave', hideTT);
    rail.appendChild(t);
  });
  const m = DATA.models.map((r, i) =>
    `${i === 0 ? '<b>' : ''}${r.model} ${r['MAPE_%']}%${i === 0 ? '</b>' : ''}`)
    .join(' · ');
  document.getElementById('models').innerHTML =
    `Forecast by <b>${DATA.bestModel}</b> — 28-day holdout error (MAPE): ${m}`;
})();

// ---------- render
document.getElementById('range').textContent = DATA.range;
function renderRhythm() { renderDow(); renderHeat(); }
function renderAll() {
  renderLedger(); renderTrend(); renderOverview(); renderRhythm();
  syncPressed();
}
renderAll();
</script>
"""


def main() -> None:
    payload = build_payload()
    html = (TEMPLATE
            .replace("__F_FRAUNCES__", font_b64("fraunces-600.woff2"))
            .replace("__F_ARCHIVO__", font_b64("archivo-var.woff2"))
            .replace("__DATA__", json.dumps(payload)))
    out = OUTPUT_DIR / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

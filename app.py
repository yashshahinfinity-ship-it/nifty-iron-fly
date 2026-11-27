"""
NIFTY iron-fly forward-test app (phone friendly).

Tab 1  Signal : is the next session CALM / NEUTRAL / HIGH-VOL, plus a rule-of-thumb strike & wing suggestion.
Tab 2  Enter  : type in the legs you actually traded (blank boxes) and save the entry.
Tab 3  Exit   : pick an open trade, type the exit prices, get P&L, save.
Tab 4  Review : all closed trades, win rate, cumulative P&L, split by signal.

Trades are saved to your Google Sheet (via the Apps Script web link) if configured in Streamlit secrets,
otherwise kept only in this browser session (download the CSV before closing).
This app is a research tool, not investment advice.
"""
import json
import math
import uuid
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
IST = ZoneInfo("Asia/Kolkata")
START = "2015-01-01"
CALM_Q = 0.40
STEP = 50                 # NIFTY strike spacing
WING_X = 1.5              # wings = 1.5 x the VIX-implied 1-day straddle (same rule as the backtest)
DEFAULT_LOT = 65          # CHECK your broker: NIFTY lot size is revised by NSE from time to time
try:
    START_CAPITAL = float(st.secrets.get("START_CAPITAL", 200000))   # forward-test starting capital (Rs)
except Exception:
    START_CAPITAL = 200000.0
RISK_RULE_PCT = 1.0       # max worst-case loss per trade, % of current capital

st.set_page_config(page_title="NIFTY Iron Fly Tracker", page_icon="📈", layout="centered")

# ============================== signal model ==============================
def _download(ticker, need):
    d = yf.download(ticker, start=START, progress=False, auto_adjust=True)
    if d is None or d.empty:
        raise ValueError(f"no data for {ticker}")
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d.dropna(subset=list(need))


def _features(d, vix):
    df = d.join(vix, how="left").ffill()
    df["ret"] = np.log(df["Close"]).diff()
    df["range_pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["gap"] = df["Open"] / df["Close"].shift(1) - 1
    for n in (5, 10, 20):
        df[f"rv_{n}"] = df["ret"].rolling(n).std() * np.sqrt(252)
    df["range_ma5"] = df["range_pct"].rolling(5).mean()
    df["range_ma20"] = df["range_pct"].rolling(20).mean()
    df["range_ratio"] = df["range_ma5"] / df["range_ma20"]
    df["ret_1"] = df["ret"]
    df["ret_5"] = df["ret"].rolling(5).sum()
    df["dist_50dma"] = df["Close"] / df["Close"].rolling(50).mean() - 1
    df["dow"] = df.index.dayofweek
    thr = df["range_pct"].rolling(60).median()
    df["thr"] = thr
    df["rel_range_1d"] = df["range_pct"] / thr
    df["rel_range_5d"] = df["range_ma5"] / thr
    df["rel_range_20d"] = df["range_ma20"] / thr
    df["rel_range_max5"] = df["range_pct"].rolling(5).max() / thr
    df["days_above_5"] = (df["range_pct"] > thr).astype(float).rolling(5).sum()
    df["vix_chg_5"] = df["vix"].pct_change(5)
    df["vix_pctile"] = df["vix"].rolling(250).rank(pct=True)
    df["vix_rel_60"] = df["vix"] / df["vix"].rolling(60).mean()
    feats = ["rv_5", "rv_10", "rv_20", "range_pct", "range_ma5", "range_ma20", "range_ratio", "ret_1", "ret_5",
             "dist_50dma", "gap", "dow", "vix", "vix_chg_5", "vix_pctile", "rel_range_1d", "rel_range_5d",
             "rel_range_20d", "rel_range_max5", "days_above_5", "vix_rel_60"]
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    nxt = df["range_pct"].shift(-1)
    df["target"] = np.where(nxt.isna() | thr.isna(), np.nan, (nxt > thr).astype(float))
    return df, feats


def _model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=42)


@st.cache_data(ttl=1800, show_spinner=False)
def compute_signal(cache_key: str):
    """cache_key only forces a refresh every ~30 min; it is not used in the maths."""
    d = _download("^NSEI", ("Open", "High", "Low", "Close"))
    vix = _download("^INDIAVIX", ("Close",))["Close"].rename("vix")
    now = datetime.now(IST)
    dropped_today = False
    if d.index[-1].date() == now.date() and (now.hour, now.minute) < (15, 45):
        d = d.iloc[:-1]                       # today's candle is still forming: never feed it to the model
        dropped_today = True
    df, feats = _features(d, vix)
    data = df.dropna(subset=feats + ["target"])
    X, y = data[feats], data["target"].astype(int)

    oos = pd.Series(np.nan, index=X.index)
    aucs, naive = [], []
    for tr, te in TimeSeriesSplit(n_splits=5, gap=1).split(X):
        m = _model().fit(X.iloc[tr], y.iloc[tr])
        oos.iloc[te] = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], oos.iloc[te]))
        naive.append(max(roc_auc_score(y.iloc[te], data["rel_range_1d"].iloc[te]),
                         roc_auc_score(y.iloc[te], data["rel_range_5d"].iloc[te])))
    o = oos.dropna()
    cut_calm, cut_wild = o.quantile(CALM_Q), o.quantile(1 - CALM_Q)
    yo = y.loc[o.index]

    final = _model().fit(X, y)
    latest = df[feats].dropna().iloc[[-1]]
    p = float(final.predict_proba(latest)[0, 1])
    signal = "CALM" if p <= cut_calm else ("HIGH-VOL" if p >= cut_wild else "NEUTRAL")
    last_date = latest.index[0]
    nxt_day = last_date + pd.offsets.BDay(1)
    return {
        "thr": float(df["thr"].loc[latest.index[0]]), "p": p, "signal": signal, "last_date": str(last_date.date()), "for_session": str(nxt_day.date()),
        "spot": float(d["Close"].loc[last_date]), "vix": float(df["vix"].loc[last_date]),
        "auc": float(np.mean(aucs)), "naive_auc": float(np.mean(naive)),
        "after_calm": float(yo[o <= cut_calm].mean()), "after_wild": float(yo[o >= cut_wild].mean()),
        "dropped_today": dropped_today, "made_at": now.strftime("%d %b %H:%M IST"),
    }


# ============================== option maths ==============================
def straddle_estimate(spot, vix):
    """1-day ATM straddle price implied by VIX (same proxy as the backtest)."""
    return 0.798 * (vix / 100.0) * math.sqrt(1 / 252) * spot


def suggest(spot, vix):
    atm = round(spot / STEP) * STEP
    s = straddle_estimate(spot, vix)
    wing = max(STEP, round(WING_X * s / STEP) * STEP)
    return atm, s, wing


def clean_legs(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ("Strike", "Lots", "Price"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["Action"].isin(["Buy", "Sell"]) & d["Type"].isin(["CE", "PE"])]
    d = d.dropna(subset=["Strike", "Lots", "Price"])
    return d[(d["Strike"] > 0) & (d["Lots"] > 0) & (d["Price"] >= 0)].reset_index(drop=True)


def payoff_at(legs: pd.DataFrame, S: float, lot: int) -> float:
    tot = 0.0
    for r in legs.itertuples():
        intrinsic = max(S - r.Strike, 0) if r.Type == "CE" else max(r.Strike - S, 0)
        sign = 1 if r.Action == "Buy" else -1
        tot += sign * (intrinsic - r.Price) * r.Lots * lot
    return tot


def risk_profile(legs: pd.DataFrame, lot: int):
    """Profit/loss AT EXPIRY of the shape you typed (a same-day exit with DTE>0 usually loses less)."""
    pts = sorted(set([0.0] + list(legs["Strike"])))
    big = max(pts) * 5
    unlimited = payoff_at(legs, big, lot) < payoff_at(legs, max(pts) + 1, lot) - 1e-6
    vals = [payoff_at(legs, s, lot) for s in pts + [max(pts) + 1]]
    return {"max_loss": None if unlimited else min(vals), "max_profit": max(vals), "unlimited_loss": unlimited}


def legs_text(legs: list) -> str:
    out = []
    for l in legs:
        s = f"{l['Action'][0]} {int(l['Strike'])}{l['Type']} {l['Lots']}L @{l['Price']}"
        if l.get("ExitPrice") not in (None, ""):
            s += f"->{l['ExitPrice']}"
        out.append(s)
    return " | ".join(out)


def pnl_rs(legs: list, lot: int) -> float:
    tot = 0.0
    for l in legs:
        sign = 1 if l["Action"] == "Buy" else -1
        tot += sign * (float(l["ExitPrice"]) - float(l["Price"])) * float(l["Lots"]) * lot
    return tot



# ============================== entry checks ==============================
def check_structure(legs: pd.DataFrame, spot=None, wing=None):
    """Wrong strikes / wrong side / naked legs. Returns (errors, warnings)."""
    errs, warns = [], []
    ex_w = wing or 300
    bad = legs[(legs["Strike"] % STEP) != 0]
    if len(bad):
        errs.append(f"Strike(s) {sorted(set(int(x) for x in bad['Strike']))} are not multiples of {STEP}. NIFTY strikes move in steps of {STEP}.")
    dup = legs[legs.duplicated(["Type", "Strike"], keep=False)]
    if len(dup):
        errs.append("The same strike and type appears twice (e.g. you both buy and sell the same option). Check the rows.")
    for typ in ("CE", "PE"):
        sells = legs[(legs["Type"] == typ) & (legs["Action"] == "Sell")]
        buys = legs[(legs["Type"] == typ) & (legs["Action"] == "Buy")]
        if len(sells) and buys.empty:
            k = int(sells["Strike"].iloc[0])
            errs.append(f"Sold {k} {typ} has no protection. Buy a farther {typ} (e.g. {k + ex_w if typ == 'CE' else k - ex_w} {typ}). "
                        "This forward test is for defined-risk iron flies only.")
        for b in buys.itertuples():
            if len(sells) == 0:
                continue
            if typ == "CE" and b.Strike <= sells["Strike"].min():
                k = int(sells["Strike"].min())
                errs.append(f"Wrong leg: bought {int(b.Strike)} CE is not above your sold {k} CE. "
                            f"Protection for a sold CE must be HIGHER, e.g. {k + ex_w} CE.")
            if typ == "PE" and b.Strike >= sells["Strike"].max():
                k = int(sells["Strike"].max())
                errs.append(f"Wrong leg: bought {int(b.Strike)} PE is not below your sold {k} PE. "
                            f"Protection for a sold PE must be LOWER, e.g. {k - ex_w} PE.")
    if legs["Lots"].nunique() > 1:
        warns.append("Lots differ between legs. Part of the position may be unprotected.")
    if spot:
        atm = round(spot / STEP) * STEP
        for r in legs[legs["Action"] == "Sell"].itertuples():
            if abs(r.Strike - atm) >= 2 * STEP:
                warns.append(f"Sold {int(r.Strike)} {r.Type} is {abs(int(r.Strike - spot))} points from spot {spot:,.0f}: that is not at-the-money.")
    return errs, warns


def check_prices(legs: pd.DataFrame, col="Price", credit=False):
    """A farther-out option must never cost more than a nearer one of the same type."""
    errs = []
    for typ in ("CE", "PE"):
        t = legs[legs["Type"] == typ]
        for a in t.itertuples():
            for b in t.itertuples():
                pa, pb = float(getattr(a, col)), float(getattr(b, col))
                nearer = (a.Strike < b.Strike) if typ == "CE" else (a.Strike > b.Strike)
                if nearer and pa < pb - 1e-9:
                    errs.append(f"Price looks wrong: {int(b.Strike)} {typ} ({b.Action}, ₹{pb:g}) costs MORE than "
                                f"{int(a.Strike)} {typ} ({a.Action}, ₹{pa:g}). A farther-out {typ} must be cheaper.")
    if credit:
        cr = sum((1 if r.Action == "Sell" else -1) * float(getattr(r, col)) * r.Lots for r in legs.itertuples())
        if cr <= 0:
            errs.append(f"Net DEBIT of {abs(cr):.2f} points: an iron fly should bring in a net credit. Check which prices went in which row.")
    return errs


@st.dialog("Please fix these first")
def show_problems(msgs):
    for m in msgs:
        st.error(m)
    st.caption("Close this box, correct the table, then press Save again.")


def block(msgs):
    """Tell the user nothing was saved (pop-up + a line on the page)."""
    st.error(f"Not saved: {len(msgs)} problem(s) found.")
    try:
        show_problems(msgs)
    except Exception:
        for m in msgs:
            st.error(m)


def next_tuesday(today=None):
    d = today or datetime.now(IST).date()
    return d + timedelta(days=(1 - d.weekday()) % 7)


# ============================== storage ==============================
HEADERS_BY = {
    "Trades": ["id", "date", "signal", "p_high_vol", "vix", "dte", "lot_size", "spot_entry", "entry_legs",
               "entry_time", "exit_time", "spot_exit", "exit_legs", "legs_text", "gross_pnl_rs", "charges_rs",
               "net_pnl_rs", "status", "notes"],
    "Calls": ["last_date", "for_session", "signal", "p_high_vol", "thr", "made_at", "status", "actual_date",
              "actual_range_pct", "actual_big", "verdict"],
}
KEY = {"Trades": "id", "Calls": "last_date"}


class LocalStore:
    kind = "local"

    def all(self, sheet="Trades"):
        return list(st.session_state.setdefault(f"local_{sheet}", []))

    def add(self, row, sheet="Trades"):
        st.session_state.setdefault(f"local_{sheet}", []).append({k: str(row.get(k, "")) for k in HEADERS_BY[sheet]})

    def update(self, key, fields, sheet="Trades"):
        for r in st.session_state.setdefault(f"local_{sheet}", []):
            if r[KEY[sheet]] == key:
                r.update({k: str(v) for k, v in fields.items()})


class SheetStore:
    kind = "sheet"

    def __init__(self, url, token):
        self.url, self.token = url, token

    def _call(self, action, sheet, **kw):
        r = requests.post(self.url, data=json.dumps({"token": self.token, "action": action, "sheet": sheet, **kw}),
                          headers={"Content-Type": "text/plain"}, timeout=40)
        r.raise_for_status()
        j = r.json()
        if not j.get("ok"):
            raise RuntimeError(j.get("error", "unknown error from the sheet script"))
        if j.get("sheet") != sheet:
            raise RuntimeError("Your Google Apps Script is out of date. Paste the new apps_script.gs, then "
                               "Deploy > Manage deployments > pencil > Version: New version > Deploy.")
        return j

    def all(self, sheet="Trades"):
        return self._call("list", sheet)["rows"]

    def add(self, row, sheet="Trades"):
        self._call("add", sheet, row={k: str(row.get(k, "")) for k in HEADERS_BY[sheet]})

    def update(self, key, fields, sheet="Trades"):
        self._call("update", sheet, key=str(key), fields={k: str(v) for k, v in fields.items()})


def get_store():
    try:
        url, token = st.secrets["SHEET_URL"], st.secrets["SHEET_TOKEN"]
        return SheetStore(url, token)
    except Exception:
        return LocalStore()


# ---------- call scorecard logic ----------
def norm_date(x) -> str:
    """Google Sheets sometimes turns '2026-11-20' into a Date and sends it back as
    'Fri Nov 20 2026 00:00:00 GMT+0530 (...)' or an ISO time. Always return YYYY-MM-DD."""
    x = str(x).strip()
    if not x:
        return ""
    try:
        return str(pd.Timestamp(x).date())
    except Exception:
        pass
    try:
        return str(datetime.strptime(x[:15], "%a %b %d %Y").date())
    except Exception:
        return x


def resolve_call(call: dict, ohlc: pd.DataFrame, now: datetime):
    """Was the call right? Looks at the first trading day AFTER the call's data date.
    Returns the fields to save, or None if that day isn't finished yet."""
    after = ohlc.index[ohlc.index > pd.Timestamp(norm_date(call["last_date"]))]
    if len(after) == 0:
        return None
    day = after[0]
    if day.date() == now.date() and (now.hour, now.minute) < (15, 45):
        return None
    row = ohlc.loc[day]
    rng = float((row["High"] - row["Low"]) / row["Close"])
    thr = float(call["thr"])
    big = rng > thr
    sig = call["signal"]
    verdict = "NO CALL" if sig == "NEUTRAL" else ("RIGHT" if (sig == "HIGH-VOL") == big else "WRONG")
    return {"status": "DONE", "actual_date": str(day.date()), "actual_range_pct": round(100 * rng, 3),
            "actual_big": "BIG" if big else "QUIET", "verdict": verdict}


def verify_pending(calls: list) -> int:
    pend = [c for c in calls if c.get("status") == "PENDING"]
    if not pend:
        return 0
    first = min(pd.Timestamp(norm_date(c["last_date"])) for c in pend) - pd.Timedelta(days=5)
    d = yf.download("^NSEI", start=first.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.dropna(subset=["High", "Low", "Close"])
    now, n = datetime.now(IST), 0
    for c in pend:
        res = resolve_call(c, d, now)
        if res:
            store.update(c["last_date"], res, "Calls")
            n += 1
    return n


store = get_store()


def _cached(name, sheet, refresh=False):
    if refresh or name not in st.session_state:
        try:
            rows = store.all(sheet)
            for r in rows:
                for col in ("date", "last_date", "for_session", "actual_date"):
                    if col in r:
                        r[col] = norm_date(r[col])
            st.session_state[name] = rows
            st.session_state.pop(name + "_err", None)
        except Exception as e:
            st.session_state[name] = []
            st.session_state[name + "_err"] = str(e)
    return st.session_state[name]


def trades(refresh=False):
    return _cached("rows_cache", "Trades", refresh)


def calls(refresh=False):
    return _cached("calls_cache", "Calls", refresh)


def capital_now():
    t = pd.DataFrame(trades()).reindex(columns=HEADERS_BY["Trades"]) if trades() else pd.DataFrame()
    if t.empty or "status" not in t:
        return START_CAPITAL, 0.0
    pnl = pd.to_numeric(t.loc[t["status"] == "CLOSED", "net_pnl_rs"], errors="coerce").sum()
    return START_CAPITAL + float(pnl), float(pnl)


def log_call(sig):
    """Record today's prediction once per data date, so we can grade it after the session."""
    if any(c.get("last_date") == sig["last_date"] for c in calls()):
        return False
    store.add({"last_date": sig["last_date"], "for_session": sig["for_session"], "signal": sig["signal"],
               "p_high_vol": f"{sig['p']:.3f}", "thr": f"{sig['thr']:.6f}", "made_at": sig["made_at"],
               "status": "PENDING"}, "Calls")
    calls(refresh=True)
    return True


def run_grading() -> int:
    """Grade every finished session and refresh the cache. Returns how many calls were graded."""
    n = verify_pending(calls(refresh=True))
    calls(refresh=True)
    return n


def grading_message(n: int):
    if n:
        st.success(f"Graded {n} call(s).")
        return
    now = datetime.now(IST)
    today_open = (now.hour, now.minute) < (15, 45) and now.weekday() < 5
    pend_today = [c for c in calls() if c.get("status") == "PENDING" and c.get("for_session") == now.strftime("%Y-%m-%d")]
    if pend_today and today_open:
        st.info("Today's session is still open. The verdict is available after 3:45 PM IST.")
    else:
        st.info("Nothing new to grade yet (no finished session waiting, or no data yet).")


def render_verdict(full: bool):
    """Latest verdict + forward-test accuracy. full=True adds the trade join and the whole call log."""
    cdf = pd.DataFrame(calls()).reindex(columns=HEADERS_BY["Calls"]) if calls() else pd.DataFrame()
    if cdf.empty or "status" not in cdf:
        st.caption("No calls recorded yet. Press 'Get today's signal' first.")
        return
    done = cdf[cdf["status"] == "DONE"].copy()
    pend = cdf[cdf["status"] == "PENDING"]
    tdf = pd.DataFrame(trades()).reindex(columns=HEADERS_BY["Trades"]) if trades() else pd.DataFrame()
    if not done.empty:
        last = done.sort_values("actual_date").iloc[-1]
        mine = "  ·  you took no trade that day"
        if not tdf.empty:
            day_tr = tdf[(tdf["date"] == last["actual_date"]) & (tdf["status"] == "CLOSED")]
            if len(day_tr):
                mine = f"  ·  your trade that day: net ₹{pd.to_numeric(day_tr['net_pnl_rs'], errors='coerce').sum():,.0f}"
        icon = {"RIGHT": "✅", "WRONG": "❌", "NO CALL": "➖"}.get(last["verdict"], "")
        st.subheader(f"{icon} Latest verdict: {last['verdict']}")
        st.write(f"Call for **{last['actual_date']}** was **{last['signal']}**. The day's range was "
                 f"**{last['actual_range_pct']}%** vs a 'normal' of {100 * float(last['thr']):.2f}% "
                 f"→ it was a **{last['actual_big']}** day{mine}.")
        dec = done[done["verdict"].isin(["RIGHT", "WRONG"])]
        if len(dec):
            acc = dec.groupby("signal")["verdict"].agg(calls="size", right=lambda x: (x == "RIGHT").sum())
            acc["accuracy"] = (100 * acc["right"] / acc["calls"]).round(0).astype(int).astype(str) + "%"
            if full:
                st.markdown("**Forward-test accuracy**")
                st.dataframe(acc, width="stretch")
            else:
                parts = " · ".join(f"{k}: {int(v.right)}/{int(v.calls)} right" for k, v in acc.iterrows())
                st.markdown(f"**Forward test so far:** {int((dec['verdict'] == 'RIGHT').sum())} of {len(dec)} "
                            f"decisive calls right ({(dec['verdict'] == 'RIGHT').mean():.0%})  ·  {parts}")
            st.caption("Backtest said about 67-68%. Under ~30 calls this is mostly noise.")
        if full and not tdf.empty:
            cl = tdf[tdf["status"] == "CLOSED"].copy()
            cl["net"] = pd.to_numeric(cl["net_pnl_rs"], errors="coerce")
            j = cl.merge(done[["actual_date", "signal", "verdict"]], left_on="date", right_on="actual_date",
                         how="inner", suffixes=("", "_call"))
            if len(j):
                st.markdown("**Your trades by call result**")
                st.dataframe(j.groupby("verdict")["net"].agg(trades="size", avg="mean", total="sum").round(0),
                             width="stretch")
    else:
        st.caption("No call has been graded yet.")
    if len(pend):
        st.caption(f"{len(pend)} call(s) waiting for their session to finish.")
    if full:
        show = cdf.copy()
        show["thr"] = (100 * pd.to_numeric(show["thr"], errors="coerce")).round(2)
        st.markdown("**All recorded calls**")
        st.dataframe(show.iloc[::-1].rename(columns={"thr": "normal range %"}), width="stretch", hide_index=True)


# ============================== UI ==============================
st.title("📈 NIFTY Iron Fly Tracker")
if store.kind == "sheet":
    st.caption("Saving to your Google Sheet ✅")
else:
    st.warning("Google Sheet not connected: trades live only in this browser session. "
               "Download the CSV in the Review tab before you close the page.", icon="⚠️")

_cap, _pnl = capital_now()
h1, h2, h3 = st.columns(3)
h1.metric("Capital now", f"₹{_cap:,.0f}")
h2.metric("Net P&L", f"₹{_pnl:,.0f}")
h3.metric("Return", f"{100 * _pnl / START_CAPITAL:.2f}%", help=f"Started with ₹{START_CAPITAL:,.0f}")
for _n in ("rows_cache_err", "calls_cache_err"):
    if st.session_state.get(_n):
        st.error(st.session_state[_n])

tab_sig, tab_in, tab_out, tab_rev, tab_sc = st.tabs(["1 Signal", "2 Enter trade", "3 Exit trade", "4 Review", "5 Scorecard"])

# ---------------------------- tab 1: signal ----------------------------
with tab_sig:
    st.write("Press the button **after the market opens** (it uses yesterday's close) or **after 3:45 PM** "
             "(it uses today's close and gives you tomorrow's call).")
    if st.button("Get today's signal", type="primary", width="stretch"):
        compute_signal.clear()
        with st.spinner("Downloading data and running the model (about a minute)…"):
            try:
                st.session_state["sig"] = compute_signal(datetime.now(IST).strftime("%Y-%m-%d %H%M"))
                try:
                    if log_call(st.session_state["sig"]):
                        st.toast("Call recorded in your Scorecard ✅")
                except Exception as e:
                    st.error(f"Signal OK, but I could not record it: {e}")
            except Exception as e:
                st.error(f"Could not run the model: {e}")

    sig = st.session_state.get("sig")
    if sig:
        colour = {"CALM": "🟢", "NEUTRAL": "⚪", "HIGH-VOL": "🔴"}[sig["signal"]]
        st.subheader(f"{colour} {sig['signal']}  ·  P(big day) {sig['p']:.0%}")
        st.caption(f"Data up to {sig['last_date']} close  ·  call is for the session of {sig['for_session']}  ·  "
                   f"made {sig['made_at']}")
        if sig["dropped_today"]:
            st.info("Today's unfinished candle was ignored on purpose.")
        if sig["signal"] == "CALM":
            st.success("Calm day expected → premium-selling conditions look OK. Defined risk only (iron fly with bought wings).")
        elif sig["signal"] == "HIGH-VOL":
            st.error("Bigger-than-normal move expected → do NOT sell premium. This is not a buy signal either: "
                     "buying options on these days lost money in our tests.")
        else:
            st.info("No clear lean → skip today.")
        st.caption(f"Track record (walk-forward): after CALM calls the next day was big {sig['after_calm']:.0%} of the time; "
                   f"after HIGH-VOL calls {sig['after_wild']:.0%}. Model AUC {sig['auc']:.2f} vs simple rule {sig['naive_auc']:.2f}.")

        st.divider()
        st.markdown("**Rule-of-thumb strikes (a starting idea, not a quote)**")
        c1, c2 = st.columns(2)
        spot = c1.number_input("NIFTY spot now", value=float(round(sig["spot"], 2)), step=1.0, key="spot_now")
        vix = c2.number_input("India VIX now", value=float(round(sig["vix"], 2)), step=0.1, key="vix_now")
        atm, s, wing = suggest(spot, vix)
        st.write(f"- **Sell** {atm} CE and {atm} PE (ATM)\n"
                 f"- **Buy** {atm + wing} CE and {atm - wing} PE (wings, {wing} points away)\n"
                 f"- VIX-implied 1-day straddle ≈ {s:.0f} points. Wings = 1.5 × that, the same rule as the backtest.\n"
                 f"- Your real fills decide everything: check the live option chain and log what you actually got.")
    else:
        st.caption("No signal yet. Press the button.")

    st.divider()
    st.subheader("Day-end verdict")
    st.caption("Press this after 3:45 PM IST. It grades the latest call against what NIFTY actually did that day, "
               "whether or not you took a trade. Every graded call is saved in your Scorecard.")
    if st.button("Get day-end verdict", type="primary", key="verdict_btn", width="stretch"):
        with st.spinner("Checking what the market did…"):
            try:
                grading_message(run_grading())
            except Exception as e:
                st.error(f"Could not grade: {e}")
    render_verdict(full=False)

# ---------------------------- tab 2: enter trade ----------------------------
with tab_in:
    st.markdown("**Step 1: set expiry, spot and wing. Step 2: type only the 4 prices.** Strikes fill in by themselves.")
    sig = st.session_state.get("sig") or {}
    if not sig:                                           # fall back to today's recorded call
        today_call = [c for c in calls() if c.get("for_session") == datetime.now(IST).strftime("%Y-%m-%d")]
        if today_call:
            sig = {"signal": today_call[-1]["signal"], "p": float(today_call[-1]["p_high_vol"] or 0), "vix": 0.0}
    today = datetime.now(IST).date()
    sp0 = st.session_state.get("spot_now") or sig.get("spot")
    vx0 = st.session_state.get("vix_now") or sig.get("vix")
    wing0 = suggest(float(sp0), float(vx0))[2] if (sp0 and vx0) else 300

    c1, c2 = st.columns(2)
    expiry = c1.date_input("Expiry date (calendar)", value=next_tuesday(today), min_value=today, format="DD/MM/YYYY",
                           help="Pick the expiry you traded. The app works out the days to expiry (DTE) from today.")
    dte = (expiry - today).days
    c1.caption(f"DTE = {dte} day(s)")
    lot = c2.number_input("Lot size", value=DEFAULT_LOT, step=1, help="Check your broker, NSE revises this.")
    c3, c4 = st.columns(2)
    spot_e = c3.number_input("NIFTY spot at entry", value=float(round(sp0)) if sp0 else None, step=1.0,
                             placeholder="type, e.g. 25900", key="spot_entry",
                             help="Type the live NIFTY level. The sold strikes jump to the nearest 50 automatically.")
    wing = c4.number_input("Wing distance (points)", value=int(wing0), step=STEP, min_value=STEP, key="wing_pts",
                           help="How far the bought strikes sit from the sold ones. Default = 1.5 x the VIX-implied move.")

    if spot_e:
        atm = int(round(spot_e / STEP) * STEP)
        blank = pd.DataFrame({"Action": ["Buy", "Buy", "Sell", "Sell"], "Type": ["CE", "PE", "CE", "PE"],
                              "Strike": [atm + int(wing), atm - int(wing), atm, atm], "Lots": [1, 1, 1, 1],
                              "Price": [None] * 4})
        fill_key = f"{atm}_{int(wing)}"
        st.info(f"Spot {spot_e:,.0f} → sell {atm} CE and {atm} PE, buy {atm + int(wing)} CE and {atm - int(wing)} PE. "
                "Change the spot above and the strikes follow. Set spot and wing first, then type prices.")
    else:
        blank = pd.DataFrame({"Action": [None] * 4, "Type": [None] * 4, "Strike": [None] * 4,
                              "Lots": [None] * 4, "Price": [None] * 4})
        fill_key = "blank"
        st.caption("Type the NIFTY spot above and the 4 legs will fill in.")

    with st.form("entry_form"):
        legs_df = st.data_editor(
            blank, num_rows="dynamic", width="stretch", hide_index=True, key=f"entry_legs_editor_{fill_key}",
            column_config={
                "Action": st.column_config.SelectboxColumn("Buy/Sell", options=["Buy", "Sell"]),
                "Type": st.column_config.SelectboxColumn("Call/Put", options=["CE", "PE"]),
                "Strike": st.column_config.NumberColumn("Strike", step=STEP, format="%d"),
                "Lots": st.column_config.NumberColumn("Lots", step=1, min_value=1),
                "Price": st.column_config.NumberColumn("Entry price", step=0.05, min_value=0.0),
            })
        notes = st.text_input("Notes (optional)")
        go = st.form_submit_button("Save entry", type="primary", width="stretch")

    if go:
        legs = clean_legs(legs_df)
        problems = []
        if not spot_e:
            problems.append("Type the NIFTY spot at entry.")
        if legs.empty:
            problems.append("Fill at least one full leg (Buy/Sell, Call/Put, Strike, Lots, Price).")
        else:
            e1, w1 = check_structure(legs, spot_e, int(wing))
            problems += e1
            problems += check_prices(legs, "Price", credit=(len(legs) == 4))
            if len(legs) == 4 and not ((legs["Action"] == "Sell").sum() == 2 and (legs["Action"] == "Buy").sum() == 2):
                problems.append("4 legs but not 2 sells + 2 buys: check the Buy/Sell column.")
            for w in w1:
                st.warning(w)
        if problems:
            block(problems)
        else:
            rp = risk_profile(legs, int(lot))
            pct = 100 * -rp["max_loss"] / _cap if (rp["max_loss"] is not None and _cap) else 0
            st.info(f"Worst case at expiry: ₹{rp['max_loss']:,.0f} ({pct:.1f}% of your capital ₹{_cap:,.0f})  ·  "
                    f"best case: ₹{rp['max_profit']:,.0f}")
            if pct > RISK_RULE_PCT:
                st.warning(f"That is above your {RISK_RULE_PCT:.0f}% per-trade risk rule. Consider narrower wings or fewer lots.")
            trade_id = uuid.uuid4().hex[:8]
            now = datetime.now(IST)
            ll = legs.to_dict("records")
            note_txt = f"Expiry {expiry:%Y-%m-%d}" + (f" | {notes}" if notes else "")
            row = {"id": trade_id, "date": now.strftime("%Y-%m-%d"), "signal": sig.get("signal", "n/a"),
                   "p_high_vol": f"{sig['p']:.3f}" if sig else "", "vix": f"{sig['vix']:.2f}" if sig.get("vix") else "",
                   "dte": int(dte), "lot_size": int(lot), "spot_entry": spot_e or "",
                   "entry_legs": json.dumps(ll), "entry_time": now.strftime("%H:%M"), "legs_text": legs_text(ll),
                   "status": "OPEN", "notes": note_txt}
            try:
                store.add(row, "Trades")
                trades(refresh=True)
                st.success(f"Saved ✅  trade id {trade_id}. Go to the Exit tab when you close it.")
            except Exception as e:
                st.error(f"Could not save to the sheet: {e}")

# ---------------------------- tab 3: exit trade ----------------------------
with tab_out:
    if st.button("Refresh open trades", width="stretch"):
        trades(refresh=True)
    rows = trades()
    open_rows = [r for r in rows if str(r.get("status")) == "OPEN"]
    if not open_rows:
        st.caption("No open trades.")
    else:
        labels = {r["id"]: f"{r['date']}  ·  {r.get('signal','')}  ·  DTE {r.get('dte','')}  ·  {r.get('legs_text','')[:60]}"
                  for r in open_rows}
        pick = st.selectbox("Open trade", list(labels), format_func=lambda k: labels[k])
        tr = next(r for r in open_rows if r["id"] == pick)
        entry_legs = json.loads(tr["entry_legs"])
        ex = pd.DataFrame(entry_legs)
        ex["ExitPrice"] = None
        with st.form("exit_form"):
            lot = st.number_input("Lot size", value=int(float(tr.get("lot_size") or DEFAULT_LOT)), step=1)
            c1, c2 = st.columns(2)
            spot_x = c1.number_input("NIFTY spot at exit", value=None, step=1.0, placeholder="type")
            charges = c2.number_input("Total charges ₹ (brokerage, STT, taxes)", value=0.0, step=10.0)
            edited = st.data_editor(
                ex, hide_index=True, width="stretch", key=f"exit_editor_{pick}",
                disabled=["Action", "Type", "Strike", "Lots", "Price"],
                column_config={"Price": st.column_config.NumberColumn("Entry price"),
                               "ExitPrice": st.column_config.NumberColumn("Exit price", step=0.05, min_value=0.0)})
            force = st.checkbox("Save even if the price checker complains (illiquid strikes can look odd)")
            gox = st.form_submit_button("Save exit", type="primary", width="stretch")
        if gox:
            exit_problems = []
            if edited["ExitPrice"].isna().any():
                st.error("Fill the exit price for every leg (use 0 if it expired worthless).")
            else:
                chk = edited.copy()
                chk["ExitPrice"] = pd.to_numeric(chk["ExitPrice"], errors="coerce")
                exit_problems = check_prices(chk, "ExitPrice")
            if exit_problems and not force:
                block(exit_problems)
            elif not edited["ExitPrice"].isna().any():
                legs_x = edited.to_dict("records")
                gross = pnl_rs(legs_x, int(lot))
                net = gross - charges
                try:
                    store.update(pick, {"exit_time": datetime.now(IST).strftime("%H:%M"), "spot_exit": spot_x or "",
                                       "exit_legs": json.dumps(legs_x), "legs_text": legs_text(legs_x),
                                       "gross_pnl_rs": round(gross, 2), "charges_rs": round(charges, 2),
                                       "net_pnl_rs": round(net, 2), "status": "CLOSED"}, "Trades")
                    trades(refresh=True)
                    msg = f"Saved ✅  gross ₹{gross:,.0f}  ·  net after charges ₹{net:,.0f}"
                    if net >= 0:
                        st.success(msg)
                    else:
                        st.error(msg)
                except Exception as e:
                    st.error(f"Could not save to the sheet: {e}")

# ---------------------------- tab 4: review ----------------------------
with tab_rev:
    if st.button("Refresh", key="rev_refresh", width="stretch"):
        trades(refresh=True)
    allr = pd.DataFrame(trades()).reindex(columns=HEADERS_BY["Trades"]) if trades() else pd.DataFrame()
    if allr.empty or "status" not in allr:
        st.caption("No trades yet.")
    else:
        cl = allr[allr["status"] == "CLOSED"].copy()
        for c in ("gross_pnl_rs", "charges_rs", "net_pnl_rs"):
            cl[c] = pd.to_numeric(cl[c], errors="coerce")
        cl = cl.dropna(subset=["net_pnl_rs"])
        st.caption(f"{len(cl)} closed · {(allr['status'] == 'OPEN').sum()} open")
        if len(cl) >= 1:
            cl = cl.sort_values(["date", "entry_time"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Net P&L", f"₹{cl['net_pnl_rs'].sum():,.0f}")
            m2.metric("Win rate", f"{(cl['net_pnl_rs'] > 0).mean():.0%}")
            m3.metric("Worst trade", f"₹{cl['net_pnl_rs'].min():,.0f}")
            equity = START_CAPITAL + cl["net_pnl_rs"].cumsum().reset_index(drop=True)
            equity.index = range(1, len(equity) + 1)                    # x axis = trade number (no date parsing)
            peak = equity.cummax()
            st.markdown(f"**Capital by trade number** (start ₹{START_CAPITAL:,.0f})")
            st.line_chart(equity.rename("Capital ₹"))
            st.caption(f"Max drawdown so far: {100 * ((equity - peak) / peak).min():.1f}%")
            by = cl.groupby("signal")["net_pnl_rs"].agg(trades="size", avg="mean", total="sum").round(0)
            st.markdown("**By signal at entry**")
            st.dataframe(by, width="stretch")
            by_dte = cl.groupby("dte")["net_pnl_rs"].agg(trades="size", avg="mean", total="sum").round(0)
            st.markdown("**By DTE**")
            st.dataframe(by_dte, width="stretch")
            if len(cl) < 20:
                st.caption("Under ~20 trades this is just noise. Keep going before drawing conclusions.")
        st.download_button("Download all trades (CSV)", allr.to_csv(index=False), "nifty_trades.csv",
                           "text/csv", width="stretch")

# ---------------------------- tab 5: scorecard ----------------------------
with tab_sc:
    st.write("Every time you press the signal button, the call is recorded. After the session ends, press "
             "**Check results** (here or on the first tab) and the app grades it: was the day really calm / big?")
    c1, c2 = st.columns(2)
    if c1.button("Check results now", type="primary", width="stretch"):
        with st.spinner("Grading finished sessions…"):
            try:
                grading_message(run_grading())
            except Exception as e:
                st.error(f"Could not grade: {e}")
    if c2.button("Refresh", key="sc_refresh", width="stretch"):
        calls(refresh=True)
        trades(refresh=True)
    render_verdict(full=True)

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
        "p": p, "signal": signal, "last_date": str(last_date.date()), "for_session": str(nxt_day.date()),
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


# ============================== storage ==============================
HEADERS = ["id", "date", "signal", "p_high_vol", "vix", "dte", "lot_size", "spot_entry", "entry_legs",
           "entry_time", "exit_time", "spot_exit", "exit_legs", "legs_text", "gross_pnl_rs", "charges_rs",
           "net_pnl_rs", "status", "notes"]


class LocalStore:
    kind = "local"

    def __init__(self):
        st.session_state.setdefault("local_rows", [])

    def all(self):
        return list(st.session_state["local_rows"])

    def add(self, row):
        st.session_state["local_rows"].append(dict(row))

    def close(self, trade_id, fields):
        for r in st.session_state["local_rows"]:
            if r["id"] == trade_id:
                r.update({k: str(v) for k, v in fields.items()})


class SheetStore:
    kind = "sheet"

    def __init__(self, url, token):
        self.url, self.token = url, token

    def _call(self, action, **kw):
        r = requests.post(self.url, data=json.dumps({"token": self.token, "action": action, **kw}),
                          headers={"Content-Type": "text/plain"}, timeout=40)
        r.raise_for_status()
        j = r.json()
        if not j.get("ok"):
            raise RuntimeError(j.get("error", "unknown error from the sheet script"))
        return j

    def all(self):
        return self._call("list")["rows"]

    def add(self, row):
        self._call("add", row={k: str(row.get(k, "")) for k in HEADERS})

    def close(self, trade_id, fields):
        self._call("close", id=trade_id, fields={k: str(v) for k, v in fields.items()})


def get_store():
    try:
        url, token = st.secrets["SHEET_URL"], st.secrets["SHEET_TOKEN"]
        return SheetStore(url, token)
    except Exception:
        return LocalStore()


store = get_store()

# ============================== UI ==============================
st.title("📈 NIFTY Iron Fly Tracker")
if store.kind == "sheet":
    st.caption("Saving to your Google Sheet ✅")
else:
    st.warning("Google Sheet not connected: trades live only in this browser session. "
               "Download the CSV in the Review tab before you close the page.", icon="⚠️")

tab_sig, tab_in, tab_out, tab_rev = st.tabs(["1 Signal", "2 Enter trade", "3 Exit trade", "4 Review"])

# ---------------------------- tab 1: signal ----------------------------
with tab_sig:
    st.write("Press the button **after the market opens** (it uses yesterday's close) or **after 3:45 PM** "
             "(it uses today's close and gives you tomorrow's call).")
    if st.button("Get today's signal", type="primary", width="stretch"):
        compute_signal.clear()
        with st.spinner("Downloading data and running the model (about a minute)…"):
            try:
                st.session_state["sig"] = compute_signal(datetime.now(IST).strftime("%Y-%m-%d %H%M"))
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

# ---------------------------- tab 2: enter trade ----------------------------
with tab_in:
    st.markdown("**Fill the boxes with what you actually traded.** Leave unused rows blank.")
    sig = st.session_state.get("sig") or {}
    blank = pd.DataFrame({"Action": [None] * 4, "Type": [None] * 4, "Strike": [None] * 4,
                          "Lots": [None] * 4, "Price": [None] * 4})
    with st.form("entry_form"):
        c1, c2, c3 = st.columns(3)
        dte = c1.selectbox("DTE (days to expiry)", list(range(0, 8)), index=None, placeholder="pick")
        lot = c2.number_input("Lot size", value=DEFAULT_LOT, step=1, help="Check your broker, NSE revises this.")
        spot_e = c3.number_input("NIFTY spot at entry", value=None, step=1.0, placeholder="type")
        legs_df = st.data_editor(
            blank, num_rows="dynamic", width="stretch", hide_index=True, key="entry_legs_editor",
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
        if dte is None:
            st.error("Pick the DTE.")
        elif legs.empty:
            st.error("Fill at least one full leg (Buy/Sell, Call/Put, Strike, Lots, Price).")
        else:
            rp = risk_profile(legs, int(lot))
            if rp["unlimited_loss"]:
                st.error("⚠️ This shape has UNLIMITED loss on the upside (a naked short leg). "
                         "Our test was for defined-risk iron flies only. Saved anyway, but think twice.")
            else:
                st.info(f"Worst case at expiry: ₹{rp['max_loss']:,.0f}  ·  best case at expiry: ₹{rp['max_profit']:,.0f}")
            n_sell, n_buy = (legs["Action"] == "Sell").sum(), (legs["Action"] == "Buy").sum()
            if len(legs) == 4 and not (n_sell == 2 and n_buy == 2):
                st.warning("4 legs but not 2 sells + 2 buys: double-check the Buy/Sell column.")
            trade_id = uuid.uuid4().hex[:8]
            now = datetime.now(IST)
            ll = legs.to_dict("records")
            row = {"id": trade_id, "date": now.strftime("%Y-%m-%d"), "signal": sig.get("signal", "n/a"),
                   "p_high_vol": f"{sig['p']:.3f}" if sig else "", "vix": f"{sig['vix']:.2f}" if sig else "",
                   "dte": int(dte), "lot_size": int(lot), "spot_entry": spot_e or "",
                   "entry_legs": json.dumps(ll), "entry_time": now.strftime("%H:%M"), "legs_text": legs_text(ll),
                   "status": "OPEN", "notes": notes}
            try:
                store.add(row)
                st.success(f"Saved ✅  trade id {trade_id}. Go to the Exit tab when you close it.")
            except Exception as e:
                st.error(f"Could not save to the sheet: {e}")

# ---------------------------- tab 3: exit trade ----------------------------
with tab_out:
    if st.button("Refresh open trades", width="stretch"):
        st.session_state.pop("rows_cache", None)
    try:
        if "rows_cache" not in st.session_state:
            st.session_state["rows_cache"] = store.all()
        rows = st.session_state["rows_cache"]
    except Exception as e:
        rows = []
        st.error(f"Could not read the sheet: {e}")
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
            gox = st.form_submit_button("Save exit", type="primary", width="stretch")
        if gox:
            if edited["ExitPrice"].isna().any():
                st.error("Fill the exit price for every leg (use 0 if it expired worthless).")
            else:
                legs_x = edited.to_dict("records")
                gross = pnl_rs(legs_x, int(lot))
                net = gross - charges
                try:
                    store.close(pick, {"exit_time": datetime.now(IST).strftime("%H:%M"), "spot_exit": spot_x or "",
                                       "exit_legs": json.dumps(legs_x), "legs_text": legs_text(legs_x),
                                       "gross_pnl_rs": round(gross, 2), "charges_rs": round(charges, 2),
                                       "net_pnl_rs": round(net, 2), "status": "CLOSED"})
                    st.session_state.pop("rows_cache", None)
                    (st.success if net >= 0 else st.error)(f"Saved ✅  gross ₹{gross:,.0f}  ·  net after charges ₹{net:,.0f}")
                except Exception as e:
                    st.error(f"Could not save to the sheet: {e}")

# ---------------------------- tab 4: review ----------------------------
with tab_rev:
    try:
        rows = store.all()
    except Exception as e:
        rows, _ = [], st.error(f"Could not read the sheet: {e}")
    allr = pd.DataFrame(rows)
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
            st.line_chart(cl.set_index(pd.to_datetime(cl["date"]) + pd.to_timedelta(range(len(cl)), unit="s"))
                          ["net_pnl_rs"].cumsum().rename("Cumulative net ₹"))
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

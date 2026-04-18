"""Streamlit web app — public dashboard for the TQQQ journey.

Deploy on Streamlit Community Cloud by connecting this repo. The app reads
directly from the JSON/CSV files in data/ — no database, no auth, all public.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

# Inline the lib helpers (simpler for Streamlit deploy than module imports)
def _N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(0.0, S - K)
    if sigma <= 0:
        return max(0.0, S - K * math.exp(-r * T))
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _N(d1) - K * math.exp(-r * T) * _N(d2)


def bs_spread(S, K_low, K_high, T, r=0.04, sigma=0.60, skew=0.90):
    return bs_call(S, K_low, T, r, sigma) - bs_call(S, K_high, T, r, sigma * skew)


# =====================================================================
# Config + data loading
# =====================================================================
DATA = Path(__file__).parent / "data"


@st.cache_data(ttl=30)
def load_config():
    return json.loads((DATA / "config.json").read_text())


@st.cache_data(ttl=30)
def load_positions():
    return json.loads((DATA / "positions.json").read_text())["positions"]


@st.cache_data(ttl=30)
def load_trades():
    return json.loads((DATA / "trades.json").read_text())["trades"]


@st.cache_data(ttl=30)
def load_equity_history():
    df = pd.read_csv(DATA / "equity_history.csv")
    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=900)  # refresh every 15 min during market hours
def get_current_tqqq_iv():
    try:
        tqqq = yf.download("TQQQ", period="3mo", auto_adjust=True, progress=False)
        if isinstance(tqqq.columns, pd.MultiIndex):
            tqqq.columns = tqqq.columns.get_level_values(0)
        if len(tqqq) == 0:
            return None, None, None
        tqqq_close = float(tqqq["Close"].iloc[-1])
        last_date = tqqq.index[-1].date()
        import numpy as np
        lr = np.log(tqqq["Close"] / tqqq["Close"].shift(1)).dropna()
        rv = float(lr.tail(60).std() * math.sqrt(252))
        iv = max(0.30, min(1.5, rv * 1.1))
        return tqqq_close, iv, last_date
    except Exception:
        return None, None, None


# =====================================================================
# Page config
# =====================================================================
config = load_config()
st.set_page_config(
    page_title=f"{config['owner']}'s TQQQ Journey",
    page_icon="[chart]",
    layout="wide",
)


# =====================================================================
# Sidebar navigation
# =====================================================================
st.sidebar.title(f"{config['owner']}'s Journey")
st.sidebar.caption(config["strategy_name"])

tqqq_now, iv_now, tqqq_date = get_current_tqqq_iv()
if tqqq_now:
    st.sidebar.metric("TQQQ", f"${tqqq_now:.2f}", help=f"Last close: {tqqq_date}")
    st.sidebar.caption(f"Est IV: {iv_now*100:.0f}%")

page = st.sidebar.radio(
    "View",
    ["Home", "Current Positions", "Trade Log", "Equity Curve", "Strategy"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption(
    f"Target: ${config['target_equity']:,.0f} by end of {config['target_year']}"
)
st.sidebar.caption(f"Starting capital: ${config['starting_capital']:,.0f}")
if config.get("weekly_contribution"):
    st.sidebar.caption(f"Adding ${config['weekly_contribution']}/week")


# =====================================================================
# Helper: compute current portfolio state from positions
# =====================================================================
def compute_portfolio_state(positions, tqqq_current, iv):
    """Returns (total_open_value, total_unrealized_pnl, total_realized_pnl, positions_enriched)."""
    today = date.today()
    enriched = []
    total_open = 0.0
    unrealized = 0.0
    realized = 0.0
    for p in positions:
        p = dict(p)  # copy
        contracts = int(p["contracts"])
        cost_basis = float(p["total_cost"])
        if p.get("status") == "open" and tqqq_current:
            expiry = datetime.strptime(p["expiry"], "%Y-%m-%d").date()
            T = max(0.001, (expiry - today).days / 365.25)
            val_per_share = bs_spread(
                tqqq_current, float(p["long_strike"]),
                float(p["short_strike"]), T, sigma=iv,
            )
            current_value = val_per_share * 100 * contracts
            p["current_value"] = current_value
            p["unrealized_pnl"] = current_value - cost_basis
            p["unrealized_pct"] = (current_value / cost_basis - 1) * 100 if cost_basis > 0 else 0
            p["days_to_expiry"] = (expiry - today).days
            total_open += current_value
            unrealized += p["unrealized_pnl"]
        elif p.get("status") in ("closed", "expired", "rolled"):
            realized += float(p.get("realized_pnl", 0))
        enriched.append(p)
    return total_open, unrealized, realized, enriched


# =====================================================================
# Page: HOME
# =====================================================================
if page == "Home":
    st.title(f"{config['owner']}'s TQQQ Journey")
    st.subheader(config["strategy_name"])
    st.write(config.get("owner_intro", ""))

    st.divider()
    positions = load_positions()
    total_open, unrealized, realized, _ = compute_portfolio_state(
        positions, tqqq_now, iv_now,
    )

    eq_df = load_equity_history()
    if len(eq_df) > 0:
        latest_eq = float(eq_df["total_equity"].iloc[-1])
        invested = float(eq_df["invested_to_date"].iloc[-1])
    else:
        latest_eq = config["starting_capital"]
        invested = config["starting_capital"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current equity", f"${latest_eq:,.0f}",
                f"{(latest_eq/config['starting_capital']-1)*100:+.1f}% vs start")
    col2.metric("Total invested", f"${invested:,.0f}",
                help="Starting cap + weekly contributions to date")
    col3.metric("Unrealized P&L", f"${unrealized:+,.0f}",
                help="Mark-to-market on open positions (BS-estimated)")
    col4.metric("Realized P&L", f"${realized:+,.0f}",
                help="Closed/expired/rolled positions, cumulative")

    # Progress to target
    target = config["target_equity"]
    pct_to_target = latest_eq / target * 100
    st.progress(min(1.0, latest_eq / target),
                text=f"${latest_eq:,.0f} / ${target:,.0f} target by {config['target_year']} "
                     f"({pct_to_target:.1f}%)")

    st.divider()
    st.markdown("### The strategy in one paragraph")
    st.write(config["strategy_summary"])

    st.markdown("### Honest caveats")
    st.markdown("""
- **This is a leveraged options strategy.** Historical backtest shows 60-90% drawdowns
  during bear markets. The commitment is to hold through them.
- **Backtest numbers are modeled, not guaranteed.** Real-world execution (bid/ask, IV
  skew, liquidity) typically haircuts returns 3-5% CAGR vs. the idealized model.
- **This is not investment advice.** It's one person's journey. Following along is
  informational; make your own decisions.
""")


# =====================================================================
# Page: CURRENT POSITIONS
# =====================================================================
elif page == "Current Positions":
    st.title("Current Positions")
    positions = load_positions()
    open_pos = [p for p in positions if p.get("status") == "open"]

    total_open, unrealized, realized, enriched = compute_portfolio_state(
        positions, tqqq_now, iv_now,
    )
    enriched_open = [p for p in enriched if p.get("status") == "open"]

    if not enriched_open:
        st.info("No open positions yet. The first tranche will appear here after it fills.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Open positions", len(enriched_open))
        col2.metric("Total open value", f"${total_open:,.0f}",
                    f"${unrealized:+,.0f} unrealized")
        total_cost = sum(float(p["total_cost"]) for p in enriched_open)
        col3.metric("Total cost basis", f"${total_cost:,.0f}")

        st.divider()
        for p in sorted(enriched_open, key=lambda x: x["expiry"]):
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 2, 1])
                c1.markdown(
                    f"**TQQQ {p['long_strike']:.0f} / {p['short_strike']:.0f} "
                    f"bull call spread** — expires {p['expiry']}"
                )
                c1.caption(
                    f"{p['contracts']} contracts at entry. Opened {p['entry_date']} "
                    f"when TQQQ was ${p['tqqq_at_entry']:.2f}."
                )
                c2.metric("Current value", f"${p.get('current_value', 0):,.0f}",
                          f"{p.get('unrealized_pct', 0):+.1f}% vs cost")
                c2.caption(f"Cost basis: ${p['total_cost']:,.0f}  |  "
                          f"Debit at entry: ${p['net_debit_per_share']:.2f}/share")
                c3.metric("Days to expiry", f"{p.get('days_to_expiry', 0)}",
                          help="Will roll at ~365 days remaining")
                c3.caption(f"Breakeven: TQQQ ${p['breakeven']:.2f}")
                if p.get("note"):
                    st.caption(f"Note: {p['note']}")


# =====================================================================
# Page: TRADE LOG
# =====================================================================
elif page == "Trade Log":
    st.title("Trade Log")
    trades = load_trades()
    if not trades:
        st.info("No trades logged yet.")
    else:
        df = pd.DataFrame(trades)
        df = df.sort_values("date", ascending=False)
        # Flatten details for display
        display_rows = []
        for t in trades:
            row = {"date": t["date"], "event": t["event"]}
            if t["event"] == "open":
                d = t.get("details", {})
                row.update({
                    "underlying": d.get("underlying", ""),
                    "strikes": f"{d.get('long_strike', '?')}/{d.get('short_strike', '?')}",
                    "expiry": d.get("expiry", ""),
                    "contracts": d.get("contracts", ""),
                    "debit/share": f"${d.get('net_debit_per_share', 0):.2f}",
                    "cost": f"${d.get('total_cost', 0):,.0f}",
                    "pnl": "",
                    "note": d.get("note", ""),
                })
            else:
                row.update({
                    "underlying": "TQQQ",
                    "strikes": "",
                    "expiry": "",
                    "contracts": "",
                    "debit/share": "",
                    "cost": "",
                    "pnl": f"${t.get('realized_pnl', 0):+,.0f} "
                           f"({t.get('realized_pnl_pct', 0):+.1f}%)",
                    "note": t.get("note", ""),
                })
            display_rows.append(row)
        display_df = pd.DataFrame(display_rows).sort_values("date", ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# =====================================================================
# Page: EQUITY CURVE
# =====================================================================
elif page == "Equity Curve":
    st.title("Equity Curve")
    eq_df = load_equity_history()
    if len(eq_df) < 2:
        st.info("Need at least 2 days of history to plot. Check back tomorrow.")
    else:
        import altair as alt
        chart_data = eq_df[["date", "total_equity", "invested_to_date"]].copy()
        chart_data = chart_data.melt("date", var_name="series", value_name="value")
        chart = alt.Chart(chart_data).mark_line().encode(
            x="date:T",
            y=alt.Y("value:Q", title="$"),
            color="series:N",
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)

        # Summary stats
        latest = eq_df.iloc[-1]
        first = eq_df.iloc[0]
        days = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days
        col1, col2, col3 = st.columns(3)
        col1.metric("Latest equity", f"${latest['total_equity']:,.0f}")
        col2.metric("Net P&L",
                    f"${latest['total_equity'] - latest['invested_to_date']:+,.0f}")
        col3.metric("Days tracked", days)


# =====================================================================
# Page: STRATEGY
# =====================================================================
elif page == "Strategy":
    st.title("The Strategy")
    st.subheader(config["strategy_name"])
    st.markdown(f"""
### What I'm doing

{config['strategy_summary']}

### The rules

- **Instrument**: TQQQ (3x leveraged Nasdaq-100 ETF) bull call spreads
- **Long leg**: Strike at ~70% of TQQQ spot (~30% ITM)
- **Short leg**: Strike at ~159% of TQQQ spot (~59% OTM)
- **Spread width**: ~89% of spot
- **Expiry**: ~24 months out (longest available LEAP cycle)
- **Roll**: Each spread is held for 13 months (rolled when 11 months remain) — long-term capital gains treatment
- **Cadence**: One new spread per month, funded by matured rolls and ongoing contributions
- **Sizing**: ~95% of available cash per monthly entry (5% reserve)
- **No stops, no profit-taking early, no filters.** Diamond hands through drawdowns.

### Why this structure

Selected from parameter robustness testing across 4 regime samples (real TQQQ 2010-26,
synthetic 3x-leveraged QQQ 2000-26, isolated bull + bear periods). The 70%/+89%
configuration tied or outperformed every alternative tested in bull regimes (Sharpe
~0.97-1.01) with meaningfully better bear-regime resilience than narrower spreads.
Upside uncapped to 159% of entry spot — aligns with a bull thesis.

Caveat: Sharpe differences within a cluster of near-equivalent configurations are
within bootstrap 90% confidence intervals (~0.5-1.3). The choice is "best of several
near-equivalents with favorable regime stability," not "statistically proven optimum."

### What could go wrong

- **Prolonged bear market.** A 2000-2002 style 3-year Nasdaq collapse would punch
  through 24-month spread expiries. The backtest window didn't include that scenario.
- **TQQQ structural risk.** ProShares fee increases, regulatory changes, or fund
  closure could impair the strategy without warning.
- **My own discipline.** The strategy requires holding through 60-90% drawdowns.
  If I panic-close during a crash, I lock in the worst outcome.
- **Model optimism.** My backtest uses Black-Scholes idealized pricing. Real options
  have bid/ask spreads, IV skew, and liquidity constraints that haircut returns.

### Not investment advice

I'm not a financial advisor. This site documents what I'm doing. It's not a
recommendation for what you should do. Your situation, risk tolerance, and
financial goals are your own.
""")


st.sidebar.divider()
st.sidebar.caption("Not investment advice. Trade history is public & append-only.")

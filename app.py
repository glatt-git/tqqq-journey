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

# Portfolio metric — current equity with dollar & percent change vs starting
_eq_sidebar = load_equity_history()
_starting = float(config["starting_capital"])
if len(_eq_sidebar) > 0:
    _current_equity = float(_eq_sidebar["total_equity"].iloc[-1])
else:
    _current_equity = _starting
_dollar_change = _current_equity - _starting
_pct_change = (_current_equity / _starting - 1) * 100 if _starting > 0 else 0.0
_dollar_str = f"+${_dollar_change:,.0f}" if _dollar_change >= 0 else f"-${abs(_dollar_change):,.0f}"
st.sidebar.metric(
    "Portfolio",
    f"${_current_equity:,.0f}",
    f"{_pct_change:+.2f}%",
)
st.sidebar.caption(f"{_dollar_str} since start")

page = st.sidebar.radio(
    "View",
    ["Home", "Strategy", "Thesis", "Backtest", "Positions", "Trade Log", "Equity Curve"],
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
# Page: THESIS
# =====================================================================
elif page == "Thesis":
    import streamlit.components.v1 as components
    thesis_path = Path(__file__).parent / "thesis.html"
    if thesis_path.exists():
        thesis_html = thesis_path.read_text(encoding="utf-8")
        # Render at full width; height = viewport-ish with internal scroll
        components.html(thesis_html, height=2400, scrolling=True)
    else:
        st.warning("Thesis document not found in repo. Expected at `thesis.html`.")


# =====================================================================
# Page: BACKTEST
# =====================================================================
elif page == "Backtest":
    import altair as alt

    BT = Path(__file__).parent / "data" / "backtest"

    @st.cache_data(ttl=300)
    def _load_backtest_summary():
        return json.loads((BT / "summary.json").read_text())

    @st.cache_data(ttl=300)
    def _load_equity_curves():
        df = pd.read_csv(BT / "equity_curves.csv", parse_dates=["date"])
        return df

    @st.cache_data(ttl=300)
    def _load_rolling():
        df = pd.read_csv(BT / "rolling_4yr.csv", parse_dates=["start"])
        return df

    @st.cache_data(ttl=300)
    def _load_stress():
        df = pd.read_csv(BT / "stress_test.csv", parse_dates=["date"])
        return df

    if not (BT / "summary.json").exists():
        st.warning("Backtest data not found. Run `python scripts/precompute_backtest.py` to generate it.")
    else:
        summary = _load_backtest_summary()
        curves = _load_equity_curves()
        rolling = _load_rolling()
        stress = _load_stress()

        st.title("Backtest Results")
        st.caption(
            f"Backtest period: {summary['period']['start']} to {summary['period']['end']} "
            f"({summary['period']['years']} years on real TQQQ data)"
        )

        # -----------------------------------------------------------------
        # Hero metrics — two clearly-labeled headline numbers
        # -----------------------------------------------------------------
        st.markdown("### Two ways to read the result")
        col1, col2 = st.columns(2)
        with col1:
            h1 = summary["headline"]["realistic_rolling_4yr_median"]
            st.markdown("#### Realistic 4-year outcome")
            st.metric(
                label="Median of 147 rolling 4-year windows",
                value=f"${h1['strategy_final_median']:,.0f}",
                delta=f"{h1['strategy_multiple']}x invested",
            )
            st.caption(
                f"$50k lump + $500/week DCA = ${h1['total_invested']:,} invested per window. "
                f"This is how the strategy will actually execute. QQQ BH median over same windows: "
                f"${h1['qqq_bh_final_median']:,.0f}."
            )
        with col2:
            h2 = summary["headline"]["single_path_16yr"]
            st.markdown("#### Theoretical 16-year maximum")
            st.metric(
                label="$50k lump sum, no contributions, 16 years on real TQQQ",
                value=f"${h2['final']:,.0f}",
                delta=f"{h2['multiple']}x starting",
            )
            st.caption(
                "Single-path backtest with all capital deployed at start, compounding across 16 years. "
                "Idealized; real-world execution at this scale would be materially constrained by "
                "options market liquidity."
            )

        st.divider()

        # -----------------------------------------------------------------
        # Chart A — Strategy DCA vs QQQ BH DCA vs TQQQ BH DCA
        # -----------------------------------------------------------------
        st.markdown("### Strategy vs. buy-and-hold alternatives (equal contributions)")
        st.markdown(
            "Same $50k starting capital, same $500/week DCA, same 16-year window. "
            "The strategy (red) leverages equity exposure via TQQQ LEAP spreads; "
            "QQQ BH (blue) is unleveraged Nasdaq-100; TQQQ BH (orange) is the raw 3x ETF "
            "held continuously."
        )
        primary = curves[["date", "strategy", "qqq_bh", "tqqq_bh"]].melt(
            "date", var_name="series", value_name="equity"
        )
        name_map = {
            "strategy": "Strategy (70/+89 DCA)",
            "qqq_bh": "QQQ buy-and-hold + DCA",
            "tqqq_bh": "TQQQ buy-and-hold + DCA",
        }
        primary["Strategy"] = primary["series"].map(name_map)
        chart_a = alt.Chart(primary).mark_line(strokeWidth=2).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("equity:Q", title="Equity ($)", scale=alt.Scale(type="log")),
            color=alt.Color(
                "Strategy:N",
                scale=alt.Scale(
                    domain=[name_map["strategy"], name_map["qqq_bh"], name_map["tqqq_bh"]],
                    range=["#8b2c2c", "#2c5f8b", "#d4852c"],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        ).properties(height=380)
        st.altair_chart(chart_a, use_container_width=True)

        # -----------------------------------------------------------------
        # Chart B — Rolling 4-year outcome distribution
        # -----------------------------------------------------------------
        st.markdown("### Distribution of 4-year outcomes")
        st.markdown(
            "Each dot is a 4-year window starting a different month between 2010 and 2022. "
            "147 windows total. For each window: $50k start + $500/week for 4 years, then measure final equity. "
            "Shows the strategy's **expected variance** rather than a single lucky/unlucky path."
        )
        dist_data = []
        for _, row in rolling.iterrows():
            dist_data.append({"Start": row["start"], "Final": row["strategy_final"], "Strategy": "Strategy"})
            dist_data.append({"Start": row["start"], "Final": row["qqq_bh_final"], "Strategy": "QQQ buy-and-hold"})
        dist_df = pd.DataFrame(dist_data)
        chart_b = alt.Chart(dist_df).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X("Start:T", title="Window start date"),
            y=alt.Y("Final:Q", title="Final equity ($)", scale=alt.Scale(type="log")),
            color=alt.Color(
                "Strategy:N",
                scale=alt.Scale(domain=["Strategy", "QQQ buy-and-hold"], range=["#8b2c2c", "#2c5f8b"]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["Start:T", "Strategy:N", alt.Tooltip("Final:Q", format="$,.0f")],
        ).properties(height=360)
        st.altair_chart(chart_b, use_container_width=True)

        # Distribution stats
        strat_vals = rolling["strategy_final"]
        qqq_vals = rolling["qqq_bh_final"]
        pct_beat = (rolling["strategy_final"] > rolling["qqq_bh_final"]).sum() / len(rolling) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Strategy median", f"${strat_vals.median():,.0f}",
                  f"{strat_vals.median()/qqq_vals.median():.1f}x vs QQQ BH median")
        c2.metric("Strategy 10th pct", f"${strat_vals.quantile(0.1):,.0f}",
                  f"${strat_vals.quantile(0.9):,.0f} 90th pct")
        c3.metric("% of windows where strategy beats QQQ BH", f"{pct_beat:.0f}%")

        st.divider()

        # -----------------------------------------------------------------
        # Chart C — Comprehensive benchmark comparison
        # -----------------------------------------------------------------
        st.markdown("### Full benchmark set")
        st.markdown(
            "Strategy vs. every reasonable passive alternative on the same starting capital and "
            "contribution schedule. Included: TQQQ, QQQ, SPY, GLD, and 60/40 (SPY+AGG)."
        )
        full = curves.melt("date", var_name="series", value_name="equity")
        full_map = {
            "strategy": "Strategy",
            "qqq_bh": "QQQ", "tqqq_bh": "TQQQ",
            "spy_bh": "SPY", "gld_bh": "GLD", "blend_6040": "60/40 SPY+AGG",
        }
        full["Strategy"] = full["series"].map(full_map)
        full = full.dropna(subset=["equity"])
        chart_c = alt.Chart(full).mark_line(strokeWidth=1.5).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("equity:Q", title="Equity ($)", scale=alt.Scale(type="log")),
            color=alt.Color("Strategy:N", legend=alt.Legend(orient="top", title=None),
                            scale=alt.Scale(
                                domain=["Strategy", "TQQQ", "QQQ", "SPY", "GLD", "60/40 SPY+AGG"],
                                range=["#8b2c2c", "#d4852c", "#2c5f8b", "#6b6b6b", "#c4a84a", "#5a7a5a"],
                            )),
        ).properties(height=420)
        st.altair_chart(chart_c, use_container_width=True)

        st.divider()

        # -----------------------------------------------------------------
        # Chart D — Stress test (2000-2010 synthetic regime)
        # -----------------------------------------------------------------
        st.markdown("### Stress test: dot-com + 2008 GFC")
        st.markdown(
            "TQQQ didn't exist before 2010. This chart uses a **synthetic 3x-leveraged QQQ** "
            "(daily rebalanced, ~0.95% fee + swap-spread drag, floored at -99% daily) to simulate "
            "what the strategy would have done through the worst historical 10-year period for "
            "leveraged long-equity: dot-com crash + 2008 financial crisis. $10k start + $500/week."
        )
        ss = summary["stress_2000_2010"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Invested", f"${ss['total_invested']:,.0f}")
        c2.metric("Final", f"${ss['final']:,.0f}",
                  f"{ss['pct_of_invested']:.0f}% of invested")
        c3.metric("Net loss", f"{100 - ss['pct_of_invested']:.0f}%")
        stress_m = stress.melt("date", var_name="series", value_name="equity")
        stress_map = {"strategy": "Strategy (synthetic)", "qqq_bh": "QQQ buy-and-hold"}
        stress_m["Strategy"] = stress_m["series"].map(stress_map)
        chart_d = alt.Chart(stress_m).mark_line(strokeWidth=2).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("equity:Q", title="Equity ($)"),
            color=alt.Color(
                "Strategy:N",
                scale=alt.Scale(domain=list(stress_map.values()), range=["#8b2c2c", "#2c5f8b"]),
                legend=alt.Legend(orient="top", title=None),
            ),
        ).properties(height=340)
        st.altair_chart(chart_d, use_container_width=True)

        st.info(
            "**Honest takeaway**: In a 2000-style regime, the strategy loses ~17% of total "
            "contributions while QQQ buy-and-hold roughly breaks even. DCA converts the crash "
            "into accumulation opportunities, but the leveraged option premiums still cost money "
            "on positions that roll through falling markets. If you expect this kind of regime, "
            "this strategy is the wrong choice."
        )

        st.divider()

        # -----------------------------------------------------------------
        # Risk-adjusted metrics table
        # -----------------------------------------------------------------
        st.markdown("### Risk-adjusted metrics (2010-2026, equal contributions)")
        ra = summary["risk_adjusted_comparison"]
        display_names = {
            "strategy": "Strategy (70/+89)", "qqq_bh": "QQQ buy-and-hold",
            "tqqq_bh": "TQQQ buy-and-hold", "spy_bh": "SPY buy-and-hold",
            "gld_bh": "GLD buy-and-hold", "blend_6040": "60/40 SPY+AGG",
        }
        ra_rows = []
        for key, name in display_names.items():
            m = ra.get(key)
            if m is None: continue
            ra_rows.append({
                "Portfolio": name,
                "CAGR %": f"{m['cagr']:.2f}",
                "Max DD %": f"{m['maxdd']:.2f}",
                "Sharpe": f"{m['sharpe']:.2f}",
                "Sortino": f"{m['sortino']:.2f}",
                "Calmar": f"{m['calmar']:.2f}",
                "Final $": f"${m['final']:,.0f}",
            })
        st.dataframe(pd.DataFrame(ra_rows), use_container_width=True, hide_index=True)
        st.caption(
            "The strategy beats every benchmark on absolute return and CAGR. On risk-adjusted "
            "metrics, it's competitive with TQQQ BH but not better than QQQ BH — the strategy's "
            "edge is leverage magnitude, not Sharpe. Max DD is severe by design; the trade-off "
            "for the return."
        )

        st.divider()

        # -----------------------------------------------------------------
        # Expandable: Methodology
        # -----------------------------------------------------------------
        with st.expander("Methodology"):
            cfg = summary["strategy_config"]
            st.markdown(f"""
**Option pricing**: Black-Scholes with implied volatility estimated as
{cfg['iv_model']}. Risk-free rate held at {cfg['risk_free_rate']*100:.0f}% (approximately
the 2010-2026 T-bill average).

**Strategy parameters** ([see Strategy page for the locked spec](./Strategy)):
- Long leg: {cfg['long_pct']*100:.0f}% of TQQQ spot (30% in-the-money)
- Short leg: {cfg['short_pct']*100:.0f}% of spot (59% out-of-the-money)
- Spread width: {cfg['width_pct']*100:.0f}% of spot
- Option duration at entry: {cfg['option_duration_months']} months
- Hold period before roll: {cfg['roll_hold_months']} months (long-term capital gains)

**Execution modeling**:
- Combined bid/ask slippage: {cfg['commission_slip_pct']}% of net debit at entry
- IV skew modeled by pricing the short leg at 90% of the long leg's IV
- Monthly entry cadence (4-week cycle)
- No filters (no IV or trend-based skip rules)

**Capital model**:
- Starting capital: ${summary['starting_capital']:,}
- Weekly DCA contribution: ${summary['weekly_contribution']}
- Cash reserve: 5% of available cash at each entry
- Minimum position size: 1 contract (skip and accumulate if insufficient)

**Data sources**:
- TQQQ: yfinance adjusted close, 2010-02-11 through 2026-04-17
- QQQ, SPY, GLD, AGG: yfinance adjusted close
- Synthetic 3x-leveraged QQQ (for pre-2010 regime): daily rebalanced from QQQ,
  0.95% annual fee, 0.5% swap financing spread, daily return floored at -99%

**What this model does not capture**:
- IV term structure (assumes flat IV surface)
- Full volatility skew (modeled only as a ratio)
- Liquidity constraints at scale (bid/ask widens during crises)
- Market impact of a large position relative to open interest
- Sequence-dependent fill quality
- Tax drag on rolls (pre-tax results shown)

Realistic execution is likely to produce 3-5 percentage points lower CAGR than the
model shows, with wider drawdowns during illiquid stress periods.
""")

        # -----------------------------------------------------------------
        # Expandable: Parameter robustness
        # -----------------------------------------------------------------
        with st.expander("Parameter robustness"):
            pr = summary["parameter_robustness_sharpe"]
            st.markdown(f"""
**Bootstrap Sharpe 90% CI for the production configuration (70/+89)**: `{pr['bootstrap_ci_90pct']}`

**Point-estimate Sharpes for neighboring configurations** (real TQQQ 2010-2026):
""")
            for k, v in pr["neighbors_point_sharpe"].items():
                is_prod = (k == pr["production_config"])
                marker = " ← production" if is_prod else ""
                st.markdown(f"- **{k}**: Sharpe {v}{marker}")
            st.markdown(f"\n{pr['interpretation']}")
            st.markdown("""
The differences between these configurations are within the bootstrap confidence
interval width (roughly 0.8 Sharpe points). Statistically, they are not
distinguishable. The 70/+89 selection is based on cross-regime stability: it tied
or exceeded every alternative in both the 2010-2026 bull regime and the synthetic
2000-2010 bear regime.

**What this means**: Think of the Sharpe for this strategy as being in the range
[0.5, 1.3] with 90% confidence, with a point estimate of 1.0. If your live Sharpe
lands anywhere in that range, the strategy is performing as modeled. If it lands
materially outside, something has changed (regime, liquidity, or your execution).
""")

        # -----------------------------------------------------------------
        # Expandable: Honest caveats
        # -----------------------------------------------------------------
        with st.expander("Honest caveats"):
            st.markdown("""
**Sample size**: The real TQQQ data spans ~16 years, about 192 monthly entries.
The standard error on a Sharpe estimate with ~200 observations is roughly 1/√200 ≈ 0.07.
Differences smaller than ~0.15 Sharpe between configurations are not statistically
distinguishable.

**Regime dependence**: TQQQ's entire existence (2010-present) has been a
historically favorable period for leveraged long-equity. The 2000-2010 synthetic
stress test is my best attempt to see how the strategy would behave in a
materially different regime, but it's a model, not a real track record.

**Survivorship in the underlying**: The Nasdaq-100 composition has changed
dramatically over 16 years. The 70/+89 strategy's performance is partly a story
about superstar firm concentration continuing to work. If the current Magnificent-7
era ends and the next decade's leaders haven't emerged yet, results will be different.

**Execution at scale**: Every dollar of the backtest assumes model-perfect
execution at mid-price. At $50k account size, this is roughly accurate. At
$10M+, you're a meaningful fraction of TQQQ LEAP open interest and real
execution costs will be materially higher than the 3% slippage modeled.

**Tax treatment not modeled**: Results are pre-tax. Real after-tax returns depend on
holding period compliance (13-month roll for long-term treatment), state residency,
and year-to-year realization timing. Factor 15-25% haircut on realized gains for
federal plus state on the long-term portion.

**What I'm not showing**: Variants that underperformed in this sweep. V4 (max 4
units), V5 (symmetric bias all markets), V9 (no 6% risk cap), IV filters, trend
filters. All tested; all underperformed the current structure in at least one
regime. Asymmetric reporting is not conservative reporting.
""")

        st.caption(
            "Source code for all backtests: "
            "[github.com/glatt-git/tqqq-journey/scripts](https://github.com/glatt-git/tqqq-journey/tree/main/scripts) "
            "— `precompute_backtest.py` regenerates every number on this page."
        )


# =====================================================================
# Page: POSITIONS
# =====================================================================
elif page == "Positions":
    st.title("Positions")
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

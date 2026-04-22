"""Daily mark-to-market update. Pulls TQQQ close, revalues open positions, appends equity row.

Run manually or via GitHub Action cron (see .github/workflows/daily.yml).

Data source strategy: try yfinance first (richer; historical quirks), fall back to the
Nasdaq public historical endpoint (no auth, no rate limit, reliable from GitHub runner IPs).
If both fail, exit 0 quietly so the cron doesn't keep emailing failure notices on
transient upstream issues.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
import urllib.request
from datetime import date as date_cls, timedelta

import pandas as pd

from lib import (
    append_equity_row,
    estimate_iv,
    git_commit_and_push,
    load_config,
    load_equity_history,
    load_positions,
    value_position,
)


def fetch_tqqq_yfinance(retries: int = 2) -> pd.DataFrame | None:
    """Try yfinance with a short retry. Returns DataFrame or None on failure."""
    try:
        import yfinance as yf
    except Exception as e:
        print(f"  yfinance import failed: {e}")
        return None
    for attempt in range(retries + 1):
        try:
            df = yf.download("TQQQ", period="6mo", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 0 and "Close" in df.columns:
                print(f"  yfinance: got {len(df)} rows")
                return df
            print(f"  yfinance attempt {attempt+1}: empty frame")
        except Exception as e:
            print(f"  yfinance attempt {attempt+1} failed: {type(e).__name__}: {e}")
        if attempt < retries:
            time.sleep(2 ** attempt)
    return None


def fetch_tqqq_nasdaq() -> pd.DataFrame | None:
    """Fallback: Nasdaq public historical endpoint. No auth, JSON response."""
    try:
        today = date_cls.today()
        start = today - timedelta(days=200)
        url = (
            "https://api.nasdaq.com/api/quote/TQQQ/historical"
            f"?assetclass=etf&fromdate={start.isoformat()}&todate={today.isoformat()}&limit=250"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8")
        payload = json.loads(body)
        rows = (((payload.get("data") or {}).get("tradesTable") or {}).get("rows")) or []
        if not rows:
            print("  nasdaq: no rows returned")
            return None
        records = []
        for r in rows:
            try:
                records.append({
                    "Date": pd.to_datetime(r["date"]),
                    "Close": float(str(r["close"]).replace(",", "").replace("$", "")),
                })
            except (KeyError, ValueError):
                continue
        if not records:
            print("  nasdaq: rows malformed")
            return None
        df = pd.DataFrame(records).set_index("Date").sort_index()
        print(f"  nasdaq: got {len(df)} rows")
        return df
    except Exception as e:
        print(f"  nasdaq failed: {type(e).__name__}: {e}")
        return None


def fetch_tqqq() -> pd.DataFrame | None:
    df = fetch_tqqq_yfinance()
    if df is not None and len(df) > 0:
        return df
    print("  yfinance unavailable; falling back to Nasdaq public API...")
    return fetch_tqqq_nasdaq()


def main():
    print(f"Daily update {date_cls.today()}")
    config = load_config()
    starting_cap = config["starting_capital"]

    # Pull TQQQ recent prices for close + IV estimate
    tqqq = fetch_tqqq()
    if tqqq is None or len(tqqq) == 0:
        print("  All data sources failed. Exiting 0 to avoid spurious failure emails.")
        sys.exit(0)
    tqqq_close = float(tqqq["Close"].iloc[-1])
    last_date = tqqq.index[-1].date()
    iv = estimate_iv(tqqq["Close"])
    print(f"  TQQQ close: ${tqqq_close:.2f} on {last_date}")
    print(f"  Estimated IV: {iv*100:.1f}%")

    # Revalue open positions
    positions = load_positions()["positions"]
    open_positions = [p for p in positions if p.get("status") == "open"]
    total_open_value = 0.0
    for p in open_positions:
        val = value_position(p, tqqq_close, iv)
        total_open_value += val["total_value"]
        print(f"  {p['id']}: cost ${val['cost_basis']:,.0f} -> value ${val['total_value']:,.0f} "
              f"({val['unrealized_pct']:+.1f}%)")

    # Compute invested to date = starting + contributions
    today = date_cls.today()
    # Estimate contributions: $500/week since account start (use earliest trade date or config)
    # For simplicity use first position entry date if present, else skip
    if positions:
        start_dt = min(p["entry_date"] for p in positions)
        start_dt = date_cls.fromisoformat(start_dt)
        weeks_elapsed = max(0, (today - start_dt).days / 7)
        contribs = config["weekly_contribution"] * weeks_elapsed
    else:
        contribs = 0
    invested = starting_cap + contribs

    # Cash = invested - cumulative entry costs + cumulative exit proceeds
    total_entry_cost = sum(float(p["total_cost"]) for p in positions)
    total_exit_value = sum(float(p.get("gross_exit_value", 0.0)) for p in positions)
    cash = invested - total_entry_cost + total_exit_value
    total_equity = cash + total_open_value
    print(f"  Invested: ${invested:,.2f}  Cash: ${cash:,.2f}  Open value: ${total_open_value:,.2f}  "
          f"Total equity: ${total_equity:,.2f}")

    # Append to equity history (replace row if already today)
    eq_df = load_equity_history()
    eq_df = eq_df[eq_df["date"] != pd.Timestamp(last_date)]
    row = {
        "date": last_date.isoformat(),
        "tqqq_close": round(tqqq_close, 2),
        "cash": round(cash, 2),
        "open_spread_value": round(total_open_value, 2),
        "total_equity": round(total_equity, 2),
        "invested_to_date": round(invested, 2),
    }
    eq_df = pd.concat([eq_df, pd.DataFrame([row])], ignore_index=True)
    eq_df = eq_df.sort_values("date").reset_index(drop=True)
    from lib import EQUITY_FILE
    eq_df.to_csv(EQUITY_FILE, index=False)
    print("  Equity row appended.")

    # Commit + push
    ok = git_commit_and_push(f"Daily MTM: {last_date} total equity ${total_equity:,.0f}")
    if ok:
        print("  Pushed.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("!!! Daily update crashed — full traceback below !!!")
        traceback.print_exc()
        sys.exit(1)

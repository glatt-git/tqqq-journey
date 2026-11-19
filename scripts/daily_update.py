"""Daily mark-to-market update. Pulls TQQQ closes, revalues open positions, writes equity rows.

Run manually or via GitHub Action cron (see .github/workflows/daily.yml).

    python scripts/daily_update.py             # update the CSV, commit, push
    python scripts/daily_update.py --no-push   # update the CSV only
    python scripts/daily_update.py --dry-run   # compute and print, write nothing

Price sources, tried in order until one has the close for the latest completed session:
  1. yfinance daily bars, with retries.
  2. Yahoo's live quote (regularMarketPrice). Yahoo sometimes returns today's daily bar
     with an empty close; the quote field is filled independently of that bar.
  3. Nasdaq public historical endpoint.

Rules:
  * Only finite, positive closes are used. A bar with an empty close is dropped, never
    read as a price.
  * A row is written only if every value in it is finite. No close means no row.
  * If the close for the latest completed session cannot be obtained, the script exits 1
    so the GitHub Action fails and sends its email. Nothing is written for that session.
  * Every run also fills sessions the cron missed and repairs blank rows already in the
    CSV, so a failed day is recovered by the next successful run.
"""
from __future__ import annotations

import argparse
import json
import math
import numbers
import sys
import time
import traceback
import urllib.request
from datetime import date as date_cls, timedelta

import numpy as np
import pandas as pd

from lib import (
    EQUITY_FILE,
    cash_on,
    estimate_iv,
    git_commit_and_push,
    last_completed_session,
    load_config,
    load_equity_history,
    load_positions,
    position_state_on,
    value_position,
)

NY = "America/New_York"
VALUE_COLS = ["tqqq_close", "cash", "open_spread_value", "total_equity", "invested_to_date"]
MIN_HISTORY = 65       # closes needed for the 60-day realized vol (see lib.estimate_iv)
CATCH_UP_DAYS = 14     # how far back to look for sessions missing from the CSV
RETRY_WAITS = (3, 10, 30)  # seconds between yfinance attempts


def now_new_york() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY)


# =====================================================================
# Price sources. Each returns a frame indexed by session date with columns
# Close (raw close, used for the mark) and AdjClose (used for IV returns),
# containing only rows whose close is usable. None means the source failed.
# =====================================================================
def _usable(df: pd.DataFrame | None, source: str) -> pd.DataFrame | None:
    if df is None or len(df) == 0 or "Close" not in df.columns:
        return None
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=float)
    adj = (pd.to_numeric(df["Adj Close"], errors="coerce").to_numpy(dtype=float)
           if "Adj Close" in df.columns else close.copy())
    out = pd.DataFrame({"Close": close, "AdjClose": adj}, index=idx.normalize())
    out = out[~out.index.duplicated(keep="last")].sort_index()
    ok = np.isfinite(out["Close"]) & (out["Close"] > 0)
    if not ok.all():
        dropped = [str(d.date()) for d in out.index[~ok]]
        print(f"  {source}: dropped {len(dropped)} bar(s) with no usable close: "
              f"{', '.join(dropped[-5:])}")
    out = out[ok].copy()
    # IV only needs returns, so the raw close stands in where the adjusted one is missing.
    bad_adj = ~(np.isfinite(out["AdjClose"]) & (out["AdjClose"] > 0))
    out.loc[bad_adj, "AdjClose"] = out.loc[bad_adj, "Close"]
    return out if len(out) else None


def _covers(df: pd.DataFrame | None, need: date_cls) -> bool:
    return df is not None and len(df) > 0 and df.index[-1].date() >= need


def fetch_tqqq_yfinance(need: date_cls) -> pd.DataFrame | None:
    """yfinance daily bars. Retries while the close for `need` is missing."""
    try:
        import yfinance as yf
    except Exception as e:
        print(f"  yfinance import failed: {e}")
        return None
    best = None
    attempts = len(RETRY_WAITS) + 1
    for attempt in range(1, attempts + 1):
        try:
            raw = yf.download("TQQQ", period="1y", auto_adjust=False, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = _usable(raw, "yfinance")
            if df is None:
                print(f"  yfinance attempt {attempt}: no usable closes")
            else:
                if best is None or df.index[-1] >= best.index[-1]:
                    best = df
                if _covers(df, need):
                    print(f"  yfinance: {len(df)} closes through {df.index[-1].date()}")
                    return df
                print(f"  yfinance attempt {attempt}: latest usable close is "
                      f"{df.index[-1].date()}, need {need}")
        except Exception as e:
            print(f"  yfinance attempt {attempt} failed: {type(e).__name__}: {e}")
        if attempt < attempts:
            time.sleep(RETRY_WAITS[attempt - 1])
    return best


def _quote_from_meta(meta: dict) -> tuple[date_cls, float]:
    price = float(meta["regularMarketPrice"])
    t = meta["regularMarketTime"]
    ts = pd.Timestamp(t, unit="s", tz="UTC") if isinstance(t, numbers.Number) else pd.Timestamp(t)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(NY).date(), price


def _yahoo_meta_yfinance() -> dict:
    import yfinance as yf
    return yf.Ticker("TQQQ").get_history_metadata()


def _yahoo_meta_http() -> dict:
    req = urllib.request.Request(
        "https://query1.finance.yahoo.com/v8/finance/chart/TQQQ?range=5d&interval=1d",
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["chart"]["result"][0]["meta"]


def fetch_quote_yahoo() -> tuple[date_cls, float] | None:
    """Yahoo's live quote as (session date in New York, regular-session last price).
    After the 16:00 ET close this is the closing price, even when the daily bar is empty."""
    for name, get_meta in (("yfinance", _yahoo_meta_yfinance), ("http", _yahoo_meta_http)):
        try:
            session, price = _quote_from_meta(get_meta())
            if math.isfinite(price) and price > 0:
                print(f"  yahoo quote ({name}): ${price:.2f} for session {session}")
                return session, price
            print(f"  yahoo quote ({name}): unusable price {price!r}")
        except Exception as e:
            print(f"  yahoo quote ({name}) failed: {type(e).__name__}: {e}")
    return None


def fetch_tqqq_nasdaq() -> pd.DataFrame | None:
    """Nasdaq public historical endpoint. No auth, JSON response."""
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
        df = _usable(pd.DataFrame(records).set_index("Date"), "nasdaq")
        if df is not None:
            print(f"  nasdaq: {len(df)} closes through {df.index[-1].date()}")
        return df
    except Exception as e:
        print(f"  nasdaq failed: {type(e).__name__}: {e}")
        return None


def _with_quote(history: pd.DataFrame, session: date_cls, price: float) -> pd.DataFrame:
    history = history.copy()
    history.loc[pd.Timestamp(session)] = {"Close": price, "AdjClose": price}
    return history.sort_index()


def fetch_prices(need: date_cls) -> pd.DataFrame | None:
    """Best available close history. Works down the source list until one has `need`.
    If none does, returns the freshest history anyway (the caller can still catch up
    earlier sessions and will fail the run for `need`). None if every source failed."""
    yahoo = fetch_tqqq_yfinance(need)
    if _covers(yahoo, need):
        return yahoo

    print(f"  yfinance has no usable close for {need}; trying Yahoo's live quote...")
    quote = fetch_quote_yahoo()
    if quote is not None and quote[0] != need:
        print(f"  yahoo quote is for {quote[0]}, not {need}; ignoring it")
        quote = None
    if yahoo is not None and quote is not None:
        print(f"  using Yahoo's quote as the {need} close (daily bar had none)")
        return _with_quote(yahoo, *quote)

    print("  trying Nasdaq public API...")
    nasdaq = fetch_tqqq_nasdaq()
    if _covers(nasdaq, need):
        return nasdaq
    if nasdaq is not None and quote is not None:
        print(f"  using Yahoo's quote as the {need} close on top of Nasdaq history")
        return _with_quote(nasdaq, *quote)

    candidates = [c for c in (yahoo, nasdaq) if c is not None]
    return max(candidates, key=lambda c: c.index[-1]) if candidates else None


# =====================================================================
# Row construction
# =====================================================================
def build_row(d: pd.Timestamp, prices: pd.DataFrame, positions: list[dict],
              invested: float, verbose: bool = False) -> dict:
    """Equity row for session d, using only information available at that day's close.
    Raises ValueError rather than return anything non-finite."""
    hist = prices.loc[:d]
    if len(hist) == 0 or hist.index[-1] != d:
        raise ValueError(f"no close for {d.date()}")
    if len(hist) < MIN_HISTORY:
        raise ValueError(f"only {len(hist)} closes up to {d.date()}, need {MIN_HISTORY} for IV")
    close = float(hist["Close"].iloc[-1])
    iv = estimate_iv(hist["AdjClose"])
    day = d.date()
    if verbose:
        print(f"  TQQQ close: ${close:.2f} on {day}")
        print(f"  Estimated IV: {iv*100:.1f}%")

    open_value = 0.0
    for p in positions:
        if position_state_on(p, day) != "open":
            continue
        val = value_position(p, close, iv, today=day)
        open_value += val["total_value"]
        if verbose:
            print(f"  {p['id']}: cost ${val['cost_basis']:,.0f} -> value ${val['total_value']:,.0f} "
                  f"({val['unrealized_pct']:+.1f}%)")

    # Invested = actual capital paid in. For now, just starting capital. Adam has not
    # logged real DCA contributions yet. When contributions are tracked (future:
    # data/contributions.json), sum them here. Do NOT simulate $500/week as if it had
    # been deposited: that inflates "Total invested" and "Total equity" with money
    # that doesn't exist, masking the real P&L.
    cash = cash_on(positions, invested, day)
    row = {
        "tqqq_close": round(close, 2),
        "cash": round(cash, 2),
        "open_spread_value": round(open_value, 2),
        "total_equity": round(cash + open_value, 2),
        "invested_to_date": round(invested, 2),
    }
    _require_finite(row, day)
    return row


def _require_finite(row: dict, day) -> None:
    bad = [c for c in VALUE_COLS if not math.isfinite(float(row[c]))]
    if bad:
        raise ValueError(f"refusing to write {day}: non-finite {', '.join(bad)}")


# =====================================================================
# Main
# =====================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print the rows; do not write, commit or push")
    parser.add_argument("--no-push", action="store_true",
                        help="Write the CSV but do not commit or push")
    args = parser.parse_args(argv)

    now_ny = now_new_york()
    need = last_completed_session(now_ny)
    print(f"Daily update {date_cls.today()} (latest completed session: {need})")
    invested = float(load_config()["starting_capital"])
    positions = load_positions()["positions"]

    prices = fetch_prices(need)
    if prices is None:
        print("  All price sources failed. No row written. Exiting 1 so the failure is visible.")
        return 1
    latest = prices.index[-1]

    eq = load_equity_history()
    eq["date"] = pd.to_datetime(eq["date"])
    for c in VALUE_COLS:
        eq[c] = pd.to_numeric(eq[c], errors="coerce")
    eq = eq.drop_duplicates("date", keep="last").set_index("date").sort_index()[VALUE_COLS]

    # What to (re)compute this run. A complete row from an earlier day is never touched.
    blank = set(eq.index[eq.isna().any(axis=1)])            # rows with empty cells
    targets = set(blank)
    if len(eq) == 0:
        targets.add(latest)
    else:
        first_row, last_row = eq.index[0], eq.index[-1]
        recent = pd.Timestamp(need) - pd.Timedelta(days=CATCH_UP_DAYS)
        targets |= {d for d in prices.index                  # sessions not in the CSV yet
                    if d not in eq.index and d > first_row and (d > last_row or d >= recent)}
        if latest.date() >= now_ny.date():
            targets.add(latest)                              # today's row: add or refresh

    written, unresolved = [], []
    for d in sorted(targets):
        try:
            row = build_row(d, prices, positions, invested, verbose=(d == latest))
            if d in blank:
                # Repair: fill only what is missing. Stored cash and invested stay as they are.
                stored = eq.loc[d]
                for c in ("cash", "invested_to_date"):
                    if math.isfinite(stored[c]):
                        row[c] = float(stored[c])
                row["total_equity"] = round(row["cash"] + row["open_spread_value"], 2)
                _require_finite(row, d.date())
        except ValueError as e:
            unresolved.append((d, str(e)))
            continue
        kind = "repaired" if d in blank else ("updated" if d in eq.index else "added")
        eq.loc[d] = [row[c] for c in VALUE_COLS]
        written.append(d)
        print(f"  {d.date()} {kind}: TQQQ ${row['tqqq_close']:.2f}  cash ${row['cash']:,.2f}  "
              f"open ${row['open_spread_value']:,.2f}  total ${row['total_equity']:,.2f}")
    eq = eq.sort_index()

    # Last line of defence: nothing touched in this run may contain a blank.
    if written and eq.loc[written].isna().any().any():
        raise RuntimeError("internal error: a computed row contains NaN; nothing written")

    for d, why in unresolved:
        print(f"  WARNING {d.date()} not written: {why}")

    ok = True
    if not written:
        print("  Nothing to write.")
    elif args.dry_run:
        print(f"  Dry run: {len(written)} row(s) computed, nothing written.")
    else:
        out = eq.reset_index()
        out.to_csv(EQUITY_FILE, index=False, lineterminator="\n")
        print(f"  Equity history written ({len(written)} row(s) changed or added).")
        if not args.no_push:
            newest = max(written)
            if newest == eq.index[-1]:
                msg = (f"Daily MTM: {newest.date()} total equity "
                       f"${eq.loc[newest, 'total_equity']:,.0f}")
                if len(written) > 1:
                    msg += f" (+{len(written) - 1} backfilled)"
            else:
                msg = f"Backfill {len(written)} equity row(s) through {newest.date()}"
            if git_commit_and_push(msg):
                print("  Pushed.")
            else:
                print("  Commit or push failed.")
                ok = False

    # Fail loudly if the session we should have by now is missing, or blanks remain.
    need_ts = pd.Timestamp(need)
    if need_ts not in eq.index or eq.loc[need_ts].isna().any():
        print(f"!!! No close could be obtained for {need}. No row was written for it. "
              "The next successful run will fill it in.")
        ok = False
    remaining = eq.index[eq.isna().any(axis=1)]
    if len(remaining):
        print(f"!!! {len(remaining)} blank row(s) could not be repaired: "
              f"{', '.join(str(d.date()) for d in remaining)}")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        print("!!! Daily update crashed. Full traceback below !!!")
        traceback.print_exc()
        sys.exit(1)

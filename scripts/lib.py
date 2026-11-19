"""Shared utilities for the TQQQ journey repo."""
from __future__ import annotations

import json
import math
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
POSITIONS_FILE = DATA / "positions.json"
TRADES_FILE = DATA / "trades.json"
EQUITY_FILE = DATA / "equity_history.csv"
CONFIG_FILE = DATA / "config.json"


# =====================================================================
# Black-Scholes (no scipy dependency)
# =====================================================================
def _N(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(0.0, S - K)
    if sigma <= 0:
        return max(0.0, S - K * math.exp(-r * T))
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _N(d1) - K * math.exp(-r * T) * _N(d2)


def bs_spread_value(S: float, K_low: float, K_high: float, T: float,
                    r: float = 0.04, sigma: float = 0.60, skew: float = 0.90) -> float:
    """Value one share of a bull call spread."""
    return bs_call(S, K_low, T, r, sigma) - bs_call(S, K_high, T, r, sigma * skew)


# =====================================================================
# Data I/O
# =====================================================================
def load_positions() -> dict:
    if POSITIONS_FILE.exists():
        return json.loads(POSITIONS_FILE.read_text())
    return {"positions": []}


def save_positions(data: dict) -> None:
    POSITIONS_FILE.write_text(json.dumps(data, indent=2, default=str))


def load_trades() -> dict:
    if TRADES_FILE.exists():
        return json.loads(TRADES_FILE.read_text())
    return {"trades": []}


def save_trades(data: dict) -> None:
    TRADES_FILE.write_text(json.dumps(data, indent=2, default=str))


def load_config() -> dict:
    return json.loads(CONFIG_FILE.read_text())


def load_equity_history() -> pd.DataFrame:
    if not EQUITY_FILE.exists() or EQUITY_FILE.stat().st_size < 50:
        return pd.DataFrame(columns=[
            "date", "tqqq_close", "cash", "open_spread_value",
            "total_equity", "invested_to_date",
        ])
    df = pd.read_csv(EQUITY_FILE, parse_dates=["date"])
    return df


def append_equity_row(row: dict) -> None:
    df = load_equity_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(EQUITY_FILE, index=False)


# =====================================================================
# Realized vol estimator (for IV proxy in MTM)
# =====================================================================
def estimate_iv(prices: pd.Series, window: int = 60, iv_mult: float = 1.1,
                floor: float = 0.30, cap: float = 1.5) -> float:
    # Drop unusable closes first so one bad bar cannot poison the estimate.
    prices = pd.to_numeric(prices, errors="coerce")
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if len(prices) < window + 5:
        return 0.60
    lr = np.log(prices / prices.shift(1)).dropna()
    rv = float(lr.tail(window).std() * math.sqrt(252))
    if not math.isfinite(rv):
        # max(floor, min(cap, nan)) silently evaluates to the cap, so refuse instead.
        raise ValueError("realized vol is not finite; cannot estimate IV")
    return max(floor, min(cap, rv * iv_mult))


# =====================================================================
# Position valuation
# =====================================================================
def value_position(pos: dict, current_tqqq: float, iv: float,
                   today: Optional[date] = None) -> dict:
    """Compute current mark-to-market value for one spread position.
    Returns dict with per_contract_value, total_value, unrealized_pnl, etc."""
    # math.log(nan) does not raise, so a missing price or IV would otherwise flow
    # through Black-Scholes and come out as a NaN valuation. Stop it here.
    if not (math.isfinite(current_tqqq) and current_tqqq > 0):
        raise ValueError(f"cannot value {pos.get('id')}: TQQQ price is {current_tqqq!r}")
    if not (math.isfinite(iv) and iv > 0):
        raise ValueError(f"cannot value {pos.get('id')}: IV is {iv!r}")
    if today is None:
        today = date.today()
    expiry = datetime.strptime(pos["expiry"], "%Y-%m-%d").date()
    days_to_exp = (expiry - today).days
    T = max(0.001, days_to_exp / 365.25)

    value_per_share = bs_spread_value(
        current_tqqq, float(pos["long_strike"]), float(pos["short_strike"]), T,
        r=0.04, sigma=iv,
    )
    value_per_contract = value_per_share * 100
    total_value = value_per_contract * int(pos["contracts"])

    entry_debit = float(pos["net_debit_per_contract"])
    cost_basis = entry_debit * int(pos["contracts"])
    unrealized_pnl = total_value - cost_basis
    unrealized_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0

    # Max profit cap
    width_dollars = float(pos["short_strike"]) - float(pos["long_strike"])
    max_value_per_contract = width_dollars * 100
    max_profit_per_contract = max_value_per_contract - entry_debit

    return {
        "days_to_expiry": days_to_exp,
        "value_per_contract": round(value_per_contract, 2),
        "total_value": round(total_value, 2),
        "cost_basis": round(cost_basis, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "unrealized_pct": round(unrealized_pct, 2),
        "max_profit_per_contract": round(max_profit_per_contract, 2),
        "max_total_profit": round(max_profit_per_contract * int(pos["contracts"]), 2),
    }


# =====================================================================
# Position state as of a given date (for valuing past dates)
# =====================================================================
def _as_date(s: str) -> date:
    return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()


def position_state_on(pos: dict, d: date) -> str:
    """'pending' (not entered yet), 'open', or 'closed' as of the close of day d."""
    if _as_date(pos["entry_date"]) > d:
        return "pending"
    if pos.get("status") == "open":
        return "open"
    exit_date = pos.get("exit_date")
    return "open" if exit_date and _as_date(exit_date) > d else "closed"


def cash_on(positions: list[dict], invested: float, d: date) -> float:
    """Cash as of day d: invested, minus entry costs paid, plus exit proceeds received."""
    cash = float(invested)
    for p in positions:
        state = position_state_on(p, d)
        if state != "pending":
            cash -= float(p["total_cost"])
        if state == "closed":
            cash += float(p.get("gross_exit_value", 0.0))
    return cash


# =====================================================================
# NYSE session calendar. Only used to decide whether a close is overdue
# (i.e. whether the cron should fail loudly); never used to compute values.
# =====================================================================
class _NYSEHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def is_nyse_session(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    holidays = _NYSEHolidays().holidays(pd.Timestamp(d.year - 1, 12, 1),
                                        pd.Timestamp(d.year + 1, 1, 31))
    return pd.Timestamp(d) not in holidays


def last_completed_session(now_ny: pd.Timestamp) -> date:
    """Most recent NYSE session whose 16:00 ET close has already happened."""
    d = now_ny.date()
    if now_ny.hour < 16:
        d -= timedelta(days=1)
    while not is_nyse_session(d):
        d -= timedelta(days=1)
    return d


# =====================================================================
# Git helpers
# =====================================================================
def git_commit_and_push(message: str, files: list[str] | None = None) -> bool:
    """Commit specified files (or everything in data/) and push. Returns True on success."""
    try:
        if files:
            subprocess.run(["git", "-C", str(REPO), "add", *files], check=True)
        else:
            subprocess.run(["git", "-C", str(REPO), "add", "data/"], check=True)
        # Check if there are changes
        result = subprocess.run(
            ["git", "-C", str(REPO), "diff", "--cached", "--quiet"],
            check=False,
        )
        if result.returncode == 0:
            print("  (no changes to commit)")
            return True
        subprocess.run(
            ["git", "-C", str(REPO), "commit", "-m", message],
            check=True,
        )
        subprocess.run(["git", "-C", str(REPO), "push"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  git error: {e}")
        return False

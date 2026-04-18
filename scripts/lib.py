"""Shared utilities for the TQQQ journey repo."""
from __future__ import annotations

import json
import math
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

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
    if len(prices) < window + 5:
        return 0.60
    lr = np.log(prices / prices.shift(1)).dropna()
    rv = float(lr.tail(window).std() * math.sqrt(252))
    return max(floor, min(cap, rv * iv_mult))


# =====================================================================
# Position valuation
# =====================================================================
def value_position(pos: dict, current_tqqq: float, iv: float,
                   today: Optional[date] = None) -> dict:
    """Compute current mark-to-market value for one spread position.
    Returns dict with per_contract_value, total_value, unrealized_pnl, etc."""
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

"""Log a new TQQQ bull call spread entry.

Usage:
    python scripts/log_trade.py \\
        --tqqq 55.50 --long 40 --short 70 --expiry 2028-01-21 \\
        --contracts 12 --debit 13.00 [--date 2026-05-05] [--note "Tranche 1"]

After logging:
  - Updates data/positions.json
  - Appends to data/trades.json
  - Commits + pushes to GitHub (public repo)
  - Prints a formatted WhatsApp/Telegram message for copy-paste
"""
from __future__ import annotations

import argparse
import uuid
from datetime import date as date_cls, datetime

from lib import (
    git_commit_and_push,
    load_config,
    load_positions,
    load_trades,
    save_positions,
    save_trades,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tqqq", type=float, required=True, help="TQQQ price at entry")
    parser.add_argument("--long", type=float, required=True, help="Long call strike")
    parser.add_argument("--short", type=float, required=True, help="Short call strike")
    parser.add_argument("--expiry", type=str, required=True, help="Expiry date YYYY-MM-DD")
    parser.add_argument("--contracts", type=int, required=True, help="Number of contracts")
    parser.add_argument("--debit", type=float, required=True,
                        help="Net debit per share (not per contract)")
    parser.add_argument("--date", type=str, default=None, help="Entry date (default: today)")
    parser.add_argument("--note", type=str, default="", help="Optional note/tranche label")
    parser.add_argument("--vxn", type=float, default=None, help="VXN at entry (optional)")
    parser.add_argument("--no-push", action="store_true",
                        help="Skip git commit+push (for testing)")
    args = parser.parse_args()

    entry_date = args.date or date_cls.today().isoformat()
    position_id = f"tqqq-{entry_date}-{uuid.uuid4().hex[:6]}"

    # Sanity checks
    if args.long >= args.short:
        raise SystemExit("ERROR: long strike must be < short strike for bull call spread")
    long_pct = args.long / args.tqqq
    short_pct = args.short / args.tqqq
    width_pct = (args.short - args.long) / args.tqqq
    print(f"\nProposed entry:")
    print(f"  Date: {entry_date}")
    print(f"  TQQQ: ${args.tqqq:.2f}")
    print(f"  Long:  ${args.long:.2f} call  ({long_pct*100:.1f}% of spot, {(1-long_pct)*100:+.1f}% ITM if <100)")
    print(f"  Short: ${args.short:.2f} call  ({short_pct*100:.1f}% of spot)")
    print(f"  Width: ${args.short - args.long:.2f}  ({width_pct*100:.1f}% of spot)")
    print(f"  Expiry: {args.expiry}")
    print(f"  Contracts: {args.contracts}")
    print(f"  Net debit: ${args.debit:.2f}/share = ${args.debit*100:.0f}/contract")
    print(f"  Total cost: ${args.debit * 100 * args.contracts:,.2f}")

    # Warnings
    warnings = []
    if not (0.60 <= long_pct <= 0.80):
        warnings.append(f"Long strike is at {long_pct*100:.0f}% of spot; strategy target is 70%")
    if not (1.15 <= short_pct <= 1.35):
        warnings.append(f"Short strike is at {short_pct*100:.0f}% of spot; strategy target is 125%")
    if not (0.40 <= width_pct <= 0.70):
        warnings.append(f"Width is {width_pct*100:.0f}%; strategy target is 55%")
    exp_date = datetime.strptime(args.expiry, "%Y-%m-%d").date()
    months_to_exp = (exp_date - date_cls.fromisoformat(entry_date)).days / 30.44
    if months_to_exp < 20:
        warnings.append(f"Expiry is only {months_to_exp:.1f} months out; strategy target is 24")
    if warnings:
        print("\nWARNINGS (proceeding anyway — verify these are intentional):")
        for w in warnings:
            print(f"  [!] {w}")

    # Build position record
    position = {
        "id": position_id,
        "entry_date": entry_date,
        "underlying": "TQQQ",
        "strategy": "bull_call_spread",
        "tqqq_at_entry": args.tqqq,
        "vxn_at_entry": args.vxn,
        "long_strike": args.long,
        "short_strike": args.short,
        "width": round(args.short - args.long, 2),
        "expiry": args.expiry,
        "contracts": args.contracts,
        "net_debit_per_share": args.debit,
        "net_debit_per_contract": args.debit * 100,
        "total_cost": round(args.debit * 100 * args.contracts, 2),
        "max_profit": round((args.short - args.long - args.debit) * 100 * args.contracts, 2),
        "breakeven": round(args.long + args.debit, 2),
        "note": args.note,
        "status": "open",
    }

    # Write to positions.json
    positions_data = load_positions()
    positions_data["positions"].append(position)
    save_positions(positions_data)

    # Append to trades.json
    trades_data = load_trades()
    trades_data["trades"].append({
        "id": position_id,
        "event": "open",
        "date": entry_date,
        "position_id": position_id,
        "details": position,
    })
    save_trades(trades_data)

    print(f"\n  [OK] Position {position_id} saved to positions.json")
    print(f"  [OK] Trade event logged to trades.json")

    # Git push
    if not args.no_push:
        print("\nCommitting and pushing to GitHub...")
        ok = git_commit_and_push(
            f"Log trade: open TQQQ {args.long}/{args.short} spread x{args.contracts}",
        )
        if ok:
            print("  [OK] Pushed to GitHub")
        else:
            print("  [!] Git push failed — commit locally, fix later")

    # Formatted message
    config = load_config()
    web_url = config.get("web_app_url", "")
    print()
    print("=" * 78)
    print("MESSAGE FOR WHATSAPP / TELEGRAM GROUP (copy-paste below):")
    print("=" * 78)
    print()
    msg_lines = [
        "[*] New trade opened!",
        f"TQQQ {_pretty_expiry(args.expiry)} {args.long:.0f}/{args.short:.0f} bull call spread",
        "",
        f"  - TQQQ @ ${args.tqqq:.2f}  ({args.contracts} contracts)",
        f"  - Debit: ${args.debit:.2f}/share  (total ${args.debit * 100 * args.contracts:,.0f})",
        f"  - Max profit: ${position['max_profit']:,.0f}",
        f"  - Breakeven: TQQQ ${position['breakeven']:.2f}",
        f"  - Expires: {args.expiry}  (~{months_to_exp:.0f} months)",
        "",
    ]
    if args.note:
        msg_lines.append(f"  Note: {args.note}")
        msg_lines.append("")
    if web_url:
        msg_lines.append(f"Follow the journey -> {web_url}")
    print("\n".join(msg_lines))
    print()
    print("=" * 78)


def _pretty_expiry(d: str) -> str:
    dt = datetime.strptime(d, "%Y-%m-%d").date()
    return dt.strftime("%b %Y")


if __name__ == "__main__":
    main()

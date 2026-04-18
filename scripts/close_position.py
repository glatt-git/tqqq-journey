"""Close or roll an existing TQQQ spread position.

Usage:
    # Close out entirely (at expiry or early close)
    python scripts/close_position.py --id <position_id> --exit-price 15.00 --event close

    # Roll (close + immediately open new — use --event roll and separately run log_trade.py)
    python scripts/close_position.py --id <position_id> --exit-price 15.00 --event roll

    # Let expire (at expiry, computes payoff automatically from strikes + final TQQQ)
    python scripts/close_position.py --id <position_id> --event expire --tqqq-at-exp 65.00
"""
from __future__ import annotations

import argparse
from datetime import date as date_cls

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
    parser.add_argument("--id", required=True, help="Position ID to close")
    parser.add_argument("--event", choices=["close", "roll", "expire"], required=True,
                        help="Close=early close, roll=close+replace, expire=let expire")
    parser.add_argument("--exit-price", type=float, default=None,
                        help="Spread value per share at exit (for close/roll)")
    parser.add_argument("--tqqq-at-exp", type=float, default=None,
                        help="TQQQ price at expiry (for expire event)")
    parser.add_argument("--date", default=None, help="Exit date (default: today)")
    parser.add_argument("--note", default="")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    exit_date = args.date or date_cls.today().isoformat()

    positions_data = load_positions()
    pos = next((p for p in positions_data["positions"] if p["id"] == args.id), None)
    if pos is None:
        raise SystemExit(f"No position found with id {args.id}")
    if pos.get("status") != "open":
        raise SystemExit(f"Position {args.id} is not open (status: {pos.get('status')})")

    # Compute exit value
    contracts = int(pos["contracts"])
    if args.event == "expire":
        if args.tqqq_at_exp is None:
            raise SystemExit("--tqqq-at-exp required for expire event")
        tqqq_exp = args.tqqq_at_exp
        K_low = float(pos["long_strike"])
        K_high = float(pos["short_strike"])
        payoff_per_share = max(0.0, min(tqqq_exp - K_low, K_high - K_low))
        exit_price = payoff_per_share
    else:
        if args.exit_price is None:
            raise SystemExit("--exit-price required for close/roll events")
        exit_price = args.exit_price

    gross_exit_value = exit_price * 100 * contracts
    entry_cost = float(pos["total_cost"])
    realized_pnl = gross_exit_value - entry_cost
    pnl_pct = (realized_pnl / entry_cost * 100) if entry_cost > 0 else 0.0

    # Update position
    pos["status"] = "closed" if args.event == "close" else (
        "rolled" if args.event == "roll" else "expired"
    )
    pos["exit_date"] = exit_date
    pos["exit_price_per_share"] = exit_price
    pos["gross_exit_value"] = round(gross_exit_value, 2)
    pos["realized_pnl"] = round(realized_pnl, 2)
    pos["realized_pnl_pct"] = round(pnl_pct, 2)
    pos["close_note"] = args.note

    save_positions(positions_data)

    # Append trade event
    trades_data = load_trades()
    trades_data["trades"].append({
        "id": f"{pos['id']}-exit",
        "event": args.event,
        "date": exit_date,
        "position_id": pos["id"],
        "exit_price": exit_price,
        "realized_pnl": round(realized_pnl, 2),
        "realized_pnl_pct": round(pnl_pct, 2),
        "note": args.note,
    })
    save_trades(trades_data)

    print(f"\n  [OK] Position {args.id} closed as '{args.event}'")
    print(f"  Exit price: ${exit_price:.2f}/share  =>  ${gross_exit_value:,.2f} total")
    print(f"  Entry cost: ${entry_cost:,.2f}")
    print(f"  Realized P&L: ${realized_pnl:,.2f}  ({pnl_pct:+.1f}%)")

    if not args.no_push:
        ok = git_commit_and_push(
            f"{args.event.capitalize()} position {args.id}: P&L ${realized_pnl:+,.0f}",
        )
        if ok:
            print("  [OK] Pushed to GitHub")

    # Formatted message
    config = load_config()
    web_url = config.get("web_app_url", "")
    emoji = "[+]" if realized_pnl >= 0 else "[-]"
    event_verb = {
        "close": "Closed", "roll": "Rolled out of", "expire": "Expired",
    }[args.event]
    print()
    print("=" * 78)
    print("MESSAGE FOR WHATSAPP / TELEGRAM GROUP:")
    print("=" * 78)
    msg = [
        f"{emoji} {event_verb} TQQQ {pos['long_strike']:.0f}/{pos['short_strike']:.0f} spread (expiry {pos['expiry']})",
        "",
        f"  - Entry: ${pos['net_debit_per_share']:.2f}/share on {pos['entry_date']}",
        f"  - Exit:  ${exit_price:.2f}/share on {exit_date}",
        f"  - Realized P&L: ${realized_pnl:+,.0f}  ({pnl_pct:+.1f}%)",
        "",
    ]
    if web_url:
        msg.append(f"Follow -> {web_url}")
    print("\n".join(msg))


if __name__ == "__main__":
    main()

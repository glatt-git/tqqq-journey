# TQQQ Journey — public trade tracker

Adam's attempt to grow $50k → $1M by end of 2030 via rolling 24-month TQQQ bull call spreads.

## For followers

The live dashboard is at **https://tqqq-journey.streamlit.app** (update URL after deploy).

Pages:
- **Home** — current equity, progress to target, strategy summary
- **Current Positions** — open spreads with live mark-to-market
- **Trade Log** — every entry, exit, roll (append-only)
- **Equity Curve** — daily P&L history
- **Strategy** — the rules + honest caveats

Not investment advice. Just one person's journey.

---

## For Adam — operational runbook

### Initial setup (one-time)

1. Create a **public** GitHub repo called `tqqq-journey` (public so Streamlit Community Cloud can deploy it for free)
2. Push this folder to that repo
3. Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub, deploy from the repo
4. Update `data/config.json` `web_app_url` field with the Streamlit URL you get
5. Commit + push that config update

### Monthly entry (after tranche #1 fills)

```bash
cd D:/Documents/Trading/tqqq-journey

# Fill in actual values from your ToS order confirmation:
python scripts/log_trade.py \
    --tqqq 55.50 \
    --long 40 --short 90 \
    --expiry 2028-01-21 \
    --contracts 1 \
    --debit 18.00 \
    --date 2026-05-05 \
    --note "Tranche 1 — 38.2% Fib pullback entry" \
    --vxn 24.5
```

The script:
- Updates `data/positions.json`
- Appends to `data/trades.json`
- `git add` + `git commit` + `git push` automatically
- **Prints a WhatsApp/Telegram message for you to copy-paste**

The web app (deployed on Streamlit Cloud) auto-detects the git push and updates within ~1 minute.

### When a spread reaches 11 months remaining (roll time — after 13 months held)

```bash
# First close the old position
python scripts/close_position.py \
    --id <position_id_from_positions.json> \
    --exit-price 22.00 \
    --event roll \
    --note "Rolled at 11mo remaining (13mo held, LT cap gains)"

# Then enter the new one
python scripts/log_trade.py ...
```

### If you want to force a daily MTM update locally

```bash
python scripts/daily_update.py
```

(Otherwise the GitHub Action at `.github/workflows/daily.yml` runs this at 4:30pm ET every weekday automatically.)

### Where do I see the message to paste to WhatsApp?

It prints to your terminal at the end of `log_trade.py` / `close_position.py`.
Copy everything between the `===` lines and paste into your WhatsApp group.

---

## Architecture

```
tqqq-journey/
├── app.py                     # Streamlit app (public dashboard)
├── data/
│   ├── config.json            # strategy metadata, target, owner info
│   ├── positions.json         # all positions (open + closed)
│   ├── trades.json            # append-only event log
│   └── equity_history.csv     # daily MTM snapshots
├── scripts/
│   ├── lib.py                 # shared: BS pricing, file I/O, git helpers
│   ├── log_trade.py           # open a new position (CLI)
│   ├── close_position.py      # close/roll/expire (CLI)
│   └── daily_update.py        # MTM + append equity row (cron)
├── .github/workflows/daily.yml  # daily cron for MTM updates
├── requirements.txt
└── README.md
```

No database. Git is the audit trail. Everything is public and versioned.

### Why not Supabase?

Supabase free tier pauses projects after 7 days of inactivity. At 1 trade/month, we'd
need a keep-alive cron anyway. Since we need a daily cron regardless (for MTM), we may
as well use that same cron to commit to git instead — one less service to maintain.

The trade-off: querying/filtering is easier in SQL. But with <100 trades over 5 years,
we're nowhere near needing SQL.

---

## Caveats

- **MTM valuations use Black-Scholes with realized vol as IV proxy.** Real option
  prices differ due to bid/ask, IV skew, and vol-of-vol. The tracker will be
  approximate on day-to-day MTM; realized P&L on close/expire is exact.
- **The `daily_update.py` script uses `yfinance` for TQQQ close prices.** Yahoo has
  occasional data delays or outages; if the action fails, the next day's run will
  backfill.
- **Equity history starts from the first trade.** Days before any position exists are
  not in the history (no meaningful MTM to record).

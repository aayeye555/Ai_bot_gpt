# AI Trader Pro — V5 adaptive AI research engine — Paper MVP

A small FastAPI dashboard for Binance USD-M Futures market scanning and **manual-approval paper trading**.

## Fixed initial configuration

- Paper equity: **$1,000 USDT**
- Symbols: BTCUSDT, ETHUSDT, SOLUSDT
- Signal timeframe: 5-minute candles
- Risk budget: 0.5% of current equity per trade (initially $5)
- Maximum notional leverage: 2x equity
- Maximum concurrent positions: 3
- Daily realized-loss pause threshold: 1.5% of starting capital ($15)
- Every candidate is queued; the user must approve it before a simulated position opens.
- **No Binance API key, no live order endpoint, and no live execution code are included.**

## Run locally

Python 3.11+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

The app reads public Binance Futures market data from `fapi.binance.com`. It does not need credentials for market data. Network access to Binance is required.

## Deploy on Render

1. Create a private GitHub repository and upload this project.
2. In Render, choose **New → Blueprint** and connect that repository.
3. Render detects `render.yaml`; create the service.
4. Wait for `/health` to report `{"status":"ok","mode":"paper-only"}`.
5. Open the service URL. The persistent disk stores the paper account state.

The included starter persistent disk may incur a Render charge. Confirm current Render pricing and region availability before deploying.

## Strategy notes

The scanner uses transparent, deterministic rules:
- Pump-exhaustion short: an extended 12-candle rise, elevated RSI, relative volume, and a rejection/breakdown confirmation.
- Trend-continuation long: EMA9 > EMA21, price above EMA9, RSI in a defined band, and elevated relative volume.

These are candidate-generation heuristics, not a validated profitable strategy. The daily scanner is manually triggered; it does not place orders automatically.

## Important MVP limitations

- Manual refresh/scan; no always-on scheduler or WebSocket stream.
- Paper positions are manually closed. Stop and target are displayed but are not yet monitored/triggered automatically.
- Exit cost is a simple estimated round-trip 0.10% of notional; funding, spread and slippage are not fully modeled.
- No authentication is included. **Do not expose this dashboard publicly until authentication and access controls are added.**
- The app is paper-only. Do not add exchange keys or enable live trading without independent security review, backtesting, testnet validation, and explicit risk controls.

## API

- `GET /api/state`
- `POST /api/scan`
- `POST /api/orders/{id}/decision` with `{"decision":"approve"|"reject"}`
- `POST /api/positions/{id}/close`
- `POST /api/pause`
- `POST /api/reset`
- `POST /api/backtest` with `{"days":90,"interval":"5m","symbols":["BTCUSDT","ETHUSDT","SOLUSDT"]}` (days 7–365; intervals 5m/15m)
- `GET /health`


## Historical backtesting (added)

The dashboard includes a historical backtest panel. It downloads Binance Futures candles for 30/90/180/365 days at 5m or 15m, then replays the same signal rules. It uses next-candle-open entries to reduce look-ahead bias, 0.5% risk per trade, a 2x notional cap, 2R targets, estimated fee/slippage, and conservative stop-first ordering when both stop and target fall inside the same candle. The $1,000 is split equally across selected symbols. Funding is excluded and assumptions are displayed in the API response. This is a research simulation, not a guarantee of live results.


## Dynamic top-coin universe (added in V3)

The scanner and historical backtest can use the top 10, 20, 30, 40 or 50 active Binance USD-M USDT perpetual contracts, ranked by 24-hour quote volume. Stablecoin bases and common leveraged-token suffixes are excluded. The universe is refreshed from Binance exchange metadata and 24-hour ticker data on each scan/backtest. This ranks liquidity, not expected profitability.

The scanner fetches 5-minute candles with bounded concurrency and creates at most three pending paper-trade approvals. The backtest uses the selected current universe and divides the $1,000 starting equity equally across its symbols. For 30–50 coins, backtests are limited to 90 days or less to avoid excessive historical-data requests. A 50-coin historical test can still take several minutes and may be subject to Binance rate limits.


## V4 — strategy-isolated backtesting

The backtest now replays the same downloaded historical candles in three independent modes:
- **LONG only** — trend-continuation long signals only.
- **SHORT only** — pump-exhaustion short signals only.
- **COMBINED** — both signal types, preserving the existing signal priority.

The dashboard shows a side-by-side strategy summary and per-coin comparison, plus detailed per-coin results for the selected mode. Each mode is independently simulated from the same $1,000 starting equity; these are comparison runs, not three portfolios that can be added together. Historical downloads are performed once per symbol and reused for all three modes.

**Important backtest limitation:** the selected universe is today's top-volume Binance USDT perpetual contracts, not the contracts that ranked in the top N at each historical date. This creates possible survivorship/universe-selection bias. Funding is excluded, fees/slippage are estimates, and historical results do not predict future returns.


## V5 — adaptive AI learning engine

- Trains a three-class HistGradientBoosting model on Binance USD-M Futures 5-minute candles.
- Predicts the 12-candle forward direction class: LONG (> +0.35%), SHORT (< -0.35%), or NEUTRAL.
- Uses causal technical features and a chronological 75/25 holdout split within each symbol.
- Reports accuracy, balanced accuracy, class recall and a confusion matrix.
- Saves the model and metadata to the Render persistent disk (`/data`) so it survives restarts.
- AI is OFF after training. The user must explicitly activate it; once active, it only filters existing rule-based signals at a configurable confidence threshold.
- Manual trade approval, 0.5% risk, 2x notional cap, paper-only execution and daily loss guard remain unchanged.

**AI limitations and safety:** this is a supervised ML direction classifier, not an autonomous profit-making agent or a guarantee of profitable trades. The label is a forward price-move classification, not a net-profit label; validation accuracy does not establish trading edge. It does not yet learn from closed paper-trade outcomes or retrain itself automatically. Retraining is a deliberate user action. Keep it in paper mode and compare out-of-sample trading results before considering any live execution.


## V6 — self-learning paper-trading system

V6 adds persistent learning from completed paper trades and guarded periodic model refresh:
- Stores each completed paper trade and groups outcomes by LONG/SHORT, setup and a simple market regime (TRENDING, RANGE, HIGH_VOLATILITY).
- Calculates recent average R-multiple, win rate and a sample-size-shrunk advisory score. Segments are shown only after at least five trades.
- Records approve/reject actions separately as user feedback; a rejected signal is **not** falsely labeled a losing trade.
- Adds `/api/learning` and a dashboard learning panel.
- Runs a weekly background candidate retraining job. A candidate model is promoted only when its chronological holdout balanced accuracy is no more than 1 percentage point below the incumbent. Otherwise the incumbent remains in place.
- Keeps fixed risk (0.5%), 2x leverage cap, daily loss stop, paper-only mode and manual approval.

### V6 limitations and safe-use notes
- The strategy/regime learner is currently advisory; it does not automatically change trade direction, sizing, stop distance or leverage. This avoids overfitting a small paper-trade sample.
- The model validation is a chronological holdout by symbol, not a fully purged walk-forward test; overlapping forward labels can still make validation optimistic.
- Weekly retraining uses today's top-volume universe, so survivorship bias remains possible.
- Render must use a persistent disk mounted at `/data` (`DATA_DIR=/data`) or model/state/learning files may be lost on redeploy. Configure one service instance; multiple instances would need shared storage and locking.
- Do not expose the dashboard publicly without authentication. No live Binance private API or live order endpoint is implemented.
- This is an experimental paper-learning tool, not financial advice or a promise of returns.

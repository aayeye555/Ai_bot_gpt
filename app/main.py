import os, json, math, asyncio
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from app.learning import TradeLearner

BASE = Path(__file__).parent
DATA = Path(os.getenv("DATA_DIR", "/data"))
DATA.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA / "state.json"
AI_MODEL_FILE = DATA / "ai_direction_model.joblib"
AI_META_FILE = DATA / "ai_model_meta.json"
LEARNING_FILE = DATA / "trade_learning_memory.json"
AI_FEATURE_NAMES = ["ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "rsi_14", "ema_gap", "rel_volume", "range_pct", "body_pct", "volatility_14"]
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
STARTING_EQUITY = 1000.0
RISK_FRACTION = 0.005
MAX_LEVERAGE = 2.0
DAILY_LOSS_LIMIT = 0.015
MAX_POSITIONS = 3
BINANCE = "https://fapi.binance.com"

app = FastAPI(title="AI Trader Pro", version="0.5.0")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

def default_state():
    return {"equity": STARTING_EQUITY, "realized_pnl": 0.0, "day": str(date.today()),
            "daily_pnl": 0.0, "pending": [], "positions": [], "trades": [],
            "last_scan": None, "mode": "PAPER", "paused": False}

def load_state():
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            # Keep the safety-critical configuration fixed in this MVP.
            return data
        except Exception:
            pass
    return default_state()

state = load_state()
AI_MODEL = None
AI_META = {"trained": False, "active": False, "confidence_threshold": 0.58}


def load_ai_model():
    global AI_MODEL, AI_META
    if AI_META_FILE.exists():
        try:
            AI_META = json.loads(AI_META_FILE.read_text())
        except Exception:
            AI_META = {"trained": False, "active": False, "confidence_threshold": 0.58}
    if AI_MODEL_FILE.exists():
        try:
            AI_MODEL = joblib.load(AI_MODEL_FILE)
            AI_META["trained"] = True
        except Exception:
            AI_MODEL = None
            AI_META["trained"] = False
            AI_META["active"] = False


def save_ai_meta():
    AI_META_FILE.write_text(json.dumps(AI_META, indent=2))


load_ai_model()
LEARNER = TradeLearner(LEARNING_FILE)


def save():
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)

def reset_day():
    today = str(date.today())
    if state.get("day") != today:
        state["day"], state["daily_pnl"] = today, 0.0
        save()

async def get_klines(symbol: str, interval="5m", limit=100):
    async with httpx.AsyncClient(timeout=12) as client:
        r = await client.get(f"{BINANCE}/fapi/v1/klines",
                             params={"symbol": symbol, "interval": interval, "limit": limit})
        r.raise_for_status()
        return r.json()

def ema(values, period):
    k = 2/(period+1)
    out = values[0]
    for v in values[1:]:
        out = v*k + out*(1-k)
    return out

def rsi(values, period=14):
    changes = [values[i]-values[i-1] for i in range(1, len(values))]
    gains = sum(max(x,0) for x in changes[-period:])/period
    losses = sum(max(-x,0) for x in changes[-period:])/period
    if losses == 0: return 100.0
    rs = gains/losses
    return 100 - 100/(1+rs)

def analyze(symbol, candles):
    closes = [float(c[4]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    volumes = [float(c[5]) for c in candles]
    last = closes[-1]
    current_rsi = rsi(closes)
    avg_vol = sum(volumes[-21:-1])/20
    rel_vol = volumes[-1]/avg_vol if avg_vol else 0
    change_12 = (last/closes[-13]-1)*100
    fast, slow = ema(closes[-50:], 9), ema(closes[-50:], 21)
    side, setup, score, rationale = None, None, 0, []

    # Conservative pump-exhaustion short: extended move + high RSI/volume
    # + rejection from recent high + close below prior candle low.
    recent_high = max(highs[-12:-1])
    prior_low = lows[-2]
    rejection = (recent_high-last)/recent_high*100
    if change_12 >= 2.0: score += 1; rationale.append("12-candle move is extended")
    if current_rsi >= 72: score += 1; rationale.append(f"RSI elevated ({current_rsi:.1f})")
    if rel_vol >= 1.5: score += 1; rationale.append(f"Relative volume {rel_vol:.1f}x")
    if rejection >= 0.25 and last < prior_low: score += 1; rationale.append("High rejection and breakdown")
    if score >= 3 and rejection >= 0.25 and last < prior_low:
        side, setup = "SHORT", "Pump exhaustion"

    # Trend-following long only when no short setup qualifies.
    if side is None and fast > slow and last > fast and 52 <= current_rsi <= 68 and rel_vol >= 1.1:
        side, setup, score = "LONG", "Trend continuation", 3
        rationale = ["EMA9 above EMA21", "Price above EMA9", f"RSI {current_rsi:.1f}", f"Volume {rel_vol:.1f}x"]

    # ATR-like stop distance from recent range; floor prevents tiny stops.
    ranges = [highs[i]-lows[i] for i in range(-14,0)]
    stop_pct = max(0.004, (sum(ranges)/len(ranges))/last * 1.5)
    stop_pct = min(stop_pct, 0.025)
    return {"symbol": symbol, "price": last, "side": side, "setup": setup,
            "score": score, "rsi": current_rsi, "relative_volume": rel_vol,
            "change_12": change_12, "stop_pct": stop_pct,
            "rationale": rationale, "time": datetime.now(timezone.utc).isoformat()}

EXCLUDED_BASES = {"USDC", "FDUSD", "TUSD", "USDP", "BUSD", "DAI", "EUR", "GBP"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")

async def get_top_usdt_perpetuals(limit: int = 10):
    """Rank active USDT-margined perpetual contracts by 24h quote volume."""
    async with httpx.AsyncClient(timeout=20) as client:
        info_r, ticker_r = await asyncio.gather(
            client.get(f"{BINANCE}/fapi/v1/exchangeInfo"),
            client.get(f"{BINANCE}/fapi/v1/ticker/24hr")
        )
        info_r.raise_for_status()
        ticker_r.raise_for_status()
        info, tickers = info_r.json(), ticker_r.json()

    volumes = {x["symbol"]: float(x.get("quoteVolume", 0)) for x in ticker_r.json()}
    eligible = []
    for item in info.get("symbols", []):
        symbol = item.get("symbol", "")
        base = item.get("baseAsset", "")
        if (item.get("status") == "TRADING" and item.get("contractType") == "PERPETUAL"
                and item.get("quoteAsset") == "USDT" and base not in EXCLUDED_BASES
                and not base.endswith(LEVERAGED_SUFFIXES) and volumes.get(symbol, 0) > 0):
            eligible.append({"symbol": symbol, "quote_volume_24h": volumes[symbol]})
    eligible.sort(key=lambda x: x["quote_volume_24h"], reverse=True)
    return eligible[:limit]

def ai_features(candles, index=None):
    """Causal features built only from candles up to index (inclusive)."""
    end = len(candles) if index is None else index + 1
    window = candles[:end]
    if len(window) < 60:
        raise ValueError("At least 60 candles are required for AI features")
    closes = [float(c[4]) for c in window]
    highs = [float(c[2]) for c in window]
    lows = [float(c[3]) for c in window]
    opens = [float(c[1]) for c in window]
    volumes = [float(c[5]) for c in window]
    last = closes[-1]
    def ret(n):
        return last / closes[-1-n] - 1 if len(closes) > n and closes[-1-n] else 0.0
    avg_volume = sum(volumes[-21:-1]) / 20 if len(volumes) >= 21 else 0
    rel_volume = volumes[-1] / avg_volume if avg_volume else 0.0
    avg_range = sum((highs[i]-lows[i]) for i in range(-14, 0)) / 14
    volatility = avg_range / last if last else 0.0
    fast = ema(closes[-50:], 9)
    slow = ema(closes[-50:], 21)
    return [ret(1), ret(3), ret(6), ret(12), ret(24), rsi(closes),
            (fast/slow-1) if slow else 0.0, rel_volume,
            (highs[-1]-lows[-1])/last if last else 0.0,
            (closes[-1]-opens[-1])/last if last else 0.0, volatility]


def ai_probabilities(candles):
    if AI_MODEL is None:
        return None
    vector = ai_features(candles)
    probs = AI_MODEL.predict_proba([vector])[0]
    return {int(label): float(prob) for label, prob in zip(AI_MODEL.classes_, probs)}


def apply_ai_filter(signal, candles):
    if not AI_META.get("active") or AI_MODEL is None or not signal.get("side"):
        return signal
    try:
        probs = ai_probabilities(candles) or {}
        side_label = 1 if signal["side"] == "LONG" else -1
        confidence = probs.get(side_label, 0.0)
        signal["ai_confidence"] = round(confidence, 4)
        signal["ai_probabilities"] = {"short": round(probs.get(-1, 0.0), 4),
                                      "neutral": round(probs.get(0, 0.0), 4),
                                      "long": round(probs.get(1, 0.0), 4)}
        threshold = float(AI_META.get("confidence_threshold", 0.58))
        if confidence < threshold:
            signal["side"] = None
            signal["setup"] = None
            signal["rationale"].append(f"AI filter rejected setup ({confidence:.0%} < {threshold:.0%})")
        else:
            signal["rationale"].append(f"AI confirms direction ({confidence:.0%} confidence)")
    except Exception as exc:
        # Fail closed while AI mode is active: do not silently bypass a broken model.
        signal["side"] = None
        signal["setup"] = None
        signal["rationale"].append(f"AI filter unavailable; setup blocked ({str(exc)[:70]})")
    return signal


async def scan_market(symbols):
    semaphore = asyncio.Semaphore(5)
    async def one(symbol):
        async with semaphore:
            try:
                candles = await get_klines(symbol, interval="5m", limit=100)
                signal = analyze(symbol, candles)
                try:
                    f = ai_features(candles)
                    signal["entry_ema_gap"], signal["entry_volatility"] = f[6], f[10]
                except Exception:
                    signal["entry_ema_gap"], signal["entry_volatility"] = 0, 0
                return apply_ai_filter(signal, candles)
            except Exception as exc:
                return {"symbol": symbol, "side": None, "error": str(exc)[:140]}
    return await asyncio.gather(*(one(symbol) for symbol in symbols))

def make_order(signal):
    equity = float(state["equity"])
    risk_usd = equity * RISK_FRACTION
    stop_pct = signal["stop_pct"]
    notional = risk_usd / stop_pct
    max_notional = equity * MAX_LEVERAGE
    notional = min(notional, max_notional)
    qty = notional / signal["price"]
    stop = signal["price"] * (1-stop_pct if signal["side"] == "LONG" else 1+stop_pct)
    target = signal["price"] * (1+2*stop_pct if signal["side"] == "LONG" else 1-2*stop_pct)
    return {"id": f'{int(datetime.now().timestamp()*1000)}-{signal["symbol"]}',
            "symbol": signal["symbol"], "side": signal["side"], "setup": signal["setup"],
            "entry": signal["price"], "qty": qty, "notional": notional,
            "risk_usd": min(risk_usd, notional*stop_pct), "stop": stop, "target": target,
            "status": "PENDING_APPROVAL", "created_at": datetime.now(timezone.utc).isoformat(),
            "rationale": signal["rationale"], "score": signal["score"],
            "ai_confidence": signal.get("ai_confidence"), "ai_probabilities": signal.get("ai_probabilities"),
            "entry_ema_gap": signal.get("entry_ema_gap", 0), "entry_volatility": signal.get("entry_volatility", 0)}

class Decision(BaseModel):
    decision: str = Field(pattern="^(approve|reject)$")

@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE / "static/index.html").read_text()

@app.get("/api/state")
def get_state():
    reset_day()
    return {**state, "config": {"symbols": SYMBOLS, "capital": STARTING_EQUITY,
            "risk_per_trade": RISK_FRACTION, "max_leverage": MAX_LEVERAGE,
            "daily_loss_limit": DAILY_LOSS_LIMIT, "execution": "MANUAL APPROVAL / PAPER ONLY",
            "ai": {"trained": AI_META.get("trained", False), "active": AI_META.get("active", False),
                   "confidence_threshold": AI_META.get("confidence_threshold", 0.58),
                   "validation": AI_META.get("validation")}}}

@app.get("/api/learning")
def learning_status():
    return LEARNER.advice()


class AITrainRequest(BaseModel):
    days: int = Field(default=60, ge=30, le=90)
    top_n: int = Field(default=10, ge=10, le=30)


class AIActivationRequest(BaseModel):
    active: bool
    confidence_threshold: float = Field(default=0.58, ge=0.55, le=0.80)


@app.get("/api/ai/status")
def ai_status():
    return {**AI_META, "model_loaded": AI_MODEL is not None,
            "feature_count": len(AI_FEATURE_NAMES),
            "description": "Three-class model estimates 12-candle direction: SHORT, NEUTRAL or LONG."}


@app.post("/api/ai/train")
async def train_ai_model(req: AITrainRequest):
    """Train a direction classifier with per-symbol chronological holdout validation."""
    global AI_MODEL, AI_META
    try:
        universe_rows = await get_top_usdt_perpetuals(req.top_n)
        symbols = [x["symbol"] for x in universe_rows]
        if not symbols:
            raise HTTPException(502, "No eligible Binance USDT perpetual contracts")
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = end_ms - req.days * 24 * 60 * 60 * 1000
        semaphore = asyncio.Semaphore(4)
        async def fetch(symbol):
            async with semaphore:
                candles = await get_historical_klines(symbol, "5m", start_ms, end_ms)
                return symbol, candles
        fetched = await asyncio.gather(*(fetch(symbol) for symbol in symbols), return_exceptions=True)
        train_x, train_y, test_x, test_y = [], [], [], []
        used_symbols, skipped = [], []
        for item in fetched:
            if isinstance(item, Exception):
                skipped.append(str(item)[:100]); continue
            symbol, candles = item
            if len(candles) < 500:
                skipped.append(f"{symbol}: insufficient candles"); continue
            xs, ys = [], []
            # Every second candle limits correlated samples; labels use the next 12 completed bars.
            for i in range(60, len(candles)-12, 2):
                try:
                    features = ai_features(candles, i)
                    future_return = float(candles[i+12][4]) / float(candles[i][4]) - 1
                    label = 1 if future_return > 0.0035 else (-1 if future_return < -0.0035 else 0)
                    xs.append(features); ys.append(label)
                except Exception:
                    continue
            if len(xs) < 200:
                skipped.append(f"{symbol}: too few feature samples"); continue
            split = int(len(xs) * 0.75)
            train_x.extend(xs[:split]); train_y.extend(ys[:split])
            test_x.extend(xs[split:]); test_y.extend(ys[split:])
            used_symbols.append(symbol)
        if len(train_x) < 1500 or len(test_x) < 300:
            raise HTTPException(422, f"Not enough training data. Training rows={len(train_x)}, validation rows={len(test_x)}. Try more coins or 90 days.")
        if len(set(train_y)) < 3:
            raise HTTPException(422, "Training data does not contain all three market classes; try a different period/universe")
        model = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.07,
                    max_leaf_nodes=15, l2_regularization=1.5, min_samples_leaf=35,
                    random_state=42)
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        accuracy = float(accuracy_score(test_y, predictions))
        balanced = float(balanced_accuracy_score(test_y, predictions))
        matrix = confusion_matrix(test_y, predictions, labels=[-1, 0, 1]).tolist()
        report = classification_report(test_y, predictions, labels=[-1, 0, 1],
                    target_names=["SHORT", "NEUTRAL", "LONG"], output_dict=True, zero_division=0)
        previous_meta = dict(AI_META)
        previous_model_exists = AI_MODEL is not None and AI_META.get("trained")
        previous_balanced = float((previous_meta.get("validation") or {}).get("balanced_accuracy_pct", 0))
        candidate_balanced = balanced * 100
        promote = (not previous_model_exists) or candidate_balanced >= previous_balanced - 1.0
        if promote:
            candidate_path = AI_MODEL_FILE.with_suffix(".candidate")
            joblib.dump(model, candidate_path)
            candidate_path.replace(AI_MODEL_FILE)
            AI_MODEL = model
        AI_META = {"trained": True, "active": bool(previous_meta.get("active")) if promote else bool(previous_meta.get("active")),
                   "confidence_threshold": previous_meta.get("confidence_threshold", 0.58),
                   "trained_at": datetime.now(timezone.utc).isoformat(),
                   "training_days": req.days, "timeframe": "5m", "symbols": used_symbols,
                   "training_samples": len(train_x), "validation_samples": len(test_x),
                   "validation": {"accuracy_pct": round(accuracy*100, 2),
                                  "balanced_accuracy_pct": round(balanced*100, 2),
                                  "confusion_matrix_labels": ["SHORT", "NEUTRAL", "LONG"],
                                  "confusion_matrix": matrix,
                                  "class_recall": {k: round(v["recall"]*100, 2) for k,v in report.items() if k in ("SHORT", "NEUTRAL", "LONG")}},
                   "target_definition": "12-bar forward return: LONG > +0.35%, SHORT < -0.35%, otherwise NEUTRAL",
                   "feature_names": AI_FEATURE_NAMES,
                   "promotion": {"candidate_promoted": promote, "previous_balanced_accuracy_pct": round(previous_balanced, 2), "candidate_balanced_accuracy_pct": round(candidate_balanced, 2), "rule": "Promote only if new balanced accuracy is no more than 1 percentage point below incumbent; otherwise retain incumbent model."},
                   "warning": "Directional classification accuracy is not trading profitability. Use paper validation before relying on this filter.",
                   "skipped": skipped[:10]}
        if not promote:
            # Preserve incumbent model + metadata; expose candidate evaluation without persisting it as active.
            candidate_evaluation = AI_META.get("promotion")
            candidate_validation = AI_META.get("validation")
            AI_META = previous_meta
            save_ai_meta()
            return {"ok": True, **AI_META, "candidate_evaluation": candidate_evaluation,
                    "candidate_validation": candidate_validation, "candidate_promoted": False}
        save_ai_meta()
        return {"ok": True, **AI_META}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"AI training failed: {str(exc)[:220]}")


@app.post("/api/ai/activate")
def activate_ai(body: AIActivationRequest):
    global AI_META
    if body.active:
        if AI_MODEL is None or not AI_META.get("trained"):
            raise HTTPException(409, "Train the AI model before activation")
        validation = AI_META.get("validation", {})
        if validation.get("balanced_accuracy_pct", 0) < 35:
            raise HTTPException(409, "Validation is too weak to activate safely (balanced accuracy below 35%)")
    AI_META["active"] = body.active
    AI_META["confidence_threshold"] = body.confidence_threshold
    save_ai_meta()
    return ai_status()



async def periodic_retraining():
    """Weekly background candidate training; incumbent stays active unless candidate passes promotion gate."""
    while True:
        try:
            await asyncio.sleep(7 * 24 * 60 * 60)
            await train_ai_model(AITrainRequest(days=60, top_n=10))
        except asyncio.CancelledError:
            raise
        except Exception:
            # Keep the service alive; next scheduled cycle retries.
            await asyncio.sleep(60)


@app.on_event("startup")
async def start_periodic_retraining():
    app.state.learning_task = asyncio.create_task(periodic_retraining())


@app.on_event("shutdown")
async def stop_periodic_retraining():
    task = getattr(app.state, "learning_task", None)
    if task:
        task.cancel()

class ScanRequest(BaseModel):
    top_n: int = Field(default=10, ge=10, le=50)

@app.get("/api/universe")
async def universe(limit: int = 10):
    if limit not in (10, 20, 30, 40, 50):
        raise HTTPException(400, "Choose 10, 20, 30, 40 or 50 coins")
    try:
        coins = await get_top_usdt_perpetuals(limit)
        return {"count": len(coins), "ranking": "24h USDT quote volume", "coins": coins}
    except Exception as e:
        raise HTTPException(502, f"Could not load Binance Futures universe: {str(e)[:160]}")

@app.post("/api/scan")
async def scan(req: ScanRequest):
    reset_day()
    if state["paused"]: raise HTTPException(423, "Trading is paused")
    if state["daily_pnl"] <= -STARTING_EQUITY*DAILY_LOSS_LIMIT:
        state["paused"] = True; save()
        raise HTTPException(423, "Daily loss limit reached")
    try:
        universe_rows = await get_top_usdt_perpetuals(req.top_n)
        symbols = [x["symbol"] for x in universe_rows]
        if not symbols:
            raise HTTPException(502, "Binance returned no eligible USDT perpetual contracts")
        signals = await scan_market(symbols)
    except Exception as e:
        raise HTTPException(502, f"Binance market data unavailable: {str(e)[:180]}")
    state["last_scan"] = datetime.now(timezone.utc).isoformat()
    active_symbols = {p["symbol"] for p in state["positions"]} | {p["symbol"] for p in state["pending"]}
    for sig in signals:
        if sig.get("side") and sig["symbol"] not in active_symbols and len(state["positions"])+len(state["pending"]) < MAX_POSITIONS:
            state["pending"].append(make_order(sig))
            active_symbols.add(sig["symbol"])
    save()
    return {"universe": symbols, "ranking": "24h USDT quote volume",
            "signals": signals, "pending": state["pending"]}

@app.post("/api/orders/{order_id}/decision")
def decide(order_id: str, body: Decision):
    reset_day()
    order = next((x for x in state["pending"] if x["id"] == order_id), None)
    if not order: raise HTTPException(404, "Pending order not found")
    state["pending"].remove(order)
    LEARNER.record_decision(body.decision)
    if body.decision == "approve":
        if state["paused"]: raise HTTPException(423, "Trading is paused")
        order["status"] = "OPEN"
        order["approved_at"] = datetime.now(timezone.utc).isoformat()
        state["positions"].append(order)
    else:
        order["status"] = "REJECTED"
        state["trades"].append(order)
    save()
    return {"ok": True, "state": state}

@app.post("/api/positions/{order_id}/close")
def close_position(order_id: str):
    pos = next((x for x in state["positions"] if x["id"] == order_id), None)
    if not pos: raise HTTPException(404, "Open paper position not found")
    # Manual close uses latest mark price from Binance.
    import requests
    try:
        r = requests.get(f"{BINANCE}/fapi/v1/ticker/price", params={"symbol": pos["symbol"]}, timeout=8)
        r.raise_for_status(); price = float(r.json()["price"])
    except Exception as e:
        raise HTTPException(502, f"Could not get mark price: {e}")
    pnl = (price-pos["entry"])*pos["qty"]*(1 if pos["side"]=="LONG" else -1)
    # Conservative simulated trading cost: 0.05% each side of notional.
    fees = pos["notional"]*0.001
    net = pnl-fees
    pos.update({"exit": price, "gross_pnl": pnl, "fees_estimate": fees, "net_pnl": net,
                "status": "CLOSED", "closed_at": datetime.now(timezone.utc).isoformat()})
    state["equity"] += net
    state["realized_pnl"] += net
    state["daily_pnl"] += net
    state["positions"].remove(pos)
    state["trades"].append(pos)
    LEARNER.record_closed_trade(pos)
    if state["daily_pnl"] <= -STARTING_EQUITY*DAILY_LOSS_LIMIT: state["paused"] = True
    save()
    return {"ok": True, "closed": pos, "state": state}

@app.post("/api/pause")
def pause():
    state["paused"] = not state["paused"]; save()
    return {"paused": state["paused"]}

@app.post("/api/reset")
def reset():
    state.clear(); state.update(default_state()); save()
    return {"ok": True, "message": "Paper account reset to $1,000. Trade history cleared."}



async def get_historical_klines(symbol: str, interval: str, start_ms: int, end_ms: int):
    """Fetch Binance USD-M Futures candles in bounded pages."""
    rows = []
    cursor = start_ms
    async with httpx.AsyncClient(timeout=25) as client:
        while cursor < end_ms:
            r = await client.get(f"{BINANCE}/fapi/v1/klines",
                                 params={"symbol": symbol, "interval": interval,
                                         "startTime": cursor, "endTime": end_ms, "limit": 1500})
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1][0]) + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < 1500:
                break
    # Drop the currently forming candle, if present.
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return [c for c in rows if int(c[6]) < now_ms]

def simulate_symbol_backtest(symbol: str, candles: list, allocation: float, strategy: str = "COMBINED"):
    """One-position-per-symbol, next-bar entry, conservative intrabar exit model."""
    equity = allocation
    peak = allocation
    max_dd = 0.0
    trades = []
    position = None
    daily_pnl = {}
    halted_days = set()
    fee_rate = 0.0005       # assumed taker fee per side; adjust to account tier
    slippage_rate = 0.0002  # conservative assumed slippage per side
    cost_rate = fee_rate + slippage_rate

    def close_trade(pos, raw_exit, reason, timestamp):
        nonlocal equity, peak, max_dd
        exit_price = raw_exit * (1 - cost_rate if pos["side"] == "LONG" else 1 + cost_rate)
        gross = (exit_price - pos["entry"]) * pos["qty"] * (1 if pos["side"] == "LONG" else -1)
        entry_cost = pos["entry"] * pos["qty"] * cost_rate
        net = gross - entry_cost
        equity += net
        day = datetime.fromtimestamp(timestamp/1000, tz=timezone.utc).date().isoformat()
        daily_pnl[day] = daily_pnl.get(day, 0.0) + net
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak-equity)/peak if peak else 0.0)
        trades.append({"symbol": symbol, "side": pos["side"], "entry_time": pos["entry_time"],
                       "exit_time": datetime.fromtimestamp(timestamp/1000, tz=timezone.utc).isoformat(),
                       "entry": round(pos["raw_entry"], 8), "exit": round(raw_exit, 8),
                       "stop": round(pos["stop"], 8), "target": round(pos["target"], 8),
                       "qty": round(pos["qty"], 8), "notional": round(pos["notional"], 2),
                       "gross_pnl": round(gross, 4), "estimated_costs": round(entry_cost + abs(exit_price)*pos["qty"]*cost_rate, 4),
                       "net_pnl": round(net, 4), "exit_reason": reason,
                       "equity_after": round(equity, 2)})
        return None

    for i in range(60, len(candles)-1):
        candle = candles[i]
        ts = int(candle[0])
        day = datetime.fromtimestamp(ts/1000, tz=timezone.utc).date().isoformat()

        if position:
            high, low, op = float(candle[2]), float(candle[3]), float(candle[1])
            if position["side"] == "LONG":
                stop_hit, target_hit = low <= position["stop"], high >= position["target"]
                if stop_hit or target_hit:
                    # If both levels are touched in one candle, assume the adverse stop happens first.
                    if stop_hit:
                        raw_exit = min(op, position["stop"])
                        reason = "STOP (conservative same-bar ordering)" if target_hit else "STOP"
                    else:
                        raw_exit, reason = position["target"], "TARGET"
                    position = close_trade(position, raw_exit, reason, ts)
            else:
                stop_hit, target_hit = high >= position["stop"], low <= position["target"]
                if stop_hit or target_hit:
                    if stop_hit:
                        raw_exit = max(op, position["stop"])
                        reason = "STOP (conservative same-bar ordering)" if target_hit else "STOP"
                    else:
                        raw_exit, reason = position["target"], "TARGET"
                    position = close_trade(position, raw_exit, reason, ts)

        if position is None and day not in halted_days and daily_pnl.get(day, 0.0) > -(allocation * DAILY_LOSS_LIMIT):
            sig = analyze(symbol, candles[max(0, i-99):i+1])
            if sig["side"] and (strategy == "COMBINED" or sig["side"] == strategy):
                entry = float(candles[i+1][1])  # next candle open avoids same-candle look-ahead
                stop_pct = sig["stop_pct"]
                risk_budget = equity * RISK_FRACTION
                notional = min(risk_budget / stop_pct, equity * MAX_LEVERAGE)
                qty = notional / entry
                if notional > 5 and qty > 0:
                    side = sig["side"]
                    effective_entry = entry * (1 + cost_rate if side == "LONG" else 1 - cost_rate)
                    stop = effective_entry * (1-stop_pct if side == "LONG" else 1+stop_pct)
                    target = effective_entry * (1+2*stop_pct if side == "LONG" else 1-2*stop_pct)
                    position = {"side": side, "entry": effective_entry, "raw_entry": entry,
                                "entry_time": datetime.fromtimestamp(int(candles[i+1][0])/1000, tz=timezone.utc).isoformat(),
                                "qty": qty, "notional": notional, "stop": stop, "target": target}

    # Mark any still-open trade at final candle close.
    if position:
        last = candles[-1]
        position = close_trade(position, float(last[4]), "END OF TEST", int(last[0]))

    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    losses = sum(1 for t in trades if t["net_pnl"] <= 0)
    gross_profit = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    gross_loss = abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0))
    return {"symbol": symbol, "starting_equity": round(allocation, 2), "ending_equity": round(equity, 2),
            "net_pnl": round(equity-allocation, 2), "return_pct": round((equity/allocation-1)*100, 2),
            "trades": len(trades), "wins": wins, "losses": losses,
            "win_rate_pct": round(wins/len(trades)*100, 2) if trades else 0,
            "profit_factor": round(gross_profit/gross_loss, 2) if gross_loss else (None if gross_profit == 0 else "∞"),
            "max_drawdown_pct": round(max_dd*100, 2), "trade_history": trades}

class BacktestRequest(BaseModel):
    days: int = Field(default=90, ge=7, le=365)
    interval: str = Field(default="5m", pattern="^(5m|15m)$")
    top_n: int = Field(default=10, ge=10, le=50)
    strategy: str = Field(default="COMBINED", pattern="^(LONG|SHORT|COMBINED)$")


def summarize_portfolio(results):
    total_pnl = sum(x["net_pnl"] for x in results)
    total_trades = sum(x["trades"] for x in results)
    total_wins = sum(x["wins"] for x in results)
    gross_profit = sum(sum(t["net_pnl"] for t in x["trade_history"] if t["net_pnl"] > 0) for x in results)
    gross_loss = abs(sum(sum(t["net_pnl"] for t in x["trade_history"] if t["net_pnl"] < 0) for x in results))
    return {
        "starting_equity": STARTING_EQUITY,
        "ending_equity": round(STARTING_EQUITY + total_pnl, 2),
        "net_pnl": round(total_pnl, 2),
        "return_pct": round(total_pnl / STARTING_EQUITY * 100, 2),
        "trades": total_trades,
        "wins": total_wins,
        "losses": total_trades - total_wins,
        "win_rate_pct": round(total_wins / total_trades * 100, 2) if total_trades else 0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else (None if gross_profit == 0 else "∞"),
    }


@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest):
    try:
        universe_rows = await get_top_usdt_perpetuals(req.top_n)
        symbols = [x["symbol"] for x in universe_rows]
    except Exception as e:
        raise HTTPException(502, f"Could not load Binance Futures universe: {str(e)[:160]}")
    if not symbols:
        raise HTTPException(502, "Binance returned no eligible USDT perpetual contracts")
    if req.top_n >= 30 and req.days > 90:
        raise HTTPException(400, "For 30–50 coins, choose 90 days or less to keep historical data requests manageable")

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - req.days * 24 * 60 * 60 * 1000
    allocation = STARTING_EQUITY / len(symbols)
    strategies = ("LONG", "SHORT", "COMBINED")
    results_by_strategy = {name: [] for name in strategies}

    # Download each symbol once, then replay the exact same candles for all three modes.
    for symbol in symbols:
        try:
            candles = await get_historical_klines(symbol, req.interval, start_ms, end_ms)
        except Exception as e:
            raise HTTPException(502, f"Historical Binance data failed for {symbol}: {str(e)[:160]}")
        if len(candles) < 100:
            raise HTTPException(422, f"Not enough historical candles for {symbol}")

        for strategy_name in strategies:
            result = simulate_symbol_backtest(symbol, candles, allocation, strategy_name)
            result["candles"] = len(candles)
            result["first_candle"] = datetime.fromtimestamp(int(candles[0][0]) / 1000, tz=timezone.utc).isoformat()
            result["last_candle"] = datetime.fromtimestamp(int(candles[-1][0]) / 1000, tz=timezone.utc).isoformat()
            results_by_strategy[strategy_name].append(result)

    summaries = {name: summarize_portfolio(rows) for name, rows in results_by_strategy.items()}
    selected_rows = results_by_strategy[req.strategy]
    selected_summary = summaries[req.strategy]

    # Per-symbol side-by-side comparison makes coin-level strengths/weaknesses visible.
    per_symbol_comparison = []
    for index, symbol in enumerate(symbols):
        row = {"symbol": symbol, "candles": results_by_strategy["COMBINED"][index]["candles"]}
        for strategy_name in strategies:
            item = results_by_strategy[strategy_name][index]
            row[strategy_name.lower()] = {
                "trades": item["trades"],
                "win_rate_pct": item["win_rate_pct"],
                "net_pnl": item["net_pnl"],
                "return_pct": item["return_pct"],
                "max_drawdown_pct": item["max_drawdown_pct"],
                "profit_factor": item["profit_factor"],
            }
        per_symbol_comparison.append(row)

    return {
        "mode": "HISTORICAL_BACKTEST",
        "selected_strategy": req.strategy,
        "comparison": summaries,
        **selected_summary,
        "universe": symbols,
        "ranking": "Current top USDT perpetuals by 24h quote volume",
        "symbols": selected_rows,
        "per_symbol_comparison": per_symbol_comparison,
        "assumptions": {
            "allocation": "Equal allocation across selected symbols; each strategy is independently simulated from the same $1,000 starting capital",
            "risk_per_trade": "0.5% of each symbol sleeve equity",
            "max_leverage": "2x",
            "reward_risk": "2R",
            "fees_per_side": "0.05% assumed",
            "slippage_per_side": "0.02% assumed",
            "intrabar_rule": "If stop and target both touch in one candle, stop is assumed first",
            "funding": "Not included",
            "daily_loss_limit": "1.5% of each symbol sleeve starting equity; no new trades for that UTC day after limit",
            "universe_bias": "Uses today's top-volume contracts, not the historical point-in-time universe; survivorship bias is possible",
        },
    }

@app.get("/health")
def health():
    return {"status": "ok", "mode": "paper-only"}

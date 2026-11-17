"""Conservative online learning from completed paper trades."""
import json
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

class TradeLearner:
    def __init__(self, path: Path):
        self.path = path
        self.data = {"version": 1, "closed_trade_count": 0, "approved_count": 0,
                     "rejected_count": 0, "groups": {}, "recent": []}
        self.load()

    def load(self):
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
                if isinstance(loaded, dict):
                    self.data.update(loaded)
            except Exception:
                pass

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)

    @staticmethod
    def regime_from_trade(trade):
        gap = float(trade.get("entry_ema_gap", 0) or 0)
        vol = float(trade.get("entry_volatility", 0) or 0)
        if vol >= 0.012:
            return "HIGH_VOLATILITY"
        if abs(gap) >= 0.004:
            return "TRENDING"
        return "RANGE"

    def record_decision(self, decision):
        key = "approved_count" if decision == "approve" else "rejected_count"
        self.data[key] = int(self.data.get(key, 0)) + 1
        self.save()

    def record_closed_trade(self, trade):
        side = trade.get("side", "UNKNOWN")
        setup = trade.get("setup", "UNKNOWN")
        regime = self.regime_from_trade(trade)
        pnl = float(trade.get("net_pnl", 0) or 0)
        risk = max(float(trade.get("risk_usd", 5) or 5), 0.01)
        r_multiple = pnl / risk
        group_keys = [f"side:{side}", f"setup:{setup}", f"regime:{regime}",
                      f"setup_regime:{setup}|{regime}", f"side_regime:{side}|{regime}"]
        for key in group_keys:
            group = self.data.setdefault("groups", {}).setdefault(
                key, {"trades": 0, "wins": 0, "net_pnl": 0.0, "sum_r": 0.0, "recent_r": []})
            group["trades"] += 1
            group["wins"] += int(pnl > 0)
            group["net_pnl"] = round(float(group["net_pnl"]) + pnl, 4)
            group["sum_r"] = round(float(group["sum_r"]) + r_multiple, 4)
            group["recent_r"] = (group.get("recent_r", []) + [round(r_multiple, 4)])[-50:]
        self.data["closed_trade_count"] = int(self.data.get("closed_trade_count", 0)) + 1
        self.data["recent"] = (self.data.get("recent", []) + [{
            "symbol": trade.get("symbol"), "side": side, "setup": setup, "regime": regime,
            "net_pnl": round(pnl, 4), "r_multiple": round(r_multiple, 3),
            "closed_at": trade.get("closed_at") or datetime.now(timezone.utc).isoformat()
        }])[-100:]
        self.save()

    def advice(self):
        """Return conservative, sample-size-aware performance insights; never changes risk."""
        groups = self.data.get("groups", {})
        insights = []
        for key, value in groups.items():
            n = int(value.get("trades", 0))
            if n < 5:
                continue
            recent = value.get("recent_r", [])[-20:]
            avg_r = sum(recent) / len(recent) if recent else 0.0
            # Shrink estimates toward zero until there is a meaningful sample.
            confidence = min(1.0, n / 20.0)
            adjusted = avg_r * confidence
            insights.append({"segment": key, "trades": n,
                "win_rate_pct": round(value.get("wins", 0) / n * 100, 1),
                "recent_avg_r": round(avg_r, 3),
                "confidence": round(confidence, 2),
                "adjusted_score": round(adjusted, 3)})
        insights.sort(key=lambda x: x["adjusted_score"], reverse=True)
        return {"closed_trade_count": self.data.get("closed_trade_count", 0),
                "approved_count": self.data.get("approved_count", 0),
                "rejected_count": self.data.get("rejected_count", 0),
                "insights": insights[:12], "recent": self.data.get("recent", [])[-15:],
                "policy": "Advisory only: outcomes are grouped by side, setup and market regime. No automatic risk increase or trade execution."}

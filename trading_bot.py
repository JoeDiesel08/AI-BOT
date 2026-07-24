"""Continuous/paper/live trading bot using a trained TradingAgent and Kraken."""
import os
import json
import time
from pathlib import Path
from datetime import datetime

from kraken_executor import KrakenExecutor
from trading_agents import TradingAgent, HOLD, BUY, SELL


# Configuration from environment
TRADING_PAIR = os.environ.get("TRADING_PAIR", "BTC/USD")
TIMEFRAME = os.environ.get("TIMEFRAME", "4h")
DATA_LIMIT = int(os.environ.get("DATA_LIMIT", "200"))
PAPER_TRADING = os.environ.get("PAPER_TRADING", "true").lower() in ("1", "true", "yes")
KRAKEN_API_KEY = os.environ.get("KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.environ.get("KRAKEN_API_SECRET")
LOOP_INTERVAL_SECONDS = int(os.environ.get("LOOP_INTERVAL_SECONDS", "0"))
DATA_DIR = os.environ.get("DATA_DIR", "/data/kraken_paper")
PAPER_CASH = float(os.environ.get("PAPER_CASH", "1000.0"))
PAPER_CRYPTO = float(os.environ.get("PAPER_CRYPTO", "0.0"))
_seed_path = os.environ.get("SEED_FILE", "best_agent.json")
SEED_FILE = Path(_seed_path) if os.path.isabs(_seed_path) else Path(__file__).parent / _seed_path
KRAKEN_VALIDATE = os.environ.get("KRAKEN_VALIDATE", "false").lower() in ("1", "true", "yes")
FEE_MAKER = float(os.environ.get("KRAKEN_FEE_MAKER", "0.0025"))
FEE_TAKER = float(os.environ.get("KRAKEN_FEE_TAKER", "0.0040"))
SLIPPAGE = float(os.environ.get("KRAKEN_SLIPPAGE", "0.0005"))


class TradingBot:
    """Runs the best trained agent against live Kraken spot data."""

    def __init__(self):
        self.executor = KrakenExecutor(
            pair=TRADING_PAIR,
            is_paper=PAPER_TRADING,
            api_key=KRAKEN_API_KEY,
            api_secret=KRAKEN_API_SECRET,
            data_dir=DATA_DIR,
            fee_maker=FEE_MAKER,
            fee_taker=FEE_TAKER,
            slippage=SLIPPAGE,
            validate=KRAKEN_VALIDATE,
            paper_cash=PAPER_CASH,
            paper_crypto=PAPER_CRYPTO,
        )
        self.agent = self._load_best_agent()

    def _load_best_agent(self) -> TradingAgent:
        if not SEED_FILE.exists():
            raise FileNotFoundError(
                f"No trained agent seed found at {SEED_FILE}. "
                "Run `python main.py` with MODE=optimize first to generate best_agent.json."
            )
        with open(SEED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        params = data.get("params", data)
        return TradingAgent.from_dict(params)

    def run_once(self):
        """Fetch latest Kraken data, get a signal, execute one iteration, and log telemetry."""
        now = datetime.utcnow().isoformat()
        df = self.executor.fetch_ohlcv(timeframe=TIMEFRAME, limit=DATA_LIMIT)
        if df.empty:
            print(f"[{now}] No market data returned; skipping iteration.")
            return

        decisions = self.agent.evaluate_market(df)
        current_decision = int(decisions[-1])

        action_label = {HOLD: "HOLD", BUY: "BUY", SELL: "SELL"}.get(current_decision, "HOLD")
        print(f"\n[{now}] Kraken signal: {action_label} ({len(df)} candles evaluated)")

        side, volume, fill_price = self.executor.execute_decision(current_decision)
        if side != "hold":
            print(f"Executed {side}: volume={volume:.8f} BTC, fill_price=${fill_price:,.2f}")

        self.executor.log_telemetry()

    def run(self):
        """Run one iteration, then loop if LOOP_INTERVAL_SECONDS is set."""
        self.run_once()

        if LOOP_INTERVAL_SECONDS > 0:
            print(f"\nTrading loop active (interval={LOOP_INTERVAL_SECONDS}s). Press Ctrl+C to stop.")
            while True:
                time.sleep(LOOP_INTERVAL_SECONDS)
                try:
                    self.run_once()
                except Exception as e:
                    print(f"Iteration error: {e}")


def main():
    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()

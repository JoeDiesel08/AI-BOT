"""Kraken execution wrapper supporting both paper and live trading modes."""
import os
import json
import math
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import ccxt

# Pull in the same decision codes used by TradingAgent
from trading_agents import HOLD, BUY, SELL

# Kraken starter-tier fee schedule defaults
DEFAULT_FEE_MAKER = 0.0025
DEFAULT_FEE_TAKER = 0.0040
DEFAULT_SLIPPAGE = 0.0005


class KrakenExecutor:
    """
    Dual-mode Kraken executor.

    - Paper mode: fetches live public Kraken market data and simulates fills with
      Kraken-like fees + slippage. All state is persisted to a local directory.
    - Live mode: forwards orders to Kraken's authenticated private REST API
      via CCXT (which signs and sends the /0/private/AddOrder payload).
    """

    def __init__(
        self,
        pair: str = "BTC/USD",
        is_paper: bool = True,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        data_dir: str = "/data/kraken_paper",
        fee_maker: float = DEFAULT_FEE_MAKER,
        fee_taker: float = DEFAULT_FEE_TAKER,
        slippage: float = DEFAULT_SLIPPAGE,
        validate: bool = False,
        paper_cash: float = 1000.0,
        paper_crypto: float = 0.0,
    ):
        self.pair = pair
        self.is_paper = is_paper
        self.data_dir = Path(data_dir)
        self.fee_maker = fee_maker
        self.fee_taker = fee_taker
        self.slippage = slippage
        self.validate = validate

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio_file = self.data_dir / "portfolio.json"
        self.trades_file = self.data_dir / "trades.jsonl"
        self.equity_file = self.data_dir / "equity_log.jsonl"

        config = {"enableRateLimit": True, "options": {"defaultType": "spot"}}
        if not is_paper:
            if not api_key or not api_secret:
                raise ValueError("Live trading requires KRAKEN_API_KEY and KRAKEN_API_SECRET")
            config["apiKey"] = api_key
            config["secret"] = api_secret

        self.exchange = ccxt.kraken(config)

        self.state = self._load_state()
        if self.state is None:
            self.state = {
                "cash": float(paper_cash) if is_paper else 0.0,
                "crypto": float(paper_crypto) if is_paper else 0.0,
                "entry_price": 0.0,
                "realized_pnl": 0.0,
                "total_fees": 0.0,
                "trade_count": 0,
                "last_price": 0.0,
                "last_updated": None,
            }
            self._save_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def _load_state(self):
        if not self.portfolio_file.exists():
            return None
        try:
            with open(self.portfolio_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_state(self):
        try:
            with open(self.portfolio_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"Warning: could not save portfolio state: {e}")

    def _append_line(self, path: Path, record: dict):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            print(f"Warning: could not write {path}: {e}")

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def fetch_price(self) -> float:
        """Live last traded price from Kraken public REST."""
        ticker = self.exchange.fetch_ticker(self.pair)
        price = float(ticker.get("last", 0.0) or 0.0)
        if price <= 0:
            raise ValueError(f"Invalid ticker price for {self.pair}: {ticker}")
        self.state["last_price"] = price
        self.state["last_updated"] = datetime.utcnow().isoformat()
        return price

    def fetch_ohlcv(self, timeframe: str = "4h", limit: int = 200):
        """Fetch historical OHLCV candles from Kraken."""
        ohlcv = self.exchange.fetch_ohlcv(self.pair, timeframe=timeframe, limit=limit)
        if not ohlcv:
            raise ValueError(f"No OHLCV data returned for {self.pair}")

        import pandas as pd

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------
    def execute_decision(self, decision: int, price: Optional[float] = None):
        """
        Convert a TradingAgent decision (HOLD/BUY/SELL) into a trade.

        Returns (side, volume, fill_price).  side == 'hold' means no action.
        """
        if price is None:
            price = self.fetch_price()
        else:
            self.state["last_price"] = float(price)
            self.state["last_updated"] = datetime.utcnow().isoformat()

        side = None
        if decision == BUY:
            if self.state["crypto"] <= 1e-12:
                side = "buy"
        elif decision == SELL:
            if self.state["crypto"] > 1e-12:
                side = "sell"

        if side is None:
            self._log_equity(price)
            self._save_state()
            return "hold", 0.0, price

        amount = self._get_trade_size(side, price)
        if amount <= 1e-12:
            self._log_equity(price)
            self._save_state()
            return "hold", 0.0, price

        return self.place_order(side, amount, order_type="market", price=price)

    def _get_trade_size(self, side: str, price: float) -> float:
        if side == "buy":
            cash = self.state["cash"]
            if cash <= 0:
                return 0.0
            # Use 95% of cash to leave room for fees + slippage
            return (cash * 0.95) / price
        return self.state["crypto"]

    def place_order(self, side: str, volume: float, order_type: str = "market", price: Optional[float] = None):
        if price is None:
            price = self.fetch_price()
        price = float(price)

        if self.is_paper:
            return self._paper_fill(side, volume, order_type, price)
        return self._live_order(side, volume, order_type, price)

    def _paper_fill(self, side: str, volume: float, order_type: str, price: float):
        """Simulate a Kraken fill with fees and slippage."""
        if side == "buy":
            fill_price = price * (1 + self.slippage)
        else:
            fill_price = price * (1 - self.slippage)

        fee_rate = self.fee_maker if order_type == "limit" else self.fee_taker
        notional = volume * fill_price
        fee = notional * fee_rate

        if side == "buy":
            cost = notional + fee
            if cost > self.state["cash"]:
                volume = (self.state["cash"] / (fill_price * (1 + fee_rate))) * 0.999
                if volume <= 1e-12:
                    return "hold", 0.0, price
                notional = volume * fill_price
                fee = notional * fee_rate
                cost = notional + fee

            old_crypto = self.state["crypto"]
            self.state["cash"] -= cost
            self.state["crypto"] += volume
            # Weighted average entry for pyramiding, but since we only enter flat, it's simple
            if old_crypto > 1e-12:
                old_cost = old_crypto * self.state["entry_price"]
                new_cost = volume * fill_price
                self.state["entry_price"] = (old_cost + new_cost) / (old_crypto + volume)
            else:
                self.state["entry_price"] = fill_price
        else:
            proceeds = notional - fee
            old_entry = self.state.get("entry_price", 0.0) or 0.0
            realized = proceeds - (old_entry * volume)
            self.state["cash"] += proceeds
            self.state["crypto"] -= volume
            self.state["realized_pnl"] += realized
            if self.state["crypto"] <= 1e-12:
                self.state["entry_price"] = 0.0

        self.state["total_fees"] += fee
        self.state["trade_count"] += 1
        self.state["last_price"] = price
        self.state["last_updated"] = datetime.utcnow().isoformat()

        trade_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "pair": self.pair,
            "side": side,
            "type": order_type,
            "volume": volume,
            "price": price,
            "fill_price": fill_price,
            "fee": fee,
            "fee_rate": fee_rate,
            "pnl": self.state["realized_pnl"] if side == "sell" else 0.0,
            "mode": "paper",
        }
        self._append_line(self.trades_file, trade_record)
        self._log_equity(price)
        self._save_state()
        return side, volume, fill_price

    def _live_order(self, side: str, volume: float, order_type: str, price: float):
        """Send a real order to Kraken's authenticated /0/private/AddOrder endpoint."""
        params = {}
        if self.validate:
            # Kraken's validate flag performs a dry run without placing an order
            params["validate"] = True

        order = self.exchange.create_order(
            symbol=self.pair,
            type=order_type,
            side=side,
            amount=volume,
            price=None if order_type == "market" else price,
            params=params,
        )
        order_id = order.get("id", "unknown")
        print(
            f"Live {'VALIDATE ' if self.validate else ''}order placed: "
            f"id={order_id}, side={side}, volume={volume:.8f}, pair={self.pair}"
        )

        self._sync_balances()
        self.state["trade_count"] += 1
        self._save_state()

        trade_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "pair": self.pair,
            "side": side,
            "type": order_type,
            "volume": volume,
            "price": price,
            "order_id": order_id,
            "mode": "live",
            "validate": self.validate,
        }
        self._append_line(self.trades_file, trade_record)
        self._log_equity(self.state.get("last_price", price))
        return side, volume, price

    def _sync_balances(self):
        """Pull live USD/BTC balances from Kraken and update local state."""
        try:
            balance = self.exchange.fetch_balance()
            free = balance.get("free", {})
            self.state["cash"] = float(free.get("USD", 0.0) or free.get("ZUSD", 0.0))
            self.state["crypto"] = float(
                free.get("BTC", 0.0) or free.get("XBT", 0.0) or free.get("XXBT", 0.0)
            )
            self.state["last_updated"] = datetime.utcnow().isoformat()
        except Exception as e:
            print(f"Warning: could not sync live balances: {e}")

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------
    def _log_equity(self, price: float):
        equity = self.state["cash"] + self.state["crypto"] * price
        unrealized = 0.0
        if self.state["crypto"] > 1e-12 and self.state.get("entry_price", 0) > 0:
            unrealized = (price - self.state["entry_price"]) * self.state["crypto"]

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "equity": equity,
            "cash": self.state["cash"],
            "crypto": self.state["crypto"],
            "price": price,
            "realized_pnl": self.state["realized_pnl"],
            "unrealized_pnl": unrealized,
            "total_fees": self.state["total_fees"],
        }
        self._append_line(self.equity_file, record)

    def log_telemetry(self):
        """Print a highly readable markdown balance/position/equity report to stdout."""
        try:
            price = self.fetch_price()
        except Exception:
            price = self.state.get("last_price", 0.0) or 0.0

        equity = self.state["cash"] + self.state["crypto"] * price
        unrealized = 0.0
        if self.state["crypto"] > 1e-12 and self.state.get("entry_price", 0) > 0:
            unrealized = (price - self.state["entry_price"]) * self.state["crypto"]

        mode_label = "PAPER" if self.is_paper else "LIVE"

        print("\n## Kraken Trading Telemetry\n")
        print(f"**Mode:** {mode_label}  **Pair:** {self.pair}  **Price:** ${price:,.2f}")
        print(f"**Data Directory:** {self.data_dir}")
        print(f"**Last Updated:** {self.state.get('last_updated', 'N/A')}\n")

        print("| Metric | Value |")
        print("|--------|-------|")
        print(f"| Cash (USD) | ${self.state['cash']:,.2f} |")
        print(f"| Crypto (BTC) | {self.state['crypto']:.8f} |")
        print(f"| Total Equity | ${equity:,.2f} |")
        print(f"| Unrealized P&L | ${unrealized:,.2f} |")
        print(f"| Realized P&L | ${self.state['realized_pnl']:,.2f} |")
        print(f"| Total Fees | ${self.state['total_fees']:,.2f} |")
        if self.is_paper:
            start_value = float(os.environ.get("PAPER_CASH", 1000.0))
            total_return_pct = ((equity / start_value) - 1) * 100 if start_value else 0.0
            print(f"| Total Return | {total_return_pct:+.2f}% |")
        print("")

        print("### Open Position\n")
        if self.state["crypto"] > 1e-12:
            print("| Side | Entry Price | Current Price | Size | Unrealized P&L |")
            print("|------|-------------|---------------|------|----------------|")
            print(
                f"| LONG | ${self.state['entry_price']:,.2f} | ${price:,.2f} | "
                f"{self.state['crypto']:.8f} | ${unrealized:,.2f} |"
            )
        else:
            print("None (flat)")
        print("")

        recent = []
        if self.trades_file.exists():
            try:
                with open(self.trades_file, "r", encoding="utf-8") as f:
                    lines = [line for line in f.readlines() if line.strip()]
                for line in lines[-5:]:
                    recent.append(json.loads(line))
            except Exception:
                pass

        if recent:
            print("### Recent Trades\n")
            print("| Time | Mode | Side | Type | Volume | Fill/Order Price | Fee |")
            print("|------|------|------|------|--------|------------------|-----|")
            for t in recent:
                fill_price = t.get("fill_price") or t.get("price") or 0
                fee = t.get("fee") or 0.0
                print(
                    f"| {t.get('timestamp','')} | {t.get('mode','')} | {t.get('side','')} | "
                    f"{t.get('type','')} | {t.get('volume',0):.8f} | ${fill_price:,.2f} | ${fee:,.4f} |"
                )
            print("")

import pandas as pd
import numpy as np
import math
from decimal import Decimal, ROUND_FLOOR


def _floor_to_precision(value: float, precision: float) -> float:
    """
    Floor a numeric value to a given quantity precision.

    Supports both integer decimal places (e.g., 4) and float step sizes
    (e.g., 0.0001). Uses Decimal to avoid floating-point rounding errors.
    """
    if precision < 0:
        return 0.0

    frac, _ = math.modf(precision)
    value_dec = Decimal(str(value))

    if math.isclose(frac, 0.0, abs_tol=1e-9):
        # Precision is an integer count of decimal places
        places = int(round(precision))
        if places == 0:
            quantizer = Decimal("1")
        else:
            quantizer = Decimal("1." + "0" * places)
        return float(value_dec.quantize(quantizer, rounding=ROUND_FLOOR))
    else:
        # Precision is a step size (e.g., 0.0001)
        step = Decimal(str(precision))
        steps = (value_dec / step).to_integral_value(rounding=ROUND_FLOOR)
        return float(steps * step)


def calculate_position_size(
    account_balance: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    market_limits: dict
) -> float:
    """
    Calculate the exact trade quantity for a crypto position based on risk rules.

    The quantity is derived from the dollar amount willing to be risked divided by
    the per-unit risk distance, then floored to the exchange's quantity precision.
    It is capped by available equity and validated against the minimum order size.

    Args:
        account_balance: Total equity in the quote currency (e.g., USDT).
        risk_pct: Percentage of account balance to risk per trade (e.g., 0.01 for 1%).
        entry_price: Planned position entry price.
        stop_loss: Planned stop-loss price.
        market_limits: Dictionary with exchange market metadata.
                       Must contain 'min_qty' and 'qty_precision' keys.
                       'qty_precision' can be an integer (decimal places) or a
                       float step size (e.g., 0.0001).

    Returns:
        The floored, validated trade quantity. Returns 0.0 if the trade cannot be
        placed within risk/market constraints or if inputs are invalid.
    """
    try:
        # Validate numeric inputs are positive and finite
        if any(
            not isinstance(x, (int, float)) or math.isnan(x) or math.isinf(x) or x <= 0
            for x in [account_balance, risk_pct, entry_price, stop_loss]
        ):
            return 0.0

        # Validate market limits structure
        if not isinstance(market_limits, dict):
            return 0.0
        if "min_qty" not in market_limits or "qty_precision" not in market_limits:
            return 0.0

        min_qty = float(market_limits["min_qty"])
        qty_precision = float(market_limits["qty_precision"])

        if min_qty < 0 or qty_precision < 0:
            return 0.0

        # Prevent division-by-zero for equal entry and stop-loss
        risk_distance = abs(entry_price - stop_loss)
        if risk_distance == 0:
            return 0.0

        # Risk-based position size
        risk_amount = account_balance * risk_pct
        position_qty = risk_amount / risk_distance

        # Cap position at maximum affordable quantity (no leverage assumed)
        max_affordable_qty = account_balance / entry_price
        if position_qty > max_affordable_qty:
            position_qty = max_affordable_qty

        # Apply exchange quantity precision (always floor, never round up)
        position_qty = _floor_to_precision(position_qty, qty_precision)

        # Enforce minimum order size
        if position_qty < min_qty:
            return 0.0

        return float(position_qty)

    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return 0.0


class TradingAgent:
    def __init__(self, agent_id, sma_short_period=5, sma_long_period=20, 
                 rsi_period=14, rsi_oversold=30, rsi_overbought=70,
                 macd_fast=12, macd_slow=26, macd_signal=9,
                 volume_sma_period=20, volume_threshold=1.5,
                 stop_loss_pct=0.05, risk_reward_ratio=2.0, use_trailing_stop=False,
                 risk_pct=0.02, max_drawdown_pct=0.15,
                 use_trend_filter=False, trend_sma_period=50,
                 commission_pct=0.001, slippage_pct=0.001,
                 signal_threshold=2, require_volume=False):
        """
        Initializes an individual trading agent with multiple technical indicator parameters
        and risk management parameters.
        These parameters will be tweaked by the genetic optimizer.
        """
        self.agent_id = agent_id
        # SMA parameters
        self.sma_short_period = sma_short_period
        self.sma_long_period = sma_long_period
        # RSI parameters
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        # MACD parameters
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        # Volume parameters
        self.volume_sma_period = volume_sma_period
        self.volume_threshold = volume_threshold
        # Risk management parameters
        self.stop_loss_pct = stop_loss_pct  # Stop loss as percentage (e.g., 0.05 = 5%)
        self.risk_reward_ratio = risk_reward_ratio  # Take-profit distance as multiple of stop-loss distance
        self.use_trailing_stop = use_trailing_stop  # Whether to use trailing stop loss
        self.risk_pct = risk_pct  # Percentage of balance to risk per trade (e.g., 0.02 = 2%)
        self.max_drawdown_pct = max_drawdown_pct  # Max portfolio drawdown before halting (e.g., 0.15 = 15%)
        # Trend filter: only enter long positions when price is above a long-term SMA
        self.use_trend_filter = use_trend_filter
        self.trend_sma_period = trend_sma_period
        # Transaction costs and slippage to make simulation more realistic
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        # Signal filter parameters for stronger entry/exit control
        self.signal_threshold = max(1, int(signal_threshold))
        self.require_volume = require_volume
        
        self.portfolio_value = 1000.0  # Starting fake cash ($1000 USD)
        self.crypto_held = 0.0  # Starting crypto balance
        self.trade_history = []

    def evaluate_market(self, df):
        """
        Calculates multiple technical indicators based on the agent's parameters and
        returns a decision list for the entire dataset using a multi-signal strategy.
        The strength of the combined signal is controlled by signal_threshold, and
        volume confirmation can be required or used as a bonus.
        """
        required_len = max(self.sma_short_period, self.sma_long_period, self.rsi_period)
        if self.use_trend_filter:
            required_len = max(required_len, self.trend_sma_period)
        if df.empty or len(df) < required_len:
            return ["HOLD"] * len(df)

        df = df.copy()

        # Calculate long-term trend SMA for trend filter
        if self.use_trend_filter:
            df['trend_sma'] = df['close'].rolling(window=self.trend_sma_period).mean().bfill()

        # Calculate SMA
        df['sma_short'] = df['close'].rolling(window=self.sma_short_period).mean().bfill()
        df['sma_long'] = df['close'].rolling(window=self.sma_long_period).mean().bfill()

        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].bfill()

        # Calculate MACD
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd'] = df['macd'].bfill()
        df['macd_signal'] = df['macd_signal'].bfill()

        # Calculate Volume SMA
        df['volume_sma'] = df['volume'].rolling(window=self.volume_sma_period).mean().bfill()

        decisions = []
        for i in range(len(df)):
            core_buy = 0
            core_sell = 0

            # SMA crossover signal
            if (df['sma_short'].iloc[i] > df['sma_long'].iloc[i] and
                    df['sma_short'].iloc[i - 1] <= df['sma_long'].iloc[i - 1]):
                core_buy += 1
            elif (df['sma_short'].iloc[i] < df['sma_long'].iloc[i] and
                  df['sma_short'].iloc[i - 1] >= df['sma_long'].iloc[i - 1]):
                core_sell += 1

            # RSI signal
            if df['rsi'].iloc[i] < self.rsi_oversold:
                core_buy += 1
            elif df['rsi'].iloc[i] > self.rsi_overbought:
                core_sell += 1

            # MACD crossover signal
            if (df['macd'].iloc[i] > df['macd_signal'].iloc[i] and
                    df['macd'].iloc[i - 1] <= df['macd_signal'].iloc[i - 1]):
                core_buy += 1
            elif (df['macd'].iloc[i] < df['macd_signal'].iloc[i] and
                  df['macd'].iloc[i - 1] >= df['macd_signal'].iloc[i - 1]):
                core_sell += 1

            # Volume confirmation
            volume_confirmed = df['volume'].iloc[i] > df['volume_sma'].iloc[i] * self.volume_threshold

            if self.require_volume:
                # Volume is a hard gate: no signal unless volume is above average
                if volume_confirmed:
                    buy_signals = core_buy + 1
                    sell_signals = core_sell + 1
                else:
                    buy_signals = 0
                    sell_signals = 0
            else:
                # Volume adds a bonus signal when it confirms an existing directional bias
                buy_signals = core_buy
                sell_signals = core_sell
                if volume_confirmed:
                    if core_buy > 0:
                        buy_signals += 1
                    if core_sell > 0:
                        sell_signals += 1

            # Trend filter: avoid new long entries in a downtrend and exit if trend flips
            if self.use_trend_filter:
                in_uptrend = df['close'].iloc[i] > df['trend_sma'].iloc[i]
                if not in_uptrend:
                    decisions.append("SELL")
                    continue

            # Decision based on configurable signal strength
            if buy_signals >= self.signal_threshold:
                decisions.append("BUY")
            elif sell_signals >= self.signal_threshold:
                decisions.append("SELL")
            else:
                decisions.append("HOLD")

        return decisions

    def simulate_trading(self, df, decisions, market_limits=None):
        """
        Simulates trading based on decisions with stop-loss, take-profit, and
        position sizing mechanisms. Returns final portfolio value, number of trades,
        and observed max drawdown for risk-adjusted fitness scoring.
        """
        if market_limits is None:
            market_limits = {'min_qty': 0.00001, 'qty_precision': 8}

        cash = self.portfolio_value
        crypto = self.crypto_held
        entry_price = None  # Track entry price for stop-loss calculations
        highest_price_since_entry = None  # For trailing stop-loss
        peak_value = self.portfolio_value  # Track peak portfolio value for drawdown
        max_drawdown = 0.0  # Observed peak-to-trough drawdown
        trade_count = 0  # Number of executed orders
        halted = False  # Circuit breaker flag

        for i in range(len(df)):
            if halted:
                continue

            price = df['close'].iloc[i]
            current_value = cash + (crypto * price)

            # Update peak portfolio value and observed max drawdown
            if current_value > peak_value:
                peak_value = current_value
            if peak_value > 0:
                drawdown = (peak_value - current_value) / peak_value
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

            # Max drawdown circuit breaker: liquidate if drawdown exceeds limit
            if peak_value > 0 and (peak_value - current_value) / peak_value > self.max_drawdown_pct:
                if crypto > 0:
                    executed_price = price * (1 - self.slippage_pct)
                    gross_proceeds = crypto * executed_price
                    commission = gross_proceeds * self.commission_pct
                    cash += gross_proceeds - commission
                    trade_count += 1
                    self.trade_history.append((df.index[i], "MAX_DRAWDOWN", executed_price, commission))
                    crypto = 0
                    entry_price = None
                    highest_price_since_entry = None
                halted = True
                continue

            action = decisions[i]

            # Check stop-loss/take-profit if holding crypto
            if crypto > 0 and entry_price is not None:
                # Calculate current P&L percentage
                pnl_pct = (price - entry_price) / entry_price
                
                # Update highest price for trailing stop
                if self.use_trailing_stop:
                    highest_price_since_entry = max(highest_price_since_entry or entry_price, price)
                    trailing_stop_price = highest_price_since_entry * (1 - self.stop_loss_pct)
                    
                    # Check trailing stop
                    if price <= trailing_stop_price:
                        executed_price = price * (1 - self.slippage_pct)
                        gross_proceeds = crypto * executed_price
                        commission = gross_proceeds * self.commission_pct
                        cash += gross_proceeds - commission
                        trade_count += 1
                        self.trade_history.append((df.index[i], "TRAILING_STOP", executed_price, commission))
                        crypto = 0
                        entry_price = None
                        highest_price_since_entry = None
                        continue
                
                # Check fixed stop-loss
                if pnl_pct <= -self.stop_loss_pct:
                    executed_price = price * (1 - self.slippage_pct)
                    gross_proceeds = crypto * executed_price
                    commission = gross_proceeds * self.commission_pct
                    cash += gross_proceeds - commission
                    trade_count += 1
                    self.trade_history.append((df.index[i], "STOP_LOSS", executed_price, commission))
                    crypto = 0
                    entry_price = None
                    highest_price_since_entry = None
                    continue
                
                # Check take-profit (derived from stop-loss distance and risk/reward ratio)
                take_profit_pct = self.stop_loss_pct * self.risk_reward_ratio
                if pnl_pct >= take_profit_pct:
                    executed_price = price * (1 - self.slippage_pct)
                    gross_proceeds = crypto * executed_price
                    commission = gross_proceeds * self.commission_pct
                    cash += gross_proceeds - commission
                    trade_count += 1
                    self.trade_history.append((df.index[i], "TAKE_PROFIT", executed_price, commission))
                    crypto = 0
                    entry_price = None
                    highest_price_since_entry = None
                    continue

            # Execute trading signals with risk-based position sizing
            if action == "BUY" and cash > 0:
                # Determine stop-loss price for long position
                stop_loss_price = price * (1 - self.stop_loss_pct)
                
                # Calculate position size based on risk rules
                qty = calculate_position_size(
                    account_balance=cash,
                    risk_pct=self.risk_pct,
                    entry_price=price,
                    stop_loss=stop_loss_price,
                    market_limits=market_limits
                )
                
                if qty > 0:
                    executed_price = price * (1 + self.slippage_pct)
                    gross_cost = qty * executed_price
                    commission = gross_cost * self.commission_pct
                    cost = gross_cost + commission
                    if cost <= cash:
                        crypto += qty
                        cash -= cost
                        entry_price = executed_price
                        highest_price_since_entry = executed_price
                        trade_count += 1
                        self.trade_history.append((df.index[i], "BUY", executed_price, qty, commission))
            elif action == "SELL" and crypto > 0:
                # Sell all crypto holdings, applying slippage and commission
                executed_price = price * (1 - self.slippage_pct)
                gross_proceeds = crypto * executed_price
                commission = gross_proceeds * self.commission_pct
                cash += gross_proceeds - commission
                crypto = 0
                entry_price = None
                highest_price_since_entry = None
                trade_count += 1
                self.trade_history.append((df.index[i], "SELL", executed_price, commission))

        # Calculate final total value in USD at the last available price
        final_price = df['close'].iloc[-1]
        final_value = cash + (crypto * final_price)
        return final_value, trade_count, max_drawdown


# --- TEST UTILITY ---
if __name__ == "__main__":
    # Fake minimal data to test the logic locally before connecting files
    dates = pd.date_range(start="2026-01-01", periods=5)
    test_df = pd.DataFrame({'close': [100, 105, 102, 110, 108]}, index=dates)

    # Create a test agent
    agent = TradingAgent(agent_id="Agent_Beta_1", sma_short_period=2, sma_long_period=3)
    decisions = agent.evaluate_market(test_df)

    print(f"Decisions made: {decisions}")
    final_balance = agent.simulate_trading(test_df, decisions)
    print(f"Starting Balance: $1000 -> Ending Balance: ${final_balance:.2f}")

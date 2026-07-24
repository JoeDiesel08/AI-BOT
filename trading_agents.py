import pandas as pd
import numpy as np
import math
from decimal import Decimal, ROUND_FLOOR

# Internal decision codes used by TradingAgent for fast vectorized logic
HOLD = 0
BUY = 1
SELL = 2


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
                 signal_threshold=2, require_volume=False,
                 use_adx_filter=False, adx_period=14, adx_threshold=25.0):
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
        # ADX filter: only trade when trend strength is above a threshold
        self.use_adx_filter = use_adx_filter
        self.adx_period = max(1, int(adx_period))
        self.adx_threshold = float(adx_threshold)
        
        self.portfolio_value = 1000.0  # Starting fake cash ($1000 USD)
        self.crypto_held = 0.0  # Starting crypto balance
        self.trade_history = []

    def evaluate_market(self, df):
        """
        Calculates multiple technical indicators based on the agent's parameters and
        returns a decision array for the entire dataset using a multi-signal strategy.
        Uses vectorized pandas operations instead of row-by-row Python loops for speed.
        The strength of the combined signal is controlled by signal_threshold, and
        volume confirmation can be required or used as a bonus.
        Returns a list of integer decision codes (HOLD=0, BUY=1, SELL=2).
        """
        n = len(df)
        required_len = max(self.sma_short_period, self.sma_long_period, self.rsi_period)
        if self.use_trend_filter:
            required_len = max(required_len, self.trend_sma_period)
        if self.use_adx_filter:
            required_len = max(required_len, self.adx_period * 2)
        if df.empty or n < required_len:
            return [HOLD] * n

        close = df['close']
        volume = df['volume']

        # SMA and crossover signals (vectorized)
        sma_short = close.rolling(window=self.sma_short_period).mean().bfill()
        sma_long = close.rolling(window=self.sma_long_period).mean().bfill()
        sma_buy = ((sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))).fillna(False).astype(int)
        sma_sell = ((sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))).fillna(False).astype(int)

        # RSI (vectorized)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).bfill()
        rsi_buy = (rsi < self.rsi_oversold).astype(int)
        rsi_sell = (rsi > self.rsi_overbought).astype(int)

        # MACD and crossover signals (vectorized)
        macd_line = close.ewm(span=self.macd_fast, adjust=False).mean() - close.ewm(span=self.macd_slow, adjust=False).mean()
        macd_signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        macd_line = macd_line.bfill()
        macd_signal_line = macd_signal_line.bfill()
        macd_buy = ((macd_line > macd_signal_line) & (macd_line.shift(1) <= macd_signal_line.shift(1))).fillna(False).astype(int)
        macd_sell = ((macd_line < macd_signal_line) & (macd_line.shift(1) >= macd_signal_line.shift(1))).fillna(False).astype(int)

        # Volume SMA
        volume_sma = volume.rolling(window=self.volume_sma_period).mean().bfill()
        volume_confirmed = volume > volume_sma * self.volume_threshold

        # Core signal counts
        core_buy = sma_buy + rsi_buy + macd_buy
        core_sell = sma_sell + rsi_sell + macd_sell

        if self.require_volume:
            # Volume is a hard gate: no signal unless volume is above average
            buy_signals = (core_buy + 1).where(volume_confirmed, 0).to_numpy()
            sell_signals = (core_sell + 1).where(volume_confirmed, 0).to_numpy()
        else:
            # Volume adds a bonus signal when it confirms an existing directional bias
            bonus = volume_confirmed.astype(int)
            buy_signals = (core_buy + bonus * (core_buy > 0).astype(int)).to_numpy()
            sell_signals = (core_sell + bonus * (core_sell > 0).astype(int)).to_numpy()

        # Start with HOLD; trend filter can force SELL
        decisions = np.full(n, HOLD, dtype=np.int8)

        if self.use_trend_filter:
            trend_sma = close.rolling(window=self.trend_sma_period).mean().bfill()
            not_uptrend = (close <= trend_sma).to_numpy()
            decisions[not_uptrend] = SELL

        # ADX trend-strength regime filter
        strong = np.ones(n, dtype=bool)
        if self.use_adx_filter:
            high = df['high']
            low = df['low']
            prev_high = high.shift(1)
            prev_low = low.shift(1)
            prev_close = close.shift(1)

            plus_dm = (high - prev_high).clip(lower=0)
            minus_dm = (prev_low - low).clip(lower=0)
            plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
            minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1.0 / self.adx_period, adjust=False).mean()

            plus_di = 100 * plus_dm.ewm(alpha=1.0 / self.adx_period, adjust=False).mean() / atr
            minus_di = 100 * minus_dm.ewm(alpha=1.0 / self.adx_period, adjust=False).mean() / atr
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di) * 100).fillna(0)
            adx = dx.ewm(alpha=1.0 / self.adx_period, adjust=False).mean().bfill()
            strong = (adx > self.adx_threshold).to_numpy()

        # Apply signal threshold with BUY priority over SELL
        mask = decisions != SELL
        buy_cond = mask & strong & (buy_signals >= self.signal_threshold)
        sell_cond = mask & (~buy_cond) & strong & (sell_signals >= self.signal_threshold)
        decisions[buy_cond] = BUY
        decisions[sell_cond] = SELL

        return decisions.tolist()

    def simulate_trading(self, df, decisions, market_limits=None):
        """
        Simulates trading based on decisions with stop-loss, take-profit, and
        position sizing mechanisms. Returns final portfolio value, number of trades,
        and observed max drawdown for risk-adjusted fitness scoring.
        Decisions can be integer codes (HOLD=0, BUY=1, SELL=2) or legacy strings.
        """
        if market_limits is None:
            market_limits = {'min_qty': 0.00001, 'qty_precision': 8}

        # Normalize decisions to integer codes for fast comparison
        if isinstance(decisions, (list, tuple)) and len(decisions) > 0 and isinstance(decisions[0], str):
            str_to_code = {"HOLD": HOLD, "BUY": BUY, "SELL": SELL}
            decisions = np.array([str_to_code.get(d, HOLD) for d in decisions], dtype=np.int8)
        else:
            decisions = np.asarray(decisions, dtype=np.int8)

        prices = df['close'].to_numpy()
        index_values = df.index.to_numpy()
        n = len(prices)

        cash = self.portfolio_value
        crypto = self.crypto_held
        entry_price = None  # Track entry price for stop-loss calculations
        highest_price_since_entry = None  # For trailing stop-loss
        peak_value = self.portfolio_value  # Track peak portfolio value for drawdown
        max_drawdown = 0.0  # Observed peak-to-trough drawdown
        trade_count = 0  # Number of executed orders
        halted = False  # Circuit breaker flag

        for i in range(n):
            if halted:
                continue

            price = prices[i]
            current_value = cash + (crypto * price)

            # Update peak portfolio value and observed max drawdown
            if current_value > peak_value:
                peak_value = current_value
            if peak_value > 0:
                drawdown = (peak_value - current_value) / peak_value
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

            # Max drawdown circuit breaker: liquidate if drawdown exceeds limit
            if peak_value > 0 and drawdown > self.max_drawdown_pct:
                if crypto > 0:
                    executed_price = price * (1 - self.slippage_pct)
                    gross_proceeds = crypto * executed_price
                    commission = gross_proceeds * self.commission_pct
                    cash += gross_proceeds - commission
                    trade_count += 1
                    self.trade_history.append((index_values[i], "MAX_DRAWDOWN", executed_price, commission))
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
                    if highest_price_since_entry is None or price > highest_price_since_entry:
                        highest_price_since_entry = price
                    trailing_stop_price = highest_price_since_entry * (1 - self.stop_loss_pct)

                    # Check trailing stop
                    if price <= trailing_stop_price:
                        executed_price = price * (1 - self.slippage_pct)
                        gross_proceeds = crypto * executed_price
                        commission = gross_proceeds * self.commission_pct
                        cash += gross_proceeds - commission
                        trade_count += 1
                        self.trade_history.append((index_values[i], "TRAILING_STOP", executed_price, commission))
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
                    self.trade_history.append((index_values[i], "STOP_LOSS", executed_price, commission))
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
                    self.trade_history.append((index_values[i], "TAKE_PROFIT", executed_price, commission))
                    crypto = 0
                    entry_price = None
                    highest_price_since_entry = None
                    continue

            # Execute trading signals with risk-based position sizing
            if action == BUY and cash > 0:
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
                        self.trade_history.append((index_values[i], "BUY", executed_price, qty, commission))
            elif action == SELL and crypto > 0:
                # Sell all crypto holdings, applying slippage and commission
                executed_price = price * (1 - self.slippage_pct)
                gross_proceeds = crypto * executed_price
                commission = gross_proceeds * self.commission_pct
                cash += gross_proceeds - commission
                crypto = 0
                entry_price = None
                highest_price_since_entry = None
                trade_count += 1
                self.trade_history.append((index_values[i], "SELL", executed_price, commission))

        # Calculate final total value in USD at the last available price
        final_price = prices[-1]
        final_value = cash + (crypto * final_price)
        return final_value, trade_count, max_drawdown


# --- TEST UTILITY ---
if __name__ == "__main__":
    # Fake minimal data to test the logic locally before connecting files
    dates = pd.date_range(start="2026-01-01", periods=50)
    np.random.seed(1)
    prices = 100 + np.cumsum(np.random.normal(0, 1, 50))
    test_df = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.005,
        'low': prices * 0.995,
        'close': prices,
        'volume': np.random.uniform(100, 1000, 50),
    }, index=dates)

    # Create a test agent
    agent = TradingAgent(agent_id="Agent_Beta_1", sma_short_period=5, sma_long_period=10)
    decisions = agent.evaluate_market(test_df)

    print(f"Decisions made: {decisions}")
    final_balance, trade_count, max_dd = agent.simulate_trading(test_df, decisions)
    print(f"Starting Balance: $1000 -> Ending Balance: ${final_balance:.2f} | Trades: {trade_count} | MaxDD: {max_dd:.2%}")

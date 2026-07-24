import random
import math
import numpy as np
import copy
from trading_agents import TradingAgent

class GenerationOptimizer:
    def __init__(self, population_size=10, mutation_rate=0.2, market_limits=None, risk_penalty=0.5, seed_agent=None):
        """
        Manages a population of trading agents and evolves them over time.
        
        Args:
            population_size: Number of agents in the population.
            mutation_rate: Probability of mutating each parameter.
            market_limits: Optional dict with 'min_qty' and 'qty_precision' for
                           position sizing. Defaults to BTC/USDT-like limits.
            risk_penalty: Weight applied to observed max drawdown when computing
                          risk-adjusted fitness. Higher = penalize drawdown more.
            seed_agent: Optional dict or TradingAgent used as the starting point
                        for the first generation. If provided, the first agent is the
                        seed and the rest are slightly mutated copies.
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.market_limits = market_limits if market_limits is not None else {'min_qty': 0.00001, 'qty_precision': 8}
        self.risk_penalty = risk_penalty
        self.seed_agent = seed_agent
        
    def _mutate_agent(self, parent, agent_id):
        """Return a new TradingAgent that is a mutated copy of the parent."""
        sma_short = parent.sma_short_period
        sma_long = parent.sma_long_period
        rsi_period = parent.rsi_period
        rsi_oversold = parent.rsi_oversold
        rsi_overbought = parent.rsi_overbought
        macd_fast = parent.macd_fast
        macd_slow = parent.macd_slow
        macd_signal = parent.macd_signal
        volume_sma_period = parent.volume_sma_period
        volume_threshold = parent.volume_threshold
        stop_loss_pct = parent.stop_loss_pct
        risk_reward_ratio = parent.risk_reward_ratio
        use_trailing_stop = parent.use_trailing_stop
        risk_pct = parent.risk_pct
        max_drawdown_pct = parent.max_drawdown_pct
        use_trend_filter = parent.use_trend_filter
        trend_sma_period = parent.trend_sma_period
        signal_threshold = parent.signal_threshold
        require_volume = parent.require_volume
        use_adx_filter = parent.use_adx_filter
        adx_period = parent.adx_period
        adx_threshold = parent.adx_threshold

        if random.random() < self.mutation_rate:
            sma_short += random.choice([-2, -1, 1, 2])
            sma_short = max(2, sma_short)

        if random.random() < self.mutation_rate:
            sma_long += random.choice([-2, -1, 1, 2])
            sma_long = max(sma_short + 2, sma_long)

        if random.random() < self.mutation_rate:
            rsi_period += random.choice([-2, -1, 1, 2])
            rsi_period = max(5, rsi_period)

        if random.random() < self.mutation_rate:
            rsi_oversold += random.choice([-5, -2, 2, 5])
            rsi_oversold = max(10, min(40, rsi_oversold))

        if random.random() < self.mutation_rate:
            rsi_overbought += random.choice([-5, -2, 2, 5])
            rsi_overbought = max(50, min(90, rsi_overbought))

        if random.random() < self.mutation_rate:
            macd_fast += random.choice([-2, -1, 1, 2])
            macd_fast = max(5, macd_fast)

        if random.random() < self.mutation_rate:
            macd_slow += random.choice([-2, -1, 1, 2])
            macd_slow = max(macd_fast + 3, macd_slow)

        if random.random() < self.mutation_rate:
            macd_signal += random.choice([-2, -1, 1, 2])
            macd_signal = max(3, macd_signal)

        if random.random() < self.mutation_rate:
            volume_sma_period += random.choice([-5, -2, 2, 5])
            volume_sma_period = max(5, volume_sma_period)

        if random.random() < self.mutation_rate:
            volume_threshold += random.uniform(-0.3, 0.3)
            volume_threshold = max(0.5, min(3.0, volume_threshold))

        if random.random() < self.mutation_rate:
            stop_loss_pct += random.uniform(-0.01, 0.01)
            stop_loss_pct = max(0.005, min(0.20, stop_loss_pct))

        if random.random() < self.mutation_rate:
            risk_reward_ratio += random.uniform(-0.5, 0.5)
            risk_reward_ratio = max(0.25, min(10.0, risk_reward_ratio))

        if random.random() < self.mutation_rate:
            use_trailing_stop = not use_trailing_stop

        if random.random() < self.mutation_rate:
            risk_pct += random.uniform(-0.01, 0.01)
            risk_pct = max(0.005, min(0.50, risk_pct))

        if random.random() < self.mutation_rate:
            max_drawdown_pct += random.uniform(-0.02, 0.02)
            max_drawdown_pct = max(0.01, min(0.50, max_drawdown_pct))

        if random.random() < self.mutation_rate:
            use_trend_filter = not use_trend_filter

        if random.random() < self.mutation_rate:
            trend_sma_period += random.choice([-10, -5, 5, 10])
            trend_sma_period = max(10, min(150, trend_sma_period))

        if random.random() < self.mutation_rate:
            signal_threshold += random.choice([-1, 1])
            signal_threshold = max(2, min(4, signal_threshold))

        if random.random() < self.mutation_rate:
            require_volume = not require_volume

        if random.random() < self.mutation_rate:
            use_adx_filter = not use_adx_filter

        if random.random() < self.mutation_rate:
            adx_period += random.choice([-3, -1, 1, 3])
            adx_period = max(5, min(50, adx_period))

        if random.random() < self.mutation_rate:
            adx_threshold += random.uniform(-5.0, 5.0)
            adx_threshold = max(5.0, min(60.0, adx_threshold))

        return TradingAgent(
            agent_id=agent_id,
            sma_short_period=sma_short,
            sma_long_period=sma_long,
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            volume_sma_period=volume_sma_period,
            volume_threshold=volume_threshold,
            stop_loss_pct=stop_loss_pct,
            risk_reward_ratio=risk_reward_ratio,
            use_trailing_stop=use_trailing_stop,
            risk_pct=risk_pct,
            max_drawdown_pct=max_drawdown_pct,
            use_trend_filter=use_trend_filter,
            trend_sma_period=trend_sma_period,
            signal_threshold=signal_threshold,
            require_volume=require_volume,
            use_adx_filter=use_adx_filter,
            adx_period=adx_period,
            adx_threshold=adx_threshold
        )

    def initialize_population(self):
        """
        Creates the first generation of agents. If a seed agent was supplied, the first
        agent is the seed and the rest are slightly mutated copies; otherwise all agents
        are created with completely random strategy values.
        """
        self.population = []
        if self.seed_agent is not None:
            seed = self.seed_agent if isinstance(self.seed_agent, TradingAgent) else TradingAgent.from_dict(self.seed_agent)
            self.population.append(seed)
            for i in range(1, self.population_size):
                self.population.append(self._mutate_agent(seed, f"Gen1_Seed_{i}"))
            print(f"Successfully initialized {self.population_size} agents seeded from previous best ({seed.agent_id}).")
            return

        for i in range(self.population_size):
            # SMA parameters
            sma_short = random.randint(2, 15)
            sma_long = random.randint(sma_short + 2, 40)
            
            # RSI parameters
            rsi_period = random.randint(10, 20)
            rsi_oversold = random.randint(20, 35)
            rsi_overbought = random.randint(65, 80)
            
            # MACD parameters
            macd_fast = random.randint(8, 15)
            macd_slow = random.randint(macd_fast + 5, 30)
            macd_signal = random.randint(5, 12)
            
            # Volume parameters
            volume_sma_period = random.randint(10, 30)
            volume_threshold = random.uniform(1.0, 2.5)
            
            # Risk management parameters
            stop_loss_pct = random.uniform(0.01, 0.10)  # 1% to 10%
            risk_reward_ratio = random.uniform(0.5, 5.0)  # Reward distance as multiple of risk
            use_trailing_stop = random.choice([True, False])
            risk_pct = random.uniform(0.01, 0.10)  # 1% to 10% of balance risked
            max_drawdown_pct = random.uniform(0.05, 0.25)  # 5% to 25% max drawdown

            # Trend filter parameters
            use_trend_filter = random.choice([True, False])
            trend_sma_period = random.randint(20, 100)

            # Signal filter parameters
            signal_threshold = random.randint(2, 4)
            require_volume = random.choice([True, False])

            # ADX trend-strength filter parameters
            use_adx_filter = random.choice([True, False])
            adx_period = random.randint(10, 30)
            adx_threshold = random.uniform(15.0, 40.0)
            
            agent = TradingAgent(
                agent_id=f"Gen1_Agent_{i}",
                sma_short_period=sma_short,
                sma_long_period=sma_long,
                rsi_period=rsi_period,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought,
                macd_fast=macd_fast,
                macd_slow=macd_slow,
                macd_signal=macd_signal,
                volume_sma_period=volume_sma_period,
                volume_threshold=volume_threshold,
                stop_loss_pct=stop_loss_pct,
                risk_reward_ratio=risk_reward_ratio,
                use_trailing_stop=use_trailing_stop,
                risk_pct=risk_pct,
                max_drawdown_pct=max_drawdown_pct,
                use_trend_filter=use_trend_filter,
                trend_sma_period=trend_sma_period,
                signal_threshold=signal_threshold,
                require_volume=require_volume,
                use_adx_filter=use_adx_filter,
                adx_period=adx_period,
                adx_threshold=adx_threshold
            )
            self.population.append(agent)
        print(f"Successfully initialized {self.population_size} random agents with multi-indicator strategies.")

    def _fitness(self, final_value, max_drawdown, equity_curve, trade_pnls, closed_trades):
        """
        Robust fitness score that rewards consistent, repeatable strategy behavior
        over one-off lucky runs. Combines drawdown, Sharpe-like consistency,
        profit factor, trade diversification, and linearity of the equity curve.
        """
        STARTING_VALUE = 1000.0
        net_profit = final_value - STARTING_VALUE
        num_bars = len(equity_curve) if equity_curve else 0

        # --- Trade count quality (avoid one-trade lucky outcomes / overtrading)
        min_required_trades = max(4, num_bars // 100)
        max_trades = max(min_required_trades, num_bars / 15.0)
        if closed_trades < min_required_trades:
            trade_quality = closed_trades / min_required_trades
        elif closed_trades <= max_trades:
            trade_quality = 1.0
        else:
            trade_quality = max_trades / closed_trades

        # --- Sharpe-like quality from equity-curve returns (logistic map)
        sharpe_quality = 0.5  # neutral default
        if num_bars > 1:
            returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
            returns = returns[np.isfinite(returns)]
            if len(returns) > 1 and np.std(returns) > 1e-12:
                sharpe = np.mean(returns) / np.std(returns)
                # Logistic map: negative Sharpe < 0.5, positive > 0.5, saturates at 1
                sharpe_quality = 1.0 / (1.0 + np.exp(-sharpe))
            elif len(returns) > 0:
                # No variance: reward positive drift
                sharpe_quality = 1.0 / (1.0 + np.exp(-np.mean(returns) * 1000.0))

        # --- Drawdown quality (exponential penalty)
        drawdown_quality = math.exp(-max_drawdown * 4.0)

        # --- Profit factor quality
        gross_profit = sum(p for p in trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in trade_pnls if p < 0))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = 10.0 if gross_profit > 0 else 1.0
        profit_factor_quality = float(np.clip(profit_factor - 1.0, 0.0, 1.0))

        # --- Consistency quality: linearity of the equity curve (R^2)
        consistency_quality = 0.0
        if num_bars > 2:
            x = np.arange(num_bars)
            y = np.array(equity_curve, dtype=float)
            if np.std(y) > 1e-9:
                slope, intercept = np.polyfit(x, y, 1)
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                if ss_tot > 0:
                    r2 = 1.0 - ss_res / ss_tot
                    consistency_quality = float(np.clip(r2, 0.0, 1.0))

        # --- Concentration quality: penalize if a single trade drives most profit
        concentration_quality = 1.0
        if net_profit > 1e-9 and trade_pnls:
            max_trade_pnl = max(trade_pnls)
            concentration = float(np.clip(max_trade_pnl / net_profit, 0.0, 1.0))
            if concentration > 0.5:
                concentration_quality = max(0.0, 1.0 - (concentration - 0.5) * 2.0)

        # --- Combine quality factors (product = all must be decent)
        quality = (
            drawdown_quality *
            consistency_quality *
            trade_quality *
            profit_factor_quality *
            sharpe_quality *
            concentration_quality
        )

        if net_profit > 0:
            # Reward net profit only when quality is high enough
            fitness = net_profit * quality
        else:
            # For losing strategies, use legacy risk-adjusted score
            fitness = net_profit - (max_drawdown * 1000.0 * max(self.risk_penalty, 0.01))

        return float(fitness)

    def score_population(self, market_df):
        """
        Runs every agent against the real market data and ranks them by a
        robust risk-adjusted fitness score. Returns a list of tuples:
        (agent, fitness_score, final_portfolio_value, max_drawdown, closed_trade_count)
        """
        ranked_agents = []
        
        for agent in self.population:
            # Get decisions from the agent based on market data
            decisions = agent.evaluate_market(market_df)
            final_value, order_count, max_drawdown, equity_curve, trade_pnls, closed_trades = agent.simulate_trading(
                market_df, decisions, market_limits=self.market_limits
            )
            # Robust fitness: reward strategy, not one-off lucky runs
            fitness = self._fitness(
                final_value, max_drawdown, equity_curve, trade_pnls, closed_trades
            )
            ranked_agents.append((agent, fitness, final_value, max_drawdown, closed_trades))
            
        # Sort agents by highest risk-adjusted fitness first
        ranked_agents.sort(key=lambda x: x[1], reverse=True)
        return ranked_agents

    def evolve_population(self, ranked_agents, generation_number):
        """
        Takes the best agents, breeds/mutates them, and updates the population for the next round.
        Now handles mutation across all technical indicator parameters.
        """
        new_population = []
        
        # Elitist Strategy: Keep the top 2 best-performing agents exactly as they are
        new_population.append(copy.deepcopy(ranked_agents[0][0]))
        new_population.append(copy.deepcopy(ranked_agents[1][0]))
        
        # Add some random diversity (2 new random agents each generation)
        for _ in range(2):
            sma_short = random.randint(2, 15)
            sma_long = random.randint(sma_short + 2, 40)
            rsi_period = random.randint(10, 20)
            rsi_oversold = random.randint(20, 35)
            rsi_overbought = random.randint(65, 80)
            macd_fast = random.randint(8, 15)
            macd_slow = random.randint(macd_fast + 5, 30)
            macd_signal = random.randint(5, 12)
            volume_sma_period = random.randint(10, 30)
            volume_threshold = random.uniform(1.0, 2.5)
            stop_loss_pct = random.uniform(0.01, 0.10)
            risk_reward_ratio = random.uniform(0.5, 5.0)
            use_trailing_stop = random.choice([True, False])
            risk_pct = random.uniform(0.01, 0.10)
            max_drawdown_pct = random.uniform(0.05, 0.25)

            use_trend_filter = random.choice([True, False])
            trend_sma_period = random.randint(20, 100)

            signal_threshold = random.randint(2, 4)
            require_volume = random.choice([True, False])

            use_adx_filter = random.choice([True, False])
            adx_period = random.randint(10, 30)
            adx_threshold = random.uniform(15.0, 40.0)
            
            agent = TradingAgent(
                agent_id=f"Gen{generation_number}_Random_{len(new_population)}",
                sma_short_period=sma_short,
                sma_long_period=sma_long,
                rsi_period=rsi_period,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought,
                macd_fast=macd_fast,
                macd_slow=macd_slow,
                macd_signal=macd_signal,
                volume_sma_period=volume_sma_period,
                volume_threshold=volume_threshold,
                stop_loss_pct=stop_loss_pct,
                risk_reward_ratio=risk_reward_ratio,
                use_trailing_stop=use_trailing_stop,
                risk_pct=risk_pct,
                max_drawdown_pct=max_drawdown_pct,
                use_trend_filter=use_trend_filter,
                trend_sma_period=trend_sma_period,
                signal_threshold=signal_threshold,
                require_volume=require_volume,
                use_adx_filter=use_adx_filter,
                adx_period=adx_period,
                adx_threshold=adx_threshold
            )
            new_population.append(agent)
        
        # Fill up the rest of the population with mutated versions of the top agents
        while len(new_population) < self.population_size:
            # Randomly pick one of the successful top 3 parent agents
            parent = random.choice(ranked_agents[:3])[0]
            child_id = f"Gen{generation_number}_Agent_{len(new_population)}"
            new_population.append(self._mutate_agent(parent, child_id))
            
        self.population = new_population

# --- TEST UTILITY ---
if __name__ == "__main__":
    print("Testing Generation Optimizer structure...")
    optimizer = GenerationOptimizer(population_size=5)
    optimizer.initialize_population()
    print("Optimizer setup verification passed!")

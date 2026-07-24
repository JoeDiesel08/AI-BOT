import random
import copy
from trading_agents import TradingAgent

class GenerationOptimizer:
    def __init__(self, population_size=10, mutation_rate=0.2, market_limits=None, risk_penalty=0.5):
        """
        Manages a population of trading agents and evolves them over time.
        
        Args:
            population_size: Number of agents in the population.
            mutation_rate: Probability of mutating each parameter.
            market_limits: Optional dict with 'min_qty' and 'qty_precision' for
                           position sizing. Defaults to BTC/USDT-like limits.
            risk_penalty: Weight applied to observed max drawdown when computing
                          risk-adjusted fitness. Higher = penalize drawdown more.
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.market_limits = market_limits if market_limits is not None else {'min_qty': 0.00001, 'qty_precision': 8}
        self.risk_penalty = risk_penalty
        
    def initialize_population(self):
        """
        Creates the first generation of agents with completely random strategy values
        across all technical indicators and risk management parameters.
        """
        self.population = []
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

    def score_population(self, market_df):
        """
        Runs every agent against the real market data and ranks them by a
        risk-adjusted fitness score. Returns a list of tuples:
        (agent, fitness_score, final_portfolio_value, max_drawdown, trade_count)
        """
        ranked_agents = []
        
        for agent in self.population:
            # Get decisions from the agent based on market data
            decisions = agent.evaluate_market(market_df)
            final_value, trade_count, max_drawdown = agent.simulate_trading(
                market_df, decisions, market_limits=self.market_limits
            )
            # Risk-adjusted fitness: reward profit, penalize severe drawdowns
            fitness = final_value - (max_drawdown * 1000 * self.risk_penalty)
            ranked_agents.append((agent, fitness, final_value, max_drawdown, trade_count))
            
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
            
            # Clone parent parameters
            child_sma_short = parent.sma_short_period
            child_sma_long = parent.sma_long_period
            child_rsi_period = parent.rsi_period
            child_rsi_oversold = parent.rsi_oversold
            child_rsi_overbought = parent.rsi_overbought
            child_macd_fast = parent.macd_fast
            child_macd_slow = parent.macd_slow
            child_macd_signal = parent.macd_signal
            child_volume_sma_period = parent.volume_sma_period
            child_volume_threshold = parent.volume_threshold
            child_stop_loss_pct = parent.stop_loss_pct
            child_risk_reward_ratio = parent.risk_reward_ratio
            child_use_trailing_stop = parent.use_trailing_stop
            child_risk_pct = parent.risk_pct
            child_max_drawdown_pct = parent.max_drawdown_pct
            child_use_trend_filter = parent.use_trend_filter
            child_trend_sma_period = parent.trend_sma_period
            
            # Apply mutations based on mutation rate
            if random.random() < self.mutation_rate:
                child_sma_short += random.choice([-2, -1, 1, 2])
                child_sma_short = max(2, child_sma_short)
                
            if random.random() < self.mutation_rate:
                child_sma_long += random.choice([-2, -1, 1, 2])
                child_sma_long = max(child_sma_short + 2, child_sma_long)
                
            if random.random() < self.mutation_rate:
                child_rsi_period += random.choice([-2, -1, 1, 2])
                child_rsi_period = max(5, child_rsi_period)
                
            if random.random() < self.mutation_rate:
                child_rsi_oversold += random.choice([-5, -2, 2, 5])
                child_rsi_oversold = max(10, min(40, child_rsi_oversold))
                
            if random.random() < self.mutation_rate:
                child_rsi_overbought += random.choice([-5, -2, 2, 5])
                child_rsi_overbought = max(50, min(90, child_rsi_overbought))
                
            if random.random() < self.mutation_rate:
                child_macd_fast += random.choice([-2, -1, 1, 2])
                child_macd_fast = max(5, child_macd_fast)
                
            if random.random() < self.mutation_rate:
                child_macd_slow += random.choice([-2, -1, 1, 2])
                child_macd_slow = max(child_macd_fast + 3, child_macd_slow)
                
            if random.random() < self.mutation_rate:
                child_macd_signal += random.choice([-2, -1, 1, 2])
                child_macd_signal = max(3, child_macd_signal)
                
            if random.random() < self.mutation_rate:
                child_volume_sma_period += random.choice([-5, -2, 2, 5])
                child_volume_sma_period = max(5, child_volume_sma_period)
                
            if random.random() < self.mutation_rate:
                child_volume_threshold += random.uniform(-0.3, 0.3)
                child_volume_threshold = max(0.5, min(3.0, child_volume_threshold))
                
            # Mutate risk management parameters
            if random.random() < self.mutation_rate:
                child_stop_loss_pct += random.uniform(-0.01, 0.01)
                child_stop_loss_pct = max(0.005, min(0.20, child_stop_loss_pct))
                
            if random.random() < self.mutation_rate:
                child_risk_reward_ratio += random.uniform(-0.5, 0.5)
                child_risk_reward_ratio = max(0.25, min(10.0, child_risk_reward_ratio))
                
            if random.random() < self.mutation_rate:
                child_use_trailing_stop = not child_use_trailing_stop
                
            if random.random() < self.mutation_rate:
                child_risk_pct += random.uniform(-0.01, 0.01)
                child_risk_pct = max(0.005, min(0.50, child_risk_pct))
                
            if random.random() < self.mutation_rate:
                child_max_drawdown_pct += random.uniform(-0.02, 0.02)
                child_max_drawdown_pct = max(0.01, min(0.50, child_max_drawdown_pct))

            if random.random() < self.mutation_rate:
                child_use_trend_filter = not child_use_trend_filter

            if random.random() < self.mutation_rate:
                child_trend_sma_period += random.choice([-10, -5, 5, 10])
                child_trend_sma_period = max(10, min(150, child_trend_sma_period))

            # Clone and mutate signal filter parameters
            child_signal_threshold = parent.signal_threshold
            child_require_volume = parent.require_volume

            if random.random() < self.mutation_rate:
                child_signal_threshold += random.choice([-1, 1])
                child_signal_threshold = max(2, min(4, child_signal_threshold))

            if random.random() < self.mutation_rate:
                child_require_volume = not child_require_volume

            # Clone and mutate ADX trend-strength filter parameters
            child_use_adx_filter = parent.use_adx_filter
            child_adx_period = parent.adx_period
            child_adx_threshold = parent.adx_threshold

            if random.random() < self.mutation_rate:
                child_use_adx_filter = not child_use_adx_filter

            if random.random() < self.mutation_rate:
                child_adx_period += random.choice([-3, -1, 1, 3])
                child_adx_period = max(5, min(50, child_adx_period))

            if random.random() < self.mutation_rate:
                child_adx_threshold += random.uniform(-5.0, 5.0)
                child_adx_threshold = max(5.0, min(60.0, child_adx_threshold))
                
            # Instantiate the new mutated child agent
            child_id = f"Gen{generation_number}_Agent_{len(new_population)}"
            child_agent = TradingAgent(
                agent_id=child_id,
                sma_short_period=child_sma_short,
                sma_long_period=child_sma_long,
                rsi_period=child_rsi_period,
                rsi_oversold=child_rsi_oversold,
                rsi_overbought=child_rsi_overbought,
                macd_fast=child_macd_fast,
                macd_slow=child_macd_slow,
                macd_signal=child_macd_signal,
                volume_sma_period=child_volume_sma_period,
                volume_threshold=child_volume_threshold,
                stop_loss_pct=child_stop_loss_pct,
                risk_reward_ratio=child_risk_reward_ratio,
                use_trailing_stop=child_use_trailing_stop,
                risk_pct=child_risk_pct,
                max_drawdown_pct=child_max_drawdown_pct,
                use_trend_filter=child_use_trend_filter,
                trend_sma_period=child_trend_sma_period,
                signal_threshold=child_signal_threshold,
                require_volume=child_require_volume,
                use_adx_filter=child_use_adx_filter,
                adx_period=child_adx_period,
                adx_threshold=child_adx_threshold
            )
            new_population.append(child_agent)
            
        self.population = new_population

# --- TEST UTILITY ---
if __name__ == "__main__":
    print("Testing Generation Optimizer structure...")
    optimizer = GenerationOptimizer(population_size=5)
    optimizer.initialize_population()
    print("Optimizer setup verification passed!")

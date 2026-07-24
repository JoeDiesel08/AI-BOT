import random
import copy
from trading_agents import TradingAgent

class GenerationOptimizer:
    def __init__(self, population_size=10, mutation_rate=0.2, market_limits=None):
        """
        Manages a population of trading agents and evolves them over time.
        
        Args:
            population_size: Number of agents in the population.
            mutation_rate: Probability of mutating each parameter.
            market_limits: Optional dict with 'min_qty' and 'qty_precision' for
                           position sizing. Defaults to BTC/USDT-like limits.
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.market_limits = market_limits if market_limits is not None else {'min_qty': 0.00001, 'qty_precision': 8}
        
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
                max_drawdown_pct=max_drawdown_pct
            )
            self.population.append(agent)
        print(f"Successfully initialized {self.population_size} random agents with multi-indicator strategies.")

    def score_population(self, market_df):
        """
        Runs every agent against the real market data and ranks them by performance.
        Returns a list of tuples: (agent, final_portfolio_value)
        """
        ranked_agents = []
        
        for agent in self.population:
            # Get decisions from the agent based on market data
            decisions = agent.evaluate_market(market_df)
            # Track how much money the agent finishes with
            final_value = agent.simulate_trading(market_df, decisions, market_limits=self.market_limits)
            ranked_agents.append((agent, final_value))
            
        # Sort agents by highest ending value (profit) first
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
                max_drawdown_pct=max_drawdown_pct
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
                max_drawdown_pct=child_max_drawdown_pct
            )
            new_population.append(child_agent)
            
        self.population = new_population

# --- TEST UTILITY ---
if __name__ == "__main__":
    print("Testing Generation Optimizer structure...")
    optimizer = GenerationOptimizer(population_size=5)
    optimizer.initialize_population()
    print("Optimizer setup verification passed!")

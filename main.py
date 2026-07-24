import os

from data_engine import get_real_data
from genetic_optimizer import GenerationOptimizer


def main():
    # Web deployments often need smaller/faster defaults; local runs can use full settings.
    population_size = int(os.environ.get("POPULATION_SIZE", "20"))
    num_generations = int(os.environ.get("NUM_GENERATIONS", "10"))
    data_limit = int(os.environ.get("DATA_LIMIT", "200"))

    # Step 1: Fetch real Bitcoin data from Binance
    print("Fetching real Bitcoin data...")
    market_data = get_real_data(symbol="BTC/USDT", timeframe="15m", limit=data_limit)
    
    if market_data.empty:
        print("Failed to fetch market data. Exiting.")
        return
    
    print(f"Successfully fetched {len(market_data)} data points.\n")
    
    # Step 2: Initialize the genetic optimizer
    print("Initializing genetic optimizer...")
    print(f"Population: {population_size}, Generations: {num_generations}")
    btc_usdt_limits = {'min_qty': 0.00001, 'qty_precision': 8}
    optimizer = GenerationOptimizer(population_size=population_size, mutation_rate=0.3, market_limits=btc_usdt_limits)
    optimizer.initialize_population()
    
    # Step 3: Run evolution for more generations
    for generation in range(1, num_generations + 1):
        print(f"\n=== Generation {generation} ===")
        
        # Score the current population
        ranked_agents = optimizer.score_population(market_data)
        
        # Print top performers
        print(f"Top 3 agents in Generation {generation}:")
        for i, (agent, value) in enumerate(ranked_agents[:3], 1):
            print(f"  {i}. {agent.agent_id}:")
            print(f"     SMA({agent.sma_short_period}, {agent.sma_long_period}) | RSI({agent.rsi_period}, {agent.rsi_oversold}, {agent.rsi_overbought}) | MACD({agent.macd_fast}, {agent.macd_slow}, {agent.macd_signal}) | Vol({agent.volume_sma_period}, {agent.volume_threshold:.2f})")
            trailing = "Yes" if agent.use_trailing_stop else "No"
            effective_profit = agent.stop_loss_pct * agent.risk_reward_ratio
            print(f"     Risk: Stop={agent.stop_loss_pct:.2%}, RR={agent.risk_reward_ratio:.2f}, EffectiveProfit={effective_profit:.2%}, Trailing={trailing}, RiskPct={agent.risk_pct:.2%}, MaxDD={agent.max_drawdown_pct:.2%}")
            print(f"     -> ${value:.2f}")
        
        # Evolve to next generation (skip after the last generation)
        if generation < num_generations:
            optimizer.evolve_population(ranked_agents, generation + 1)
    
    # Step 4: Print the most profitable agent's parameters
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    best_agent, best_value = ranked_agents[0]
    print(f"Most Profitable Agent: {best_agent.agent_id}")
    print(f"Parameters:")
    print(f"  - SMA: Short={best_agent.sma_short_period}, Long={best_agent.sma_long_period}")
    print(f"  - RSI: Period={best_agent.rsi_period}, Oversold={best_agent.rsi_oversold}, Overbought={best_agent.rsi_overbought}")
    print(f"  - MACD: Fast={best_agent.macd_fast}, Slow={best_agent.macd_slow}, Signal={best_agent.macd_signal}")
    print(f"  - Volume: SMA Period={best_agent.volume_sma_period}, Threshold={best_agent.volume_threshold:.2f}")
    trailing = "Yes" if best_agent.use_trailing_stop else "No"
    best_effective_profit = best_agent.stop_loss_pct * best_agent.risk_reward_ratio
    print(f"  - Risk: Stop Loss={best_agent.stop_loss_pct:.2%}, RR={best_agent.risk_reward_ratio:.2f}, EffectiveProfit={best_effective_profit:.2%}, Trailing Stop={trailing}, RiskPct={best_agent.risk_pct:.2%}, MaxDD={best_agent.max_drawdown_pct:.2%}")
    print(f"  - Final Portfolio Value: ${best_value:.2f}")
    print(f"  - Starting Value: $1000.00")
    print(f"  - Profit/Loss: ${best_value - 1000:.2f}")

if __name__ == "__main__":
    main()

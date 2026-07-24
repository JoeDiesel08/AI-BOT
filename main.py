import os
import json
from pathlib import Path

from data_engine import get_real_data
from genetic_optimizer import GenerationOptimizer


MODE = os.environ.get("MODE", "optimize").lower()


seed_path = os.environ.get("SEED_FILE", "best_agent.json")
SEED_FILE = Path(seed_path) if os.path.isabs(seed_path) else Path(__file__).parent / seed_path
SEED_VERSION = 3  # bump when fitness scale changes so old seeds are replaced


def load_seed():
    """Load a previously saved best agent seed if it exists."""
    if not SEED_FILE.exists():
        return None
    try:
        with open(SEED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded previous best agent seed from {SEED_FILE.name}.\n")
        return data.get("params", data)
    except Exception as e:
        print(f"Could not load seed file: {e}")
        return None


def save_seed(best_agent, fitness, final_value):
    """Save the best agent from this run if it beats the previous seed."""
    payload = {
        "version": SEED_VERSION,
        "fitness": fitness,
        "final_value": final_value,
        "params": best_agent.to_dict(),
    }
    try:
        existing_fitness = float("-inf")
        existing_final_value = float("-inf")
        existing_version = None
        if SEED_FILE.exists():
            with open(SEED_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
                existing_version = existing.get("version")
                existing_fitness = existing.get("fitness", float("-inf"))
                existing_final_value = existing.get("final_value", float("-inf"))

        # If the saved seed predates a fitness-scale change, ignore its scores.
        if existing_version != SEED_VERSION:
            existing_fitness = float("-inf")
            existing_final_value = float("-inf")

        if fitness > existing_fitness or final_value > existing_final_value:
            SEED_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SEED_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"\nSaved new best agent seed to {SEED_FILE.name} (fitness: {fitness:.2f}).")
        else:
            print(f"\nKept existing best agent seed (fitness: {existing_fitness:.2f}) > current ({fitness:.2f}).")
    except Exception as e:
        print(f"\nWarning: could not save best agent seed: {e}")


def main():
    # Dispatch to the live/paper trading loop if requested, otherwise run the GA optimizer.
    if MODE in ("paper", "live"):
        if MODE == "live":
            os.environ["PAPER_TRADING"] = "false"
        from trading_bot import main as trading_bot_main
        trading_bot_main()
        return

    # Web deployments often need smaller/faster defaults; local runs can use full settings.
    population_size = int(os.environ.get("POPULATION_SIZE", "20"))
    num_generations = int(os.environ.get("NUM_GENERATIONS", "10"))
    data_limit = int(os.environ.get("DATA_LIMIT", "500"))
    timeframe = os.environ.get("TIMEFRAME", "4h")
    risk_penalty = float(os.environ.get("RISK_PENALTY", "0.5"))
    trading_pair = os.environ.get("TRADING_PAIR", "BTC/USD")

    # Step 1: Fetch real Bitcoin data from Kraken (with fallbacks)
    print(f"Fetching real Bitcoin data ({trading_pair}, {timeframe}, {data_limit} candles)...")
    market_data = get_real_data(symbol=trading_pair, timeframe=timeframe, limit=data_limit)
    
    if market_data.empty:
        print("Failed to fetch market data. Exiting.")
        return
    
    print(f"Successfully fetched {len(market_data)} data points.\n")
    
    # Step 2: Initialize the genetic optimizer
    print("Initializing genetic optimizer...")
    print(f"Population: {population_size}, Generations: {num_generations}")
    btc_usdt_limits = {'min_qty': 0.00001, 'qty_precision': 8}
    seed_agent = load_seed()
    optimizer = GenerationOptimizer(
        population_size=population_size,
        mutation_rate=0.3,
        market_limits=btc_usdt_limits,
        risk_penalty=risk_penalty,
        seed_agent=seed_agent
    )
    optimizer.initialize_population()
    
    # Step 3: Run evolution for more generations
    for generation in range(1, num_generations + 1):
        print(f"\n=== Generation {generation} ===")
        
        # Score the current population
        ranked_agents = optimizer.score_population(market_data)
        
        # Print top performers
        print(f"Top 3 agents in Generation {generation}:")
        for i, (agent, fitness, value, max_dd, trade_count) in enumerate(ranked_agents[:3], 1):
            print(f"  {i}. {agent.agent_id}:")
            print(f"     SMA({agent.sma_short_period}, {agent.sma_long_period}) | RSI({agent.rsi_period}, {agent.rsi_oversold}, {agent.rsi_overbought}) | MACD({agent.macd_fast}, {agent.macd_slow}, {agent.macd_signal}) | Vol({agent.volume_sma_period}, {agent.volume_threshold:.2f})")
            trailing = "Yes" if agent.use_trailing_stop else "No"
            trend_filter = "Yes" if agent.use_trend_filter else "No"
            require_volume = "Yes" if agent.require_volume else "No"
            effective_profit = agent.stop_loss_pct * agent.risk_reward_ratio
            print(f"     Risk: Stop={agent.stop_loss_pct:.2%}, RR={agent.risk_reward_ratio:.2f}, EffectiveProfit={effective_profit:.2%}, Trailing={trailing}, RiskPct={agent.risk_pct:.2%}, MaxDD={agent.max_drawdown_pct:.2%}")
            adx_filter = "Yes" if agent.use_adx_filter else "No"
            print(f"     Trend Filter: {trend_filter}, Trend SMA Period: {agent.trend_sma_period}")
            print(f"     Signal Filter: Threshold={agent.signal_threshold}, RequireVolume={require_volume}")
            print(f"     ADX Filter: {adx_filter}, ADX Period={agent.adx_period}, ADX Threshold={agent.adx_threshold:.1f}")
            print(f"     Costs: Commission={agent.commission_pct:.3%}, Slippage={agent.slippage_pct:.3%}")
            print(f"     -> Fitness Score={fitness:.2f} | Final=${value:.2f} | MaxDD={max_dd:.2%} | Trades={trade_count}")
        
        # Evolve to next generation (skip after the last generation)
        if generation < num_generations:
            optimizer.evolve_population(ranked_agents, generation + 1)
    
    # Step 4: Print the most profitable agent's parameters
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    best_agent, best_fitness, best_value, best_max_dd, best_trade_count = ranked_agents[0]
    print(f"Most Profitable Agent: {best_agent.agent_id}")
    print(f"Parameters:")
    print(f"  - SMA: Short={best_agent.sma_short_period}, Long={best_agent.sma_long_period}")
    print(f"  - RSI: Period={best_agent.rsi_period}, Oversold={best_agent.rsi_oversold}, Overbought={best_agent.rsi_overbought}")
    print(f"  - MACD: Fast={best_agent.macd_fast}, Slow={best_agent.macd_slow}, Signal={best_agent.macd_signal}")
    print(f"  - Volume: SMA Period={best_agent.volume_sma_period}, Threshold={best_agent.volume_threshold:.2f}")
    trailing = "Yes" if best_agent.use_trailing_stop else "No"
    trend_filter = "Yes" if best_agent.use_trend_filter else "No"
    require_volume = "Yes" if best_agent.require_volume else "No"
    best_effective_profit = best_agent.stop_loss_pct * best_agent.risk_reward_ratio
    print(f"  - Risk: Stop Loss={best_agent.stop_loss_pct:.2%}, RR={best_agent.risk_reward_ratio:.2f}, EffectiveProfit={best_effective_profit:.2%}, Trailing Stop={trailing}, RiskPct={best_agent.risk_pct:.2%}, MaxDD={best_agent.max_drawdown_pct:.2%}")
    adx_filter = "Yes" if best_agent.use_adx_filter else "No"
    print(f"  - Trend Filter: {trend_filter}, Trend SMA Period: {best_agent.trend_sma_period}")
    print(f"  - Signal Filter: Threshold={best_agent.signal_threshold}, RequireVolume={require_volume}")
    print(f"  - ADX Filter: {adx_filter}, ADX Period={best_agent.adx_period}, ADX Threshold={best_agent.adx_threshold:.1f}")
    print(f"  - Costs: Commission={best_agent.commission_pct:.3%}, Slippage={best_agent.slippage_pct:.3%}")
    print(f"  - Timeframe: {timeframe}, Data Candles: {len(market_data)}")
    print(f"  - Fitness Score: {best_fitness:.2f} | Final Portfolio Value: ${best_value:.2f} | Observed MaxDD: {best_max_dd:.2%} | Trades: {best_trade_count}")
    print(f"  - Starting Value: $1000.00")
    print(f"  - Profit/Loss: ${best_value - 1000:.2f}")

    save_seed(best_agent, best_fitness, best_value)

if __name__ == "__main__":
    main()

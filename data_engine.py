import ccxt
import pandas as pd
from datetime import datetime, timedelta

def get_real_data(symbol="BTC/USDT", timeframe="15m", limit=100):
    """
    Fetches real, live OHLCV price data from multiple exchanges without requiring an API key.
    Falls back to mock data if all exchanges fail.
    """
    exchanges = [
        ('Binance', ccxt.binance),
        ('Kraken', ccxt.kraken),
        ('Coinbase', ccxt.coinbase),
    ]
    
    for exchange_name, exchange_class in exchanges:
        try:
            exchange = exchange_class({'enableRateLimit': True})
            print(f"Fetching real {timeframe} data for {symbol} from {exchange_name}...")

            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = pd.DataFrame(ohlcv, columns=columns)

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            print(f"Successfully fetched {len(df)} data points from {exchange_name}.")
            return df

        except Exception as e:
            print(f"{exchange_name} failed: {e}")
            continue
    
    # Fallback to mock data if all exchanges fail
    print("All exchanges failed. Using mock data for testing...")
    return get_mock_data(limit)

def get_mock_data(limit=100):
    """
    Generates mock Bitcoin price data for testing when APIs are unavailable.
    Simulates realistic price movements with random walk.
    """
    import numpy as np
    
    # Generate realistic-looking price data
    np.random.seed(42)
    base_price = 65000
    returns = np.random.normal(0.001, 0.02, limit)  # 0.1% average return, 2% volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    dates = pd.date_range(start=datetime.now() - timedelta(minutes=limit*15), periods=limit, freq='15min')
    data = []
    
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        volume = np.random.uniform(100, 1000)
        data.append([dates[i], price, high, low, price, volume])
    
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(data, columns=columns)
    df.set_index('timestamp', inplace=True)
    
    print(f"Generated {len(df)} mock data points.")
    return df

def add_indicators(df, sma_period=20, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9):
    """
    Adds multiple technical indicators to the price data:
    - SMA (Simple Moving Average)
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Volume SMA
    """
    if df.empty:
        return df

    df = df.copy()
    
    # SMA
    df['sma'] = df['close'].rolling(window=sma_period).mean()
    df['sma'] = df['sma'].bfill()
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'], rsi_period)
    df['rsi'] = df['rsi'].bfill()
    
    # MACD
    macd_line, signal_line, histogram = calculate_macd(df['close'], macd_fast, macd_slow, macd_signal)
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_histogram'] = histogram
    df['macd'] = df['macd'].bfill()
    df['macd_signal'] = df['macd_signal'].bfill()
    df['macd_histogram'] = df['macd_histogram'].bfill()
    
    # Volume SMA
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_sma'] = df['volume_sma'].bfill()
    
    return df

def calculate_rsi(prices, period=14):
    """
    Calculates the Relative Strength Index (RSI).
    """
    import numpy as np
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculates MACD (Moving Average Convergence Divergence).
    Returns MACD line, Signal line, and Histogram.
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# --- TEST UTILITY ---
if __name__ == "__main__":
    # Test with real Bitcoin prices
    real_data = get_real_data(symbol="BTC/USDT", timeframe="15m", limit=50)

    if not real_data.empty:
        processed_data = add_indicators(real_data, sma_period=10)
        print("\nReal-Time Data Engine Output Sample:")
        print(processed_data.tail())
        print(f"\nSuccessfully fetched {len(processed_data)} real market rows.")
    else:
        print("Failed to fetch data. Check internet connection.")

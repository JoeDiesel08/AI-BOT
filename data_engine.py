import os
import time
import hashlib
from pathlib import Path

import ccxt
import pandas as pd
from datetime import datetime, timedelta

# Disk cache for fetched OHLCV so repeated runs on Fly load instantly
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_TTL_SECONDS = int(os.environ.get("DATA_CACHE_TTL_SECONDS", "3600"))


def _cache_key(symbol, timeframe, limit):
    return hashlib.md5(f"{symbol}|{timeframe}|{limit}".encode()).hexdigest()


def _load_cached_df(cache_file):
    if not cache_file.exists():
        return None
    try:
        age = time.time() - cache_file.stat().st_mtime
        if age > CACHE_TTL_SECONDS:
            return None
        df = pd.read_csv(cache_file, index_col='timestamp', parse_dates=True)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _save_cached_df(df, cache_file):
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=True)
    except Exception:
        pass


def get_real_data(symbol="BTC/USDT", timeframe="15m", limit=100):
    """
    Fetches real, live OHLCV price data from multiple exchanges without requiring an API key.
    Uses a disk cache so repeated runs on Fly load the data instantly.
    Falls back to mock data if all exchanges fail.
    """
    cache_file = CACHE_DIR / f"{_cache_key(symbol, timeframe, limit)}.csv"
    cached = _load_cached_df(cache_file)
    if cached is not None and len(cached) >= limit * 0.9:
        print(f"Using cached {timeframe} data for {symbol} ({len(cached)} rows).")
        return cached

    # Kraken is the primary source per the new dual-mode architecture.
    # Binance and Coinbase remain as resilient fallbacks.
    exchanges = [
        ('Kraken', ccxt.kraken),
        ('Binance', ccxt.binance),
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
            _save_cached_df(df, cache_file)
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

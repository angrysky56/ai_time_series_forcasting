import ccxt
import pandas as pd
from datetime import datetime

def fetch_crypto_data(symbol: str, timeframe: str = '1d', limit: int = 100):
    """
    Fetches historical cryptocurrency data from Binance.

    Args:
        symbol (str): The cryptocurrency symbol (e.g., 'BTC/USDT').
        timeframe (str): The timeframe for the data (e.g., '1d', '1h', '5m').
        limit (int): The number of data points to retrieve.

    Returns:
        pd.DataFrame: A DataFrame containing the historical data, or None if an error occurs.
    """
    try:
        exchange = ccxt.kraken()  # Switched to Kraken due to Binance geo-restrictions
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        if not ohlcv:
            print(f"No data found for symbol {symbol}")
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df

    except ccxt.NetworkError as e:
        print(f"Network error while fetching data for {symbol}: {e}")
        return None
    except ccxt.ExchangeError as e:
        print(f"Exchange error for symbol {symbol}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for {symbol}: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    btc_symbol = 'BTC/USDT'
    btc_data = fetch_crypto_data(btc_symbol, timeframe='1d', limit=365)

    if btc_data is not None:
        print(f"Successfully fetched {len(btc_data)} data points for {btc_symbol}.")
        print(btc_data.head())
        print(btc_data.tail())

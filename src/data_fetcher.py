import ccxt
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Free exchanges with good API support for smaller timeframes
SUPPORTED_EXCHANGES = {
    'binance': ccxt.binance,
    'kraken': ccxt.kraken,
    'coinbase': ccxt.coinbase,
    'bitstamp': ccxt.bitstamp,
    'bitfinex': ccxt.bitfinex,
    'huobi': ccxt.huobi,
    'okx': ccxt.okx,
    'kucoin': ccxt.kucoin,
    'bybit': ccxt.bybit,
    'gateio': ccxt.gateio
}

def get_available_exchanges() -> list[str]:
    """Get list of supported exchange names."""
    return list(SUPPORTED_EXCHANGES.keys())

def discover_symbols(exchange_name: str = 'binance', limit: int = 100) -> list[str]:
    """
    Discover available trading symbols from an exchange.

    Args:
        exchange_name (str): Name of the exchange to query
        limit (int): Maximum number of symbols to return

    Returns:
        list[str]: List of available trading symbols
    """
    try:
        if exchange_name not in SUPPORTED_EXCHANGES:
            logger.error(f"Exchange {exchange_name} not supported")
            return []

        exchange_class = SUPPORTED_EXCHANGES[exchange_name]
        exchange = exchange_class()

        # Load markets to get symbols
        markets = exchange.load_markets()
        symbols = list(markets.keys())

        # Filter for USDT pairs primarily (most liquid)
        usdt_symbols = [s for s in symbols if 'USDT' in s]
        if usdt_symbols:
            symbols = usdt_symbols

        # Sort by symbol name and limit results
        symbols.sort()
        return symbols[:limit]

    except Exception as e:
        logger.error(f"Error discovering symbols from {exchange_name}: {e}")
        return []

def fetch_crypto_data(
    symbol: str,
    timeframe: str = '1d',
    limit: int = 100,
    exchange_name: str = 'binance'
) -> pd.DataFrame | None:
    """
    Fetches historical cryptocurrency data from multiple exchanges with support for smaller timeframes.

    Args:
        symbol (str): The cryptocurrency symbol (e.g., 'BTC/USDT').
        timeframe (str): The timeframe for the data (e.g., '1d', '1h', '4h', '15m', '5m', '1m').
        limit (int): The number of data points to retrieve.
        exchange_name (str): Name of the exchange to use.

    Returns:
        pd.DataFrame: A DataFrame containing the historical data, or None if an error occurs.
    """
    try:
        if exchange_name not in SUPPORTED_EXCHANGES:
            logger.error(f"Exchange {exchange_name} not supported. Available: {list(SUPPORTED_EXCHANGES.keys())}")
            return None

        exchange_class = SUPPORTED_EXCHANGES[exchange_name]
        exchange = exchange_class()

        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        if not ohlcv:
            logger.warning(f"No data found for symbol {symbol} on {exchange_name}")
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Add exchange info for tracking
        df['exchange'] = exchange_name
        df['symbol'] = symbol
        df['timeframe'] = timeframe

        logger.info(f"Successfully fetched {len(df)} records for {symbol} from {exchange_name}")
        return df

    except ccxt.NetworkError as e:
        logger.error(f"Network error while fetching data for {symbol} from {exchange_name}: {e}")
        return None
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error for symbol {symbol} on {exchange_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error occurred for {symbol} on {exchange_name}: {e}")
        return None

def try_multiple_exchanges(
    symbol: str,
    timeframe: str = '1d',
    limit: int = 100,
    preferred_exchanges: list[str] | None = None
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Try to fetch data from multiple exchanges until successful.

    Args:
        symbol (str): The cryptocurrency symbol
        timeframe (str): The timeframe for the data
        limit (int): Number of data points to retrieve
        preferred_exchanges (list[str]): List of exchanges to try in order

    Returns:
        tuple: (DataFrame, successful_exchange_name) or (None, None)
    """
    if preferred_exchanges is None:
        # Default order prioritizing major exchanges with good free API limits
        preferred_exchanges = ['binance', 'kraken', 'coinbase', 'kucoin', 'bybit']

    for exchange_name in preferred_exchanges:
        if exchange_name in SUPPORTED_EXCHANGES:
            logger.info(f"Trying to fetch {symbol} from {exchange_name}")
            data = fetch_crypto_data(symbol, timeframe, limit, exchange_name)
            if data is not None:
                return data, exchange_name

    logger.error(f"Failed to fetch {symbol} from all tried exchanges")
    return None, None

def validate_timeframe_support(exchange_name: str, timeframe: str) -> bool:
    """
    Validate if an exchange supports a specific timeframe.

    Args:
        exchange_name (str): Name of the exchange
        timeframe (str): Timeframe to validate

    Returns:
        bool: True if supported, False otherwise
    """
    try:
        if exchange_name not in SUPPORTED_EXCHANGES:
            return False

        exchange_class = SUPPORTED_EXCHANGES[exchange_name]
        exchange = exchange_class()

        # Get supported timeframes
        if hasattr(exchange, 'timeframes'):
            return timeframe in exchange.timeframes
        else:
            # Fallback: try common timeframes
            common_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
            return timeframe in common_timeframes

    except Exception:
        return False

if __name__ == '__main__':
    # Example usage:
    btc_symbol = 'BTC/USDT'
    btc_data = fetch_crypto_data(btc_symbol, timeframe='1d', limit=365)

    if btc_data is not None:
        print(f"Successfully fetched {len(btc_data)} data points for {btc_symbol}.")
        print(btc_data.head())
        print(btc_data.tail())

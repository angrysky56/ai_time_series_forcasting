import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Literal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DataSource = Literal['crypto', 'stock', 'auto']
TimeFrame = Literal['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']

# Map timeframes to yfinance intervals
TIMEFRAME_TO_YFINANCE = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '1h',        # Fixed: Use proper 1h interval
    '4h': '1d',        # yfinance doesn't support 4h, use 1d and resample if needed
    '1d': '1d',
    '1w': '1wk',       # Fixed: Use proper 1wk interval
    '1wk': '1wk',      # Added alias for consistency
    '1M': '1mo',
    '1mo': '1mo'       # Added alias for consistency
}

# Maximum historical data periods for yfinance
TIMEFRAME_TO_PERIOD = {
    '1m': '7d',      # Max 7 days for minute data
    '5m': '60d',     # Max 60 days for 5m data
    '15m': '60d',
    '30m': '60d',
    '1h': 'max',     # yfinance supports hourly for longer periods
    '4h': 'max',
    '1d': 'max',     # All available daily data
    '1w': 'max',     # All available weekly data
    '1wk': 'max',    # Alias for weekly
    '1M': 'max',
    '1mo': 'max'     # Alias for monthly
}




def get_available_exchanges() -> list[str]:
    """Get list of supported data sources."""
    return ['yfinance']

def _fetch_screener_symbols(quote_type: str, limit: int, region: str = 'us') -> list[str]:
    """Fetch symbols from Yahoo Finance screener by quote_type with pagination.

    quote_type: 'EQUITY' | 'ETF' | 'CRYPTOCURRENCY' | 'INDEX'
    """
    import requests

    url = "https://query2.finance.yahoo.com/v1/finance/screener"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    symbols: list[str] = []
    size = 250
    offset = 0
    while len(symbols) < limit:
        payload = {
            "offset": offset,
            "size": min(size, limit - len(symbols)),
            "sortField": "symbol",
            "sortType": "asc",
            "quoteType": quote_type,
            "query": {
                "operator": "AND",
                "operands": [
                    {"operator": "eq", "operands": ["region", region]}
                ]
            },
            "userId": "",
            "userIdType": "guid"
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=15)
            logger.info(f"[DEBUG] Screener POST {url} status={resp.status_code} ok={resp.ok}")
            if not resp.ok:
                logger.error(f"[DEBUG] Screener API error: {resp.text}")
                break
            data = resp.json()
            logger.info(f"[DEBUG] Screener API data sample: {str(data)[:500]}")
            quotes = (
                data.get('finance', {})
                    .get('result', [{}])[0]
                    .get('quotes', [])
            )
            logger.info(f"[DEBUG] Screener quotes count: {len(quotes)}")
            if not quotes:
                break
            for q in quotes:
                sym = q.get('symbol')
                if sym:
                    symbols.append(sym)
                    if len(symbols) >= limit:
                        break
            offset += size
        except Exception as e:
            logger.error(f"[DEBUG] Screener exception: {e}")
            break
    return symbols



def discover_symbols(exchange_name: str = 'yfinance', limit: int = 1000, category: str = 'all') -> list[str]:
    """
    Discover available trading symbols using yfinance.Lookup by category.

    Args:
        exchange_name (str): Data source name (kept for compatibility)
        limit (int): Maximum number of symbols to return
        category (str): Symbol category ('all', 'stocks', 'crypto', 'etfs', 'index')

    Returns:
        list[str]: List of available trading symbols
    """
    import yfinance as yf

    cat = category.lower()
    symbols = []
    if cat in ('stock', 'stocks'):
        try:
            symbols = yf.Lookup("*").get_stock(count=limit)
        except Exception:
            symbols = []
    elif cat in ('etf', 'etfs'):
        try:
            symbols = yf.Lookup("*").get_etf(count=limit)
        except Exception:
            symbols = []
    elif cat in ('crypto', 'cryptocurrency', 'cryptocurrencies'):
        try:
            symbols = yf.Lookup("*").get_cryptocurrency(count=limit)
        except Exception:
            symbols = []
    elif cat in ('index', 'indexes', 'indices'):
        try:
            symbols = yf.Lookup("*").get_index(count=limit)
        except Exception:
            symbols = []
    elif cat == 'all':
        # Fill from all categories until limit reached
        for func in [
            lambda n: yf.Lookup("*").get_stock(count=n),
            lambda n: yf.Lookup("*").get_etf(count=n),
            lambda n: yf.Lookup("*").get_cryptocurrency(count=n),
            lambda n: yf.Lookup("*").get_index(count=n)
        ]:
            if len(symbols) >= limit:
                break
            needed = limit - len(symbols)
            try:
                symbols.extend(func(needed))
            except Exception:
                continue
    else:
        # Unknown category: default to stocks
        try:
            symbols = yf.Lookup("*").get_stock(count=limit)
        except Exception:
            symbols = []

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in symbols:
        if isinstance(s, dict):
            sym = s.get('symbol')
        else:
            sym = str(s)
        if sym and sym not in seen:
            seen.add(sym)
            unique.append(sym)
            if len(unique) >= limit:
                break
    return unique


def search_symbols(query: str, limit: int = 1000) -> list[dict[str, str]]:
    """
    Search for symbols by company name, ticker, or keyword.

    Args:
        query (str): Search query (company name, ticker, etc.)
        limit (int): Maximum number of results to return

    Returns:
        list[dict]: List of symbol matches with metadata
    """
    import requests

    results: list[dict[str, str]] = []
    try:
        # Use Yahoo Finance public search API
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": query, "quotesCount": limit, "newsCount": 0}
        resp = requests.get(url, params=params, timeout=10)
        if resp.ok:
            data = resp.json()
            for q in data.get('quotes', []):
                sym = q.get('symbol')
                if not sym:
                    continue
                name = q.get('longname') or q.get('shortname') or q.get('name') or sym
                exch = q.get('exchDisp') or q.get('exchange') or 'Yahoo Finance'
                quote_type = (q.get('quoteType') or '').lower()
                if quote_type == 'crypto' or sym.endswith('-USD'):
                    typ = 'crypto'
                elif quote_type == 'etf':
                    typ = 'etf'
                elif quote_type == 'index' or sym.startswith('^'):
                    typ = 'index'
                else:
                    typ = 'stock'
                results.append({'symbol': sym, 'name': name, 'exchange': exch, 'type': typ})
                if len(results) >= limit:
                    return results[:limit]

        return results[:limit]

    except Exception as e:
        logger.error(f"Symbol search failed for '{query}': {e}")
        return []


def _infer_symbol_type(symbol: str, info: dict | None) -> str:
    """Infer human-friendly type from Yahoo info or symbol shape."""
    if isinstance(info, dict):
        qt = (info.get('quoteType') or info.get('quote_type') or '').upper()
        if qt == 'CRYPTOCURRENCY':
            return 'crypto'
        if qt == 'ETF':
            return 'etf'
        if qt == 'INDEX':
            return 'index'
        if qt == 'EQUITY':
            return 'stock'
    # Heuristics
    if symbol.startswith('^'):
        return 'index'
    if symbol.endswith('-USD') or symbol.endswith('-USDT') or '/' in symbol:
        return 'crypto'
    return 'stock'


def validate_symbol(symbol: str) -> dict[str, str | bool | list[str]]:
    """
    Validate if a symbol exists and return metadata.

    Args:
        symbol (str): Symbol to validate

    Returns:
        dict: Validation result with metadata
    """
    import yfinance as yf


    try:
        # Normalize crypto symbols to yfinance format (generic)
        if '/' in symbol:
            symbol = symbol.replace('/USDT', '-USD').replace('/USD', '-USD')

        ticker = yf.Ticker(symbol)

        # Lightweight validation via small history fetch
        has_data = False
        try:
            hist = ticker.history(period='5d', interval='1d')
            has_data = not hist.empty
        except Exception:
            has_data = False

        name = symbol
        exchange = 'Yahoo Finance'
        currency = 'USD'
        try:
            # Use fast_info when available
            fast = getattr(ticker, 'fast_info', None)
            if fast is not None:
                currency = getattr(fast, 'currency', currency) or currency
        except Exception:
            pass
        info: dict | None = None
        try:
            # get_info is lighter than .info when available
            if hasattr(ticker, 'get_info'):
                info = ticker.get_info()
            else:
                raw = getattr(ticker, 'info', None)
                if isinstance(raw, dict):
                    info = raw
            if isinstance(info, dict) and info:
                name = info.get('longName') or info.get('shortName') or name
                exchange = info.get('exchange', exchange)
                currency = info.get('currency', currency)
        except Exception:
            pass

        if has_data:
            return {
                'valid': True,
                'symbol': symbol,
                'name': name,
                'type': _infer_symbol_type(symbol, info),
                'exchange': exchange,
                'currency': currency,
                'message': 'Symbol validated successfully'
            }
        else:
            return {
                'valid': False,
                'symbol': symbol,
                'message': f"Symbol '{symbol}' not found on Yahoo Finance. Try searching for similar symbols.",
                'suggestions': _get_symbol_suggestions(symbol)
            }

    except Exception as e:
        return {
            'valid': False,
            'symbol': symbol,
            'message': f"Validation failed: {str(e)}",
            'suggestions': _get_symbol_suggestions(symbol)
        }


def _classify_symbol_type(symbol: str) -> str:
    """Deprecated: kept for compatibility. Use _infer_symbol_type instead."""
    return _infer_symbol_type(symbol, None)


def _get_symbol_suggestions(symbol: str) -> list[str]:
    """Get suggestions using Yahoo search API (no curated lists)."""
    suggestions = [r['symbol'] for r in search_symbols(symbol, limit=5)]
    return suggestions

def fetch_universal_data(
    symbol: str,
    timeframe: str = '1d',
    limit: int = 1000
) -> pd.DataFrame | None:
    """
    Fetches historical data using yfinance for any symbol type.

    Args:
        symbol: Trading symbol (e.g., 'AAPL', 'BTC-USD', 'BTC/USDT')
        timeframe: The timeframe for the data
        limit: Number of data points to retrieve

    Returns:
        pd.DataFrame: Historical data or None if failed
    """
    try:
        # Normalize crypto symbols to yfinance format (generic)
        if '/' in symbol:
            symbol = symbol.replace('/USDT', '-USD').replace('/USD', '-USD')
        # Validate symbol first
        validation = validate_symbol(symbol)
        if not validation['valid']:
            logger.error(f"Invalid symbol '{symbol}': {validation['message']}")
            suggestions = validation.get('suggestions')
            if isinstance(suggestions, list):
                logger.info(f"Suggestions: {', '.join(suggestions)}")
            return None

        # Use the validated symbol
        validated_symbol = validation['symbol']

        yf_interval = TIMEFRAME_TO_YFINANCE.get(timeframe, '1d')
        period = TIMEFRAME_TO_PERIOD.get(timeframe, 'max')

        # Create ticker and fetch data
        ticker = yf.Ticker(validated_symbol)

        # Calculate appropriate date range based on limit and timeframe
        if timeframe in ['1m', '5m', '15m', '30m']:
            # For intraday data below 1h, use explicit date range with strict limits
            end_date = datetime.now()

            # Initialize days_back with a default value
            days_back = 1
            # Calculate start date based on timeframe and limit
            if timeframe == '1m':
                days_back = min(7, limit // 390 + 1)  # ~390 minutes per trading day
            elif timeframe == '5m':
                days_back = min(60, limit // 78 + 1)  # ~78 5-min bars per day
            elif timeframe == '15m':
                days_back = min(60, limit // 26 + 1)
            elif timeframe == '30m':
                days_back = min(60, limit // 13 + 1)

            start_date = end_date - timedelta(days=days_back)
            df = ticker.history(start=start_date, end=end_date, interval=yf_interval)
        else:
            # For 1h and above, use period-based fetching for better data quality
            period = TIMEFRAME_TO_PERIOD.get(timeframe, 'max')
            df = ticker.history(period=period, interval=yf_interval)

        if df.empty:
            logger.error(f"No data returned for {validated_symbol} with interval {yf_interval}")
            # Provide helpful suggestions
            if timeframe in ['1m', '5m', '15m', '30m']:
                logger.info("💡 Tip: Intraday data may have limited history. Try daily ('1d') or weekly ('1w') timeframes")
            elif validation['type'] == 'crypto':
                logger.info("💡 Tip: Some crypto symbols may have limited data. Try major cryptos like BTC-USD or ETH-USD")
            return None

        # Standardize column names
        df = df.reset_index()
        df.columns = df.columns.str.lower()

        # Handle different index column names from yfinance
        if 'date' in df.columns:
            df.rename(columns={'date': 'timestamp'}, inplace=True)
        elif 'datetime' in df.columns:
            df.rename(columns={'datetime': 'timestamp'}, inplace=True)

        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Clean data - remove any NaN values
        df = df.dropna()

        # Sort by timestamp to ensure proper order
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Limit to requested number of rows (take most recent)
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)

        # Add metadata
        df['symbol'] = validated_symbol
        df['timeframe'] = timeframe
        df['source'] = 'yfinance'
        df['symbol_type'] = validation['type']

        logger.info(f"✅ Successfully fetched {len(df)} records for {validated_symbol} ({validation['type']})")
        return df

    except Exception as e:
        logger.error(f"❌ Failed to fetch {symbol}: {e}")
        # Provide contextual help based on error type
        if "404" in str(e) or "No data found" in str(e):
            logger.info("💡 Tip: Symbol not found. Use search_symbols() to find similar symbols")
        elif "429" in str(e) or "rate limit" in str(e).lower():
            logger.info("💡 Tip: Rate limited by Yahoo Finance. Wait a moment and try again")
        elif "timeout" in str(e).lower():
            logger.info("💡 Tip: Network timeout. Check connection or try again later")

        return None

# Simplified wrapper functions for compatibility
def fetch_stock_data(symbol: str, timeframe: str = '1d', limit: int = 1000) -> pd.DataFrame | None:
    """Wrapper for fetching stock data."""
    return fetch_universal_data(symbol, timeframe, limit)

def fetch_crypto_data(symbol: str, timeframe: str = '1d', limit: int = 1000, exchange_name: str = 'yfinance') -> pd.DataFrame | None:
    """Wrapper for fetching crypto data."""
    return fetch_universal_data(symbol, timeframe, limit)

def try_multiple_exchanges(symbol: str, timeframe: str = '1d', limit: int = 1000, preferred_exchanges: list[str] | None = None) -> tuple[pd.DataFrame | None, str | None]:
    """Compatibility wrapper - just uses yfinance."""
    data = fetch_universal_data(symbol, timeframe, limit)
    if data is not None:
        return data, 'yfinance'
    return None, None

def validate_timeframe_support(exchange_name: str, timeframe: str) -> bool:
    """Check if timeframe is supported."""
    return timeframe in TIMEFRAME_TO_YFINANCE

if __name__ == '__main__':
    # Example usage:
    btc_symbol = 'BTC-USD'
    btc_data = fetch_universal_data(btc_symbol, timeframe='1d', limit=365)

    if btc_data is not None:
        print(f"Successfully fetched {len(btc_data)} data points for {btc_symbol}.")
        print(btc_data.head())
        print(btc_data.tail())

import pandas as pd

def calculate_sma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculates the Simple Moving Average (SMA)."""
    df[f'SMA_{window}'] = df['close'].rolling(window=window).mean()
    return df

def calculate_ema(df: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    """Calculates the Exponential Moving Average (EMA)."""
    df[f'EMA_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
    return df

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculates the Relative Strength Index (RSI)."""
    close = pd.to_numeric(df['close'], errors='coerce').astype(float)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    return df

if __name__ == '__main__':
    # Example Usage
    from .data_fetcher import fetch_crypto_data

    symbol = 'BTC/USDT'
    data = fetch_crypto_data(symbol, limit=200)

    if data is not None:
        data = calculate_sma(data)
        data = calculate_ema(data)
        data = calculate_rsi(data)

        print(f"Data with Technical Indicators for {symbol}:")
        print(data.tail(10))

"""
Technical analysis module using talipp library for incremental indicator calculation.
Provides comprehensive technical indicators with optimal parameter selection.
"""

import pandas as pd
from typing import Any
from talipp.indicators import (
    SMA, EMA, RSI, MACD, Stoch, BB,
    ATR, OBV, VWAP,
    ADX, CCI, ROC
)
from talipp.ohlcv import OHLCVFactory
import logging

logger = logging.getLogger(__name__)

def calculate_sma(df: pd.DataFrame, window: int = 55) -> pd.DataFrame:
    """Calculate Simple Moving Average using talipp."""
    try:
        sma = SMA(period=window, input_values=df['close'].tolist())
        df[f'SMA_{window}'] = list(sma)
    except Exception as e:
        logger.error(f"Error calculating SMA: {e}")
        # Fallback to pandas
        df[f'SMA_{window}'] = df['close'].rolling(window=window).mean()
    return df

def calculate_ema(df: pd.DataFrame, span: int = 13) -> pd.DataFrame:
    """Calculate Exponential Moving Average using talipp."""
    try:
        ema = EMA(period=span, input_values=df['close'].tolist())
        df[f'EMA_{span}'] = list(ema)
    except Exception as e:
        logger.error(f"Error calculating EMA: {e}")
        # Fallback to pandas
        df[f'EMA_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
    return df

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate Relative Strength Index using talipp."""
    try:
        rsi = RSI(period=window, input_values=df['close'].tolist())
        df['RSI'] = list(rsi)
    except Exception as e:
        logger.error(f"Error calculating RSI with talipp: {e}")
        # Fallback to pandas calculation
        close = pd.to_numeric(df['close'], errors='coerce').astype(float)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi_values = 100 - (100 / (1 + rs))
        df['RSI'] = rsi_values
    return df

def calculate_comprehensive_indicators(
    df: pd.DataFrame,
    indicators: dict[str, dict[str, Any]] | None = None
) -> pd.DataFrame:
    """
    Calculate multiple technical indicators using talipp.

    Args:
        df: DataFrame with OHLCV data
        indicators: Dictionary of indicators to calculate with parameters

    Returns:
        DataFrame with calculated indicators
    """
    if indicators is None:
        # Default indicators
        indicators = {
            'SMA': {'period': 20},
            'EMA': {'period': 12},
            'RSI': {'period': 14},
            'BB': {'period': 20, 'deviation': 2},
            'MACD': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        }

    # Prepare OHLCV data for talipp
    try:
        ohlcv_data = {
            'open': df['open'].tolist(),
            'high': df['high'].tolist(),
            'low': df['low'].tolist(),
            'close': df['close'].tolist(),
        }
        if 'volume' in df:
            ohlcv_data['volume'] = df['volume'].tolist()

        ohlcv = OHLCVFactory.from_dict(ohlcv_data)
    except Exception as e:
        logger.error(f"Error preparing OHLCV data: {e}")
        return df

    # Calculate each indicator
    for indicator_name, params in indicators.items():
        try:
            if indicator_name == 'SMA':
                indicator = SMA(period=params.get('period', 20), input_values=df['close'].tolist())
                df[f'SMA_{params.get("period", 20)}'] = list(indicator)

            elif indicator_name == 'EMA':
                indicator = EMA(period=params.get('period', 12), input_values=df['close'].tolist())
                df[f'EMA_{params.get("period", 12)}'] = list(indicator)

            elif indicator_name == 'RSI':
                indicator = RSI(period=params.get('period', 14), input_values=df['close'].tolist())
                df['RSI'] = list(indicator)

            elif indicator_name == 'BB':
                indicator = BB(
                    period=params.get('period', 20),
                    std_dev_mult=params.get('deviation', 2),
                    input_values=df['close'].tolist()
                )
                bb_values = list(indicator)
                # Extract upper, middle, and lower bands
                df['BB_upper'] = [val.ub if val else None for val in bb_values]
                df['BB_middle'] = [val.cb if val else None for val in bb_values]
                df['BB_lower'] = [val.lb if val else None for val in bb_values]

            elif indicator_name == 'MACD':
                indicator = MACD(
                    fast_period=params.get('fast_period', 12),
                    slow_period=params.get('slow_period', 26),
                    signal_period=params.get('signal_period', 9),
                    input_values=df['close'].tolist()
                )
                macd_values = list(indicator)
                df['MACD'] = [val.value if val else None for val in macd_values]
                df['MACD_signal'] = [val.signal if val else None for val in macd_values]
                df['MACD_histogram'] = [val.histogram if val else None for val in macd_values]

            elif indicator_name == 'ATR':
                indicator = ATR(period=params.get('period', 14), input_values=ohlcv)
                df['ATR'] = list(indicator)

            elif indicator_name == 'ADX':
                indicator = ADX(
                    di_period=params.get('di_period', params.get('period', 14)),
                    adx_period=params.get('adx_period', params.get('period', 14)),
                    input_values=ohlcv
                )
                adx_values = list(indicator)
                df['ADX'] = [val.adx if val else None for val in adx_values]
                df['ADX_plus_di'] = [val.plus_di if val else None for val in adx_values]
                df['ADX_minus_di'] = [val.minus_di if val else None for val in adx_values]

            elif indicator_name == 'Stoch':
                indicator = Stoch(
                    period=params.get('period', 14),
                    smoothing_period=params.get('smoothing_period', 3),
                    input_values=ohlcv
                )
                stoch_values = list(indicator)
                df['Stoch_K'] = [val.k if val else None for val in stoch_values]
                df['Stoch_D'] = [val.d if val else None for val in stoch_values]

            elif indicator_name == 'CCI':
                indicator = CCI(period=params.get('period', 20), input_values=ohlcv)
                df['CCI'] = list(indicator)

            elif indicator_name == 'ROC':
                indicator = ROC(period=params.get('period', 12), input_values=df['close'].tolist())
                df['ROC'] = list(indicator)

            elif indicator_name == 'OBV':
                if 'volume' in df:
                    indicator = OBV(input_values=ohlcv)
                    df['OBV'] = list(indicator)

            elif indicator_name == 'VWAP':
                if 'volume' in df:
                    indicator = VWAP(input_values=ohlcv)
                    df['VWAP'] = list(indicator)

        except Exception as e:
            logger.error(f"Error calculating {indicator_name}: {e}")
            continue

    return df

def analyze_indicator_signals(df: pd.DataFrame) -> dict[str, str]:
    """
    Analyze technical indicators and generate trading signals.

    Args:
        df: DataFrame with calculated indicators

    Returns:
        Dictionary of signals and recommendations
    """
    signals = {}

    # RSI Analysis
    if 'RSI' in df.columns:
        last_rsi = df['RSI'].iloc[-1]
        if pd.notna(last_rsi):
            if last_rsi > 70:
                signals['RSI'] = 'Overbought - Consider selling'
            elif last_rsi < 30:
                signals['RSI'] = 'Oversold - Consider buying'
            else:
                signals['RSI'] = f'Neutral ({last_rsi:.1f})'

    # MACD Analysis
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        macd = df['MACD'].iloc[-1]
        signal = df['MACD_signal'].iloc[-1]
        if pd.notna(macd) and pd.notna(signal):
            if macd > signal:
                signals['MACD'] = 'Bullish crossover'
            else:
                signals['MACD'] = 'Bearish crossover'

    # Bollinger Bands Analysis
    if all(col in df.columns for col in ['BB_upper', 'BB_lower', 'close']):
        close = df['close'].iloc[-1]
        upper = df['BB_upper'].iloc[-1]
        lower = df['BB_lower'].iloc[-1]
        if pd.notna(upper) and pd.notna(lower):
            if close > upper:
                signals['BB'] = 'Price above upper band - Overbought'
            elif close < lower:
                signals['BB'] = 'Price below lower band - Oversold'
            else:
                signals['BB'] = 'Price within bands - Normal'

    # Moving Average Analysis
    if 'SMA_20' in df.columns and 'EMA_12' in df.columns:
        sma = df['SMA_20'].iloc[-1]
        ema = df['EMA_12'].iloc[-1]
        close = df['close'].iloc[-1]
        if pd.notna(sma) and pd.notna(ema):
            if close > sma and close > ema:
                signals['MA'] = 'Price above moving averages - Bullish'
            elif close < sma and close < ema:
                signals['MA'] = 'Price below moving averages - Bearish'
            else:
                signals['MA'] = 'Mixed signals from moving averages'

    # ADX Trend Strength
    if 'ADX' in df.columns:
        adx = df['ADX'].iloc[-1]
        if pd.notna(adx):
            if adx > 25:
                signals['ADX'] = f'Strong trend ({adx:.1f})'
            else:
                signals['ADX'] = f'Weak trend ({adx:.1f})'

    # Stochastic Oscillator
    if 'Stoch_K' in df.columns and 'Stoch_D' in df.columns:
        k = df['Stoch_K'].iloc[-1]
        d = df['Stoch_D'].iloc[-1]
        if pd.notna(k) and pd.notna(d):
            if k > 80:
                signals['Stoch'] = 'Overbought zone'
            elif k < 20:
                signals['Stoch'] = 'Oversold zone'
            else:
                signals['Stoch'] = f'Neutral ({k:.1f})'

    return signals

def generate_trading_recommendation(signals: dict[str, str]) -> str:
    """
    Generate overall trading recommendation based on indicator signals.

    Args:
        signals: Dictionary of indicator signals

    Returns:
        Trading recommendation string
    """
    bullish_count = 0
    bearish_count = 0

    # Count bullish and bearish signals
    for signal in signals.values():
        signal_lower = signal.lower()
        if any(word in signal_lower for word in ['bullish', 'buy', 'oversold']):
            bullish_count += 1
        elif any(word in signal_lower for word in ['bearish', 'sell', 'overbought']):
            bearish_count += 1

    # Generate recommendation
    if bullish_count > bearish_count + 1:
        return "🟢 BULLISH - Multiple indicators suggest upward momentum"
    elif bearish_count > bullish_count + 1:
        return "🔴 BEARISH - Multiple indicators suggest downward pressure"
    else:
        return "🟡 NEUTRAL - Mixed signals, wait for clearer direction"

if __name__ == '__main__':
    # Example usage
    from data_fetcher import fetch_universal_data

    symbol = 'BTC-USD'
    data = fetch_universal_data(symbol, '1d', limit=500)

    if data is not None:
        # Calculate indicators
        data = calculate_comprehensive_indicators(data)

        # Analyze signals
        signals = analyze_indicator_signals(data)
        print(f"\nTechnical Analysis for {symbol}:")
        for indicator, signal in signals.items():
            print(f"{indicator}: {signal}")

        # Generate recommendation
        recommendation = generate_trading_recommendation(signals)
        print(f"\n{recommendation}")

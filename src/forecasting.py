import pandas as pd
from prophet import Prophet
from statsmodels.tsa.api import ARIMA, ExponentialSmoothing
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# Mapping timeframes to pandas frequency strings
TIMEFRAME_TO_FREQ = {
    '1m': '1T',     # 1 minute
    '5m': '5T',     # 5 minutes
    '15m': '15T',   # 15 minutes
    '30m': '30T',   # 30 minutes
    '1h': '1H',     # 1 hour
    '4h': '4H',     # 4 hours
    '1d': '1D',     # 1 day
    '1w': '1W',     # 1 week
}

def infer_frequency_from_data(df: pd.DataFrame) -> str:
    """
    Infer the frequency of the time series data.

    Args:
        df (pd.DataFrame): DataFrame with 'timestamp' column

    Returns:
        str: Pandas frequency string
    """
    if len(df) < 2:
        return '1D'  # Default fallback

    # Calculate the most common time difference
    time_diffs = df['timestamp'].diff().dropna()
    most_common_diff = time_diffs.mode().iloc[0] if not time_diffs.empty else pd.Timedelta(days=1)

    # Ensure most_common_diff is a pd.Timedelta for comparison
    try:
        if not isinstance(most_common_diff, pd.Timedelta):
            most_common_diff = pd.to_timedelta(str(most_common_diff))
    except Exception:
        most_common_diff = pd.Timedelta(days=1)

    # Map common differences to frequency strings
    if most_common_diff <= pd.Timedelta(minutes=1):
        return '1T'
    elif most_common_diff <= pd.Timedelta(minutes=5):
        return '5T'
    elif most_common_diff <= pd.Timedelta(minutes=15):
        return '15T'
    elif most_common_diff <= pd.Timedelta(minutes=30):
        return '30T'
    elif most_common_diff <= pd.Timedelta(hours=1):
        return '1H'
    elif most_common_diff <= pd.Timedelta(hours=4):
        return '4H'
    elif most_common_diff <= pd.Timedelta(days=1):
        return '1D'
    else:
        return '1W'

def generate_forecast(df: pd.DataFrame, model_name: str, periods: int = 30) -> pd.DataFrame | None:
    """
    Generates a forecast using the selected model with dynamic frequency support.

    Args:
        df (pd.DataFrame): The input DataFrame with historical data.
        model_name (str): The name of the model to use ('Prophet', 'ARIMA', 'ETS').
        periods (int): The number of future periods to forecast.

    Returns:
        pd.DataFrame: A DataFrame containing the forecast data.
    """
    if df is None or df.empty:
        logger.error("Input DataFrame is None or empty")
        return None

    try:
        # Infer frequency from data or use timeframe if available
        if 'timeframe' in df.columns and not df['timeframe'].empty:
            timeframe = df['timeframe'].iloc[0]
            freq = TIMEFRAME_TO_FREQ.get(timeframe, infer_frequency_from_data(df))
        else:
            freq = infer_frequency_from_data(df)

        logger.info(f"Using frequency: {freq} for {model_name} model")

        # Prepare data
        data = df.set_index('timestamp')['close']

        if model_name == 'Prophet':
            return _forecast_prophet(df, periods, freq)
        elif model_name == 'ARIMA':
            return _forecast_arima(data, periods, freq)
        elif model_name == 'ETS':
            return _forecast_ets(data, periods, freq)
        else:
            raise ValueError("Invalid model name. Choose from 'Prophet', 'ARIMA', 'ETS'.")

    except Exception as e:
        logger.error(f"Error in forecast generation for {model_name}: {e}")
        return None

def _forecast_prophet(df: pd.DataFrame, periods: int, freq: str) -> pd.DataFrame:
    """Generate forecast using Prophet model."""
    # Prepare Prophet DataFrame
    prophet_df = df.rename(columns={'timestamp': 'ds', 'close': 'y'})[['ds', 'y']]

    # Initialize Prophet model with appropriate parameters
    model = Prophet(
        changepoint_prior_scale=0.05,  # Flexibility of trend changes
        seasonality_prior_scale=10.0,   # Flexibility of seasonality
    )

    # Add seasonality based on frequency
    if freq in ['1T', '5T', '15T', '30T']:  # Minute-level data
        model.add_seasonality(name='hourly', period=1/24, fourier_order=5)
        model.add_seasonality(name='daily', period=1, fourier_order=10)
    elif freq in ['1H', '4H']:  # Hourly data
        model.add_seasonality(name='daily', period=1, fourier_order=10)
        model.add_seasonality(name='weekly', period=7, fourier_order=5)
    else:  # Daily or weekly data
        model.add_seasonality(name='weekly', period=7, fourier_order=5)
        model.add_seasonality(name='yearly', period=365.25, fourier_order=10)

    model.fit(prophet_df)

    # Create future dataframe with correct frequency
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def _forecast_arima(data: pd.Series, periods: int, freq: str) -> pd.DataFrame:
    """Generate forecast using ARIMA model with auto-order selection."""
    from statsmodels.tsa.stattools import adfuller
    
    # Test stationarity and determine integration order
    adf_result = adfuller(data)
    integration_order = 1 if adf_result[1] > 0.05 else 0
    
    # Auto-select ARIMA order using information criteria
    best_aic = float('inf')
    best_order = (1, integration_order, 1)
    
    # Test different ARIMA configurations
    for p in range(0, 4):
        for q in range(0, 4):
            try:
                temp_model = ARIMA(data, order=(p, integration_order, q))
                temp_fitted = temp_model.fit()
                if temp_fitted.aic < best_aic:
                    best_aic = temp_fitted.aic
                    best_order = (p, integration_order, q)
            except Exception:
                continue
    
    # Fit final model with best order
    model = ARIMA(data, order=best_order)
    fitted_model = model.fit()
    
    logger.info(f"ARIMA order selected: {best_order}, AIC: {fitted_model.aic:.2f}")
    
    # Generate forecast
    forecast_result = fitted_model.get_forecast(steps=periods)
    forecast_df = forecast_result.summary_frame()
    
    # Generate future dates with proper frequency handling
    last_date = data.index.max()
    
    # Convert frequency string to proper timedelta
    freq_map = {
        '1T': pd.Timedelta(minutes=1),
        '5T': pd.Timedelta(minutes=5),
        '15T': pd.Timedelta(minutes=15),
        '30T': pd.Timedelta(minutes=30),
        '1H': pd.Timedelta(hours=1),
        '4H': pd.Timedelta(hours=4),
        '1D': pd.Timedelta(days=1),
        '1W': pd.Timedelta(weeks=1)
    }
    
    time_delta = freq_map.get(freq, pd.Timedelta(days=1))
    future_dates = pd.date_range(
        start=last_date + time_delta,
        periods=periods,
        freq=freq
    )
    
    forecast_df = forecast_df.reset_index(drop=True)
    forecast_df['ds'] = future_dates
    forecast_df.rename(columns={
        'mean': 'yhat',
        'mean_ci_lower': 'yhat_lower',
        'mean_ci_upper': 'yhat_upper'
    }, inplace=True)
    
    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def _forecast_ets(data: pd.Series, periods: int, freq: str) -> pd.DataFrame:
    """Generate forecast using ETS (Exponential Smoothing) model."""
    seasonal_periods = _get_seasonal_periods(freq)

    try:
        # Configure seasonality based on data frequency and length
        if len(data) >= seasonal_periods * 2 and seasonal_periods > 1:
            model = ExponentialSmoothing(
                data,
                seasonal='add',
                seasonal_periods=seasonal_periods,
                trend='add',
                damped_trend=True
            )
        else:
            # No seasonality if insufficient data
            model = ExponentialSmoothing(
                data,
                trend='add',
                damped_trend=True
            )

        fitted_model = model.fit()
        yhat = fitted_model.forecast(steps=periods)

    except Exception as e:
        logger.warning(f"ETS model failed, using simple exponential smoothing: {e}")
        # Fallback to simple exponential smoothing
        model = ExponentialSmoothing(data)
        fitted_model = model.fit()
        yhat = fitted_model.forecast(steps=periods)

    # Generate future dates with correct frequency
    last_date = data.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(freq), periods=periods, freq=freq)

    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': yhat,
        'yhat_lower': None,  # ETS doesn't provide confidence intervals by default
        'yhat_upper': None
    })

    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def _get_seasonal_periods(freq: str) -> int:
    """
    Get seasonal periods based on frequency.

    Args:
        freq (str): Pandas frequency string

    Returns:
        int: Number of periods in a seasonal cycle
    """
    if freq in ['1T', '5T']:      # Minutes - hourly seasonality
        return 60 // int(freq[:-1]) if freq[:-1].isdigit() else 12
    elif freq in ['15T', '30T']:  # Minutes - daily seasonality
        return 96 if freq == '15T' else 48
    elif freq == '1H':            # Hours - daily seasonality
        return 24
    elif freq == '4H':            # 4-hour - weekly seasonality
        return 42  # 7 days * 6 periods per day
    elif freq == '1D':            # Days - weekly seasonality
        return 7
    elif freq == '1W':            # Weeks - yearly seasonality
        return 52
    else:
        return 1  # No clear seasonality

if __name__ == '__main__':
    from .data_fetcher import fetch_crypto_data

    symbol = 'BTC/USDT'
    btc_data = fetch_crypto_data(symbol, timeframe='1d', limit=365)

    if btc_data is not None:
        print(f"--- Testing Prophet for {symbol} ---")
        prophet_forecast = generate_forecast(btc_data.copy(), 'Prophet', 90)
        if prophet_forecast is not None:
            print(prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        print(f"\n--- Testing ARIMA for {symbol} ---")
        arima_forecast = generate_forecast(btc_data.copy(), 'ARIMA', 90)
        if arima_forecast is not None:
            print(arima_forecast.tail())

        print(f"\n--- Testing ETS for {symbol} ---")
        ets_forecast = generate_forecast(btc_data.copy(), 'ETS', 90)
        if ets_forecast is not None:
            print(ets_forecast.tail())
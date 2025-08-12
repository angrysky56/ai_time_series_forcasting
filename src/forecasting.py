import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.api import ARIMA, ExponentialSmoothing
import logging
from typing import cast

# Neural network imports
from darts import TimeSeries
from darts.models import (
    RNNModel, NBEATSModel, TransformerModel,
    TCNModel, TFTModel, NLinearModel
)
from darts.dataprocessing.transformers import Scaler

# Suppress Prophet warnings
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

def _to_naive_utc(series: pd.Series) -> pd.Series:
    """Return a timezone-naive datetime series in UTC.

    - Parses input to pandas datetime
    - If tz-aware, converts to UTC then drops tz info
    - If naive, leaves as naive
    """
    s = pd.to_datetime(series, errors='coerce')
    if hasattr(s.dt, 'tz') and s.dt.tz is not None:
        # Convert any tz to UTC, then drop tz info
        return s.dt.tz_convert('UTC').dt.tz_localize(None)
    return s

def _ts_to_pandas_dataframe(ts: TimeSeries) -> pd.DataFrame:
    """Return a pandas DataFrame (time as first column) from a Darts TimeSeries.

    Uses whatever conversion API is available across Darts versions.
    """
    for attr in ('pd_dataframe', 'pd_series', 'to_pandas'):
        if hasattr(ts, attr):
            conv = getattr(ts, attr)
            try:
                obj = conv()
            except TypeError:
                # In unlikely case method needs different signature, skip
                continue
            if isinstance(obj, pd.DataFrame):
                return obj.reset_index()
            if isinstance(obj, pd.Series):
                return obj.to_frame().reset_index()
    raise AttributeError("Cannot convert TimeSeries to pandas DataFrame: no known conversion method available.")

def _timeseries_to_df(ts: TimeSeries) -> pd.DataFrame:
    """Convert a Darts TimeSeries to the unified forecast DataFrame schema.

    Returns columns: ds, yhat, yhat_lower, yhat_upper
    Raises on failure so caller can surface meaningful error.
    """
    df = _ts_to_pandas_dataframe(ts)
    # Expect at least two columns: time index + value column
    if len(df.columns) < 2:
        raise ValueError("Unexpected TimeSeries DataFrame shape from Darts")
    time_col = df.columns[0]
    value_col = df.columns[1]
    df.rename(columns={time_col: 'ds', value_col: 'yhat'}, inplace=True)
    df['yhat_lower'] = None
    df['yhat_upper'] = None
    return df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
# Enhanced mapping for timeframes to pandas frequency strings
TIMEFRAME_TO_FREQ = {
    '1m': '1min',     # 1 minute
    '5m': '5min',     # 5 minutes
    '15m': '15min',   # 15 minutes
    '30m': '30min',   # 30 minutes
    '1h': '1h',       # 1 hour (updated from deprecated 'H')
    '4h': '4h',       # 4 hours (updated from deprecated 'H')
    '1d': '1D',       # 1 day
    '1w': '1W',       # 1 week
    '1M': '1M',       # 1 month
}

def create_darts_timeseries(df: pd.DataFrame, freq: str | None = None) -> 'TimeSeries | None':
    """
    Convert pandas DataFrame to Darts TimeSeries with proper frequency.

    Args:
        df: DataFrame with 'timestamp' and 'close' columns
        freq: Frequency string

    Returns:
        TimeSeries object or None if conversion fails
    """

    try:
        # Ensure proper datetime index
        data = df.copy()
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.set_index('timestamp').sort_index()

        # Infer frequency if not provided
        if freq is None:
            # Ensure index is a DatetimeIndex before inferring frequency
            if not isinstance(data.index, pd.DatetimeIndex):
                datetime_index = pd.to_datetime(data.index)
            else:
                datetime_index = data.index
            freq = pd.infer_freq(datetime_index)
            if freq is None:
                # Fall back to most common difference
                time_diffs = data.index.to_series().diff().dropna()
                if len(time_diffs) > 0:
                    most_common_diff = time_diffs.mode().iloc[0]
                    # Ensure most_common_diff is a pd.Timedelta
                    if not isinstance(most_common_diff, pd.Timedelta):
                        try:
                            most_common_diff = pd.to_timedelta(str(most_common_diff)) if not isinstance(most_common_diff, pd.Timedelta) else most_common_diff
                        except Exception:
                            most_common_diff = pd.Timedelta(0)
                    if most_common_diff <= pd.Timedelta(minutes=5):
                        freq = '5min'
                    elif most_common_diff <= pd.Timedelta(hours=1):
                        freq = '1H'
                    elif most_common_diff <= pd.Timedelta(days=1):
                        freq = '1D'
                    else:
                        freq = '1W'

        # Create TimeSeries with explicit frequency
        ts = TimeSeries.from_dataframe(
            data[['close']],
            value_cols=['close'],
            freq=freq
        )
        return ts

    except Exception as e:
        logger.warning(f"Failed to create Darts TimeSeries: {e}")
        return None

def _forecast_rnn(df: pd.DataFrame, periods: int, freq: str, model_type: str = 'LSTM') -> pd.DataFrame:
    """Generate forecast using RNN-based models (LSTM, GRU, RNN)."""

    ts = create_darts_timeseries(df, freq)
    if ts is None:
        raise ValueError("Failed to create TimeSeries object")

    # Scale the data
    scaler = Scaler()
    ts_scaled = scaler.fit_transform(ts)

    # Configure model parameters based on data frequency
    input_length = min(30, len(ts_scaled) // 3)
    output_length = min(periods, 12)

    model = RNNModel(
        model=model_type,
        input_chunk_length=input_length,
        output_chunk_length=output_length,
        hidden_dim=64,
        n_rnn_layers=2,
        dropout=0.1,
        batch_size=16,
        n_epochs=100,
        optimizer_kwargs={"lr": 1e-3},
        random_state=42,
        force_reset=True,
        save_checkpoints=False
    )

    # Fit and predict
    model.fit(ts_scaled)
    forecast_scaled = model.predict(n=periods)
    forecast_ts = scaler.inverse_transform(forecast_scaled)

    # Convert back to DataFrame format
    return _timeseries_to_df(cast(TimeSeries, forecast_ts))

def _forecast_nbeats(df: pd.DataFrame, periods: int, freq: str) -> pd.DataFrame:
    """Generate forecast using N-BEATS model."""

    ts = create_darts_timeseries(df, freq)
    if ts is None:
        raise ValueError("Failed to create TimeSeries object")

    scaler = Scaler()
    ts_scaled = scaler.fit_transform(ts)

    input_length = min(30, len(ts_scaled) // 3)
    output_length = min(periods, 12)

    model = NBEATSModel(
        input_chunk_length=input_length,
        output_chunk_length=output_length,
        num_stacks=30,
        num_blocks=1,
        num_layers=4,
        layer_widths=512,
        n_epochs=100,
        batch_size=16,
        optimizer_kwargs={"lr": 1e-3},
        random_state=42,
        force_reset=True,
        save_checkpoints=False
    )

    model.fit(ts_scaled)
    forecast_scaled = model.predict(n=periods)
    forecast_ts = scaler.inverse_transform(forecast_scaled)

    forecast_df = _ts_to_pandas_dataframe(cast(TimeSeries, forecast_ts))
    forecast_df.rename(columns={forecast_df.columns[0]: 'ds', forecast_df.columns[1]: 'yhat'}, inplace=True)
    forecast_df['yhat_lower'] = None
    forecast_df['yhat_upper'] = None

    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def _forecast_transformer(df: pd.DataFrame, periods: int, freq: str) -> pd.DataFrame:
    """Generate forecast using Transformer model."""

    ts = create_darts_timeseries(df, freq)
    if ts is None:
        raise ValueError("Failed to create TimeSeries object")

    scaler = Scaler()
    ts_scaled = scaler.fit_transform(ts)

    model = TransformerModel(
        input_chunk_length=min(24, len(ts_scaled) // 3),
        output_chunk_length=min(periods, 12),
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        activation='relu',
        batch_size=16,
        n_epochs=100,
        optimizer_kwargs={"lr": 1e-3},
        random_state=42,
        force_reset=True,
        save_checkpoints=False
    )

    model.fit(ts_scaled)
    forecast_scaled = model.predict(n=periods)
    forecast_ts = scaler.inverse_transform(forecast_scaled)

    forecast_df = _ts_to_pandas_dataframe(cast(TimeSeries, forecast_ts))
    forecast_df.columns = ['ds', 'yhat']
    forecast_df['yhat_lower'] = None
    forecast_df['yhat_lower'] = None
    forecast_df['yhat_upper'] = None

    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def _forecast_tcn(df: pd.DataFrame, periods: int, freq: str) -> pd.DataFrame:

    ts = create_darts_timeseries(df, freq)
    if ts is None:
        raise ValueError("Failed to create TimeSeries object")

    scaler = Scaler()
    ts_scaled = scaler.fit_transform(ts)

    model = TCNModel(
        input_chunk_length=min(30, len(ts_scaled) // 3),
        output_chunk_length=min(periods, 12),
        kernel_size=3,
        num_filters=32,
        num_layers=3,
        dilation_base=2,
        dropout=0.1,
        batch_size=16,
        n_epochs=100,
        optimizer_kwargs={"lr": 1e-3},
        random_state=42,
        force_reset=True,
        save_checkpoints=False
    )

    model.fit(ts_scaled)
    forecast_scaled = model.predict(n=periods)
    forecast_ts = scaler.inverse_transform(forecast_scaled)

    forecast_df = _ts_to_pandas_dataframe(cast(TimeSeries, forecast_ts))
    forecast_df.columns = ['ds', 'yhat']
    forecast_df['yhat_lower'] = None
    forecast_df['yhat_upper'] = None
    forecast_df['yhat_lower'] = None
    forecast_df['yhat_upper'] = None

    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def _forecast_tft(df: pd.DataFrame, periods: int, freq: str) -> pd.DataFrame:
    ts = create_darts_timeseries(df, freq)
    if ts is None:
        raise ValueError("Failed to create TimeSeries object")

    scaler = Scaler()
    ts_scaled = scaler.fit_transform(ts)

    model = TFTModel(
        input_chunk_length=min(24, len(ts_scaled) // 3),
        output_chunk_length=min(periods, 12),
        hidden_size=64,
        lstm_layers=2,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=16,
        n_epochs=100,
        optimizer_kwargs={"lr": 1e-3},
        random_state=42,
        force_reset=True,
        save_checkpoints=False
    )

    model.fit(ts_scaled)
    forecast_scaled = model.predict(n=periods)
    forecast_ts = scaler.inverse_transform(forecast_scaled)

    forecast_df = _ts_to_pandas_dataframe(cast(TimeSeries, forecast_ts))
    forecast_df.columns = ['ds', 'yhat']
    forecast_df['yhat_lower'] = None
    forecast_df['yhat_upper'] = None

    forecast_df['yhat_lower'] = None
    forecast_df['yhat_upper'] = None

    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def _forecast_nlinear(df: pd.DataFrame, periods: int, freq: str) -> pd.DataFrame:
    ts = create_darts_timeseries(df, freq)
    if ts is None:
        raise ValueError("Failed to create TimeSeries object")

    scaler = Scaler()
    ts_scaled = scaler.fit_transform(ts)

    model = NLinearModel(
        input_chunk_length=min(30, len(ts_scaled) // 3),
        output_chunk_length=min(periods, 12),
        batch_size=16,
        n_epochs=100,
        optimizer_kwargs={"lr": 1e-3},
        random_state=42,
        force_reset=True,
        save_checkpoints=False
    )

    model.fit(ts_scaled)
    forecast_scaled = model.predict(n=periods)
    forecast_ts = scaler.inverse_transform(forecast_scaled)

    forecast_df = _ts_to_pandas_dataframe(cast(TimeSeries, forecast_ts))
    forecast_df.columns = ['ds', 'yhat']
    forecast_df['yhat_lower'] = None
    forecast_df['yhat_upper'] = None

    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def generate_forecast(df: pd.DataFrame, model_name: str, periods: int = 30, freq: str = '1d') -> pd.DataFrame | None:
    """Generate a forecast for a given time series.

    Args:
        df (pd.DataFrame): The input DataFrame with historical data.
        model_name (str): The name of the model to use.
        periods (int): The number of future periods to forecast.
        freq (str): The frequency of the data (e.g., '1d', '1h', '4h'). Defaults to '1d'.

    Returns:
        pd.DataFrame: A DataFrame containing the forecast data.
    """

    try:
        # Use passed frequency first, fall back to inference if needed
        if freq and freq in TIMEFRAME_TO_FREQ:
            # Use the explicitly passed frequency
            pandas_freq = TIMEFRAME_TO_FREQ[freq]
        elif 'timeframe' in df.columns and not df['timeframe'].empty:
            # Fall back to timeframe column if available
            timeframe = df['timeframe'].iloc[0]
            freq = timeframe
            pandas_freq = TIMEFRAME_TO_FREQ.get(timeframe, infer_frequency_from_data(df))
        else:
            # Last resort: infer from data
            freq = infer_frequency_from_data(df)
            pandas_freq = freq

        logger.info(f"Using frequency: {freq} (pandas: {pandas_freq}) for {model_name} model")

        # Route to appropriate model
        if model_name == 'Prophet':
            # Prophet expects a valid pandas frequency string
            return _forecast_prophet(df, periods, pandas_freq)
        elif model_name == 'ARIMA':
            data = df.set_index('timestamp')['close']
            # Ensure timezone-naive index
            if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
                data.index = data.index.tz_convert('UTC').tz_localize(None)
            # Don't use asfreq() as it introduces NaN values - pass data directly
            return _forecast_arima(data, periods, freq)
        elif model_name == 'ETS':
            data = df.set_index('timestamp')['close']
            # Ensure timezone-naive index
            if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
                data.index = data.index.tz_convert('UTC').tz_localize(None)
            # Don't use asfreq() as it introduces NaN values - pass data directly
            return _forecast_ets(data, periods, freq)
        elif model_name in ['LSTM', 'RNN', 'GRU']:
            return _forecast_rnn(df, periods, freq, model_type=model_name)
        elif model_name == 'NBEATS':
            return _forecast_nbeats(df, periods, freq)
        elif model_name == 'Transformer':
            return _forecast_transformer(df, periods, freq)
        elif model_name == 'TCN':
            return _forecast_tcn(df, periods, freq)
        elif model_name == 'TFT':
            return _forecast_tft(df, periods, freq)
        elif model_name == 'NLinear':
            return _forecast_nlinear(df, periods, freq)
        else:
            raise ValueError(f"Invalid model name: {model_name}. Choose from supported models.")

    except Exception as e:
        logger.error(f"Error in forecast generation for {model_name}: {e}")
        return None

def _forecast_prophet(df: pd.DataFrame, periods: int, freq: str) -> pd.DataFrame:
    """Generate forecast using Prophet model."""
    # Prepare Prophet DataFrame
    prophet_df = df.rename(columns={'timestamp': 'ds', 'close': 'y'})[['ds', 'y']]
    # Prophet does not support timezone-aware datetimes
    prophet_df['ds'] = _to_naive_utc(prophet_df['ds'])

    # Initialize Prophet model with appropriate parameters
    model = Prophet(
        changepoint_prior_scale=0.05,  # Flexibility of trend changes
        seasonality_prior_scale=10.0,   # Flexibility of seasonality
    )

    # Add seasonality based on frequency
    if freq in ['1T', '5T', '15T', '30T', '1min', '5min', '15min', '30min']:  # Minute-level data
        model.add_seasonality(name='hourly', period=1/24, fourier_order=5)
        model.add_seasonality(name='daily', period=1, fourier_order=10)
    elif freq in ['1H', '4H', '1h', '4h']:  # Hourly data
        model.add_seasonality(name='daily', period=1, fourier_order=10)
        model.add_seasonality(name='weekly', period=7, fourier_order=5)
    else:  # Daily or weekly data
        model.add_seasonality(name='weekly', period=7, fourier_order=5)
        model.add_seasonality(name='yearly', period=365.25, fourier_order=10)

    model.fit(prophet_df)

    # Create future dataframe with correct frequency  
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    
    # Return only future forecasts, not historical + future combined
    # Get the last historical timestamp to filter future predictions
    last_historical_ts = prophet_df['ds'].max()
    future_forecast = forecast[forecast['ds'] > last_historical_ts]
    
    # If no future data (edge case), return the last prediction
    if future_forecast.empty:
        future_forecast = forecast.tail(periods) if len(forecast) >= periods else forecast.tail(1)

    return future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def _forecast_arima(data: pd.Series, periods: int, freq: str) -> pd.DataFrame:
    """Generate forecast using ARIMA model with auto-order selection."""
    from statsmodels.tsa.stattools import adfuller
    
    # Data validation and cleaning
    if data.empty:
        raise ValueError("Empty data series provided to ARIMA model")
    
    # Remove any infinite values and NaN values
    original_length = len(data)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    
    if data.empty:
        raise ValueError("No valid data points after removing inf/NaN values")
    
    if len(data) < 10:
        raise ValueError(f"Insufficient data for ARIMA modeling: {len(data)} points (minimum 10 required)")
    
    if len(data) < original_length:
        logger.warning(f"Removed {original_length - len(data)} invalid data points for ARIMA modeling")

    # Test stationarity and determine integration order
    try:
        adf_result = adfuller(data)
        integration_order = 1 if adf_result[1] > 0.05 else 0
    except Exception as e:
        logger.warning(f"Stationarity test failed: {e}, using integration order 1")
        integration_order = 1

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

def infer_frequency_from_data(df: pd.DataFrame) -> str:
    """
    Infers the frequency string from the DataFrame's 'timestamp' column.

    Args:
        df (pd.DataFrame): DataFrame with a 'timestamp' column.

    Returns:
        str: Inferred frequency string.
    """
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column to infer frequency.")
    datetime_index = pd.to_datetime(df['timestamp'])
    freq = pd.infer_freq(datetime_index)
    if freq is not None:
        return freq
    # Fallback: use mode of time differences
    time_diffs = datetime_index.diff().dropna()
    if len(time_diffs) > 0:
        most_common_diff = time_diffs.mode().iloc[0]
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
        elif most_common_diff <= pd.Timedelta(weeks=1):
            return '1W'
    return '1D'

if __name__ == '__main__':
    from .data_fetcher import fetch_universal_data

    symbol = 'BTC-USD'
    btc_data = fetch_universal_data(symbol, timeframe='1d', limit=365)

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
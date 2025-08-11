import pandas as pd
from prophet import Prophet
from statsmodels.tsa.api import ARIMA, ExponentialSmoothing

def generate_forecast(df: pd.DataFrame, model_name: str, periods: int = 30):
    """
    Generates a forecast using the selected model.

    Args:
        df (pd.DataFrame): The input DataFrame with historical data.
        model_name (str): The name of the model to use ('Prophet', 'ARIMA', 'ETS').
        periods (int): The number of future periods to forecast.

    Returns:
        pd.DataFrame: A DataFrame containing the forecast data.
    """
    if df is None or df.empty:
        return None


    # Prepare data
    data = df.set_index('timestamp')['close']

    if model_name == 'Prophet':
        prophet_df = df.rename(columns={'timestamp': 'ds', 'close': 'y'})[['ds', 'y']]
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast

    elif model_name == 'ARIMA':
        model = ARIMA(data, order=(5, 1, 0))
        fitted_model = model.fit()
        forecast_result = fitted_model.get_forecast(steps=periods)
        forecast_df = forecast_result.summary_frame()
        # Generate future dates for forecast periods
        last_date = df['timestamp'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
        forecast_df = forecast_df.reset_index(drop=True)
        forecast_df['ds'] = future_dates
        forecast_df.rename(columns={'mean': 'yhat', 'mean_ci_lower': 'yhat_lower', 'mean_ci_upper': 'yhat_upper'}, inplace=True)
        return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    elif model_name == 'ETS':
        model = ExponentialSmoothing(data, seasonal='add', seasonal_periods=7, trend='add', damped_trend=True)
        fitted_model = model.fit()
        yhat = fitted_model.forecast(steps=periods)
        # Generate future dates for forecast periods
        last_date = df['timestamp'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
        forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': yhat})
        forecast_df['yhat_lower'] = None
        forecast_df['yhat_upper'] = None
        return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    else:
        raise ValueError("Invalid model name. Choose from 'Prophet', 'ARIMA', 'ETS'.")

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
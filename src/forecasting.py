import pandas as pd
from prophet import Prophet

def generate_forecast(df: pd.DataFrame, periods: int = 30):
    """
    Generates a forecast using the Prophet model.

    Args:
        df (pd.DataFrame): The input DataFrame with historical data.
        periods (int): The number of future periods to forecast.

    Returns:
        pd.DataFrame: A DataFrame containing the forecast data.
    """
    if df is None or df.empty:
        return None

    # Prophet requires columns 'ds' (timestamp) and 'y' (value)
    prophet_df = df.rename(columns={'timestamp': 'ds', 'close': 'y'})[['ds', 'y']]

    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return forecast

if __name__ == '__main__':
    # Example usage:
    from data_fetcher import fetch_crypto_data

    btc_symbol = 'BTC/USDT'
    btc_data = fetch_crypto_data(btc_symbol, timeframe='1d', limit=365)

    if btc_data is not None:
        print(f"Successfully fetched data for {btc_symbol}. Generating forecast...")
        forecast_data = generate_forecast(btc_data, periods=90)
        if forecast_data is not None:
            print("Forecast generated successfully.")
            # Print the last few rows of the forecast including the predicted values
            print(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

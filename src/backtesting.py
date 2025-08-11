import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from src.forecasting import generate_forecast

def perform_backtest(df: pd.DataFrame, model_name: str, periods: int):
    """
    Performs a backtest for a given model.

    Args:
        df (pd.DataFrame): The historical data.
        model_name (str): The name of the model to backtest.
        periods (int): The number of periods to forecast and evaluate.

    Returns:
        dict: A dictionary containing the backtest metrics (MAE, RMSE).
    """
    if df is None or len(df) <= periods:
        return {"error": "Not enough data to perform backtest."}

    # Split data into training and testing sets
    train_df = df.iloc[:-periods]
    test_df = df.iloc[-periods:]

    # Generate a forecast on the training data
    forecast_df = generate_forecast(train_df, model_name, periods)

    if forecast_df is None or forecast_df.empty:
        return {"error": f"Forecast generation failed for {model_name} during backtest."}

    # Align forecast with the actual test data
    # The forecast 'ds' should match the test data 'timestamp'
    y_true = np.array(test_df['close'].values)
    y_pred = np.array(forecast_df['yhat'].iloc[-periods:])

    # Ensure the lengths match
    if len(y_true) != len(y_pred):
        return {"error": "Mismatch between test data and forecast length."}

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {"MAE": mae, "RMSE": rmse}

if __name__ == '__main__':
    from .data_fetcher import fetch_crypto_data

    symbol = 'BTC/USDT'
    btc_data = fetch_crypto_data(symbol, timeframe='1d', limit=365)

    if btc_data is not None:
        models_to_test = ['Prophet', 'ARIMA', 'ETS']
        for model in models_to_test:
            print(f"--- Backtesting {model} for {symbol} ---")
            metrics = perform_backtest(btc_data.copy(), model, periods=90)
            print(metrics)

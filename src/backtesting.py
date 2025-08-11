import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from src.forecasting import generate_forecast
import logging
from typing import Any
logger = logging.getLogger(__name__)

def perform_backtest(df: pd.DataFrame, model_name: str, periods: int) -> dict[str, Any]:
    """
    Performs a simple backtest for a given model (legacy method).

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
    y_true = np.array(test_df['close'].values)
    y_pred = np.array(forecast_df['yhat'].iloc[-periods:])

    # Ensure the lengths match
    if len(y_true) != len(y_pred):
        return {"error": "Mismatch between test data and forecast length."}

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def walk_forward_validation(
    df: pd.DataFrame,
    model_name: str,
    forecast_periods: int,
    min_train_size: int | None = None,
    step_size: int = 1,
    window_type: str = 'expanding',
    max_train_size: int | None = None
) -> dict[str, Any]:
    """
    Performs walk-forward validation (time series cross-validation).

    This is the gold standard for time series model evaluation.

    Args:
        df (pd.DataFrame): The historical data with 'timestamp' and 'close' columns.
        model_name (str): The name of the model to validate.
        forecast_periods (int): Number of periods to forecast in each iteration.
        min_train_size (int): Minimum training set size. Defaults to 10 * forecast_periods.
        step_size (int): Number of periods to step forward each iteration.
        window_type (str): 'expanding' (growing train set) or 'rolling' (fixed train set).
        max_train_size (int): Maximum training set size for 'rolling' window.

    Returns:
        dict: Comprehensive validation results with metrics and fold details.
    """
    if df is None or len(df) < forecast_periods * 3:
        return {"error": "Insufficient data for walk-forward validation."}

    if min_train_size is None:
        min_train_size = max(50, forecast_periods * 10)  # Ensure sufficient training data

    if window_type == 'rolling' and max_train_size is None:
        max_train_size = min_train_size * 2

    # Ensure we have enough data
    if len(df) < min_train_size + forecast_periods:
        return {"error": f"Need at least {min_train_size + forecast_periods} data points."}

    folds = []
    fold_metrics = []
    all_predictions = []
    all_actuals = []

    # Start from min_train_size
    current_start = 0
    current_train_end = min_train_size

    fold_num = 1

    while current_train_end + forecast_periods <= len(df):
        try:
            # Define training window
            if window_type == 'expanding':
                train_start = 0
                train_end = current_train_end
            else:  # rolling
                effective_max_train_size = max_train_size if max_train_size is not None else min_train_size * 2
                train_start = max(0, current_train_end - effective_max_train_size)
                train_end = current_train_end

            # Define test window
            test_start = current_train_end
            test_end = min(current_train_end + forecast_periods, len(df))

            # Extract train and test data
            train_data = df.iloc[train_start:train_end].copy()
            test_data = df.iloc[test_start:test_end].copy()

            if len(test_data) < forecast_periods:
                break  # Not enough test data

            # Generate forecast
            forecast_df = generate_forecast(train_data, model_name, forecast_periods)

            if forecast_df is None or forecast_df.empty:
                logger.warning(f"Fold {fold_num}: Forecast generation failed")
                current_train_end += step_size
                fold_num += 1
                continue

            # Calculate metrics for this fold
            y_true = np.asarray(test_data['close'].values)
            y_pred = np.asarray(forecast_df['yhat'].iloc[:len(y_true)].values)

            if len(y_true) != len(y_pred):
                logger.warning(f"Fold {fold_num}: Length mismatch - true: {len(y_true)}, pred: {len(y_pred)}")
                current_train_end += step_size
                fold_num += 1
                continue

            # Calculate fold metrics
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            fold_result = {
                'fold': fold_num,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_size': train_end - train_start,
                'test_size': test_end - test_start,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'predictions': y_pred.tolist(),
                'actuals': y_true.tolist(),
                'test_dates': test_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
            }

            folds.append(fold_result)
            fold_metrics.append({'MAE': mae, 'RMSE': rmse, 'MAPE': mape})
            all_predictions.extend(y_pred)
            all_actuals.extend(y_true)

            logger.info(f"Fold {fold_num} completed - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

        except Exception as e:
            logger.error(f"Error in fold {fold_num}: {e}")

        # Move to next fold
        current_train_end += step_size
        fold_num += 1

    if not fold_metrics:
        return {"error": "No successful validation folds completed."}

    # Calculate overall metrics
    overall_mae = np.mean([m['MAE'] for m in fold_metrics])
    overall_rmse = np.mean([m['RMSE'] for m in fold_metrics])
    overall_mape = np.mean([m['MAPE'] for m in fold_metrics])

    # Calculate standard deviations
    std_mae = np.std([m['MAE'] for m in fold_metrics])
    std_rmse = np.std([m['RMSE'] for m in fold_metrics])
    std_mape = np.std([m['MAPE'] for m in fold_metrics])

    # Calculate combined metrics on all predictions
    if all_predictions and all_actuals:
        combined_mae = mean_absolute_error(all_actuals, all_predictions)
        combined_rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
        combined_mape = np.mean(np.abs((np.array(all_actuals) - np.array(all_predictions)) / np.array(all_actuals))) * 100
    else:
        combined_mae = combined_rmse = combined_mape = None

    return {
        'model': model_name,
        'validation_type': 'walk_forward',
        'window_type': window_type,
        'num_folds': len(folds),
        'forecast_periods': forecast_periods,
        'step_size': step_size,
        'metrics': {
            'mean_MAE': overall_mae,
            'mean_RMSE': overall_rmse,
            'mean_MAPE': overall_mape,
            'std_MAE': std_mae,
            'std_RMSE': std_rmse,
            'std_MAPE': std_mape,
            'combined_MAE': combined_mae,
            'combined_RMSE': combined_rmse,
            'combined_MAPE': combined_mape
        },
        'folds': folds,
        'fold_metrics': fold_metrics
    }

def compare_models_walkforward(
    df: pd.DataFrame,
    model_names: list[str],
    forecast_periods: int = 30,
    min_train_size: int | None = None,
    step_size: int = 1
) -> dict[str, Any]:
    """
    Compare multiple models using walk-forward validation.

    Args:
        df (pd.DataFrame): Historical data
        model_names (list[str]): List of model names to compare
        forecast_periods (int): Number of periods to forecast
        min_train_size (int): Minimum training set size
        step_size (int): Step size for walk-forward

    Returns:
        dict: Comparison results with best model recommendation
    """
    results = {}

    for model_name in model_names:
        logger.info(f"Running walk-forward validation for {model_name}")
        result = walk_forward_validation(
            df, model_name, forecast_periods, min_train_size, step_size
        )
        results[model_name] = result

    # Find best model based on combined RMSE
    best_model = None
    best_rmse = float('inf')

    valid_results = {k: v for k, v in results.items() if 'error' not in v}

    if valid_results:
        for model_name, result in valid_results.items():
            combined_rmse = result['metrics'].get('combined_RMSE')
            if combined_rmse is not None and combined_rmse < best_rmse:
                best_rmse = combined_rmse
                best_model = model_name

    return {
        'comparison_results': results,
        'best_model': best_model,
        'best_rmse': best_rmse,
        'summary': {model: res.get('metrics', {}).get('combined_RMSE')
                   for model, res in valid_results.items()}
    }

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

"""
Ensemble forecasting system that combines multiple models for improved accuracy.
Builds on existing forecasting.py infrastructure with intelligent model combination.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List
import logging
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from pathlib import Path

from .forecasting import generate_forecast
from .backtesting import walk_forward_validation

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Track performance metrics for individual models."""
    model_name: str
    rmse: float
    mae: float
    mape: float
    weight: float
    last_updated: str
    fold_count: int = 0

class EnsembleForecaster:
    """
    Intelligent ensemble forecasting system that combines multiple models.
    
    Features:
    - Performance-weighted combination
    - Bayesian Model Averaging
    - Dynamic weight adjustment
    - Model confidence scoring
    """
    
    def __init__(
        self, 
        models: list[str] | None = None,
        performance_window: int = 50,
        performance_file: str | None = None
    ):
        """
        Initialize ensemble forecaster.
        
        Args:
            models: List of model names to include in ensemble
            performance_window: Number of recent forecasts to use for performance calculation
            performance_file: Path to store performance history
        """
        self.models = models or ['Prophet', 'ARIMA', 'ETS']
        self.performance_window = performance_window
        self.performance_file = performance_file or 'model_performance.json'
        self.model_performances: dict[str, ModelPerformance] = {}
        self.performance_history: dict[str, list[dict]] = {model: [] for model in self.models}
        
        # Load existing performance data
        self._load_performance_history()
    
    def _load_performance_history(self) -> None:
        """Load historical performance data from file."""
        try:
            if Path(self.performance_file).exists():
                with open(self.performance_file, 'r') as f:
                    data = json.load(f)
                    
                # Reconstruct performance objects
                for model_name, perf_data in data.get('model_performances', {}).items():
                    self.model_performances[model_name] = ModelPerformance(**perf_data)
                    
                # Load performance history
                self.performance_history = data.get('performance_history', 
                                                   {model: [] for model in self.models})
                
                logger.info(f"Loaded performance history for {len(self.model_performances)} models")
        except Exception as e:
            logger.warning(f"Could not load performance history: {e}")
            self._initialize_performance()
    
    def _save_performance_history(self) -> None:
        """Save performance data to file."""
        try:
            data = {
                'model_performances': {
                    name: {
                        'model_name': perf.model_name,
                        'rmse': perf.rmse,
                        'mae': perf.mae,
                        'mape': perf.mape,
                        'weight': perf.weight,
                        'last_updated': perf.last_updated,
                        'fold_count': perf.fold_count
                    }
                    for name, perf in self.model_performances.items()
                },
                'performance_history': self.performance_history
            }
            
            with open(self.performance_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Could not save performance history: {e}")
    
    def _initialize_performance(self) -> None:
        """Initialize performance tracking for all models."""
        for model in self.models:
            self.model_performances[model] = ModelPerformance(
                model_name=model,
                rmse=1.0,  # Default neutral performance
                mae=1.0,
                mape=10.0,
                weight=1.0 / len(self.models),  # Equal weights initially
                last_updated=pd.Timestamp.now().isoformat(),
                fold_count=0
            )
    
    def update_model_performance(
        self, 
        data: pd.DataFrame, 
        freq: str = '1d',
        validation_periods: int = 30
    ) -> dict[str, Any]:
        """
        Update model performance metrics using walk-forward validation.
        
        Args:
            data: Historical data for validation
            freq: Data frequency
            validation_periods: Number of periods for validation
            
        Returns:
            Updated performance metrics for all models
        """
        if len(data) < validation_periods * 3:
            logger.warning("Insufficient data for performance validation")
            return self.get_current_weights()
        
        logger.info("Updating model performance metrics...")
        
        for model_name in self.models:
            try:
                # Run walk-forward validation
                validation_result = walk_forward_validation(
                    data.copy(),
                    model_name,
                    forecast_periods=validation_periods,
                    min_train_size=min(100, len(data) // 3),
                    step_size=max(1, validation_periods // 10),
                    freq=freq
                )
                
                if 'error' not in validation_result:
                    metrics = validation_result['metrics']
                    
                    # Update performance tracking
                    self.model_performances[model_name] = ModelPerformance(
                        model_name=model_name,
                        rmse=metrics['combined_RMSE'] or metrics['mean_RMSE'],
                        mae=metrics['combined_MAE'] or metrics['mean_MAE'],
                        mape=metrics['combined_MAPE'] or metrics['mean_MAPE'],
                        weight=0.0,  # Will be calculated below
                        last_updated=pd.Timestamp.now().isoformat(),
                        fold_count=validation_result['num_folds']
                    )
                    
                    # Store in history
                    self.performance_history[model_name].append({
                        'timestamp': pd.Timestamp.now().isoformat(),
                        'rmse': self.model_performances[model_name].rmse,
                        'mae': self.model_performances[model_name].mae,
                        'mape': self.model_performances[model_name].mape,
                        'folds': validation_result['num_folds']
                    })
                    
                    # Keep only recent performance window
                    if len(self.performance_history[model_name]) > self.performance_window:
                        self.performance_history[model_name] = \
                            self.performance_history[model_name][-self.performance_window:]
                    
                    logger.info(f"{model_name} - RMSE: {self.model_performances[model_name].rmse:.6f}")
                
                else:
                    logger.warning(f"Validation failed for {model_name}: {validation_result['error']}")
                    
            except Exception as e:
                logger.error(f"Error updating performance for {model_name}: {e}")
        
        # Calculate dynamic weights
        self._calculate_weights()
        
        # Save updated performance
        self._save_performance_history()
        
        return self.get_current_weights()
    
    def _calculate_weights(self, method: str = 'inverse_rmse') -> None:
        """
        Calculate ensemble weights based on model performance.
        
        Args:
            method: Weighting method ('inverse_rmse', 'softmax', 'equal')
        """
        if not self.model_performances:
            self._initialize_performance()
            return
        
        if method == 'equal':
            # Equal weights
            weight = 1.0 / len(self.models)
            for model_name in self.models:
                if model_name in self.model_performances:
                    self.model_performances[model_name].weight = weight
        
        elif method == 'inverse_rmse':
            # Inverse RMSE weighting
            rmse_values = []
            valid_models = []
            
            for model_name in self.models:
                if model_name in self.model_performances:
                    rmse = self.model_performances[model_name].rmse
                    if rmse > 0:  # Avoid division by zero
                        rmse_values.append(rmse)
                        valid_models.append(model_name)
            
            if rmse_values:
                # Calculate inverse weights
                inverse_rmse = [1.0 / rmse for rmse in rmse_values]
                total_inverse = sum(inverse_rmse)
                
                # Normalize weights
                for i, model_name in enumerate(valid_models):
                    self.model_performances[model_name].weight = inverse_rmse[i] / total_inverse
            else:
                # Fallback to equal weights
                self._calculate_weights('equal')
        
        elif method == 'softmax':
            # Softmax on negative RMSE (lower RMSE gets higher weight)
            rmse_values = []
            valid_models = []
            
            for model_name in self.models:
                if model_name in self.model_performances:
                    rmse_values.append(-self.model_performances[model_name].rmse)
                    valid_models.append(model_name)
            
            if rmse_values:
                # Apply softmax
                exp_values = np.exp(rmse_values - max(rmse_values))  # Stability
                softmax_weights = exp_values / np.sum(exp_values)
                
                for i, model_name in enumerate(valid_models):
                    self.model_performances[model_name].weight = softmax_weights[i]
            else:
                self._calculate_weights('equal')
    
    def generate_ensemble_forecast(
        self,
        data: pd.DataFrame,
        periods: int = 30,
        freq: str = '1d',
        method: str = 'weighted_average',
        confidence_intervals: bool = True
    ) -> pd.DataFrame | None:
        """
        Generate ensemble forecast by combining multiple models.
        
        Args:
            data: Historical data
            periods: Number of periods to forecast
            freq: Data frequency
            method: Ensemble method ('weighted_average', 'median', 'bayesian')
            confidence_intervals: Whether to calculate ensemble confidence intervals
            
        Returns:
            Ensemble forecast DataFrame
        """
        logger.info(f"Generating ensemble forecast with {len(self.models)} models")
        
        # Generate individual forecasts
        forecasts = {}
        for model_name in self.models:
            try:
                forecast = generate_forecast(data.copy(), model_name, periods, freq)
                if forecast is not None and not forecast.empty:
                    forecasts[model_name] = forecast
                    logger.info(f"Generated {model_name} forecast: {len(forecast)} periods")
                else:
                    logger.warning(f"Failed to generate forecast for {model_name}")
            except Exception as e:
                logger.error(f"Error generating forecast for {model_name}: {e}")
        
        if not forecasts:
            logger.error("No successful forecasts generated")
            return None
        
        # Combine forecasts
        return self._combine_forecasts(forecasts, method, confidence_intervals)
    
    def _combine_forecasts(
        self,
        forecasts: dict[str, pd.DataFrame],
        method: str,
        confidence_intervals: bool
    ) -> pd.DataFrame:
        """Combine individual forecasts into ensemble prediction."""
        
        # Ensure all forecasts have the same timestamps
        common_timestamps = None
        for model_name, forecast in forecasts.items():
            if common_timestamps is None:
                common_timestamps = forecast['ds'].copy()
            else:
                # Use intersection of timestamps
                common_timestamps = pd.Series(
                    list(set(common_timestamps) & set(forecast['ds']))
                ).sort_values().reset_index(drop=True)
        
        if common_timestamps is None or len(common_timestamps) == 0:
            logger.error("No common timestamps found across forecasts")
            return None
        
        # Align all forecasts to common timestamps
        aligned_forecasts = {}
        for model_name, forecast in forecasts.items():
            aligned = forecast[forecast['ds'].isin(common_timestamps)].copy()
            aligned = aligned.sort_values('ds').reset_index(drop=True)
            aligned_forecasts[model_name] = aligned
        
        if method == 'weighted_average':
            return self._weighted_average_combination(aligned_forecasts, confidence_intervals)
        elif method == 'median':
            return self._median_combination(aligned_forecasts, confidence_intervals)
        elif method == 'bayesian':
            return self._bayesian_combination(aligned_forecasts, confidence_intervals)
        else:
            logger.warning(f"Unknown ensemble method: {method}, using weighted_average")
            return self._weighted_average_combination(aligned_forecasts, confidence_intervals)
    
    def _weighted_average_combination(
        self, 
        forecasts: dict[str, pd.DataFrame], 
        confidence_intervals: bool
    ) -> pd.DataFrame:
        """Combine forecasts using weighted average."""
        
        # Use first forecast as template
        first_forecast = list(forecasts.values())[0]
        ensemble_forecast = first_forecast[['ds']].copy()
        
        # Weighted average of predictions
        weighted_predictions = np.zeros(len(ensemble_forecast))
        total_weight = 0.0
        
        for model_name, forecast in forecasts.items():
            weight = self.model_performances.get(model_name, ModelPerformance(
                model_name, 1.0, 1.0, 10.0, 1.0/len(forecasts), "", 0
            )).weight
            
            weighted_predictions += weight * forecast['yhat'].values
            total_weight += weight
        
        # Normalize if needed
        if total_weight > 0:
            ensemble_forecast['yhat'] = weighted_predictions / total_weight
        else:
            ensemble_forecast['yhat'] = weighted_predictions / len(forecasts)
        
        # Calculate ensemble confidence intervals
        if confidence_intervals:
            lower_bounds = []
            upper_bounds = []
            
            for i in range(len(ensemble_forecast)):
                predictions_at_t = []
                weights_at_t = []
                
                for model_name, forecast in forecasts.items():
                    if i < len(forecast):
                        predictions_at_t.append(forecast['yhat'].iloc[i])
                        weight = self.model_performances.get(model_name, ModelPerformance(
                            model_name, 1.0, 1.0, 10.0, 1.0/len(forecasts), "", 0
                        )).weight
                        weights_at_t.append(weight)
                
                if predictions_at_t:
                    # Calculate weighted standard deviation
                    weighted_mean = np.average(predictions_at_t, weights=weights_at_t)
                    variance = np.average((predictions_at_t - weighted_mean)**2, weights=weights_at_t)
                    std_dev = np.sqrt(variance)
                    
                    # 95% confidence interval
                    lower_bounds.append(weighted_mean - 1.96 * std_dev)
                    upper_bounds.append(weighted_mean + 1.96 * std_dev)
                else:
                    lower_bounds.append(ensemble_forecast['yhat'].iloc[i])
                    upper_bounds.append(ensemble_forecast['yhat'].iloc[i])
            
            ensemble_forecast['yhat_lower'] = lower_bounds
            ensemble_forecast['yhat_upper'] = upper_bounds
        else:
            ensemble_forecast['yhat_lower'] = None
            ensemble_forecast['yhat_upper'] = None
        
        return ensemble_forecast
    
    def _median_combination(
        self, 
        forecasts: dict[str, pd.DataFrame], 
        confidence_intervals: bool
    ) -> pd.DataFrame:
        """Combine forecasts using median."""
        
        first_forecast = list(forecasts.values())[0]
        ensemble_forecast = first_forecast[['ds']].copy()
        
        # Calculate median predictions
        predictions_matrix = np.array([
            forecast['yhat'].values for forecast in forecasts.values()
        ])
        
        ensemble_forecast['yhat'] = np.median(predictions_matrix, axis=0)
        
        if confidence_intervals:
            # Use percentiles for confidence intervals
            ensemble_forecast['yhat_lower'] = np.percentile(predictions_matrix, 2.5, axis=0)
            ensemble_forecast['yhat_upper'] = np.percentile(predictions_matrix, 97.5, axis=0)
        else:
            ensemble_forecast['yhat_lower'] = None
            ensemble_forecast['yhat_upper'] = None
        
        return ensemble_forecast
    
    def _bayesian_combination(
        self, 
        forecasts: dict[str, pd.DataFrame], 
        confidence_intervals: bool
    ) -> pd.DataFrame:
        """Combine forecasts using Bayesian Model Averaging."""
        
        # For now, implement as uncertainty-weighted combination
        # TODO: Implement full Bayesian approach with prior distributions
        
        first_forecast = list(forecasts.values())[0]
        ensemble_forecast = first_forecast[['ds']].copy()
        
        # Weight by inverse of historical RMSE (Bayesian-inspired)
        predictions = np.zeros(len(ensemble_forecast))
        uncertainty_weights = []
        
        for model_name, forecast in forecasts.items():
            performance = self.model_performances.get(model_name)
            if performance:
                # Use inverse of RMSE as uncertainty weight
                weight = 1.0 / (performance.rmse + 1e-8)  # Avoid division by zero
                uncertainty_weights.append(weight)
                predictions += weight * forecast['yhat'].values
            else:
                uncertainty_weights.append(1.0)
                predictions += forecast['yhat'].values
        
        # Normalize
        total_weight = sum(uncertainty_weights)
        ensemble_forecast['yhat'] = predictions / total_weight
        
        if confidence_intervals:
            # Calculate Bayesian confidence intervals
            variance = np.zeros(len(ensemble_forecast))
            
            for i, (model_name, forecast) in enumerate(forecasts.items()):
                weight = uncertainty_weights[i] / total_weight
                diff = forecast['yhat'].values - ensemble_forecast['yhat'].values
                variance += weight * (diff ** 2)
            
            std_dev = np.sqrt(variance)
            ensemble_forecast['yhat_lower'] = ensemble_forecast['yhat'] - 1.96 * std_dev
            ensemble_forecast['yhat_upper'] = ensemble_forecast['yhat'] + 1.96 * std_dev
        else:
            ensemble_forecast['yhat_lower'] = None
            ensemble_forecast['yhat_upper'] = None
        
        return ensemble_forecast
    
    def get_current_weights(self) -> dict[str, float]:
        """Get current ensemble weights for all models."""
        return {
            model_name: perf.weight 
            for model_name, perf in self.model_performances.items()
        }
    
    def get_performance_summary(self) -> dict[str, Any]:
        """Get detailed performance summary for all models."""
        summary = {}
        for model_name, perf in self.model_performances.items():
            summary[model_name] = {
                'rmse': perf.rmse,
                'mae': perf.mae,
                'mape': perf.mape,
                'weight': perf.weight,
                'last_updated': perf.last_updated,
                'fold_count': perf.fold_count,
                'history_length': len(self.performance_history.get(model_name, []))
            }
        return summary
    
    def recommend_best_ensemble_method(
        self, 
        data: pd.DataFrame, 
        test_periods: int = 20
    ) -> str:
        """
        Recommend best ensemble method based on backtest performance.
        
        Args:
            data: Historical data for testing
            test_periods: Number of periods to use for method comparison
            
        Returns:
            Recommended ensemble method name
        """
        if len(data) < test_periods * 3:
            return 'weighted_average'  # Default fallback
        
        methods = ['weighted_average', 'median', 'bayesian']
        method_scores = {}
        
        # Split data for testing
        train_data = data.iloc[:-test_periods].copy()
        test_data = data.iloc[-test_periods:].copy()
        
        for method in methods:
            try:
                # Generate ensemble forecast
                forecast = self.generate_ensemble_forecast(
                    train_data, test_periods, method=method
                )
                
                if forecast is not None and len(forecast) >= test_periods:
                    # Calculate error
                    y_true = test_data['close'].values
                    y_pred = forecast['yhat'].iloc[:len(y_true)].values
                    
                    if len(y_true) == len(y_pred):
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        method_scores[method] = rmse
                        logger.info(f"Method {method} RMSE: {rmse:.6f}")
                
            except Exception as e:
                logger.warning(f"Error testing ensemble method {method}: {e}")
        
        if method_scores:
            best_method = min(method_scores, key=method_scores.get)
            logger.info(f"Recommended ensemble method: {best_method}")
            return best_method
        else:
            logger.warning("Could not evaluate ensemble methods, using default")
            return 'weighted_average'

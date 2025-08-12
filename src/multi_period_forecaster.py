"""
Multi-period forecasting system for coherent time-based analysis.
Provides integrated forecasts across monthly, weekly, daily, and hourly timeframes.
"""

import pandas as pd
import numpy as np
from typing import Any
import logging
from src.forecasting import generate_forecast
from src.data_fetcher import fetch_universal_data
from src.ensemble_forecaster import EnsembleForecaster

logger = logging.getLogger(__name__)

class MultiPeriodForecaster:
    """Generates coherent forecasts across multiple time periods."""

    def __init__(self):
        # Direct timeframe mapping to yfinance intervals with proper periods
        self.timeframes = {
            'monthly': {'interval': '1mo', 'periods': 12, 'limit': 60},      # 5 years of monthly data for better analysis
            'weekly': {'interval': '1wk', 'periods': 26, 'limit': 104},      # 2 years of weekly data
            'daily': {'interval': '1d', 'periods': 30, 'limit': 365},        # 1 year of daily data
            'hourly': {'interval': '1h', 'periods': 168, 'limit': 720}       # 30 days of hourly data (168 = 1 week forecast)
        }

        self.forecasts = {}
        self.consistency_scores = {}

    def fetch_multi_period_data(
        self,
        symbol: str,
        source: str = 'auto',
        exchange: str | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch data for all time periods with proper aggregation.

        Args:
            symbol: Trading symbol
            source: 'crypto', 'stock', or 'auto'
            exchange: Compatibility parameter (not used with yfinance)

        Returns:
            Dictionary of timeframe -> DataFrame
        """
        data = {}

        # Convert timeframe keys to match yfinance intervals
        yf_interval_map = {
            'monthly': '1mo',
            'weekly': '1wk',
            'daily': '1d',
            'hourly': '1h'
        }

        for period_name, config in self.timeframes.items():
            try:
                # Use the proper yfinance interval for each timeframe
                yf_interval = yf_interval_map[period_name]

                # Fetch data using the correct interval
                df = fetch_universal_data(symbol, yf_interval, config['limit'])

                if df is not None and not df.empty:
                    # Add period-specific metadata
                    df['period'] = period_name

                    # Calculate proper period statistics with correct open/close logic
                    df = df.sort_values('timestamp').reset_index(drop=True)

                    # For all periods, calculate period-over-period changes
                    if len(df) > 1:
                        # For multi-period analysis: period_open should be the earliest open price
                        # and period_close should be the latest close price across the entire dataset
                        earliest_open = df['open'].iloc[0]  # First/earliest opening price
                        latest_close = df['close'].iloc[-1]  # Last/latest closing price

                        # Set period_open and period_close for the entire range analysis
                        df['period_open'] = earliest_open  # Same for all rows - represents dataset start
                        df['period_close'] = latest_close  # Same for all rows - represents dataset end

                        # Calculate total period change (from earliest open to latest close)
                        total_period_change = ((latest_close - earliest_open) / earliest_open * 100)
                        df['period_change'] = round(total_period_change, 2)  # Same for all rows

                        # Keep individual period-to-period changes (this logic was correct)
                        df['period_to_period_change'] = df['close'].pct_change() * 100
                        df['period_to_period_change'] = df['period_to_period_change'].round(2)

                        # Add high/low information for better context
                        df['period_high'] = df['high']
                        df['period_low'] = df['low']
                        df['period_range'] = ((df['high'] - df['low']) / df['low'] * 100).round(2)

                    # Add volume information if available
                    if 'volume' in df.columns:
                        df['volume'] = df['volume']

                    data[period_name] = df
                    logger.info(f"Fetched {len(df)} {period_name} records for {symbol}")

                    # Log sample data for debugging with proper period info
                    if len(df) > 3:
                        latest_period = df.iloc[-1]
                        logger.info(f"{period_name} latest period - Open: {latest_period['period_open']:.2f}, "
                                  f"Close: {latest_period['period_close']:.2f}, "
                                  f"Change: {latest_period['period_change']:.2f}%")
                else:
                    logger.warning(f"Failed to fetch {period_name} data for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching {period_name} data: {e}")

        return data

    def generate_multi_period_forecasts(
        self,
        data: dict[str, pd.DataFrame],
        model_name: str = 'Prophet',
        use_ensemble: bool = False,
        ensemble_models: list[str] | None = None,
        ensemble_method: str = 'weighted_average'
    ) -> dict[str, pd.DataFrame]:
        """
        Generate forecasts for all time periods using single model or ensemble.

        Args:
            data: Dictionary of timeframe -> DataFrame
            model_name: Single model to use (when use_ensemble=False)
            use_ensemble: Whether to use ensemble forecasting
            ensemble_models: List of models for ensemble (defaults to ['Prophet', 'ARIMA', 'ETS'])
            ensemble_method: Ensemble combination method ('weighted_average', 'median', 'bayesian')

        Returns:
            Dictionary of timeframe -> forecast DataFrame
        """
        forecasts = {}

        # Initialize ensemble forecaster if requested
        ensemble_forecaster = None
        if use_ensemble:
            ensemble_models = ensemble_models or ['Prophet', 'ARIMA', 'ETS']
            ensemble_forecaster = EnsembleForecaster(
                models=ensemble_models,
                performance_file=f'performance_{hash(str(sorted(ensemble_models)))}.json'
            )
            logger.info(f"Using ensemble forecasting with models: {ensemble_models}")

        for period_name, df in data.items():
            try:
                config = self.timeframes[period_name]

                if use_ensemble:
                    # Update ensemble performance for this timeframe
                    if ensemble_forecaster is not None and len(df) > config['periods'] * 3:  # Sufficient data for validation
                        logger.info(f"Updating ensemble performance for {period_name} timeframe")
                        ensemble_forecaster.update_model_performance(df.copy(), config['interval'])

                    # Generate ensemble forecast only if ensemble_forecaster is not None
                    if ensemble_forecaster is not None:
                        forecast = ensemble_forecaster.generate_ensemble_forecast(
                            df.copy(),
                            config['periods'],
                            config['interval'],
                            method=ensemble_method
                        )

                        if forecast is not None:
                            # Add ensemble metadata
                            forecast['ensemble_method'] = ensemble_method
                            forecast['ensemble_models'] = str(ensemble_models)
                            weights = ensemble_forecaster.get_current_weights()
                            forecast['model_weights'] = str(weights)

                            forecasts[period_name] = forecast
                            logger.info(f"Generated ensemble {period_name} forecast with weights: {weights}")
                        else:
                            logger.warning(f"Ensemble forecast failed for {period_name}, falling back to single model")
                            # Fallback to single model
                            forecast = generate_forecast(df.copy(), model_name, config['periods'], config['interval'])
                            if forecast is not None:
                                forecasts[period_name] = forecast
                    else:
                        logger.warning(f"Ensemble forecaster is None for {period_name}, skipping ensemble forecast.")
                        # Fallback to single model
                        forecast = generate_forecast(df.copy(), model_name, config['periods'], config['interval'])
                        if forecast is not None:
                            forecasts[period_name] = forecast
                else:
                    # Single model forecasting (existing logic)
                    forecast = generate_forecast(df.copy(), model_name, config['periods'], config['interval'])

                    if forecast is not None:
                        forecasts[period_name] = forecast
                        logger.info(f"Generated {period_name} forecast with {model_name} ({len(forecast)} points)")
                    else:
                        logger.warning(f"Failed to generate {model_name} forecast for {period_name}")

            except Exception as e:
                logger.error(f"Error generating {period_name} forecast: {e}")

                # If ensemble fails, try fallback to single model
                if use_ensemble:
                    try:
                        logger.info(f"Attempting fallback to {model_name} for {period_name}")
                        config = self.timeframes[period_name]
                        forecast = generate_forecast(df.copy(), model_name, config['periods'], config['interval'])
                        if forecast is not None:
                            forecasts[period_name] = forecast
                            logger.info(f"Fallback successful for {period_name}")
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed for {period_name}: {fallback_error}")

        self.forecasts = forecasts

        # Store ensemble information for consistency calculations
        if use_ensemble and ensemble_forecaster:
            self.ensemble_info = {
                'models': ensemble_models,
                'method': ensemble_method,
                'performance': ensemble_forecaster.get_performance_summary()
            }

        return forecasts

    def calculate_consistency_scores(self) -> dict[str, float]:
        """
        Calculate consistency scores between different timeframe forecasts.

        Returns:
            Dictionary of consistency scores
        """
        scores = {}

        # Check if we have at least 2 forecasts to compare
        if len(self.forecasts) < 2:
            return scores

        # Compare adjacent timeframes
        comparisons = [
            ('monthly', 'weekly'),
            ('weekly', 'daily'),
            ('daily', 'hourly')
        ]

        for longer_tf, shorter_tf in comparisons:
            if longer_tf in self.forecasts and shorter_tf in self.forecasts:
                score = self._calculate_trend_consistency(
                    self.forecasts[longer_tf],
                    self.forecasts[shorter_tf]
                )
                scores[f"{longer_tf}_vs_{shorter_tf}"] = score

        # Calculate overall consistency
        if scores:
            scores['overall'] = np.mean(list(scores.values()))

        self.consistency_scores = scores
        return scores

    def _calculate_trend_consistency(
        self,
        forecast1: pd.DataFrame,
        forecast2: pd.DataFrame
    ) -> float:
        """
        Calculate trend consistency between two forecasts.

        Args:
            forecast1: First forecast DataFrame
            forecast2: Second forecast DataFrame

        Returns:
            Consistency score between 0 and 1
        """
        try:
            # Calculate trend direction for each forecast
            trend1 = np.sign(forecast1['yhat'].iloc[-1] - forecast1['yhat'].iloc[0])
            trend2 = np.sign(forecast2['yhat'].iloc[-1] - forecast2['yhat'].iloc[0])

            # Calculate percentage changes
            pct_change1 = (forecast1['yhat'].iloc[-1] - forecast1['yhat'].iloc[0]) / forecast1['yhat'].iloc[0]
            pct_change2 = (forecast2['yhat'].iloc[-1] - forecast2['yhat'].iloc[0]) / forecast2['yhat'].iloc[0]

            # Direction consistency (0.5 weight)
            direction_score = 1.0 if trend1 == trend2 else 0.0

            # Magnitude consistency (0.5 weight)
            magnitude_diff = abs(pct_change1 - pct_change2)
            magnitude_score = max(0, 1 - magnitude_diff * 10)  # Penalize large differences

            return 0.5 * direction_score + 0.5 * magnitude_score

        except Exception as e:
            logger.error(f"Error calculating trend consistency: {e}")
            return 0.5  # Neutral score on error

    def generate_coherent_report(self) -> dict[str, Any]:
        """
        Generate a coherent multi-period forecast report.

        Returns:
            Comprehensive forecast report
        """
        report = {
            'forecasts': self.forecasts,
            'consistency_scores': self.consistency_scores,
            'summary': self._generate_summary(),
            'recommendations': self._generate_recommendations()
        }

        return report

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary of multi-period forecasts with ensemble information."""
        summary = {
            'trend_direction': {},
            'volatility': {},
            'confidence': {},
            'ensemble_info': getattr(self, 'ensemble_info', None)
        }

        for period_name, forecast in self.forecasts.items():
            if forecast is not None and not forecast.empty:
                # Trend direction
                start_price = forecast['yhat'].iloc[0]
                end_price = forecast['yhat'].iloc[-1]
                change_pct = ((end_price - start_price) / start_price) * 100

                summary['trend_direction'][period_name] = {
                    'direction': 'up' if change_pct > 0 else 'down',
                    'change_percent': round(change_pct, 2),
                    'is_ensemble': 'ensemble_method' in forecast.columns
                }

                # Add ensemble-specific information if available
                if 'ensemble_method' in forecast.columns:
                    summary['trend_direction'][period_name]['ensemble_method'] = forecast['ensemble_method'].iloc[0]
                    summary['trend_direction'][period_name]['ensemble_models'] = forecast['ensemble_models'].iloc[0]

                # Volatility (if confidence intervals available)
                if 'yhat_upper' in forecast.columns and 'yhat_lower' in forecast.columns:
                    if forecast['yhat_upper'].notna().any() and forecast['yhat_lower'].notna().any():
                        avg_range = (forecast['yhat_upper'] - forecast['yhat_lower']).mean()
                        avg_price = forecast['yhat'].mean()
                        volatility = (avg_range / avg_price) * 100 if avg_price > 0 else 0
                        summary['volatility'][period_name] = round(volatility, 2)

                # Enhanced confidence calculation for ensemble
                base_consistency = self.consistency_scores.get('overall', 0.5)

                # Boost confidence if using ensemble with good model agreement
                if hasattr(self, 'ensemble_info') and self.ensemble_info:
                    performance_info = self.ensemble_info.get('performance', {})
                    if performance_info:
                        # Calculate ensemble confidence boost based on model agreement
                        model_weights = []
                        for model_perf in performance_info.values():
                            if isinstance(model_perf, dict) and 'weight' in model_perf:
                                model_weights.append(model_perf['weight'])

                        if model_weights:
                            # Higher weight concentration = higher confidence
                            weight_entropy = -sum(w * np.log(w + 1e-10) for w in model_weights if w > 0)
                            max_entropy = np.log(len(model_weights))
                            confidence_boost = 1 - (weight_entropy / max_entropy) if max_entropy > 0 else 0
                            base_consistency = min(1.0, base_consistency + confidence_boost * 0.2)

                summary['confidence'][period_name] = round(base_consistency * 100, 1)

        return summary

    def _generate_recommendations(self) -> list[str]:
        """Generate trading recommendations based on multi-period analysis with ensemble insights."""
        recommendations = []

        # Check overall trend consistency
        overall_consistency = self.consistency_scores.get('overall', 0)

        if overall_consistency > 0.7:
            recommendations.append("✅ Strong consistency across timeframes - higher confidence in predictions")
        elif overall_consistency < 0.3:
            recommendations.append("⚠️ Low consistency across timeframes - exercise caution")

        # Ensemble-specific recommendations
        if hasattr(self, 'ensemble_info') and self.ensemble_info:
            ensemble_performance = self.ensemble_info.get('performance', {})
            ensemble_method = self.ensemble_info.get('method', 'weighted_average')

            # Analyze ensemble model agreement
            if ensemble_performance:
                model_weights = []
                model_rmse = []
                for model_name, perf in ensemble_performance.items():
                    if isinstance(perf, dict):
                        model_weights.append(perf.get('weight', 0))
                        model_rmse.append(perf.get('rmse', 1.0))

                if model_weights:
                    # Check weight distribution
                    max_weight = max(model_weights)
                    min_weight = min(model_weights)
                    weight_ratio = max_weight / min_weight if min_weight > 0 else float('inf')

                    if weight_ratio > 3:
                        dominant_model = None
                        for model_name, perf in ensemble_performance.items():
                            if isinstance(perf, dict) and perf.get('weight', 0) == max_weight:
                                dominant_model = model_name
                                break
                        if dominant_model:
                            recommendations.append(f"📊 {dominant_model} model dominates ensemble - consider single model approach")
                    elif weight_ratio < 1.5:
                        recommendations.append("🎯 Balanced ensemble with good model diversity - high confidence in predictions")

                    # Performance-based recommendations
                    avg_rmse = np.mean(model_rmse) if model_rmse else 1.0
                    if avg_rmse < 0.01:  # Very low error
                        recommendations.append("🎯 Ensemble showing excellent accuracy - strong signal reliability")
                    elif avg_rmse > 0.1:  # High error
                        recommendations.append("⚠️ Ensemble accuracy concerns - consider model retraining or regime change")

                    # Method-specific insights
                    if ensemble_method == 'bayesian':
                        recommendations.append("🔬 Bayesian ensemble accounting for model uncertainty - robust predictions")
                    elif ensemble_method == 'median':
                        recommendations.append("🛡️ Median ensemble reducing outlier impact - stable predictions")

        # Check trend alignment (existing logic enhanced)
        trends = {}
        ensemble_count = 0
        for period_name in self.forecasts:
            summary = self._generate_summary()
            if period_name in summary['trend_direction']:
                trend_info = summary['trend_direction'][period_name]
                trends[period_name] = trend_info['direction']
                if trend_info.get('is_ensemble', False):
                    ensemble_count += 1

        # Enhanced trend analysis with ensemble consideration
        if all(t == 'up' for t in trends.values()):
            confidence_modifier = "high confidence" if ensemble_count > 0 else "moderate confidence"
            recommendations.append(f"📈 All timeframes show upward trend ({confidence_modifier}) - consider long positions")
        elif all(t == 'down' for t in trends.values()):
            confidence_modifier = "high confidence" if ensemble_count > 0 else "moderate confidence"
            recommendations.append(f"📉 All timeframes show downward trend ({confidence_modifier}) - consider short positions or waiting")
        else:
            recommendations.append("🔄 Mixed signals across timeframes - wait for clearer direction")

        # Check volatility with ensemble enhancement
        volatilities = self._generate_summary().get('volatility', {})
        if volatilities:
            avg_volatility = np.mean(list(volatilities.values()))
            volatility_modifier = "ensemble-validated" if ensemble_count > 0 else "single-model"

            if avg_volatility > 10:
                recommendations.append(f"⚡ High volatility ({avg_volatility:.1f}%, {volatility_modifier}) - adjust position sizes")
            elif avg_volatility < 3:
                recommendations.append(f"😴 Low volatility ({avg_volatility:.1f}%, {volatility_modifier}) - limited movement expected")

        # Ensemble model performance recommendations
        if hasattr(self, 'ensemble_info') and self.ensemble_info:
            models_used = self.ensemble_info.get('models', [])
            if len(models_used) >= 3:
                recommendations.append(f"🎪 Using {len(models_used)}-model ensemble for robustness")
            else:
                recommendations.append("🎯 Consider adding more models to ensemble for improved robustness")

        return recommendations

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
                        # Period open is the opening price of each period
                        df['period_open'] = df['open']
                        df['period_close'] = df['close']
                        
                        # Calculate period change (close vs open of same period)
                        df['period_change'] = ((df['close'] - df['open']) / df['open'] * 100).round(2)
                        
                        # Calculate period-to-period change (close vs previous close)
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
        model_name: str = 'Prophet'
    ) -> dict[str, pd.DataFrame]:
        """
        Generate forecasts for all time periods.
        
        Args:
            data: Dictionary of timeframe -> DataFrame
            model_name: Forecasting model to use
        
        Returns:
            Dictionary of timeframe -> forecast DataFrame
        """
        forecasts = {}
        
        for period_name, df in data.items():
            try:
                config = self.timeframes[period_name]
                # Pass the correct frequency interval from config
                forecast = generate_forecast(df.copy(), model_name, config['periods'], config['interval'])
                
                if forecast is not None:
                    forecasts[period_name] = forecast
                    logger.info(f"Generated {period_name} forecast with {len(forecast)} points")
                else:
                    logger.warning(f"Failed to generate forecast for {period_name}")
            except Exception as e:
                logger.error(f"Error generating {period_name} forecast: {e}")
        
        self.forecasts = forecasts
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
        """Generate summary of multi-period forecasts."""
        summary = {
            'trend_direction': {},
            'volatility': {},
            'confidence': {}
        }
        
        for period_name, forecast in self.forecasts.items():
            if forecast is not None and not forecast.empty:
                # Trend direction
                start_price = forecast['yhat'].iloc[0]
                end_price = forecast['yhat'].iloc[-1]
                change_pct = ((end_price - start_price) / start_price) * 100
                
                summary['trend_direction'][period_name] = {
                    'direction': 'up' if change_pct > 0 else 'down',
                    'change_percent': round(change_pct, 2)
                }
                
                # Volatility (if confidence intervals available)
                if 'yhat_upper' in forecast.columns and 'yhat_lower' in forecast.columns:
                    if forecast['yhat_upper'].notna().any() and forecast['yhat_lower'].notna().any():
                        avg_range = (forecast['yhat_upper'] - forecast['yhat_lower']).mean()
                        avg_price = forecast['yhat'].mean()
                        volatility = (avg_range / avg_price) * 100 if avg_price > 0 else 0
                        summary['volatility'][period_name] = round(volatility, 2)
                
                # Confidence based on consistency
                overall_consistency = self.consistency_scores.get('overall', 0.5)
                summary['confidence'][period_name] = round(overall_consistency * 100, 1)
        
        return summary
    
    def _generate_recommendations(self) -> list[str]:
        """Generate trading recommendations based on multi-period analysis."""
        recommendations = []
        
        # Check overall trend consistency
        overall_consistency = self.consistency_scores.get('overall', 0)
        
        if overall_consistency > 0.7:
            recommendations.append("✅ Strong consistency across timeframes - higher confidence in predictions")
        elif overall_consistency < 0.3:
            recommendations.append("⚠️ Low consistency across timeframes - exercise caution")
        
        # Check trend alignment
        trends = {}
        for period_name in self.forecasts:
            if period_name in self._generate_summary()['trend_direction']:
                trends[period_name] = self._generate_summary()['trend_direction'][period_name]['direction']
        
        if all(t == 'up' for t in trends.values()):
            recommendations.append("📈 All timeframes show upward trend - consider long positions")
        elif all(t == 'down' for t in trends.values()):
            recommendations.append("📉 All timeframes show downward trend - consider short positions or waiting")
        else:
            recommendations.append("🔄 Mixed signals across timeframes - wait for clearer direction")
        
        # Check volatility
        volatilities = self._generate_summary().get('volatility', {})
        if volatilities:
            avg_volatility = np.mean(list(volatilities.values()))
            if avg_volatility > 10:
                recommendations.append(f"⚡ High volatility ({avg_volatility:.1f}%) - adjust position sizes")
            elif avg_volatility < 3:
                recommendations.append(f"😴 Low volatility ({avg_volatility:.1f}%) - limited movement expected")
        
        return recommendations

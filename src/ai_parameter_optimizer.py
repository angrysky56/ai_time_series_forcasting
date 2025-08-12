"""
AI-driven parameter optimizer for technical indicator selection and configuration.
Automatically selects optimal indicators and parameters from talipp library.
"""

import pandas as pd
import numpy as np
from typing import Any
import logging
from talipp.indicators import (
    SMA, EMA, WMA, DEMA, TEMA, T3, KAMA, McGinleyDynamic,
    RSI, StochRSI,
    BB, KeltnerChannels, DonchianChannels,
    ATR, NATR, CHOP,
    OBV, VWAP, KVO,
    ADX, Aroon, VTX,
    CCI, TSI, UO,
    AO, ROC
)
from talipp.ohlcv import OHLCVFactory
import optuna

logger = logging.getLogger(__name__)

# Indicator categories and their parameters
INDICATOR_CATALOG = {
    'trend': {
        'SMA': {'params': {'period': (5, 200)}, 'class': SMA},
        'EMA': {'params': {'period': (5, 200)}, 'class': EMA},
        'WMA': {'params': {'period': (5, 200)}, 'class': WMA},
        'DEMA': {'params': {'period': (5, 100)}, 'class': DEMA},
        'TEMA': {'params': {'period': (5, 100)}, 'class': TEMA},
        'T3': {'params': {'period': (5, 100), 'volume_factor': (0.5, 0.9)}, 'class': T3},
        'KAMA': {'params': {'period': (5, 50), 'fast_period': (2, 10), 'slow_period': (20, 50)}, 'class': KAMA},
        'McGinleyDynamic': {'params': {'period': (10, 50), 'k': (0.4, 0.8)}, 'class': McGinleyDynamic}
    },
    'momentum': {
        'RSI': {'params': {'period': (7, 28)}, 'class': RSI},
        'StochRSI': {'params': {'period': (7, 21), 'smoothing_period': (3, 7)}, 'class': StochRSI},
        'CCI': {'params': {'period': (10, 40)}, 'class': CCI},
        'TSI': {'params': {'fast_period': (10, 20), 'slow_period': (20, 40)}, 'class': TSI},
        'UO': {'params': {'fast_period': (5, 10), 'medium_period': (10, 20), 'slow_period': (20, 40)}, 'class': UO},
        'AO': {'params': {'fast_period': (5, 15), 'slow_period': (20, 50)}, 'class': AO},
        'ROC': {'params': {'period': (5, 30)}, 'class': ROC}
    },
    'volatility': {
        'BB': {'params': {'period': (10, 30), 'deviation': (1.5, 3.0)}, 'class': BB},
        'KeltnerChannels': {'params': {'period': (10, 30), 'multiplier': (1.0, 3.0)}, 'class': KeltnerChannels},
        'DonchianChannels': {'params': {'period': (10, 50)}, 'class': DonchianChannels},
        'ATR': {'params': {'period': (7, 21)}, 'class': ATR},
        'NATR': {'params': {'period': (7, 21)}, 'class': NATR},
        'CHOP': {'params': {'period': (10, 30)}, 'class': CHOP}
    },
    'volume': {
        'OBV': {'params': {}, 'class': OBV},
        'VWAP': {'params': {'period': (10, 50)}, 'class': VWAP},
        'KVO': {'params': {'fast_period': (20, 40), 'slow_period': (40, 80)}, 'class': KVO}
    },
    'trend_strength': {
        'ADX': {'params': {'period': (7, 21)}, 'class': ADX},
        'Aroon': {'params': {'period': (10, 30)}, 'class': Aroon},
        'VTX': {'params': {'period': (10, 30)}, 'class': VTX}
    }
}

class AIParameterOptimizer:
    """AI-driven optimizer for selecting indicators and parameters."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize optimizer with historical data.

        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data.copy()
        # Ensure 'close' column is numeric to prevent type errors in calculations
        if 'close' in self.data.columns:
            self.data['close'] = pd.to_numeric(self.data['close'], errors='coerce')
            self.data.dropna(subset=['close'], inplace=True)

        self.ohlcv = self._prepare_ohlcv_data()
        self.selected_indicators = {}
        self.optimal_parameters = {}
        self.performance_scores = {}

    def _prepare_ohlcv_data(self):
        """Convert DataFrame to OHLCV format for talipp."""
        try:
            ohlcv_dict = {
                'open': self.data['open'].tolist(),
                'high': self.data['high'].tolist(),
                'low': self.data['low'].tolist(),
                'close': self.data['close'].tolist()
            }
            if 'volume' in self.data.columns:
                ohlcv_dict['volume'] = self.data['volume'].tolist()

            return OHLCVFactory.from_dict(ohlcv_dict)
        except Exception as e:
            logger.error(f"Error preparing OHLCV data: {e}")
            return None

    def auto_select_indicators(
        self,
        max_indicators: int = 5,
        categories: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Automatically select optimal indicators based on data characteristics.

        Args:
            max_indicators: Maximum number of indicators to select
            categories: Specific categories to select from (optional)

        Returns:
            Dictionary of selected indicators with optimal parameters
        """
        if categories is None:
            categories = list(INDICATOR_CATALOG.keys())

        selected = {}
        scores = {}

        # Analyze data characteristics
        characteristics = self._analyze_data_characteristics()

        # Select indicators based on characteristics
        for category in categories:
            if category not in INDICATOR_CATALOG:
                continue

            # Score indicators in category
            category_scores = self._score_indicators_for_category(
                category,
                characteristics
            )

            # Select best indicator from category
            if category_scores:
                best_indicator = max(category_scores, key=lambda k: category_scores[k])
                selected[best_indicator] = {
                    'category': category,
                    'score': category_scores[best_indicator]
                }
                scores[best_indicator] = category_scores[best_indicator]

        # Keep only top indicators
        top_indicators = dict(sorted(
            selected.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:max_indicators])

        # Optimize parameters for selected indicators
        for indicator_name in top_indicators:
            params = self._optimize_indicator_parameters(indicator_name)
            top_indicators[indicator_name]['parameters'] = params

        self.selected_indicators = top_indicators
        return top_indicators

    def _analyze_data_characteristics(self) -> dict[str, float]:
        """Analyze data characteristics to guide indicator selection."""
        characteristics = {}

        try:
            closes = self.data['close'].to_numpy()

            # Trend strength (0-1)
            trend_line = np.polyfit(range(len(closes)), closes, 1)[0]
            trend_strength = abs(trend_line) / np.std(closes) if np.std(closes) > 0 else 0
            characteristics['trend_strength'] = min(1.0, trend_strength)

            # Volatility (0-1)
            returns = pd.Series(closes).pct_change().dropna()
            volatility = returns.std()
            characteristics['volatility'] = min(1.0, volatility * 10)

            # Momentum (0-1)
            momentum = (closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0
            characteristics['momentum'] = min(1.0, abs(momentum))

            # Cyclicality (0-1) - using autocorrelation
            if len(returns) > 20:
                autocorr = returns.autocorr(lag=10)
                characteristics['cyclicality'] = abs(autocorr) if not np.isnan(autocorr) else 0
            else:
                characteristics['cyclicality'] = 0

            # Volume trend (if available)
            if 'volume' in self.data.columns:
                volumes = self.data['volume'].to_numpy()
                vol_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
                characteristics['volume_trend'] = min(1.0, abs(vol_trend) / np.mean(volumes))
            else:
                characteristics['volume_trend'] = 0

        except Exception as e:
            logger.error(f"Error analyzing data characteristics: {e}")
            # Return neutral characteristics on error
            characteristics = {
                'trend_strength': 0.5,
                'volatility': 0.5,
                'momentum': 0.5,
                'cyclicality': 0.5,
                'volume_trend': 0.5
            }

        return characteristics

    def _score_indicators_for_category(
        self,
        category: str,
        characteristics: dict[str, float]
    ) -> dict[str, float]:
        """Score indicators in a category based on data characteristics."""
        scores = {}

        # Scoring weights based on category and characteristics
        if category == 'trend':
            weight = characteristics['trend_strength']
        elif category == 'momentum':
            weight = characteristics['momentum'] * 0.5 + characteristics['cyclicality'] * 0.5
        elif category == 'volatility':
            weight = characteristics['volatility']
        elif category == 'volume':
            weight = characteristics['volume_trend']
        elif category == 'trend_strength':
            weight = characteristics['trend_strength'] * 0.7 + characteristics['volatility'] * 0.3
        else:
            weight = 0.5

        # Score each indicator in category
        for indicator_name in INDICATOR_CATALOG[category]:
            # Base score from category relevance
            base_score = weight

            # Adjust based on complexity (simpler is better for clear trends)
            if indicator_name in ['SMA', 'EMA', 'RSI', 'ATR', 'OBV']:
                complexity_bonus = 0.1
            elif indicator_name in ['DEMA', 'TEMA', 'BB', 'MACD', 'ADX']:
                complexity_bonus = 0.05
            else:
                complexity_bonus = 0

            scores[indicator_name] = base_score + complexity_bonus

        return scores

    def _optimize_indicator_parameters(self, indicator_name: str) -> dict[str, Any]:
        """
        Optimize parameters for a specific indicator using Optuna.

        Args:
            indicator_name: Name of the indicator

        Returns:
            Dictionary of optimal parameters
        """
        # Find indicator config
        indicator_config = None
        for category, indicators in INDICATOR_CATALOG.items():
            if indicator_name in indicators:
                indicator_config = indicators[indicator_name]
                break

        if not indicator_config:
            return {}

        # If no parameters to optimize, return empty dict
        if not indicator_config['params']:
            return {}

        # Use Optuna for parameter optimization
        def objective(trial):
            params = {}
            for param_name, param_range in indicator_config['params'].items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])

            # Calculate performance score with these parameters
            score = self._calculate_parameter_score(indicator_name, params)
            return score

        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20, show_progress_bar=False)

        return study.best_params

    def _calculate_parameter_score(self, indicator_name: str, params: dict) -> float:
        """Calculate performance score for specific indicator parameters."""
        try:
            # Simple scoring based on profitability of signals
            # This is a simplified version - in production, use backtesting

            # For trend indicators, check trend following ability
            if indicator_name in ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA']:
                period = params.get('period', 20)
                if period < len(self.data):
                    ma = self.data['close'].rolling(period).mean()
                    signals = (self.data['close'] > ma).astype(int).diff()
                    returns = self.data['close'].pct_change()
                    signal_returns = returns[signals == 1].sum()
                    return signal_returns

            # For momentum indicators, check reversal detection
            elif indicator_name in ['RSI', 'CCI', 'StochRSI']:
                period = params.get('period', 14)
                if period < len(self.data):
                    # Simplified RSI calculation
                    delta = self.data['close'].diff().astype(float)
                    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))

                    # Check oversold/overbought signals
                    oversold_signals = (rsi < 30).astype(int).diff()
                    overbought_signals = (rsi > 70).astype(int).diff()

                    returns = self.data['close'].pct_change()
                    oversold_returns = returns[oversold_signals == 1].sum()
                    overbought_returns = -returns[overbought_signals == 1].sum()

                    return oversold_returns + overbought_returns

            # Default score
            return 0.5

        except Exception as e:
            logger.error(f"Error calculating parameter score: {e}")
            return 0.0

    def get_optimized_config(self) -> dict[str, Any]:
        """Get the complete optimized configuration."""
        return {
            'selected_indicators': self.selected_indicators,
            'optimal_parameters': self.optimal_parameters,
            'performance_scores': self.performance_scores,
            'data_characteristics': self._analyze_data_characteristics()
        }

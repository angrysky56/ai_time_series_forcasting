"""
Market indicators integration for broader market context analysis.
Fetches and analyzes key market indicators for both traditional and crypto markets.
"""

import pandas as pd
import numpy as np
from typing import Any
import logging
from src.data_fetcher import fetch_universal_data

logger = logging.getLogger(__name__)

# Market indicator configurations
MARKET_INDICATORS = {
    'traditional': {
        'SPY': {'name': 'S&P 500 ETF', 'category': 'equity'},
        'QQQ': {'name': 'NASDAQ 100 ETF', 'category': 'equity'},
        'DIA': {'name': 'Dow Jones ETF', 'category': 'equity'},
        'IWM': {'name': 'Russell 2000 ETF', 'category': 'equity'},
        '^VIX': {'name': 'Volatility Index', 'category': 'volatility'},
        '^TNX': {'name': '10-Year Treasury Yield', 'category': 'bonds'},
        'DX-Y.NYB': {'name': 'US Dollar Index', 'category': 'currency'},
        'GLD': {'name': 'Gold ETF', 'category': 'commodity'},
        'USO': {'name': 'Oil ETF', 'category': 'commodity'},
        'TLT': {'name': '20+ Year Treasury ETF', 'category': 'bonds'}
    },
    'crypto': {
        'BTC/USDT': {'name': 'Bitcoin', 'category': 'crypto_major'},
        'ETH/USDT': {'name': 'Ethereum', 'category': 'crypto_major'},
        'BNB/USDT': {'name': 'Binance Coin', 'category': 'crypto_major'},
        'ADA/USDT': {'name': 'Cardano', 'category': 'crypto_alt'},
        'SOL/USDT': {'name': 'Solana', 'category': 'crypto_alt'},
        'DOT/USDT': {'name': 'Polkadot', 'category': 'crypto_alt'},
        'AVAX/USDT': {'name': 'Avalanche', 'category': 'crypto_alt'},
        'MATIC/USDT': {'name': 'Polygon', 'category': 'crypto_alt'}
    }
}

class MarketIndicatorsAnalyzer:
    """Analyzes broader market context for better forecasting."""

    def __init__(self):
        self.market_data = {}
        self.correlations = {}
        self.market_regime = None
        self.fundamental_data = {}  # New: Store fundamental analysis
        self.sector_industry_data = {}  # New: Store sector/industry insights

    def fetch_comprehensive_market_data(
        self,
        symbol: str,
        market_type: str = 'auto',
        timeframe: str = '1d',
        limit: int = 100,
        include_fundamentals: bool = True
    ) -> dict[str, Any]:
        """
        Fetch comprehensive market data including fundamentals and sector analysis.

        Args:
            symbol: Target trading symbol
            market_type: 'traditional', 'crypto', or 'auto'
            timeframe: Data timeframe
            limit: Number of data points
            include_fundamentals: Whether to fetch financial statements and analyst data

        Returns:
            Dictionary containing market data and comprehensive analysis
        """
        # First get basic market context (existing functionality)
        market_data = self.fetch_market_context(symbol, market_type, timeframe, limit)

        # Enhanced: Add comprehensive fundamental analysis if symbol is traditional
        if include_fundamentals and market_type in ['traditional', 'auto']:
            self._fetch_fundamental_analysis(symbol)
            self._fetch_sector_industry_insights(symbol)

        return {
            'market_data': market_data,
            'fundamental_data': self.fundamental_data.get(symbol, {}),
            'sector_industry': self.sector_industry_data.get(symbol, {})
        }

    def _fetch_fundamental_analysis(self, symbol: str) -> None:
        """Fetch comprehensive fundamental data using yfinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)

            # Company basics
            info = ticker.info

            fundamental_data = {
                'company_info': {
                    'name': info.get('longName', 'Unknown'),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', None),
                    'forward_pe': info.get('forwardPE', None),
                    'peg_ratio': info.get('pegRatio', None),
                    'beta': info.get('beta', None),
                    'dividend_yield': info.get('dividendYield', None)
                },
                'analyst_data': {},
                'financial_health': {}
            }

            # Analyst recommendations
            try:
                rec_summary = ticker.recommendations_summary
                if isinstance(rec_summary, pd.DataFrame) and not rec_summary.empty:
                    summary_row = rec_summary.loc['strongBuy':'strongSell']
                    if not summary_row.empty:
                        fundamental_data['analyst_data'] = {
                            'strong_buy': summary_row.get('strongBuy', 0),
                            'buy': summary_row.get('buy', 0),
                            'hold': summary_row.get('hold', 0),
                            'sell': summary_row.get('sell', 0),
                            'strong_sell': summary_row.get('strongSell', 0)
                        }
            except Exception as e:
                logger.debug(f"Could not fetch analyst data for {symbol}: {e}")

            # Financial statements summary
            try:
                financials = ticker.financials
                if financials is not None and not financials.empty:
                    # Get most recent year data
                    recent_year = financials.columns[0]
                    fundamental_data['financial_health'] = {
                        'revenue': financials.loc['Total Revenue', recent_year] if 'Total Revenue' in financials.index else None,
                        'net_income': financials.loc['Net Income', recent_year] if 'Net Income' in financials.index else None,
                        'gross_profit': financials.loc['Gross Profit', recent_year] if 'Gross Profit' in financials.index else None
                    }
            except Exception as e:
                logger.debug(f"Could not fetch financial statements for {symbol}: {e}")

            self.fundamental_data[symbol] = fundamental_data
            logger.info(f"Fetched fundamental data for {symbol}")

        except Exception as e:
            logger.error(f"Error fetching fundamental analysis for {symbol}: {e}")

    def _fetch_sector_industry_insights(self, symbol: str) -> None:
        """Fetch sector and industry performance insights."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info

            sector = info.get('sector')
            industry = info.get('industry')

            if sector:
                # Map common sectors to ETF proxies for performance comparison
                sector_etfs = {
                    'Technology': 'XLK',
                    'Healthcare': 'XLV',
                    'Financial Services': 'XLF',
                    'Consumer Cyclical': 'XLY',
                    'Consumer Defensive': 'XLP',
                    'Energy': 'XLE',
                    'Industrials': 'XLI',
                    'Materials': 'XLB',
                    'Utilities': 'XLU',
                    'Real Estate': 'XLRE',
                    'Communication Services': 'XLC'
                }

                sector_etf = sector_etfs.get(sector)
                sector_performance = None

                if sector_etf:
                    try:
                        sector_ticker = yf.Ticker(sector_etf)
                        sector_data = sector_ticker.history(period='1mo')
                        if not sector_data.empty:
                            sector_performance = {
                                'symbol': sector_etf,
                                'monthly_return': ((sector_data['Close'].iloc[-1] - sector_data['Close'].iloc[0]) / sector_data['Close'].iloc[0]) * 100
                            }
                    except Exception as e:
                        logger.debug(f"Could not fetch sector ETF data for {sector}: {e}")

                self.sector_industry_data[symbol] = {
                    'sector': sector,
                    'industry': industry,
                    'sector_etf': sector_etf,
                    'sector_performance': sector_performance
                }

                logger.info(f"Fetched sector/industry insights for {symbol}")

        except Exception as e:
            logger.error(f"Error fetching sector insights for {symbol}: {e}")

    def generate_enhanced_market_summary(
        self,
        target_symbol: str,
        target_data: pd.DataFrame,
        include_fundamentals: bool = True
    ) -> dict[str, Any]:
        """
        Generate comprehensive market summary including fundamentals.

        Args:
            target_symbol: Target trading symbol
            target_data: Target symbol price data
            include_fundamentals: Whether to include fundamental analysis

        Returns:
            Enhanced market context analysis
        """
        # Get basic market context (existing functionality)
        basic_summary = self.generate_market_context_summary(target_symbol, target_data)

        # Add fundamental insights if available
        enhanced_summary = basic_summary.copy()

        if include_fundamentals and target_symbol in self.fundamental_data:
            fund_data = self.fundamental_data[target_symbol]

            # Add fundamental insights
            fundamental_insights = []

            company_info = fund_data.get('company_info', {})
            if company_info.get('pe_ratio'):
                pe_ratio = company_info['pe_ratio']
                if pe_ratio < 15:
                    fundamental_insights.append(f"Low P/E ratio ({pe_ratio:.1f}) suggests potential value opportunity")
                elif pe_ratio > 25:
                    fundamental_insights.append(f"High P/E ratio ({pe_ratio:.1f}) indicates growth expectations or overvaluation")

            if company_info.get('beta'):
                beta = company_info['beta']
                if beta > 1.2:
                    fundamental_insights.append(f"High beta ({beta:.2f}) indicates higher volatility than market")
                elif beta < 0.8:
                    fundamental_insights.append(f"Low beta ({beta:.2f}) suggests defensive characteristics")

            # Add analyst sentiment
            analyst_data = fund_data.get('analyst_data', {})
            if analyst_data:
                total_ratings = sum(analyst_data.values())
                if total_ratings > 0:
                    bullish_ratio = (analyst_data.get('strong_buy', 0) + analyst_data.get('buy', 0)) / total_ratings
                    if bullish_ratio > 0.6:
                        fundamental_insights.append("Strong analyst bullish sentiment")
                    elif bullish_ratio < 0.3:
                        fundamental_insights.append("Analyst bearish sentiment")

            enhanced_summary['fundamental_insights'] = fundamental_insights
            enhanced_summary['company_metrics'] = company_info

        # Add sector performance if available
        if target_symbol in self.sector_industry_data:
            sector_data = self.sector_industry_data[target_symbol]
            enhanced_summary['sector_analysis'] = sector_data

            # Add sector performance insight
            if sector_data.get('sector_performance'):
                sector_perf = sector_data['sector_performance']['monthly_return']
                if sector_perf > 5:
                    enhanced_summary['insights'].append(f"Sector ({sector_data['sector']}) showing strong performance (+{sector_perf:.1f}% monthly)")
                elif sector_perf < -5:
                    enhanced_summary['insights'].append(f"Sector ({sector_data['sector']}) under pressure ({sector_perf:.1f}% monthly)")

        return enhanced_summary

    def fetch_market_context(
        self,
        symbol: str,
        market_type: str = 'auto',
        timeframe: str = '1d',
        limit: int = 100,
        exchange: str | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch relevant market indicators based on the target symbol.

        Args:
            symbol: Target trading symbol
            market_type: 'traditional', 'crypto', or 'auto'
            timeframe: Data timeframe
            limit: Number of data points
            exchange: Crypto exchange (if applicable)

        Returns:
            Dictionary of indicator -> DataFrame
        """
        # Determine market type
        if market_type == 'auto':
            if '/' in symbol or symbol.endswith('USDT') or symbol.endswith('USD'):
                market_type = 'crypto'
            else:
                market_type = 'traditional'

        # Select relevant indicators
        if market_type == 'crypto':
            indicators_to_fetch = list(MARKET_INDICATORS['crypto'].keys())
            # Add some traditional indicators for broader context
            indicators_to_fetch.extend(['SPY', 'VIX', 'DXY'])
        else:
            indicators_to_fetch = list(MARKET_INDICATORS['traditional'].keys())
            # Remove the target symbol if it's in the list
            if symbol in indicators_to_fetch:
                indicators_to_fetch.remove(symbol)

        market_data = {}

        for indicator in indicators_to_fetch:
            data = None  # Initialize data to None
            try:
                # Fetch market data using unified fetcher for all indicators
                data = fetch_universal_data(indicator, timeframe, limit)

                if data is not None and not data.empty:
                    market_data[indicator] = data
                    logger.info(f"Fetched market data for {indicator}")
                else:
                    logger.warning(f"Failed to fetch data for {indicator}")
            except Exception as e:
                logger.error(f"Error fetching {indicator}: {e}")
                continue

        self.market_data = market_data
        return market_data

    def calculate_correlations(
        self,
        target_data: pd.DataFrame,
        symbol: str
    ) -> dict[str, float]:
        """
        Calculate correlations between target symbol and market indicators.

        Args:
            target_data: Target symbol price data
            symbol: Target symbol name

        Returns:
            Dictionary of correlations
        """
        correlations = {}

        if target_data is None or target_data.empty:
            return correlations

        target_returns = target_data['close'].pct_change().dropna()

        for indicator, data in self.market_data.items():
            try:
                if data is None or data.empty:
                    continue

                # Calculate returns
                indicator_returns = data['close'].pct_change().dropna()

                # Align data lengths
                min_length = min(len(target_returns), len(indicator_returns))
                if min_length < 10:  # Need at least 10 data points
                    continue

                target_aligned = target_returns.tail(min_length)
                indicator_aligned = indicator_returns.tail(min_length)

                # Calculate correlation
                correlation = target_aligned.corr(indicator_aligned)

                if not np.isnan(correlation):
                    correlations[indicator] = correlation

            except Exception as e:
                logger.error(f"Error calculating correlation with {indicator}: {e}")
                continue

        self.correlations = correlations
        return correlations

    def assess_market_regime(self) -> dict[str, Any]:
        """
        Assess current market regime based on key indicators.

        Returns:
            Dictionary with market regime analysis
        """
        regime_analysis = {
            'regime_type': 'unknown',
            'risk_level': 'medium',
            'volatility_state': 'normal',
            'trend_direction': 'sideways',
            'signals': []
        }

        try:
            # VIX Analysis (Fear/Greed indicator)
            if '^VIX' in self.market_data:
                vix_data = self.market_data['^VIX']
                current_vix = vix_data['close'].iloc[-1]

                if current_vix > 30:
                    regime_analysis['volatility_state'] = 'high'
                    regime_analysis['risk_level'] = 'high'
                    regime_analysis['signals'].append('High market fear (VIX > 30)')
                elif current_vix < 15:
                    regime_analysis['volatility_state'] = 'low'
                    regime_analysis['risk_level'] = 'low'
                    regime_analysis['signals'].append('Low market fear (VIX < 15)')
                else:
                    regime_analysis['volatility_state'] = 'normal'

            # SPY/Market Trend Analysis
            if 'SPY' in self.market_data:
                spy_data = self.market_data['SPY']
                if len(spy_data) > 20:
                    sma_20 = spy_data['close'].rolling(20).mean().iloc[-1]
                    current_price = spy_data['close'].iloc[-1]

                    if current_price > sma_20 * 1.02:
                        regime_analysis['trend_direction'] = 'bullish'
                        regime_analysis['signals'].append('Market above 20-day SMA')
                    elif current_price < sma_20 * 0.98:
                        regime_analysis['trend_direction'] = 'bearish'
                        regime_analysis['signals'].append('Market below 20-day SMA')

            # Dollar Strength (DXY)
            if 'DX-Y.NYB' in self.market_data:
                dxy_data = self.market_data['DX-Y.NYB']
                if len(dxy_data) > 10:
                    dxy_change = ((dxy_data['close'].iloc[-1] - dxy_data['close'].iloc[-10]) /
                                 dxy_data['close'].iloc[-10]) * 100

                    if dxy_change > 2:
                        regime_analysis['signals'].append('Strong dollar strength')
                    elif dxy_change < -2:
                        regime_analysis['signals'].append('Dollar weakness')

            # Bitcoin Dominance (for crypto markets)
            if 'BTC/USDT' in self.market_data and 'ETH/USDT' in self.market_data:
                btc_data = self.market_data['BTC/USDT']
                eth_data = self.market_data['ETH/USDT']

                if len(btc_data) > 5 and len(eth_data) > 5:
                    btc_change = ((btc_data['close'].iloc[-1] - btc_data['close'].iloc[-5]) /
                                 btc_data['close'].iloc[-5]) * 100
                    eth_change = ((eth_data['close'].iloc[-1] - eth_data['close'].iloc[-5]) /
                                 eth_data['close'].iloc[-5]) * 100

                    if btc_change > eth_change + 5:
                        regime_analysis['signals'].append('Bitcoin outperforming altcoins')
                    elif eth_change > btc_change + 5:
                        regime_analysis['signals'].append('Altcoin season potential')

            # Determine overall regime
            if regime_analysis['volatility_state'] == 'high':
                if regime_analysis['trend_direction'] == 'bearish':
                    regime_analysis['regime_type'] = 'bear_market'
                else:
                    regime_analysis['regime_type'] = 'volatile_market'
            elif regime_analysis['trend_direction'] == 'bullish':
                if regime_analysis['volatility_state'] == 'low':
                    regime_analysis['regime_type'] = 'bull_market'
                else:
                    regime_analysis['regime_type'] = 'growing_market'
            else:
                regime_analysis['regime_type'] = 'sideways_market'

        except Exception as e:
            logger.error(f"Error in market regime assessment: {e}")

        self.market_regime = regime_analysis
        return regime_analysis

    def generate_market_context_summary(
        self,
        target_symbol: str,
        target_data: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Generate comprehensive market context summary.

        Args:
            target_symbol: Target trading symbol
            target_data: Target symbol price data

        Returns:
            Comprehensive market context analysis
        """
        # Calculate correlations
        correlations = self.calculate_correlations(target_data, target_symbol)

        # Assess market regime
        regime = self.assess_market_regime()

        # Find strongest correlations
        strong_correlations = {
            k: v for k, v in correlations.items()
            if abs(v) > 0.3
        }

        # Sort by absolute correlation strength
        sorted_correlations = dict(
            sorted(strong_correlations.items(),
                  key=lambda x: abs(x[1]), reverse=True)
        )

        # Generate insights
        insights = []

        # Correlation insights
        if sorted_correlations:
            strongest = list(sorted_correlations.items())[0]
            indicator_name = MARKET_INDICATORS.get('traditional', {}).get(
                strongest[0], MARKET_INDICATORS.get('crypto', {}).get(
                    strongest[0], {'name': strongest[0]}
                )
            )['name']

            if strongest[1] > 0.5:
                insights.append(f"Strong positive correlation with {indicator_name} ({strongest[1]:.2f})")
            elif strongest[1] < -0.5:
                insights.append(f"Strong negative correlation with {indicator_name} ({strongest[1]:.2f})")

        # Regime insights
        insights.extend(regime['signals'])

        return {
            'correlations': sorted_correlations,
            'market_regime': regime,
            'insights': insights,
            'risk_factors': self._identify_risk_factors(),
            'opportunities': self._identify_opportunities()
        }

    def _identify_risk_factors(self) -> list[str]:
        """Identify current market risk factors."""
        risks = []

        if self.market_regime:
            if self.market_regime['volatility_state'] == 'high':
                risks.append("High market volatility increases uncertainty")

            if self.market_regime['risk_level'] == 'high':
                risks.append("Elevated market stress levels")

        # Check for extreme correlations
        if self.correlations:
            high_correlations = [k for k, v in self.correlations.items() if abs(v) > 0.8]
            if high_correlations:
                risks.append("High correlation with market indicators reduces diversification")

        return risks

    def _identify_opportunities(self) -> list[str]:
        """Identify current market opportunities."""
        opportunities = []

        if self.market_regime:
            if self.market_regime['volatility_state'] == 'low':
                opportunities.append("Low volatility environment may favor trend following")

            if self.market_regime['trend_direction'] == 'bullish':
                opportunities.append("Bullish market trend supports long positions")

        # Check for negative correlations (hedging opportunities)
        if self.correlations:
            negative_corr = [k for k, v in self.correlations.items() if v < -0.3]
            if negative_corr:
                opportunities.append("Negative correlations provide hedging opportunities")

        return opportunities
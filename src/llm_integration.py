import requests
import pandas as pd
from typing import Any
import logging

logger = logging.getLogger(__name__)

# LM Studio API Configuration
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"

def get_ai_analysis(
    symbol: str,
    historical_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    model_name: str | None = None,
    validation_results: dict[str, Any] | None = None,
    technical_indicators: dict[str, Any] | None = None,
    exchange_info: dict[str, Any] | None = None
) -> str:
    """
    Generates comprehensive AI analysis using all available data.

    Args:
        symbol (str): The cryptocurrency symbol.
        historical_data (pd.DataFrame): Historical price data.
        forecast_data (pd.DataFrame): Forecast data.
        model_name (str): Selected forecasting model.
        validation_results (dict): Walk-forward validation results.
        technical_indicators (dict): Technical indicator values.
        exchange_info (dict): Exchange and timeframe information.

    Returns:
        str: Comprehensive AI-generated analysis.
    """
    if forecast_data is None or forecast_data.empty or historical_data is None or historical_data.empty:
        return "Insufficient data for AI analysis."

    # Extract basic forecast metrics
    last_known_price = historical_data['close'].iloc[-1]
    forecast_end_price = forecast_data['yhat'].iloc[-1]
    forecast_period = len(forecast_data) - len(historical_data)
    percentage_change = ((forecast_end_price - last_known_price) / last_known_price) * 100

    # Build comprehensive analysis context
    analysis_context = _build_analysis_context(
        symbol, last_known_price, forecast_end_price, percentage_change, forecast_period,
        model_name, validation_results, technical_indicators, exchange_info, historical_data
    )

    # Generate enhanced prompt
    prompt = _create_enhanced_prompt(analysis_context)

    # Call LLM API
    return _call_llm_api(prompt)

def _build_analysis_context(
    symbol: str,
    last_price: float,
    forecast_price: float,
    price_change: float,
    forecast_period: int,
    model_name: str | None,
    validation_results: dict[str, Any] | None,
    technical_indicators: dict[str, Any] | None,
    exchange_info: dict[str, Any] | None,
    historical_data: pd.DataFrame
) -> dict[str, Any]:
    """Build comprehensive context for AI analysis."""

    context = {
        'basic_forecast': {
            'symbol': symbol,
            'last_price': last_price,
            'forecast_price': forecast_price,
            'price_change_pct': price_change,
            'forecast_period': forecast_period,
            'model_used': model_name or 'Unknown'
        }
    }

    # Add price statistics
    context['price_statistics'] = {
        'volatility_pct': historical_data['close'].pct_change().std() * 100,
        'price_range_24h': {
            'high': historical_data['high'].tail(24).max() if len(historical_data) >= 24 else historical_data['high'].max(),
            'low': historical_data['low'].tail(24).min() if len(historical_data) >= 24 else historical_data['low'].min()
        },
        'avg_volume': historical_data['volume'].mean(),
        'trend_direction': 'bullish' if price_change > 0 else 'bearish' if price_change < 0 else 'sideways'
    }

    # Add validation metrics if available
    if validation_results and 'error' not in validation_results:
        metrics = validation_results.get('metrics', {})
        context['model_performance'] = {
            'validation_type': validation_results.get('validation_type'),
            'num_folds': validation_results.get('num_folds'),
            'mean_rmse': metrics.get('mean_RMSE'),
            'mean_mape': metrics.get('mean_MAPE'),
            'std_rmse': metrics.get('std_RMSE'),
            'combined_rmse': metrics.get('combined_RMSE'),
            'reliability_score': _calculate_reliability_score(metrics)
        }

    # Add technical indicators if available
    if technical_indicators:
        context['technical_analysis'] = technical_indicators

    # Add exchange/timeframe info if available
    if exchange_info:
        context['market_context'] = exchange_info

    return context

def _calculate_reliability_score(metrics: dict[str, Any]) -> str:
    """Calculate model reliability based on validation metrics."""
    mean_mape = metrics.get('mean_MAPE', 100)
    std_rmse = metrics.get('std_RMSE', float('inf'))
    mean_rmse = metrics.get('mean_RMSE', float('inf'))

    # Calculate coefficient of variation for RMSE
    cv_rmse = (std_rmse / mean_rmse) if mean_rmse > 0 else float('inf')

    if mean_mape < 2 and cv_rmse < 0.2:
        return 'Very High'
    elif mean_mape < 5 and cv_rmse < 0.4:
        return 'High'
    elif mean_mape < 10 and cv_rmse < 0.6:
        return 'Moderate'
    else:
        return 'Low'

def _create_enhanced_prompt(context: dict[str, Any]) -> str:
    """Create comprehensive analysis prompt."""
    basic = context['basic_forecast']
    stats = context.get('price_statistics', {})
    performance = context.get('model_performance')
    technical = context.get('technical_analysis')
    market = context.get('market_context')

    prompt = f"""
    Act as a professional cryptocurrency analyst providing comprehensive market analysis.

    FORECAST OVERVIEW:
    - Symbol: {basic['symbol']}
    - Current Price: ${basic['last_price']:,.2f}
    - Predicted Price ({basic['forecast_period']} periods): ${basic['forecast_price']:,.2f}
    - Expected Change: {basic['price_change_pct']:+.2f}%
    - Model Used: {basic['model_used']}
    - Trend Direction: {stats.get('trend_direction', 'Unknown')}

    MARKET STATISTICS:
    - Recent Volatility: {stats.get('volatility_pct', 0):.2f}%
    - Average Volume: {stats.get('avg_volume', 0):,.0f}
    """

    if performance:
        prompt += f"""

    MODEL VALIDATION METRICS:
    - Validation Method: {performance.get('validation_type', 'Unknown')}
    - Cross-Validation Folds: {performance.get('num_folds', 0)}
    - Mean Prediction Error (MAPE): {performance.get('mean_mape', 0):.2f}%
    - Model Consistency (RMSE): {performance.get('mean_rmse', 0):.2f}
    - Reliability Score: {performance.get('reliability_score', 'Unknown')}
    """

    if technical:
        prompt += """

    TECHNICAL INDICATORS:
    """
        for indicator, value in technical.items():
            if isinstance(value, (int, float)):
                prompt += f"- {indicator}: {value:.2f}\n    "

    if market:
        prompt += f"""

    MARKET CONTEXT:
    - Exchange: {market.get('exchange', 'Unknown')}
    - Timeframe: {market.get('timeframe', 'Unknown')}
    - Data Points: {market.get('data_points', 0)}
    """

    prompt += """

    ANALYSIS REQUIREMENTS:
    Provide a comprehensive analysis covering:

    1. **Price Forecast Assessment**: Evaluate the predicted price movement and its implications
    2. **Model Reliability**: Assess the forecasting model's performance and trustworthiness
    3. **Technical Analysis**: Interpret Any technical indicators in market context
    4. **Risk Assessment**: Identify key risks and market factors that could affect the forecast
    5. **Market Context**: Consider timeframe, volatility, and trading conditions

    Keep the analysis professional, data-driven, and objective. Include specific metrics where relevant.
    Conclude with a clear summary of the forecast confidence level and key considerations.

    Format: Use clear headings and bullet points for readability.
    Length: 4-6 comprehensive paragraphs.
    """

    return prompt

def _call_llm_api(prompt: str) -> str:
    """Call the LLM API with error handling."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": "You are a professional cryptocurrency market analyst with expertise in quantitative analysis, technical indicators, and risk assessment."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6,
        "max_tokens": 8164
    }

    try:
        response = requests.post(LM_STUDIO_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        ai_response = result['choices'][0]['message']['content']
        return ai_response

    except requests.exceptions.RequestException as e:
        error_msg = f"LLM API Connection Error: {e}"
        logger.error(error_msg)
        return f"**API Connection Failed**: {error_msg}\n\nPlease ensure LM Studio is running and accessible at {LM_STUDIO_API_URL}"

    except (KeyError, IndexError) as e:
        error_msg = f"LLM API Response Error: {e}"
        logger.error(error_msg)
        return f"**API Response Error**: {error_msg}\n\nThe LLM response format may have changed."

    except Exception as e:
        error_msg = f"Unexpected LLM Error: {e}"
        logger.error(error_msg)
        return f"**Analysis Error**: {error_msg}"

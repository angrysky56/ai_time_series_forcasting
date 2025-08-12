import requests
import pandas as pd
from typing import Any, Dict
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
    exchange_info: dict[str, Any] | None = None,
    market_context: dict[str, Any] | None = None
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
        market_context (dict): Market indicators, correlations, and regime analysis.

    Returns:
        str: Comprehensive AI-generated analysis.
    """
    if forecast_data is None or forecast_data.empty or historical_data is None or historical_data.empty:
        return "Insufficient data for AI analysis."

    # Extract basic forecast metrics (use only true future horizon)
    last_known_price = historical_data['close'].iloc[-1]
    last_hist_ts = pd.Timestamp(pd.to_datetime(historical_data['timestamp']).max())
    # Ensure 'ds' is datetime
    forecast_df = forecast_data.copy()
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'], errors='coerce')
    future_mask = forecast_df['ds'] > last_hist_ts
    future_forecast = forecast_df[future_mask]
    if future_forecast.empty:
        # Fallback: use tail(1) if future mask fails (e.g., models that only return future are fine too)
        future_forecast = forecast_df.tail(1)

    forecast_end_price = future_forecast['yhat'].iloc[-1]
    forecast_period = len(future_forecast)
    percentage_change = ((forecast_end_price - last_known_price) / last_known_price) * 100

    # Build comprehensive analysis context
    analysis_context = _build_analysis_context(
        symbol, last_known_price, forecast_end_price, percentage_change, forecast_period,
        model_name, validation_results, technical_indicators, exchange_info, historical_data, market_context
    )

    # Generate enhanced prompt
    prompt = _create_enhanced_prompt(analysis_context)

    # Call LLM API
    return _call_llm_api(prompt)

def get_multi_period_ai_analysis(
    symbol: str,
    multi_historical: dict[str, pd.DataFrame],
    multi_forecasts: dict[str, pd.DataFrame],
    report: dict[str, Any],
    exchange_info: dict[str, Any] | None = None
) -> str:
    """Generate an AI narrative for multi-period coherent analysis.

    Summarizes trends across monthly/weekly/daily/hourly, includes consistency scores
    and volatility, then requests a cohesive interpretation and recommendations.
    """
    # Build a compact summary table per timeframe
    tf_summaries: list[str] = []
    for tf in ['monthly', 'weekly', 'daily', 'hourly']:
        hist = multi_historical.get(tf)
        fc = multi_forecasts.get(tf)
        if hist is None or fc is None or hist.empty or fc.empty:
            continue
        last_price = hist['close'].iloc[-1]
        
        # Fix timezone comparison issue - normalize both to timezone-naive UTC
        hist_timestamps = pd.to_datetime(hist['timestamp'])
        if hist_timestamps.dt.tz is not None:
            hist_timestamps = hist_timestamps.dt.tz_convert('UTC').dt.tz_localize(None)
        last_hist_ts = hist_timestamps.max()
        
        fcf = fc.copy()
        fcf['ds'] = pd.to_datetime(fcf['ds'], errors='coerce')
        # Ensure forecast timestamps are also timezone-naive UTC
        if fcf['ds'].dt.tz is not None:
            fcf['ds'] = fcf['ds'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        # Now both are timezone-naive UTC - safe to compare
        fut = fcf[fcf['ds'] > last_hist_ts]
        if fut.empty:
            fut = fcf.tail(1)
        end_price = fut['yhat'].iloc[-1]
        change_pct = (end_price - last_price) / last_price * 100 if last_price else 0
        tf_summaries.append(f"- {tf.title()}: last={last_price:,.2f}, forecast_end={end_price:,.2f}, change={change_pct:+.2f}% ({len(fut)} steps)")

    # Pull key pieces from the coherent report
    summary = report.get('summary', {}) if isinstance(report, dict) else {}
    trends = summary.get('trend_direction', {})
    vol = summary.get('volatility', {})
    consistency = report.get('consistency_scores', {}) if isinstance(report, dict) else {}
    overall_consistency = consistency.get('overall', None)

    # Create a focused prompt
    details = "\n".join(tf_summaries) if tf_summaries else "(no timeframe summaries available)"
    trend_lines = []
    for tf, info in trends.items():
        if isinstance(info, dict) and 'direction' in info and 'change_percent' in info:
            arrow = '📈' if info['direction'] == 'up' else '📉'
            trend_lines.append(f"  • {tf.title()}: {arrow} {info['change_percent']:+.2f}%")
    trend_block = "\n".join(trend_lines) if trend_lines else "  • Not available"

    vol_lines = []
    for tf, v in vol.items():
        vol_lines.append(f"  • {tf.title()}: {v:.2f}%")
    vol_block = "\n".join(vol_lines) if vol_lines else "  • Not available"

    exch_block = ""
    if exchange_info:
        exch_block = f"Exchange: {exchange_info.get('exchange', 'Unknown')} | Timeframe: multi | Data completeness varies by tf."

    prompt = f"""
    Act as a senior market strategist. Provide a cohesive, multi-timeframe narrative for {symbol} using the details below.

    TIMEFRAME SNAPSHOTS (last price vs forecast end):
    {details}

    TRENDS BY TIMEFRAME:
    {trend_block}

    VOLATILITY ESTIMATES (avg CI width / mean):
    {vol_block}

    CONSISTENCY SCORES:
      • Monthly vs Weekly: {consistency.get('monthly_vs_weekly', 'n/a')}
      • Weekly vs Daily: {consistency.get('weekly_vs_daily', 'n/a')}
      • Daily vs Hourly: {consistency.get('daily_vs_hourly', 'n/a')}
      • Overall: {overall_consistency if overall_consistency is not None else 'n/a'}

    {exch_block}

    Requirements:
    - Synthesize a clear story across horizons (long/medium/short).
    - Reconcile disagreements (if any) and indicate confidence based on consistency.
    - Highlight risks/opportunities and actionable timeframes for entries/exits.
    - Keep it concise and structured with bullet points where helpful.
    """

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
    historical_data: pd.DataFrame,
    market_context: dict[str, Any] | None = None
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

    # Add broader market context if available
    if market_context:
        context['broader_market_analysis'] = {
            'correlations': market_context.get('correlations', {}),
            'market_regime': market_context.get('market_regime', {}),
            'insights': market_context.get('insights', []),
            'risk_factors': market_context.get('risk_factors', []),
            'opportunities': market_context.get('opportunities', [])
        }

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
    broader_market = context.get('broader_market_analysis')

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

    if broader_market:
        prompt += """

    BROADER MARKET ANALYSIS:
    """
        regime = broader_market.get('market_regime', {})
        if regime:
            prompt += f"- Market Regime: {regime.get('regime_type', 'unknown').replace('_', ' ').title()}\n    "
            prompt += f"- Risk Level: {regime.get('risk_level', 'unknown').title()}\n    "
            prompt += f"- Volatility State: {regime.get('volatility_state', 'unknown').title()}\n    "
            prompt += f"- Trend Direction: {regime.get('trend_direction', 'unknown').title()}\n    "

        correlations = broader_market.get('correlations', {})
        if correlations:
            prompt += "- Key Correlations:\n    "
            for indicator, corr in list(correlations.items())[:3]:
                prompt += f"  • {indicator}: {corr:+.3f}\n    "

        insights = broader_market.get('insights', [])
        if insights:
            prompt += "- Market Insights:\n    "
            for insight in insights[:3]:
                prompt += f"  • {insight}\n    "

        risks = broader_market.get('risk_factors', [])
        if risks:
            prompt += "- Risk Factors:\n    "
            for risk in risks[:2]:
                prompt += f"  • {risk}\n    "

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
    """Call the LLM API with enhanced error handling and connectivity checks."""
    
    # Pre-flight connectivity check
    connectivity_check = _check_lm_studio_connectivity()
    if not connectivity_check['available']:
        return _generate_fallback_analysis(connectivity_check['error'])
    
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
        response = requests.post(LM_STUDIO_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        
        # Enhanced response validation
        if 'choices' not in result or not result['choices']:
            raise ValueError("Invalid API response: no choices returned")
        
        if 'message' not in result['choices'][0] or 'content' not in result['choices'][0]['message']:
            raise ValueError("Invalid API response: no content in message")
        
        ai_response = result['choices'][0]['message']['content']
        
        # Validate response quality
        if not ai_response or len(ai_response.strip()) < 50:
            logger.warning("LLM returned very short response, may indicate an issue")
            return ai_response + "\n\n⚠️ *Note: LLM response was unusually short*"
        
        return ai_response

    except requests.exceptions.ConnectionError:
        error_msg = "Cannot connect to LM Studio. Please ensure LM Studio is running on localhost:1234"
        logger.error(error_msg)
        return _generate_fallback_analysis(error_msg)

    except requests.exceptions.Timeout:
        error_msg = "LM Studio request timed out (60s). The model may be overloaded"
        logger.error(error_msg)
        return _generate_fallback_analysis(error_msg)

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            error_msg = "LM Studio endpoint not found. Check if the local server is properly configured"
        elif e.response.status_code == 500:
            error_msg = "LM Studio internal error. Try restarting the model or checking model compatibility"
        else:
            error_msg = f"LM Studio HTTP error: {e.response.status_code}"
        logger.error(error_msg)
        return _generate_fallback_analysis(error_msg)

    except (KeyError, IndexError, ValueError) as e:
        error_msg = f"Invalid LLM response format: {e}"
        logger.error(error_msg)
        return _generate_fallback_analysis(error_msg)

    except Exception as e:
        error_msg = f"Unexpected LLM error: {e}"
        logger.error(error_msg, exc_info=True)
        return _generate_fallback_analysis(error_msg)


def _check_lm_studio_connectivity() -> Dict[str, Any]:
    """Check if LM Studio is accessible with quick health check.
    
    Returns:
        dict: {'available': bool, 'error': str, 'response_time': float}
    """
    import time
    
    start_time = time.time()
    try:
        # Quick health check with minimal payload
        response = requests.get(f"{LM_STUDIO_API_URL.replace('/chat/completions', '/health')}", timeout=5)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            return {'available': True, 'error': None, 'response_time': response_time}
        else:
            # Try the main endpoint with a minimal request
            test_payload = {
                "model": "local-model",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }
            response = requests.post(LM_STUDIO_API_URL, json=test_payload, timeout=5)
            response_time = time.time() - start_time
            
            if response.status_code in [200, 400]:  # 400 might be expected for minimal request
                return {'available': True, 'error': None, 'response_time': response_time}
            else:
                return {'available': False, 'error': f"HTTP {response.status_code}", 'response_time': response_time}
                
    except requests.exceptions.ConnectionError:
        return {'available': False, 'error': "Connection refused - LM Studio not running", 'response_time': time.time() - start_time}
    except requests.exceptions.Timeout:
        return {'available': False, 'error': "Health check timeout", 'response_time': time.time() - start_time}
    except Exception as e:
        return {'available': False, 'error': str(e), 'response_time': time.time() - start_time}


def _generate_fallback_analysis(error_reason: str) -> str:
    """Generate a basic technical analysis when LLM is unavailable.
    
    Args:
        error_reason: Reason why LLM is unavailable
        
    Returns:
        str: Fallback analysis with technical indicators
    """
    return f"""
## 📊 Technical Analysis (LLM Unavailable)

**🔧 LLM Status**: {error_reason}

### 📈 Forecast Summary
- **Model Performance**: Based on validation metrics provided
- **Price Direction**: Review the forecast trend line and confidence intervals
- **Volatility**: Check the width of confidence bands for risk assessment

### 🛠 Troubleshooting LLM Integration
1. **Check LM Studio**: Ensure LM Studio is running on localhost:1234
2. **Model Status**: Verify a model is loaded and responding in LM Studio
3. **Network**: Check if any firewall is blocking localhost connections
4. **Memory**: Ensure sufficient RAM for the LLM model

### 📋 Manual Analysis Guidelines
- **Trend Analysis**: Look at the forecast direction relative to recent price action
- **Confidence Assessment**: Wider confidence bands indicate higher uncertainty
- **Technical Indicators**: Review any displayed moving averages, RSI, or other indicators
- **Market Context**: Consider current market volatility and trading volume

### 💡 Next Steps
- Restart LM Studio and reload the model
- Check the LM Studio console for error messages
- Ensure the model is compatible with the API format
- Try reducing the analysis complexity if the model is struggling

*Note: This fallback analysis provides basic guidance. For detailed market insights, please resolve the LLM connectivity issue.*
"""

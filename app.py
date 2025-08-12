import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.data_fetcher import (
    fetch_universal_data,
    discover_symbols,
    validate_timeframe_support
)
from src.multi_period_forecaster import MultiPeriodForecaster
from src.ai_parameter_optimizer import AIParameterOptimizer
from src.market_indicators import MarketIndicatorsAnalyzer
from src.forecasting import generate_forecast
from src.llm_integration import get_ai_analysis, get_multi_period_ai_analysis
from src.technical_analysis import calculate_sma, calculate_ema, calculate_rsi
from src.backtesting import walk_forward_validation, compare_models_walkforward
from src.utils.chart_renderer import StreamlitChartRenderer, ChartRenderError

st.set_page_config(layout="wide", page_title="AI Crypto Forecaster")
st.title("🤖 AI-Assisted Crypto Forecaster")
st.markdown("*Enhanced with Walk-Forward Validation & Multi-Exchange Support*")

# Initialize chart renderer with error boundaries
chart_renderer = StreamlitChartRenderer()

# Initialize session state
if 'discovered_symbols' not in st.session_state:
    st.session_state.discovered_symbols = []
if 'selected_exchange' not in st.session_state:
    st.session_state.selected_exchange = 'coinbase'

# --- Sidebar Controls ---
st.sidebar.header("📊 Data Configuration")

# Data source selection
data_source = st.sidebar.selectbox(
    "Data Source",
    ['auto', 'crypto', 'stock'],
    help="Auto-detect based on symbol format"
)

# Since we're using yfinance exclusively, simplify the exchange selection
st.sidebar.info("📈 Using Yahoo Finance for all data")
exchange_name = 'yfinance'

# Symbol discovery - show available symbols
if st.sidebar.button("🔍 Show Available Symbols", help="Show available trading symbols"):
    with st.spinner("Loading available symbols..."):
        symbols = discover_symbols('yfinance', limit=100)
        st.session_state.discovered_symbols = symbols
        st.session_state.selected_exchange = 'yfinance'
    if symbols:
        st.sidebar.success(f"Found {len(symbols)} symbols")
    else:
        st.sidebar.error("No symbols found")

# Symbol input/selection
symbol_options = []
if data_source == 'stock':
    symbol_options = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ', 'BTC-USD', 'ETH-USD']
    default_symbol = 'AAPL'
elif st.session_state.discovered_symbols:
    symbol_options = st.session_state.discovered_symbols[:50]  # Top 50 for performance
    default_symbol = 'BTC/USDT'
else:
    default_symbol = 'BTC/USDT' if data_source in ['auto', 'crypto'] else 'AAPL'

if symbol_options:
    symbol_method = st.sidebar.radio("Symbol Input", ["Select from list", "Manual entry"])
    if symbol_method == "Select from list":
        symbol = st.sidebar.selectbox("Symbol", symbol_options)
    else:
        symbol = st.sidebar.text_input("Enter Symbol", default_symbol).upper()
else:
    symbol = st.sidebar.text_input("Enter Symbol", default_symbol).upper()

st.sidebar.header("🔮 Forecasting")

# Analysis mode selection
analysis_mode = st.sidebar.selectbox(
    "Analysis Mode",
    ["Single Period", "Multi-Period Coherent"],
    help="Multi-period provides linked forecasts across timeframes"
)

# Initialize variables with defaults
timeframe = '1d'
data_limit = 5000

if analysis_mode == "Single Period":
    # Timeframe selection with validation
    timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
    timeframe = st.sidebar.selectbox("Timeframe", timeframes, index=6)  # Default to 1d

    # Validate timeframe support
    if data_source in ['auto', 'crypto'] and not validate_timeframe_support(exchange_name, timeframe):
        st.sidebar.warning(f"⚠️ {timeframe} may not be supported by {exchange_name}")

    # Data limit
    data_limit = st.sidebar.slider("Data Points", 100, 2000, 500)
else:
    st.sidebar.info("📈 Multi-period analysis uses optimized timeframes for coherent forecasting")
    # Multi-period uses predefined timeframes from MultiPeriodForecaster

# Model selection with comparison option
use_ai_indicators = st.sidebar.checkbox("🎯 AI-Select Indicators", value=False, help="Use AI to select optimal technical indicators")

# Initialize with default values to avoid unbound errors
max_indicators = 5
indicator_categories = ['trend', 'momentum']
if use_ai_indicators:
    max_indicators = st.sidebar.slider("Max Indicators", 3, 8, 5)
    indicator_categories = st.sidebar.multiselect(
        "Indicator Categories",
        ['trend', 'momentum', 'volatility', 'volume', 'trend_strength'],
        default=['trend', 'momentum']
    )

use_auto_select = st.sidebar.checkbox("🎯 Auto-Select Best Model (Walk-Forward)", value=False)

model_name = None  # Ensure model_name is always defined
models_to_compare = []  # Ensure models_to_compare is always defined

if use_auto_select:
    models_to_compare = st.sidebar.multiselect(
        "Models to Compare",
        ['Prophet', 'ARIMA', 'ETS'],
        default=['Prophet', 'ARIMA']
    )
    st.sidebar.info("Will use walk-forward validation to select best model")
else:
    model_name = st.sidebar.selectbox("Choose Model", ['Prophet', 'ARIMA', 'ETS'])

forecast_periods = st.sidebar.slider("Forecast Periods", 5, 100, 30)

# Advanced backtesting options
st.sidebar.header("🔬 Validation Settings")
use_walkforward = st.sidebar.checkbox("Use Walk-Forward Validation", value=False)

# Set default values for walk-forward parameters to avoid unbound errors
wf_min_train = 100
wf_step_size = 1

if use_walkforward:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        wf_step_size = st.sidebar.slider("Step Size", 1, 10, 1)
    with col2:
        wf_min_train = st.sidebar.slider("Min Train Size", 50, 500, 100)

# Technical indicators
st.sidebar.header("📈 Technical Indicators")
show_sma = st.sidebar.checkbox("SMA")
if show_sma:
    st.session_state.sma_window = st.sidebar.number_input("SMA Window", 5, 100, 55)

show_ema = st.sidebar.checkbox("EMA")
if show_ema:
    st.session_state.ema_span = st.sidebar.number_input("EMA Span", 5, 100, 13)

show_rsi = st.sidebar.checkbox("RSI")
if show_rsi:
    st.session_state.rsi_window = st.sidebar.number_input("RSI Window", 7, 50, 14)

# Main action button
start_button = st.sidebar.button("🚀 Generate Analysis", use_container_width=True, type="primary")

# --- Main Content ---
if start_button:
    # Create main layout
    header_col1, header_col2 = st.columns([2, 1])

    with header_col1:
        if analysis_mode == "Multi-Period Coherent":
            st.header(f"📊 Multi-Period Analysis for {symbol}")
        else:
            if use_auto_select:
                st.header(f"📊 Auto-Selected Analysis for {symbol}")
            else:
                st.header(f"📊 Analysis for {symbol} using {model_name}")

    with header_col2:
        st.metric("Source", exchange_name)
        if analysis_mode == "Single Period":
            st.metric("Timeframe", timeframe)

    if analysis_mode == "Multi-Period Coherent":
        # === MULTI-PERIOD ANALYSIS ===
        st.subheader("🔄 Multi-Period Coherent Analysis")

        # Initialize multi-period forecaster
        forecaster = MultiPeriodForecaster()

        # Fetch multi-period data
        with st.spinner("Fetching multi-timeframe data..."):
            multi_data = forecaster.fetch_multi_period_data(
                symbol=symbol,
                source=data_source,
                exchange=exchange_name if data_source in ['auto', 'crypto'] else None
            )

        if not multi_data:
            st.error("❌ Failed to fetch multi-period data")
        else:
            # Display data overview
            st.success(f"✅ Fetched data for {len(multi_data)} timeframes")

            # Data summary table with proper period information
            summary_data = []
            for period, df in multi_data.items():
                if len(df) > 0:
                    latest = df.iloc[-1]
                    previous = df.iloc[-2] if len(df) > 1 else df.iloc[-1]

                    summary_data.append({
                        'Timeframe': period.title(),
                        'Data Points': len(df),
                        'Period Open': f"${latest.get('period_open', latest['open']):.4f}",
                        'Period Close': f"${latest.get('period_close', latest['close']):.4f}",
                        'Period Change': f"{latest.get('period_change', 0.0):+.2f}%",
                        'Period-to-Period': f"{latest.get('period_to_period_change', 0.0):+.2f}%"
                    })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

            # AI Indicator Optimization (if enabled)
            if use_ai_indicators and multi_data:
                st.subheader("🤖 AI Indicator Optimization")

                # Use daily data for indicator optimization
                daily_data = multi_data.get('daily')
                if daily_data is None or daily_data.empty:
                    daily_data = list(multi_data.values())[0]

                with st.spinner("AI analyzing data characteristics and selecting optimal indicators..."):
                    optimizer = AIParameterOptimizer(daily_data)
                    selected_indicators = optimizer.auto_select_indicators(
                        max_indicators=max_indicators,
                        categories=indicator_categories
                    )

                if selected_indicators:
                    st.success(f"🎯 AI selected {len(selected_indicators)} optimal indicators")

                    # Display selected indicators
                    indicator_data = []
                    for indicator, config in selected_indicators.items():
                        indicator_data.append({
                            'Indicator': indicator,
                            'Category': config['category'].title(),
                            'Score': f"{config['score']:.3f}",
                            'Parameters': str(config.get('parameters', {}))
                        })

                    indicators_df = pd.DataFrame(indicator_data)
                    st.dataframe(indicators_df, use_container_width=True)
                else:
                    st.warning("⚠️ AI indicator selection failed, using default indicators")

            # Generate multi-period forecasts
            st.subheader("🔮 Multi-Period Forecasting")

            model_to_use = 'Prophet'  # Default
            if use_auto_select:
                with st.spinner("Selecting best model..."):
                    # Use daily data for model comparison
                    daily_data = multi_data.get('daily')
                    if daily_data is None or daily_data.empty:
                        daily_data = list(multi_data.values())[0]
                    comparison = compare_models_walkforward(
                        daily_data, ['Prophet', 'ARIMA', 'ETS'], 30, 100
                    )
                    if comparison['best_model']:
                        model_to_use = comparison['best_model']
                        st.info(f"🏆 AI selected: {model_to_use}")
            elif model_name:
                model_to_use = model_name

            with st.spinner("Generating multi-period forecasts..."):
                forecasts = forecaster.generate_multi_period_forecasts(
                    multi_data,
                    model_name=model_to_use
                )

            if forecasts:
                st.success(f"✅ Generated forecasts for {len(forecasts)} timeframes")

                # Calculate consistency scores
                with st.spinner("Calculating consistency scores..."):
                    consistency_scores = forecaster.calculate_consistency_scores()

                if consistency_scores:
                    st.subheader("📊 Forecast Consistency Analysis")

                    col1, col2 = st.columns(2)
                    with col1:
                        overall_score = consistency_scores.get('overall', 0)
                        st.metric(
                            "Overall Consistency",
                            f"{overall_score:.1%}",
                            help="Higher scores indicate better alignment between timeframes"
                        )

                    with col2:
                        if overall_score > 0.7:
                            st.success("🟢 High consistency - Strong signal")
                        elif overall_score > 0.4:
                            st.warning("🟡 Moderate consistency - Mixed signals")
                        else:
                            st.error("🔴 Low consistency - Conflicting signals")

                # Generate coherent report
                with st.spinner("Generating coherent analysis report..."):
                    report = forecaster.generate_coherent_report()

                # Multi-Period Forecast Visualization with error boundaries
                st.subheader("📈 Multi-Period Forecast Visualization")

                try:
                    fig = chart_renderer.create_multi_period_chart(multi_data, forecasts, symbol)
                    if fig is not None:
                        chart_renderer.render_safe_chart(
                            fig,
                            "Multi-Period Forecast Analysis",
                            use_container_width=True
                        )
                    else:
                        st.error("❌ Chart creation returned no figure.")
                except ChartRenderError as e:
                    st.error(f"❌ Multi-period chart creation failed: {e}")
                    st.info("💡 Try adjusting the data range or timeframe selection")
                except Exception as e:
                    st.error(f"❌ Unexpected chart error: {e}")
                    st.info("💡 Please refresh the page and try again")

                # Summary and Recommendations
                st.subheader("📋 Multi-Period Summary & Recommendations")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Trend Analysis by Timeframe:**")
                    summary = report['summary']
                    for period in ['monthly', 'weekly', 'daily', 'hourly']:
                        if period in summary['trend_direction']:
                            trend_info = summary['trend_direction'][period]
                            direction = "📈" if trend_info['direction'] == 'up' else "📉"
                            st.write(f"{direction} {period.title()}: {trend_info['change_percent']:+.1f}%")

                with col2:
                    st.write("**AI Recommendations:**")
                    for rec in report['recommendations']:
                        st.write(f"• {rec}")

                # Optional: Multi-period AI narrative
                with st.spinner("AI Analyst is synthesizing multi-period insights..."):
                    try:
                        ai_multi = get_multi_period_ai_analysis(
                            symbol=symbol,
                            multi_historical=multi_data,
                            multi_forecasts=forecasts,
                            report=report,
                            exchange_info={"exchange": exchange_name}
                        )
                        st.info(ai_multi)
                    except Exception as e:
                        st.info(f"AI multi-period analysis unavailable: {e}")
            else:
                st.error("❌ Failed to generate multi-period forecasts")

    else:
        # === SINGLE PERIOD ANALYSIS ===
        # Data fetching with yfinance
        with st.spinner(f"Fetching {symbol} data..."):
            data = fetch_universal_data(symbol, timeframe, data_limit)
            successful_exchange = 'yfinance' if data is not None else None

        if data is None or data.empty:
            st.error(f"❌ Could not fetch data for {symbol}")
            st.info("💡 Try a different symbol or check if the symbol exists on Yahoo Finance")
        else:
            # Data summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Data Points", len(data))
            with col2:
                st.metric("Date Range", f"{len(data)} {timeframe} periods")
            with col3:
                latest_price = data['close'].iloc[-1]
                price_change = ((latest_price - data['close'].iloc[-2]) / data['close'].iloc[-2] * 100) if len(data) > 1 else 0
                st.metric("Latest Price", f"${latest_price:.4f}", f"{price_change:+.2f}%")
            with col4:
                st.metric("Volatility", f"{data['close'].pct_change().std()*100:.2f}%")

        # Calculate technical indicators (if enabled)
        if show_sma and 'sma_window' in st.session_state and data is not None:
            data = calculate_sma(data, st.session_state.sma_window)
        if show_ema and 'ema_span' in st.session_state and data is not None:
            data = calculate_ema(data, st.session_state.ema_span)
        if show_rsi and 'rsi_window' in st.session_state and data is not None:
            data = calculate_rsi(data, st.session_state.rsi_window)

        # Model selection and forecasting
        if use_auto_select and models_to_compare:
            st.subheader("🔄 Model Comparison & Selection")

            if data is not None:
                with st.spinner("Running walk-forward validation on all models..."):
                    comparison_results = compare_models_walkforward(
                        data.copy(), models_to_compare, forecast_periods, wf_min_train
                    )

                if comparison_results['best_model']:
                    best_model = comparison_results['best_model']
                    best_rmse = comparison_results['best_rmse']

                    # Display comparison results
                    st.success(f"🏆 Best Model: **{best_model}** (RMSE: {best_rmse:.6f})")

                    # Show model comparison table
                    summary_df = pd.DataFrame([
                        {'Model': model, 'RMSE': rmse}
                        for model, rmse in comparison_results['summary'].items()
                    ]).sort_values('RMSE')

                    st.dataframe(summary_df, use_container_width=True)

                    # Use best model for final forecast
                    model_name = best_model
                else:
                    st.error("❌ Model comparison failed")
                    model_name = models_to_compare[0]  # Fallback
            else:
                st.error("❌ No data available for model comparison.")

        # Generate final forecast
        with st.spinner(f"Generating forecast with {model_name}..."):
            safe_model_name = model_name if isinstance(model_name, str) and model_name else "Prophet"
            if data is not None:
                forecast = generate_forecast(data.copy(), safe_model_name, forecast_periods)
            else:
                forecast = None

        # Generate final forecast chart with error boundaries
        if forecast is None or forecast.empty:
            st.error("❌ Forecast generation failed")
        else:
            st.success("✅ Forecast generated successfully!")

            # Prepare technical indicators context for chart renderer
            tech_indicators = {}
            if show_sma and 'sma_window' in st.session_state:
                tech_indicators[f'SMA_{st.session_state.sma_window}'] = st.session_state.sma_window
            if show_ema and 'ema_span' in st.session_state:
                tech_indicators[f'EMA_{st.session_state.ema_span}'] = st.session_state.ema_span

            # Create and render chart with error boundaries
            try:
                if data is not None:
                    fig = chart_renderer.create_single_period_chart(
                        data=data,
                        forecast=forecast,
                        symbol=symbol,
                        model_name=safe_model_name,
                        timeframe=timeframe,
                        technical_indicators=tech_indicators if tech_indicators else None,
                        show_rsi=show_rsi
                    )

                    if fig is not None:
                        chart_renderer.render_safe_chart(
                            fig,
                            f"{symbol} {safe_model_name} Forecast",
                            use_container_width=True
                        )
                    else:
                        st.error("❌ Chart creation returned no figure.")
                else:
                    st.error("❌ No data available for chart creation.")

            except ChartRenderError as e:
                st.error(f"❌ Chart creation failed: {e}")
                st.info("💡 Try adjusting the data range or technical indicator settings")
            except Exception as e:
                st.error(f"❌ Unexpected chart error: {e}")
                st.info("💡 Please refresh the page and try again")

            # Walk-forward validation results
            wf_results = None  # Ensure wf_results is always defined
            if use_walkforward:
                st.subheader("📊 Walk-Forward Validation Results")

                with st.spinner("Running walk-forward validation..."):
                    safe_model_name = model_name if isinstance(model_name, str) and model_name else "Prophet"
                    if data is not None:
                        wf_results = walk_forward_validation(
                            data.copy(), safe_model_name, forecast_periods, wf_min_train, wf_step_size,
                            'expanding', None, timeframe
                        )
                    else:
                        wf_results = {'error': 'No data available for walk-forward validation.'}

                if 'error' not in wf_results:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Validation Folds", wf_results['num_folds'])
                        mean_rmse = wf_results['metrics']['mean_RMSE'] if isinstance(wf_results['metrics'], dict) and 'mean_RMSE' in wf_results['metrics'] else 0
                        st.metric("Mean RMSE", f"{mean_rmse:.6f}")

                    with col2:
                        mean_mae = wf_results['metrics']['mean_MAE'] if isinstance(wf_results['metrics'], dict) and 'mean_MAE' in wf_results['metrics'] else 0
                        mean_mape = wf_results['metrics']['mean_MAPE'] if isinstance(wf_results['metrics'], dict) and 'mean_MAPE' in wf_results['metrics'] else 0
                        st.metric("Mean MAE", f"{mean_mae:.6f}")
                        st.metric("Mean MAPE", f"{mean_mape:.2f}%")

                    with col3:
                        std_rmse = wf_results['metrics']['std_RMSE'] if isinstance(wf_results['metrics'], dict) and 'std_RMSE' in wf_results['metrics'] else 0
                        combined_rmse = wf_results['metrics']['combined_RMSE'] if isinstance(wf_results['metrics'], dict) and 'combined_RMSE' in wf_results['metrics'] else 0
                        st.metric("Std RMSE", f"{std_rmse:.6f}")
                        st.metric("Combined RMSE", f"{combined_rmse:.6f}")

                    # Fold performance chart
                    fold_df = pd.DataFrame(wf_results['fold_metrics'])

                    fig_validation = go.Figure()
                    fig_validation.add_trace(
                        go.Scatter(
                            x=list(range(1, len(fold_df) + 1)),
                            y=fold_df['RMSE'],
                            mode='lines+markers',
                            name='RMSE per Fold',
                            line=dict(color='red')
                        )
                    )

                    fig_validation.update_layout(
                        title="Walk-Forward Validation: RMSE per Fold",
                        xaxis_title="Fold Number",
                        yaxis_title="RMSE",
                        height=400
                    )

                    st.plotly_chart(fig_validation, use_container_width=True)
                else:
                    st.error(f"Validation failed: {wf_results['error']}")

            # AI Analysis with Enhanced Context
            st.subheader("🤖 AI-Generated Analysis")

            # Market Context Analysis
            with st.spinner("Analyzing market context..."):
                market_analyzer = MarketIndicatorsAnalyzer()

                # Determine market type and fetch context
                market_type = 'crypto' if '/' in symbol or symbol.endswith('USDT') else 'traditional'
                market_context = market_analyzer.fetch_market_context(
                    symbol=symbol,
                    market_type=market_type,
                    timeframe=timeframe,
                    limit=100,
                    exchange=exchange_name if market_type == 'crypto' else None
                )

                # Generate market context summary
                market_summary = None
                if data is not None:
                    market_summary = market_analyzer.generate_market_context_summary(symbol, data)

            # Display market context insights
            if market_summary and market_summary.get('insights'):
                st.subheader("📊 Market Context & Risk Assessment")

                col1, col2, col3 = st.columns(3)

                with col1:
                    regime = market_summary['market_regime']
                    st.metric("Market Regime", regime['regime_type'].replace('_', ' ').title())
                    st.metric("Risk Level", regime['risk_level'].title())
                    st.metric("Volatility", regime['volatility_state'].title())

                with col2:
                    st.write("**Key Correlations:**")
                    correlations = market_summary.get('correlations', {})
                    for indicator, corr in list(correlations.items())[:3]:
                        direction = "↗️" if corr > 0 else "↘️"
                        st.write(f"{direction} {indicator}: {corr:.3f}")

                with col3:
                    trend = regime['trend_direction']
                    trend_emoji = "📈" if trend == 'bullish' else "📉" if trend == 'bearish' else "↔️"
                    st.metric("Market Trend", f"{trend_emoji} {trend.title()}")

                # Market insights
                if market_summary['insights']:
                    st.write("**Market Insights:**")
                    for insight in market_summary['insights']:
                        st.write(f"• {insight}")

                # Risk factors and opportunities
                col_risk, col_opp = st.columns(2)

                with col_risk:
                    risks = market_summary.get('risk_factors', [])
                    if risks:
                        st.write("**⚠️ Risk Factors:**")
                        for risk in risks:
                            st.write(f"• {risk}")

                with col_opp:
                    opportunities = market_summary.get('opportunities', [])
                    if opportunities:
                        st.write("**💡 Opportunities:**")
                        for opp in opportunities:
                            st.write(f"• {opp}")

            with st.spinner("AI Analyst is analyzing the forecast..."):

                # Build technical indicators context
                technical_context = {}
                if data is not None and show_sma and f'SMA_{st.session_state.sma_window}' in data.columns:
                    technical_context[f'SMA_{st.session_state.sma_window}'] = data[f'SMA_{st.session_state.sma_window}'].iloc[-1]
                if data is not None and show_rsi and 'RSI' in data.columns:
                    rsi_value = data['RSI'].iloc[-1]
                    technical_context['RSI'] = rsi_value
                    if rsi_value > 70:
                        technical_context['RSI_Signal'] = 'Overbought'
                    elif rsi_value < 30:
                        technical_context['RSI_Signal'] = 'Oversold'
                    else:
                        technical_context['RSI_Signal'] = 'Neutral'

                # Build exchange context
                exchange_context = {
                    'exchange': exchange_name,
                    'timeframe': timeframe,
                    'data_points': len(data) if data is not None else 0
                }

                # Get validation results if walk-forward was used
                validation_context = None
                comparison_results = None
                comparison_results = comparison_results if 'comparison_results' in locals() else None
                if use_walkforward and 'wf_results' in locals() and wf_results is not None and 'error' not in wf_results:
                    validation_context = wf_results
                elif use_auto_select and comparison_results is not None:
                    validation_context = comparison_results.get('comparison_results', {}).get(model_name)

                # Call enhanced AI analysis with market context
                ai_summary = get_ai_analysis(
                    symbol=symbol,
                    historical_data=data if data is not None else pd.DataFrame(),
                    forecast_data=forecast,
                    model_name=model_name,
                    validation_results=validation_context,
                    technical_indicators=technical_context if technical_context else None,
                    exchange_info=exchange_context,
                    market_context=market_summary if 'market_summary' in locals() else None
                )
                st.info(ai_summary)

            # Forecast data table
            st.subheader("📋 Forecast Data")
            forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods)
            forecast_display['ds'] = forecast_display['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(forecast_display, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("💡 **Tips**: Use smaller timeframes (1h, 4h) for short-term trading signals, daily/weekly for longer-term analysis.")

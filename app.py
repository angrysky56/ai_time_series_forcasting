import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from src.data_fetcher import (
    fetch_crypto_data,
    try_multiple_exchanges,
    discover_symbols,
    get_available_exchanges,
    validate_timeframe_support
)
from src.forecasting import generate_forecast
from src.llm_integration import get_ai_analysis
from src.technical_analysis import calculate_sma, calculate_ema, calculate_rsi
from src.backtesting import walk_forward_validation, compare_models_walkforward

st.set_page_config(layout="wide", page_title="AI Crypto Forecaster")
st.title("🤖 AI-Assisted Crypto Forecaster")
st.markdown("*Enhanced with Walk-Forward Validation & Multi-Exchange Support*")

# Initialize session state
if 'discovered_symbols' not in st.session_state:
    st.session_state.discovered_symbols = []
if 'selected_exchange' not in st.session_state:
    st.session_state.selected_exchange = 'coinbase'

# --- Sidebar Controls ---
st.sidebar.header("📊 Data Configuration")

# Exchange selection
available_exchanges = get_available_exchanges()
exchange_name = st.sidebar.selectbox(
    "Exchange",
    available_exchanges,
    index=available_exchanges.index(st.session_state.selected_exchange)
    if st.session_state.selected_exchange in available_exchanges else 0
)

# Symbol discovery
if st.sidebar.button("🔍 Discover Symbols", help="Find available trading pairs"):
    with st.spinner(f"Discovering symbols from {exchange_name}..."):
        symbols = discover_symbols(exchange_name, limit=500)
        st.session_state.discovered_symbols = symbols
        st.session_state.selected_exchange = exchange_name
    if symbols:
        st.sidebar.success(f"Found {len(symbols)} symbols")
    else:
        st.sidebar.error("No symbols found")

# Symbol input/selection
if st.session_state.discovered_symbols:
    symbol_input_method = st.sidebar.radio("Symbol Input", ["Manual", "From Discovery"])
    if symbol_input_method == "From Discovery":
        symbol = st.sidebar.selectbox("Select Symbol", st.session_state.discovered_symbols)
    else:
        symbol = st.sidebar.text_input("Enter Symbol", "BTC/USDT").upper()
else:
    symbol = st.sidebar.text_input("Enter Symbol", "BTC/USDT").upper()

# Timeframe selection with validation
timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
timeframe = st.sidebar.selectbox("Timeframe", timeframes, index=6)  # Default to 1d

# Validate timeframe support
if not validate_timeframe_support(exchange_name, timeframe):
    st.sidebar.warning(f"⚠️ {timeframe} may not be supported by {exchange_name}")

# Data limit
data_limit = st.sidebar.slider("Data Points", 100, 2000, 500)

st.sidebar.header("🔮 Forecasting")

# Model selection with comparison option
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
        if use_auto_select:
            st.header(f"📊 Auto-Selected Analysis for {symbol}")
        else:
            st.header(f"📊 Analysis for {symbol} using {model_name}")

    with header_col2:
        st.metric("Exchange", exchange_name)
        st.metric("Timeframe", timeframe)

    # Data fetching with fallback
    with st.spinner(f"Fetching {symbol} data from {exchange_name}..."):
        if exchange_name == 'auto':
            data, successful_exchange = try_multiple_exchanges(symbol, timeframe, data_limit)
            if successful_exchange:
                st.success(f"✅ Data fetched from {successful_exchange}")
                exchange_name = successful_exchange
        else:
            data = fetch_crypto_data(symbol, timeframe, data_limit, exchange_name)
            successful_exchange = exchange_name if data is not None else None

    if data is None or data.empty:
        st.error(f"❌ Could not fetch data for {symbol} from {exchange_name}")
        st.info("💡 Try a different symbol or exchange")
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

        # Calculate technical indicators
        if show_sma and 'sma_window' in st.session_state:
            data = calculate_sma(data, st.session_state.sma_window)
        if show_ema and 'ema_span' in st.session_state:
            data = calculate_ema(data, st.session_state.ema_span)
        if show_rsi and 'rsi_window' in st.session_state:
            data = calculate_rsi(data, st.session_state.rsi_window)

        # Model selection and forecasting
        if use_auto_select and models_to_compare:
            st.subheader("🔄 Model Comparison & Selection")

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

        # Generate final forecast
        with st.spinner(f"Generating forecast with {model_name}..."):
            safe_model_name = model_name if isinstance(model_name, str) and model_name else "Prophet"
            forecast = generate_forecast(data.copy(), safe_model_name, forecast_periods)

        if forecast is None or forecast.empty:
            st.error("❌ Forecast generation failed")
        else:
            st.success("✅ Forecast generated successfully!")

            # Create main chart
            fig = make_subplots(
                rows=2 if show_rsi else 1,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3] if show_rsi else [1.0]
            )

            # Price data
            fig.add_trace(
                go.Candlestick(
                    x=data['timestamp'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name='Price'
                ),
                row=1, col=1
            )

            # Forecast
            fig.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name=f'{model_name} Forecast',
                    line=dict(color='#ff7f0e', width=3)
                ),
                row=1, col=1
            )

            # Confidence intervals if available
            if 'yhat_lower' in forecast.columns and forecast['yhat_lower'].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255, 127, 14, 0.2)',
                        showlegend=False
                    ),
                    row=1, col=1
                )

            # Technical indicators
            if show_sma:
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data[f'SMA_{st.session_state.sma_window}'],
                        mode='lines',
                        name=f'SMA {st.session_state.sma_window}',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
            if show_ema and 'ema_span' in st.session_state:
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data[f'EMA_{st.session_state.ema_span}'],
                        mode='lines',
                        name=f'EMA {st.session_state.ema_span}',
                        line=dict(color='green')
                    ),
                    row=1, col=1
                )
            if show_rsi and 'RSI' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple')
                    ),
                    row=2, col=1
                )
                fig.add_shape(
                    type="line",
                    x0=data['timestamp'].iloc[0],
                    x1=data['timestamp'].iloc[-1],
                    y0=70,
                    y1=70,
                    line=dict(dash="dash", color="red"),
                    xref="x2",
                    yref="y2"
                )
                fig.add_shape(
                    type="line",
                    x0=data['timestamp'].iloc[0],
                    x1=data['timestamp'].iloc[-1],
                    y0=30,
                    y1=30,
                    line=dict(dash="dash", color="green"),
                    xref="x2",
                    yref="y2"
                )
                fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

            # Update layout
            fig.update_layout(
                title=f'{symbol} - {model_name} Forecast ({timeframe} timeframe)',
                xaxis_rangeslider_visible=False,
                height=700,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
            if show_rsi:
                fig.update_xaxes(showticklabels=False, row=1, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # Walk-forward validation results
            wf_results = None  # Ensure wf_results is always defined
            if use_walkforward:
                st.subheader("📊 Walk-Forward Validation Results")

                with st.spinner("Running walk-forward validation..."):
                    safe_model_name = model_name if isinstance(model_name, str) and model_name else "Prophet"
                    wf_results = walk_forward_validation(
                        data.copy(), safe_model_name, forecast_periods, wf_min_train, wf_step_size
                    )

                if 'error' not in wf_results:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Validation Folds", wf_results['num_folds'])
                        st.metric("Mean RMSE", f"{wf_results['metrics']['mean_RMSE']:.6f}")

                    with col2:
                        st.metric("Mean MAE", f"{wf_results['metrics']['mean_MAE']:.6f}")
                        st.metric("Mean MAPE", f"{wf_results['metrics']['mean_MAPE']:.2f}%")

                    with col3:
                        st.metric("Std RMSE", f"{wf_results['metrics']['std_RMSE']:.6f}")
                        st.metric("Combined RMSE", f"{wf_results['metrics']['combined_RMSE']:.6f}")

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
            with st.spinner("AI Analyst is analyzing the forecast..."):

                # Build technical indicators context
                technical_context = {}
                if show_sma and f'SMA_{st.session_state.sma_window}' in data.columns:
                    technical_context[f'SMA_{st.session_state.sma_window}'] = data[f'SMA_{st.session_state.sma_window}'].iloc[-1]
                if show_rsi and 'RSI' in data.columns:
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
                    'data_points': len(data)
                }

                # Get validation results if walk-forward was used
                validation_context = None
                comparison_results = None
                comparison_results = comparison_results if 'comparison_results' in locals() else None
                if use_walkforward and 'wf_results' in locals() and wf_results is not None and 'error' not in wf_results:
                    validation_context = wf_results
                elif use_auto_select and comparison_results is not None:
                    validation_context = comparison_results.get('comparison_results', {}).get(model_name)

                # Call enhanced AI analysis
                ai_summary = get_ai_analysis(
                    symbol=symbol,
                    historical_data=data,
                    forecast_data=forecast,
                    model_name=model_name,
                    validation_results=validation_context,
                    technical_indicators=technical_context if technical_context else None,
                    exchange_info=exchange_context
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

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.data_fetcher import fetch_crypto_data
from src.forecasting import generate_forecast
from src.llm_integration import get_ai_analysis
from src.technical_analysis import calculate_sma, calculate_ema, calculate_rsi

st.set_page_config(layout="wide")
st.title("AI-Assisted Crypto Forecaster")

# --- User Inputs in Sidebar ---
st.sidebar.header("Controls")
symbol = st.sidebar.text_input("Enter Crypto Symbol (e.g., BTC/USDT)", "BTC/USDT").upper()
model_name = st.sidebar.selectbox("Choose Forecast Model", ['Prophet', 'ARIMA', 'ETS'])
forecast_periods = st.sidebar.slider("Forecast Period (Days)", 7, 365, 90)

st.sidebar.header("Technical Indicators")
show_sma = st.sidebar.checkbox("Show Simple Moving Average (SMA)")
sma_window = st.sidebar.number_input("SMA Window", 5, 100, 20)
show_ema = st.sidebar.checkbox("Show Exponential Moving Average (EMA)")
ema_span = st.sidebar.number_input("EMA Span", 5, 100, 20)
show_rsi = st.sidebar.checkbox("Show Relative Strength Index (RSI)")
rsi_window = st.sidebar.number_input("RSI Window", 7, 50, 14)

start_button = st.sidebar.button("Generate Forecast", use_container_width=True)

# Caching the data fetching function
@st.cache_data
def cached_fetch_data(symbol, timeframe, limit):
    return fetch_crypto_data(symbol, timeframe, limit)

if start_button:
    st.header(f"Analysis for {symbol} using {model_name}")
    
    with st.spinner(f"Fetching historical data for {symbol}..."):
        data = cached_fetch_data(symbol, '1d', 365)

    if data is None or data.empty:
        st.error(f"Could not fetch data for {symbol}. Please check the symbol or try another exchange.")
    else:
        st.success(f"Successfully fetched {len(data)} days of historical data.")

        # --- Calculate Technical Indicators ---
        if show_sma:
            data = calculate_sma(data, sma_window)
        if show_ema:
            data = calculate_ema(data, ema_span)
        if show_rsi:
            data = calculate_rsi(data, rsi_window)

        # --- Generate Forecast ---
        with st.spinner(f"AI is generating the forecast with {model_name}..."):
            forecast = generate_forecast(data.copy(), model_name, forecast_periods)
        st.success("Forecast generated successfully!")

        # --- Plotting with Plotly ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.05, row_heights=[0.7, 0.3])

        fig.add_trace(go.Candlestick(x=data['timestamp'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name='Historical Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='#ff7f0e', width=3)), row=1, col=1)
        
        # Conditionally plot confidence intervals if they exist
        if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns and forecast['yhat_lower'].notna().all():
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 127, 14, 0.2)', showlegend=False), row=1, col=1)

        if show_sma:
            fig.add_trace(go.Scatter(x=data['timestamp'], y=data[f'SMA_{sma_window}'], mode='lines', name=f'SMA {sma_window}'), row=1, col=1)
        if show_ema:
            fig.add_trace(go.Scatter(x=data['timestamp'], y=data[f'EMA_{ema_span}'], mode='lines', name=f'EMA {ema_span}'), row=1, col=1)
        
        if show_rsi:
            fig.add_trace(go.Scatter(x=data['timestamp'], y=data['RSI'], mode='lines', name='RSI'), row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(
            title_text=f'{symbol} Price Forecast & Analysis ({model_name})',
            xaxis_rangeslider_visible=False,
            yaxis_title_text="Price (USD)",
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(showticklabels=False, row=1, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # --- AI Analysis ---
        st.subheader("AI-Generated Analysis")
        with st.spinner("AI Analyst is thinking..."):
            ai_summary = get_ai_analysis(symbol, data, forecast)
            st.info(ai_summary)

        # --- Forecast Data Table ---
        st.subheader("Forecast Data")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods))
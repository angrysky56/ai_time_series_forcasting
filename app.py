import streamlit as st
import plotly.graph_objects as go
from src.data_fetcher import fetch_crypto_data
from src.forecasting import generate_forecast
from src.llm_integration import get_ai_analysis

st.title("AI-Assisted Crypto Forecaster")

# --- User Inputs in Sidebar ---
st.sidebar.header("Controls")
symbol = st.sidebar.text_input("Enter Crypto Symbol (e.g., BTC/USDT)", "BTC/USDT").upper()
forecast_periods = st.sidebar.slider("Forecast Period (Days)", 7, 365, 90)
start_button = st.sidebar.button("Generate Forecast", use_container_width=True)

# Caching the data fetching function
@st.cache_data
def cached_fetch_data(symbol, timeframe, limit):
    return fetch_crypto_data(symbol, timeframe, limit)

if start_button:
    st.header(f"Analysis for {symbol}")

    # --- Fetch Data ---
    with st.spinner(f"Fetching historical data for {symbol}..."):
        data = cached_fetch_data(symbol, '1d', 365)

    if data is None or data.empty:
        st.error(f"Could not fetch data for {symbol}. Please check the symbol or try another exchange.")
    else:
        st.success(f"Successfully fetched {len(data)} days of historical data.")

        # --- Generate Forecast ---
        with st.spinner("AI is generating the forecast..."):
            forecast = generate_forecast(data, forecast_periods)

        if forecast is None or forecast.empty:
            st.error("Forecast generation failed. Please check your data or try again.")
        else:
            st.success("Forecast generated successfully!")

            # --- Plotting with Plotly ---
            fig = go.Figure()

            fig.add_trace(go.Candlestick(x=data['timestamp'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name='Historical Price'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='#ff7f0e', width=3)))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, name='Upper Confidence'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 127, 14, 0.2)', showlegend=False, name='Lower Confidence'))

            fig.update_layout(
                title=f'{symbol} Price Forecast',
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True)

            # --- AI Analysis ---
            st.subheader("AI-Generated Analysis")
            with st.spinner("AI Analyst is thinking..."):
                ai_summary = get_ai_analysis(symbol, data, forecast)
                st.info(ai_summary)

            # --- Forecast Data Table ---
            st.subheader("Forecast Data")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods))
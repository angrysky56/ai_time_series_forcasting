import requests
import pandas as pd

# --- Configuration for LM Studio API ---
# Make sure your local server is running and accessible at this address.
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"

def get_ai_analysis(symbol: str, historical_data: pd.DataFrame, forecast_data: pd.DataFrame):
    """
    Generates a qualitative analysis of the forecast using a local LLM.

    Args:
        symbol (str): The cryptocurrency symbol.
        historical_data (pd.DataFrame): The historical price data.
        forecast_data (pd.DataFrame): The forecast data from Prophet.

    Returns:
        str: The AI-generated analysis, or an error message.
    """
    if forecast_data is None or forecast_data.empty or historical_data is None or historical_data.empty:
        return "Not enough data to generate an AI analysis."

    # --- Prompt Engineering ---
    # Extract key data points for the prompt
    last_known_price = historical_data['close'].iloc[-1]
    forecast_end_price = forecast_data['yhat'].iloc[-1]
    forecast_period = len(forecast_data) - len(historical_data)
    percentage_change = ((forecast_end_price - last_known_price) / last_known_price) * 100

    prompt = f"""
    Act as a neutral and objective cryptocurrency market analyst. Do not give financial advice.
    Your task is to provide a brief, high-level analysis of a price forecast for {symbol}.

    Here is the data:
    - The last known price is: ${last_known_price:,.2f}
    - The forecast predicts a price of: ${forecast_end_price:,.2f} in {forecast_period} days.
    - This represents a projected change of: {percentage_change:.2f}%.

    Based on this data, please provide a short analysis covering the following points:
    1.  **Trend Summary:** Briefly describe the predicted trend (e.g., bullish, bearish, sideways, volatile).
    2.  **Key Price Levels:** Mention the current price and the predicted future price.
    3.  **Concluding Remark:** End with a neutral concluding sentence about the nature of market forecasts.

    Keep the entire analysis to about 3-4 sentences. Be objective and data-driven.
    Example of a good response:
    'The forecast for BTC/USDT suggests a bullish trend over the next 90 days, with the price projected to move from its current level of $60,000.00 to $75,000.00. This represents a potential increase of 25.00%. As with all market predictions, this forecast is based on historical data and does not account for future real-world events.'
    """

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "local-model", # This can be any name, LM Studio doesn't enforce it
        "messages": [
            {"role": "system", "content": "You are a cryptocurrency market analyst providing objective insights based on forecast data."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
    }

    try:
        response = requests.post(LM_STUDIO_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        result = response.json()
        ai_response = result['choices'][0]['message']['content']
        return ai_response

    except requests.exceptions.RequestException as e:
        return f"Error connecting to the LLM API: {e}. Please ensure LM Studio is running and the server is active."
    except (KeyError, IndexError) as e:
        return f"Error parsing the LLM API response: {e}. The response format may have changed."
    except Exception as e:
        return f"An unexpected error occurred during AI analysis: {e}"

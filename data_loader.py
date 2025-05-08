# data_loader.py
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, date
import pytz # For timezone handling
import requests # For API calls

# --- Finnhub API Configuration ---
# Base URL for Finnhub API
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# Function to get API key from Streamlit secrets
def get_finnhub_api_key():
    """Retrieves the Finnhub API key from Streamlit secrets."""
    try:
        return st.secrets["FINNHUB_API_KEY"]
    except (KeyError, FileNotFoundError): # FileNotFoundError for local dev without secrets.toml
        st.error("🚨 Finnhub API key not found. Please add `FINNHUB_API_KEY = \"YOUR_KEY\"` to your Streamlit secrets (e.g., .streamlit/secrets.toml).")
        return None

@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_economic_calendar_from_finnhub(api_key, from_date_str, to_date_str):
    """
    Fetches economic calendar data from Finnhub API for a given date range.

    Args:
        api_key (str): Your Finnhub API key.
        from_date_str (str): Start date in YYYY-MM-DD format.
        to_date_str (str): End date in YYYY-MM-DD format.

    Returns:
        pd.DataFrame: DataFrame containing economic events, or empty DataFrame on error.
    """
    if not api_key:
        return pd.DataFrame()

    params = {
        "token": api_key,
        "from": from_date_str,
        "to": to_date_str
    }
    try:
        response = requests.get(f"{FINNHUB_BASE_URL}/calendar/economic", params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()

        if not data or "economicCalendar" not in data or not data["economicCalendar"]:
            st.info(f"No economic events found on Finnhub for {from_date_str} to {to_date_str}.")
            return pd.DataFrame()

        df = pd.DataFrame(data["economicCalendar"])

        # --- Data Transformation and Cleaning ---
        # Rename columns to match existing app structure
        column_mapping = {
            "time": "Timestamp", # This is Unix timestamp (seconds)
            "country": "Currency", # Finnhub uses country, map to currency (e.g., US -> USD) - needs refinement
            "event": "EventName",
            "impact": "Impact", # Finnhub impact: "low", "medium", "high"
            "estimate": "Forecast",
            "actual": "Actual",
            "prev": "Previous"
        }
        df.rename(columns=column_mapping, inplace=True)

        # Convert Unix timestamp to datetime objects (UTC)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s', utc=True)
        
        # Map country codes to currency codes (simplified - needs a robust mapping)
        country_to_currency_map = {
            "US": "USD", "CA": "CAD", "GB": "GBP", "EU": "EUR", "JP": "JPY", "AU": "AUD", "NZ": "NZD", "CH": "CHF", "CN": "CNY"
            # Add more mappings as needed
        }
        df['Currency'] = df['Currency'].map(country_to_currency_map).fillna(df['Currency']) # Keep original if no map

        # Ensure impact is capitalized ("Low", "Medium", "High")
        if 'Impact' in df.columns:
            df['Impact'] = df['Impact'].astype(str).str.capitalize()


        # Select and order relevant columns
        relevant_columns = ["Timestamp", "Currency", "EventName", "Impact", "Previous", "Forecast", "Actual"]
        df = df[[col for col in relevant_columns if col in df.columns]] # Keep only existing relevant columns

        # Ensure numeric columns are numeric
        numeric_cols_finnhub = ['Previous', 'Forecast', 'Actual']
        for col in numeric_cols_finnhub:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else: # Add column with NaNs if missing from API response
                df[col] = np.nan
        
        df['id'] = df.index # Add unique ID
        return df.sort_values(by='Timestamp').reset_index(drop=True)

    except requests.exceptions.RequestException as e:
        st.error(f"🚨 API Request Error fetching data from Finnhub: {e}")
        return pd.DataFrame()
    except ValueError as e: # For JSON decoding errors
        st.error(f"🚨 Error decoding Finnhub API response: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"🚨 An unexpected error occurred while processing Finnhub data: {e}")
        # import traceback
        # st.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

# This function now orchestrates fetching from Finnhub
@st.cache_data(ttl=1800) # Cache combined data for 30 minutes
def load_economic_data():
    """
    Loads economic calendar data, primarily from Finnhub API.
    Fetches data for the current week (Monday to Sunday).
    """
    api_key = get_finnhub_api_key()
    if not api_key:
        st.warning("API key not available. Displaying no live data.")
        return pd.DataFrame() # Return empty if no API key

    # Determine date range for fetching (e.g., current week or next few days)
    today = date.today()
    # For example, fetch from last Monday to next Sunday to cover a good range
    start_of_week = today - timedelta(days=today.weekday()) 
    end_of_week = start_of_week + timedelta(days=13) # Fetch for two weeks

    from_date_str = start_of_week.strftime("%Y-%m-%d")
    to_date_str = end_of_week.strftime("%Y-%m-%d")
    
    st.info(f"Fetching live economic data from Finnhub for {from_date_str} to {to_date_str}...")
    df_finnhub = fetch_economic_calendar_from_finnhub(api_key, from_date_str, to_date_str)

    if df_finnhub.empty:
        st.warning("Could not retrieve live data from Finnhub. Check API key and network.")
    
    return df_finnhub

@st.cache_data
def load_historical_data(event_name):
    """
    Loads sample historical data. This part is NOT yet connected to Finnhub
    and still uses sample data.
    """
    today_date = datetime.now().date()
    all_historical_data = {
        "Non-Farm Employment Change": pd.DataFrame({
            'Date': [today_date - timedelta(days=30*i) for i in range(6, 0, -1)],
            'Actual': [150.0, 220.0, 180.0, 250.0, 160.0, 175.0],
            'Forecast': [160.0, 200.0, 190.0, 230.0, 180.0, 185.0],
            'Previous': [140.0, 150.0, 220.0, 180.0, 250.0, 160.0]
        }),
        "Unemployment Rate": pd.DataFrame({
            'Date': [today_date - timedelta(days=30*i) for i in range(6, 0, -1)],
            'Actual': [4.0, 3.8, 3.9, 3.7, 3.9, 3.9],
            'Forecast': [3.9, 3.8, 3.9, 3.8, 3.9, 3.9],
            'Previous': [4.1, 4.0, 3.8, 3.9, 3.7, 3.9]
        }),
         "Core CPI m/m": pd.DataFrame({
            'Date': [today_date - timedelta(days=30*i) for i in range(6, 0, -1)],
            'Actual': [0.2, 0.4, 0.3, 0.3, 0.5, 0.3],
            'Forecast': [0.3, 0.3, 0.3, 0.4, 0.4, 0.3],
            'Previous': [0.1, 0.2, 0.4, 0.3, 0.3, 0.5]
        })
    }
    for key in all_historical_data:
        if key.lower() in event_name.lower():
            df = all_historical_data[key].copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df
    return pd.DataFrame()

if __name__ == '__main__':
    # For local testing, you might need to mock st.secrets or provide a key directly
    # This is a simplified test; Streamlit context is needed for full st.secrets functionality
    print("Attempting to load economic data (requires FINNHUB_API_KEY in secrets or mocked):")
    # To test locally without Streamlit running, you'd need to manually provide a key
    # or mock st.secrets. For now, this will likely show an error if key isn't available.
    # For a real test, run `streamlit run app.py` after setting up secrets.
    
    # Mocking st.secrets for local test if needed:
    # class MockSecrets:
    #     def __getitem__(self, key):
    #         if key == "FINNHUB_API_KEY":
    #             return "YOUR_ACTUAL_FINNHUB_KEY_FOR_TESTING" # Replace with your key for local test
    #         raise KeyError(key)
    # st.secrets = MockSecrets()

    live_data = load_economic_data()
    if not live_data.empty:
        print("\nLive Economic Data Sample (from Finnhub if successful):")
        print(live_data[['Timestamp', 'Currency', 'EventName', 'Impact', 'Forecast']].head())
        if pd.notna(live_data['Timestamp'].iloc[0]):
            print(f"\nFirst timestamp object type: {type(live_data['Timestamp'].iloc[0])}")
            print(f"First timestamp timezone info: {live_data['Timestamp'].iloc[0].tzinfo}")
    else:
        print("\nFailed to load live economic data. Check API key or Finnhub service.")

    nfp_hist = load_historical_data("Non-Farm Employment Change")
    print("\nSample NFP Historical Data (still sample):")
    print(nfp_hist.head())

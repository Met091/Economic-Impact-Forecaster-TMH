# data_loader.py
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, date
import pytz 
import investpy 
import requests # For Alpha Vantage API calls

# --- Alpha Vantage API Configuration ---
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

def get_alpha_vantage_api_key():
    """Retrieves the Alpha Vantage API key from Streamlit secrets."""
    try:
        return st.secrets["ALPHA_VANTAGE_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error("🚨 Alpha Vantage API key not found. Please add `ALPHA_VANTAGE_API_KEY = \"YOUR_KEY\"` to your Streamlit secrets (e.g., .streamlit/secrets.toml) to fetch real US historical data.")
        return None

# --- investpy Data Fetching (Main Calendar) ---
@st.cache_data(ttl=1800) 
def fetch_economic_calendar_from_investpy(from_date_obj, to_date_obj):
    # ... (fetch_economic_calendar_from_investpy function remains the same as V9) ...
    if not isinstance(from_date_obj, date) or not isinstance(to_date_obj, date):
        st.error("🚨 Invalid date objects provided to fetch_economic_calendar_from_investpy.")
        return pd.DataFrame()
    from_date_str = from_date_obj.strftime("%d/%m/%Y")
    to_date_str = to_date_obj.strftime("%d/%m/%Y")
    try:
        df_investpy = investpy.economic_calendar(from_date=from_date_str, to_date=to_date_str)
        if df_investpy.empty: return pd.DataFrame()
        df = df_investpy.copy()
        def create_timestamp(row):
            try:
                time_str = row['time']
                if time_str == 'All Day' or pd.isna(time_str): time_str = '00:00'
                datetime_str = f"{row['date']} {time_str}"
                naive_dt = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M")
                return pytz.utc.localize(naive_dt)
            except Exception: return pd.NaT
        df['Timestamp'] = df.apply(create_timestamp, axis=1)
        df.dropna(subset=['Timestamp'], inplace=True)
        column_mapping = {'zone': 'Zone', 'currency': 'Currency', 'importance': 'Impact', 'event': 'EventName', 'actual': 'Actual', 'forecast': 'Forecast', 'previous': 'Previous'}
        df.rename(columns=column_mapping, inplace=True)
        impact_map = {'low': 'Low', 'medium': 'Medium', 'high': 'High'}
        if 'Impact' in df.columns: df['Impact'] = df['Impact'].map(impact_map).fillna('N/A')
        else: df['Impact'] = 'N/A'
        def clean_numeric_value(value):
            if pd.isna(value) or value == ' ': return np.nan
            if isinstance(value, (int, float)): return float(value)
            text = str(value).strip().replace(' ', '').replace('$', '').replace('€', '').replace('£', '')
            multiplier = 1
            if 'K' in text.upper(): multiplier = 1000; text = text.upper().replace('K', '')
            elif 'M' in text.upper(): multiplier = 1000000; text = text.upper().replace('M', '')
            elif 'B' in text.upper(): multiplier = 1000000000; text = text.upper().replace('B', '')
            text = text.replace('%', '')
            try: return float(text) * multiplier
            except ValueError: return np.nan
        numeric_cols_investpy = ['Actual', 'Forecast', 'Previous']
        for col in numeric_cols_investpy:
            if col in df.columns: df[col] = df[col].apply(clean_numeric_value)
            else: df[col] = np.nan
        app_columns = ['id', 'Timestamp', 'Currency', 'EventName', 'Impact', 'Previous', 'Forecast', 'Actual', 'Zone']
        df_final = df[[col for col in app_columns if col in df.columns]].copy()
        df_final['app_id'] = range(len(df_final))
        return df_final.sort_values(by='Timestamp').reset_index(drop=True)
    except RuntimeError as e: st.error(f"🚨 investpy Runtime Error: {e}."); return pd.DataFrame()
    except ConnectionError as e: st.error(f"🚨 investpy Connection Error: {e}."); return pd.DataFrame()
    except Exception as e: st.error(f"🚨 Unexpected error with investpy: {e}"); return pd.DataFrame()

def load_economic_data(start_date, end_date):
    if not start_date or not end_date:
        st.error("🚨 Start date or end date not provided to load_economic_data.")
        return pd.DataFrame()
    df_investpy = fetch_economic_calendar_from_investpy(start_date, end_date)
    if 'app_id' in df_investpy.columns: df_investpy.rename(columns={'app_id': 'id'}, inplace=True)
    elif 'id' not in df_investpy.columns and not df_investpy.empty: df_investpy['id'] = range(len(df_investpy))
    return df_investpy

# --- Alpha Vantage Historical Data Fetching ---
@st.cache_data(ttl=86400) # Cache Alpha Vantage historical data for 1 day
def fetch_us_indicator_history_alphavantage(indicator_function_name, api_key, interval=None):
    """
    Fetches historical data for a specific US indicator from Alpha Vantage.
    Returns a DataFrame with 'Date' (index) and 'Actual' value.
    """
    if not api_key:
        return pd.DataFrame()

    params = {
        "function": indicator_function_name,
        "apikey": api_key,
        "datatype": "json" # JSON is easier to parse here
    }
    if interval and indicator_function_name in ["REAL_GDP", "CPI", "TREASURY_YIELD", "FEDERAL_FUNDS_RATE"]: # Add other functions that accept interval
        params["interval"] = interval
    
    try:
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if "Note" in data and "Thank you for using Alpha Vantage!" in data["Note"]: # Rate limit message
            st.warning(f"Alpha Vantage API rate limit likely reached. Please try again later. ({data['Note']})")
            return pd.DataFrame()
        if "Error Message" in data:
            st.error(f"🚨 Alpha Vantage API Error for {indicator_function_name}: {data['Error Message']}")
            return pd.DataFrame()
        if not data or "data" not in data or not data["data"]:
            # st.info(f"No historical data returned from Alpha Vantage for {indicator_function_name}.")
            return pd.DataFrame()

        hist_df = pd.DataFrame(data["data"])
        if 'date' not in hist_df.columns or 'value' not in hist_df.columns:
            # st.warning(f"Unexpected data format from Alpha Vantage for {indicator_function_name}. Missing 'date' or 'value'.")
            return pd.DataFrame()
            
        hist_df['Date'] = pd.to_datetime(hist_df['date'])
        hist_df['Actual'] = pd.to_numeric(hist_df['value'], errors='coerce')
        hist_df.set_index('Date', inplace=True)
        hist_df = hist_df[['Actual']].dropna().sort_index() # Keep only 'Actual', drop NaNs, sort
        
        # For indicators like NFP, Alpha Vantage values are in thousands.
        # Our sample data for NFP was also in thousands (e.g., 175.0 for 175K).
        # No explicit unit conversion needed here if AV also provides it in K.
        # If AV provides full numbers, then divide by 1000 for NFP.
        # For now, assume values are directly comparable or in the expected unit.

        return hist_df
    except requests.exceptions.RequestException as e:
        st.error(f"🚨 API Request Error fetching historical data from Alpha Vantage for {indicator_function_name}: {e}")
        return pd.DataFrame()
    except ValueError as e: 
        st.error(f"🚨 Error decoding Alpha Vantage API response for {indicator_function_name}: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"🚨 Unexpected error with Alpha Vantage historical data for {indicator_function_name}: {e}")
        return pd.DataFrame()

# --- Load Historical Data (Main Function) ---
# No @st.cache_data on this main loader, caching is on the fetch_ function
def load_historical_data(event_name):
    """
    Loads historical data. Tries Alpha Vantage for supported US indicators,
    otherwise falls back to sample data.
    """
    api_key = get_alpha_vantage_api_key()
    av_df = pd.DataFrame()

    # Map app event names to Alpha Vantage function names and required intervals
    event_to_av_map = {
        "Non-Farm Employment Change": {"function": "NONFARM_PAYROLL", "interval": None}, # Monthly by default
        "Unemployment Rate": {"function": "UNEMPLOYMENT", "interval": None}, # Monthly by default
        "Core CPI m/m": {"function": "CPI", "interval": "monthly"}, # AV provides general CPI
        "Retail Sales m/m": {"function": "RETAIL_SALES", "interval": None}, # Monthly by default
        "Real GDP": {"function": "REAL_GDP", "interval": "quarterly"}, # Or annual
        # Add more mappings here
    }

    matched_av_indicator = None
    for key_event, av_params in event_to_av_map.items():
        if key_event.lower() in event_name.lower():
            matched_av_indicator = av_params
            break
            
    if matched_av_indicator and api_key:
        # st.caption(f"Attempting to fetch real historical data for {event_name} from Alpha Vantage...")
        av_df = fetch_us_indicator_history_alphavantage(
            matched_av_indicator["function"], 
            api_key,
            interval=matched_av_indicator.get("interval")
        )
        if not av_df.empty:
            # Alpha Vantage typically provides 'Actual'. We might not have 'Forecast' or 'Previous' for these historical points.
            # The plotting function expects 'Actual', 'Forecast', 'Previous'.
            # We will return only 'Actual' from AV, and the plot function can adapt or show only 'Actual'.
            # Or, we can create dummy Forecast/Previous if needed for consistent plotting.
            # For now, just return the DataFrame with 'Actual'.
            return av_df # Contains 'Actual' column, Date index

    # Fallback to sample data if Alpha Vantage fails or event not mapped
    if av_df.empty:
        # st.caption(f"Using sample historical data for {event_name}.")
        pass # Message will be in app.py

    # --- Sample Historical Data (Fallback) ---
    today_date = datetime.now().date()
    sample_historical_data = {
        "Non-Farm Employment Change": pd.DataFrame({
            'Date': [today_date - timedelta(days=30*i) for i in range(12, 0, -1)], # More data points
            'Actual': [187.0, 150.0, 275.0, 216.0, 353.0, 175.0, 200.0, 220.0, 180.0, 190.0, 210.0, 205.0],
            'Forecast': [170.0, 180.0, 190.0, 175.0, 185.0, 200.0, 190.0, 210.0, 185.0, 195.0, 200.0, 200.0],
            'Previous': [165.0, 187.0, 150.0, 275.0, 216.0, 353.0, 175.0, 200.0, 220.0, 180.0, 190.0, 210.0]
        }),
        "Unemployment Rate": pd.DataFrame({
            'Date': [today_date - timedelta(days=30*i) for i in range(12, 0, -1)],
            'Actual': [3.8, 3.9, 3.7, 3.7, 3.7, 3.9, 3.6, 3.8, 3.7, 3.9, 3.8, 3.7],
            'Forecast': [3.8, 3.8, 3.8, 3.7, 3.8, 3.9, 3.7, 3.8, 3.7, 3.8, 3.8, 3.8],
            'Previous': [3.7, 3.8, 3.9, 3.7, 3.7, 3.7, 3.8, 3.6, 3.8, 3.7, 3.9, 3.8]
        }),
         "Core CPI m/m": pd.DataFrame({ # Sample data for Core CPI
            'Date': [today_date - timedelta(days=30*i) for i in range(12, 0, -1)],
            'Actual': [0.3, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.3, 0.5, 0.3, 0.2],
            'Forecast': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.3, 0.4, 0.3, 0.3],
            'Previous': [0.2, 0.3, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.3, 0.5, 0.3]
        }),
        "Retail Sales m/m": pd.DataFrame({ # Sample data for Retail Sales
            'Date': [today_date - timedelta(days=30*i) for i in range(12, 0, -1)],
            'Actual': [0.7, -0.8, 0.4, 0.9, -1.1, 0.6, 0.3, 0.5, -0.3, 0.8, 0.1, 0.0],
            'Forecast': [0.4, -0.5, 0.5, 0.6, -0.8, 0.5, 0.2, 0.4, -0.2, 0.6, 0.2, 0.1],
            'Previous': [-0.2, 0.7, -0.8, 0.4, 0.9, -1.1, 0.6, 0.3, 0.5, -0.3, 0.8, 0.1]
        })
    }
    for key_event, df_sample in sample_historical_data.items():
        if key_event.lower() in event_name.lower():
            df = df_sample.copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df # Returns 'Actual', 'Forecast', 'Previous'
    return pd.DataFrame()


if __name__ == '__main__':
    # Test investpy calendar
    today = date.today()
    start_test_date = today - timedelta(days=today.weekday()) 
    end_test_date = start_test_date + timedelta(days=6)     
    print(f"Fetching investpy calendar for: {start_test_date.strftime('%d/%m/%Y')} to {end_test_date.strftime('%d/%m/%Y')}")
    calendar_data = load_economic_data(start_test_date, end_test_date)
    if not calendar_data.empty: print("\nInvestpy Calendar Data Sample:\n", calendar_data.head())
    else: print("\nFailed to load investpy calendar data.")

    # Test Alpha Vantage historical data
    print("\n--- Testing Alpha Vantage Historical Data ---")
    # Mock st.secrets for local test if needed:
    # class MockSecretsAV:
    #     def __getitem__(self, key):
    #         if key == "ALPHA_VANTAGE_API_KEY":
    #             return "YOUR_AV_KEY_FOR_LOCAL_TEST" 
    #         raise KeyError(key)
    # st.secrets = MockSecretsAV()
    
    nfp_hist_av = load_historical_data("Non-Farm Employment Change")
    if not nfp_hist_av.empty: print("\nNFP Historical Data (from AV if key valid, else sample):\n", nfp_hist_av.head())
    else: print("\nNFP Historical Data: Empty or failed to load.")
    
    cpi_hist_av = load_historical_data("Core CPI m/m")
    if not cpi_hist_av.empty: print("\nCPI Historical Data (from AV if key valid, else sample):\n", cpi_hist_av.head())
    else: print("\nCPI Historical Data: Empty or failed to load.")

    gdp_hist_av = load_historical_data("Real GDP") # Example of another mapped indicator
    if not gdp_hist_av.empty: print("\nReal GDP Historical Data (from AV if key valid, else sample):\n", gdp_hist_av.head())
    else: print("\nReal GDP Historical Data: Empty or failed to load.")

    unmapped_hist = load_historical_data("Some Unmapped Event") # Should use sample or be empty
    if not unmapped_hist.empty: print("\nUnmapped Event Historical Data (should be sample if defined, else empty):\n", unmapped_hist.head())
    else: print("\nUnmapped Event Historical Data: Empty (as expected if no sample).")

# data_loader.py
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, date
import pytz 
import investpy 
import requests 
import time # For retry delay

# --- Alpha Vantage API Configuration ---
# ... (get_alpha_vantage_api_key function remains the same) ...
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
def get_alpha_vantage_api_key():
    try: return st.secrets["ALPHA_VANTAGE_API_KEY"]
    except (KeyError, FileNotFoundError): st.error("🚨 Alpha Vantage API key not found in secrets."); return None

# --- Simulated Data Generation (Fallback) ---
def generate_simulated_economic_data(start_date, end_date):
    """Generates sample economic data for a given date range."""
    st.warning("⚠️ Generating simulated economic calendar data as live fetch failed.")
    simulated_data = []
    current_date = start_date
    event_counter = 0
    base_events = [
        {"EventName": "Simulated NFP", "Currency": "USD", "Impact": "High", "Hour": 14, "Minute": 30, "OffsetDays": 4}, # Friday
        {"EventName": "Simulated CAD Jobs", "Currency": "CAD", "Impact": "High", "Hour": 14, "Minute": 30, "OffsetDays": 4}, # Friday
        {"EventName": "Simulated US Unemp. Rate", "Currency": "USD", "Impact": "High", "Hour": 14, "Minute": 30, "OffsetDays": 4}, # Friday
        {"EventName": "Simulated GBP GDP", "Currency": "GBP", "Impact": "Medium", "Hour": 6, "Minute": 0, "OffsetDays": 1}, # Tuesday
        {"EventName": "Simulated EUR Speech", "Currency": "EUR", "Impact": "High", "Hour": 10, "Minute": 30, "OffsetDays": 2}, # Wednesday
        {"EventName": "Simulated US CPI", "Currency": "USD", "Impact": "High", "Hour": 14, "Minute": 30, "OffsetDays": 9}, # ~Next week Tue/Wed
    ]

    while current_date <= end_date:
        weekday = current_date.weekday() # Monday is 0, Sunday is 6
        for event_template in base_events:
            # Schedule event based on weekday offset relative to Monday (0)
            if event_template["OffsetDays"] == weekday:
                 # Create a UTC timestamp for that day at the specified hour/minute
                 event_dt_naive = datetime.combine(current_date, datetime.min.time()).replace(hour=event_template["Hour"], minute=event_template["Minute"])
                 event_dt_utc = pytz.utc.localize(event_dt_naive)

                 simulated_data.append({
                    "Timestamp": event_dt_utc,
                    "Currency": event_template["Currency"],
                    "EventName": event_template["EventName"],
                    "Impact": event_template["Impact"],
                    # Add some very basic random-ish previous/forecast
                    "Previous": round(np.random.uniform(-0.5, 5.0) * (100 if 'NFP' in event_template["EventName"] or 'Jobs' in event_template["EventName"] else 1), 1) if event_template["Impact"] != "High" else round(np.random.uniform(-0.1, 0.5), 1),
                    "Forecast": round(np.random.uniform(-0.5, 5.0) * (100 if 'NFP' in event_template["EventName"] or 'Jobs' in event_template["EventName"] else 1), 1) if event_template["Impact"] != "High" else round(np.random.uniform(0.0, 0.6), 1),
                    "Actual": np.nan,
                    "Zone": event_template["Currency"], # Use currency as zone for simplicity
                    "id": f"sim_{event_counter}"
                 })
                 event_counter += 1
        current_date += timedelta(days=1)
        
    if not simulated_data:
        return pd.DataFrame() # Return empty if no events generated

    df = pd.DataFrame(simulated_data)
    # Ensure correct types
    numeric_cols = ['Previous', 'Forecast', 'Actual']
    for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)
    
    return df.sort_values(by='Timestamp').reset_index(drop=True)


# --- investpy Data Fetching (Main Calendar) ---
# No cache here, caching applied within load_economic_data now
def fetch_economic_calendar_from_investpy(from_date_obj, to_date_obj, retries=2, delay=1):
    """
    Fetches economic calendar data from Investing.com using investpy with retries.
    """
    if not isinstance(from_date_obj, date) or not isinstance(to_date_obj, date):
        st.error("🚨 Invalid date objects provided to fetch_economic_calendar_from_investpy.")
        return pd.DataFrame()
    from_date_str = from_date_obj.strftime("%d/%m/%Y")
    to_date_str = to_date_obj.strftime("%d/%m/%Y")
    
    for attempt in range(retries + 1):
        try:
            # st.info(f"Attempt {attempt + 1}: Fetching investpy data for {from_date_str} to {to_date_str}...") # Optional debug info
            df_investpy = investpy.economic_calendar(from_date=from_date_str, to_date=to_date_str)
            
            # --- Data Transformation (same as before) ---
            if df_investpy.empty: return pd.DataFrame() # No data found is not an error
            df = df_investpy.copy()
            def create_timestamp(row):
                try:
                    time_str = row['time']; date_str = row['date']
                    if time_str == 'All Day' or pd.isna(time_str): time_str = '00:00'
                    if pd.isna(date_str): return pd.NaT # Cannot proceed without date
                    datetime_str = f"{date_str} {time_str}"
                    naive_dt = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M")
                    return pytz.utc.localize(naive_dt)
                except Exception: return pd.NaT
            df['Timestamp'] = df.apply(create_timestamp, axis=1)
            df.dropna(subset=['Timestamp'], inplace=True)
            if df.empty: return pd.DataFrame() # Return empty if all timestamps failed
            column_mapping = {'zone': 'Zone', 'currency': 'Currency', 'importance': 'Impact', 'event': 'EventName', 'actual': 'Actual', 'forecast': 'Forecast', 'previous': 'Previous'}
            df.rename(columns=column_mapping, inplace=True)
            impact_map = {'low': 'Low', 'medium': 'Medium', 'high': 'High'}; df['Impact'] = df['Impact'].map(impact_map).fillna('N/A') if 'Impact' in df.columns else 'N/A'
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
            for col in numeric_cols_investpy: df[col] = df[col].apply(clean_numeric_value) if col in df.columns else np.nan
            app_columns = ['id', 'Timestamp', 'Currency', 'EventName', 'Impact', 'Previous', 'Forecast', 'Actual', 'Zone']
            df_final = df[[col for col in app_columns if col in df.columns]].copy()
            df_final['app_id'] = range(len(df_final)) # Use this for Streamlit keys
            
            return df_final.sort_values(by='Timestamp').reset_index(drop=True) # SUCCESS

        except (RuntimeError, ConnectionError, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            st.warning(f"⚠️ investpy fetch attempt {attempt + 1} failed: {e}")
            if attempt < retries:
                time.sleep(delay) # Wait before retrying
            else:
                st.error(f"🚨 investpy failed after {retries + 1} attempts.")
                return pd.DataFrame() # Return empty after final retry fails
        except Exception as e:
            st.error(f"🚨 Unexpected error during investpy fetch attempt {attempt + 1}: {e}")
            # import traceback # Uncomment for detailed debugging
            # st.error(traceback.format_exc())
            return pd.DataFrame() # Return empty on unexpected error
            
    return pd.DataFrame() # Should not be reached if logic is correct, but acts as fallback

# --- Main Data Loading Function ---
# Cache the final result (either live or simulated)
@st.cache_data(ttl=900) # Cache combined data for 15 minutes
def load_economic_data(start_date, end_date):
    """
    Loads economic calendar data using investpy with fallback to simulation.
    Returns tuple: (DataFrame, status_message)
    """
    if not start_date or not end_date:
        st.error("🚨 Start date or end date not provided.")
        return pd.DataFrame(), "Error: Dates missing"
        
    df_live = fetch_economic_calendar_from_investpy(start_date, end_date)

    if not df_live.empty:
        status = f"Live (investpy @ {datetime.now().strftime('%H:%M:%S %Z')})"
        # Rename 'app_id' to 'id' for consistency
        if 'app_id' in df_live.columns:
            df_live.rename(columns={'app_id': 'id'}, inplace=True)
        elif 'id' not in df_live.columns: # Ensure an 'id' column exists
            df_live['id'] = range(len(df_live))
        return df_live, status
    else:
        # Fallback to simulated data
        df_simulated = generate_simulated_economic_data(start_date, end_date)
        status = "Simulated (investpy fetch failed)"
        # 'id' is already generated in generate_simulated_economic_data
        return df_simulated, status


# --- Alpha Vantage Historical Data Fetching ---
@st.cache_data(ttl=86400) 
def fetch_us_indicator_history_alphavantage(indicator_function_name, api_key, interval=None):
    # ... (Function remains the same as V12) ...
    if not api_key: return pd.DataFrame()
    params = {"function": indicator_function_name, "apikey": api_key, "datatype": "json"}
    if interval and indicator_function_name in ["REAL_GDP", "CPI", "TREASURY_YIELD", "FEDERAL_FUNDS_RATE"]: params["interval"] = interval
    try:
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params); response.raise_for_status(); data = response.json()
        if "Note" in data: st.warning(f"AV API limit? {data['Note']}"); return pd.DataFrame()
        if "Error Message" in data: st.error(f"🚨 AV API Error: {data['Error Message']}"); return pd.DataFrame()
        if not data or "data" not in data or not data["data"]: return pd.DataFrame()
        hist_df = pd.DataFrame(data["data"])
        if 'date' not in hist_df.columns or 'value' not in hist_df.columns: return pd.DataFrame()
        hist_df['Date'] = pd.to_datetime(hist_df['date']); hist_df['Actual'] = pd.to_numeric(hist_df['value'], errors='coerce')
        hist_df.set_index('Date', inplace=True); hist_df = hist_df[['Actual']].dropna().sort_index()
        return hist_df
    except requests.exceptions.RequestException as e: st.error(f"🚨 AV API Request Error: {e}"); return pd.DataFrame()
    except ValueError as e: st.error(f"🚨 AV JSON Error: {e}"); return pd.DataFrame()
    except Exception as e: st.error(f"🚨 Unexpected AV Error: {e}"); return pd.DataFrame()

# --- Load Historical Data (Main Function) ---
def load_historical_data(event_name):
    # ... (Function largely the same, but added more sample data below) ...
    api_key = get_alpha_vantage_api_key()
    av_df = pd.DataFrame()
    event_to_av_map = {
        "Non-Farm Employment Change": {"function": "NONFARM_PAYROLL", "interval": None},
        "Unemployment Rate": {"function": "UNEMPLOYMENT", "interval": None},
        "Core CPI m/m": {"function": "CPI", "interval": "monthly"}, # AV provides general CPI
        "CPI m/m": {"function": "CPI", "interval": "monthly"}, # Map headline CPI too
        "Retail Sales m/m": {"function": "RETAIL_SALES", "interval": None},
        "Real GDP": {"function": "REAL_GDP", "interval": "quarterly"},
        "Treasury Yield": {"function": "TREASURY_YIELD", "interval": "daily"}, # Example, needs maturity param logic
        "Federal Funds Rate": {"function": "FEDERAL_FUNDS_RATE", "interval": "daily"},
    }
    matched_av_indicator = None
    for key_event, av_params in event_to_av_map.items():
        if key_event.lower() in event_name.lower(): matched_av_indicator = av_params; break
    if matched_av_indicator and api_key:
        av_df = fetch_us_indicator_history_alphavantage(matched_av_indicator["function"], api_key, interval=matched_av_indicator.get("interval"))
        if not av_df.empty: return av_df # Return AV data if successful

    # --- Sample Historical Data (Fallback or for non-AV events) ---
    today_date = datetime.now().date()
    # Added more sample series and extended data points
    sample_historical_data = {
        "Non-Farm Employment Change": pd.DataFrame({'Date': [today_date - timedelta(days=30*i) for i in range(12, 0, -1)], 'Actual': [187.0, 150.0, 275.0, 216.0, 353.0, 175.0, 200.0, 220.0, 180.0, 190.0, 210.0, 205.0], 'Forecast': [170.0, 180.0, 190.0, 175.0, 185.0, 200.0, 190.0, 210.0, 185.0, 195.0, 200.0, 200.0], 'Previous': [165.0, 187.0, 150.0, 275.0, 216.0, 353.0, 175.0, 200.0, 220.0, 180.0, 190.0, 210.0]}),
        "Unemployment Rate": pd.DataFrame({'Date': [today_date - timedelta(days=30*i) for i in range(12, 0, -1)], 'Actual': [3.8, 3.9, 3.7, 3.7, 3.7, 3.9, 3.6, 3.8, 3.7, 3.9, 3.8, 3.7], 'Forecast': [3.8, 3.8, 3.8, 3.7, 3.8, 3.9, 3.7, 3.8, 3.7, 3.8, 3.8, 3.8], 'Previous': [3.7, 3.8, 3.9, 3.7, 3.7, 3.7, 3.8, 3.6, 3.8, 3.7, 3.9, 3.8]}),
        "Core CPI m/m": pd.DataFrame({'Date': [today_date - timedelta(days=30*i) for i in range(12, 0, -1)], 'Actual': [0.3, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.3, 0.5, 0.3, 0.2], 'Forecast': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.3, 0.4, 0.3, 0.3], 'Previous': [0.2, 0.3, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.3, 0.5, 0.3]}),
        "Retail Sales m/m": pd.DataFrame({'Date': [today_date - timedelta(days=30*i) for i in range(12, 0, -1)], 'Actual': [0.7, -0.8, 0.4, 0.9, -1.1, 0.6, 0.3, 0.5, -0.3, 0.8, 0.1, 0.0], 'Forecast': [0.4, -0.5, 0.5, 0.6, -0.8, 0.5, 0.2, 0.4, -0.2, 0.6, 0.2, 0.1], 'Previous': [-0.2, 0.7, -0.8, 0.4, 0.9, -1.1, 0.6, 0.3, 0.5, -0.3, 0.8, 0.1]}),
        "GDP": pd.DataFrame({'Date': [today_date - timedelta(days=90*i) for i in range(8, 0, -1)], 'Actual': [2.1, 1.8, 2.5, 2.9, 3.2, 2.0, 1.1, 0.8], 'Forecast': [2.0, 1.9, 2.4, 2.7, 3.0, 2.1, 1.0, 0.9], 'Previous': [1.9, 2.1, 1.8, 2.5, 2.9, 3.2, 2.0, 1.1]}), # Sample GDP q/q
        "PMI": pd.DataFrame({'Date': [today_date - timedelta(days=30*i) for i in range(12, 0, -1)], 'Actual': [50.3, 49.1, 50.9, 50.0, 52.2, 53.0, 52.8, 51.9, 50.6, 49.8, 50.1, 50.5], 'Forecast': [50.1, 49.5, 50.5, 50.2, 52.0, 52.8, 52.5, 51.5, 50.5, 50.0, 50.0, 50.4], 'Previous': [49.9, 50.3, 49.1, 50.9, 50.0, 52.2, 53.0, 52.8, 51.9, 50.6, 49.8, 50.1]}), # Sample PMI
    }
    for key_event, df_sample in sample_historical_data.items():
        # Use broader matching for sample data keys
        if key_event.lower() in event_name.lower() or \
           (key_event == "GDP" and "gdp" in event_name.lower()) or \
           (key_event == "PMI" and "pmi" in event_name.lower()):
            df = df_sample.copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df 
    return pd.DataFrame() # Return empty if no match

# --- (Optional: Add __main__ block for testing if desired) ---

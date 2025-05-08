# updated_data_loader.py
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, date
import pytz
import investpy
import requests
import time # For retry delay
import logging
import yfinance as yf
import os

# --- Logging Configuration ---
LOG_FILE_NAME = "app_data.log"

# Determine Log Level: st.secrets > Environment Variable > Default (INFO)
# Valid log levels: CRITICAL, ERROR, WARNING, INFO, DEBUG
DEFAULT_LOG_LEVEL = "INFO"
config_log_level_str = DEFAULT_LOG_LEVEL

try:
    # Check Streamlit secrets first
    if "LOG_LEVEL" in st.secrets:
        config_log_level_str = str(st.secrets["LOG_LEVEL"]).upper()
    else:
        # Fallback to environment variable
        env_log_level = os.environ.get("APP_LOG_LEVEL")
        if env_log_level:
            config_log_level_str = env_log_level.upper()
except Exception as e:
    # This might happen if st.secrets is not available in all contexts (e.g. testing outside Streamlit)
    env_log_level = os.environ.get("APP_LOG_LEVEL")
    if env_log_level:
        config_log_level_str = env_log_level.upper()
    # If st.secrets access fails and no env var, it will use DEFAULT_LOG_LEVEL

# Validate the log level string
numeric_level = getattr(logging, config_log_level_str, None)
if not isinstance(numeric_level, int):
    print(f"Warning: Invalid log level '{config_log_level_str}'. Defaulting to '{DEFAULT_LOG_LEVEL}'.")
    config_log_level_str = DEFAULT_LOG_LEVEL
    numeric_level = getattr(logging, config_log_level_str)


log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(numeric_level)

if not logger.handlers:
    try:
        file_handler = logging.FileHandler(LOG_FILE_NAME)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        st.error(f"🚨 Failed to configure file logger: {e}")
        print(f"Error setting up file logger: {e}")

    # Optional: Console Handler for local debugging or if Streamlit Cloud picks it up
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(log_formatter)
    # logger.addHandler(stream_handler)

logger.info(f"Data loader module initialized. Log level set to: {config_log_level_str}")

# --- Alpha Vantage API Configuration ---
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
def get_alpha_vantage_api_key():
    try:
        return st.secrets["ALPHA_VANTAGE_API_KEY"]
    except (KeyError, FileNotFoundError):
        logger.error("Alpha Vantage API key not found in st.secrets.")
        # st.error("🚨 Alpha Vantage API key not found in secrets.") # Avoid st.error in utility if app handles it
        return None

# --- yfinance Ticker Mapping ---
# More explicit mapping for event_name to yfinance tickers
EVENT_TO_YFINANCE_TICKER_MAP = {
    # Indices
    "S&P 500 Index": "^GSPC",
    "Dow Jones Industrial Average": "^DJI",
    "NASDAQ Composite": "^IXIC",
    "FTSE 100": "^FTSE",
    "DAX PERFORMANCE-INDEX": "^GDAXI",
    "Nikkei 225": "^N225",
    # Commodities
    "Crude Oil WTI Futures": "CL=F",
    "Gold Futures": "GC=F",
    "Silver Futures": "SI=F",
    "Copper Futures": "HG=F",
    # Bonds/Rates (Examples - yfinance coverage for specific yields can vary)
    "US 10 Year Treasury Yield": "^TNX", # Treasury Yield 10 Years
    "US 2 Year Treasury Yield": "^IRX", # Typically 13 Week Treasury Bill, adjust if specific 2Y ticker found
    "US 30 Year Treasury Yield": "^TYX", # Treasury Yield 30 Years
    "Federal Funds Rate": "FEDFUNDS", # FRED data, yfinance might not have direct equivalent always check
    # Currencies (Example: EURUSD=X for exchange rates)
    "EUR/USD Exchange Rate": "EURUSD=X",
    "GBP/USD Exchange Rate": "GBPUSD=X",
    "USD/JPY Exchange Rate": "USDJPY=X",
    # Specific Economic Indicators (if a proxy ticker exists)
    # "Philly Fed Manufacturing Index": "PATSLPHILLYFED" # Example if such a ticker existed on yfinance
}


# --- Simulated Data Generation (Fallback) ---
def generate_simulated_economic_data(start_date, end_date):
    """Generates sample economic data for a given date range."""
    logger.warning("Generating simulated economic calendar data as live fetch failed or was not attempted first.")
    st.warning("⚠️ Generating simulated economic calendar data as live data fetch failed.")
    simulated_data = []
    current_date = start_date
    event_counter = 0
    base_events = [
        {"EventName": "Simulated NFP", "Currency": "USD", "Impact": "High", "Hour": 14, "Minute": 30, "OffsetDays": 4},
        {"EventName": "Simulated CAD Jobs", "Currency": "CAD", "Impact": "High", "Hour": 14, "Minute": 30, "OffsetDays": 4},
        {"EventName": "Simulated US Unemp. Rate", "Currency": "USD", "Impact": "High", "Hour": 14, "Minute": 30, "OffsetDays": 4},
        {"EventName": "Simulated GBP GDP", "Currency": "GBP", "Impact": "Medium", "Hour": 6, "Minute": 0, "OffsetDays": 1},
        {"EventName": "Simulated EUR Speech", "Currency": "EUR", "Impact": "High", "Hour": 10, "Minute": 30, "OffsetDays": 2},
        {"EventName": "Simulated US CPI", "Currency": "USD", "Impact": "High", "Hour": 14, "Minute": 30, "OffsetDays": 9}, # Example for a future event
    ]

    while current_date <= end_date:
        weekday = current_date.weekday()
        for event_template in base_events:
            if event_template["OffsetDays"] == weekday:
                 event_dt_naive = datetime.combine(current_date, datetime.min.time()).replace(hour=event_template["Hour"], minute=event_template["Minute"])
                 event_dt_utc = pytz.utc.localize(event_dt_naive)
                 simulated_data.append({
                    "Timestamp": event_dt_utc, "Currency": event_template["Currency"],
                    "EventName": event_template["EventName"], "Impact": event_template["Impact"],
                    "Previous": round(np.random.uniform(-0.5, 5.0) * (100 if 'NFP' in event_template["EventName"] or 'Jobs' in event_template["EventName"] else 1), 1) if event_template["Impact"] != "High" else round(np.random.uniform(-0.1, 0.5), 1),
                    "Forecast": round(np.random.uniform(-0.5, 5.0) * (100 if 'NFP' in event_template["EventName"] or 'Jobs' in event_template["EventName"] else 1), 1) if event_template["Impact"] != "High" else round(np.random.uniform(0.0, 0.6), 1),
                    "Actual": np.nan, "Zone": event_template["Currency"], "id": f"sim_{event_counter}"
                 })
                 event_counter += 1
        current_date += timedelta(days=1)
        
    if not simulated_data: return pd.DataFrame()
    df = pd.DataFrame(simulated_data)
    numeric_cols = ['Previous', 'Forecast', 'Actual']
    for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)
    return df.sort_values(by='Timestamp').reset_index(drop=True)


# --- investpy Data Fetching (Main Calendar) ---
def fetch_economic_calendar_from_investpy(from_date_obj, to_date_obj, retries=2, delay=1):
    if not isinstance(from_date_obj, date) or not isinstance(to_date_obj, date):
        logger.error("Invalid date objects provided to fetch_economic_calendar_from_investpy.", extra={"from_date": from_date_obj, "to_date": to_date_obj})
        # st.error("🚨 Invalid date objects provided to fetch_economic_calendar_from_investpy.")
        return pd.DataFrame()
    from_date_str = from_date_obj.strftime("%d/%m/%Y")
    to_date_str = to_date_obj.strftime("%d/%m/%Y")
    
    logger.info(f"Attempting to fetch investpy economic calendar for {from_date_str} to {to_date_str}")
    for attempt in range(retries + 1):
        try:
            logger.info(f"investpy fetch attempt {attempt + 1}/{retries + 1}")
            df_investpy = investpy.economic_calendar(from_date=from_date_str, to_date=to_date_str)
            
            if df_investpy.empty:
                logger.info("investpy returned no data for the given date range.")
                return pd.DataFrame()

            df = df_investpy.copy()
            def create_timestamp(row):
                try:
                    time_str, date_str = row['time'], row['date']
                    if time_str == 'All Day' or pd.isna(time_str): time_str = '00:00'
                    if pd.isna(date_str): return pd.NaT
                    return pytz.utc.localize(datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M"))
                except Exception as e:
                    logger.warning(f"Failed to parse timestamp for row: {row}. Error: {e}", exc_info=False)
                    return pd.NaT
            df['Timestamp'] = df.apply(create_timestamp, axis=1)
            df.dropna(subset=['Timestamp'], inplace=True)
            if df.empty: logger.warning("All timestamps failed parsing in investpy data."); return pd.DataFrame()

            column_mapping = {'zone': 'Zone', 'currency': 'Currency', 'importance': 'Impact', 'event': 'EventName', 'actual': 'Actual', 'forecast': 'Forecast', 'previous': 'Previous'}
            df.rename(columns=column_mapping, inplace=True)
            impact_map = {'low': 'Low', 'medium': 'Medium', 'high': 'High'}
            df['Impact'] = df['Impact'].map(impact_map).fillna('N/A') if 'Impact' in df.columns else 'N/A'
            
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
                except ValueError: logger.debug(f"Could not parse numeric value: {value}"); return np.nan
            numeric_cols_investpy = ['Actual', 'Forecast', 'Previous']
            for col in numeric_cols_investpy: df[col] = df[col].apply(clean_numeric_value) if col in df.columns else np.nan
            
            app_columns = ['id', 'Timestamp', 'Currency', 'EventName', 'Impact', 'Previous', 'Forecast', 'Actual', 'Zone']
            df_final = df[[col for col in app_columns if col in df.columns]].copy()
            if 'id' not in df_final.columns: df_final['id'] = range(len(df_final))
            
            logger.info(f"Successfully fetched and processed {len(df_final)} events from investpy.")
            return df_final.sort_values(by='Timestamp').reset_index(drop=True)

        except (RuntimeError, ConnectionError, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            logger.error(f"investpy fetch attempt {attempt + 1} failed due to network/runtime error.", exc_info=True)
            # st.warning(f"⚠️ investpy fetch attempt {attempt + 1} failed: {e}") # App should show generic failure
            if attempt < retries: time.sleep(delay)
            else: # st.error(f"🚨 investpy failed after {retries + 1} attempts."); # App should show generic failure
                  return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error during investpy fetch attempt {attempt + 1}.", exc_info=True)
            # st.error(f"🚨 Unexpected error during investpy fetch: {e}") # App should show generic failure
            return pd.DataFrame()
            
    logger.error("investpy fetch failed after all retries.")
    return pd.DataFrame()

# --- Main Data Loading Function ---
@st.cache_data(ttl=900)
def load_economic_data(start_date, end_date):
    logger.info(f"Loading economic data for range: {start_date} to {end_date}")
    if not start_date or not end_date:
        logger.error("Start date or end date not provided to load_economic_data.")
        st.error("🚨 Start date or end date not provided.") # This is a critical app error
        return pd.DataFrame(), "Error: Dates missing"
        
    df_live = fetch_economic_calendar_from_investpy(start_date, end_date)

    if not df_live.empty:
        status = f"Live (investpy @ {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S %Z')})"
        logger.info("Successfully loaded live economic data from investpy.")
        return df_live, status
    else:
        logger.warning("Live data fetch from investpy failed or returned empty. Falling back to simulated data.")
        df_simulated = generate_simulated_economic_data(start_date, end_date)
        status = "Simulated (investpy fetch failed)"
        if df_simulated.empty:
             logger.warning("Simulated data generation also resulted in an empty dataset.")
        else:
            logger.info(f"Generated {len(df_simulated)} simulated economic events.")
        return df_simulated, status

# --- Alpha Vantage Historical Data Fetching ---
@st.cache_data(ttl=86400) 
def fetch_us_indicator_history_alphavantage(indicator_function_name, api_key, interval=None):
    if not api_key:
        logger.error("Alpha Vantage API key not provided for fetching historical data.")
        return pd.DataFrame()
    
    params = {"function": indicator_function_name, "apikey": api_key, "datatype": "json"}
    if interval and indicator_function_name in ["REAL_GDP", "CPI", "TREASURY_YIELD", "FEDERAL_FUNDS_RATE"]:
        params["interval"] = interval
    
    logger.info(f"Fetching Alpha Vantage historical data for function: {indicator_function_name}, interval: {interval}")
    try:
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=15) # Added timeout
        response.raise_for_status()
        data = response.json()

        if "Note" in data:
            logger.warning(f"Alpha Vantage API Note (likely limit): {data['Note']}")
            # st.warning(f"AV API limit? {data['Note']}") # Let app handle UI message if needed
            return pd.DataFrame()
        if "Error Message" in data:
            logger.error(f"Alpha Vantage API Error: {data['Error Message']}")
            # st.error(f"🚨 AV API Error: {data['Error Message']}")
            return pd.DataFrame()
        if not data or "data" not in data or not data["data"]:
            logger.warning(f"No data found in Alpha Vantage response for {indicator_function_name}.")
            return pd.DataFrame()

        hist_df = pd.DataFrame(data["data"])
        if 'date' not in hist_df.columns or 'value' not in hist_df.columns:
            logger.error(f"Unexpected data format from Alpha Vantage (missing 'date' or 'value'): {hist_df.columns}")
            return pd.DataFrame()
            
        hist_df['Date'] = pd.to_datetime(hist_df['date'])
        hist_df['Actual'] = pd.to_numeric(hist_df['value'], errors='coerce')
        hist_df.set_index('Date', inplace=True)
        hist_df = hist_df[['Actual']].dropna().sort_index()
        logger.info(f"Successfully fetched {len(hist_df)} historical data points from Alpha Vantage for {indicator_function_name}.")
        return hist_df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Alpha Vantage API Request Error for {indicator_function_name}.", exc_info=True)
        # st.error(f"🚨 AV API Request Error: {e}")
        return pd.DataFrame()
    except ValueError as e: 
        logger.error(f"Alpha Vantage JSON parsing error for {indicator_function_name}.", exc_info=True)
        # st.error(f"🚨 AV JSON Error: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error during Alpha Vantage fetch for {indicator_function_name}.", exc_info=True)
        # st.error(f"🚨 Unexpected AV Error: {e}")
        return pd.DataFrame()

# --- yfinance Historical Data Fetching ---
@st.cache_data(ttl=86400)
def fetch_yfinance_historical_data(ticker_symbol, period="2y", interval="1d"): # Added interval, increased period
    """Fetches historical 'Close' price data for a given ticker using yfinance."""
    logger.info(f"Attempting to fetch historical data for ticker '{ticker_symbol}' from yfinance for period '{period}', interval '{interval}'.")
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist_df = ticker.history(period=period, interval=interval)

        if hist_df.empty:
            logger.warning(f"yfinance returned no data for ticker '{ticker_symbol}'. It might be an invalid ticker or no data for the period/interval.")
            return pd.DataFrame()

        if 'Close' not in hist_df.columns:
            logger.warning(f"'Close' column not found in yfinance data for {ticker_symbol}. Available: {hist_df.columns}")
            return pd.DataFrame()

        hist_df_standardized = hist_df[['Close']].copy()
        hist_df_standardized.rename(columns={'Close': 'Actual'}, inplace=True)
        hist_df_standardized.index.name = 'Date'
        
        if hist_df_standardized.index.tz is not None:
             hist_df_standardized.index = hist_df_standardized.index.tz_localize(None) # Make timezone naive

        logger.info(f"Successfully fetched {len(hist_df_standardized)} data points for ticker '{ticker_symbol}' from yfinance.")
        return hist_df_standardized.sort_index()

    except Exception as e:
        logger.error(f"Error fetching or processing data for ticker '{ticker_symbol}' from yfinance.", exc_info=True)
        return pd.DataFrame()


# --- Load Historical Data (Main Function) ---
def load_historical_data(event_name):
    logger.info(f"Loading historical data for event: '{event_name}'")
    api_key = get_alpha_vantage_api_key() # Fetch once
    
    # Attempt 1: Alpha Vantage (for specific US economic indicators)
    event_to_av_map = {
        "Non-Farm Employment Change": {"function": "NONFARM_PAYROLL", "interval": None}, # US specific
        "Unemployment Rate": {"function": "UNEMPLOYMENT", "interval": None}, # US specific
        "Core CPI m/m": {"function": "CPI", "interval": "monthly"}, # US specific (AV provides headline)
        "CPI m/m": {"function": "CPI", "interval": "monthly"}, # US specific
        "Retail Sales m/m": {"function": "RETAIL_SALES", "interval": None}, # US specific
        "Real GDP": {"function": "REAL_GDP", "interval": "quarterly"}, # US specific
        # "Treasury Yield": {"function": "TREASURY_YIELD", "interval": "daily"}, # Needs maturity param
        # "Federal Funds Rate": {"function": "FEDERAL_FUNDS_RATE", "interval": "daily"}, # US specific
    }
    matched_av_indicator = None
    for key_event, av_params in event_to_av_map.items():
        if key_event.lower() in event_name.lower(): # Simple substring match
            # Add a check for currency if available, e.g. if event is for "USD"
            # This requires currency context to be passed or inferred for event_name
            matched_av_indicator = av_params
            break
    
    if matched_av_indicator and api_key:
        logger.info(f"Found Alpha Vantage mapping for '{event_name}'. Function: {matched_av_indicator['function']}")
        av_df = fetch_us_indicator_history_alphavantage(
            matched_av_indicator["function"],
            api_key,
            interval=matched_av_indicator.get("interval")
        )
        if not av_df.empty:
            logger.info(f"Successfully loaded data from Alpha Vantage for '{event_name}'.")
            return av_df
        else:
            logger.warning(f"Alpha Vantage fetch was attempted for '{event_name}' but returned no data.")

    # Attempt 2: yfinance (using explicit map first, then heuristic)
    yf_ticker_to_try = None
    # Check explicit map first
    if event_name in EVENT_TO_YFINANCE_TICKER_MAP:
        yf_ticker_to_try = EVENT_TO_YFINANCE_TICKER_MAP[event_name]
        logger.info(f"Found direct yfinance ticker map for '{event_name}': {yf_ticker_to_try}")
    else:
        # Fallback to heuristic if no direct map (less reliable)
        common_market_terms = ["index", "stock", "equity", "futures", "treasury", "bond", "oil", "gold", "silver", "rate"]
        is_potential_ticker_like = any(term in event_name.lower() for term in common_market_terms) or \
                                   (event_name.isupper() and len(event_name) < 6 and not event_name.isalpha()) or \
                                   any(char in event_name for char in ['=','^','.'])
        if is_potential_ticker_like:
            yf_ticker_to_try = event_name # Use event_name itself as potential ticker
            logger.info(f"No direct yfinance map for '{event_name}', attempting heuristic: '{yf_ticker_to_try}'")


    if yf_ticker_to_try:
        yf_df = fetch_yfinance_historical_data(yf_ticker_to_try)
        if not yf_df.empty:
            logger.info(f"Successfully loaded data from yfinance for ticker '{yf_ticker_to_try}' (mapped from event '{event_name}').")
            return yf_df
        else:
            logger.warning(f"yfinance fetch was attempted for '{yf_ticker_to_try}' but returned no data.")

    # Attempt 3: Sample Historical Data (Fallback)
    logger.warning(f"Failed to fetch live historical data for '{event_name}' from available sources. Falling back to sample data.")
    today_date = datetime.now(pytz.utc).date() # Use timezone-aware date for consistency
    sample_historical_data = {
        # NFP (USD)
        "Non-Farm Employment Change": pd.DataFrame({
            'Date': pd.to_datetime([today_date - timedelta(days=30*i) for i in range(12, 0, -1)]), 
            'Actual': [187.0, 150.0, 275.0, 216.0, 353.0, 175.0, 200.0, 220.0, 180.0, 190.0, 210.0, 205.0], 
            'Forecast': [170.0, 180.0, 190.0, 175.0, 185.0, 200.0, 190.0, 210.0, 185.0, 195.0, 200.0, 200.0], 
            'Previous': [165.0, 187.0, 150.0, 275.0, 216.0, 353.0, 175.0, 200.0, 220.0, 180.0, 190.0, 210.0]
        }).set_index('Date'),
        # Unemployment Rate (USD)
        "Unemployment Rate": pd.DataFrame({
            'Date': pd.to_datetime([today_date - timedelta(days=30*i) for i in range(12, 0, -1)]), 
            'Actual': [3.8, 3.9, 3.7, 3.7, 3.7, 3.9, 3.6, 3.8, 3.7, 3.9, 3.8, 3.7], 
            'Forecast': [3.8, 3.8, 3.8, 3.7, 3.8, 3.9, 3.7, 3.8, 3.7, 3.8, 3.8, 3.8], 
            'Previous': [3.7, 3.8, 3.9, 3.7, 3.7, 3.7, 3.8, 3.6, 3.8, 3.7, 3.9, 3.8]
        }).set_index('Date'),
        # Core CPI (USD)
        "Core CPI m/m": pd.DataFrame({
            'Date': pd.to_datetime([today_date - timedelta(days=30*i) for i in range(12, 0, -1)]), 
            'Actual': [0.3, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.3, 0.5, 0.3, 0.2], 
            'Forecast': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.3, 0.4, 0.3, 0.3], 
            'Previous': [0.2, 0.3, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.3, 0.5, 0.3]
        }).set_index('Date'),
        # Retail Sales (USD)
        "Retail Sales m/m": pd.DataFrame({
            'Date': pd.to_datetime([today_date - timedelta(days=30*i) for i in range(12, 0, -1)]), 
            'Actual': [0.7, -0.8, 0.4, 0.9, -1.1, 0.6, 0.3, 0.5, -0.3, 0.8, 0.1, 0.0], 
            'Forecast': [0.4, -0.5, 0.5, 0.6, -0.8, 0.5, 0.2, 0.4, -0.2, 0.6, 0.2, 0.1], 
            'Previous': [-0.2, 0.7, -0.8, 0.4, 0.9, -1.1, 0.6, 0.3, 0.5, -0.3, 0.8, 0.1]
        }).set_index('Date'),
        # GDP (generic, could be for any major economy)
        "GDP q/q": pd.DataFrame({ # Renamed to match a common event name format
            'Date': pd.to_datetime([today_date - timedelta(days=90*i) for i in range(8, 0, -1)]), 
            'Actual': [2.1, 1.8, 2.5, 2.9, 3.2, 2.0, 1.1, 0.8], 
            'Forecast': [2.0, 1.9, 2.4, 2.7, 3.0, 2.1, 1.0, 0.9], 
            'Previous': [1.9, 2.1, 1.8, 2.5, 2.9, 3.2, 2.0, 1.1]
        }).set_index('Date'),
        # PMI (generic, could be for any major economy)
        "Manufacturing PMI": pd.DataFrame({ # Renamed for specificity
            'Date': pd.to_datetime([today_date - timedelta(days=30*i) for i in range(12, 0, -1)]), 
            'Actual': [50.3, 49.1, 50.9, 50.0, 52.2, 53.0, 52.8, 51.9, 50.6, 49.8, 50.1, 50.5], 
            'Forecast': [50.1, 49.5, 50.5, 50.2, 52.0, 52.8, 52.5, 51.5, 50.5, 50.0, 50.0, 50.4], 
            'Previous': [49.9, 50.3, 49.1, 50.9, 50.0, 52.2, 53.0, 52.8, 51.9, 50.6, 49.8, 50.1]
        }).set_index('Date'),
         # Example for a CAD event
        "CAD Employment Change": pd.DataFrame({
            'Date': pd.to_datetime([today_date - timedelta(days=30*i) for i in range(12, 0, -1)]),
            'Actual': [10.0, 15.0, -5.0, 20.0, 12.0, 8.0, 25.0, -10.0, 18.0, 22.0, 5.0, 13.0],
            'Forecast': [12.0, 10.0, 0.0, 18.0, 15.0, 10.0, 20.0, -8.0, 15.0, 20.0, 8.0, 15.0],
            'Previous': [8.0, 10.0, 15.0, -5.0, 20.0, 12.0, 8.0, 25.0, -10.0, 18.0, 22.0, 5.0]
        }).set_index('Date'),
    }
    # Ensure sample data index is timezone-naive
    for key in sample_historical_data:
        if sample_historical_data[key].index.tz is not None:
            sample_historical_data[key].index = sample_historical_data[key].index.tz_localize(None)


    # Try to match event_name to sample data keys
    for key_event_sample, df_sample in sample_historical_data.items():
        if key_event_sample.lower() in event_name.lower(): # Simple substring match
            logger.info(f"Using sample historical data for '{event_name}' matching key '{key_event_sample}'.")
            return df_sample.copy() # Return a copy
            
    logger.warning(f"No historical data (live, yfinance, or specific sample) found for '{event_name}'. Returning empty DataFrame.")
    return pd.DataFrame()

# --- (Optional: Add __main__ block for testing if desired) ---
# if __name__ == '__main__':
#     # Test logging (ensure APP_LOG_LEVEL is set in your environment or st.secrets for this to change)
#     logger.debug("This is a debug message.")
#     logger.info("This is an info message.")
#     logger.warning("This is a warning message.")
#     logger.error("This is an error message.")
#     logger.critical("This is a critical message.")

#     # Test yfinance ticker mapping and data fetching
#     print("\n--- Testing yfinance Gold Futures ---")
#     gold_data = load_historical_data("Gold Futures")
#     if not gold_data.empty:
#         print(gold_data.tail())
#     else:
#         print("No data for Gold Futures.")

#     print("\n--- Testing yfinance S&P 500 ---")
#     sp500_data = load_historical_data("S&P 500 Index")
#     if not sp500_data.empty:
#         print(sp500_data.tail())
#     else:
#         print("No data for S&P 500 Index.")
    
#     print("\n--- Testing Non-Existent Event ---")
#     non_existent_data = load_historical_data("Imaginary Economic Event")
#     if non_existent_data.empty:
#         print("Correctly returned empty DataFrame for non-existent event.")
#     else:
#         print("Error: Should have returned empty DataFrame.")

#     print("\n--- Testing Sample NFP Data ---")
#     nfp_sample = load_historical_data("Non-Farm Employment Change") # Should hit AV or sample
#     if not nfp_sample.empty:
#         print(nfp_sample.tail())
#     else:
#         print("No NFP data.")

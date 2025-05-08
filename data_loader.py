# data_loader.py
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, date
import pytz # For timezone handling
import investpy # For fetching data from Investing.com

# Note: Caching for fetch_economic_calendar_from_investpy directly,
# as load_economic_data will now have dynamic date parameters.
@st.cache_data(ttl=1800) # Cache investpy raw fetch for 30 minutes
def fetch_economic_calendar_from_investpy(from_date_obj, to_date_obj):
    """
    Fetches economic calendar data from Investing.com using investpy.

    Args:
        from_date_obj (date): Start date object.
        to_date_obj (date): End date object.

    Returns:
        pd.DataFrame: DataFrame containing economic events, or empty DataFrame on error.
    """
    if not isinstance(from_date_obj, date) or not isinstance(to_date_obj, date):
        st.error("🚨 Invalid date objects provided to fetch_economic_calendar_from_investpy.")
        return pd.DataFrame()

    from_date_str = from_date_obj.strftime("%d/%m/%Y") # investpy expects dd/mm/yyyy
    to_date_str = to_date_obj.strftime("%d/%m/%Y")
    
    # This info message can be moved to app.py if preferred, to avoid multiple prints on reruns
    # st.info(f"Attempting to fetch data from Investing.com for {from_date_str} to {to_date_str} using investpy...")

    try:
        df_investpy = investpy.economic_calendar(
            from_date=from_date_str,
            to_date=to_date_str
            # Consider adding countries=['united states', 'euro zone', 'japan', 'united kingdom', 'canada', 'australia', 'new zealand', 'switzerland']
            # to narrow down results and potentially speed up, if only major currencies are needed.
        )

        if df_investpy.empty:
            # This message is fine here as it's specific to the fetch result
            # st.info(f"No economic events found on Investing.com for {from_date_str} to {to_date_str} via investpy.")
            return pd.DataFrame()

        # --- Data Transformation and Cleaning ---
        df = df_investpy.copy()

        def create_timestamp(row):
            try:
                time_str = row['time']
                if time_str == 'All Day' or pd.isna(time_str):
                    time_str = '00:00'
                
                datetime_str = f"{row['date']} {time_str}"
                naive_dt = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M")
                return pytz.utc.localize(naive_dt) # Assume UTC
            except Exception:
                return pd.NaT

        df['Timestamp'] = df.apply(create_timestamp, axis=1)
        df.dropna(subset=['Timestamp'], inplace=True)

        column_mapping = {
            'zone': 'Zone', 'currency': 'Currency', 'importance': 'Impact',
            'event': 'EventName', 'actual': 'Actual', 'forecast': 'Forecast', 'previous': 'Previous'
        }
        df.rename(columns=column_mapping, inplace=True)

        impact_map = {'low': 'Low', 'medium': 'Medium', 'high': 'High'}
        if 'Impact' in df.columns:
            df['Impact'] = df['Impact'].map(impact_map).fillna('N/A')
        else:
            df['Impact'] = 'N/A'

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
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric_value)
            else:
                df[col] = np.nan
        
        app_columns = ['id', 'Timestamp', 'Currency', 'EventName', 'Impact', 'Previous', 'Forecast', 'Actual', 'Zone']
        df_final = df[[col for col in app_columns if col in df.columns]].copy()
        df_final['app_id'] = range(len(df_final)) # Use this for Streamlit keys

        return df_final.sort_values(by='Timestamp').reset_index(drop=True)

    except RuntimeError as e:
        st.error(f"🚨 investpy Runtime Error: {e}.")
        return pd.DataFrame()
    except ConnectionError as e:
        st.error(f"🚨 investpy Connection Error: {e}.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"🚨 Unexpected error with investpy: {e}")
        return pd.DataFrame()

# This function is now a wrapper that calls the cached fetch function with specific dates.
# No @st.cache_data here because its inputs (start_date, end_date) change frequently.
# The caching is handled by fetch_economic_calendar_from_investpy.
def load_economic_data(start_date, end_date):
    """
    Loads economic calendar data using investpy for the given date range.
    """
    if not start_date or not end_date:
        st.error("🚨 Start date or end date not provided to load_economic_data.")
        return pd.DataFrame()
        
    df_investpy = fetch_economic_calendar_from_investpy(start_date, end_date)

    if df_investpy.empty:
        # Message now handled in app.py or fetch function
        pass 
    
    # Rename 'app_id' to 'id' for consistency if 'id' from investpy is not preferred
    if 'app_id' in df_investpy.columns:
        df_investpy.rename(columns={'app_id': 'id'}, inplace=True)
    elif 'id' not in df_investpy.columns: # Ensure an 'id' column exists for keys
        df_investpy['id'] = range(len(df_investpy))
        
    return df_investpy

# Historical data remains sample data
@st.cache_data
def load_historical_data(event_name):
    """ Loads sample historical data for a given event name. """
    # ... (sample historical data generation remains the same as previous version) ...
    today_date = datetime.now().date()
    all_historical_data = {
        "Non-Farm Employment Change": pd.DataFrame({
            'Date': [today_date - timedelta(days=30*i) for i in range(6, 0, -1)],
            'Actual': [187.0, 150.0, 275.0, 216.0, 353.0, 175.0],
            'Forecast': [170.0, 180.0, 190.0, 175.0, 185.0, 200.0],
            'Previous': [165.0, 187.0, 150.0, 275.0, 216.0, 353.0]
        }),
        "Unemployment Rate": pd.DataFrame({
            'Date': [today_date - timedelta(days=30*i) for i in range(6, 0, -1)],
            'Actual': [3.8, 3.9, 3.7, 3.7, 3.7, 3.9],
            'Forecast': [3.8, 3.8, 3.8, 3.7, 3.8, 3.9],
            'Previous': [3.7, 3.8, 3.9, 3.7, 3.7, 3.7]
        }),
         "Core CPI m/m": pd.DataFrame({
            'Date': [today_date - timedelta(days=30*i) for i in range(6, 0, -1)],
            'Actual': [0.3, 0.4, 0.4, 0.3, 0.3, 0.3],
            'Forecast': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
            'Previous': [0.2, 0.3, 0.4, 0.4, 0.3, 0.3]
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
    print("Attempting to load economic data using investpy for a specific date range:")
    today = date.today()
    start_test_date = today - timedelta(days=today.weekday()) # Monday of current week
    end_test_date = start_test_date + timedelta(days=6)     # Sunday of current week
    
    print(f"Fetching for: {start_test_date.strftime('%d/%m/%Y')} to {end_test_date.strftime('%d/%m/%Y')}")
    live_data = load_economic_data(start_test_date, end_test_date)
    
    if not live_data.empty:
        print("\nEconomic Data Sample (from investpy if successful):")
        print(live_data[['Timestamp', 'Currency', 'EventName', 'Impact', 'Forecast']].head())
    else:
        print("\nFailed to load economic data using investpy for the test range.")

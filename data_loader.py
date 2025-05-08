# data_loader.py
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, date
import pytz # For timezone handling
import investpy # For fetching data from Investing.com

@st.cache_data(ttl=1800) # Cache for 30 minutes
def fetch_economic_calendar_from_investpy(from_date_obj, to_date_obj):
    """
    Fetches economic calendar data from Investing.com using investpy.

    Args:
        from_date_obj (date): Start date object.
        to_date_obj (date): End date object.

    Returns:
        pd.DataFrame: DataFrame containing economic events, or empty DataFrame on error.
    """
    from_date_str = from_date_obj.strftime("%d/%m/%Y") # investpy expects dd/mm/yyyy
    to_date_str = to_date_obj.strftime("%d/%m/%Y")
    
    st.info(f"Attempting to fetch data from Investing.com for {from_date_str} to {to_date_str} using investpy...")

    try:
        # Fetching for all countries. Can be narrowed down with `countries` parameter if needed.
        df_investpy = investpy.economic_calendar(
            from_date=from_date_str,
            to_date=to_date_str
        )

        if df_investpy.empty:
            st.info(f"No economic events found on Investing.com for {from_date_str} to {to_date_str} via investpy.")
            return pd.DataFrame()

        # --- Data Transformation and Cleaning ---
        df = df_investpy.copy()

        # 1. Combine 'date' and 'time' into a single datetime column, assuming UTC
        # investpy 'date' is string 'dd/mm/yyyy', 'time' is string 'HH:MM' or 'All Day'
        def create_timestamp(row):
            try:
                time_str = row['time']
                if time_str == 'All Day' or pd.isna(time_str): # Handle missing time as 'All Day'
                    time_str = '00:00' # Default to midnight for 'All Day' events
                
                # Combine date and time strings
                datetime_str = f"{row['date']} {time_str}"
                # Parse to naive datetime object
                naive_dt = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M")
                # IMPORTANT ASSUMPTION: Assume parsed datetime is UTC
                return pytz.utc.localize(naive_dt)
            except Exception as e:
                # st.warning(f"Could not parse date/time for event: {row.get('event', 'Unknown')} on {row.get('date', 'Unknown Date')} - {e}")
                return pd.NaT # Return Not-a-Time for parsing errors

        df['Timestamp'] = df.apply(create_timestamp, axis=1)
        df.dropna(subset=['Timestamp'], inplace=True) # Remove rows where timestamp creation failed

        # 2. Rename columns
        column_mapping = {
            # 'id' is already present from investpy
            'zone': 'Zone', # Country/Zone name from investpy
            'currency': 'Currency',
            'importance': 'Impact',
            'event': 'EventName',
            'actual': 'Actual',
            'forecast': 'Forecast',
            'previous': 'Previous'
        }
        df.rename(columns=column_mapping, inplace=True)

        # 3. Map Impact values
        impact_map = {'low': 'Low', 'medium': 'Medium', 'high': 'High'}
        if 'Impact' in df.columns:
            df['Impact'] = df['Impact'].map(impact_map).fillna('N/A')
        else:
            df['Impact'] = 'N/A'


        # 4. Clean and convert numeric columns (Actual, Forecast, Previous)
        # This is a challenging part as investpy returns strings with units (K, M, %, etc.)
        # A robust solution requires complex parsing. This is a simplified attempt.
        def clean_numeric_value(value):
            if pd.isna(value) or value == ' ':
                return np.nan
            if isinstance(value, (int, float)):
                return float(value)
            
            text = str(value).strip().replace(' ', '') # Remove spaces
            
            # Remove common currency symbols or non-numeric characters if they are at the start/end
            # This is a very basic attempt.
            text = text.replace('$', '').replace('€', '').replace('£', '')
            
            multiplier = 1
            if 'K' in text.upper():
                multiplier = 1000
                text = text.upper().replace('K', '')
            elif 'M' in text.upper():
                multiplier = 1000000
                text = text.upper().replace('M', '')
            elif 'B' in text.upper(): # For Billions, if ever present
                multiplier = 1000000000
                text = text.upper().replace('B', '')
            
            text = text.replace('%', '') # Remove percentage sign

            try:
                return float(text) * multiplier
            except ValueError:
                return np.nan # Return NaN if conversion fails

        numeric_cols_investpy = ['Actual', 'Forecast', 'Previous']
        for col in numeric_cols_investpy:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric_value)
            else:
                df[col] = np.nan
        
        # Select and order relevant columns for the app
        # 'id' from investpy can be used directly
        app_columns = ['id', 'Timestamp', 'Currency', 'EventName', 'Impact', 'Previous', 'Forecast', 'Actual', 'Zone']
        df_final = df[[col for col in app_columns if col in df.columns]].copy()
        
        # Add a unique 'app_id' if 'id' from investpy is not suitable or to ensure uniqueness for Streamlit keys
        df_final['app_id'] = range(len(df_final))


        return df_final.sort_values(by='Timestamp').reset_index(drop=True)

    except RuntimeError as e: # investpy often raises RuntimeError for various issues
        st.error(f"🚨 investpy Runtime Error: {e}. This might be due to changes on Investing.com or network issues.")
        return pd.DataFrame()
    except ConnectionError as e: # investpy can raise this
        st.error(f"🚨 investpy Connection Error: {e}. Check your internet connection.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"🚨 An unexpected error occurred while fetching or processing data with investpy: {e}")
        # import traceback
        # st.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

# This function now orchestrates fetching from investpy
@st.cache_data(ttl=900) # Cache investpy data for 15 minutes due to potential for changes
def load_economic_data():
    """
    Loads economic calendar data using investpy.
    Fetches data for a range (e.g., today and next 7 days).
    """
    today = date.today()
    # Fetch for the current day up to 14 days ahead for a good range
    from_date_obj = today
    to_date_obj = today + timedelta(days=13) 
    
    df_investpy = fetch_economic_calendar_from_investpy(from_date_obj, to_date_obj)

    if df_investpy.empty:
        st.warning("Could not retrieve data using investpy. The service might be temporarily unavailable or website structure might have changed.")
    
    # Rename 'app_id' to 'id' for consistency with how the app uses it for keys
    if 'app_id' in df_investpy.columns:
        df_investpy.rename(columns={'app_id': 'id'}, inplace=True)
        
    return df_investpy

# Historical data remains sample data
@st.cache_data
def load_historical_data(event_name):
    """
    Loads sample historical data for a given event name.
    """
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
    } # ... (rest of the sample data)
    for key in all_historical_data:
        if key.lower() in event_name.lower():
            df = all_historical_data[key].copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df
    return pd.DataFrame()


if __name__ == '__main__':
    print("Attempting to load economic data using investpy:")
    live_data = load_economic_data() # This will call fetch_economic_calendar_from_investpy
    
    if not live_data.empty:
        print("\nEconomic Data Sample (from investpy if successful):")
        print(live_data[['Timestamp', 'Currency', 'EventName', 'Impact', 'Forecast', 'Actual', 'Previous']].head())
        if not live_data.empty and 'Timestamp' in live_data.columns and pd.notna(live_data['Timestamp'].iloc[0]):
            print(f"\nFirst timestamp object type: {type(live_data['Timestamp'].iloc[0])}")
            print(f"First timestamp timezone info: {live_data['Timestamp'].iloc[0].tzinfo}")
    else:
        print("\nFailed to load economic data using investpy.")

    # nfp_hist = load_historical_data("Non-Farm Employment Change") # Test historical data
    # print("\nSample NFP Historical Data (still sample):")
    # print(nfp_hist.head())

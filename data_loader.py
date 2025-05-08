# data_loader.py
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import pytz # For timezone handling

# Define the base timezone for the source data (assuming EST is US/Eastern)
SOURCE_TIMEZONE = pytz.timezone('US/Eastern')

@st.cache_data
def load_economic_data():
    """
    Loads sample economic calendar data with timezone-aware timestamps.
    Timestamps are converted from string format (assumed to be in SOURCE_TIMEZONE)
    to timezone-aware datetime objects.
    """
    try:
        base_time_utc = datetime.now(pytz.utc) # Use UTC as a consistent base for generating sample times
        
        # Sample data with string timestamps (implicitly in SOURCE_TIMEZONE)
        # For demonstration, we'll make them appear as if they were originally EST strings
        # then convert them.
        data = [
            # EventName: Non-Farm Employment Change (High Impact)
            {"TimestampStr": (base_time_utc.astimezone(SOURCE_TIMEZONE) + timedelta(days=1, hours=2)).strftime("%Y-%m-%d %H:%M"), "Currency": "USD", "EventName": "Non-Farm Employment Change", "Impact": "High", "Previous": 175.0, "Forecast": 200.0, "Actual": np.nan},
            # EventName: Employment Change (High Impact)
            {"TimestampStr": (base_time_utc.astimezone(SOURCE_TIMEZONE) + timedelta(days=1, hours=2)).strftime("%Y-%m-%d %H:%M"), "Currency": "CAD", "EventName": "Employment Change", "Impact": "High", "Previous": -2.2, "Forecast": 15.0, "Actual": np.nan},
            # EventName: Unemployment Rate (High Impact)
            {"TimestampStr": (base_time_utc.astimezone(SOURCE_TIMEZONE) + timedelta(days=1, hours=2, minutes=30)).strftime("%Y-%m-%d %H:%M"), "Currency": "USD", "EventName": "Unemployment Rate", "Impact": "High", "Previous": 3.9, "Forecast": 3.9, "Actual": np.nan},
            # EventName: GDP m/m (Medium Impact)
            {"TimestampStr": (base_time_utc.astimezone(SOURCE_TIMEZONE) + timedelta(days=2, hours=0)).strftime("%Y-%m-%d %H:%M"), "Currency": "GBP", "EventName": "GDP m/m", "Impact": "Medium", "Previous": 0.1, "Forecast": 0.2, "Actual": np.nan},
            # EventName: ECB President Speaks (High Impact) - Qualitative
            {"TimestampStr": (base_time_utc.astimezone(SOURCE_TIMEZONE) + timedelta(days=2, hours=4)).strftime("%Y-%m-%d %H:%M"), "Currency": "EUR", "EventName": "ECB President Speaks", "Impact": "High", "Previous": np.nan, "Forecast": np.nan, "Actual": np.nan},
            # EventName: Core CPI m/m (High Impact)
            {"TimestampStr": (base_time_utc.astimezone(SOURCE_TIMEZONE) + timedelta(days=3, hours=2, minutes=30)).strftime("%Y-%m-%d %H:%M"), "Currency": "USD", "EventName": "Core CPI m/m", "Impact": "High", "Previous": 0.3, "Forecast": 0.3, "Actual": np.nan},
            # EventName: BoJ Policy Rate (High Impact)
            {"TimestampStr": (base_time_utc.astimezone(SOURCE_TIMEZONE) + timedelta(days=3, hours=4)).strftime("%Y-%m-%d %H:%M"), "Currency": "JPY", "EventName": "BoJ Policy Rate", "Impact": "High", "Previous": -0.1, "Forecast": -0.1, "Actual": np.nan},
            # EventName: Retail Sales m/m (Medium Impact)
            {"TimestampStr": (base_time_utc.astimezone(SOURCE_TIMEZONE) + timedelta(days=4, hours=1, minutes=15)).strftime("%Y-%m-%d %H:%M"), "Currency": "AUD", "EventName": "Retail Sales m/m", "Impact": "Medium", "Previous": -0.4, "Forecast": 0.3, "Actual": np.nan},
        ]
        
        df = pd.DataFrame(data)

        # Convert TimestampStr to timezone-aware datetime objects
        # 1. Parse string to naive datetime
        # 2. Localize to SOURCE_TIMEZONE
        df['Timestamp'] = df['TimestampStr'].apply(
            lambda x: SOURCE_TIMEZONE.localize(datetime.strptime(x, "%Y-%m-%d %H:%M")) if pd.notna(x) else pd.NaT
        )
        df.drop(columns=['TimestampStr'], inplace=True) # Drop the original string column
            
        numeric_cols = ['Previous', 'Forecast', 'Actual']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df['id'] = df.index # Unique ID for selection
        return df.sort_values(by='Timestamp').reset_index(drop=True) # Sort by time
    except Exception as e:
        st.error(f"🚨 Error loading economic data: {e}")
        # import traceback
        # st.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

@st.cache_data
def load_historical_data(event_name):
    """
    Loads sample historical data for a given event name.
    Dates are currently naive but could be made timezone-aware if source provides TZ.
    For simplicity, historical data dates are kept as naive datetimes for now.
    """
    # Generate sample dates relative to today, ensuring they are just dates (no time part for simplicity)
    today = datetime.now().date()
    
    all_historical_data = {
        "Non-Farm Employment Change": pd.DataFrame({
            'Date': [today - timedelta(days=30*i) for i in range(6, 0, -1)],
            'Actual': [150.0, 220.0, 180.0, 250.0, 160.0, 175.0],
            'Forecast': [160.0, 200.0, 190.0, 230.0, 180.0, 185.0],
            'Previous': [140.0, 150.0, 220.0, 180.0, 250.0, 160.0]
        }),
        "Unemployment Rate": pd.DataFrame({
            'Date': [today - timedelta(days=30*i) for i in range(6, 0, -1)],
            'Actual': [4.0, 3.8, 3.9, 3.7, 3.9, 3.9],
            'Forecast': [3.9, 3.8, 3.9, 3.8, 3.9, 3.9],
            'Previous': [4.1, 4.0, 3.8, 3.9, 3.7, 3.9]
        }),
        "Core CPI m/m": pd.DataFrame({
            'Date': [today - timedelta(days=30*i) for i in range(6, 0, -1)],
            'Actual': [0.2, 0.4, 0.3, 0.3, 0.5, 0.3],
            'Forecast': [0.3, 0.3, 0.3, 0.4, 0.4, 0.3],
            'Previous': [0.1, 0.2, 0.4, 0.3, 0.3, 0.5]
        })
    }
    for key in all_historical_data:
        if key.lower() in event_name.lower():
            df = all_historical_data[key].copy()
            df['Date'] = pd.to_datetime(df['Date']) # Ensure 'Date' is datetime
            df.set_index('Date', inplace=True)
            return df
    return pd.DataFrame()

if __name__ == '__main__':
    sample_data = load_economic_data()
    print("Sample Economic Data (with Timezone-Aware Timestamps):")
    print(sample_data[['Timestamp', 'EventName']].head())
    if not sample_data.empty and pd.notna(sample_data['Timestamp'].iloc[0]):
        print(f"\nFirst timestamp object type: {type(sample_data['Timestamp'].iloc[0])}")
        print(f"First timestamp timezone info: {sample_data['Timestamp'].iloc[0].tzinfo}")

    nfp_hist = load_historical_data("Non-Farm Employment Change")
    print("\nSample NFP Historical Data:")
    print(nfp_hist.head())

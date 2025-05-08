# data_loader.py
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import pytz # For timezone handling

# Define the base timezone for the source data (assuming EST is US/Eastern for sample data)
# This is relevant for how we initially define our sample timestamps if they were strings.
# Since we are generating them programmatically, we will make them UTC then convert for display if needed.
SOURCE_TIMEZONE_NAME = 'US/Eastern' # For context if we were parsing strings
SOURCE_TIMEZONE = pytz.timezone(SOURCE_TIMEZONE_NAME)

@st.cache_data
def load_economic_data():
    """
    Loads sample economic calendar data with timezone-aware timestamps.
    The timestamps are generated as UTC and then can be converted to any target timezone in the app.
    """
    try:
        # Base time for generating sample data, set to UTC for consistency
        base_time_utc = datetime.now(pytz.utc)
        
        # Sample data structure
        data = [
            {"EventName": "Non-Farm Employment Change", "Currency": "USD", "Impact": "High", "Previous": 175.0, "Forecast": 200.0, "Actual": np.nan, "TimeOffsetDays": 1, "TimeOffsetHours": 14, "TimeOffsetMinutes": 30}, # Approx 9:30 AM EST if base is midnight EST
            {"EventName": "Employment Change", "Currency": "CAD", "Impact": "High", "Previous": -2.2, "Forecast": 15.0, "Actual": np.nan, "TimeOffsetDays": 1, "TimeOffsetHours": 14, "TimeOffsetMinutes": 30},
            {"EventName": "Unemployment Rate", "Currency": "USD", "Impact": "High", "Previous": 3.9, "Forecast": 3.9, "Actual": np.nan, "TimeOffsetDays": 1, "TimeOffsetHours": 14, "TimeOffsetMinutes": 30},
            {"EventName": "GDP m/m", "Currency": "GBP", "Impact": "Medium", "Previous": 0.1, "Forecast": 0.2, "Actual": np.nan, "TimeOffsetDays": 2, "TimeOffsetHours": 6, "TimeOffsetMinutes": 0}, # Early morning London
            {"EventName": "ECB President Speaks", "Currency": "EUR", "Impact": "High", "Previous": np.nan, "Forecast": np.nan, "Actual": np.nan, "TimeOffsetDays": 2, "TimeOffsetHours": 10, "TimeOffsetMinutes": 30}, # Mid-morning Frankfurt
            {"EventName": "Core CPI m/m", "Currency": "USD", "Impact": "High", "Previous": 0.3, "Forecast": 0.3, "Actual": np.nan, "TimeOffsetDays": 13, "TimeOffsetHours": 14, "TimeOffsetMinutes": 30}, # Approx 2 weeks later
            {"EventName": "BoJ Policy Rate", "Currency": "JPY", "Impact": "High", "Previous": -0.1, "Forecast": -0.1, "Actual": np.nan, "TimeOffsetDays": 3, "TimeOffsetHours": 3, "TimeOffsetMinutes": 0}, # Midnight Tokyo
            {"EventName": "Retail Sales m/m", "Currency": "AUD", "Impact": "Medium", "Previous": -0.4, "Forecast": 0.3, "Actual": np.nan, "TimeOffsetDays": 4, "TimeOffsetHours": 1, "TimeOffsetMinutes": 30}, # Early morning Sydney
        ]
        
        processed_data = []
        for item in data:
            # Generate timestamp in UTC
            event_time_utc = base_time_utc.replace(hour=0, minute=0, second=0, microsecond=0) + \
                             timedelta(days=item["TimeOffsetDays"], 
                                       hours=item["TimeOffsetHours"], 
                                       minutes=item["TimeOffsetMinutes"])
            processed_data.append({
                "Timestamp": event_time_utc, # Store as UTC
                "Currency": item["Currency"],
                "EventName": item["EventName"],
                "Impact": item["Impact"],
                "Previous": item["Previous"],
                "Forecast": item["Forecast"],
                "Actual": item["Actual"]
            })
            
        df = pd.DataFrame(processed_data)
            
        numeric_cols = ['Previous', 'Forecast', 'Actual']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df['id'] = df.index # Unique ID for selection
        return df.sort_values(by='Timestamp').reset_index(drop=True)
    except Exception as e:
        st.error(f"🚨 Error generating sample economic data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_historical_data(event_name):
    """
    Loads sample historical data for a given event name.
    Dates are naive datetimes (just dates).
    """
    today_date = datetime.now().date() # Use current date for relevance
    
    all_historical_data = {
        # Using more realistic NFP numbers (in thousands)
        "Non-Farm Employment Change": pd.DataFrame({
            'Date': [today_date - timedelta(days=30*i) for i in range(6, 0, -1)],
            'Actual': [187.0, 150.0, 275.0, 216.0, 353.0, 175.0], # Sample NFP data
            'Forecast': [170.0, 180.0, 190.0, 175.0, 185.0, 200.0],
            'Previous': [165.0, 187.0, 150.0, 275.0, 216.0, 353.0]
        }),
        "Unemployment Rate": pd.DataFrame({
            'Date': [today_date - timedelta(days=30*i) for i in range(6, 0, -1)],
            'Actual': [3.8, 3.9, 3.7, 3.7, 3.7, 3.9], # Sample Unemployment data
            'Forecast': [3.8, 3.8, 3.8, 3.7, 3.8, 3.9],
            'Previous': [3.7, 3.8, 3.9, 3.7, 3.7, 3.7]
        }),
         "Core CPI m/m": pd.DataFrame({
            'Date': [today_date - timedelta(days=30*i) for i in range(6, 0, -1)],
            'Actual': [0.3, 0.4, 0.4, 0.3, 0.3, 0.3], # Sample Core CPI m/m data
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
    sample_data = load_economic_data()
    print("Sample Economic Data (Simulated, Timezone-Aware UTC Timestamps):")
    print(sample_data[['Timestamp', 'EventName', 'Currency']].head())
    if not sample_data.empty and pd.notna(sample_data['Timestamp'].iloc[0]):
        print(f"\nFirst timestamp object type: {type(sample_data['Timestamp'].iloc[0])}")
        print(f"First timestamp timezone info: {sample_data['Timestamp'].iloc[0].tzinfo}")

    nfp_hist = load_historical_data("Non-Farm Employment Change")
    print("\nSample NFP Historical Data:")
    print(nfp_hist.head())

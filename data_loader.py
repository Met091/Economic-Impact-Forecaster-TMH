# data_loader.py
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

@st.cache_data # Cache the data to avoid reloading on every interaction
def load_economic_data():
    """
    Loads sample economic calendar data.
    In a real application, this would fetch data from an API or database.
    """
    try:
        base_time = datetime.now()
        data = [
            {"Timestamp": (base_time + timedelta(days=1, hours=2)).strftime("%Y-%m-%d %H:%M EST"), "Currency": "USD", "EventName": "Non-Farm Employment Change", "Impact": "High", "Previous": 175.0, "Forecast": 200.0, "Actual": np.nan},
            {"Timestamp": (base_time + timedelta(days=1, hours=2)).strftime("%Y-%m-%d %H:%M EST"), "Currency": "CAD", "EventName": "Employment Change", "Impact": "High", "Previous": -2.2, "Forecast": 15.0, "Actual": np.nan},
            {"Timestamp": (base_time + timedelta(days=1, hours=2, minutes=30)).strftime("%Y-%m-%d %H:%M EST"), "Currency": "USD", "EventName": "Unemployment Rate", "Impact": "High", "Previous": 3.9, "Forecast": 3.9, "Actual": np.nan},
            {"Timestamp": (base_time + timedelta(days=2, hours=0)).strftime("%Y-%m-%d %H:%M EST"), "Currency": "GBP", "EventName": "GDP m/m", "Impact": "Medium", "Previous": 0.1, "Forecast": 0.2, "Actual": np.nan},
            {"Timestamp": (base_time + timedelta(days=2, hours=4)).strftime("%Y-%m-%d %H:%M EST"), "Currency": "EUR", "EventName": "ECB President Speaks", "Impact": "High", "Previous": np.nan, "Forecast": np.nan, "Actual": np.nan},
            {"Timestamp": (base_time + timedelta(days=3, hours=2, minutes=30)).strftime("%Y-%m-%d %H:%M EST"), "Currency": "USD", "EventName": "Core CPI m/m", "Impact": "High", "Previous": 0.3, "Forecast": 0.3, "Actual": np.nan},
            {"Timestamp": (base_time + timedelta(days=3, hours=4)).strftime("%Y-%m-%d %H:%M EST"), "Currency": "JPY", "EventName": "BoJ Policy Rate", "Impact": "High", "Previous": -0.1, "Forecast": -0.1, "Actual": np.nan},
            {"Timestamp": (base_time + timedelta(days=4, hours=1, minutes=15)).strftime("%Y-%m-%d %H:%M EST"), "Currency": "AUD", "EventName": "Retail Sales m/m", "Impact": "Medium", "Previous": -0.4, "Forecast": 0.3, "Actual": np.nan},
        ]
        
        df = pd.DataFrame(data)
        
        numeric_cols = ['Previous', 'Forecast', 'Actual']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df['id'] = df.index # Unique ID for selection
        return df
    except Exception as e:
        st.error(f"🚨 Error loading economic data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_historical_data(event_name):
    """
    Loads sample historical data for a given event name.
    In a real application, this would fetch from a database or API.
    Returns a DataFrame with 'Date', 'Actual', 'Forecast', 'Previous'.
    """
    all_historical_data = {
        "Non-Farm Employment Change": pd.DataFrame({
            'Date': pd.to_datetime([datetime.now() - timedelta(days=30*i) for i in range(6, 0, -1)]),
            'Actual': [150.0, 220.0, 180.0, 250.0, 160.0, 175.0],
            'Forecast': [160.0, 200.0, 190.0, 230.0, 180.0, 185.0],
            'Previous': [140.0, 150.0, 220.0, 180.0, 250.0, 160.0]
        }),
        "Unemployment Rate": pd.DataFrame({
            'Date': pd.to_datetime([datetime.now() - timedelta(days=30*i) for i in range(6, 0, -1)]),
            'Actual': [4.0, 3.8, 3.9, 3.7, 3.9, 3.9],
            'Forecast': [3.9, 3.8, 3.9, 3.8, 3.9, 3.9],
            'Previous': [4.1, 4.0, 3.8, 3.9, 3.7, 3.9]
        }),
        "Core CPI m/m": pd.DataFrame({
            'Date': pd.to_datetime([datetime.now() - timedelta(days=30*i) for i in range(6, 0, -1)]),
            'Actual': [0.2, 0.4, 0.3, 0.3, 0.5, 0.3],
            'Forecast': [0.3, 0.3, 0.3, 0.4, 0.4, 0.3],
            'Previous': [0.1, 0.2, 0.4, 0.3, 0.3, 0.5]
        })
    }
    # Find a match (case-insensitive, partial)
    for key in all_historical_data:
        if key.lower() in event_name.lower():
            df = all_historical_data[key].copy()
            df.set_index('Date', inplace=True)
            return df
    return pd.DataFrame() # Return empty if no specific historical data found

if __name__ == '__main__':
    sample_data = load_economic_data()
    print("Sample Economic Data:")
    print(sample_data.head())
    
    nfp_hist = load_historical_data("Non-Farm Employment Change")
    print("\nSample NFP Historical Data:")
    print(nfp_hist)

    unemp_hist = load_historical_data("Unemployment Rate")
    print("\nSample Unemployment Rate Historical Data:")
    print(unemp_hist)

    random_hist = load_historical_data("Some Random Event")
    print("\nSample Random Event Historical Data (should be empty):")
    print(random_hist)

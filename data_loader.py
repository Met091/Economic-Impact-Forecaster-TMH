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
        # Sample data for demonstration
        # Timestamps are strings for now, will be converted
        # Using a slightly more diverse set of examples including None/NaN for Previous/Forecast
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
        
        # Ensure numeric columns are indeed numeric, coercing errors to NaN
        numeric_cols = ['Previous', 'Forecast', 'Actual']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Add a unique ID for selection if needed, using index for now
        df['id'] = df.index
        return df
    except Exception as e:
        st.error(f"Error loading economic data: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

if __name__ == '__main__':
    # For testing the data loader independently
    sample_data = load_economic_data()
    print("Sample Economic Data:")
    print(sample_data)
    print("\nData Types:")
    print(sample_data.dtypes)

# visualization.py
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime # Ensure datetime is imported

def plot_historical_trend(df_historical, event_name, indicator_type="normal"):
    """
    Plots the historical trend. Adapts to available columns ('Actual', 'Forecast', 'Previous').
    If only 'Actual' is present (e.g., from Alpha Vantage), it plots only that series.
    """
    if df_historical.empty:
        st.info(f"No historical data available to plot for '{event_name}'.")
        return

    # Determine which value columns are present
    available_value_vars = [col for col in ['Actual', 'Forecast', 'Previous'] if col in df_historical.columns]

    if not available_value_vars:
        st.info(f"No standard data series (Actual, Forecast, Previous) found in historical data for '{event_name}'.")
        return

    # Melt the DataFrame to long format for Altair using only available columns
    df_melted = df_historical.reset_index().melt(
        id_vars=['Date'], 
        value_vars=available_value_vars, # Use only present columns
        var_name='Measure', 
        value_name='Value'
    )
    df_melted.dropna(subset=['Value'], inplace=True) # Drop rows where 'Value' is NaN after melting

    if df_melted.empty:
        st.info(f"No valid data points to plot for '{event_name}' after processing.")
        return

    # Create the Altair chart
    try:
        y_axis_title = f"Value ({'Lower is Better' if indicator_type == 'inverted' else 'Higher is Better'})"

        chart = alt.Chart(df_melted).mark_line(point=True).encode(
            x=alt.X('Date:T', title='Date', axis=alt.Axis(format='%Y-%m-%d')),
            y=alt.Y('Value:Q', title=y_axis_title, scale=alt.Scale(zero=False)),
            color=alt.Color('Measure:N', legend=alt.Legend(title="Data Series")),
            tooltip=['Date:T', 'Measure:N', alt.Tooltip('Value:Q', format='.2f')] # Format tooltip value
        ).properties(
            title=f"Historical Trend for {event_name}",
            height=350 # Slightly increased height
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"🚨 Error plotting historical data for '{event_name}': {e}")

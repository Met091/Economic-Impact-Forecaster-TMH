# visualization.py
import streamlit as st
import pandas as pd
import altair as alt

def plot_historical_trend(df_historical, event_name, indicator_type="normal"):
    """
    Plots the historical trend of Actual, Forecast, and Previous values for an economic indicator.

    Args:
        df_historical (pd.DataFrame): DataFrame with 'Date' index and columns 'Actual', 'Forecast', 'Previous'.
        event_name (str): Name of the economic event for the chart title.
        indicator_type (str): "normal" or "inverted". Used for y-axis title context.
    """
    if df_historical.empty:
        st.info(f"No historical data available to plot for '{event_name}'.")
        return

    # Melt the DataFrame to long format for Altair
    df_melted = df_historical.reset_index().melt(
        id_vars=['Date'], 
        value_vars=['Actual', 'Forecast', 'Previous'],
        var_name='Measure', 
        value_name='Value'
    )

    # Create the Altair chart
    try:
        y_axis_title = f"Value ({'Lower is Better' if indicator_type == 'inverted' else 'Higher is Better'})"

        chart = alt.Chart(df_melted).mark_line(point=True).encode(
            x=alt.X('Date:T', title='Date', axis=alt.Axis(format='%Y-%m-%d')),
            y=alt.Y('Value:Q', title=y_axis_title, scale=alt.Scale(zero=False)), # zero=False often better for economic data
            color=alt.Color('Measure:N', legend=alt.Legend(title="Data Series")),
            tooltip=['Date:T', 'Measure:N', 'Value:Q']
        ).properties(
            title=f"Historical Trend for {event_name}",
            height=300
        ).interactive() # Allows zoom and pan

        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"🚨 Error plotting historical data for '{event_name}': {e}")
        # For debugging:
        # import traceback
        # st.error(f"Traceback: {traceback.format_exc()}")

if __name__ == '__main__':
    # Sample data for testing the plotting function
    sample_event_name = "Test Indicator (Higher is Better)"
    sample_hist_data_normal = pd.DataFrame({
        'Date': pd.to_datetime([datetime.now() - timedelta(days=30*i) for i in range(6, 0, -1)]),
        'Actual': [10, 12, 11, 13, 10, 12],
        'Forecast': [11, 11, 12, 12, 11, 11],
        'Previous': [9, 10, 12, 11, 13, 10]
    }).set_index('Date')

    sample_event_name_inverted = "Test Indicator (Lower is Better)"
    sample_hist_data_inverted = pd.DataFrame({
        'Date': pd.to_datetime([datetime.now() - timedelta(days=30*i) for i in range(6, 0, -1)]),
        'Actual': [5.0, 4.8, 4.9, 4.7, 4.9, 4.8],
        'Forecast': [4.9, 4.8, 4.9, 4.8, 4.9, 4.9],
        'Previous': [5.1, 5.0, 4.8, 4.9, 4.7, 4.9]
    }).set_index('Date')

    st.header("Test Plot Normal")
    plot_historical_trend(sample_hist_data_normal, sample_event_name)
    
    st.header("Test Plot Inverted")
    plot_historical_trend(sample_hist_data_inverted, sample_event_name_inverted, indicator_type="inverted")

    st.header("Test Plot Empty Data")
    plot_historical_trend(pd.DataFrame(), "Empty Data Event")

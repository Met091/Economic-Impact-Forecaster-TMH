# app.py
import streamlit as st
import pandas as pd
import numpy as np
from data_loader import load_economic_data, load_historical_data
from strategy_engine import (
    predict_actual_condition_for_outcome, 
    infer_market_outlook_from_data,
    classify_actual_release,
    get_indicator_properties # Import to get indicator type for plot
)
from visualization import plot_historical_trend

# --- Page Configuration ---
st.set_page_config(
    page_title="Economic Impact Forecaster V3",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Data ---
# This is cached in data_loader.py
economic_df_master = load_economic_data()

# --- Application Title ---
st.title("🌍 Economic Impact Forecaster V3")
st.markdown("""
Analyze economic data releases by currency, view historical trends, and simulate impacts of hypothetical 'Actual' values.
The system infers market bias (Forecast vs. Previous) and helps interpret outcomes.
""")

# --- Main Application Logic ---
if economic_df_master.empty:
    st.error("🚨 Failed to load economic data. Please check `data_loader.py` or the data source.")
else:
    # --- Sidebar ---
    st.sidebar.header("⚙️ Configuration")

    # --- Currency Filter ---
    st.sidebar.subheader("💱 Currency Filter")
    # Get unique, sorted currencies from the master DataFrame, handling potential NaNs
    available_currencies = sorted([curr for curr in economic_df_master['Currency'].unique() if pd.notna(curr)])
    
    # Add "All" option
    currency_options = ["All Currencies"] + available_currencies
    selected_currencies = st.sidebar.multiselect(
        "Select Currencies:",
        options=currency_options,
        default=["All Currencies"] # Default to showing all
    )

    # Filter DataFrame based on selected currencies
    if "All Currencies" in selected_currencies or not selected_currencies: # If "All" is selected or nothing is selected, show all
        economic_df_filtered = economic_df_master.copy()
    else:
        economic_df_filtered = economic_df_master[economic_df_master['Currency'].isin(selected_currencies)].copy()

    st.sidebar.markdown("---")
    st.sidebar.header("🗓️ Event Selection")

    if economic_df_filtered.empty:
        st.sidebar.warning("No events match the selected currency filter.")
        # Display a message in the main panel as well, or disable tabs
        st.error("No economic events found for the selected currency/currencies. Please adjust the filter in the sidebar.")
    else:
        # Create a display name for the selectbox using the filtered DataFrame
        economic_df_filtered['display_name'] = economic_df_filtered.apply(
            lambda row: f"{row['Timestamp']} - {row['Currency']} - {row['EventName']}"
            if pd.notna(row['Currency']) and pd.notna(row['EventName'])
            else f"{row['Timestamp']} - Event details missing",
            axis=1
        )
        event_options = economic_df_filtered['display_name'].tolist()
        
        # Check if there are any event options before trying to display the selectbox
        if not event_options:
            st.sidebar.warning("No events available for selection with the current filter.")
            st.error("No economic events available for selection. Please adjust the currency filter.")
        else:
            selected_event_display_name = st.sidebar.selectbox(
                "Select Economic Event:",
                options=event_options,
                index=0 # Default to the first event in the filtered list
            )
            selected_event_row = economic_df_filtered[economic_df_filtered['display_name'] == selected_event_display_name].iloc[0]
            
            # --- Event Details Display in Sidebar ---
            st.sidebar.markdown("---")
            st.sidebar.subheader(f"Selected: {selected_event_row['EventName']}")
            st.sidebar.caption(f"Currency: {selected_event_row['Currency']} | Impact: {selected_event_row['Impact']} | Time: {selected_event_row['Timestamp']}")
            
            previous_val = selected_event_row['Previous']
            forecast_val = selected_event_row['Forecast']
            event_name_str = str(selected_event_row['EventName'])
            currency_str = str(selected_event_row['Currency'])

            st.sidebar.markdown(f"**Previous:** `{previous_val if pd.notna(previous_val) else 'N/A'}`")
            st.sidebar.markdown(f"**Forecast:** `{forecast_val if pd.notna(forecast_val) else 'N/A'}`")

            # --- Tabs for Main Content ---
            tab1, tab2, tab3 = st.tabs(["🔍 Interpretation & Outlook", "📈 Historical Trends", "🔬 Simulate Actual Release"])

            with tab1:
                st.header(f"🔍 Interpretation for: {event_name_str} ({currency_str})")
                
                inferred_outcome = infer_market_outlook_from_data(
                    previous_val,
                    forecast_val,
                    event_name_str
                )
                st.info(f"System-Inferred Bias (Forecast vs. Previous): **{inferred_outcome}** for {currency_str}")

                st.subheader("🎯 Desired Market Outcome Analysis")
                outcome_options_list = ["Bullish", "Bearish", "Consolidating"]
                try:
                    # Handle complex inferred_outcome strings like "Consolidating (Qualitative...)"
                    default_outcome_index = 2 # Default to Consolidating
                    if "bullish" in inferred_outcome.lower(): default_outcome_index = 0
                    elif "bearish" in inferred_outcome.lower(): default_outcome_index = 1
                except ValueError: 
                    default_outcome_index = 2 
                
                desired_outcome = st.radio(
                    f"Select desired outcome for {currency_str} to analyze:",
                    options=outcome_options_list,
                    index=default_outcome_index,
                    key=f"outcome_radio_{selected_event_row['id']}",
                    horizontal=True
                )

                prediction_text = predict_actual_condition_for_outcome(
                    previous_val,
                    forecast_val,
                    desired_outcome,
                    currency_str,
                    event_name_str
                )
                outcome_color_map = {
                    "Bullish": "#1E4620", "Bearish": "#541B1B", "Consolidating": "#333333",
                    "Qualitative": "#2E4053", "Indeterminate": "#4A235A", "Error": "#641E16"
                }
                bg_color = outcome_color_map.get(desired_outcome, "#333333")
                st.markdown(f"<div style='background-color: {bg_color}; color: #FAFAFA; padding: 15px; border-radius: 8px; border: 1px solid #4F4F4F; margin-top:10px;'>{prediction_text}</div>", unsafe_allow_html=True)

            with tab2:
                st.header(f"📈 Historical Trends for: {event_name_str}")
                df_hist = load_historical_data(event_name_str) # Assumes load_historical_data is not currency specific yet
                if not df_hist.empty:
                    indicator_props = get_indicator_properties(event_name_str)
                    plot_historical_trend(df_hist, event_name_str, indicator_props.get("type", "normal"))
                else:
                    st.info(f"No specific historical data found for '{event_name_str}'.")

            with tab3:
                st.header(f"🔬 Simulate Actual Release Impact for: {event_name_str}")
                st.markdown("Enter a hypothetical 'Actual' value to see how it might be classified.")

                indicator_props_sim = get_indicator_properties(event_name_str)
                unit_sim = indicator_props_sim.get("unit", "")
                
                step_value = 0.1 if "%" in unit_sim else 1.0 if "K" in unit_sim else 0.01
                if indicator_props_sim["type"] == "qualitative":
                     st.warning(f"'{event_name_str}' is a qualitative event. Numerical simulation is not applicable.")
                else:
                    actual_input_val_default = forecast_val if pd.notna(forecast_val) else (previous_val if pd.notna(previous_val) else 0.0)
                    
                    hypothetical_actual = st.number_input(
                        f"Enter Hypothetical 'Actual' Value ({unit_sim}):",
                        value=float(actual_input_val_default) if pd.notna(actual_input_val_default) else 0.0,
                        step=step_value,
                        format="%.2f",
                        key=f"actual_input_{selected_event_row['id']}"
                    )

                    if st.button("Classify Hypothetical Actual", key=f"classify_btn_{selected_event_row['id']}", use_container_width=True):
                        if hypothetical_actual is not None: # number_input ensures it's a float
                            classification, explanation = classify_actual_release(
                                hypothetical_actual,
                                forecast_val,
                                previous_val,
                                event_name_str,
                                currency_str
                            )
                            
                            class_bg_color = outcome_color_map.get(classification, "#333333")
                            st.markdown(f"**Classification: <span style='color:{class_bg_color}; font-weight:bold;'>{classification}</span>**", unsafe_allow_html=True)
                            st.markdown(f"<div style='background-color: {class_bg_color}; color: #FAFAFA; padding: 10px; border-radius: 5px; border: 1px solid #4F4F4F; margin-top:5px;'>{explanation}</div>", unsafe_allow_html=True)
                        # No else needed as number_input handles non-valid entries by not updating value or showing Streamlit's default error.
    
    # --- Economic Calendar Overview in Sidebar ---
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Economic Calendar Overview")
    if not economic_df_filtered.empty:
        display_df_calendar = economic_df_filtered[['Timestamp', 'Currency', 'EventName', 'Impact', 'Previous', 'Forecast']].copy()
        display_df_calendar.rename(columns={'EventName': 'Event Name'}, inplace=True)

        st.sidebar.dataframe(
            display_df_calendar,
            column_config={
                "Timestamp": st.column_config.TextColumn("Time (EST)", width="medium"),
                "Currency": st.column_config.TextColumn("CCY", width="small"),
                "Event Name": st.column_config.TextColumn("Event", width="large"),
                "Impact": st.column_config.TextColumn("Impact", width="small"),
                "Previous": st.column_config.NumberColumn("Prev.", format="%.2f", width="small"),
                "Forecast": st.column_config.NumberColumn("Fcst.", format="%.2f", width="small"),
            },
            use_container_width=True,
            hide_index=True,
            height=300 
        )
    else:
        st.sidebar.info("No events to display based on the current currency filter.")


    # --- Footer & Disclaimer ---
    st.markdown("---")
    st.caption("""
    **Disclaimer:** This tool provides generalized interpretations based on common market reactions and should not be considered financial advice.
    Actual market movements can be influenced by a wide array of factors. The simulated data is for demonstration purposes only.
    Historical data is sample data. For live and comprehensive data, API integration is required.
    Latency: Data loading is cached. For live API data, consider asynchronous fetching or background updates for optimal performance.
    """)

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pytz # For timezone handling
from datetime import datetime

from data_loader import load_economic_data, load_historical_data
from strategy_engine import (
    predict_actual_condition_for_outcome, 
    infer_market_outlook_from_data,
    classify_actual_release,
    get_indicator_properties
)
from visualization import plot_historical_trend

# --- Page Configuration ---
st.set_page_config(
    page_title="Economic Impact Forecaster V4",
    page_icon="🕰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper function for timezone conversion ---
def convert_and_format_time(dt_object, target_tz_str, fmt="%Y-%m-%d %H:%M %Z"):
    """Converts a datetime object to a target timezone and formats it."""
    if pd.isna(dt_object) or not isinstance(dt_object, datetime):
        return "N/A"
    try:
        target_tz = pytz.timezone(target_tz_str)
        # If dt_object is naive, assume it's UTC (or handle as error/default)
        # However, data_loader should now provide timezone-aware objects.
        if dt_object.tzinfo is None:
             # This case should ideally not happen if data_loader is correct
            dt_object = pytz.utc.localize(dt_object) 
        return dt_object.astimezone(target_tz).strftime(fmt)
    except Exception as e:
        # st.error(f"Error converting time: {e}") # Avoid flooding UI with errors
        # print(f"Error converting time: {dt_object} to {target_tz_str} - {e}")
        return "Invalid Time"


# --- Load Data ---
economic_df_master = load_economic_data()

# --- Application Title ---
st.title("🕰️ Economic Impact Forecaster V4")
st.markdown("""
Select your timezone, filter by currency, analyze economic data releases, view historical trends, and simulate impacts.
""")

# --- Main Application Logic ---
if economic_df_master.empty:
    st.error("🚨 Failed to load economic data. Please check `data_loader.py` or the data source.")
else:
    # --- Sidebar ---
    st.sidebar.header("⚙️ Configuration")

    # --- Timezone Selection ---
    st.sidebar.subheader("🌐 Timezone")
    common_timezones = pytz.common_timezones
    # Try to guess user's local timezone or default to UTC / US/Eastern
    default_tz = 'UTC'
    try:
        # This is a browser-dependent feature and might not always work.
        # It's also not directly available in Streamlit's Python backend.
        # For a robust solution, would need JavaScript or a user profile setting.
        # Defaulting to a common one like UTC or 'US/Eastern'.
        if 'selected_timezone' not in st.session_state:
            st.session_state.selected_timezone = 'US/Eastern' # Or 'UTC'
    except Exception:
        st.session_state.selected_timezone = 'US/Eastern'

    selected_tz_name = st.sidebar.selectbox(
        "Select Display Timezone:",
        options=common_timezones,
        index=common_timezones.index(st.session_state.selected_timezone) if st.session_state.selected_timezone in common_timezones else common_timezones.index('US/Eastern'),
        key="selected_timezone_widget" # Use a distinct key if needed
    )
    # Update session state if changed by widget
    st.session_state.selected_timezone = selected_tz_name


    # --- Currency Filter ---
    st.sidebar.subheader("💱 Currency Filter")
    available_currencies = sorted([curr for curr in economic_df_master['Currency'].unique() if pd.notna(curr)])
    currency_options = ["All Currencies"] + available_currencies
    
    if 'selected_currencies_filter' not in st.session_state:
        st.session_state.selected_currencies_filter = ["All Currencies"]

    selected_currencies = st.sidebar.multiselect(
        "Select Currencies:",
        options=currency_options,
        default=st.session_state.selected_currencies_filter,
        key="selected_currencies_widget"
    )
    st.session_state.selected_currencies_filter = selected_currencies


    # Filter DataFrame based on selected currencies
    if "All Currencies" in selected_currencies or not selected_currencies:
        economic_df_filtered = economic_df_master.copy()
    else:
        economic_df_filtered = economic_df_master[economic_df_master['Currency'].isin(selected_currencies)].copy()

    st.sidebar.markdown("---")
    st.sidebar.header("🗓️ Event Selection")

    if economic_df_filtered.empty:
        st.sidebar.warning("No events match the selected filters.")
        st.error("No economic events found for the selected filters. Please adjust filters in the sidebar.")
    else:
        # Create a display name for the selectbox using the filtered DataFrame AND selected timezone
        economic_df_filtered['display_name'] = economic_df_filtered.apply(
            lambda row: (f"{convert_and_format_time(row['Timestamp'], selected_tz_name, '%Y-%m-%d %H:%M')} "
                         f"({pytz.timezone(selected_tz_name).localize(datetime.now()).strftime('%Z')}) - " # Show TZ abbreviation
                         f"{row['Currency']} - {row['EventName']}")
            if pd.notna(row['Currency']) and pd.notna(row['EventName']) and pd.notna(row['Timestamp'])
            else f"Invalid Event Data",
            axis=1
        )
        event_options = economic_df_filtered['display_name'].tolist()
        
        if not event_options:
            st.sidebar.warning("No events available for selection with the current filters.")
            st.error("No economic events available for selection. Please adjust the filters.")
        else:
            # Manage selectbox state to avoid issues on filter changes
            current_event_selection_key = "current_event_selectbox"
            if current_event_selection_key not in st.session_state or st.session_state[current_event_selection_key] not in event_options:
                st.session_state[current_event_selection_key] = event_options[0] if event_options else None
            
            selected_event_display_name = st.sidebar.selectbox(
                "Select Economic Event:",
                options=event_options,
                key=current_event_selection_key
            )
            
            if selected_event_display_name is None: # Should not happen if event_options is not empty
                st.error("Error: No event selected.")
                st.stop()

            selected_event_row = economic_df_filtered[economic_df_filtered['display_name'] == selected_event_display_name].iloc[0]
            
            st.sidebar.markdown("---")
            st.sidebar.subheader(f"Selected: {selected_event_row['EventName']}")
            
            # Display event time in selected timezone
            formatted_event_time = convert_and_format_time(selected_event_row['Timestamp'], selected_tz_name)
            st.sidebar.caption(f"Currency: {selected_event_row['Currency']} | Impact: {selected_event_row['Impact']} | Time: {formatted_event_time}")
            
            previous_val = selected_event_row['Previous']
            forecast_val = selected_event_row['Forecast']
            event_name_str = str(selected_event_row['EventName'])
            currency_str = str(selected_event_row['Currency'])

            st.sidebar.markdown(f"**Previous:** `{previous_val if pd.notna(previous_val) else 'N/A'}`")
            st.sidebar.markdown(f"**Forecast:** `{forecast_val if pd.notna(forecast_val) else 'N/A'}`")

            tab1, tab2, tab3 = st.tabs(["🔍 Interpretation & Outlook", "📈 Historical Trends", "🔬 Simulate Actual Release"])

            with tab1:
                st.header(f"🔍 Interpretation for: {event_name_str} ({currency_str})")
                # ... (rest of tab1 logic unchanged, uses event_name_str, currency_str, etc.)
                inferred_outcome = infer_market_outlook_from_data(
                    previous_val,
                    forecast_val,
                    event_name_str
                )
                st.info(f"System-Inferred Bias (Forecast vs. Previous): **{inferred_outcome}** for {currency_str}")

                st.subheader("🎯 Desired Market Outcome Analysis")
                outcome_options_list = ["Bullish", "Bearish", "Consolidating"]
                try:
                    default_outcome_index = 2 
                    if "bullish" in inferred_outcome.lower(): default_outcome_index = 0
                    elif "bearish" in inferred_outcome.lower(): default_outcome_index = 1
                except ValueError: 
                    default_outcome_index = 2 
                
                desired_outcome = st.radio(
                    f"Select desired outcome for {currency_str} to analyze:",
                    options=outcome_options_list,
                    index=default_outcome_index,
                    key=f"outcome_radio_{selected_event_row['id']}", # id should be unique
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
                bg_color = outcome_color_map.get(desired_outcome, "#333333") # Default for complex strings
                if "Bullish" in desired_outcome: bg_color = outcome_color_map["Bullish"] # Simplified check
                elif "Bearish" in desired_outcome: bg_color = outcome_color_map["Bearish"]
                elif "Consolidating" in desired_outcome: bg_color = outcome_color_map["Consolidating"]


                st.markdown(f"<div style='background-color: {bg_color}; color: #FAFAFA; padding: 15px; border-radius: 8px; border: 1px solid #4F4F4F; margin-top:10px;'>{prediction_text}</div>", unsafe_allow_html=True)


            with tab2:
                st.header(f"📈 Historical Trends for: {event_name_str}")
                # Historical data dates are typically just dates, not specific times, so TZ conversion might not be critical here
                # unless the source data provides specific times for historical releases.
                df_hist = load_historical_data(event_name_str) 
                if not df_hist.empty:
                    indicator_props = get_indicator_properties(event_name_str)
                    plot_historical_trend(df_hist, event_name_str, indicator_props.get("type", "normal"))
                else:
                    st.info(f"No specific historical data found for '{event_name_str}'.")

            with tab3:
                st.header(f"🔬 Simulate Actual Release Impact for: {event_name_str}")
                # ... (rest of tab3 logic unchanged)
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
                        if hypothetical_actual is not None: 
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

    # --- Economic Calendar Overview in Sidebar ---
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Economic Calendar Overview")
    if not economic_df_filtered.empty:
        # Create a temporary column for display with converted times
        calendar_display_df = economic_df_filtered[['Timestamp', 'Currency', 'EventName', 'Impact', 'Previous', 'Forecast']].copy()
        calendar_display_df['FormattedTimestamp'] = calendar_display_df['Timestamp'].apply(
            lambda x: convert_and_format_time(x, selected_tz_name, "%Y-%m-%d %H:%M %Z")
        )
        
        # Select and rename columns for the final display
        calendar_display_df = calendar_display_df[['FormattedTimestamp', 'Currency', 'EventName', 'Impact', 'Previous', 'Forecast']]
        calendar_display_df.rename(columns={'FormattedTimestamp': 'Time', 'EventName': 'Event Name'}, inplace=True)


        st.sidebar.dataframe(
            calendar_display_df,
            column_config={
                "Time": st.column_config.TextColumn("Time", width="medium"), # Will show selected TZ
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
        st.sidebar.info("No events to display based on the current filters.")

    st.markdown("---")
    st.caption("""
    **Disclaimer:** This tool provides generalized interpretations. Not financial advice. Data is simulated.
    Timezone conversion relies on `pytz`. Ensure system time and base data timezone are accurate.
    """)

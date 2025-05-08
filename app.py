# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pytz 
from datetime import datetime, date, timedelta # Added date, timedelta

# Assuming these modules are in the same directory
from data_loader import load_economic_data, load_historical_data
from strategy_engine import (
    predict_actual_condition_for_outcome,
    infer_market_outlook_from_data,
    classify_actual_release,
    get_indicator_properties,
    INDICATOR_CONFIG 
)
from visualization import plot_historical_trend

# --- Page Configuration ---
st.set_page_config(
    page_title="Economic Impact Forecaster V9 (Date Range)",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper function for timezone conversion ---
def convert_and_format_time(dt_object, target_tz_str, fmt="%Y-%m-%d %H:%M %Z"):
    # ... (function remains the same as V8) ...
    if pd.isna(dt_object) or not isinstance(dt_object, datetime):
        return "N/A"
    try:
        target_tz = pytz.timezone(target_tz_str)
        if dt_object.tzinfo is None or dt_object.tzinfo.utcoffset(dt_object) is None: 
             dt_object = pytz.utc.localize(dt_object)
        return dt_object.astimezone(target_tz).strftime(fmt)
    except Exception:
        return "Invalid Time"

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # --- Date Range Selection ---
    st.subheader("🗓️ Date Range for Calendar")
    today = date.today() # Current date: Thursday, May 8, 2025
    
    # Default: Current Week (Monday to Sunday)
    default_start_date = today - timedelta(days=today.weekday()) # Monday: 2025-05-05
    default_end_date = default_start_date + timedelta(days=6)    # Sunday: 2025-05-11

    # Initialize session state for dates if not already set
    if 'start_date_filter' not in st.session_state:
        st.session_state.start_date_filter = default_start_date
    if 'end_date_filter' not in st.session_state:
        st.session_state.end_date_filter = default_end_date

    col_start_date, col_end_date = st.columns(2)
    with col_start_date:
        start_date_input = st.date_input(
            "Start Date", 
            value=st.session_state.start_date_filter,
            key="start_date_widget"
        )
    with col_end_date:
        end_date_input = st.date_input(
            "End Date", 
            value=st.session_state.end_date_filter,
            min_value=start_date_input, # End date cannot be before start date
            key="end_date_widget"
        )
    
    # Update session state with new selections
    st.session_state.start_date_filter = start_date_input
    st.session_state.end_date_filter = end_date_input

    # --- Timezone Selection ---
    st.subheader("🌐 Timezone")
    # ... (Timezone selection code remains the same as V8) ...
    common_timezones = pytz.common_timezones
    default_tz_sidebar = 'US/Eastern' 
    if 'selected_timezone' not in st.session_state:
        st.session_state.selected_timezone = default_tz_sidebar
    selected_tz_name = st.selectbox(
        "Display Timezone:", # Shortened label
        options=common_timezones,
        index=common_timezones.index(st.session_state.selected_timezone) if st.session_state.selected_timezone in common_timezones else common_timezones.index(default_tz_sidebar),
        key="selected_timezone_widget"
    )
    st.session_state.selected_timezone = selected_tz_name

    # --- Currency Filter ---
    st.subheader("💱 Currency Filter")
    # ... (Currency filter code remains the same, but available_currencies will be based on newly fetched data) ...
    # Note: available_currencies will be populated after data is loaded.
    # This means the currency filter options might briefly be out of sync if data changes significantly.
    # For a more robust solution, one might update currency options after data load.

# --- Load Data based on selected date range ---
# Pass the selected dates from session state to the data loader
# st.info message moved here to show what dates are being used for fetching
st.info(f"Fetching economic data from {st.session_state.start_date_filter.strftime('%Y-%m-%d')} to {st.session_state.end_date_filter.strftime('%Y-%m-%d')}...")
economic_df_master = load_economic_data(st.session_state.start_date_filter, st.session_state.end_date_filter)

# --- Update Currency Filter Options based on loaded data ---
with st.sidebar: # Re-enter sidebar context to update currency filter
    if not economic_df_master.empty and 'Currency' in economic_df_master.columns:
        available_currencies = sorted([curr for curr in economic_df_master['Currency'].unique() if pd.notna(curr) and curr != ''])
    else: 
        available_currencies = ["USD", "EUR", "JPY", "GBP", "CAD", "AUD"] # Fallback

    currency_options = ["All Currencies"] + available_currencies
    
    # Preserve previous selection if possible, otherwise default
    current_currency_selection = st.session_state.get('selected_currencies_filter', ["All Currencies"])
    valid_current_selection = [c for c in current_currency_selection if c in currency_options]
    if not valid_current_selection and "All Currencies" not in current_currency_selection : # if previous selection is no longer valid
        default_currency_selection = ["All Currencies"]
    elif not valid_current_selection and "All Currencies" in current_currency_selection:
         default_currency_selection = ["All Currencies"]
    elif not valid_current_selection: # if empty
        default_currency_selection = ["All Currencies"]
    else:
        default_currency_selection = valid_current_selection


    selected_currencies = st.multiselect(
        "Select Currencies:",
        options=currency_options,
        default=default_currency_selection, # Use the potentially updated default
        key="selected_currencies_widget_updated" # Use a new key if re-rendering, or manage state carefully
    )
    st.session_state.selected_currencies_filter = selected_currencies


# --- Application Title ---
st.title("📅 Economic Impact Forecaster V9 (Date Range)") # Updated title
st.markdown("""
Select date range, timezone, and currency preferences. Then, choose an economic event to analyze.
**Calendar data is fetched from Investing.com using `investpy`. This method relies on web scraping and may be unstable.**
""")
if economic_df_master.empty: # Check after attempting to load with selected dates
    st.error("🚨 Failed to load economic data for the selected date range using `investpy`. Investing.com might be temporarily unavailable, its website structure may have changed, or there might be a network issue. Please try different dates or try again later.")


# --- Filter DataFrame ---
# ... (Filtering logic remains the same as V8) ...
if not economic_df_master.empty:
    if "All Currencies" in selected_currencies or not selected_currencies:
        economic_df_filtered = economic_df_master.copy()
    else:
        if 'Currency' in economic_df_master.columns:
            economic_df_filtered = economic_df_master[economic_df_master['Currency'].isin(selected_currencies)].copy()
        else:
            st.warning("Currency information not available in the fetched data to apply filter.")
            economic_df_filtered = economic_df_master.copy()
else:
    economic_df_filtered = pd.DataFrame() 


# --- Main Application Area ---
# ... (Rest of the main app layout and tab logic remains largely the same as V8) ...
if economic_df_master.empty:
    pass # Error message already shown above
elif economic_df_filtered.empty:
    st.warning("⚠️ No economic events match the selected filters. Please adjust your Date Range, Timezone, or Currency selection.")
else:
    # ... (Event Selection, Details, Tabs, Calendar Overview as in V8) ...
    col_event_selection, col_event_details = st.columns([2, 3])

    with col_event_selection:
        st.subheader("🗓️ Select Economic Event")
        if 'Timestamp' in economic_df_filtered.columns and pd.api.types.is_datetime64_any_dtype(economic_df_filtered['Timestamp']):
            economic_df_filtered['display_name'] = economic_df_filtered.apply(
                lambda row: (f"{convert_and_format_time(row['Timestamp'], selected_tz_name, '%Y-%m-%d %H:%M')} "
                             f"({pytz.timezone(selected_tz_name).localize(datetime.now()).strftime('%Z')}) - "
                             f"{row.get('Currency','N/A')} - {row.get('EventName','Unknown Event')}")
                if pd.notna(row.get('EventName')) else f"Invalid Event Data @ {convert_and_format_time(row.get('Timestamp'), selected_tz_name)}",
                axis=1
            )
        else: 
            economic_df_filtered['display_name'] = economic_df_filtered.apply(
                lambda row: f"Data Error - {row.get('Currency','N/A')} - {row.get('EventName','Unknown Event')}", axis=1
            )
            st.error("Timestamp data is missing or in an incorrect format from investpy.")

        event_options = economic_df_filtered['display_name'].tolist()

        current_event_selection_key = "current_event_selectbox_main"
        if current_event_selection_key not in st.session_state or st.session_state[current_event_selection_key] not in event_options:
            st.session_state[current_event_selection_key] = event_options[0] if event_options else None
        
        selected_event_display_name = st.selectbox(
            "Choose an event from the filtered list:",
            options=event_options, key=current_event_selection_key, label_visibility="collapsed"
        )

    if selected_event_display_name is None or selected_event_display_name.startswith("Invalid Event Data") or selected_event_display_name.startswith("Data Error"):
        st.error("🚨 No valid event selected or available with current filters. Please check data source or filters.")
        if not event_options: # If there are truly no options, stop to prevent errors.
             st.stop()
        else: # If there are options but none selected (shouldn't happen with default), try to recover or stop.
             selected_event_display_name = event_options[0] # Attempt to pick first valid one
             if selected_event_display_name is None: st.stop()


    selected_event_row = economic_df_filtered[economic_df_filtered['display_name'] == selected_event_display_name].iloc[0]
    
    previous_val = selected_event_row.get('Previous')
    forecast_val = selected_event_row.get('Forecast')
    actual_val = selected_event_row.get('Actual') 
    event_name_str = str(selected_event_row.get('EventName', 'N/A'))
    currency_str = str(selected_event_row.get('Currency', 'N/A'))
    impact_str = str(selected_event_row.get('Impact', 'N/A'))
    event_timestamp = selected_event_row.get('Timestamp')
    formatted_event_time = convert_and_format_time(event_timestamp, selected_tz_name)

    with col_event_details:
        st.subheader(f"🔍 Details for: {event_name_str}")
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        with detail_col1:
            st.metric(label="Currency", value=currency_str)
            st.metric(label="Previous", value=f"{previous_val:.2f}" if pd.notna(previous_val) else "N/A")
        with detail_col2:
            st.metric(label="Impact", value=impact_str)
            st.metric(label="Forecast", value=f"{forecast_val:.2f}" if pd.notna(forecast_val) else "N/A")
        with detail_col3:
            time_part = formatted_event_time.split(' ')[1] if formatted_event_time != "N/A" and ' ' in formatted_event_time else "N/A"
            date_part = formatted_event_time.split(' ')[0] if formatted_event_time != "N/A" and ' ' in formatted_event_time else formatted_event_time
            st.metric(label="Scheduled Time", value=time_part)
            st.caption(f"Date: {date_part}")
        if pd.notna(actual_val):
            st.metric(label="Actual", value=f"{actual_val:.2f}", delta_color="off")


    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🎯 Interpretation & Outlook", "📈 Historical Trends (Sample)", "🔬 Simulate Actual Release"])

    with tab1:
        # ... (Tab 1 logic as before) ...
        inferred_outcome = infer_market_outlook_from_data(
            previous_val, forecast_val, event_name_str
        )
        st.info(f"System-Inferred Bias (Forecast vs. Previous): **{inferred_outcome}** for {currency_str}")
        st.subheader("📊 Desired Market Outcome Analysis")
        outcome_options_list = ["Bullish", "Bearish", "Consolidating"]
        try:
            default_outcome_index = 2 
            if "bullish" in inferred_outcome.lower(): default_outcome_index = 0
            elif "bearish" in inferred_outcome.lower(): default_outcome_index = 1
        except ValueError: default_outcome_index = 2 
        
        desired_outcome = st.radio(
            f"Select desired outcome for {currency_str} to analyze:",
            options=outcome_options_list, index=default_outcome_index,
            key=f"outcome_radio_main_{selected_event_row.get('id', event_name_str + currency_str)}",
            horizontal=True
        )
        prediction_text = predict_actual_condition_for_outcome(
            previous_val, forecast_val, desired_outcome, currency_str, event_name_str
        )
        outcome_color_map = {
            "Bullish": "#1E4620", "Bearish": "#541B1B", "Consolidating": "#333333",
            "Qualitative": "#2E4053", "Indeterminate": "#4A235A", "Error": "#641E16"
        }
        bg_color = outcome_color_map.get(desired_outcome, "#333333")
        st.markdown(f"<div style='background-color: {bg_color}; color: #FAFAFA; padding: 15px; border-radius: 8px; border: 1px solid #4F4F4F; margin-top:10px;'>{prediction_text}</div>", unsafe_allow_html=True)


    with tab2: # Historical data is still sample
        # ... (Tab 2 logic as before) ...
        st.header(f"Historical Trends for: {event_name_str}")
        st.caption("Note: Historical data below is sample data.")
        df_hist = load_historical_data(event_name_str) 
        if not df_hist.empty:
            indicator_props = get_indicator_properties(event_name_str)
            plot_historical_trend(df_hist, event_name_str, indicator_props.get("type", "normal"))
        else:
            st.info(f"No specific sample historical data found for '{event_name_str}'.")

    with tab3:
        # ... (Tab 3 logic as before) ...
        st.header(f"Simulate Actual Release Impact for: {event_name_str}")
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
                step=step_value, format="%.2f", 
                key=f"actual_input_main_{selected_event_row.get('id', event_name_str + currency_str)}"
            )
            if st.button("Classify Hypothetical Actual", 
                         key=f"classify_btn_main_{selected_event_row.get('id', event_name_str + currency_str)}", 
                         use_container_width=True):
                classification, explanation = classify_actual_release(
                    hypothetical_actual, forecast_val, previous_val, event_name_str, currency_str
                )
                class_bg_color = outcome_color_map.get(classification, "#333333")
                st.markdown(f"**Classification: <span style='color:{class_bg_color}; font-weight:bold;'>{classification}</span>**", unsafe_allow_html=True)
                st.markdown(f"<div style='background-color: {class_bg_color}; color: #FAFAFA; padding: 10px; border-radius: 5px; border: 1px solid #4F4F4F; margin-top:5px;'>{explanation}</div>", unsafe_allow_html=True)


    st.markdown("---")
    with st.expander("🗓️ Full Economic Calendar Overview (Data via investpy)", expanded=False):
        # ... (Calendar overview logic as before) ...
        if not economic_df_filtered.empty:
            calendar_display_df = economic_df_filtered.copy()
            if 'Timestamp' in calendar_display_df.columns and pd.api.types.is_datetime64_any_dtype(calendar_display_df['Timestamp']):
                calendar_display_df['FormattedTimestamp'] = calendar_display_df['Timestamp'].apply(
                    lambda x: convert_and_format_time(x, selected_tz_name, "%Y-%m-%d %H:%M %Z")
                )
                display_cols = ['FormattedTimestamp', 'Currency', 'EventName', 'Impact', 'Previous', 'Forecast', 'Actual', 'Zone']
                calendar_display_df = calendar_display_df[[col for col in display_cols if col in calendar_display_df.columns]]
                calendar_display_df.rename(columns={'FormattedTimestamp': 'Time', 'EventName': 'Event Name'}, inplace=True)
            else: 
                calendar_display_df['Time'] = "Data Error"
                display_cols_err = ['Time', 'Currency', 'EventName', 'Impact', 'Previous', 'Forecast', 'Actual', 'Zone']
                calendar_display_df = calendar_display_df[[col for col in display_cols_err if col in calendar_display_df.columns]]

            st.dataframe( calendar_display_df, use_container_width=True, hide_index=True, height=400,
                column_config={
                    "Time": st.column_config.TextColumn("Time", width="medium"),
                    "Currency": st.column_config.TextColumn("CCY", width="small"),
                    "Event Name": st.column_config.TextColumn("Event", width="large"),
                    "Impact": st.column_config.TextColumn("Impact", width="small"),
                    "Previous": st.column_config.NumberColumn("Prev.", format="%.2f", width="small"),
                    "Forecast": st.column_config.NumberColumn("Fcst.", format="%.2f", width="small"),
                    "Actual": st.column_config.NumberColumn("Actual", format="%.2f", width="small"),
                    "Zone": st.column_config.TextColumn("Zone", width="medium"),
                }
            )
        else:
            st.info("No events to display based on the current filters or data availability from investpy.")


    st.markdown("---")
    st.caption("""
    **Disclaimer:** Generalized interpretations, not financial advice. Calendar data from Investing.com via `investpy` (unofficial, may be unstable). Historical data is sample.
    Timezone conversion relies on `pytz`. Data accuracy/availability depends on Investing.com.
    """)

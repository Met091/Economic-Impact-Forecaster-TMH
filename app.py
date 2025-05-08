# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, date, timedelta

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
    page_title="Economic Forecaster (Layout Fix)", # Updated Page Title
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Injection for Metric Size ---
st.markdown("""
<style>
div[data-testid="stMetricLabel"] > div { font-size: 0.8rem; }
div[data-testid="stMetricValue"] > div { font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)
# --- End Custom CSS ---

# --- Helper function for timezone conversion ---
def convert_and_format_time(dt_object, target_tz_str, fmt="%Y-%m-%d %I:%M %p %Z"):
    if pd.isna(dt_object) or not isinstance(dt_object, datetime): return "N/A"
    try:
        target_tz = pytz.timezone(target_tz_str)
        if dt_object.tzinfo is None or dt_object.tzinfo.utcoffset(dt_object) is None:
             dt_object = pytz.utc.localize(dt_object)
        return dt_object.astimezone(target_tz).strftime(fmt)
    except Exception: return "Invalid Time"

# --- Sidebar for Configuration ---
with st.sidebar:
    # ... (Sidebar configuration remains the same) ...
    st.header("⚙️ Configuration Filters")
    st.subheader("🗓️ Date Range")
    today = date.today(); default_start_date = today - timedelta(days=today.weekday()); default_end_date = default_start_date + timedelta(days=6)
    if 'start_date_filter' not in st.session_state: st.session_state.start_date_filter = default_start_date
    if 'end_date_filter' not in st.session_state: st.session_state.end_date_filter = default_end_date
    col_start_date, col_end_date = st.columns(2)
    with col_start_date: start_date_input = st.date_input("Start", value=st.session_state.start_date_filter, key="start_date_widget")
    with col_end_date: end_date_input = st.date_input("End", value=st.session_state.end_date_filter, min_value=start_date_input, key="end_date_widget")
    st.session_state.start_date_filter = start_date_input; st.session_state.end_date_filter = end_date_input
    st.subheader("🌐 Timezone")
    common_timezones = pytz.common_timezones; default_tz_sidebar = 'US/Eastern'
    if 'selected_timezone' not in st.session_state: st.session_state.selected_timezone = default_tz_sidebar
    selected_tz_name = st.selectbox("Display Timezone:", options=common_timezones, index=common_timezones.index(st.session_state.selected_timezone) if st.session_state.selected_timezone in common_timezones else common_timezones.index(default_tz_sidebar), key="selected_timezone_widget")
    st.session_state.selected_timezone = selected_tz_name

# --- Load Data ---
economic_df_master, data_status_message = load_economic_data(
    st.session_state.start_date_filter, st.session_state.end_date_filter
)

# --- Sidebar Config (Continued) ---
with st.sidebar:
    st.subheader("💱 Currencies")
    if not economic_df_master.empty and 'Currency' in economic_df_master.columns: available_currencies = sorted([curr for curr in economic_df_master['Currency'].unique() if pd.notna(curr) and curr != ''])
    else: available_currencies = ["USD", "EUR", "JPY", "GBP", "CAD", "AUD"]
    currency_options = ["All"] + available_currencies
    if 'selected_currencies_filter' not in st.session_state: st.session_state.selected_currencies_filter = ["All"]
    current_currency_selection = st.session_state.selected_currencies_filter; valid_current_selection_curr = [c for c in current_currency_selection if c in currency_options]; default_currency_sel = valid_current_selection_curr if valid_current_selection_curr else ["All"]
    selected_currencies = st.multiselect("Filter Currencies:", options=currency_options, default=default_currency_sel, key="selected_currencies_widget_updated")
    st.session_state.selected_currencies_filter = selected_currencies

    st.subheader("⚡ Impact Level")
    impact_level_options_std = ["High", "Medium", "Low"]
    if not economic_df_master.empty and 'Impact' in economic_df_master.columns:
        data_impact_values = sorted([str(imp) for imp in economic_df_master['Impact'].unique() if pd.notna(imp) and str(imp) != 'N/A'])
        combined_impact_options = []; [combined_impact_options.append(opt) for opt in impact_level_options_std if opt in data_impact_values and opt not in combined_impact_options]; [combined_impact_options.append(opt) for opt in data_impact_values if opt not in combined_impact_options]
        impact_filter_options = ["All"] + (combined_impact_options if combined_impact_options else impact_level_options_std)
    else: impact_filter_options = ["All"] + impact_level_options_std
    if 'selected_impact_filter' not in st.session_state: st.session_state.selected_impact_filter = ["High"] if "High" in impact_filter_options else ["All"]
    current_impact_selection = st.session_state.selected_impact_filter; valid_current_selection_imp = [i for i in current_impact_selection if i in impact_filter_options]; default_impact_sel = valid_current_selection_imp if valid_current_selection_imp else ["High"] if "High" in impact_filter_options else ["All"]
    selected_impacts = st.multiselect("Filter Impact:", options=impact_filter_options, default=default_impact_sel, key="selected_impact_widget")
    st.session_state.selected_impact_filter = selected_impacts

    st.markdown("---")
    st.caption(f"Calendar Status: {data_status_message}")
    if 'ALPHA_VANTAGE_API_KEY' not in st.secrets: st.caption("AV Key: Missing")
    else: st.caption("AV Key: Configured")

# --- Application Title & Info ---
st.title("✨ Economic Impact Forecaster V14")
st.markdown("Powered by Trading Mastery Hub")

# --- Apply Filters ---
economic_df_filtered = economic_df_master.copy()
if 'Currency' in economic_df_filtered.columns and not ("All" in selected_currencies or not selected_currencies) : economic_df_filtered = economic_df_filtered[economic_df_filtered['Currency'].isin(selected_currencies)]
if 'Impact' in economic_df_filtered.columns and not ("All" in selected_impacts or not selected_impacts): economic_df_filtered = economic_df_filtered[economic_df_filtered['Impact'].isin(selected_impacts)]

# --- Main Application Area ---
if economic_df_master.empty:
    if "Simulated" in data_status_message: st.warning("⚠️ Displaying simulated data as live data fetch failed.")
    else: st.error("🚨 Failed to load any economic data.")
elif economic_df_filtered.empty:
    st.warning("⚠️ No economic events match the selected filters.")
else:
    # --- Event Selection ---
    st.subheader("🗓️ Select Economic Event")
    # ... (Event selection logic remains the same) ...
    if 'Timestamp' in economic_df_filtered.columns and pd.api.types.is_datetime64_any_dtype(economic_df_filtered['Timestamp']):
        economic_df_filtered['display_name'] = economic_df_filtered.apply(lambda row: (f"{convert_and_format_time(row['Timestamp'], selected_tz_name)} - {row.get('Currency','N/A')} - {row.get('Impact','N/A')} - {row.get('EventName','Unknown Event')}"), axis=1)
    else: economic_df_filtered['display_name'] = economic_df_filtered.apply(lambda row: f"Data Error - {row.get('Currency','N/A')} - {row.get('EventName','Unknown Event')}", axis=1); st.error("Timestamp data issue.")
    event_options = economic_df_filtered['display_name'].tolist()
    current_event_selection_key = "current_event_selectbox_main"
    if not event_options: st.session_state[current_event_selection_key] = None
    elif current_event_selection_key not in st.session_state or st.session_state[current_event_selection_key] not in event_options: st.session_state[current_event_selection_key] = event_options[0]
    selected_event_display_name = st.selectbox("Choose an event from the filtered list:", options=event_options, key=current_event_selection_key, label_visibility="collapsed", index=0 if not event_options or st.session_state[current_event_selection_key] is None else event_options.index(st.session_state[current_event_selection_key]))

    if not event_options or selected_event_display_name is None or selected_event_display_name.startswith("Invalid") or selected_event_display_name.startswith("Data Error"):
        st.error("🚨 No valid event selected or available. Adjust filters."); st.stop()

    selected_event_row = economic_df_filtered[economic_df_filtered['display_name'] == selected_event_display_name].iloc[0]
    event_id_for_keys = selected_event_row.get('id', str(selected_event_row.get('EventName','')) + str(selected_event_row.get('Currency','')))
    previous_val, forecast_val, actual_val = selected_event_row.get('Previous'), selected_event_row.get('Forecast'), selected_event_row.get('Actual')
    event_name_str, currency_str, impact_str = str(selected_event_row.get('EventName', 'N/A')), str(selected_event_row.get('Currency', 'N/A')), str(selected_event_row.get('Impact', 'N/A'))
    event_timestamp = selected_event_row.get('Timestamp'); formatted_event_time = convert_and_format_time(event_timestamp, selected_tz_name)

    st.markdown("---") # Separator

    # --- Event Details (Improved Layout) ---
    st.subheader(f"🔍 Details: {event_name_str}")
    # Use columns to group related metrics for better spacing
    detail_cols = st.columns(4) # Use 4 columns for better distribution
    with detail_cols[0]:
        st.metric(label="Currency", value=currency_str)
        st.metric(label="Impact", value=impact_str)
    with detail_cols[1]:
        time_part, date_part = "N/A", formatted_event_time
        if formatted_event_time not in ["N/A", "Invalid Time"] and ' ' in formatted_event_time: parts = formatted_event_time.split(' '); date_part = parts[0]; time_part = f"{parts[1]} {parts[2]}" if len(parts) >=3 else parts[1]
        st.metric(label="Scheduled Time", value=time_part)
        st.caption(f"Date: {date_part}")
    with detail_cols[2]:
        st.metric(label="Previous", value=f"{previous_val:.2f}" if pd.notna(previous_val) else "N/A")
        st.metric(label="Forecast", value=f"{forecast_val:.2f}" if pd.notna(forecast_val) else "N/A")
    with detail_cols[3]:
         if pd.notna(actual_val):
             st.metric(label="Actual", value=f"{actual_val:.2f}", delta_color="off")
         else:
             # Optional: Add a placeholder or leave blank if no actual
             st.metric(label="Actual", value="N/A")


    # --- Interpretation & Outlook (Below Details) ---
    st.subheader("🎯 Interpretation & Outlook")
    inferred_outcome = infer_market_outlook_from_data(previous_val, forecast_val, event_name_str)
    st.info(f"System-Inferred Bias (Forecast vs. Previous): **{inferred_outcome}** for {currency_str}")
    st.markdown("**Desired Market Outcome Analysis**")
    outcome_options_list = ["Bullish", "Bearish", "Consolidating"]
    default_outcome_index = 2
    if inferred_outcome:
         if "bullish" in inferred_outcome.lower(): default_outcome_index = 0
         elif "bearish" in inferred_outcome.lower(): default_outcome_index = 1
    desired_outcome = st.radio(f"Select desired outcome for {currency_str}:", options=outcome_options_list, index=default_outcome_index, key=f"outcome_radio_main_{event_id_for_keys}", horizontal=True)
    prediction_text = predict_actual_condition_for_outcome(previous_val, forecast_val, desired_outcome, currency_str, event_name_str)
    outcome_color_map_desired = {"Bullish": "#1E4620", "Bearish": "#541B1B", "Consolidating": "#333333"}
    bg_color_desired = outcome_color_map_desired.get(desired_outcome, "#333333")
    st.markdown(f"<div style='background-color: {bg_color_desired}; color: #FAFAFA; padding: 15px; border-radius: 8px; border: 1px solid #4F4F4F; margin-top:10px;'>{prediction_text}</div>", unsafe_allow_html=True)


    st.markdown("---") # Separator before remaining tabs

    # --- Remaining Tabs ---
    tab2, tab3 = st.tabs(["📈 Historical Trends", "🔬 Simulate Actual Release"])

    with tab2: # Historical Trends Tab
        # ... (Tab 2 logic remains the same) ...
        st.header(f"Historical Trends for: {event_name_str}")
        is_av_source = False; # Determine if AV is source
        if 'ALPHA_VANTAGE_API_KEY' in st.secrets: event_to_av_map = {"Non-Farm Employment Change": {}, "Unemployment Rate": {}, "Core CPI m/m": {}, "CPI m/m": {}, "Retail Sales m/m": {}, "Real GDP": {}, "Treasury Yield": {}, "Federal Funds Rate": {}}; [is_av_source := True for key_event in event_to_av_map if key_event.lower() in event_name_str.lower()]
        df_hist = load_historical_data(event_name_str)
        if not df_hist.empty:
            if is_av_source and 'Forecast' not in df_hist.columns: st.caption(f"Displaying real historical 'Actual' values from Alpha Vantage for {event_name_str} (US Data).")
            else: st.caption(f"Displaying sample historical data for {event_name_str}.")
            indicator_props = get_indicator_properties(event_name_str); plot_historical_trend(df_hist, event_name_str, indicator_props.get("type", "normal"))
        else: st.info(f"No historical data found for '{event_name_str}'.")

    with tab3: # Simulate Actual Release Tab
        # ... (Tab 3 logic remains the same) ...
        st.header(f"Simulate Actual Release Impact for: {event_name_str}")
        st.markdown("Enter a hypothetical 'Actual' value to see how it might be classified using a nuanced scale.")
        indicator_props_sim = get_indicator_properties(event_name_str)
        unit_sim = indicator_props_sim.get("unit", ""); step_value = 0.1 if "%" in unit_sim else 1.0 if "K" in unit_sim else 0.01
        if indicator_props_sim["type"] == "qualitative": st.warning(f"'{event_name_str}' is qualitative. Numerical simulation not applicable.")
        else:
            actual_input_val_default = forecast_val if pd.notna(forecast_val) else (previous_val if pd.notna(previous_val) else 0.0)
            hypothetical_actual = st.number_input(f"Hypothetical 'Actual' ({unit_sim}):", value=float(actual_input_val_default) if pd.notna(actual_input_val_default) else 0.0, step=step_value, format="%.2f", key=f"actual_input_main_{event_id_for_keys}")
            if st.button("Classify Hypothetical Actual", key=f"classify_btn_main_{event_id_for_keys}", use_container_width=True):
                classification, explanation = classify_actual_release(hypothetical_actual, forecast_val, previous_val, event_name_str, currency_str)
                nuanced_color_map = {"Strongly Bullish": "#145A32", "Mildly Bullish": "#27AE60", "Neutral/In-Line": "#808B96", "Mildly Bearish": "#E74C3C", "Strongly Bearish": "#922B21", "Qualitative": "#2E4053", "Indeterminate": "#4A235A", "Error": "#641E16"}
                class_bg_color = nuanced_color_map.get(classification, "#333333")
                st.markdown(f"**Classification: <span style='background-color:{class_bg_color}; color:white; padding: 2px 6px; border-radius: 4px; font-weight:bold;'>{classification}</span>**", unsafe_allow_html=True)
                st.markdown(f"<div style='background-color: #262730; color: #FAFAFA; border: 1px solid {class_bg_color}; padding: 10px; border-radius: 5px; margin-top:5px; white-space: pre-wrap;'>{explanation}</div>", unsafe_allow_html=True)

    # --- Economic Calendar Overview ---
    st.markdown("---")
    with st.expander("🗓️ Full Economic Calendar Overview (Filtered - Data via investpy)", expanded=False):
        # ... (Calendar display logic remains the same) ...
        if not economic_df_filtered.empty:
            calendar_display_df = economic_df_filtered.copy()
            if 'Timestamp' in calendar_display_df.columns and pd.api.types.is_datetime64_any_dtype(calendar_display_df['Timestamp']):
                calendar_display_df['FormattedTimestamp'] = calendar_display_df['Timestamp'].apply(lambda x: convert_and_format_time(x, selected_tz_name))
                display_cols = ['FormattedTimestamp', 'Currency', 'EventName', 'Impact', 'Previous', 'Forecast', 'Actual', 'Zone']
                calendar_display_df = calendar_display_df[[col for col in display_cols if col in calendar_display_df.columns]]
                calendar_display_df.rename(columns={'FormattedTimestamp': 'Time', 'EventName': 'Event Name'}, inplace=True)
            else: calendar_display_df['Time'] = "Data Error"; display_cols_err = ['Time', 'Currency', 'EventName', 'Impact', 'Previous', 'Forecast', 'Actual', 'Zone']; calendar_display_df = calendar_display_df[[col for col in display_cols_err if col in calendar_display_df.columns]]
            st.dataframe( calendar_display_df, use_container_width=True, hide_index=True, height=400, column_config={"Time": st.column_config.TextColumn("Time", width="medium"), "Currency": st.column_config.TextColumn("CCY", width="small"),"Event Name": st.column_config.TextColumn("Event", width="large"),"Impact": st.column_config.TextColumn("Impact", width="small"),"Previous": st.column_config.NumberColumn("Prev.", format="%.2f", width="small"),"Forecast": st.column_config.NumberColumn("Fcst.", format="%.2f", width="small"),"Actual": st.column_config.NumberColumn("Actual", format="%.2f", width="small"),"Zone": st.column_config.TextColumn("Zone", width="medium"),})
        else: st.info("No events to display based on current filters or investpy data availability.")

    # --- Footer ---
    st.markdown("---")
    st.caption("Disclaimer: Generalized interpretations, not financial advice. Calendar: investpy (unofficial). US Historicals: Alpha Vantage. Other historicals: sample. Data accuracy depends on sources.")


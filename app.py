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
)
from visualization import plot_historical_trend

# --- Page Configuration ---
st.set_page_config(
    page_title="Economic Impact Forecaster",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Function to Load External CSS ---
def load_css(file_name):
    """Loads an external CSS file into the Streamlit app."""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"🚨 CSS File Not Found: '{file_name}'. Please ensure it's in the correct directory.")
    except Exception as e:
        st.error(f"🚨 An error occurred while loading CSS: {e}")

# --- Load Custom CSS ---
load_css("style.css") # Assumes 'style.css' is in the same directory as app.py
# --- End Custom CSS Loading ---


# --- Helper function for timezone conversion ---
def convert_and_format_time(dt_object, target_tz_str, fmt="%Y-%m-%d %I:%M %p %Z"):
    """Converts a datetime object to a target timezone and formats it."""
    if pd.isna(dt_object) or not isinstance(dt_object, datetime):
        return "N/A"
    try:
        target_tz = pytz.timezone(target_tz_str)
        if dt_object.tzinfo is None or dt_object.tzinfo.utcoffset(dt_object) is None:
            dt_object = pytz.utc.localize(dt_object)
        return dt_object.astimezone(target_tz).strftime(fmt)
    except Exception as e:
        st.warning(f"⚠️ Time conversion issue for an event: {e}. Displaying as 'Invalid Time'.")
        return "Invalid Time"

# --- Sidebar for Configuration ---
with st.sidebar:
    st.image("https://placehold.co/300x100/0F1116/007BFF?text=Impact+Forecaster&font=roboto", use_column_width=True)
    st.markdown("## ⚙️ Configuration Filters")
    st.markdown("---")

    st.subheader("🗓️ Date Range")
    today = date.today()
    default_start_date = today - timedelta(days=today.weekday())
    default_end_date = default_start_date + timedelta(days=6)

    if 'start_date_filter' not in st.session_state:
        st.session_state.start_date_filter = default_start_date
    if 'end_date_filter' not in st.session_state:
        st.session_state.end_date_filter = default_end_date

    col_start_date, col_end_date = st.columns(2)
    with col_start_date:
        start_date_input = st.date_input(
            "Start",
            value=st.session_state.start_date_filter,
            key="start_date_widget",
            help="Select the start date for the economic calendar."
        )
    with col_end_date:
        end_date_input = st.date_input(
            "End",
            value=st.session_state.end_date_filter,
            min_value=start_date_input,
            key="end_date_widget",
            help="Select the end date for the economic calendar."
        )
    st.session_state.start_date_filter = start_date_input
    st.session_state.end_date_filter = end_date_input

    st.subheader("🌐 Timezone")
    common_timezones = pytz.common_timezones
    default_tz_sidebar = 'US/Eastern'
    if 'selected_timezone' not in st.session_state:
        st.session_state.selected_timezone = default_tz_sidebar

    selected_tz_name = st.selectbox(
        "Display Timezone:",
        options=common_timezones,
        index=common_timezones.index(st.session_state.selected_timezone) if st.session_state.selected_timezone in common_timezones else common_timezones.index(default_tz_sidebar),
        key="selected_timezone_widget",
        help="Choose the timezone for displaying event times."
    )
    st.session_state.selected_timezone = selected_tz_name
    st.markdown("---")

# --- Load Data (Cached in data_loader.py) ---
economic_df_master, data_status_message = load_economic_data(
    st.session_state.start_date_filter,
    st.session_state.end_date_filter
)

# --- Sidebar Config (Continued) ---
with st.sidebar:
    st.subheader("💱 Currencies")
    if not economic_df_master.empty and 'Currency' in economic_df_master.columns:
        available_currencies = sorted([curr for curr in economic_df_master['Currency'].unique() if pd.notna(curr) and curr != ''])
    else:
        available_currencies = ["USD", "EUR", "JPY", "GBP", "CAD", "AUD"]
    currency_options = ["All"] + available_currencies

    if 'selected_currencies_filter' not in st.session_state:
        st.session_state.selected_currencies_filter = ["All"]

    current_currency_selection = st.session_state.selected_currencies_filter
    valid_current_selection_curr = [c for c in current_currency_selection if c in currency_options]
    default_currency_sel = valid_current_selection_curr if valid_current_selection_curr else ["All"]

    selected_currencies = st.multiselect(
        "Filter Currencies:",
        options=currency_options,
        default=default_currency_sel,
        key="selected_currencies_widget_updated",
        help="Select currencies to filter events."
    )
    st.session_state.selected_currencies_filter = selected_currencies

    st.subheader("⚡ Impact Level")
    impact_level_options_std = ["High", "Medium", "Low"]
    if not economic_df_master.empty and 'Impact' in economic_df_master.columns:
        data_impact_values = sorted([str(imp) for imp in economic_df_master['Impact'].unique() if pd.notna(imp) and str(imp) != 'N/A'])
        combined_impact_options = []
        [combined_impact_options.append(opt) for opt in impact_level_options_std if opt in data_impact_values and opt not in combined_impact_options]
        [combined_impact_options.append(opt) for opt in data_impact_values if opt not in combined_impact_options]
        impact_filter_options = ["All"] + (combined_impact_options if combined_impact_options else impact_level_options_std)
    else:
        impact_filter_options = ["All"] + impact_level_options_std

    if 'selected_impact_filter' not in st.session_state:
        st.session_state.selected_impact_filter = ["High"] if "High" in impact_filter_options else ["All"]

    current_impact_selection = st.session_state.selected_impact_filter
    valid_current_selection_imp = [i for i in current_impact_selection if i in impact_filter_options]
    default_impact_sel = valid_current_selection_imp if valid_current_selection_imp else (["High"] if "High" in impact_filter_options else ["All"])

    selected_impacts = st.multiselect(
        "Filter Impact:",
        options=impact_filter_options,
        default=default_impact_sel,
        key="selected_impact_widget",
        help="Select impact levels to filter events."
    )
    st.session_state.selected_impact_filter = selected_impacts

    st.markdown("---")
    st.caption(f"Calendar Status: {data_status_message}")
    if 'ALPHA_VANTAGE_API_KEY' not in st.secrets:
        st.caption("AV Key: ⚠️ Missing (US Historicals limited)")
    else:
        st.caption("AV Key: 🔑 Configured")
    st.markdown("---")
    st.markdown("<div class='custom-info-box' style='background-color: rgba(0,123,255,0.05); border-left-color: #00A0B0;'>Tip: Use filters to narrow events. Click an event below for analysis.</div>", unsafe_allow_html=True)


# --- Application Title & Info ---
st.title("📊 Economic Impact Forecaster")
st.markdown("<div class='app-subtitle'>Powered by <strong>Trading Mastery Hub</strong> | <em>Navigating Economic Tides with Data</em></div>", unsafe_allow_html=True)
st.markdown("---")


# --- Apply Filters to Data ---
economic_df_filtered = economic_df_master.copy()
if 'Currency' in economic_df_filtered.columns and not ("All" in selected_currencies or not selected_currencies) :
    economic_df_filtered = economic_df_filtered[economic_df_filtered['Currency'].isin(selected_currencies)]
if 'Impact' in economic_df_filtered.columns and not ("All" in selected_impacts or not selected_impacts):
    economic_df_filtered = economic_df_filtered[economic_df_filtered['Impact'].isin(selected_impacts)]

# --- Main Application Area ---
if economic_df_master.empty:
    if "Simulated" in data_status_message:
        st.warning("⚠️ Displaying simulated data as live data fetch failed. Functionality may be limited.")
    else:
        st.error("🚨 Failed to load any economic data. Please check data sources or network connection, then try again.")
elif economic_df_filtered.empty:
    st.warning("⚠️ No economic events match the selected filters. Try adjusting the date range or filter criteria in the sidebar.")
else:
    # --- Event Selection Section ---
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.subheader("🗓️ Select Economic Event")
    if 'Timestamp' in economic_df_filtered.columns and pd.api.types.is_datetime64_any_dtype(economic_df_filtered['Timestamp']):
        economic_df_filtered['display_name'] = economic_df_filtered.apply(
            lambda row: (
                f"{convert_and_format_time(row['Timestamp'], selected_tz_name, fmt='%Y-%m-%d %I:%M %p')} | "
                f"{row.get('Currency','N/A')} | {row.get('Impact','N/A')} | {row.get('EventName','Unknown Event')}"
            ), axis=1
        )
    else:
        economic_df_filtered['display_name'] = economic_df_filtered.apply(
            lambda row: f"Data Error - {row.get('Currency','N/A')} - {row.get('EventName','Unknown Event')}", axis=1
        )
        st.error("🚨 Timestamp data issue. Event selection might be affected. Please check data integrity.")

    event_options = economic_df_filtered.sort_values(by='Timestamp')['display_name'].tolist()
    current_event_selection_key = "current_event_selectbox_main"

    if not event_options:
        st.session_state[current_event_selection_key] = None
    elif current_event_selection_key not in st.session_state or st.session_state[current_event_selection_key] not in event_options:
        st.session_state[current_event_selection_key] = event_options[0]

    selected_event_display_name = st.selectbox(
        "Choose an event from the filtered list:",
        options=event_options,
        key=current_event_selection_key,
        label_visibility="collapsed",
        index=0 if not event_options or st.session_state[current_event_selection_key] is None else event_options.index(st.session_state[current_event_selection_key]),
        help="Select an economic event to view its details and analysis."
    )
    st.markdown('</div>', unsafe_allow_html=True)


    if not event_options or selected_event_display_name is None or selected_event_display_name.startswith("Invalid") or selected_event_display_name.startswith("Data Error"):
        st.error("🚨 No valid event selected or available. Adjust filters or check data integrity.");
        st.stop()

    selected_event_row = economic_df_filtered[economic_df_filtered['display_name'] == selected_event_display_name].iloc[0]
    event_id_for_keys = str(selected_event_row.get('id', str(selected_event_row.get('EventName','')) + str(selected_event_row.get('Currency','')) + str(selected_event_row.get('Timestamp',''))))
    previous_val, forecast_val, actual_val = selected_event_row.get('Previous'), selected_event_row.get('Forecast'), selected_event_row.get('Actual')
    event_name_str, currency_str, impact_str = str(selected_event_row.get('EventName', 'N/A')), str(selected_event_row.get('Currency', 'N/A')), str(selected_event_row.get('Impact', 'N/A'))
    event_timestamp = selected_event_row.get('Timestamp')
    formatted_event_time = convert_and_format_time(event_timestamp, selected_tz_name)

    # --- Event Details Section ---
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.subheader(f"🔍 Details: {event_name_str}")
    detail_cols = st.columns([1.5, 1.5, 1, 1, 1])
    with detail_cols[0]:
        st.metric(label="Currency Pair", value=f"{currency_str}/USD" if currency_str != "USD" and currency_str != "N/A" else currency_str)
    with detail_cols[1]:
        st.metric(label="Impact Level", value=impact_str)
    
    time_part, date_part = "N/A", formatted_event_time
    if formatted_event_time not in ["N/A", "Invalid Time"] and ' ' in formatted_event_time:
        parts = formatted_event_time.split(' ')
        date_part = parts[0]
        time_part = f"{parts[1]} {parts[2]}" if len(parts) >=3 else parts[1]
    
    with detail_cols[2]:
        st.metric(label="Previous", value=f"{previous_val:.2f}" if pd.notna(previous_val) else "N/A")
    with detail_cols[3]:
        st.metric(label="Forecast", value=f"{forecast_val:.2f}" if pd.notna(forecast_val) else "N/A")
    with detail_cols[4]:
         st.metric(label="Actual", value=f"{actual_val:.2f}" if pd.notna(actual_val) else "Pending")
    
    st.caption(f"Scheduled Release: {date_part} at {time_part} ({selected_tz_name})")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Interpretation & Outlook Section ---
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.subheader("🎯 Interpretation & Outlook")
    inferred_outcome = infer_market_outlook_from_data(previous_val, forecast_val, event_name_str)
    st.markdown(f"<div class='custom-info-box'>System-Inferred Bias (Forecast vs. Previous): <strong>{inferred_outcome}</strong> for {currency_str}</div>", unsafe_allow_html=True)

    st.markdown("<h5>Desired Market Scenario Analysis</h5>", unsafe_allow_html=True)
    outcome_options_list = ["Bullish", "Bearish", "Consolidating"]
    default_outcome_index = 2
    if inferred_outcome:
         if "bullish" in inferred_outcome.lower(): default_outcome_index = 0
         elif "bearish" in inferred_outcome.lower(): default_outcome_index = 1

    desired_outcome = st.radio(
        f"Select desired outcome for {currency_str}:",
        options=outcome_options_list, index=default_outcome_index,
        key=f"outcome_radio_main_{event_id_for_keys}", horizontal=True,
        help="Choose your desired market direction to see the required 'Actual' conditions."
    )
    
    # Get the prediction as a list of strings
    prediction_points = predict_actual_condition_for_outcome(previous_val, forecast_val, desired_outcome, currency_str, event_name_str)
    
    outcome_color_map_desired = {"Bullish": "#28a745", "Bearish": "#dc3545", "Consolidating": "#6c757d"}
    bg_color_desired = outcome_color_map_desired.get(desired_outcome, "#6c757d")

    # Construct the HTML for the bullet list within the custom box
    prediction_html_list = "".join([f"<li>{point}</li>" for point in prediction_points])
    
    st.markdown(
        f"""
        <div class='custom-prediction-box' style='background-color: {bg_color_desired}; border-left: 5px solid {bg_color_desired};'>
            <ul style='margin-bottom: 0; padding-left: 20px;'> {prediction_html_list}
            </ul>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Tabs for Historical Data and Simulation ---
    tab_hist, tab_sim = st.tabs(["📈 Historical Trends", "🔬 Simulate Actual Release"])

    with tab_hist:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.subheader(f"Historical Trends for: {event_name_str}")
        is_av_source = False
        if 'ALPHA_VANTAGE_API_KEY' in st.secrets:
            event_to_av_map = {"Non-Farm Employment Change": {}, "Unemployment Rate": {}, "Core CPI m/m": {}, "CPI m/m": {}, "Retail Sales m/m": {}, "Real GDP": {}, "Treasury Yield": {}, "Federal Funds Rate": {}}
            if any(key_event.lower() in event_name_str.lower() for key_event in event_to_av_map):
                is_av_source = True

        df_hist = load_historical_data(event_name_str)
        if not df_hist.empty:
            caption_text = f"Displaying historical data for {event_name_str}. "
            if is_av_source and 'Forecast' not in df_hist.columns:
                caption_text += "Sourced from Alpha Vantage (US Data - 'Actual' values). Forecast/Previous may not be available from this source."
            else:
                caption_text += "May include sample data if live source is unavailable or lacks all fields."
            st.caption(caption_text)
            
            indicator_props = get_indicator_properties(event_name_str)
            plot_historical_trend(df_hist, event_name_str, indicator_props.get("type", "normal"))
        else:
            st.info(f"ℹ️ No historical data found or loaded for '{event_name_str}'. This may be due to data source limitations or the specific indicator selected.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_sim:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.subheader(f"Simulate Actual Release Impact for: {event_name_str}")
        st.markdown("Enter a hypothetical 'Actual' value to see its classified impact based on the forecast and indicator type.")

        indicator_props_sim = get_indicator_properties(event_name_str)
        unit_sim = indicator_props_sim.get("unit", "")
        step_value = 0.1 if "%" in unit_sim else (1.0 if "K" in unit_sim or "M" in unit_sim else 0.01)

        if indicator_props_sim["type"] == "qualitative":
            st.warning(f"⚠️ '{event_name_str}' is a qualitative event (e.g., speech). Numerical simulation is not directly applicable. Interpretations are based on rhetoric and sentiment.")
        else:
            actual_input_val_default = forecast_val if pd.notna(forecast_val) else (previous_val if pd.notna(previous_val) else 0.0)
            
            hypothetical_actual = st.number_input(
                f"Hypothetical 'Actual' Value ({unit_sim}):",
                value=float(actual_input_val_default) if pd.notna(actual_input_val_default) else 0.0,
                step=float(step_value), format="%.2f",
                key=f"actual_input_main_{event_id_for_keys}",
                help=f"Enter a hypothetical actual value for {event_name_str} to simulate its impact."
            )

            if st.button("Classify Hypothetical Actual", key=f"classify_btn_main_{event_id_for_keys}", use_container_width=True):
                classification, explanation = classify_actual_release(hypothetical_actual, forecast_val, previous_val, event_name_str, currency_str)
                nuanced_color_map = {
                    "Strongly Bullish": "#145A32", "Mildly Bullish": "#27AE60", "Neutral/In-Line": "#808B96",
                    "Mildly Bearish": "#E74C3C", "Strongly Bearish": "#922B21",
                    "Qualitative": "#2E4053", "Indeterminate": "#4A235A", "Error": "#641E16"
                }
                class_bg_color = nuanced_color_map.get(classification, "#333333")

                st.markdown(f"**Classification Result:** <span style='background-color:{class_bg_color}; color:white; padding: 3px 7px; border-radius: 4px; font-weight:bold;'>{classification}</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='custom-classification-box' style='border-color: {class_bg_color}; background-color: #1c1e22;'>{explanation}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Economic Calendar Overview (Filtered) ---
    st.markdown("---")
    with st.expander("🗓️ Full Economic Calendar Overview (Filtered)", expanded=False):
        if not economic_df_filtered.empty:
            calendar_display_df = economic_df_filtered.copy()
            if 'Timestamp' in calendar_display_df.columns and pd.api.types.is_datetime64_any_dtype(calendar_display_df['Timestamp']):
                calendar_display_df['FormattedTimestamp'] = calendar_display_df['Timestamp'].apply(
                    lambda x: convert_and_format_time(x, selected_tz_name, fmt="%Y-%m-%d %I:%M %p")
                )
                display_cols = ['FormattedTimestamp', 'Currency', 'Impact', 'EventName', 'Previous', 'Forecast', 'Actual', 'Zone']
                calendar_display_df = calendar_display_df[[col for col in display_cols if col in calendar_display_df.columns]]
                calendar_display_df.rename(columns={'FormattedTimestamp': 'Time', 'EventName': 'Event Name', 'Currency': 'Ccy'}, inplace=True)
            else:
                calendar_display_df['Time'] = "Data Error"
                display_cols_err = ['Time', 'Currency', 'Impact', 'EventName', 'Previous', 'Forecast', 'Actual', 'Zone']
                calendar_display_df = calendar_display_df[[col for col in display_cols_err if col in calendar_display_df.columns]]

            st.dataframe(
                calendar_display_df.sort_values(by='Time'),
                use_container_width=True, hide_index=True, height=450,
                column_config={
                    "Time": st.column_config.TextColumn("Time", width="medium", help="Scheduled time in selected timezone"),
                    "Ccy": st.column_config.TextColumn("Ccy", width="small", help="Currency"),
                    "Impact": st.column_config.TextColumn("Impact", width="small", help="Event impact level"),
                    "Event Name": st.column_config.TextColumn("Event", width="large", help="Name of the economic event"),
                    "Previous": st.column_config.NumberColumn("Prev.", format="%.2f", width="small", help="Previous actual value"),
                    "Forecast": st.column_config.NumberColumn("Fcst.", format="%.2f", width="small", help="Forecasted value"),
                    "Actual": st.column_config.NumberColumn("Actual", format="%.2f", width="small", help="Actual released value (if available)"),
                    "Zone": st.column_config.TextColumn("Zone", width="small", help="Geographic zone"),
                }
            )
        else:
            st.info("ℹ️ No events to display in the calendar based on current filters or data availability.")

# --- Footer ---
st.markdown("---")
st.caption(
    "Trading Mastery Hub © 2024-2025 | Disclaimer: Generalized interpretations, not financial advice. "
    "Calendar data primarily via investpy (unofficial). US Historicals may use Alpha Vantage. Other historicals are sample data. "
    "Data accuracy depends on sources and is not guaranteed. Always conduct your own research."
)

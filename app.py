# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, date, timedelta
import logging # Import logging

# Assuming these modules are in the same directory
# Ensure data_loader.py uses the version with configurable logging and yfinance ticker map
from data_loader import load_economic_data, load_historical_data, logger as data_loader_logger
from strategy_engine import (
    predict_actual_condition_for_outcome,
    infer_market_outlook_from_data,
    classify_actual_release,
    get_indicator_properties,
    INDICATOR_CONFIG
)
from visualization import plot_historical_trend
from ai_models import analyze_qualitative_event_llm

# --- Application Logger Setup ---
app_logger = logging.getLogger(__name__)
if data_loader_logger.hasHandlers():
    app_logger.setLevel(data_loader_logger.level)
    # Not adding handlers from data_loader_logger to app_logger by default
    # to avoid duplicate logs if root logger is also configured.
    # If app.py specific logging to the same file is needed, uncomment addHandler lines.
else:
    if not app_logger.handlers:
        app_logger.setLevel(logging.INFO) # Default level for app if no other config

app_logger.info("Application starting...")


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
        app_logger.info(f"Successfully loaded CSS file: {file_name}")
    except FileNotFoundError:
        app_logger.error(f"CSS File Not Found: '{file_name}'.")
        st.error(f"🚨 CSS File Not Found: '{file_name}'. Please ensure it's in the correct directory.")
    except Exception as e:
        app_logger.error(f"An error occurred while loading CSS '{file_name}'.", exc_info=True)
        st.error(f"🚨 An error occurred while loading CSS: {e}")

# --- Load Custom CSS ---
load_css("style.css")
# --- End Custom CSS Loading ---


# --- Helper function for timezone conversion ---
def convert_and_format_time(dt_object, target_tz_str, fmt="%Y-%m-%d %I:%M %p %Z"):
    if pd.isna(dt_object) or not isinstance(dt_object, datetime): return "N/A"
    try:
        target_tz = pytz.timezone(target_tz_str)
        if dt_object.tzinfo is None or dt_object.tzinfo.utcoffset(dt_object) is None:
            try:
                dt_object = pytz.utc.localize(dt_object)
            except pytz.exceptions.AmbiguousTimeError:
                dt_object = pytz.utc.localize(dt_object, is_dst=None)
            except pytz.exceptions.InvalidTimeError:
                 app_logger.warning(f"Could not localize naive datetime {dt_object} to UTC due to invalid time (e.g. during DST change). Treating as naive.")
        return dt_object.astimezone(target_tz).strftime(fmt)
    except Exception as e:
        app_logger.warning(f"Time conversion issue for an event: {dt_object}. Error: {e}", exc_info=False)
        return "Invalid Time"

# --- Sidebar for Configuration ---
with st.sidebar:
    try:
        # Attempt to load the image for the sidebar
        # Removed use_container_width to avoid error with older Streamlit versions
        st.image("https://placehold.co/300x100/0F1116/007BFF?text=Impact+Forecaster&font=roboto")
    except Exception as e:
        app_logger.error("Failed to load sidebar image from placehold.co.", exc_info=True)
        st.error(f"🚨 Sidebar image could not be loaded. Error: {e}")

    st.markdown("## ⚙️ Configuration Filters")
    st.markdown("---")

    st.subheader("🗓️ Date Range")
    today_date_obj = date(2025, 5, 8) 
    tomorrow_date_obj = today_date_obj + timedelta(days=1)
    start_of_week = today_date_obj - timedelta(days=today_date_obj.weekday())
    end_of_week = start_of_week + timedelta(days=6)

    if 'start_date_filter' not in st.session_state:
        st.session_state.start_date_filter = start_of_week
        app_logger.debug("Session state 'start_date_filter' initialized.")
    if 'end_date_filter' not in st.session_state:
        st.session_state.end_date_filter = end_of_week
        app_logger.debug("Session state 'end_date_filter' initialized.")

    def set_date_to_today_tomorrow():
        st.session_state.start_date_filter = today_date_obj
        st.session_state.end_date_filter = tomorrow_date_obj
        app_logger.info("Date filter set to Today & Tomorrow.")
    def set_date_to_current_week():
        st.session_state.start_date_filter = start_of_week
        st.session_state.end_date_filter = end_of_week
        app_logger.info("Date filter set to Current Week.")

    col_start_date, col_end_date = st.columns(2)
    with col_start_date:
        start_date_input = st.date_input(
            "Start",
            value=st.session_state.start_date_filter,
            key="start_date_widget",
            help="Select start date for the economic calendar."
        )
    with col_end_date:
        end_date_input = st.date_input(
            "End",
            value=st.session_state.end_date_filter,
            min_value=start_date_input,
            key="end_date_widget",
            help="Select end date for the economic calendar."
        )

    if start_date_input != st.session_state.start_date_filter:
        st.session_state.start_date_filter = start_date_input
        app_logger.debug(f"start_date_filter updated to: {start_date_input}")
    if end_date_input != st.session_state.end_date_filter:
        st.session_state.end_date_filter = end_date_input
        app_logger.debug(f"end_date_filter updated to: {end_date_input}")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1: st.button("Today & Tomorrow", on_click=set_date_to_today_tomorrow, use_container_width=True, key="today_tomorrow_btn")
    with col_btn2: st.button("Current Week", on_click=set_date_to_current_week, use_container_width=True, key="current_week_btn")

    st.subheader("🌐 Timezone")
    common_timezones = sorted(pytz.common_timezones)
    default_tz_sidebar = 'US/Eastern'
    if 'selected_timezone' not in st.session_state:
        st.session_state.selected_timezone = default_tz_sidebar
        app_logger.debug("Session state 'selected_timezone' initialized.")

    try:
        default_tz_index = common_timezones.index(st.session_state.selected_timezone)
    except ValueError:
        app_logger.warning(f"Previously selected timezone '{st.session_state.selected_timezone}' not in common_timezones. Defaulting to '{default_tz_sidebar}'.")
        st.session_state.selected_timezone = default_tz_sidebar
        default_tz_index = common_timezones.index(default_tz_sidebar)

    selected_tz_name = st.selectbox(
        "Display Timezone:",
        options=common_timezones,
        index=default_tz_index,
        key="selected_timezone_widget",
        help="Choose the timezone for displaying event times."
    )
    if selected_tz_name != st.session_state.selected_timezone:
        st.session_state.selected_timezone = selected_tz_name
        app_logger.info(f"Selected timezone updated to: {selected_tz_name}")

    st.markdown("---")

economic_df_master, data_status_message = load_economic_data(
    st.session_state.start_date_filter,
    st.session_state.end_date_filter
)
app_logger.info(f"Economic data loaded. Status: {data_status_message}. Shape: {economic_df_master.shape}")

with st.sidebar:
    st.subheader("💱 Currencies")
    if not economic_df_master.empty and 'Currency' in economic_df_master.columns:
        available_currencies = sorted([curr for curr in economic_df_master['Currency'].unique() if pd.notna(curr) and curr != ''])
    else:
        available_currencies = ["USD", "EUR", "JPY", "GBP", "CAD", "AUD"]
    currency_options = ["All"] + available_currencies
    app_logger.debug(f"Available currencies for filter: {available_currencies}")

    default_initial_selection_curr = ["USD"] if "USD" in available_currencies else ["All"]
    if 'selected_currencies_filter' not in st.session_state:
        st.session_state.selected_currencies_filter = default_initial_selection_curr
        app_logger.debug("Session state 'selected_currencies_filter' initialized.")

    current_currency_selection = st.session_state.selected_currencies_filter
    valid_current_selection_curr = [c for c in current_currency_selection if c in currency_options]
    default_for_widget_curr = valid_current_selection_curr if valid_current_selection_curr else default_initial_selection_curr

    selected_currencies = st.multiselect(
        "Filter Currencies:",
        options=currency_options,
        default=default_for_widget_curr,
        key="selected_currencies_widget",
        help="Select currencies to filter events. 'All' includes events with any or no specified currency."
    )
    if selected_currencies != st.session_state.selected_currencies_filter:
        st.session_state.selected_currencies_filter = selected_currencies
        app_logger.info(f"Selected currencies filter updated to: {selected_currencies}")

    st.subheader("⚡ Impact Level")
    impact_level_options_std = ["High", "Medium", "Low"]
    if not economic_df_master.empty and 'Impact' in economic_df_master.columns:
        data_impact_values = sorted(list(set(str(imp) for imp in economic_df_master['Impact'].unique() if pd.notna(imp) and str(imp) not in ['N/A', ''])))
        combined_impact_options = []
        for opt in impact_level_options_std:
            if opt in data_impact_values and opt not in combined_impact_options:
                combined_impact_options.append(opt)
        for opt in data_impact_values:
            if opt not in combined_impact_options:
                combined_impact_options.append(opt)
        impact_filter_options = ["All"] + (combined_impact_options if combined_impact_options else impact_level_options_std)
    else:
        impact_filter_options = ["All"] + impact_level_options_std
    app_logger.debug(f"Available impact levels for filter: {impact_filter_options}")

    default_initial_selection_imp = ["High"] if "High" in impact_filter_options else ["All"]
    if 'selected_impact_filter' not in st.session_state:
        st.session_state.selected_impact_filter = default_initial_selection_imp
        app_logger.debug("Session state 'selected_impact_filter' initialized.")

    current_impact_selection = st.session_state.selected_impact_filter
    valid_current_selection_imp = [i for i in current_impact_selection if i in impact_filter_options]
    default_for_widget_imp = valid_current_selection_imp if valid_current_selection_imp else default_initial_selection_imp

    selected_impacts = st.multiselect(
        "Filter Impact:",
        options=impact_filter_options,
        default=default_for_widget_imp,
        key="selected_impact_widget",
        help="Select impact levels to filter events. 'All' includes events of any impact."
    )
    if selected_impacts != st.session_state.selected_impact_filter:
        st.session_state.selected_impact_filter = selected_impacts
        app_logger.info(f"Selected impact filter updated to: {selected_impacts}")

    st.markdown("---")
    st.caption(f"Calendar Status: {data_status_message}")
    if 'ALPHA_VANTAGE_API_KEY' not in st.secrets:
        st.caption("AV Key: ⚠️ Missing")
        app_logger.warning("Alpha Vantage API Key is missing from secrets.")
    else:
        st.caption("AV Key: 🔑 Configured")
    st.markdown("---")
    st.markdown("<div class='custom-info-box' style='background-color: rgba(0,123,255,0.05); border-left-color: #00A0B0;'>Tip: Use filters to narrow events. Click an event below for analysis.</div>", unsafe_allow_html=True)

# --- Main Application Area ---
st.title("📊 Economic Impact Forecaster")
st.markdown("<div class='app-subtitle'>Powered by <strong>Trading Mastery Hub</strong> | <em>Navigating Economic Tides with Data</em></div>", unsafe_allow_html=True)
st.markdown("---")

economic_df_filtered = economic_df_master.copy()
if 'Currency' in economic_df_filtered.columns and selected_currencies and "All" not in selected_currencies:
    economic_df_filtered = economic_df_filtered[economic_df_filtered['Currency'].isin(selected_currencies)]
if 'Impact' in economic_df_filtered.columns and selected_impacts and "All" not in selected_impacts:
    economic_df_filtered = economic_df_filtered[economic_df_filtered['Impact'].isin(selected_impacts)]

app_logger.info(f"Data filtered. Shape after filtering: {economic_df_filtered.shape}")

if economic_df_master.empty:
    if "Simulated" in data_status_message:
        st.warning("⚠️ Displaying simulated data as live data fetch failed. Check logs for details.")
        app_logger.warning("Displaying simulated data due to live fetch failure.")
    else:
        st.error("🚨 Failed to load any economic data. Check data sources/network, then retry. Consult logs for more details.")
        app_logger.error("Failed to load any economic data (master is empty).")
elif economic_df_filtered.empty:
    st.warning("⚠️ No economic events match the current filter criteria. Adjust date range or filters in the sidebar.")
    app_logger.info("No events match current filters, displaying warning.")
else:
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.subheader("🗓️ Select Economic Event")

    if 'Timestamp' in economic_df_filtered.columns and pd.api.types.is_datetime64_any_dtype(economic_df_filtered['Timestamp']):
        try:
            economic_df_filtered['display_name'] = economic_df_filtered.apply(
                lambda r: (f"{convert_and_format_time(r['Timestamp'], st.session_state.selected_timezone, fmt='%Y-%m-%d %I:%M %p')} | "
                           f"{r.get('Currency','N/A')} | {r.get('Impact','N/A')} | {r.get('EventName','Unknown Event')}"),
                axis=1
            )
        except Exception as e:
            app_logger.error(f"Error creating display_name for events: {e}", exc_info=True)
            st.error(f"🚨 Error formatting event display names: {e}")
            economic_df_filtered['display_name'] = "Error in event display"
    else:
        app_logger.error("Timestamp data issue: 'Timestamp' column missing or not datetime type in filtered data.")
        st.error("🚨 Timestamp data issue. Event selection might be affected.")
        economic_df_filtered['display_name'] = economic_df_filtered.apply(
            lambda r: f"Data Error - {r.get('Currency','N/A')} - {r.get('EventName','Unknown Event')}", axis=1
        )

    event_options = economic_df_filtered.sort_values(by='Timestamp')['display_name'].tolist()
    
    sel_key_event_selectbox = "current_event_selectbox_main"
    if not event_options:
        st.session_state[sel_key_event_selectbox] = None
        app_logger.info("No event options available for selection.")
    elif sel_key_event_selectbox not in st.session_state or st.session_state[sel_key_event_selectbox] not in event_options:
        st.session_state[sel_key_event_selectbox] = event_options[0]
        app_logger.debug(f"Event selectbox state initialized or reset to: {event_options[0]}")

    selected_event_display_name = st.selectbox(
        "Choose event:",
        options=event_options,
        key=sel_key_event_selectbox,
        label_visibility="collapsed",
        index=event_options.index(st.session_state[sel_key_event_selectbox]) if st.session_state[sel_key_event_selectbox] and event_options else 0,
        help="Select an economic event from the filtered list to view its details and analysis."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if not event_options or selected_event_display_name is None or \
       selected_event_display_name.startswith("Invalid") or \
       selected_event_display_name.startswith("Data Error") or \
       selected_event_display_name == "Error in event display":
        st.error("🚨 No valid event selected or selection error. Adjust filters or check data integrity. Consult logs.")
        app_logger.error(f"No valid event selected. Selected display name: {selected_event_display_name}")
        st.stop()

    try:
        selected_event_row = economic_df_filtered[economic_df_filtered['display_name'] == selected_event_display_name].iloc[0]
        app_logger.info(f"Event selected: {selected_event_display_name}")
    except IndexError:
        st.error("🚨 Could not retrieve details for the selected event. It might have disappeared due to a filter change or data update. Please re-select.")
        app_logger.error(f"IndexError retrieving selected event row for: {selected_event_display_name}. Filters might have changed.")
        st.stop()

    event_id_parts = [
        str(selected_event_row.get('EventName', '')),
        str(selected_event_row.get('Currency', '')),
        str(selected_event_row.get('Timestamp', ''))
    ]
    event_id = "".join(filter(None, event_id_parts))
    if not event_id: event_id = str(np.random.randint(100000))
    app_logger.debug(f"Generated event_id: {event_id}")

    prev_val = selected_event_row.get('Previous')
    fcst_val = selected_event_row.get('Forecast')
    act_val = selected_event_row.get('Actual')
    evt_name = str(selected_event_row.get('EventName', 'N/A'))
    cur_str = str(selected_event_row.get('Currency', 'N/A'))
    imp_str = str(selected_event_row.get('Impact', 'N/A'))
    evt_ts = selected_event_row.get('Timestamp')
    
    fmt_evt_time = convert_and_format_time(evt_ts, st.session_state.selected_timezone)
    indicator_props = get_indicator_properties(evt_name)
    app_logger.debug(f"Indicator properties for '{evt_name}': {indicator_props}")

    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.subheader(f"🔍 Details: {evt_name}")
    det_cols = st.columns([1.5, 1.5, 1, 1, 1])
    with det_cols[0]: st.metric(label="Currency Pair", value=f"{cur_str}/USD" if cur_str not in ["USD", "N/A", ""] else cur_str)
    with det_cols[1]: st.metric(label="Impact Level", value=imp_str if pd.notna(imp_str) and imp_str else "N/A")
    
    time_p, date_p = "N/A", "N/A"
    if fmt_evt_time not in ["N/A", "Invalid Time"] and ' ' in fmt_evt_time:
        parts = fmt_evt_time.split(' ', 2)
        date_p = parts[0]
        time_p = f"{parts[1]}"
        if len(parts) > 2:
            time_p += f" {parts[2]}"
            
    with det_cols[2]: st.metric(label="Previous", value=f"{prev_val:.2f}" if pd.notna(prev_val) and isinstance(prev_val, (int, float)) else "N/A")
    with det_cols[3]: st.metric(label="Forecast", value=f"{fcst_val:.2f}" if pd.notna(fcst_val) and isinstance(fcst_val, (int, float)) else "N/A")
    with det_cols[4]: st.metric(label="Actual", value=f"{act_val:.2f}" if pd.notna(act_val) and isinstance(act_val, (int, float)) else "Pending")
    st.caption(f"Scheduled: {date_p} at {time_p} ({st.session_state.selected_timezone})")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.subheader("🎯 Baseline Interpretation")
    inferred_bias = infer_market_outlook_from_data(prev_val, fcst_val, evt_name)
    st.markdown(f"<div class='custom-info-box'>System-Inferred Bias (Forecast vs. Previous): <strong>{inferred_bias}</strong> for {cur_str}</div>", unsafe_allow_html=True)
    
    st.markdown("<h5>Desired Market Scenario Analysis (Baseline)</h5>", unsafe_allow_html=True)
    outcome_opts = ["Bullish", "Bearish", "Consolidating"]
    def_idx_outcome = 2
    if inferred_bias and "qualitative" not in inferred_bias.lower() and "insufficient" not in inferred_bias.lower() and "invalid" not in inferred_bias.lower():
        if "bullish" in inferred_bias.lower(): def_idx_outcome = 0
        elif "bearish" in inferred_bias.lower(): def_idx_outcome = 1
    
    desired_outcome = st.radio(
        f"Select desired outcome for {cur_str}:",
        options=outcome_opts,
        index=def_idx_outcome,
        key=f"outcome_radio_{event_id}",
        horizontal=True,
        help="Choose the market direction for which you want a baseline scenario analysis."
    )
    app_logger.debug(f"Desired outcome selected: {desired_outcome} for event {event_id}")

    prediction_pts = predict_actual_condition_for_outcome(prev_val, fcst_val, desired_outcome, cur_str, evt_name)
    outcome_colors = {"Bullish": "#28a745", "Bearish": "#dc3545", "Consolidating": "#6c757d"}
    box_color = outcome_colors.get(desired_outcome, "#6c757d")
    pred_html_list = "".join([f"<li>{pt}</li>" for pt in prediction_pts])
    st.markdown(f"<div class='custom-prediction-box' style='background-color: {box_color}1A; border-left: 5px solid {box_color}; color: #E0E0E0;'><ul style='margin-bottom:0; padding-left:20px;'>{pred_html_list}</ul></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if indicator_props.get("type") == "qualitative":
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.subheader("🤖 AI Sentiment Analysis (Qualitative Event)")
        st.markdown(f"Since **{evt_name}** is a qualitative event, its impact is driven by the perceived tone and message. Use the AI assistant below to explore potential interpretations based on a selected sentiment.")

        qual_sentiment_options = ["Hawkish", "Neutral", "Dovish"]
        user_qual_sentiment = st.radio(
            "Select perceived sentiment:",
            options=qual_sentiment_options,
            index=1,
            key=f"qual_sentiment_{event_id}",
            horizontal=True,
            help="Indicate the perceived tone of the qualitative event for AI analysis."
        )

        if st.button("Analyze Qualitative Sentiment with AI", key=f"analyze_qual_ai_{event_id}", use_container_width=True):
            app_logger.info(f"Initiating AI sentiment analysis for '{evt_name}' with sentiment '{user_qual_sentiment}'.")
            with st.spinner(f"🧠 AI is analyzing '{evt_name}' with a '{user_qual_sentiment}' sentiment for {cur_str}..."):
                event_desc_for_ai = indicator_props.get("description", "A significant qualitative economic event.")
                try:
                    ai_analysis_result = analyze_qualitative_event_llm(evt_name, user_qual_sentiment, cur_str, event_desc_for_ai)
                    app_logger.info(f"AI analysis completed for '{evt_name}'. Confidence: {ai_analysis_result.get('confidence')}")
                except Exception as e:
                    app_logger.error(f"Error during AI qualitative analysis for '{evt_name}'.", exc_info=True)
                    st.error(f"🚨 AI Analysis Failed: {e}")
                    ai_analysis_result = {}

            if ai_analysis_result:
                st.markdown(f"<h5>AI Analysis Result (Sentiment: {user_qual_sentiment})</h5>", unsafe_allow_html=True)
                res_color = outcome_colors.get(user_qual_sentiment, "#6c757d")
                if user_qual_sentiment == "Hawkish": res_color = outcome_colors["Bullish"]
                elif user_qual_sentiment == "Dovish": res_color = outcome_colors["Bearish"]
                
                st.markdown(f"<div class='custom-info-box' style='border-left-color: {res_color};'><strong>Summary:</strong> {ai_analysis_result.get('summary', 'N/A')}</div>", unsafe_allow_html=True)
                
                cols_ai = st.columns(2)
                with cols_ai[0]:
                    st.markdown("**Potential Bullish Points:**")
                    bullish_pts = ai_analysis_result.get("bullish_points", [])
                    if bullish_pts: [st.markdown(f"- {pt}") for pt in bullish_pts]
                    else: st.caption("N/A")
                with cols_ai[1]:
                    st.markdown("**Potential Bearish Points:**")
                    bearish_pts = ai_analysis_result.get("bearish_points", [])
                    if bearish_pts: [st.markdown(f"- {pt}") for pt in bearish_pts]
                    else: st.caption("N/A")
                
                st.markdown("**Key Considerations & Factors:**")
                key_considerations = ai_analysis_result.get("key_considerations", [])
                if key_considerations: [st.markdown(f"- {pt}") for pt in key_considerations]
                else: st.caption("N/A")

                st.markdown(f"<div style='font-size:0.8rem;color:#A0A0A0;margin-top:10px;'><strong>Impact Scale:</strong> {ai_analysis_result.get('potential_impact','N/A')} | <strong>AI Confidence:</strong> {ai_analysis_result.get('confidence','N/A')}</div>", unsafe_allow_html=True)
                st.caption(f"ℹ️ {ai_analysis_result.get('disclaimer', 'AI analysis is experimental and not financial advice.')}")
            else:
                st.warning("AI analysis did not return a result.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    tab_hist_label = "📈 Historical Trends"
    tab_sim_label = "🔬 Simulate Actual Release"
    if indicator_props.get("type") == "qualitative":
        tab_sim_label = "🔬 Numerical Simulation (N/A for Qualitative)"
    
    tab_hist, tab_sim = st.tabs([tab_hist_label, tab_sim_label])

    with tab_hist:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.subheader(f"Historical Trends for: {evt_name}")
        app_logger.debug(f"Loading historical data for tab: {evt_name}")
        
        is_av_source = False
        if 'ALPHA_VANTAGE_API_KEY' in st.secrets:
            av_indicator_keywords = ["Non-Farm", "Unemployment", "CPI", "Retail Sales", "GDP", "Federal Funds"]
            if any(keyword.lower() in evt_name.lower() for keyword in av_indicator_keywords) and \
               (cur_str == "USD" or not cur_str or cur_str == "N/A"):
                is_av_source = True
        
        df_hist = load_historical_data(evt_name)
        
        if not df_hist.empty:
            cap_txt = f"Displaying historical data for {evt_name}. "
            if is_av_source and 'Forecast' not in df_hist.columns and 'Previous' not in df_hist.columns:
                 cap_txt += "Likely sourced from Alpha Vantage (US Data - 'Actual' values). Forecast/Previous may not be available from this source."
            elif not df_hist[['Actual', 'Forecast', 'Previous']].dropna(how='all', axis=1).empty:
                 cap_txt += "Data may be from yfinance, Alpha Vantage, or sample series."
            else:
                 cap_txt += "Sample data may be shown if live sources are unavailable or lack all fields."
            st.caption(cap_txt)
            plot_historical_trend(df_hist, evt_name, indicator_props.get("type", "normal"))
        else:
            st.info(f"ℹ️ No historical data found or able to be plotted for '{evt_name}'. This could be due to data source limitations or the nature of the event.")
            app_logger.info(f"No historical data to plot for '{evt_name}'.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_sim:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.subheader(f"Simulate Actual Release Impact for: {evt_name}")
        if indicator_props.get("type") == "qualitative":
            st.info(f"ℹ️ Numerical simulation is not applicable for purely qualitative events like '{evt_name}'. Use the AI Sentiment Analysis tab.")
        else:
            st.markdown("Enter a hypothetical 'Actual' value to see its classified impact on currency sentiment.")
            unit_sim = indicator_props.get("unit","")
            if "%" in unit_sim: step_v = 0.1
            elif "K" in unit_sim.upper() or "M" in unit_sim.upper() or "B" in unit_sim.upper(): step_v = 1.0 if pd.isna(fcst_val) or abs(fcst_val) < 10 else 10.0
            elif pd.notna(fcst_val) and abs(fcst_val) < 1: step_v = 0.01
            elif pd.notna(fcst_val) and abs(fcst_val) < 10: step_v = 0.1
            else: step_v = 0.1

            actual_in_def = fcst_val if pd.notna(fcst_val) and isinstance(fcst_val, (int,float)) else \
                            (prev_val if pd.notna(prev_val) and isinstance(prev_val, (int,float)) else 0.0)
            
            hyp_actual = st.number_input(
                f"Hypothetical 'Actual' ({unit_sim}):",
                value=float(actual_in_def),
                step=float(step_v),
                format="%.2f" if "%" in unit_sim or (pd.notna(fcst_val) and abs(fcst_val) < 10 and abs(fcst_val) != 0) else "%.1f",
                key=f"actual_input_{event_id}",
                help=f"Enter a hypothetical 'Actual' release value for {evt_name} to simulate its impact."
            )

            if st.button("Classify Hypothetical Actual", key=f"classify_btn_{event_id}", use_container_width=True):
                app_logger.info(f"Classifying hypothetical actual: {hyp_actual} for event '{evt_name}'.")
                classification, explanation = classify_actual_release(hyp_actual, fcst_val, prev_val, evt_name, cur_str)
                
                nu_colors = {
                    "Strongly Bullish":"#145A32", "Mildly Bullish":"#27AE60",
                    "Neutral/In-Line":"#808B96",
                    "Mildly Bearish":"#E74C3C", "Strongly Bearish":"#922B21",
                    "Qualitative":"#2E4053", "Indeterminate":"#4A235A", "Error":"#641E16"
                }
                cls_bg_color = nu_colors.get(classification, "#333333")

                st.markdown(f"**Classification:** <span style='background-color:{cls_bg_color};color:white;padding:3px 7px;border-radius:4px;font-weight:bold;'>{classification}</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='custom-classification-box' style='border-color:{cls_bg_color};background-color:#1c1e22;'>{explanation}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("🗓️ Full Economic Calendar Overview (Filtered)", expanded=False):
        if not economic_df_filtered.empty:
            cal_df = economic_df_filtered.copy()
            if 'Timestamp' in cal_df.columns and pd.api.types.is_datetime64_any_dtype(cal_df['Timestamp']):
                cal_df['FormattedTimestamp'] = cal_df['Timestamp'].apply(
                    lambda x: convert_and_format_time(x, st.session_state.selected_timezone, fmt="%Y-%m-%d %I:%M %p")
                )
                display_cols = ['FormattedTimestamp','Currency','Impact','EventName','Previous','Forecast','Actual','Zone']
                cols_to_display = [col for col in display_cols if col in cal_df.columns]
                cal_df_display = cal_df[cols_to_display]
                cal_df_display = cal_df_display.rename(columns={'FormattedTimestamp':'Time','EventName':'Event Name','Currency':'Ccy'})
            else:
                app_logger.warning("Timestamp issue in filtered calendar view. Displaying with errors.")
                cal_df_display = pd.DataFrame({
                    'Time': ["Data Error"] * len(cal_df),
                    'Event Name': cal_df.get('EventName', pd.Series(["N/A"] * len(cal_df))),
                    'Ccy': cal_df.get('Currency', pd.Series(["N/A"] * len(cal_df)))
                })

            st.dataframe(
                cal_df_display.sort_values(by='Time'),
                use_container_width=True,
                hide_index=True,
                height=450,
                column_config={
                    "Time": st.column_config.TextColumn("Time", width="medium", help="Scheduled time in selected timezone"),
                    "Ccy": st.column_config.TextColumn("Ccy", width="small", help="Currency"),
                    "Impact": st.column_config.TextColumn("Impact", width="small", help="Impact level"),
                    "Event Name": st.column_config.TextColumn("Event", width="large", help="Name of the economic event"),
                    "Previous": st.column_config.NumberColumn("Prev.", format="%.2f", width="small", help="Previous value"),
                    "Forecast": st.column_config.NumberColumn("Fcst.", format="%.2f", width="small", help="Forecast value"),
                    "Actual": st.column_config.NumberColumn("Actual", format="%.2f", width="small", help="Actual released value (if available)"),
                    "Zone": st.column_config.TextColumn("Zone", width="small", help="Geographic zone of the event")
                }
            )
        else:
            st.info("ℹ️ No events to display in calendar based on current filters.")

st.markdown("---")
st.caption("Trading Mastery Hub © 2024-2025 | Disclaimer: Generalized interpretations, not financial advice. Data accuracy depends on sources and is not guaranteed. Always conduct your own research.")
app_logger.info("Application rendering complete.")


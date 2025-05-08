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
from ai_models import analyze_qualitative_event_llm

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
load_css("style.css")
# --- End Custom CSS Loading ---


# --- Helper function for timezone conversion ---
def convert_and_format_time(dt_object, target_tz_str, fmt="%Y-%m-%d %I:%M %p %Z"):
    if pd.isna(dt_object) or not isinstance(dt_object, datetime): return "N/A"
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
    st.image("https://placehold.co/300x100/0F1116/007BFF?text=Impact+Forecaster&font=roboto", use_container_width=True)
    st.markdown("## ⚙️ Configuration Filters")
    st.markdown("---")

    st.subheader("🗓️ Date Range")
    today = date.today(); default_start_date = today - timedelta(days=today.weekday()); default_end_date = default_start_date + timedelta(days=6)
    if 'start_date_filter' not in st.session_state: st.session_state.start_date_filter = default_start_date
    if 'end_date_filter' not in st.session_state: st.session_state.end_date_filter = default_end_date
    col_start_date, col_end_date = st.columns(2)
    with col_start_date: start_date_input = st.date_input("Start", value=st.session_state.start_date_filter, key="start_date_widget", help="Select the start date.")
    with col_end_date: end_date_input = st.date_input("End", value=st.session_state.end_date_filter, min_value=start_date_input, key="end_date_widget", help="Select the end date.")
    st.session_state.start_date_filter = start_date_input; st.session_state.end_date_filter = end_date_input

    st.subheader("🌐 Timezone")
    common_timezones = pytz.common_timezones; default_tz_sidebar = 'US/Eastern'
    if 'selected_timezone' not in st.session_state: st.session_state.selected_timezone = default_tz_sidebar
    selected_tz_name = st.selectbox("Display Timezone:", options=common_timezones, index=common_timezones.index(st.session_state.selected_timezone) if st.session_state.selected_timezone in common_timezones else common_timezones.index(default_tz_sidebar), key="selected_timezone_widget", help="Choose timezone for event times.")
    st.session_state.selected_timezone = selected_tz_name
    st.markdown("---")

economic_df_master, data_status_message = load_economic_data(st.session_state.start_date_filter, st.session_state.end_date_filter)

with st.sidebar:
    st.subheader("💱 Currencies")
    if not economic_df_master.empty and 'Currency' in economic_df_master.columns: available_currencies = sorted([curr for curr in economic_df_master['Currency'].unique() if pd.notna(curr) and curr != ''])
    else: available_currencies = ["USD", "EUR", "JPY", "GBP", "CAD", "AUD"]
    currency_options = ["All"] + available_currencies
    if 'selected_currencies_filter' not in st.session_state: st.session_state.selected_currencies_filter = ["All"]
    current_currency_selection = st.session_state.selected_currencies_filter; valid_current_selection_curr = [c for c in current_currency_selection if c in currency_options]; default_currency_sel = valid_current_selection_curr if valid_current_selection_curr else ["All"]
    selected_currencies = st.multiselect("Filter Currencies:", options=currency_options, default=default_currency_sel, key="selected_currencies_widget_updated", help="Select currencies.")
    st.session_state.selected_currencies_filter = selected_currencies

    st.subheader("⚡ Impact Level")
    # !!! IMPORTANT: Check THIS SECTION in YOUR app.py file carefully !!!
    # The unwanted debug output appears before the "Filter Impact:" multiselect.
    # Ensure there are NO st.write() or print() statements here in your local file.
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
    
    # Example of what to look for and REMOVE in your file if it exists:
    # st.write(current_impact_selection)  # <--- REMOVE THIS IF PRESENT
    # print(default_impact_sel)           # <--- REMOVE THIS IF PRESENT
    # st.write(impact_filter_options)     # <--- REMOVE THIS IF PRESENT

    selected_impacts = st.multiselect(
        "Filter Impact:",
        options=impact_filter_options,
        default=default_impact_sel,
        key="selected_impact_widget",
        help="Select impact levels to filter events."
    )
    st.session_state.selected_impact_filter = selected_impacts
    # !!! End of section to check carefully !!!

    st.markdown("---")
    st.caption(f"Calendar Status: {data_status_message}")
    if 'ALPHA_VANTAGE_API_KEY' not in st.secrets: st.caption("AV Key: ⚠️ Missing")
    else: st.caption("AV Key: 🔑 Configured")
    st.markdown("---")
    st.markdown("<div class='custom-info-box' style='background-color: rgba(0,123,255,0.05); border-left-color: #00A0B0;'>Tip: Use filters to narrow events. Click an event below for analysis.</div>", unsafe_allow_html=True)

st.title("📊 Economic Impact Forecaster")
st.markdown("<div class='app-subtitle'>Powered by <strong>Trading Mastery Hub</strong> | <em>Navigating Economic Tides with Data</em></div>", unsafe_allow_html=True)
st.markdown("---")

economic_df_filtered = economic_df_master.copy()
if 'Currency' in economic_df_filtered.columns and not ("All" in selected_currencies or not selected_currencies) : economic_df_filtered = economic_df_filtered[economic_df_filtered['Currency'].isin(selected_currencies)]
if 'Impact' in economic_df_filtered.columns and not ("All" in selected_impacts or not selected_impacts): economic_df_filtered = economic_df_filtered[economic_df_filtered['Impact'].isin(selected_impacts)]

if economic_df_master.empty:
    if "Simulated" in data_status_message: st.warning("⚠️ Displaying simulated data as live data fetch failed.")
    else: st.error("🚨 Failed to load any economic data. Check sources/network, then retry.")
elif economic_df_filtered.empty:
    st.warning("⚠️ No economic events match filters. Adjust date range or criteria in sidebar.")
else:
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.subheader("🗓️ Select Economic Event")
    if 'Timestamp' in economic_df_filtered.columns and pd.api.types.is_datetime64_any_dtype(economic_df_filtered['Timestamp']):
        economic_df_filtered['display_name'] = economic_df_filtered.apply(lambda r: (f"{convert_and_format_time(r['Timestamp'], selected_tz_name, fmt='%Y-%m-%d %I:%M %p')} | " f"{r.get('Currency','N/A')} | {r.get('Impact','N/A')} | {r.get('EventName','Unknown Event')}"), axis=1)
    else:
        economic_df_filtered['display_name'] = economic_df_filtered.apply(lambda r: f"Data Error - {r.get('Currency','N/A')} - {r.get('EventName','Unknown Event')}", axis=1)
        st.error("🚨 Timestamp data issue. Event selection might be affected.")
    event_options = economic_df_filtered.sort_values(by='Timestamp')['display_name'].tolist()
    sel_key = "current_event_selectbox_main"
    if not event_options: st.session_state[sel_key] = None
    elif sel_key not in st.session_state or st.session_state[sel_key] not in event_options: st.session_state[sel_key] = event_options[0]
    selected_event_display_name = st.selectbox("Choose event:", options=event_options, key=sel_key, label_visibility="collapsed", index=0 if not event_options or st.session_state[sel_key] is None else event_options.index(st.session_state[sel_key]), help="Select event for details.")
    st.markdown('</div>', unsafe_allow_html=True)

    if not event_options or selected_event_display_name is None or selected_event_display_name.startswith("Invalid") or selected_event_display_name.startswith("Data Error"):
        st.error("🚨 No valid event selected. Adjust filters or check data integrity."); st.stop()

    selected_event_row = economic_df_filtered[economic_df_filtered['display_name'] == selected_event_display_name].iloc[0]
    event_id = str(selected_event_row.get('id', str(selected_event_row.get('EventName','')) + str(selected_event_row.get('Currency','')) + str(selected_event_row.get('Timestamp',''))))
    prev_val, fcst_val, act_val = selected_event_row.get('Previous'), selected_event_row.get('Forecast'), selected_event_row.get('Actual')
    evt_name, cur_str, imp_str = str(selected_event_row.get('EventName', 'N/A')), str(selected_event_row.get('Currency', 'N/A')), str(selected_event_row.get('Impact', 'N/A'))
    evt_ts = selected_event_row.get('Timestamp'); fmt_evt_time = convert_and_format_time(evt_ts, selected_tz_name)
    
    indicator_props = get_indicator_properties(evt_name) 

    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.subheader(f"🔍 Details: {evt_name}")
    det_cols = st.columns([1.5, 1.5, 1, 1, 1])
    with det_cols[0]: st.metric(label="Currency Pair", value=f"{cur_str}/USD" if cur_str!="USD" and cur_str!="N/A" else cur_str)
    with det_cols[1]: st.metric(label="Impact Level", value=imp_str)
    time_p, date_p = "N/A", fmt_evt_time
    if fmt_evt_time not in ["N/A", "Invalid Time"] and ' ' in fmt_evt_time: parts = fmt_evt_time.split(' '); date_p=parts[0]; time_p=f"{parts[1]} {parts[2]}" if len(parts)>=3 else parts[1]
    with det_cols[2]: st.metric(label="Previous", value=f"{prev_val:.2f}" if pd.notna(prev_val) else "N/A")
    with det_cols[3]: st.metric(label="Forecast", value=f"{fcst_val:.2f}" if pd.notna(fcst_val) else "N/A")
    with det_cols[4]: st.metric(label="Actual", value=f"{act_val:.2f}" if pd.notna(act_val) else "Pending")
    st.caption(f"Scheduled: {date_p} at {time_p} ({selected_tz_name})")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.subheader("🎯 Baseline Interpretation")
    inferred_bias = infer_market_outlook_from_data(prev_val, fcst_val, evt_name)
    st.markdown(f"<div class='custom-info-box'>System-Inferred Bias (Forecast vs. Previous): <strong>{inferred_bias}</strong> for {cur_str}</div>", unsafe_allow_html=True)
    
    st.markdown("<h5>Desired Market Scenario Analysis (Baseline)</h5>", unsafe_allow_html=True)
    outcome_opts = ["Bullish", "Bearish", "Consolidating"]; def_idx = 2
    if inferred_bias and "qualitative" not in inferred_bias.lower(): 
        if "bullish" in inferred_bias.lower(): def_idx = 0
        elif "bearish" in inferred_bias.lower(): def_idx = 1
    
    desired_outcome = st.radio(f"Select desired outcome for {cur_str}:", options=outcome_opts, index=def_idx, key=f"outcome_radio_{event_id}", horizontal=True, help="Choose market direction for baseline scenario.")
    prediction_pts = predict_actual_condition_for_outcome(prev_val, fcst_val, desired_outcome, cur_str, evt_name)
    outcome_colors = {"Bullish": "#28a745", "Bearish": "#dc3545", "Consolidating": "#6c757d"}
    box_color = outcome_colors.get(desired_outcome, "#6c757d")
    pred_html_list = "".join([f"<li>{pt}</li>" for pt in prediction_pts])
    st.markdown(f"<div class='custom-prediction-box' style='background-color: {box_color}; border-left: 5px solid {box_color};'><ul style='margin-bottom:0; padding-left:20px;'>{pred_html_list}</ul></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if indicator_props["type"] == "qualitative":
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.subheader("🤖 AI Sentiment Analysis (Qualitative Event)")
        st.markdown(f"Since **{evt_name}** is a qualitative event, its impact is driven by the perceived tone and message. Use the AI assistant below to explore potential interpretations based on a selected sentiment.")

        qual_sentiment_options = ["Hawkish", "Neutral", "Dovish"]
        user_qual_sentiment = st.radio("Select perceived sentiment:",options=qual_sentiment_options,index=1,key=f"qual_sentiment_{event_id}",horizontal=True,help="Indicate perceived tone.")

        if st.button("Analyze Qualitative Sentiment with AI", key=f"analyze_qual_ai_{event_id}", use_container_width=True):
            with st.spinner(f"🧠 AI is analyzing '{evt_name}' with a '{user_qual_sentiment}' sentiment for {cur_str}..."):
                event_desc_for_ai = indicator_props.get("description", "A significant qualitative economic event.")
                ai_analysis_result = analyze_qualitative_event_llm(evt_name, user_qual_sentiment, cur_str, event_desc_for_ai)
            
            st.markdown(f"<h5>AI Analysis Result (Sentiment: {user_qual_sentiment})</h5>", unsafe_allow_html=True)
            res_color = "#6c757d" 
            if user_qual_sentiment == "Hawkish": res_color = "#28a745"
            elif user_qual_sentiment == "Dovish": res_color = "#dc3545"
            st.markdown(f"<div class='custom-info-box' style='border-left-color: {res_color};'><strong>Summary:</strong> {ai_analysis_result.get('summary', 'N/A')}</div>", unsafe_allow_html=True)
            cols_ai = st.columns(2)
            with cols_ai[0]:
                st.markdown("**Potential Bullish Points:**"); [st.markdown(f"- {pt}") for pt in ai_analysis_result.get("bullish_points", [])] if ai_analysis_result.get("bullish_points") else st.caption("N/A")
            with cols_ai[1]:
                st.markdown("**Potential Bearish Points:**"); [st.markdown(f"- {pt}") for pt in ai_analysis_result.get("bearish_points", [])] if ai_analysis_result.get("bearish_points") else st.caption("N/A")
            st.markdown("**Key Considerations & Factors:**"); [st.markdown(f"- {pt}") for pt in ai_analysis_result.get("key_considerations", [])] if ai_analysis_result.get("key_considerations") else st.caption("N/A")
            st.markdown(f"<div style='font-size:0.8rem;color:#A0A0A0;margin-top:10px;'><strong>Impact Scale:</strong> {ai_analysis_result.get('potential_impact','N/A')} | <strong>AI Confidence:</strong> {ai_analysis_result.get('confidence','N/A')}</div>", unsafe_allow_html=True)
            st.caption(f"ℹ️ {ai_analysis_result.get('disclaimer', 'AI analysis is experimental.')}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    tab_hist_label = "📈 Historical Trends"
    tab_sim_label = "🔬 Simulate Actual Release"
    if indicator_props["type"] == "qualitative": tab_sim_label = "🔬 Numerical Simulation (N/A for Purely Qualitative)"
    tab_hist, tab_sim = st.tabs([tab_hist_label, tab_sim_label])

    with tab_hist:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.subheader(f"Historical Trends for: {evt_name}")
        is_av_source = False
        if 'ALPHA_VANTAGE_API_KEY' in st.secrets:
            evt_to_av_map = {"Non-Farm Employment Change":{},"Unemployment Rate":{},"Core CPI m/m":{},"CPI m/m":{},"Retail Sales m/m":{},"Real GDP":{},"Treasury Yield":{},"Federal Funds Rate":{}}
            if any(key_evt.lower() in evt_name.lower() for key_evt in evt_to_av_map): is_av_source = True
        df_hist = load_historical_data(evt_name)
        if not df_hist.empty:
            cap_txt = f"Displaying historical data for {evt_name}. "
            if is_av_source and 'Forecast' not in df_hist.columns: cap_txt += "Sourced from Alpha Vantage (US Data - 'Actual' values). Forecast/Previous may not be available."
            else: cap_txt += "May include sample data if live source is unavailable or lacks all fields."
            st.caption(cap_txt); plot_historical_trend(df_hist, evt_name, indicator_props.get("type", "normal"))
        else: st.info(f"ℹ️ No historical data found for '{evt_name}'.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_sim:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.subheader(f"Simulate Actual Release Impact for: {evt_name}")
        if indicator_props["type"] == "qualitative":
            st.info(f"ℹ️ Numerical simulation is not applicable for purely qualitative events like '{evt_name}'.")
        else:
            st.markdown("Enter a hypothetical 'Actual' value to see its classified impact.")
            unit_sim = indicator_props.get("unit",""); step_v = 0.1 if "%" in unit_sim else (1.0 if "K" in unit_sim or "M" in unit_sim else 0.01)
            actual_in_def = fcst_val if pd.notna(fcst_val) else (prev_val if pd.notna(prev_val) else 0.0)
            hyp_actual = st.number_input(f"Hypothetical 'Actual' ({unit_sim}):",value=float(actual_in_def) if pd.notna(actual_in_def) else 0.0,step=float(step_v),format="%.2f",key=f"actual_input_{event_id}",help=f"Enter hypothetical actual for {evt_name}.")
            if st.button("Classify Hypothetical Actual", key=f"classify_btn_{event_id}", use_container_width=True):
                classification, explanation = classify_actual_release(hyp_actual, fcst_val, prev_val, evt_name, cur_str)
                nu_colors = {"Strongly Bullish":"#145A32","Mildly Bullish":"#27AE60","Neutral/In-Line":"#808B96","Mildly Bearish":"#E74C3C","Strongly Bearish":"#922B21","Qualitative":"#2E4053","Indeterminate":"#4A235A","Error":"#641E16"}
                cls_bg_color = nu_colors.get(classification, "#333333")
                st.markdown(f"**Classification:** <span style='background-color:{cls_bg_color};color:white;padding:3px 7px;border-radius:4px;font-weight:bold;'>{classification}</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='custom-classification-box' style='border-color:{cls_bg_color};background-color:#1c1e22;'>{explanation}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("🗓️ Full Economic Calendar Overview (Filtered)", expanded=False):
        if not economic_df_filtered.empty:
            cal_df = economic_df_filtered.copy()
            if 'Timestamp' in cal_df.columns and pd.api.types.is_datetime64_any_dtype(cal_df['Timestamp']):
                cal_df['FormattedTimestamp'] = cal_df['Timestamp'].apply(lambda x: convert_and_format_time(x, selected_tz_name, fmt="%Y-%m-%d %I:%M %p"))
                disp_cols = ['FormattedTimestamp','Currency','Impact','EventName','Previous','Forecast','Actual','Zone']
                cal_df = cal_df[[col for col in disp_cols if col in cal_df.columns]]
                cal_df.rename(columns={'FormattedTimestamp':'Time','EventName':'Event Name','Currency':'Ccy'},inplace=True)
            else:
                cal_df['Time']="Data Error"; disp_cols_err=['Time','Currency','Impact','EventName','Previous','Forecast','Actual','Zone']
                cal_df = cal_df[[col for col in disp_cols_err if col in cal_df.columns]]
            st.dataframe(cal_df.sort_values(by='Time'),use_container_width=True,hide_index=True,height=450,
                column_config={
                    "Time":st.column_config.TextColumn("Time",width="medium",help="Scheduled time"),"Ccy":st.column_config.TextColumn("Ccy",width="small"),
                    "Impact":st.column_config.TextColumn("Impact",width="small"),"Event Name":st.column_config.TextColumn("Event",width="large"),
                    "Previous":st.column_config.NumberColumn("Prev.",format="%.2f",width="small"),"Forecast":st.column_config.NumberColumn("Fcst.",format="%.2f",width="small"),
                    "Actual":st.column_config.NumberColumn("Actual",format="%.2f",width="small"),"Zone":st.column_config.TextColumn("Zone",width="small"),
                })
        else: st.info("ℹ️ No events to display in calendar based on current filters.")

st.markdown("---")
st.caption("Trading Mastery Hub © 2024-2025 | Disclaimer: Generalized interpretations, not financial advice. Data accuracy depends on sources and is not guaranteed. Always conduct your own research.")

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
    page_title="Economic Impact Forecaster V2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Data ---
economic_df = load_economic_data()

# --- Application Title ---
st.title("📊 Economic Impact Forecaster V2")
st.markdown("""
Analyze economic data releases, view historical trends, and simulate impacts of hypothetical 'Actual' values.
The system infers market bias (Forecast vs. Previous) and helps interpret outcomes.
""")

# --- Main Application Logic ---
if economic_df.empty:
    st.error("🚨 Failed to load economic data. Please check `data_loader.py` or the data source.")
else:
    # --- Sidebar ---
    st.sidebar.header("🗓️ Event Selection & Configuration")
    economic_df['display_name'] = economic_df.apply(
        lambda row: f"{row['Timestamp']} - {row['Currency']} - {row['EventName']}"
        if pd.notna(row['Currency']) and pd.notna(row['EventName'])
        else f"{row['Timestamp']} - Event details missing",
        axis=1
    )
    event_options = economic_df['display_name'].tolist()
    selected_event_display_name = st.sidebar.selectbox(
        "Select Economic Event:",
        options=event_options,
        index=0
    )
    selected_event_row = economic_df[economic_df['display_name'] == selected_event_display_name].iloc[0]
    
    # --- Event Details Display in Sidebar ---
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"Selected: {selected_event_row['EventName']}")
    st.sidebar.caption(f"Currency: {selected_event_row['Currency']} | Impact: {selected_event_row['Impact']} | Time: {selected_event_row['Timestamp']}")
    
    previous_val = selected_event_row['Previous']
    forecast_val = selected_event_row['Forecast']
    event_name_str = str(selected_event_row['EventName']) # Ensure it's a string
    currency_str = str(selected_event_row['Currency'])

    st.sidebar.markdown(f"**Previous:** `{previous_val if pd.notna(previous_val) else 'N/A'}`")
    st.sidebar.markdown(f"**Forecast:** `{forecast_val if pd.notna(forecast_val) else 'N/A'}`")

    # --- Tabs for Main Content ---
    tab1, tab2, tab3 = st.tabs(["🔍 Interpretation & Outlook", "📈 Historical Trends", "🔬 Simulate Actual Release"])

    with tab1:
        st.header(f"🔍 Interpretation for: {event_name_str} ({currency_str})")
        
        # Inferred Market Outlook
        inferred_outcome = infer_market_outlook_from_data(
            previous_val,
            forecast_val,
            event_name_str
        )
        st.info(f"System-Inferred Bias (Forecast vs. Previous): **{inferred_outcome}** for {currency_str}")

        # Desired Outcome Analysis
        st.subheader("🎯 Desired Market Outcome Analysis")
        outcome_options = ["Bullish", "Bearish", "Consolidating"]
        try:
            default_outcome_index = outcome_options.index(inferred_outcome)
        except ValueError: # Handle cases where inferred_outcome might be like "Consolidating (Qualitative...)"
            if "bullish" in inferred_outcome.lower(): default_outcome_index = 0
            elif "bearish" in inferred_outcome.lower(): default_outcome_index = 1
            else: default_outcome_index = 2
        
        desired_outcome = st.radio(
            f"Select desired outcome for {currency_str} to analyze:",
            options=outcome_options,
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
        # Custom styling for interpretation text
        outcome_color_map = {
            "Bullish": "#1E4620", "Bearish": "#541B1B", "Consolidating": "#333333",
            "Qualitative": "#2E4053", "Indeterminate": "#4A235A", "Error": "#641E16"
        }
        bg_color = outcome_color_map.get(desired_outcome, "#333333")
        st.markdown(f"<div style='background-color: {bg_color}; color: #FAFAFA; padding: 15px; border-radius: 8px; border: 1px solid #4F4F4F; margin-top:10px;'>{prediction_text}</div>", unsafe_allow_html=True)

    with tab2:
        st.header(f"📈 Historical Trends for: {event_name_str}")
        df_hist = load_historical_data(event_name_str)
        if not df_hist.empty:
            indicator_props = get_indicator_properties(event_name_str)
            plot_historical_trend(df_hist, event_name_str, indicator_props.get("type", "normal"))
        else:
            st.info(f"No specific historical data found for '{event_name_str}'. Displaying generic message.")
            # Optionally, show a placeholder or a message about API integration for live history.

    with tab3:
        st.header(f"🔬 Simulate Actual Release Impact for: {event_name_str}")
        st.markdown("Enter a hypothetical 'Actual' value to see how it might be classified.")

        indicator_props_sim = get_indicator_properties(event_name_str)
        unit_sim = indicator_props_sim.get("unit", "")
        
        # Determine a reasonable step for number_input based on indicator type or typical values
        step_value = 0.1 if "%" in unit_sim else 1.0 if "K" in unit_sim else 0.01
        if indicator_props_sim["type"] == "qualitative":
             st.warning(f"'{event_name_str}' is a qualitative event. Numerical simulation is not applicable. Interpretation depends on announcement content.")
        else:
            actual_input_val = forecast_val if pd.notna(forecast_val) else (previous_val if pd.notna(previous_val) else 0.0)
            
            hypothetical_actual = st.number_input(
                f"Enter Hypothetical 'Actual' Value ({unit_sim}):",
                value=float(actual_input_val) if pd.notna(actual_input_val) else 0.0, # Default to forecast or previous if available
                step=step_value,
                format="%.2f", # Adjust format as needed
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

                else:
                    st.error("🚨 Please enter a valid 'Actual' value.")
    
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Economic Calendar Overview")
    # Using st.data_editor for a more interactive table (read-only for now)
    # Configure column widths or other properties as needed
    column_config = {
        "Timestamp": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
        "EventName": st.column_config.TextColumn("Event", width="large"),
        "Impact": st.column_config.TextColumn("Impact", width="small"),
        "Previous": st.column_config.NumberColumn("Prev.", format="%.2f"),
        "Forecast": st.column_config.NumberColumn("Fcst.", format="%.2f"),
    }
    # Select and rename columns for display
    display_df = economic_df[['Timestamp', 'Currency', 'EventName', 'Impact', 'Previous', 'Forecast']].copy()
    display_df.rename(columns={'EventName': 'Event Name'}, inplace=True)

    st.sidebar.dataframe( # Using dataframe for sidebar as data_editor might be too wide
        display_df,
        column_config={
            "Timestamp": st.column_config.TextColumn("Time (EST)"), # Simpler text for sidebar
            "Impact": st.column_config.TextColumn("Impact", width="small"),
            "Previous": st.column_config.NumberColumn("Prev.", format="%.2f"),
            "Forecast": st.column_config.NumberColumn("Fcst.", format="%.2f"),
        },
        use_container_width=True,
        hide_index=True,
        height=300 # Limit height in sidebar
    )


    # --- Footer & Disclaimer ---
    st.markdown("---")
    st.caption("""
    **Disclaimer:** This tool provides generalized interpretations based on common market reactions and should not be considered financial advice.
    Actual market movements can be influenced by a wide array of factors. The simulated data is for demonstration purposes only.
    Historical data is sample data. For live and comprehensive data, API integration is required.
    Latency: Data loading is cached. For live API data, consider asynchronous fetching or background updates for optimal performance.
    """)

# --- Developer Suggestions & Next Steps ---
# (Keep the suggestions from the previous response or update as needed)
"""
## Developer Suggestions & Next Steps:

1.  **Live Data Integration (`data_loader.py`):**
    * **Economic Calendar API:** Integrate a free-tier API (e.g., Finnhub, Econdb, Alpha Vantage) for real-time economic calendar data.
    * **Historical Data API:** Use the same or another API to fetch actual historical data series for indicators.
    * **API Key Management:** Use `st.secrets` for secure API key storage.
    * **Error Handling & Retries:** Implement robust error handling (timeouts, rate limits) for API calls.

2.  **Refine `strategy_engine.py`:**
    * **Expand `INDICATOR_CONFIG`:** Add more indicators with fine-tuned `significance_threshold_pct`, `buffer_pct`, and `unit`. Consider regional variations if applicable.
    * **Advanced Qualitative Analysis:** For speeches, explore basic NLP techniques (e.g., keyword spotting for hawkish/dovish terms) if summaries are available via API.
    * **Volatility-Adjusted Buffers:** Instead of fixed percentages, buffers could be dynamic based on an indicator's historical realized volatility or implied volatility from options markets (if available).

3.  **Enhance `visualization.py`:**
    * **More Chart Types:** Offer bar charts for deviations (Actual vs. Forecast), scatter plots for correlations, etc.
    * **Interactive Chart Elements:** Allow users to click on data points in charts to see more details or link to related news.
    * **Overlay Economic Events on Price Charts:** If integrating with a financial charting library, allow overlaying event release times on asset price charts.

4.  **`app.py` UI/UX and Performance:**
    * **`st.data_editor` for Main Calendar:** If you want the main calendar in the central area to be editable (e.g., for users to add notes or their own expectations), `st.data_editor` is powerful. For now, the sidebar uses `st.dataframe`.
    * **Custom CSS for Table Styling:** For more advanced styling of dataframes/tables beyond `st.data_editor`'s capabilities, use `st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)`.
    * **Asynchronous Operations:** For API calls that might take time, investigate `st.experimental_async` or threading to prevent UI blocking, especially for live data refreshes. (Note: Streamlit's execution model needs careful handling of async operations).
    * **State Management:** For more complex interactions, explore Streamlit's session state (`st.session_state`) more extensively.

5.  **AI/ML Integration (`ai_models.py` - Future):**
    * **Predict Market Impact:** Train a model (e.g., classification or regression) to predict the probability or magnitude of market reaction based on `(Actual - Forecast)`, indicator type, and other features.
    * **LLM for Summaries/Explanations:** Use an LLM to generate more natural language summaries of an event's potential impact or explain the model's prediction.

6.  **Code Quality and Deployment:**
    * **Unit Tests:** Add unit tests for functions in `strategy_engine.py`, `data_loader.py`, and `visualization.py`.
    * **Logging:** Implement more structured logging for easier debugging, especially for API interactions.
    * **Streamlit Cloud Deployment:** Ensure `requirements.txt` is complete for seamless deployment.
"""

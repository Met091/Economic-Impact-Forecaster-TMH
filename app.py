# app.py
import streamlit as st
import pandas as pd
import numpy as np # Import numpy for np.isnan checks
from data_loader import load_economic_data
from strategy_engine import predict_actual_condition_for_outcome, infer_market_outlook_from_data

# --- Page Configuration ---
st.set_page_config(
    page_title="Economic Impact Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Data ---
# This will run once and cache the data due to @st.cache_data in data_loader
economic_df = load_economic_data()

# --- Application Title ---
st.title("📈 Economic Impact Forecaster")
st.markdown("""
This application displays an economic calendar and helps interpret how an economic data release's 'Actual' value,
relative to its 'Forecast' and 'Previous' values, might influence currency markets.
The system will infer a likely market bias based on Forecast vs. Previous data.
""")

# --- Main Application Logic ---
if economic_df.empty:
    st.error("🚨 Failed to load economic data. Please check `data_loader.py` or the data source.")
else:
    # --- Sidebar for Event Selection ---
    st.sidebar.header("🗓️ Event Selection")
    
    # Create a display name for the selectbox: "Time - Currency - EventName"
    # Handle potential NaNs in event components for display
    economic_df['display_name'] = economic_df.apply(
        lambda row: f"{row['Timestamp']} - {row['Currency']} - {row['EventName']}"
        if pd.notna(row['Currency']) and pd.notna(row['EventName'])
        else f"{row['Timestamp']} - Event details missing",
        axis=1
    )
    
    event_options = economic_df['display_name'].tolist()
    # Use index of display_name to get original row id if needed, or directly use display_name to filter
    selected_event_display_name = st.sidebar.selectbox(
        "Select an Economic Event:",
        options=event_options,
        index=0 # Default to the first event
    )

    # Find the selected event details from the DataFrame
    # It's safer to select based on a unique ID if available, or ensure display_name is unique.
    # For now, assuming display_name is sufficiently unique for selection in this sample.
    selected_event_row = economic_df[economic_df['display_name'] == selected_event_display_name].iloc[0]

    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Market Outcome Analysis")

    # Infer market outcome based on Previous and Forecast
    # Ensure that event_name is passed to the function
    inferred_outcome = infer_market_outlook_from_data(
        selected_event_row['Previous'],
        selected_event_row['Forecast'],
        str(selected_event_row['EventName']) # Ensure EventName is a string
    )
    
    # Determine default index for radio button
    outcome_options = ["Bullish", "Bearish", "Consolidating"]
    try:
        default_outcome_index = outcome_options.index(inferred_outcome)
    except ValueError:
        default_outcome_index = 2 # Default to Consolidating if inferred_outcome is unexpected

    st.sidebar.info(f"System-Inferred Bias (Forecast vs. Previous for '{selected_event_row['EventName']}'): **{inferred_outcome}** for {selected_event_row['Currency']}")

    desired_outcome = st.sidebar.radio(
        f"Select desired outcome for {selected_event_row['Currency']} to analyze:",
        options=outcome_options,
        index=default_outcome_index,
        key=f"outcome_{selected_event_row['id']}" # Unique key to reset radio on event change
    )

    # --- Main Panel for Displaying Information ---
    st.header(f"📊 Analysis for: {selected_event_row['EventName']} ({selected_event_row['Currency']})")
    
    # Using st.columns for a cleaner layout of metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Scheduled Time", value=str(selected_event_row['Timestamp']))
    with col2:
        # Simple color coding for impact
        impact_color_map = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
        impact_display = f"{impact_color_map.get(str(selected_event_row['Impact']), '⚪')} {selected_event_row['Impact']}"
        st.metric(label="Impact", value=impact_display)
    with col3:
        st.metric(label="Currency", value=str(selected_event_row['Currency']))

    st.subheader("📋 Event Data Snapshot")
    # Using columns for Previous, Forecast, and Deviation for better alignment
    data_col1, data_col2, data_col3 = st.columns(3)
    previous_val = selected_event_row['Previous']
    forecast_val = selected_event_row['Forecast']

    with data_col1:
        st.markdown(f"**Previous:**")
        st.markdown(f"<h3 style='text-align: left; color: #FFBF00;'>{previous_val if pd.notna(previous_val) else 'N/A'}</h3>", unsafe_allow_html=True)
        
    with data_col2:
        st.markdown(f"**Forecast:**")
        st.markdown(f"<h3 style='text-align: left; color: #00A0B0;'>{forecast_val if pd.notna(forecast_val) else 'N/A'}</h3>", unsafe_allow_html=True)

    with data_col3:
        st.markdown(f"**Deviation (Fcst - Prev):**")
        if pd.notna(previous_val) and pd.notna(forecast_val):
            try:
                # Ensure they are floats for subtraction
                deviation = float(forecast_val) - float(previous_val)
                deviation_color = "green" if deviation > 0 else "red" if deviation < 0 else "gray" # Using more standard green/red
                st.markdown(f"<h3 style='text-align: left; color: {deviation_color};'>{deviation:.2f}</h3>", unsafe_allow_html=True)
                
                # Adding a small textual interpretation of the deviation
                if deviation > 0.001: # Using a small epsilon for float comparison
                    st.caption("📈 Forecast suggests improvement/increase.")
                elif deviation < -0.001:
                    st.caption("📉 Forecast suggests decline/decrease.")
                else:
                    st.caption("➖ Forecast suggests no change.")
            except ValueError: # Handle cases where conversion to float might fail
                st.markdown("<h3 style='text-align: left; color: orange;'>N/A</h3>", unsafe_allow_html=True)
                st.caption("⚠️ Non-numeric data for deviation calculation.")
        else:
            st.markdown("<h3 style='text-align: left; color: orange;'>N/A</h3>", unsafe_allow_html=True)
            st.caption("⚠️ Data missing for deviation calculation.")


    st.subheader(f"💡 Interpretive Outlook for a {desired_outcome} {selected_event_row['Currency']}")
    # Interpretation generates automatically on selection change
    try:
        prediction_text = predict_actual_condition_for_outcome(
            previous=selected_event_row['Previous'],
            forecast=selected_event_row['Forecast'],
            desired_outcome=desired_outcome,
            currency=selected_event_row['Currency'],
            event_name=str(selected_event_row['EventName']) # Ensure EventName is a string
        )
        
        # Using st.info, st.warning, st.success for different outcomes could be an option for semantic coloring
        # For dark mode, custom background colors provide better contrast
        if desired_outcome == "Bullish":
            st.markdown(f"<div style='background-color: #1E4620; color: #FAFAFA; padding: 10px; border-radius: 5px; border: 1px solid #2A642D;'>{prediction_text}</div>", unsafe_allow_html=True)
        elif desired_outcome == "Bearish":
            st.markdown(f"<div style='background-color: #541B1B; color: #FAFAFA; padding: 10px; border-radius: 5px; border: 1px solid #7A2828;'>{prediction_text}</div>", unsafe_allow_html=True)
        else: # Consolidating
            st.markdown(f"<div style='background-color: #333333; color: #FAFAFA; padding: 10px; border-radius: 5px; border: 1px solid #4F4F4F;'>{prediction_text}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"🚨 An error occurred during interpretation: {e}")
        # For detailed debugging in development:
        # import traceback
        # st.error(f"Traceback: {traceback.format_exc()}")

    st.markdown("---")
    st.subheader("📆 Upcoming Economic Events Overview")
    # Displaying a subset of columns for brevity, and styling the dataframe for dark mode
    # Note: DataFrame styling might need adjustment for optimal dark mode visibility.
    # Pandas styler has limitations with complex CSS in Streamlit.
    # Consider converting to HTML table with custom CSS if more control is needed.
    st.dataframe(
        economic_df[['Timestamp', 'Currency', 'EventName', 'Impact', 'Previous', 'Forecast']],
        use_container_width=True
    )

    # --- Footer & Disclaimer ---
    st.markdown("---")
    st.caption("""
    **Disclaimer:** This tool provides generalized interpretations based on common market reactions and should not be considered financial advice.
    Actual market movements can be influenced by a wide array of factors. The simulated data is for demonstration purposes only.
    The 'System-Inferred Bias' is a simple heuristic based on Forecast vs. Previous and its accuracy depends on the correct classification of indicator types (e.g., standard vs. inverted).
    """)

# --- Developer Suggestions & Next Steps (from previous responses) ---
# (This section is for developer reference and not displayed in the app)
"""
## Developer Suggestions & Next Steps:

1.  **Refine `infer_market_outlook_from_data` and `predict_actual_condition_for_outcome` in `strategy_engine.py`:**
    * **Indicator-Specific Logic (CRITICAL):** Implement a robust system (e.g., a dictionary lookup or a configuration file) to define the nature of each economic indicator (e.g., `{"Unemployment Rate": {"type": "inverted", "volatility_band": 0.1}}`). This is essential for accurate inference and prediction.
    * **Dynamic Thresholds:** Adjust `significance_threshold` and `buffer` based on the indicator's historical volatility or typical market impact.
    * **Qualitative Event Handling:** Improve interpretation for non-numeric events like speeches.

2.  **API Integration for Live Data (`data_loader.py`):**
    * Integrate a free-tier API (e.g., Finnhub, Econdb, Alpha Vantage) for real-time economic calendar data.
    * Manage API keys securely using `st.secrets`.
    * Implement robust error handling for API calls (timeouts, rate limits, data parsing).

3.  **Advanced AI/ML Integration (New Module: `ai_models.py`):**
    * **Sentiment Analysis:** For speeches or news, use NLP models (e.g., from Hugging Face Transformers) to gauge sentiment if textual data is available.
    * **Predictive Modeling:** Train a classification model (e.g., Logistic Regression, RandomForest, XGBoost) on historical data:
        * Features: `(Actual - Forecast)`, `(Forecast - Previous)`, `Indicator_Type_Encoded`, `Currency_Volatility_Index`.
        * Target: `Market_Reaction_Class` (e.g., Bullish_High_Prob, Bearish_Low_Prob).
    * **Model Explainability (XAI):** Use SHAP or LIME to explain predictions from ML models.

4.  **UI/UX Enhancements (`app.py` and custom CSS):**
    * **Visualizations (`visualization.py`):** Add charts for historical trends of selected indicators (if API provides historical data).
    * **User Input for 'Actual':** Allow users to input a hypothetical 'Actual' value and see the app's classification.
    * **Enhanced Table Styling:** For the economic calendar, consider `st.data_editor` for a more interactive experience or custom HTML/CSS for better dark mode styling.
    * **Latency Optimization:** Ensure all data operations are efficient, especially with live API data.

5.  **System Architecture and Deployment:**
    * **Modularization:** Continue to ensure distinct responsibilities for each Python module.
    * **Testing:** Implement unit tests for core logic in `strategy_engine.py` and `data_loader.py`.
    * **Deployment:** Streamline deployment to Streamlit Cloud via GitHub.

6.  **Error Patterns & Risk Flags:**
    * **Data Quality Alerts:** Implement checks for stale or anomalous data from APIs.
    * **Model Performance Monitoring:** If ML models are used, track their predictive accuracy over time.
"""

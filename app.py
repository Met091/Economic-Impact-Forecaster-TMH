# app.py
import streamlit as st
import pandas as pd
from data_loader import load_economic_data
from strategy_engine import predict_actual_condition_for_outcome

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
This application helps interpret how an economic data release's 'Actual' value, relative to its 'Forecast' and 'Previous' values,
might influence currency markets. Select an event and a desired market outcome to see the analysis.
""")

# --- Main Application Logic ---
if economic_df.empty:
    st.error("Failed to load economic data. Please check the `data_loader.py` or data source.")
else:
    # --- Sidebar for Event Selection ---
    st.sidebar.header("Event Selection")
    
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
    st.sidebar.header("Desired Market Outcome")
    desired_outcome = st.sidebar.radio(
        f"Select desired outcome for {selected_event_row['Currency']}:",
        options=["Bullish", "Bearish", "Consolidating"],
        index=0,
        key=f"outcome_{selected_event_row['id']}" # Unique key to reset radio on event change
    )

    # --- Main Panel for Displaying Information ---
    st.header(f"Analysis for: {selected_event_row['EventName']} ({selected_event_row['Currency']})")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Scheduled Time", value=str(selected_event_row['Timestamp']))
    with col2:
        st.metric(label="Impact", value=str(selected_event_row['Impact']))
    with col3:
        st.metric(label="Currency", value=str(selected_event_row['Currency']))

    st.subheader("Event Data Snapshot")
    data_col1, data_col2 = st.columns(2)
    previous_val = selected_event_row['Previous']
    forecast_val = selected_event_row['Forecast']

    with data_col1:
        st.markdown(f"**Previous:** `{previous_val if pd.notna(previous_val) else 'N/A'}`")
    with data_col2:
        st.markdown(f"**Forecast:** `{forecast_val if pd.notna(forecast_val) else 'N/A'}`")

    # Calculate and display deviation (Forecast - Previous)
    if pd.notna(previous_val) and pd.notna(forecast_val):
        try:
            deviation = float(forecast_val) - float(previous_val)
            st.markdown(f"**Deviation (Forecast - Previous):** `{deviation:.2f}`")
            if deviation > 0:
                st.info("💡 Forecast suggests an improvement or increase compared to the previous period.")
            elif deviation < 0:
                st.warning("💡 Forecast suggests a decline or decrease compared to the previous period.")
            else:
                st.info("💡 Forecast suggests no change compared to the previous period.")
        except ValueError:
            st.markdown("**Deviation (Forecast - Previous):** `N/A (Non-numeric data)`")
            st.warning("Cannot calculate deviation due to non-numeric Previous or Forecast values.")
    else:
        st.markdown("**Deviation (Forecast - Previous):** `N/A (Data missing)`")


    st.subheader("Interpretive Outlook")
    if st.sidebar.button("Generate Interpretation", key="gen_interp_btn", use_container_width=True):
        try:
            prediction_text = predict_actual_condition_for_outcome(
                previous=selected_event_row['Previous'],
                forecast=selected_event_row['Forecast'],
                desired_outcome=desired_outcome,
                currency=selected_event_row['Currency'],
                event_name=selected_event_row['EventName']
            )
            st.success(f"**Interpretation for a {desired_outcome} {selected_event_row['Currency']}:**")
            st.markdown(prediction_text)

        except Exception as e:
            st.error(f"An error occurred during interpretation: {e}")
            # Add more detailed logging for dev/debug builds if needed
            # print(f"Error in predict_actual_condition_for_outcome: {traceback.format_exc()}")

    st.markdown("---")
    st.subheader("Upcoming Economic Events Overview")
    st.dataframe(economic_df[['Timestamp', 'Currency', 'EventName', 'Impact', 'Previous', 'Forecast']], use_container_width=True)

    # --- Footer & Disclaimer ---
    st.markdown("---")
    st.caption("""
    **Disclaimer:** This tool provides generalized interpretations based on common market reactions and should not be considered financial advice.
    Actual market movements can be influenced by a wide array of factors. The simulated data is for demonstration purposes only.
    """)

# --- Suggestions for Next Steps (as per user preferences) ---
# This section could be conditionally displayed or logged for development
# For now, it's a comment block.
"""
## Developer Suggestions & Next Steps:

1.  **API Integration for Live Data:**
    * Replace `data_loader.py`'s sample data with a live API feed.
    * Consider free tiers from Finnhub (economic calendar) or other financial data providers.
    * Implement robust error handling for API calls (timeouts, rate limits, data parsing issues).
    * Ensure API keys are managed securely (e.g., using Streamlit secrets).

2.  **Enhance Prediction Logic (`strategy_engine.py`):**
    * **Event-Specific Buffers/Logic:** Different economic indicators have different volatility and market sensitivity. The `buffer` in `predict_actual_condition_for_outcome` could be made dynamic based on the event type or historical volatility.
    * **Machine Learning Model:** For a more advanced "prediction" of market reaction (rather than interpretation), train a classification model (e.g., Logistic Regression, SVM, RandomForest) on historical data:
        * Features: `(Forecast - Previous)`, `(Actual - Forecast)`, `Impact_Level`, `Currency_Volatility_Index_Pre_Event`.
        * Target: `Market_Reaction_Category` (Bullish, Bearish, Neutral post-release).
        * Use `ai_models.py` for this.
    * **Consider 'Surprise' Magnitude:** The degree of "beat" or "miss" (Actual vs. Forecast) often matters more than the absolute numbers. Quantify this.

3.  **User Interface (UI) and User Experience (UX) Refinements:**
    * **Visualizations:** Add charts showing historical trends for a selected indicator if data is available.
    * **User Input for 'Actual':** Allow users to input a hypothetical 'Actual' value and see the app's classification of the outcome.
    * **Filtering/Sorting Calendar:** Add options to filter the economic calendar by currency, impact, or date range.
    * **Real-time Updates (Advanced):** For live data, implement mechanisms for periodic refreshes or streaming updates (e.g., using `st.experimental_rerun` with a timer, or more complex WebSocket integrations if the API supports it).

4.  **Code Optimization and Modularity:**
    * **Configuration File for Strategy:** Move thresholds or model parameters from `strategy_engine.py` to a configuration file (e.g., YAML or JSON) for easier tuning.
    * **Unit Tests:** Add unit tests for `data_loader.py` and `strategy_engine.py` functions.

5.  **Error Patterns & Risk Flags:**
    * **Data Quality Checks:** Implement checks for missing/stale data from APIs.
    * **API Rate Limiting:** Handle API rate limits gracefully (e.g., with backoff strategies).
    * **Model Confidence:** If using ML models, display confidence scores for predictions.
"""

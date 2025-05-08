# Economic Impact Forecaster Streamlit App

This Streamlit application displays an economic calendar and provides a tool to interpret potential economic data releases.

## Features

* **Economic Calendar**: Shows a list of upcoming (simulated) economic events with their currency, impact, previous values, and forecast values.
* **Impact Interpretation**: For a selected event, users can choose a desired market outcome (Bullish, Bearish, Consolidating for the currency). The app then describes what the 'Actual' data release would likely need to be to achieve that outcome, considering the 'Forecast' and 'Previous' figures.

## Structure

* `app.py`: The main Streamlit application file containing the UI logic.
* `data_loader.py`: Handles loading of economic calendar data (currently simulated).
* `strategy_engine.py`: Contains the logic for interpreting event outcomes.
* `.streamlit/config.toml`: Streamlit configuration for theme and settings.
* `requirements.txt`: Python dependencies for the project.

## How to Run

1.  **Clone the repository or create the files locally.**
    ```bash
    mkdir economic_impact_app
    cd economic_impact_app
    # Create the files as specified
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

The application should open in your default web browser.

## Future Enhancements (Suggestions from AI)
* Integrate a free tier API (e.g., Finnhub, Econdb) for live economic calendar data.
* Allow users to input their own 'Actual' values and see the resulting market sentiment.
* Develop a more sophisticated prediction model using historical data and machine learning (e.g., classifying market reaction based on `Actual - Forecast` deviation).
* Add charts to visualize historical data for selected indicators.
* Implement user accounts or data persistence if custom settings or analyses are needed.

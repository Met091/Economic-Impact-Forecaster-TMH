# strategy_engine.py
import numpy as np
import streamlit as st # Used only for potential error logging if needed

# --- Indicator Configuration ---
NUANCED_MULTIPLIER = 2.0 # How much stronger a "strong" beat/miss is than a mild one

INDICATOR_CONFIG = {
    "Non-Farm Employment Change": { # USA NFP
        "type": "normal", "unit": "K",
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05,
        "abs_significance": 30.0, "abs_buffer": 20.0, # Absolute K deviation
        "description": "Measures the change in the number of employed people during the previous month, excluding the farming industry. Higher is generally better for the currency."
    },
    "Employment Change": { # Assuming CAD Employment Change or similar
        "type": "normal", "unit": "K",
        "significance_threshold_pct": 0.15, "buffer_pct": 0.10,
        "abs_significance": 7.0, "abs_buffer": 5.0, # Absolute K deviation
        "description": "Measures the change in the number of employed people. Higher is generally better for the currency."
    },
    "Unemployment Rate": { # Most countries
        "type": "inverted", "unit": "%",
        "significance_threshold_pct": 0.03, "buffer_pct": 0.02, # e.g., 3% of 3.5% rate = ~0.1%
        "abs_significance": 0.1, "abs_buffer": 0.1, # Absolute % point deviation
        "description": "Measures the percentage of the total labor force that is unemployed but actively seeking employment and willing to work. Lower is generally better for the currency."
    },
    "GDP m/m": { # Monthly GDP Growth
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.20, "buffer_pct": 0.10,
        "abs_significance": 0.1, "abs_buffer": 0.1, # Absolute % point deviation
        "description": "Measures the month-over-month change in the inflation-adjusted value of all goods and services produced by the economy. Higher is generally better."
    },
     "GDP q/q": { # Quarterly GDP Growth
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.15, "buffer_pct": 0.10,
        "abs_significance": 0.2, "abs_buffer": 0.1, # Absolute % point deviation
        "description": "Measures the quarter-over-quarter change in the inflation-adjusted value of all goods and services produced by the economy. Higher is generally better."
    },
    "Core CPI m/m": { # Monthly Core Inflation
        "type": "normal", "unit": "%", # Higher inflation can lead to rate hikes, strengthening currency (context-dependent)
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05,
        "abs_significance": 0.1, "abs_buffer": 0.1, # Absolute % point deviation
        "description": "Measures the month-over-month change in the price of goods and services, excluding food and energy. Higher can indicate inflationary pressure."
    },
     "CPI m/m": { # Monthly Headline Inflation
        "type": "normal", "unit": "%", # Similar to Core CPI in interpretation for FX in many contexts
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05,
        "abs_significance": 0.1, "abs_buffer": 0.1, # Absolute % point deviation
        "description": "Measures the month-over-month change in the price of goods and services. Higher can indicate inflationary pressure."
    },
    "Policy Rate": { # Generic for Interest Rate decisions (BoJ, Fed, ECB etc.)
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.05, "buffer_pct": 0.02, # Rates often move in 0.25% increments
        "abs_significance": 0.15, "abs_buffer": 0.10, # If rates are expected to change, a 0.25% move is standard
        "description": "The interest rate at which commercial banks can borrow money from the central bank. Higher rates typically strengthen a currency."
    },
    "Retail Sales m/m": { # Monthly Retail Sales
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.15, "buffer_pct": 0.10,
        "abs_significance": 0.3, "abs_buffer": 0.2, # Retail sales can be volatile
        "description": "Measures the month-over-month change in the total value of sales at the retail level. Higher indicates stronger consumer spending."
    },
    "PMI": { # Generic for PMI (Manufacturing, Services) - Threshold is often 50
        "type": "normal", "unit": "", # PMI is an index level
        "significance_threshold_pct": 0.02, "buffer_pct": 0.01, # e.g., 2% of 50 = 1 point
        "abs_significance": 1.0, "abs_buffer": 0.8, # Absolute point deviation
        "description": "Purchasing Managers' Index; an indicator of economic health for manufacturing or services sectors. Above 50 indicates expansion, below 50 indicates contraction."
    },
    "ECB President Speaks": {"type": "qualitative", "unit": "", "description": "Speeches by central bank heads can cause significant volatility based on their tone regarding monetary policy."},
    "FOMC Press Conference": {"type": "qualitative", "unit": "", "description": "Press conference following the Federal Open Market Committee meeting, providing insights into US monetary policy."},
    "Default": { # Fallback uses percentages if absolute values are not sensible
        "type": "normal", "unit": "",
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05,
        "default_significance": 0.1, "default_buffer": 0.1, # Default absolute values if calculation from value is zero
        "description": "Default interpretation rules for unclassified economic indicators."
    }
}

def get_indicator_properties(event_name):
    """
    Fetches properties for a given economic event name.
    Prioritizes more specific matches in INDICATOR_CONFIG.
    """
    best_match_key = "Default"
    max_match_len = 0
    event_name_lower = event_name.lower()

    for key in INDICATOR_CONFIG:
        key_lower = key.lower()
        if key_lower in event_name_lower:
            if len(key_lower) > max_match_len:
                max_match_len = len(key_lower)
                best_match_key = key
        # Allow partial match for generic terms like "Policy Rate" or "PMI"
        # Ensure these are checked after more specific matches by virtue of loop order or add more logic
        elif key_lower == "policy rate" and "rate" in event_name_lower and "unemployment" not in event_name_lower:
             if len(key_lower) > max_match_len: # This condition might be redundant if "Policy Rate" is specific enough
                max_match_len = len(key_lower)
                best_match_key = key
        elif key_lower == "pmi" and "pmi" in event_name_lower:
             if len(key_lower) > max_match_len:
                max_match_len = len(key_lower)
                best_match_key = key

    return INDICATOR_CONFIG[best_match_key].copy() # Return a copy to prevent modification of original config

def _calculate_threshold(value, pct_threshold, abs_threshold, default_abs_threshold):
    """
    Helper to calculate a threshold. Prioritizes absolute threshold if available.
    If value is None or NaN, or if abs_threshold is not None, it won't use percentage.
    """
    if abs_threshold is not None:
        return abs_threshold
    if value is not None and not np.isnan(value) and value != 0 and pct_threshold is not None:
        return abs(value * pct_threshold)
    return default_abs_threshold # Fallback to default absolute threshold

# --- infer_market_outlook_from_data ---
def infer_market_outlook_from_data(previous, forecast, event_name):
    """
    Infers a general market bias (Bullish, Bearish, Consolidating) for the currency
    based on the 'Forecast' relative to 'Previous' for a given economic event.
    """
    props = get_indicator_properties(event_name)
    if props["type"] == "qualitative":
        return "Consolidating (Qualitative)"

    if forecast is None or np.isnan(forecast) or previous is None or np.isnan(previous):
        return "Consolidating (Insufficient Data)"

    try:
        prev_val = float(previous)
        fcst_val = float(forecast)
    except ValueError:
        st.error(f"🚨 Invalid data for inference: Prev={previous}, Fcst={forecast}")
        return "Consolidating (Invalid Data)"

    # Use significance threshold to determine if the forecast is notably different from previous
    significance_threshold = _calculate_threshold(
        prev_val, # Base significance on previous value
        props.get("significance_threshold_pct"),
        props.get("abs_significance"),
        props.get("default_significance", INDICATOR_CONFIG["Default"]["default_significance"])
    )

    deviation = fcst_val - prev_val

    if abs(deviation) < significance_threshold:
        return "Consolidating" # Forecast is not significantly different from previous

    # Determine direction based on deviation and indicator type
    is_bullish_deviation = deviation > 0
    if props["type"] == "inverted": # For inverted, a negative deviation (lower forecast) is bullish
        return "Bullish" if not is_bullish_deviation else "Bearish"
    else: # For normal, a positive deviation (higher forecast) is bullish
        return "Bullish" if is_bullish_deviation else "Bearish"


def predict_actual_condition_for_outcome(previous, forecast, desired_outcome, currency, event_name):
    """
    Predicts what the 'Actual' release would likely need to be for a 'desired_outcome'
    for the specified currency, considering the event's forecast and properties.
    Returns a list of strings for bullet point presentation.
    """
    props = get_indicator_properties(event_name)
    unit = props.get("unit", "")
    indicator_nature_desc = "lower is better" if props['type'] == 'inverted' else "higher is better"
    analysis_points = []

    analysis_points.append(f"**Event Context for {currency} ({event_name}):**")
    analysis_points.append(f"- Indicator Type: {props['type']} ({indicator_nature_desc}).")
    if props.get('description'):
        analysis_points.append(f"- Description: {props['description']}")


    if props["type"] == "qualitative":
        analysis_points.append(f"**For a {desired_outcome} outcome from '{event_name}':**")
        if desired_outcome == "Bullish":
            analysis_points.append(f"- The speech/announcement would likely need hawkish rhetoric (e.g., hinting at tightening monetary policy, optimistic economic outlook).")
        elif desired_outcome == "Bearish":
            analysis_points.append(f"- The speech/announcement would likely need dovish rhetoric (e.g., hinting at easing monetary policy, pessimistic economic outlook).")
        else: # Consolidating
            analysis_points.append(f"- The speech/announcement would need to be in line with current market expectations, offering no new significant policy signals.")
        return analysis_points

    if forecast is None or np.isnan(forecast):
        analysis_points.append(f"⚠️ **Quantitative interpretation for '{event_name}' ({currency}) is not possible:** 'Forecast' data is unavailable.")
        return analysis_points

    try:
        forecast_val = float(forecast)
    except ValueError:
        analysis_points.append(f"⚠️ **Error:** Forecast value '{forecast}' for '{event_name}' ({currency}) is invalid.")
        return analysis_points

    # Calculate buffer around the forecast
    buffer = _calculate_threshold(
        forecast_val,
        props.get("buffer_pct"),
        props.get("abs_buffer"),
        props.get("default_buffer", INDICATOR_CONFIG["Default"]["default_buffer"])
    )
    strong_buffer_multiplier = props.get("nuanced_multiplier", NUANCED_MULTIPLIER) # Use specific or global
    strong_buffer = buffer * strong_buffer_multiplier

    analysis_points.append(f"**Scenario for a {desired_outcome} outcome for {currency} (Forecast: {forecast_val:.2f}{unit}):**")

    if desired_outcome == "Bullish":
        if props["type"] == "inverted": # Lower actual is bullish
            mild_beat_target = forecast_val - buffer
            strong_beat_target = forecast_val - strong_buffer
            analysis_points.append(f"- **Condition:** Actual release significantly *below* forecast ({forecast_val:.2f}{unit}).")
            analysis_points.append(f"- Mildly Bullish: Actual around {mild_beat_target:.2f}{unit} (or lower).")
            analysis_points.append(f"- Strongly Bullish: Actual at or below {strong_beat_target:.2f}{unit}.")
        else: # Higher actual is bullish (normal type)
            mild_beat_target = forecast_val + buffer
            strong_beat_target = forecast_val + strong_buffer
            analysis_points.append(f"- **Condition:** Actual release significantly *above* forecast ({forecast_val:.2f}{unit}).")
            analysis_points.append(f"- Mildly Bullish: Actual around {mild_beat_target:.2f}{unit} (or higher).")
            analysis_points.append(f"- Strongly Bullish: Actual at or above {strong_beat_target:.2f}{unit}.")
    elif desired_outcome == "Bearish":
        if props["type"] == "inverted": # Higher actual is bearish
            mild_miss_target = forecast_val + buffer
            strong_miss_target = forecast_val + strong_buffer
            analysis_points.append(f"- **Condition:** Actual release significantly *above* forecast ({forecast_val:.2f}{unit}).")
            analysis_points.append(f"- Mildly Bearish: Actual around {mild_miss_target:.2f}{unit} (or higher).")
            analysis_points.append(f"- Strongly Bearish: Actual at or above {strong_miss_target:.2f}{unit}.")
        else: # Lower actual is bearish (normal type)
            mild_miss_target = forecast_val - buffer
            strong_miss_target = forecast_val - strong_buffer
            analysis_points.append(f"- **Condition:** Actual release significantly *below* forecast ({forecast_val:.2f}{unit}).")
            analysis_points.append(f"- Mildly Bearish: Actual around {mild_miss_target:.2f}{unit} (or lower).")
            analysis_points.append(f"- Strongly Bearish: Actual at or below {strong_miss_target:.2f}{unit}.")
    elif desired_outcome == "Consolidating":
        lower_bound = forecast_val - buffer
        upper_bound = forecast_val + buffer
        analysis_points.append(f"- **Condition:** Actual release broadly *in line* with forecast ({forecast_val:.2f}{unit}).")
        analysis_points.append(f"- Expected Range: Approximately between {lower_bound:.2f}{unit} and {upper_bound:.2f}{unit}.")
    else:
        analysis_points.append("⚠️ Invalid desired outcome selected.")


    # Add context based on Previous value if available
    if previous is not None and not np.isnan(previous):
        try:
            prev_val = float(previous)
            analysis_points.append(f"- **Context vs. Previous ({prev_val:.2f}{unit}):**")
            forecast_vs_previous_outlook = infer_market_outlook_from_data(prev_val, forecast_val, event_name) # Re-use inference logic
            if "Consolidating" not in forecast_vs_previous_outlook :
                 analysis_points.append(f"  - The current forecast ({forecast_val:.2f}{unit}) already suggests a **{forecast_vs_previous_outlook.lower()}** bias compared to the previous value.")
            else:
                 analysis_points.append(f"  - The current forecast ({forecast_val:.2f}{unit}) is not significantly different from the previous value.")

            if desired_outcome == "Bullish":
                if (props["type"] == "inverted" and forecast_val > prev_val) or \
                   (props["type"] == "normal" and forecast_val < prev_val):
                    analysis_points.append(f"  - Achieving a bullish outcome would require the actual to reverse or significantly outperform the current forecast-previous trend.")
            elif desired_outcome == "Bearish":
                if (props["type"] == "inverted" and forecast_val < prev_val) or \
                   (props["type"] == "normal" and forecast_val > prev_val):
                    analysis_points.append(f"  - Achieving a bearish outcome would require the actual to reverse or significantly underperform the current forecast-previous trend.")
        except ValueError:
             analysis_points.append(f"  - Previous value '{previous}' is invalid for comparison.")
    else:
        analysis_points.append(f"- Previous value is not available for additional context.")

    return analysis_points


# --- classify_actual_release ---
def classify_actual_release(actual_value, forecast_value, previous_value, event_name, currency):
    """
    Classifies an 'Actual' release into nuanced categories (e.g., Strongly Bullish)
    based on its deviation from 'Forecast' and 'Previous', considering indicator properties.
    Returns: (classification_string, explanation_string)
    """
    props = get_indicator_properties(event_name)
    unit = props.get("unit", "")

    if props["type"] == "qualitative":
        return "Qualitative", f"'{event_name}' is a qualitative event (e.g., speech, press conference). Impact is based on rhetoric and sentiment, not numerical deviation."

    if actual_value is None or np.isnan(actual_value):
        return "Indeterminate", f"Actual value for '{event_name}' is missing. Cannot classify."
    if forecast_value is None or np.isnan(forecast_value):
        # If no forecast, classification is harder. Could compare to previous if available.
        # For now, treat as indeterminate if primary comparison point (forecast) is missing.
        return "Indeterminate", f"Forecast value for '{event_name}' is missing. Cannot reliably classify deviation."

    try:
        actual = float(actual_value)
        forecast = float(forecast_value)
    except ValueError:
        st.error(f"🚨 Invalid numeric input for classification: Actual={actual_value}, Forecast={forecast_value}")
        return "Error", "Invalid numeric input provided for actual or forecast value."

    # Calculate buffer around the forecast for classification thresholds
    buffer = _calculate_threshold(
        forecast, # Base buffer on forecast value
        props.get("buffer_pct"),
        props.get("abs_buffer"),
        props.get("default_buffer", INDICATOR_CONFIG["Default"]["default_buffer"])
    )
    strong_buffer_multiplier = props.get("nuanced_multiplier", NUANCED_MULTIPLIER)
    strong_buffer = buffer * strong_buffer_multiplier

    deviation = actual - forecast
    classification = "Neutral/In-Line" # Default classification
    is_inverted = props["type"] == "inverted"

    # Determine classification based on deviation from forecast
    if is_inverted: # Lower is better for currency
        if deviation <= -strong_buffer: classification = "Strongly Bullish"
        elif deviation < -buffer: classification = "Mildly Bullish" # deviation is negative, beyond buffer
        elif deviation >= strong_buffer: classification = "Strongly Bearish"
        elif deviation > buffer: classification = "Mildly Bearish" # deviation is positive, beyond buffer
    else: # Normal type: Higher is better for currency
        if deviation >= strong_buffer: classification = "Strongly Bullish"
        elif deviation > buffer: classification = "Mildly Bullish"
        elif deviation <= -strong_buffer: classification = "Strongly Bearish"
        elif deviation < -buffer: classification = "Mildly Bearish"

    # Construct detailed explanation
    prev_text = ""
    if previous_value is not None and not np.isnan(previous_value):
        try:
            prev_val_float = float(previous_value)
            prev_text = f", Previous: {prev_val_float:.2f}{unit}"
        except ValueError:
            prev_text = f", Previous: {previous_value} (invalid format)"


    explanation_lines = [
        f"**Event:** '{event_name}' ({currency}, Type: {props['type']})",
        f"**Data:** Actual: {actual:.2f}{unit}, Forecast: {forecast:.2f}{unit}{prev_text}",
        f"**Analysis:**",
        f"  - Deviation (Actual - Forecast): {deviation:.2f}{unit}",
        f"  - Interpretation Buffer: ±{buffer:.2f}{unit} (around forecast)",
        f"  - Strong Deviation Threshold: Beyond ±{strong_buffer:.2f}{unit}",
        f"**Outcome for {currency}: {classification}**"
    ]

    # Add specific reasoning for the classification
    if classification == "Strongly Bullish":
        explanation_lines.append(f"  - Reason: Actual significantly {'better (below)' if is_inverted else 'better (above)'} than forecast, exceeding the strong deviation threshold.")
    elif classification == "Mildly Bullish":
        explanation_lines.append(f"  - Reason: Actual moderately {'better (below)' if is_inverted else 'better (above)'} than forecast, exceeding the standard buffer.")
    elif classification == "Strongly Bearish":
        explanation_lines.append(f"  - Reason: Actual significantly {'worse (above)' if is_inverted else 'worse (below)'} than forecast, exceeding the strong deviation threshold.")
    elif classification == "Mildly Bearish":
        explanation_lines.append(f"  - Reason: Actual moderately {'worse (above)' if is_inverted else 'worse (below)'} than forecast, exceeding the standard buffer.")
    else: # Neutral/In-Line
        explanation_lines.append(f"  - Reason: Actual is within the expected buffer range (±{buffer:.2f}{unit}) around the forecast.")

    # Add comparison to previous if available and meaningful
    if previous_value is not None and not np.isnan(previous_value) and pd.notna(actual) and pd.notna(previous_value):
        try:
            prev_val_float = float(previous_value)
            actual_vs_prev_dev = actual - prev_val_float
            prev_comparison_outlook = "better" if (is_inverted and actual < prev_val_float) or (not is_inverted and actual > prev_val_float) else "worse"
            if abs(actual_vs_prev_dev) < buffer : # Using forecast buffer as a generic measure of significance
                 prev_comparison_outlook = "similar to"

            explanation_lines.append(f"  - Versus Previous: Actual ({actual:.2f}{unit}) is {prev_comparison_outlook} Previous ({prev_val_float:.2f}{unit}).")
        except ValueError:
            pass # Already handled prev_text

    return classification, "\n".join(explanation_lines)

# --- (Optional: Add new test cases for added absolute thresholds or bullet list output) ---

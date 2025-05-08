# strategy_engine.py
import numpy as np
import streamlit as st # Used only for potential error logging if needed

# --- Indicator Configuration ---
# Added optional 'abs_buffer' and 'abs_significance'
# Multiplier for nuanced classification (e.g., 2x buffer for 'Strongly')
NUANCED_MULTIPLIER = 2.0 

INDICATOR_CONFIG = {
    # Example: NFP - Use absolute buffer/significance
    "Non-Farm Employment Change": {
        "type": "normal", "unit": "K",
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05, 
        "abs_significance": 30.0, "abs_buffer": 20.0 # e.g., 30K change vs prev is significant, 20K beat/miss of forecast is notable
    },
    "Employment Change": { # Assuming CAD Employment Change
        "type": "normal", "unit": "K",
        "significance_threshold_pct": 0.15, "buffer_pct": 0.10, 
        "abs_significance": 7.0, "abs_buffer": 5.0 
    },
    # Example: Unemployment Rate - Use absolute buffer/significance (small values matter)
    "Unemployment Rate": {
        "type": "inverted", "unit": "%",
        "significance_threshold_pct": 0.03, "buffer_pct": 0.02, 
        "abs_significance": 0.1, "abs_buffer": 0.1 # e.g., 0.1% deviation is notable
    },
    # Example: GDP - Use percentage as absolute might vary wildly
    "GDP m/m": {
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.20, "buffer_pct": 0.10, 
        "default_significance": 0.1, "default_buffer": 0.05 # Fallbacks if prev/forecast is 0
    },
     # Example: CPI - Use absolute buffer/significance
    "Core CPI m/m": {
        "type": "normal", "unit": "%", # Treat higher as bullish (anticipating hikes)
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05, 
        "abs_significance": 0.1, "abs_buffer": 0.1 
    },
    "BoJ Policy Rate": {
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.05, "buffer_pct": 0.02, 
        "abs_significance": 0.05, "abs_buffer": 0.02 # Small changes matter
    },
    "Retail Sales m/m": {
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.15, "buffer_pct": 0.10, 
        "default_significance": 0.2, "default_buffer": 0.1
    },
    "ECB President Speaks": {"type": "qualitative", "unit": ""}, # No thresholds needed
    "Default": { # Fallback uses percentages
        "type": "normal", "unit": "",
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05, 
        "default_significance": 0.1, "default_buffer": 0.1 
    } 
}

def get_indicator_properties(event_name):
    """Fetches properties for a given event name, with a fallback to default."""
    for key in INDICATOR_CONFIG:
        if key.lower() in event_name.lower():
            # Return a copy to prevent modification of the original config
            return INDICATOR_CONFIG[key].copy() 
    return INDICATOR_CONFIG["Default"].copy()

def _calculate_threshold(value, pct_threshold, abs_threshold, default_abs_threshold):
    """Helper to calculate threshold, prioritizing absolute value."""
    if abs_threshold is not None:
        return abs_threshold
    elif value != 0 and pct_threshold is not None:
        return abs(value * pct_threshold)
    else:
        return default_abs_threshold

def infer_market_outlook_from_data(previous, forecast, event_name):
    """
    Infers a likely market outlook based on 'Previous', 'Forecast', 
    using absolute or percentage significance thresholds.
    """
    props = get_indicator_properties(event_name)
    if props["type"] == "qualitative": return "Consolidating (Qualitative)"
    if forecast is None or np.isnan(forecast) or previous is None or np.isnan(previous): return "Consolidating (Insufficient Data)"
    try: prev_val, fcst_val = float(previous), float(forecast)
    except ValueError: return "Consolidating (Invalid Data)"

    significance_threshold = _calculate_threshold(
        prev_val, 
        props.get("significance_threshold_pct"), 
        props.get("abs_significance"), 
        props.get("default_significance", INDICATOR_CONFIG["Default"]["default_significance"])
    )
    
    deviation = fcst_val - prev_val
    if abs(deviation) < significance_threshold: return "Consolidating"
    
    is_bullish_deviation = deviation > 0
    if props["type"] == "inverted": return "Bullish" if not is_bullish_deviation else "Bearish"
    else: return "Bullish" if is_bullish_deviation else "Bearish"


def predict_actual_condition_for_outcome(previous, forecast, desired_outcome, currency, event_name):
    """
    Describes the condition the 'Actual' data would need to meet for a desired outcome,
    using absolute or percentage buffers.
    """
    props = get_indicator_properties(event_name)
    unit = props.get("unit", "")
    if props["type"] == "qualitative":
        # ... (Qualitative logic remains the same) ...
        if desired_outcome == "Bullish": return (f"For a **Bullish** outcome for {currency} from '{event_name}', the speech/announcement would need hawkish rhetoric...")
        elif desired_outcome == "Bearish": return (f"For a **Bearish** outcome for {currency} from '{event_name}', the speech/announcement would need dovish rhetoric...")
        else: return (f"For a **Consolidating/Neutral** outcome for {currency} from '{event_name}', the speech/announcement would need to be in line with expectations...")

    if forecast is None or np.isnan(forecast): return (f"Cannot provide quantitative interpretation for '{event_name}' ({currency}): 'Forecast' data unavailable.")
    try: forecast_val = float(forecast)
    except ValueError: return (f"Forecast value '{forecast}' for '{event_name}' ({currency}) is invalid.")

    buffer = _calculate_threshold(
        forecast_val, 
        props.get("buffer_pct"), 
        props.get("abs_buffer"), 
        props.get("default_buffer", INDICATOR_CONFIG["Default"]["default_buffer"])
    )
    
    outcome_description = ""
    indicator_nature_desc = "lower is better" if props['type'] == 'inverted' else "higher is better"
    strong_buffer = buffer * NUANCED_MULTIPLIER

    # --- Generate prediction text ---
    if desired_outcome == "Bullish":
        if props["type"] == "inverted":
            mild_beat = forecast_val - buffer
            strong_beat = forecast_val - strong_buffer
            outcome_description = (f"For a **Bullish** outcome for {currency} from '{event_name}' ({indicator_nature_desc}), "
                                   f"Actual < {forecast_val:.2f}{unit}. "
                                   f"(e.g., Mildly Bullish if ~{mild_beat:.2f}{unit}, Strongly Bullish if ≤ {strong_beat:.2f}{unit})")
        else: # Normal type
            mild_beat = forecast_val + buffer
            strong_beat = forecast_val + strong_buffer
            outcome_description = (f"For a **Bullish** outcome for {currency} from '{event_name}' ({indicator_nature_desc}), "
                                   f"Actual > {forecast_val:.2f}{unit}. "
                                   f"(e.g., Mildly Bullish if ~{mild_beat:.2f}{unit}, Strongly Bullish if ≥ {strong_beat:.2f}{unit})")
    elif desired_outcome == "Bearish":
        if props["type"] == "inverted":
            mild_miss = forecast_val + buffer
            strong_miss = forecast_val + strong_buffer
            outcome_description = (f"For a **Bearish** outcome for {currency} from '{event_name}' ({indicator_nature_desc}), "
                                   f"Actual > {forecast_val:.2f}{unit}. "
                                   f"(e.g., Mildly Bearish if ~{mild_miss:.2f}{unit}, Strongly Bearish if ≥ {strong_miss:.2f}{unit})")
        else: # Normal type
            mild_miss = forecast_val - buffer
            strong_miss = forecast_val - strong_buffer
            outcome_description = (f"For a **Bearish** outcome for {currency} from '{event_name}' ({indicator_nature_desc}), "
                                   f"Actual < {forecast_val:.2f}{unit}. "
                                   f"(e.g., Mildly Bearish if ~{mild_miss:.2f}{unit}, Strongly Bearish if ≤ {strong_miss:.2f}{unit})")
    elif desired_outcome == "Consolidating":
        lower_bound, upper_bound = forecast_val - buffer, forecast_val + buffer
        outcome_description = (f"For a **Consolidating/Neutral** outcome for {currency} from '{event_name}', "
                               f"Actual ≈ {forecast_val:.2f}{unit} (e.g., between {lower_bound:.2f}{unit} and {upper_bound:.2f}{unit}).")
    else: return "Invalid desired outcome selected."

    # --- Add context based on previous ---
    if previous is not None and not np.isnan(previous) and outcome_description:
        # ... (Contextual logic remains the same as before, just append to outcome_description) ...
        prev_val = float(previous)
        if desired_outcome == "Bullish":
            forecast_improves = (props["type"] == "inverted" and forecast_val < prev_val) or (props["type"] == "normal" and forecast_val > prev_val)
            outcome_description += (f" Context: Forecast ({forecast_val:.2f}{unit}) already suggests {'improvement' if forecast_improves else 'worsening'} from Previous ({prev_val:.2f}{unit}).")
        elif desired_outcome == "Bearish":
            forecast_worsens = (props["type"] == "inverted" and forecast_val > prev_val) or (props["type"] == "normal" and forecast_val < prev_val)
            outcome_description += (f" Context: Forecast ({forecast_val:.2f}{unit}) already suggests {'worsening' if forecast_worsens else 'improvement'} from Previous ({prev_val:.2f}{unit}).")
        elif desired_outcome == "Consolidating":
             if abs(prev_val - forecast_val) < (buffer * 0.1): outcome_description += (f" Context: Forecast ({forecast_val:.2f}{unit}) is very close to Previous ({prev_val:.2f}{unit}).")

    return outcome_description


def classify_actual_release(actual_value, forecast_value, previous_value, event_name, currency):
    """
    Classifies an 'Actual' release using a 5-level scale 
    (Strongly Bullish/Bearish, Mildly Bullish/Bearish, Neutral/In-Line),
    based on deviation from forecast relative to dynamic buffers.
    """
    props = get_indicator_properties(event_name)
    unit = props.get("unit", "")
    if props["type"] == "qualitative": return "Qualitative", f"'{event_name}' is qualitative..." # Abbreviated
    if actual_value is None or np.isnan(actual_value): return "Indeterminate", f"Actual value missing for '{event_name}'..."
    if forecast_value is None or np.isnan(forecast_value): return "Indeterminate", f"Forecast value missing for '{event_name}'..."
    try: actual, forecast = float(actual_value), float(forecast_value)
    except ValueError: return "Error", f"Invalid numeric input for Actual or Forecast..."

    buffer = _calculate_threshold(
        forecast, props.get("buffer_pct"), props.get("abs_buffer"), 
        props.get("default_buffer", INDICATOR_CONFIG["Default"]["default_buffer"])
    )
    strong_buffer = buffer * NUANCED_MULTIPLIER # Threshold for 'Strongly' classification
    deviation = actual - forecast
    
    # --- Determine 5-Level Classification ---
    classification = "Neutral/In-Line" # Default
    is_inverted = props["type"] == "inverted"

    if is_inverted: # Lower is better/bullish
        if deviation <= -strong_buffer: classification = "Strongly Bullish"
        elif deviation < -buffer: classification = "Mildly Bullish"
        elif deviation >= strong_buffer: classification = "Strongly Bearish"
        elif deviation > buffer: classification = "Mildly Bearish"
    else: # Normal type: Higher is better/bullish
        if deviation >= strong_buffer: classification = "Strongly Bullish"
        elif deviation > buffer: classification = "Mildly Bullish"
        elif deviation <= -strong_buffer: classification = "Strongly Bearish"
        elif deviation < -buffer: classification = "Mildly Bearish"
        
    # --- Generate Explanation ---
    prev_text = f", Previous: {float(previous_value):.2f}{unit}" if previous_value is not None and not np.isnan(previous_value) else ""
    explanation = (f"Event: '{event_name}' ({currency}, type: {props['type']}).\n"
                   f"Actual: {actual:.2f}{unit}, Forecast: {forecast:.2f}{unit}{prev_text}.\n"
                   f"Deviation (Actual - Fcst): {deviation:.2f}{unit}. Buffer: ±{buffer:.2f}{unit} (Strong: ±{strong_buffer:.2f}{unit}).\n"
                   f"Outcome: **{classification}** for {currency}. ")

    # Add reasoning based on classification
    if "Strongly Bullish" in classification: explanation += f"Actual significantly {'below' if is_inverted else 'above'} forecast (beyond {NUANCED_MULTIPLIER}x buffer)."
    elif "Mildly Bullish" in classification: explanation += f"Actual moderately {'below' if is_inverted else 'above'} forecast (beyond 1x buffer)."
    elif "Strongly Bearish" in classification: explanation += f"Actual significantly {'above' if is_inverted else 'below'} forecast (beyond {NUANCED_MULTIPLIER}x buffer)."
    elif "Mildly Bearish" in classification: explanation += f"Actual moderately {'above' if is_inverted else 'below'} forecast (beyond 1x buffer)."
    else: explanation += "Actual is within the expected buffer range around the forecast."

    return classification, explanation

if __name__ == '__main__':
    # ... (Existing test cases for infer and predict can be run) ...

    print("\n--- Test Cases for Nuanced classify_actual_release ---")
    # NFP (Normal: Higher is better, abs_buffer=20.0, strong_buffer=40.0)
    cl, ex = classify_actual_release(250.0, 200.0, 175.0, "Non-Farm Employment Change", "USD"); print(f"NFP Strongly Bullish: {cl}\n{ex}\n") # Dev > +40
    cl, ex = classify_actual_release(230.0, 200.0, 175.0, "Non-Farm Employment Change", "USD"); print(f"NFP Mildly Bullish: {cl}\n{ex}\n")  # +20 < Dev <= +40
    cl, ex = classify_actual_release(210.0, 200.0, 175.0, "Non-Farm Employment Change", "USD"); print(f"NFP Neutral: {cl}\n{ex}\n")          # -20 <= Dev <= +20
    cl, ex = classify_actual_release(170.0, 200.0, 175.0, "Non-Farm Employment Change", "USD"); print(f"NFP Mildly Bearish: {cl}\n{ex}\n") # -40 <= Dev < -20
    cl, ex = classify_actual_release(150.0, 200.0, 175.0, "Non-Farm Employment Change", "USD"); print(f"NFP Strongly Bearish: {cl}\n{ex}\n")# Dev < -40

    # Unemployment Rate (Inverted: Lower is better, abs_buffer=0.1, strong_buffer=0.2)
    cl, ex = classify_actual_release(3.6, 3.9, 3.9, "Unemployment Rate", "USD"); print(f"Unemp Strongly Bullish: {cl}\n{ex}\n") # Dev <= -0.2
    cl, ex = classify_actual_release(3.75, 3.9, 3.9, "Unemployment Rate", "USD"); print(f"Unemp Mildly Bullish: {cl}\n{ex}\n") # -0.2 < Dev <= -0.1
    cl, ex = classify_actual_release(3.85, 3.9, 3.9, "Unemployment Rate", "USD"); print(f"Unemp Neutral: {cl}\n{ex}\n")       # -0.1 < Dev < +0.1
    cl, ex = classify_actual_release(4.05, 3.9, 3.9, "Unemployment Rate", "USD"); print(f"Unemp Mildly Bearish: {cl}\n{ex}\n") # +0.1 <= Dev < +0.2
    cl, ex = classify_actual_release(4.2, 3.9, 3.9, "Unemployment Rate", "USD"); print(f"Unemp Strongly Bearish: {cl}\n{ex}\n") # Dev >= +0.2

# strategy_engine.py
import numpy as np
import streamlit as st # Used only for potential error logging if needed

# --- Indicator Configuration ---
# Expanded with more absolute thresholds where appropriate
NUANCED_MULTIPLIER = 2.0 

INDICATOR_CONFIG = {
    "Non-Farm Employment Change": { # USA NFP
        "type": "normal", "unit": "K",
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05, 
        "abs_significance": 30.0, "abs_buffer": 20.0 
    },
    "Employment Change": { # Assuming CAD Employment Change
        "type": "normal", "unit": "K",
        "significance_threshold_pct": 0.15, "buffer_pct": 0.10, 
        "abs_significance": 7.0, "abs_buffer": 5.0 
    },
    "Unemployment Rate": { # Most countries
        "type": "inverted", "unit": "%",
        "significance_threshold_pct": 0.03, "buffer_pct": 0.02, 
        "abs_significance": 0.1, "abs_buffer": 0.1 
    },
    "GDP m/m": { # Monthly GDP Growth
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.20, "buffer_pct": 0.10, 
        "abs_significance": 0.1, "abs_buffer": 0.1 # Small % changes matter
    },
     "GDP q/q": { # Quarterly GDP Growth
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.15, "buffer_pct": 0.10, 
        "abs_significance": 0.2, "abs_buffer": 0.2 
    },
    "Core CPI m/m": { # Monthly Core Inflation
        "type": "normal", "unit": "%", # Treat higher as bullish (anticipating hikes)
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05, 
        "abs_significance": 0.1, "abs_buffer": 0.1 
    },
     "CPI m/m": { # Monthly Headline Inflation
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05, 
        "abs_significance": 0.1, "abs_buffer": 0.1 
    },
    "Policy Rate": { # Generic for Interest Rate decisions (BoJ, Fed, ECB etc.)
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.05, "buffer_pct": 0.02, 
        "abs_significance": 0.15, "abs_buffer": 0.10 # Rate decisions often move in 0.25 steps, so buffer is smaller
    },
    "Retail Sales m/m": { # Monthly Retail Sales
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.15, "buffer_pct": 0.10, 
        "abs_significance": 0.3, "abs_buffer": 0.2 # Retail sales can be volatile
    },
    "PMI": { # Generic for PMI (Manufacturing, Services) - Threshold is often 50
        "type": "normal", "unit": "", # PMI is an index level
        "significance_threshold_pct": 0.02, "buffer_pct": 0.01, # ~1 point deviation is notable
        "abs_significance": 1.0, "abs_buffer": 0.8 # Absolute buffer around forecast
        # Special logic might be needed to compare against 50 boundary
    },
    "ECB President Speaks": {"type": "qualitative", "unit": ""}, 
    "FOMC Press Conference": {"type": "qualitative", "unit": ""},
    "Default": { # Fallback uses percentages
        "type": "normal", "unit": "",
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05, 
        "default_significance": 0.1, "default_buffer": 0.1 
    } 
}

def get_indicator_properties(event_name):
    """Fetches properties, prioritizing more specific matches."""
    best_match_key = "Default"
    max_match_len = 0
    # Find the most specific matching key (longest common substring)
    event_name_lower = event_name.lower()
    for key in INDICATOR_CONFIG:
        key_lower = key.lower()
        if key_lower in event_name_lower:
            # Prioritize longer keys if multiple match (e.g., "Core CPI m/m" vs "CPI m/m")
            if len(key_lower) > max_match_len:
                max_match_len = len(key_lower)
                best_match_key = key
        # Allow partial match for generic terms like "Policy Rate" or "PMI"
        elif key_lower == "policy rate" and "rate" in event_name_lower and "unemployment" not in event_name_lower:
             if len(key_lower) > max_match_len:
                max_match_len = len(key_lower)
                best_match_key = key
        elif key_lower == "pmi" and "pmi" in event_name_lower:
             if len(key_lower) > max_match_len:
                max_match_len = len(key_lower)
                best_match_key = key


    return INDICATOR_CONFIG[best_match_key].copy()

def _calculate_threshold(value, pct_threshold, abs_threshold, default_abs_threshold):
    """Helper to calculate threshold, prioritizing absolute value."""
    # Use absolute threshold if it's provided and not None
    if abs_threshold is not None:
        return abs_threshold # Use absolute value directly
    # Else, if value is non-zero and percentage threshold is provided
    elif value != 0 and pct_threshold is not None:
        return abs(value * pct_threshold)
    # Fallback to default absolute threshold
    else:
        return default_abs_threshold

# --- infer_market_outlook_from_data ---
def infer_market_outlook_from_data(previous, forecast, event_name):
    # ... (Function remains the same as V13, uses _calculate_threshold) ...
    props = get_indicator_properties(event_name)
    if props["type"] == "qualitative": return "Consolidating (Qualitative)"
    if forecast is None or np.isnan(forecast) or previous is None or np.isnan(previous): return "Consolidating (Insufficient Data)"
    try: prev_val, fcst_val = float(previous), float(forecast)
    except ValueError: return "Consolidating (Invalid Data)"
    significance_threshold = _calculate_threshold(prev_val, props.get("significance_threshold_pct"), props.get("abs_significance"), props.get("default_significance", INDICATOR_CONFIG["Default"]["default_significance"]))
    deviation = fcst_val - prev_val
    if abs(deviation) < significance_threshold: return "Consolidating"
    is_bullish_deviation = deviation > 0
    if props["type"] == "inverted": return "Bullish" if not is_bullish_deviation else "Bearish"
    else: return "Bullish" if is_bullish_deviation else "Bearish"

# --- predict_actual_condition_for_outcome ---
def predict_actual_condition_for_outcome(previous, forecast, desired_outcome, currency, event_name):
    # ... (Function remains the same as V13, uses _calculate_threshold) ...
    props = get_indicator_properties(event_name)
    unit = props.get("unit", "")
    if props["type"] == "qualitative":
        if desired_outcome == "Bullish": return (f"For a **Bullish** outcome for {currency} from '{event_name}', the speech/announcement would need hawkish rhetoric...")
        elif desired_outcome == "Bearish": return (f"For a **Bearish** outcome for {currency} from '{event_name}', the speech/announcement would need dovish rhetoric...")
        else: return (f"For a **Consolidating/Neutral** outcome for {currency} from '{event_name}', the speech/announcement would need to be in line with expectations...")
    if forecast is None or np.isnan(forecast): return (f"Cannot provide quantitative interpretation for '{event_name}' ({currency}): 'Forecast' data unavailable.")
    try: forecast_val = float(forecast)
    except ValueError: return (f"Forecast value '{forecast}' for '{event_name}' ({currency}) is invalid.")
    buffer = _calculate_threshold(forecast_val, props.get("buffer_pct"), props.get("abs_buffer"), props.get("default_buffer", INDICATOR_CONFIG["Default"]["default_buffer"]))
    outcome_description = ""; indicator_nature_desc = "lower is better" if props['type'] == 'inverted' else "higher is better"; strong_buffer = buffer * NUANCED_MULTIPLIER
    if desired_outcome == "Bullish":
        if props["type"] == "inverted": mild_beat, strong_beat = forecast_val - buffer, forecast_val - strong_buffer; outcome_description = (f"For a **Bullish** outcome for {currency} from '{event_name}' ({indicator_nature_desc}), Actual < {forecast_val:.2f}{unit}. (e.g., Mildly Bullish if ~{mild_beat:.2f}{unit}, Strongly Bullish if ≤ {strong_beat:.2f}{unit})")
        else: mild_beat, strong_beat = forecast_val + buffer, forecast_val + strong_buffer; outcome_description = (f"For a **Bullish** outcome for {currency} from '{event_name}' ({indicator_nature_desc}), Actual > {forecast_val:.2f}{unit}. (e.g., Mildly Bullish if ~{mild_beat:.2f}{unit}, Strongly Bullish if ≥ {strong_beat:.2f}{unit})")
    elif desired_outcome == "Bearish":
        if props["type"] == "inverted": mild_miss, strong_miss = forecast_val + buffer, forecast_val + strong_buffer; outcome_description = (f"For a **Bearish** outcome for {currency} from '{event_name}' ({indicator_nature_desc}), Actual > {forecast_val:.2f}{unit}. (e.g., Mildly Bearish if ~{mild_miss:.2f}{unit}, Strongly Bearish if ≥ {strong_miss:.2f}{unit})")
        else: mild_miss, strong_miss = forecast_val - buffer, forecast_val - strong_buffer; outcome_description = (f"For a **Bearish** outcome for {currency} from '{event_name}' ({indicator_nature_desc}), Actual < {forecast_val:.2f}{unit}. (e.g., Mildly Bearish if ~{mild_miss:.2f}{unit}, Strongly Bearish if ≤ {strong_miss:.2f}{unit})")
    elif desired_outcome == "Consolidating": lower_bound, upper_bound = forecast_val - buffer, forecast_val + buffer; outcome_description = (f"For a **Consolidating/Neutral** outcome for {currency} from '{event_name}', Actual ≈ {forecast_val:.2f}{unit} (e.g., between {lower_bound:.2f}{unit} and {upper_bound:.2f}{unit}).")
    else: return "Invalid desired outcome selected."
    if previous is not None and not np.isnan(previous) and outcome_description:
        prev_val = float(previous)
        if desired_outcome == "Bullish": forecast_improves = (props["type"] == "inverted" and forecast_val < prev_val) or (props["type"] == "normal" and forecast_val > prev_val); outcome_description += (f" Context: Forecast ({forecast_val:.2f}{unit}) already suggests {'improvement' if forecast_improves else 'worsening'} from Previous ({prev_val:.2f}{unit}).")
        elif desired_outcome == "Bearish": forecast_worsens = (props["type"] == "inverted" and forecast_val > prev_val) or (props["type"] == "normal" and forecast_val < prev_val); outcome_description += (f" Context: Forecast ({forecast_val:.2f}{unit}) already suggests {'worsening' if forecast_worsens else 'improvement'} from Previous ({prev_val:.2f}{unit}).")
        elif desired_outcome == "Consolidating":
             if abs(prev_val - forecast_val) < (buffer * 0.1): outcome_description += (f" Context: Forecast ({forecast_val:.2f}{unit}) is very close to Previous ({prev_val:.2f}{unit}).")
    return outcome_description

# --- classify_actual_release ---
def classify_actual_release(actual_value, forecast_value, previous_value, event_name, currency):
    # ... (Function remains the same as V13, uses _calculate_threshold for buffer) ...
    props = get_indicator_properties(event_name); unit = props.get("unit", "")
    if props["type"] == "qualitative": return "Qualitative", f"'{event_name}' is qualitative..."
    if actual_value is None or np.isnan(actual_value): return "Indeterminate", f"Actual value missing..."
    if forecast_value is None or np.isnan(forecast_value): return "Indeterminate", f"Forecast value missing..."
    try: actual, forecast = float(actual_value), float(forecast_value)
    except ValueError: return "Error", f"Invalid numeric input..."
    buffer = _calculate_threshold(forecast, props.get("buffer_pct"), props.get("abs_buffer"), props.get("default_buffer", INDICATOR_CONFIG["Default"]["default_buffer"]))
    strong_buffer = buffer * NUANCED_MULTIPLIER; deviation = actual - forecast
    classification = "Neutral/In-Line"; is_inverted = props["type"] == "inverted"
    if is_inverted:
        if deviation <= -strong_buffer: classification = "Strongly Bullish"
        elif deviation < -buffer: classification = "Mildly Bullish"
        elif deviation >= strong_buffer: classification = "Strongly Bearish"
        elif deviation > buffer: classification = "Mildly Bearish"
    else:
        if deviation >= strong_buffer: classification = "Strongly Bullish"
        elif deviation > buffer: classification = "Mildly Bullish"
        elif deviation <= -strong_buffer: classification = "Strongly Bearish"
        elif deviation < -buffer: classification = "Mildly Bearish"
    prev_text = f", Previous: {float(previous_value):.2f}{unit}" if previous_value is not None and not np.isnan(previous_value) else ""
    explanation = (f"Event: '{event_name}' ({currency}, type: {props['type']}).\n"
                   f"Actual: {actual:.2f}{unit}, Forecast: {forecast:.2f}{unit}{prev_text}.\n"
                   f"Deviation (Actual - Fcst): {deviation:.2f}{unit}. Buffer: ±{buffer:.2f}{unit} (Strong: ±{strong_buffer:.2f}{unit}).\n"
                   f"Outcome: **{classification}** for {currency}. ")
    if "Strongly Bullish" in classification: explanation += f"Actual significantly {'below' if is_inverted else 'above'} forecast (beyond {NUANCED_MULTIPLIER}x buffer)."
    elif "Mildly Bullish" in classification: explanation += f"Actual moderately {'below' if is_inverted else 'above'} forecast (beyond 1x buffer)."
    elif "Strongly Bearish" in classification: explanation += f"Actual significantly {'above' if is_inverted else 'below'} forecast (beyond {NUANCED_MULTIPLIER}x buffer)."
    elif "Mildly Bearish" in classification: explanation += f"Actual moderately {'above' if is_inverted else 'below'} forecast (beyond 1x buffer)."
    else: explanation += "Actual is within the expected buffer range around the forecast."
    return classification, explanation

# --- (Optional: Add new test cases for added absolute thresholds) ---

# strategy_engine.py
import numpy as np
import streamlit as st # For potential logging or error display if needed directly

# --- Indicator Configuration ---
INDICATOR_CONFIG = {
    "Non-Farm Employment Change": {"type": "normal", "significance_threshold_pct": 0.10, "buffer_pct": 0.05, "default_significance": 20.0, "default_buffer": 10.0, "unit": "K"},
    "Employment Change": {"type": "normal", "significance_threshold_pct": 0.15, "buffer_pct": 0.10, "default_significance": 5.0, "default_buffer": 2.0, "unit": "K"},
    "Unemployment Rate": {"type": "inverted", "significance_threshold_pct": 0.03, "buffer_pct": 0.02, "default_significance": 0.1, "default_buffer": 0.1, "unit": "%"},
    "GDP m/m": {"type": "normal", "significance_threshold_pct": 0.20, "buffer_pct": 0.10, "default_significance": 0.1, "default_buffer": 0.05, "unit": "%"},
    "Core CPI m/m": {"type": "normal", "significance_threshold_pct": 0.10, "buffer_pct": 0.05, "default_significance": 0.1, "default_buffer": 0.05, "unit": "%"},
    "BoJ Policy Rate": {"type": "normal", "significance_threshold_pct": 0.05, "buffer_pct": 0.02, "default_significance": 0.05, "default_buffer": 0.01, "unit": "%"},
    "Retail Sales m/m": {"type": "normal", "significance_threshold_pct": 0.15, "buffer_pct": 0.10, "default_significance": 0.2, "default_buffer": 0.1, "unit": "%"},
    "ECB President Speaks": {"type": "qualitative", "significance_threshold_pct": 0, "buffer_pct": 0, "unit": ""},
    "Default": {"type": "normal", "significance_threshold_pct": 0.10, "buffer_pct": 0.05, "default_significance": 0.1, "default_buffer": 0.1, "unit": ""} # Fallback
}

def get_indicator_properties(event_name):
    """Fetches properties for a given event name, with a fallback to default."""
    for key in INDICATOR_CONFIG:
        if key.lower() in event_name.lower():
            return INDICATOR_CONFIG[key]
    return INDICATOR_CONFIG["Default"]

def infer_market_outlook_from_data(previous, forecast, event_name):
    """
    Infers a likely market outlook (Bullish, Bearish, Consolidating) for the currency
    based on 'Previous', 'Forecast', and indicator-specific properties.
    """
    props = get_indicator_properties(event_name)

    if props["type"] == "qualitative":
        return "Consolidating (Qualitative - dependent on speech content/tone)"

    if forecast is None or np.isnan(forecast) or previous is None or np.isnan(previous):
        return "Consolidating (Insufficient Data)"

    try:
        prev_val = float(previous)
        fcst_val = float(forecast)
    except ValueError:
        return "Consolidating (Invalid Data)"

    if prev_val != 0:
        significance_threshold = abs(prev_val * props["significance_threshold_pct"])
    else:
        significance_threshold = props.get("default_significance", INDICATOR_CONFIG["Default"]["default_significance"])

    deviation = fcst_val - prev_val

    if abs(deviation) < significance_threshold:
        return "Consolidating"
    
    is_bullish_deviation = deviation > 0
    
    if props["type"] == "inverted":
        return "Bullish" if not is_bullish_deviation else "Bearish"
    else: 
        return "Bullish" if is_bullish_deviation else "Bearish"


def predict_actual_condition_for_outcome(previous, forecast, desired_outcome, currency, event_name):
    """
    Describes the condition the 'Actual' economic data would likely need to meet
    for a desired market outcome, using indicator-specific logic.
    """
    props = get_indicator_properties(event_name)
    unit = props.get("unit", "")

    if props["type"] == "qualitative":
        # ... (qualitative logic from previous version, unchanged)
        if desired_outcome == "Bullish":
            return (f"For a **Bullish** outcome for {currency} from '{event_name}', "
                    f"the speech/announcement would need to contain hawkish rhetoric, positive economic assessments, "
                    f"or hints of tighter monetary policy.")
        elif desired_outcome == "Bearish":
            return (f"For a **Bearish** outcome for {currency} from '{event_name}', "
                    f"the speech/announcement would need to contain dovish rhetoric, negative economic assessments, "
                    f"or hints of looser monetary policy.")
        else: # Consolidating
            return (f"For a **Consolidating/Neutral** outcome for {currency} from '{event_name}', "
                    f"the speech/announcement would need to be in line with current market expectations, offering no new surprises.")


    if forecast is None or np.isnan(forecast):
        return (f"Cannot provide a quantitative interpretation for '{event_name}' ({currency}) "
                f"as 'Forecast' data is unavailable.")

    try:
        forecast_val = float(forecast)
    except ValueError:
        return (f"Forecast value '{forecast}' for '{event_name}' ({currency}) is not a valid number.")

    if forecast_val != 0:
        buffer = abs(forecast_val * props["buffer_pct"])
    else:
        buffer = props.get("default_buffer", INDICATOR_CONFIG["Default"]["default_buffer"])
    
    outcome_description = ""
    indicator_nature_desc = "lower is better" if props['type'] == 'inverted' else "higher is better"
    # ... (rest of the logic from previous version, ensuring 'unit' is used in f-strings)

    if desired_outcome == "Bullish":
        if props["type"] == "inverted":
            significant_beat_value = forecast_val - buffer
            outcome_description = (f"For a **Bullish** outcome for {currency} from '{event_name}' ({indicator_nature_desc}), "
                                   f"Actual < {forecast_val:.2f}{unit} (e.g., around or below {significant_beat_value:.2f}{unit}).")
        else: 
            significant_beat_value = forecast_val + buffer
            outcome_description = (f"For a **Bullish** outcome for {currency} from '{event_name}' ({indicator_nature_desc}), "
                                   f"Actual > {forecast_val:.2f}{unit} (e.g., around or above {significant_beat_value:.2f}{unit}).")
        # ... (contextual info based on previous, unchanged)

    elif desired_outcome == "Bearish":
        if props["type"] == "inverted":
            significant_miss_value = forecast_val + buffer
            outcome_description = (f"For a **Bearish** outcome for {currency} from '{event_name}' ({indicator_nature_desc}), "
                                   f"Actual > {forecast_val:.2f}{unit} (e.g., around or above {significant_miss_value:.2f}{unit}).")
        else: 
            significant_miss_value = forecast_val - buffer
            outcome_description = (f"For a **Bearish** outcome for {currency} from '{event_name}' ({indicator_nature_desc}), "
                                   f"Actual < {forecast_val:.2f}{unit} (e.g., around or below {significant_miss_value:.2f}{unit}).")
        # ... (contextual info based on previous, unchanged)

    elif desired_outcome == "Consolidating":
        lower_bound = forecast_val - buffer
        upper_bound = forecast_val + buffer
        outcome_description = (f"For a **Consolidating/Neutral** outcome for {currency} from '{event_name}', "
                               f"Actual ≈ {forecast_val:.2f}{unit} (e.g., between {lower_bound:.2f}{unit} and {upper_bound:.2f}{unit}).")
        # ... (contextual info based on previous, unchanged)
    else:
        return "Invalid desired outcome selected."
    
    # Add context based on previous if available
    if previous is not None and not np.isnan(previous) and outcome_description: # Check if outcome_description is not empty
        prev_val = float(previous)
        # This part can be complex, simplified here
        if desired_outcome == "Bullish":
            forecast_improves_on_previous = (props["type"] == "inverted" and forecast_val < prev_val) or \
                                           (props["type"] == "normal" and forecast_val > prev_val)
            if forecast_improves_on_previous:
                outcome_description += (f" Context: Forecast ({forecast_val:.2f}{unit}) already suggests improvement "
                                        f"from Previous ({prev_val:.2f}{unit}). A strong beat of forecast would be particularly bullish.")
            else:
                 outcome_description += (f" Context: Forecast ({forecast_val:.2f}{unit}) suggests a decline/worsening "
                                        f"from Previous ({prev_val:.2f}{unit}). An actual significantly better than forecast would be needed.")
        elif desired_outcome == "Bearish":
            forecast_worsens_from_previous = (props["type"] == "inverted" and forecast_val > prev_val) or \
                                             (props["type"] == "normal" and forecast_val < prev_val)
            if forecast_worsens_from_previous:
                outcome_description += (f" Context: Forecast ({forecast_val:.2f}{unit}) already suggests worsening "
                                        f"from Previous ({prev_val:.2f}{unit}). A significant miss of forecast would be particularly bearish.")
            else:
                outcome_description += (f" Context: Forecast ({forecast_val:.2f}{unit}) suggests an improvement "
                                         f"from Previous ({prev_val:.2f}{unit}). An actual significantly worse than forecast would be needed.")
        elif desired_outcome == "Consolidating":
             if abs(prev_val - forecast_val) < (buffer * 0.1): # If previous and forecast are very close
                 outcome_description += (f" Context: Forecast ({forecast_val:.2f}{unit}) is very close to Previous ({prev_val:.2f}{unit}). An in-line actual would reinforce consolidation.")


    return outcome_description


def classify_actual_release(actual_value, forecast_value, previous_value, event_name, currency):
    """
    Classifies an 'Actual' release as Bullish, Bearish, or Consolidating for the currency.

    Args:
        actual_value (float/None): The actual released value.
        forecast_value (float/None): The forecast value.
        previous_value (float/None): The previous value (for context).
        event_name (str): The name of the economic event.
        currency (str): The currency affected.

    Returns:
        tuple: (classification_str, detailed_explanation_str)
    """
    props = get_indicator_properties(event_name)
    unit = props.get("unit", "")

    if props["type"] == "qualitative":
        return "Qualitative", (f"'{event_name}' is a qualitative event. Market reaction for {currency} "
                               f"depends on the content and tone of the announcement/speech, not a numerical 'Actual'.")

    if actual_value is None or np.isnan(actual_value):
        return "Indeterminate", f"Actual value for '{event_name}' ({currency}) is missing. Cannot classify."
    
    if forecast_value is None or np.isnan(forecast_value):
        return "Indeterminate", (f"Forecast value for '{event_name}' ({currency}) is missing. "
                                 f"Classification against Actual ({actual_value:.2f}{unit}) is difficult without forecast context.")

    try:
        actual = float(actual_value)
        forecast = float(forecast_value)
    except ValueError:
        return "Error", f"Invalid numerical input for Actual or Forecast for '{event_name}' ({currency})."

    # Determine buffer based on forecast
    if forecast != 0:
        buffer = abs(forecast * props["buffer_pct"])
    else:
        buffer = props.get("default_buffer", INDICATOR_CONFIG["Default"]["default_buffer"])

    deviation_actual_vs_forecast = actual - forecast
    classification = "Consolidating"
    
    # Main classification logic
    if props["type"] == "inverted": # Lower is better
        if deviation_actual_vs_forecast < -buffer: # Actual is significantly lower than forecast
            classification = "Bullish"
        elif deviation_actual_vs_forecast > buffer: # Actual is significantly higher than forecast
            classification = "Bearish"
    else: # Normal type, higher is better
        if deviation_actual_vs_forecast > buffer: # Actual is significantly higher than forecast
            classification = "Bullish"
        elif deviation_actual_vs_forecast < -buffer: # Actual is significantly lower than forecast
            classification = "Bearish"

    # Detailed explanation
    explanation = (f"Event: '{event_name}' for {currency} (Indicator type: {props['type']}, where "
                   f"{'lower is better' if props['type'] == 'inverted' else 'higher is better'}).\n"
                   f"Actual: {actual:.2f}{unit}, Forecast: {forecast:.2f}{unit}")
    if previous_value is not None and not np.isnan(previous_value):
        explanation += f", Previous: {float(previous_value):.2f}{unit}.\n"
    else:
        explanation += ".\n"

    explanation += f"The Actual release ({actual:.2f}{unit}) compared to Forecast ({forecast:.2f}{unit}) suggests a **{classification}** outcome for {currency}. "
    
    if classification == "Bullish":
        explanation += (f"This is because the Actual is {'lower' if props['type'] == 'inverted' else 'better'} than forecast "
                        f"by a notable margin (deviation: {deviation_actual_vs_forecast:.2f}{unit}, buffer: ±{buffer:.2f}{unit}).")
    elif classification == "Bearish":
        explanation += (f"This is because the Actual is {'higher' if props['type'] == 'inverted' else 'worse'} than forecast "
                        f"by a notable margin (deviation: {deviation_actual_vs_forecast:.2f}{unit}, buffer: ±{buffer:.2f}{unit}).")
    else: # Consolidating
        explanation += (f"This is because the Actual is relatively in line with the forecast "
                        f"(deviation: {deviation_actual_vs_forecast:.2f}{unit}, within buffer of ±{buffer:.2f}{unit}).")

    return classification, explanation


if __name__ == '__main__':
    # ... (existing test cases for infer and predict) ...

    print("\n--- Test Cases for classify_actual_release ---")
    # NFP (Normal: Higher is better)
    class_nfp_bull, exp_nfp_bull = classify_actual_release(250.0, 200.0, 175.0, "Non-Farm Employment Change", "USD")
    print(f"NFP Bullish: {class_nfp_bull}\nExplanation: {exp_nfp_bull}\n")

    class_nfp_bear, exp_nfp_bear = classify_actual_release(150.0, 200.0, 175.0, "Non-Farm Employment Change", "USD")
    print(f"NFP Bearish: {class_nfp_bear}\nExplanation: {exp_nfp_bear}\n")

    class_nfp_cons, exp_nfp_cons = classify_actual_release(205.0, 200.0, 175.0, "Non-Farm Employment Change", "USD")
    print(f"NFP Consolidating: {class_nfp_cons}\nExplanation: {exp_nfp_cons}\n")

    # Unemployment Rate (Inverted: Lower is better)
    class_unemp_bull, exp_unemp_bull = classify_actual_release(3.6, 3.9, 3.9, "Unemployment Rate", "USD")
    print(f"Unemp Bullish: {class_unemp_bull}\nExplanation: {exp_unemp_bull}\n")

    class_unemp_bear, exp_unemp_bear = classify_actual_release(4.1, 3.9, 3.9, "Unemployment Rate", "USD")
    print(f"Unemp Bearish: {class_unemp_bear}\nExplanation: {exp_unemp_bear}\n")

    class_unemp_cons, exp_unemp_cons = classify_actual_release(3.88, 3.9, 3.9, "Unemployment Rate", "USD") # Using 3.88 to be within default buffer of 0.02 for 3.9
    print(f"Unemp Consolidating: {class_unemp_cons}\nExplanation: {exp_unemp_cons}\n")

    # Qualitative
    class_qual, exp_qual = classify_actual_release(None, None, None, "ECB President Speaks", "EUR")
    print(f"Qualitative: {class_qual}\nExplanation: {exp_qual}\n")

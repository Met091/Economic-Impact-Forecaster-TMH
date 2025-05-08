# strategy_engine.py
import numpy as np
import streamlit as st # For potential logging or error display if needed directly

# --- Indicator Configuration ---
# This dictionary defines properties for known economic indicators.
# 'type': 'normal' (higher is better/bullish), 'inverted' (lower is better/bullish)
# 'significance_threshold_pct': Percentage change from 'Previous' to 'Forecast' considered significant for infer_market_outlook.
# 'buffer_pct': Percentage of 'Forecast' used as a buffer for predict_actual_condition.
# 'default_significance': Absolute value used if previous is 0 or for qualitative assessment.
# 'default_buffer': Absolute value used if forecast is 0.

INDICATOR_CONFIG = {
    "Non-Farm Employment Change": {"type": "normal", "significance_threshold_pct": 0.10, "buffer_pct": 0.05, "default_significance": 20.0, "default_buffer": 10.0},
    "Employment Change": {"type": "normal", "significance_threshold_pct": 0.15, "buffer_pct": 0.10, "default_significance": 5.0, "default_buffer": 2.0},
    "Unemployment Rate": {"type": "inverted", "significance_threshold_pct": 0.03, "buffer_pct": 0.02, "default_significance": 0.1, "default_buffer": 0.1}, # Smaller changes are significant
    "GDP m/m": {"type": "normal", "significance_threshold_pct": 0.20, "buffer_pct": 0.10, "default_significance": 0.1, "default_buffer": 0.05},
    "Core CPI m/m": {"type": "normal", "significance_threshold_pct": 0.10, "buffer_pct": 0.05, "default_significance": 0.1, "default_buffer": 0.05}, # Can be inverted if fighting inflation
    "BoJ Policy Rate": {"type": "normal", "significance_threshold_pct": 0.05, "buffer_pct": 0.02, "default_significance": 0.05, "default_buffer": 0.01}, # Interest rates
    "Retail Sales m/m": {"type": "normal", "significance_threshold_pct": 0.15, "buffer_pct": 0.10, "default_significance": 0.2, "default_buffer": 0.1},
    "ECB President Speaks": {"type": "qualitative", "significance_threshold_pct": 0, "buffer_pct": 0}, # No numerical prediction
    "Default": {"type": "normal", "significance_threshold_pct": 0.10, "buffer_pct": 0.05, "default_significance": 0.1, "default_buffer": 0.1} # Fallback
}
# Note: For CPI, 'normal' implies higher inflation is bullish for the currency (anticipating rate hikes).
# If the central bank's goal is to *lower* inflation, then CPI could be treated as 'inverted' from that policy perspective.
# This configuration keeps it simple: higher CPI reading -> currency strength expectation.

def get_indicator_properties(event_name):
    """Fetches properties for a given event name, with a fallback to default."""
    # Attempt to match event_name exactly or partially
    for key in INDICATOR_CONFIG:
        if key.lower() in event_name.lower():
            return INDICATOR_CONFIG[key]
    return INDICATOR_CONFIG["Default"]

def infer_market_outlook_from_data(previous, forecast, event_name):
    """
    Infers a likely market outlook (Bullish, Bearish, Consolidating) for the currency
    based on 'Previous', 'Forecast', and indicator-specific properties.

    Args:
        previous (float/None): The previous value.
        forecast (float/None): The forecast value.
        event_name (str): The name of the economic event.

    Returns:
        str: "Bullish", "Bearish", or "Consolidating".
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

    # Determine significance threshold
    if prev_val != 0:
        significance_threshold = abs(prev_val * props["significance_threshold_pct"])
    else:
        significance_threshold = props.get("default_significance", INDICATOR_CONFIG["Default"]["default_significance"])


    deviation = fcst_val - prev_val

    if abs(deviation) < significance_threshold:
        return "Consolidating"
    
    is_bullish_deviation = deviation > 0
    
    if props["type"] == "inverted":
        # For inverted indicators, a negative deviation (forecast < previous) is bullish
        return "Bullish" if not is_bullish_deviation else "Bearish"
    else: # For normal indicators
        return "Bullish" if is_bullish_deviation else "Bearish"


def predict_actual_condition_for_outcome(previous, forecast, desired_outcome, currency, event_name):
    """
    Describes the condition the 'Actual' economic data would likely need to meet
    for a desired market outcome, using indicator-specific logic.

    Args:
        previous (float/None): The previous value.
        forecast (float/None): The forecast value.
        desired_outcome (str): "Bullish", "Bearish", or "Consolidating".
        currency (str): The currency affected.
        event_name (str): The name of the economic event.

    Returns:
        str: A descriptive string.
    """
    props = get_indicator_properties(event_name)

    if props["type"] == "qualitative":
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
                f"as 'Forecast' data is unavailable. Market reaction will depend on the surprise element if an 'Actual' is released.")

    try:
        forecast_val = float(forecast)
    except ValueError:
        return (f"Forecast value '{forecast}' for '{event_name}' ({currency}) is not a valid number. "
                f"Quantitative interpretation is not possible.")

    # Determine buffer
    if forecast_val != 0:
        buffer = abs(forecast_val * props["buffer_pct"])
    else:
        buffer = props.get("default_buffer", INDICATOR_CONFIG["Default"]["default_buffer"])
    
    outcome_description = ""
    indicator_nature_desc = "lower is better" if props['type'] == 'inverted' else "higher is better"

    if desired_outcome == "Bullish":
        if props["type"] == "inverted":
            significant_beat_value = forecast_val - buffer
            outcome_description = (f"For a **Bullish** outcome for {currency} from '{event_name}' (an indicator where {indicator_nature_desc}), "
                                   f"the Actual release would typically need to be **notably lower than forecast**. "
                                   f"Ideally, Actual < {forecast_val:.2f} (e.g., around or below {significant_beat_value:.2f}).")
        else: # Normal type
            significant_beat_value = forecast_val + buffer
            outcome_description = (f"For a **Bullish** outcome for {currency} from '{event_name}' (an indicator where {indicator_nature_desc}), "
                                   f"the Actual release would typically need to be **notably better than forecast**. "
                                   f"Ideally, Actual > {forecast_val:.2f} (e.g., around or above {significant_beat_value:.2f}).")
        
        if previous is not None and not np.isnan(previous):
            prev_val = float(previous)
            forecast_improves_on_previous = (props["type"] == "inverted" and forecast_val < prev_val) or \
                                           (props["type"] == "normal" and forecast_val > prev_val)
            if forecast_improves_on_previous:
                outcome_description += (f" Given the forecast ({forecast_val:.2f}) already suggests improvement "
                                        f"from previous ({prev_val:.2f}), a strong beat of the forecast would be particularly bullish.")
            else:
                 outcome_description += (f" Given the forecast ({forecast_val:.2f}) suggests a decline/worsening "
                                        f"from previous ({prev_val:.2f}), an actual significantly better than forecast would be needed to turn sentiment bullish.")

    elif desired_outcome == "Bearish":
        if props["type"] == "inverted":
            significant_miss_value = forecast_val + buffer
            outcome_description = (f"For a **Bearish** outcome for {currency} from '{event_name}' (an indicator where {indicator_nature_desc}), "
                                   f"the Actual release would typically need to be **notably higher than forecast**. "
                                   f"Ideally, Actual > {forecast_val:.2f} (e.g., around or above {significant_miss_value:.2f}).")
        else: # Normal type
            significant_miss_value = forecast_val - buffer
            outcome_description = (f"For a **Bearish** outcome for {currency} from '{event_name}' (an indicator where {indicator_nature_desc}), "
                                   f"the Actual release would typically need to be **notably worse than forecast**. "
                                   f"Ideally, Actual < {forecast_val:.2f} (e.g., around or below {significant_miss_value:.2f}).")

        if previous is not None and not np.isnan(previous):
            prev_val = float(previous)
            forecast_worsens_from_previous = (props["type"] == "inverted" and forecast_val > prev_val) or \
                                             (props["type"] == "normal" and forecast_val < prev_val)
            if forecast_worsens_from_previous:
                outcome_description += (f" Given the forecast ({forecast_val:.2f}) already suggests a worsening "
                                        f"from previous ({prev_val:.2f}), a significant miss of the forecast would be particularly bearish.")
            else:
                outcome_description += (f" Given the forecast ({forecast_val:.2f}) suggests an improvement "
                                         f"from previous ({prev_val:.2f}), an actual significantly worse than forecast would be needed to turn sentiment bearish.")

    elif desired_outcome == "Consolidating":
        lower_bound = forecast_val - buffer
        upper_bound = forecast_val + buffer
        outcome_description = (f"For a **Consolidating/Neutral** outcome for {currency} from '{event_name}', "
                               f"the Actual release would typically need to be **in line with the forecast**. "
                               f"Ideally, Actual ≈ {forecast_val:.2f} (e.g., between {lower_bound:.2f} and {upper_bound:.2f}).")
        if previous is not None and not np.isnan(previous):
            prev_val = float(previous)
            if abs(prev_val - forecast_val) < (buffer * 0.1): # If previous and forecast are very close
                 outcome_description += (f" Since the forecast ({forecast_val:.2f}) is very close to the previous value ({prev_val:.2f}), an in-line actual would reinforce consolidation.")
    else:
        return "Invalid desired outcome selected. Please choose Bullish, Bearish, or Consolidating."

    return outcome_description

if __name__ == '__main__':
    # Test cases for infer_market_outlook_from_data
    print("--- Test Cases for infer_market_outlook_from_data ---")
    print(f"NFP (Prev: 175, Fcst: 200, Event: 'Non-Farm Employment Change'): Expected Bullish -> {infer_market_outlook_from_data(175.0, 200.0, 'Non-Farm Employment Change')}")
    print(f"Unemployment (Prev: 3.9, Fcst: 3.7, Event: 'Unemployment Rate'): Expected Bullish -> {infer_market_outlook_from_data(3.9, 3.7, 'Unemployment Rate')}")
    print(f"Unemployment (Prev: 3.7, Fcst: 3.9, Event: 'Unemployment Rate'): Expected Bearish -> {infer_market_outlook_from_data(3.7, 3.9, 'Unemployment Rate')}")
    print(f"GDP (Prev: 0.2, Fcst: 0.1, Event: 'GDP m/m'): Expected Bearish -> {infer_market_outlook_from_data(0.2, 0.1, 'GDP m/m')}")
    print(f"ECB Speech (Prev: NaN, Fcst: NaN, Event: 'ECB President Speaks'): Expected Consolidating (Qualitative) -> {infer_market_outlook_from_data(np.nan, np.nan, 'ECB President Speaks')}")
    print(f"Retail Sales (Prev: 0.5, Fcst: 0.52, Event: 'Retail Sales m/m'): Expected Consolidating (small change) -> {infer_market_outlook_from_data(0.5, 0.52, 'Retail Sales m/m')}")


    # Test cases for predict_actual_condition_for_outcome
    print("\n--- Test Cases for predict_actual_condition_for_outcome ---")
    print("\nBullish NFP (Prev: 175, Fcst: 200, Event: 'Non-Farm Employment Change'):")
    print(predict_actual_condition_for_outcome(175.0, 200.0, "Bullish", "USD", "Non-Farm Employment Change"))
    
    print("\nBearish GDP (Prev: 0.1, Fcst: 0.2, Event: 'GDP m/m'):") # Forecast is better, but we want bearish
    print(predict_actual_condition_for_outcome(0.1, 0.2, "Bearish", "GBP", "GDP m/m"))

    print("\nBullish Unemployment Rate (Prev: 3.9, Fcst: 3.9, Event: 'Unemployment Rate'):")
    print(predict_actual_condition_for_outcome(3.9, 3.9, "Bullish", "USD", "Unemployment Rate"))

    print("\nBearish Unemployment Rate (Prev: 3.9, Fcst: 3.7, Event: 'Unemployment Rate'):") # Fcst is already better, but we want bearish
    print(predict_actual_condition_for_outcome(3.9, 3.7, "Bearish", "USD", "Unemployment Rate"))

    print("\nBullish ECB Speech (Event: 'ECB President Speaks'):")
    print(predict_actual_condition_for_outcome(np.nan, np.nan, "Bullish", "EUR", "ECB President Speaks"))

    print("\nConsolidating CPI (Prev: 0.3, Fcst: 0.3, Event: 'Core CPI m/m'):")
    print(predict_actual_condition_for_outcome(0.3, 0.3, "Consolidating", "USD", "Core CPI m/m"))

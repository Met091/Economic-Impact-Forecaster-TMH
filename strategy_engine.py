# strategy_engine.py
import numpy as np

def infer_market_outlook_from_data(previous, forecast, event_name=""):
    """
    Infers a likely market outlook (Bullish, Bearish, Consolidating) for the currency
    based purely on the relationship between 'Previous' and 'Forecast' data.

    This is a simplified heuristic. Real market reactions are more complex and
    depend on the nature of the economic indicator (e.g., whether higher is better).

    Args:
        previous (float/None): The previous value of the economic indicator.
        forecast (float/None): The forecast value of the economic indicator.
        event_name (str, optional): The name of the economic event. Used to infer indicator type.

    Returns:
        str: "Bullish", "Bearish", or "Consolidating".
    """
    if forecast is None or np.isnan(forecast) or previous is None or np.isnan(previous):
        # If key data is missing, assume consolidation.
        return "Consolidating"

    try:
        prev_val = float(previous)
        fcst_val = float(forecast)
    except ValueError:
        # If values can't be converted to float, default to Consolidating
        return "Consolidating"

    # Determine if the indicator is inverted (lower is better)
    is_inverted_indicator = "unemployment rate" in event_name.lower() or \
                            "jobless claims" in event_name.lower()
                            # Add other inverted indicators here, e.g., "inflation rate" if target is to lower it

    # Define a threshold for "significant change"
    if prev_val != 0:
        significance_threshold = abs(prev_val * 0.05) # 5% change from previous is considered significant for this heuristic
    else:
        significance_threshold = 0.05 # If previous is 0, any small non-zero forecast is significant

    deviation = fcst_val - prev_val

    if abs(deviation) < significance_threshold:
        return "Consolidating"
    
    if is_inverted_indicator:
        # For inverted indicators, a forecast lower than previous is bullish
        if deviation < 0: # Forecast is significantly lower than Previous
            return "Bullish"
        else: # Forecast is significantly higher than Previous
            return "Bearish"
    else:
        # For standard indicators, a forecast higher than previous is bullish
        if deviation > 0: # Forecast is significantly higher than Previous
            return "Bullish"
        else: # Forecast is significantly lower than Previous
            return "Bearish"


def predict_actual_condition_for_outcome(previous, forecast, desired_outcome, currency, event_name):
    """
    Describes the condition the 'Actual' economic data would likely need to meet
    for a desired market outcome (Bullish/Bearish/Consolidating) for the specified currency.
    Considers if the indicator is 'inverted' (e.g., unemployment rate where lower is better).

    Args:
        previous (float/None): The previous value of the economic indicator.
        forecast (float/None): The forecast value of the economic indicator.
        desired_outcome (str): "Bullish", "Bearish", or "Consolidating".
        currency (str): The currency affected by the event.
        event_name (str): The name of the economic event.

    Returns:
        str: A descriptive string of the likely 'Actual' data condition.
    """
    if forecast is None or np.isnan(forecast):
        return (f"Cannot provide a quantitative interpretation for '{event_name}' "
                f"as 'Forecast' data is unavailable. Market reaction for {currency} "
                f"will depend on qualitative aspects or unexpected announcements.")

    try:
        forecast_val = float(forecast)
    except ValueError:
        return (f"Forecast value '{forecast}' for '{event_name}' is not a valid number. "
                f"Quantitative interpretation is not possible.")

    # Define a small relative or absolute buffer for "around forecast"
    if forecast_val != 0:
        buffer_percentage = 0.05 # 5% buffer relative to forecast
        buffer = abs(forecast_val * buffer_percentage)
    else:
        buffer = 0.05 # Small absolute buffer if forecast is 0

    # Determine if the indicator is inverted (lower is better)
    is_inverted_indicator = "unemployment rate" in event_name.lower() or \
                            "jobless claims" in event_name.lower()
                            # Add other inverted indicators here

    outcome_description = ""

    if desired_outcome == "Bullish":
        if is_inverted_indicator:
            # For inverted indicators, a bullish outcome means Actual < Forecast
            significant_beat_value = forecast_val - buffer
            outcome_description = (f"For a **Bullish** outcome for {currency} from '{event_name}' (an inverted indicator where lower is better), "
                                   f"the Actual release would typically need to be **notably lower than forecast**. "
                                   f"Ideally, Actual < {forecast_val:.2f} (e.g., around or below {significant_beat_value:.2f}).")
        else:
            # For standard indicators, a bullish outcome means Actual > Forecast
            significant_beat_value = forecast_val + buffer
            outcome_description = (f"For a **Bullish** outcome for {currency} from '{event_name}' (where higher is generally better), "
                                   f"the Actual release would typically need to be **notably better than forecast**. "
                                   f"Ideally, Actual > {forecast_val:.2f} (e.g., around or above {significant_beat_value:.2f}).")
        
        if previous is not None and not np.isnan(previous):
            prev_val = float(previous)
            # Context based on previous vs forecast
            if (is_inverted_indicator and forecast_val < prev_val) or \
               (not is_inverted_indicator and forecast_val > prev_val):
                outcome_description += (f" Given the forecast ({forecast_val:.2f}) already suggests improvement "
                                        f"from previous ({prev_val:.2f}), a strong beat (actual further improving on forecast) would be particularly bullish.")
            elif (is_inverted_indicator and forecast_val > prev_val) or \
                 (not is_inverted_indicator and forecast_val < prev_val):
                 outcome_description += (f" Given the forecast ({forecast_val:.2f}) suggests a decline/worsening "
                                        f"from previous ({prev_val:.2f}), an actual significantly better than forecast would be needed to turn sentiment bullish.")

    elif desired_outcome == "Bearish":
        if is_inverted_indicator:
            # For inverted indicators, a bearish outcome means Actual > Forecast
            significant_miss_value = forecast_val + buffer
            outcome_description = (f"For a **Bearish** outcome for {currency} from '{event_name}' (an inverted indicator where lower is better), "
                                   f"the Actual release would typically need to be **notably higher than forecast**. "
                                   f"Ideally, Actual > {forecast_val:.2f} (e.g., around or above {significant_miss_value:.2f}).")
        else:
            # For standard indicators, a bearish outcome means Actual < Forecast
            significant_miss_value = forecast_val - buffer
            outcome_description = (f"For a **Bearish** outcome for {currency} from '{event_name}' (where higher is generally better), "
                                   f"the Actual release would typically need to be **notably worse than forecast**. "
                                   f"Ideally, Actual < {forecast_val:.2f} (e.g., around or below {significant_miss_value:.2f}).")

        if previous is not None and not np.isnan(previous):
            prev_val = float(previous)
            # Context based on previous vs forecast
            if (is_inverted_indicator and forecast_val > prev_val) or \
               (not is_inverted_indicator and forecast_val < prev_val):
                outcome_description += (f" Given the forecast ({forecast_val:.2f}) already suggests a worsening "
                                        f"from previous ({prev_val:.2f}), a significant miss (actual further worsening on forecast) would be particularly bearish.")
            elif (is_inverted_indicator and forecast_val < prev_val) or \
                 (not is_inverted_indicator and forecast_val > prev_val):
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
    print(f"NFP (Prev: 150, Fcst: 180, Event: 'Non-Farm Employment Change'): Expected Bullish -> {infer_market_outlook_from_data(150.0, 180.0, 'Non-Farm Employment Change')}")
    print(f"GDP (Prev: 0.5, Fcst: 0.2, Event: 'GDP m/m'): Expected Bearish -> {infer_market_outlook_from_data(0.5, 0.2, 'GDP m/m')}")
    print(f"CPI (Prev: 0.3, Fcst: 0.31, Event: 'Core CPI m/m'): Expected Consolidating -> {infer_market_outlook_from_data(0.3, 0.31, 'Core CPI m/m')}")
    print(f"Unemployment (Prev: 3.9, Fcst: 3.7, Event: 'Unemployment Rate'): Expected Bullish -> {infer_market_outlook_from_data(3.9, 3.7, 'Unemployment Rate')}")
    print(f"Unemployment (Prev: 3.7, Fcst: 3.9, Event: 'Unemployment Rate'): Expected Bearish -> {infer_market_outlook_from_data(3.7, 3.9, 'Unemployment Rate')}")
    print(f"Event (Prev: None, Fcst: 10): Expected Consolidating -> {infer_market_outlook_from_data(None, 10.0, 'Some Event')}")

    # Test cases for predict_actual_condition_for_outcome
    print("\n--- Test Cases for predict_actual_condition_for_outcome ---")
    print("Bullish NFP (Prev: 175, Fcst: 200, Event: 'Non-Farm Employment Change'):")
    print(predict_actual_condition_for_outcome(175.0, 200.0, "Bullish", "USD", "Non-Farm Employment Change"))
    
    print("\nBearish GDP (Prev: 0.1, Fcst: 0.2, Event: 'GDP m/m'):") # Forecast is better, but we want bearish
    print(predict_actual_condition_for_outcome(0.1, 0.2, "Bearish", "GBP", "GDP m/m"))

    print("\nBullish Unemployment Rate (Prev: 3.9, Fcst: 3.9, Event: 'Unemployment Rate'):") # Inverted indicator
    print(predict_actual_condition_for_outcome(3.9, 3.9, "Bullish", "USD", "Unemployment Rate"))

    print("\nBearish Unemployment Rate (Prev: 3.9, Fcst: 3.7, Event: 'Unemployment Rate'):") # Inverted, Fcst is already better
    print(predict_actual_condition_for_outcome(3.9, 3.7, "Bearish", "USD", "Unemployment Rate"))

# strategy_engine.py
import numpy as np

def predict_actual_condition_for_outcome(previous, forecast, desired_outcome, currency, event_name):
    """
    Describes the condition the 'Actual' economic data would likely need to meet
    for a desired market outcome (Bullish/Bearish/Consolidating) for the specified currency.

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

    # Ensure forecast is float for calculations
    try:
        forecast_val = float(forecast)
    except ValueError:
        return (f"Forecast value '{forecast}' for '{event_name}' is not a valid number. "
                f"Quantitative interpretation is not possible.")

    # Define a small relative or absolute buffer for "around forecast"
    # This can be tuned based on typical volatility/significance of different indicators
    if forecast_val != 0:
        buffer = abs(forecast_val * 0.05) # 5% buffer relative to forecast
    else:
        buffer = 0.05 # Small absolute buffer if forecast is 0 (e.g. for rates or changes)
    
    # For some indicators (e.g., PMIs), 50 is a key threshold. This model is generic for now.
    # For rates, even minor deviations can be significant. Buffer might need to be event-specific.

    outcome_description = ""

    if desired_outcome == "Bullish":
        # Generally, Actual > Forecast is bullish.
        # If forecast is already very positive (e.g. high growth), meeting it might be enough.
        # If forecast is negative (e.g. contraction), a less negative actual is bullish.
        # Example: Forecast -0.5%. Actual -0.1% is bullish.
        significant_beat_value = forecast_val + buffer
        outcome_description = (f"For a **Bullish** outcome for {currency} from '{event_name}', "
                               f"the Actual release would typically need to be **better than forecast**. "
                               f"Ideally, Actual > {forecast_val:.2f} (e.g., around or above {significant_beat_value:.2f}).")
        if previous is not None and not np.isnan(previous):
            if forecast_val > previous:
                outcome_description += (f" Given the forecast ({forecast_val:.2f}) already indicates improvement "
                                        f"from previous ({previous:.2f}), a strong beat would be particularly bullish.")
            elif forecast_val < previous:
                 outcome_description += (f" Given the forecast ({forecast_val:.2f}) indicates a decline "
                                        f"from previous ({previous:.2f}), an actual significantly above forecast would be needed to turn sentiment bullish.")


    elif desired_outcome == "Bearish":
        # Generally, Actual < Forecast is bearish.
        significant_miss_value = forecast_val - buffer
        outcome_description = (f"For a **Bearish** outcome for {currency} from '{event_name}', "
                               f"the Actual release would typically need to be **worse than forecast**. "
                               f"Ideally, Actual < {forecast_val:.2f} (e.g., around or below {significant_miss_value:.2f}).")
        if previous is not None and not np.isnan(previous):
            if forecast_val < previous:
                outcome_description += (f" Given the forecast ({forecast_val:.2f}) already indicates a worsening "
                                        f"from previous ({previous:.2f}), a significant miss would be particularly bearish.")
            elif forecast_val > previous:
                outcome_description += (f" Given the forecast ({forecast_val:.2f}) indicates an improvement "
                                         f"from previous ({previous:.2f}), an actual significantly below forecast would be needed to turn sentiment bearish.")


    elif desired_outcome == "Consolidating":
        lower_bound = forecast_val - buffer
        upper_bound = forecast_val + buffer
        outcome_description = (f"For a **Consolidating/Neutral** outcome for {currency} from '{event_name}', "
                               f"the Actual release would typically need to be **in line with the forecast**. "
                               f"Ideally, Actual ≈ {forecast_val:.2f} (e.g., between {lower_bound:.2f} and {upper_bound:.2f}).")
        if previous is not None and not np.isnan(previous) and previous == forecast_val :
             outcome_description += (f" Since the forecast ({forecast_val:.2f}) matches the previous value, an in-line actual would reinforce consolidation.")


    else:
        return "Invalid desired outcome selected. Please choose Bullish, Bearish, or Consolidating."

    return outcome_description

if __name__ == '__main__':
    # Test cases
    print("--- Test Case 1: Bullish NFP ---")
    print(predict_actual_condition_for_outcome(previous=150.0, forecast=180.0, desired_outcome="Bullish", currency="USD", event_name="Non-Farm Payrolls"))
    
    print("\n--- Test Case 2: Bearish GDP ---")
    print(predict_actual_condition_for_outcome(previous=0.5, forecast=0.2, desired_outcome="Bearish", currency="EUR", event_name="GDP Growth q/q"))

    print("\n--- Test Case 3: Consolidating CPI ---")
    print(predict_actual_condition_for_outcome(previous=0.3, forecast=0.3, desired_outcome="Consolidating", currency="USD", event_name="CPI m/m"))

    print("\n--- Test Case 4: Event with no forecast ---")
    print(predict_actual_condition_for_outcome(previous=None, forecast=None, desired_outcome="Bullish", currency="JPY", event_name="BoJ Governor Speech"))
    
    print("\n--- Test Case 5: Forecast is negative, Bullish outcome ---")
    print(predict_actual_condition_for_outcome(previous=-0.5, forecast=-0.2, desired_outcome="Bullish", currency="GBP", event_name="Manufacturing PMI Change"))

    print("\n--- Test Case 6: Forecast is positive, Bearish outcome, prev is higher ---")
    print(predict_actual_condition_for_outcome(previous=100.0, forecast=50.0, desired_outcome="Bearish", currency="AUD", event_name="Retail Sales"))

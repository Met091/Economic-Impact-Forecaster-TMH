# strategy_engine.py
import numpy as np
import pandas as pd # Import pandas
import streamlit as st # Used only for potential error logging if needed

# --- Indicator Configuration ---
# This dictionary defines how different economic indicators are typically interpreted
# and what constitutes a significant deviation.
# - type: 'normal' (higher is better for currency), 'inverted' (lower is better), 'qualitative' (impact based on tone)
# - unit: The unit of measurement (e.g., "%", "K" for thousands)
# - significance_threshold_pct: Percentage deviation from forecast/previous considered significant (used if abs_significance is None)
# - buffer_pct: Percentage deviation from forecast considered "in-line" or neutral (used if abs_buffer is None)
# - abs_significance: Absolute deviation from forecast/previous considered significant (takes precedence over pct)
# - abs_buffer: Absolute deviation from forecast considered "in-line" or neutral (takes precedence over pct)
# - default_significance/default_buffer: Fallback absolute values if others cannot be calculated
# - description: A brief explanation of the indicator.

NUANCED_MULTIPLIER = 2.0 # Used to define "Strongly" Bullish/Bearish vs "Mildly"

INDICATOR_CONFIG = {
    "Non-Farm Employment Change": {
        "type": "normal", "unit": "K",
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05, # e.g. 10% of forecast
        "abs_significance": 30.0, "abs_buffer": 20.0, # Absolute K values
        "description": "Measures the change in the number of employed people during the previous month, excluding the farming industry. Higher is generally better for the currency (e.g., USD)."
    },
    "Employment Change": { # Generic employment change, might apply to CAD, AUD etc.
        "type": "normal", "unit": "K",
        "significance_threshold_pct": 0.15, "buffer_pct": 0.10,
        "abs_significance": 10.0, "abs_buffer": 5.0, # Smaller abs values for smaller economies typically
        "description": "Measures the change in the number of employed people. Higher is generally better for the currency."
    },
    "Unemployment Rate": {
        "type": "inverted", "unit": "%", # Lower is better
        "significance_threshold_pct": 0.03, "buffer_pct": 0.02, # e.g., 3% of the rate itself (0.1 if rate is 3.0%)
        "abs_significance": 0.1, "abs_buffer": 0.1, # Absolute percentage point change (e.g., 3.5% vs 3.6%)
        "description": "Measures the percentage of the total labor force that is unemployed but actively seeking employment and willing to work. Lower is generally better for the currency."
    },
    "GDP m/m": { # Gross Domestic Product Month-over-Month
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.20, "buffer_pct": 0.10, # e.g. 20% of the GDP growth rate
        "abs_significance": 0.2, "abs_buffer": 0.1, # Absolute percentage point change in growth rate
        "description": "Measures the month-over-month change in the inflation-adjusted value of all goods and services produced by the economy. Higher is generally better for the currency."
    },
     "GDP q/q": { # Gross Domestic Product Quarter-over-Quarter
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.15, "buffer_pct": 0.10,
        "abs_significance": 0.2, "abs_buffer": 0.1,
        "description": "Measures the quarter-over-quarter change in the inflation-adjusted value of all goods and services produced by the economy. Higher is generally better for the currency."
    },
    "Core CPI m/m": { # Core Consumer Price Index Month-over-Month
        "type": "normal", "unit": "%", # Higher inflation can lead to rate hikes, strengthening currency
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05,
        "abs_significance": 0.1, "abs_buffer": 0.1, # Absolute percentage point change
        "description": "Measures the month-over-month change in the price of goods and services, excluding food and energy. Higher can indicate inflationary pressure, potentially leading to currency strengthening via rate hike expectations."
    },
     "CPI m/m": { # Consumer Price Index Month-over-Month
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05,
        "abs_significance": 0.1, "abs_buffer": 0.1,
        "description": "Measures the month-over-month change in the price of goods and services. Higher can indicate inflationary pressure, potentially leading to currency strengthening."
    },
    "Policy Rate": { # e.g., Fed Funds Rate, ECB Main Refinancing Rate
        "type": "normal", "unit": "%", # Higher rates typically strengthen currency
        "significance_threshold_pct": 0.05, "buffer_pct": 0.02, # Not usually forecast, but change vs previous is key
        "abs_significance": 0.25, "abs_buffer": 0.0, # Any change is significant, 0.25 for a typical hike/cut
        "description": "The interest rate at which commercial banks can borrow money from the central bank. Higher rates (or hawkish outlook) typically strengthen a currency."
    },
    "Retail Sales m/m": {
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.15, "buffer_pct": 0.10,
        "abs_significance": 0.3, "abs_buffer": 0.2,
        "description": "Measures the month-over-month change in the total value of sales at the retail level. Higher indicates stronger consumer spending, generally positive for the currency."
    },
    "Manufacturing PMI": { # Purchasing Managers' Index
        "type": "normal", "unit": "", # Index value, e.g. 52.5
        "significance_threshold_pct": 0.02, "buffer_pct": 0.01, # e.g. 2% of the index value
        "abs_significance": 1.0, "abs_buffer": 0.5, # Absolute point change in index
        "description": "Purchasing Managers' Index for the manufacturing sector. Above 50 indicates expansion, below 50 indicates contraction. Higher is generally better."
    },
    "Services PMI": {
        "type": "normal", "unit": "",
        "significance_threshold_pct": 0.02, "buffer_pct": 0.01,
        "abs_significance": 1.0, "abs_buffer": 0.5,
        "description": "Purchasing Managers' Index for the services sector. Above 50 indicates expansion, below 50 indicates contraction. Higher is generally better."
    },
    # Qualitative Events - these rely on sentiment rather than numerical deviation
    "ECB President Speaks": {"type": "qualitative", "unit": "", "description": "Speeches by the ECB President can cause significant volatility based on their tone regarding monetary policy for the Eurozone."},
    "FOMC Press Conference": {"type": "qualitative", "unit": "", "description": "Press conference following the Federal Open Market Committee (FOMC) meeting, providing insights into US monetary policy and its outlook."},
    "BoE Gov Speaks": {"type": "qualitative", "unit": "", "description": "Speeches by the Bank of England Governor, influencing GBP through monetary policy signals."},
    "BoJ Gov Speaks": {"type": "qualitative", "unit": "", "description": "Speeches by the Bank of Japan Governor, impacting JPY via monetary policy stance."},
    "Default": { # Fallback for unclassified indicators
        "type": "normal", "unit": "",
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05,
        "default_significance": 0.1, "default_buffer": 0.05, # Default absolute values if forecast/prev are zero or NaN
        "description": "Default interpretation rules for unclassified economic indicators. Assumes higher is generally better."
    }
}

def get_indicator_properties(event_name):
    """
    Fetches properties for a given event_name from INDICATOR_CONFIG.
    Prioritizes more specific (longer key) matches.
    """
    if not event_name or pd.isna(event_name):
        return INDICATOR_CONFIG["Default"].copy()

    best_match_key = "Default"
    max_match_len = 0
    event_name_lower = str(event_name).lower()

    for key in INDICATOR_CONFIG:
        if key == "Default":
            continue # Skip default unless it's the only one left

        key_lower = key.lower()
        if key_lower in event_name_lower:
            # If a key is found within the event_name, it's a potential match.
            # Prioritize longer keys as they are usually more specific.
            if len(key_lower) > max_match_len:
                max_match_len = len(key_lower)
                best_match_key = key
        # Handling cases like "PMI" where it might be "Manufacturing PMI" or "Services PMI"
        # This simple "in" check might lead to "PMI" (if it existed as a key) matching "Manufacturing PMI".
        # More sophisticated matching might be needed if ambiguity arises.
        # For now, longer key match preference helps.

    # Specific handling for generic terms if not caught by longer keys
    if best_match_key == "Default": # If no specific key matched well
        if "pmi" in event_name_lower:
            if "manu" in event_name_lower: best_match_key = "Manufacturing PMI"
            elif "serv" in event_name_lower or "non-manu" in event_name_lower : best_match_key = "Services PMI"
            # If just "PMI" and not more specific, it could remain Default or map to a generic PMI if one was defined
        elif "gdp" in event_name_lower:
            if "m/m" in event_name_lower : best_match_key = "GDP m/m"
            elif "q/q" in event_name_lower: best_match_key = "GDP q/q"
        elif "cpi" in event_name_lower:
             if "core" in event_name_lower: best_match_key = "Core CPI m/m"
             else: best_match_key = "CPI m/m"
        elif "unemployment rate" in event_name_lower:
            best_match_key = "Unemployment Rate"
        elif "employment change" in event_name_lower: # Catches NFP too if not more specific
            if "non-farm" in event_name_lower or "nfp" in event_name_lower: best_match_key = "Non-Farm Employment Change"
            else: best_match_key = "Employment Change"


    return INDICATOR_CONFIG.get(best_match_key, INDICATOR_CONFIG["Default"]).copy()


def _calculate_threshold(value, pct_threshold, abs_threshold, default_abs_threshold):
    """
    Helper to calculate deviation thresholds.
    Prioritizes absolute threshold if provided, then percentage, then default.
    """
    if abs_threshold is not None and not np.isnan(abs_threshold):
        return abs_threshold
    if value is not None and not np.isnan(value) and value != 0 and pct_threshold is not None and not np.isnan(pct_threshold):
        return abs(value * pct_threshold)
    return default_abs_threshold # Fallback if value is None, NaN, or zero for percentage calculation

def infer_market_outlook_from_data(previous, forecast, event_name):
    """
    Infers a general market bias (Bullish/Bearish/Consolidating for the currency)
    based on the Forecast value relative to the Previous value for a given economic event.
    """
    props = get_indicator_properties(event_name)
    if props["type"] == "qualitative":
        return "Subjective (Qualitative)"

    # Ensure previous and forecast are numeric, handle None or NaN
    if forecast is None or np.isnan(forecast) or previous is None or np.isnan(previous):
        return "Consolidating (Insufficient Data)"
    try:
        prev_val, fcst_val = float(previous), float(forecast)
    except ValueError:
        return "Consolidating (Invalid Data)" # Should not happen if data cleaning is robust

    # Use 'significance_threshold_pct' and 'abs_significance' from config
    # The 'significance' threshold determines if the deviation is meaningful enough to infer a bias.
    significance_thresh_val = _calculate_threshold(
        prev_val, # Bias is often judged against how much forecast deviates from previous
        props.get("significance_threshold_pct"),
        props.get("abs_significance"),
        props.get("default_significance", INDICATOR_CONFIG["Default"]["default_significance"])
    )

    deviation = fcst_val - prev_val

    if abs(deviation) < significance_thresh_val:
        return "Consolidating" # Deviation not significant enough

    is_bullish_deviation = deviation > 0 # Forecast is higher than previous

    # Apply indicator type (normal or inverted)
    if props["type"] == "inverted": # Lower is better for currency
        return "Bearish" if is_bullish_deviation else "Bullish"
    else: # Normal type: Higher is better for currency
        return "Bullish" if is_bullish_deviation else "Bearish"


def predict_actual_condition_for_outcome(previous, forecast, desired_outcome, currency, event_name):
    """
    Predicts what the 'Actual' data release would likely need to be (relative to Forecast)
    to achieve a desired market outcome (Bullish, Bearish, Consolidating) for the currency.
    Returns a list of plain text strings for display.
    """
    props = get_indicator_properties(event_name)
    unit = props.get("unit", "")
    indicator_nature_desc = "lower is generally better" if props['type'] == 'inverted' else "higher is generally better"
    analysis_points = []

    analysis_points.append(f"Event Context for {currency} ({event_name}):")
    analysis_points.append(f"  Indicator Type: {props['type']} ({indicator_nature_desc} for {currency}).")
    if props.get('description'):
        analysis_points.append(f"  Event Focus: {props['description']}")

    if props["type"] == "qualitative":
        analysis_points.append(f"\nFor a '{desired_outcome}' outcome for {currency} from '{event_name}':")
        if desired_outcome == "Bullish":
            analysis_points.extend([
                f"  - Rhetoric Needed: Clearly hawkish statements suggesting tighter monetary policy or a surprisingly optimistic economic outlook.",
                f"  - Key Signals: Hints of accelerated rate hikes, reduction in asset purchases, or strong confidence in achieving inflation targets (if applicable).",
                f"  - Market Expectation: The perceived hawkishness must exceed current market expectations to drive {currency} significantly higher."
            ])
        elif desired_outcome == "Bearish":
            analysis_points.extend([
                f"  - Rhetoric Needed: Clearly dovish statements suggesting looser monetary policy or a surprisingly pessimistic economic outlook.",
                f"  - Key Signals: Hints of potential rate cuts, increase in asset purchases, or concerns about economic growth/inflation outlook.",
                f"  - Market Expectation: The perceived dovishness must exceed current market expectations to drive {currency} significantly lower."
            ])
        else: # Consolidating
            analysis_points.extend([
                f"  - Rhetoric Needed: Balanced statements, reaffirming current policy stance without new strong signals, or comments fully aligned with market consensus.",
                f"  - Key Signals: Emphasis on data-dependency, no change in forward guidance, or a mix of cautious optimism and acknowledged risks.",
                f"  - Market Expectation: Speech offers no major surprises, leading to limited net directional impact on {currency}."
            ])
        analysis_points.append(f"\nNote: Qualitative event impacts are highly subjective. Consider using the 'AI Sentiment Analysis' feature for deeper insights.")
        return analysis_points

    if forecast is None or np.isnan(forecast): # Check if forecast is NaN or None
        analysis_points.append(f"\nQuantitative interpretation for '{event_name}' ({currency}) is not possible: 'Forecast' data is unavailable or invalid.")
        return analysis_points
    try:
        forecast_val = float(forecast)
    except ValueError:
        analysis_points.append(f"\nError: Forecast value '{forecast}' for '{event_name}' ({currency}) is invalid.")
        return analysis_points

    # Use 'buffer_pct' and 'abs_buffer' for defining "in-line" vs "beat/miss"
    # The 'buffer' defines the range around the forecast considered neutral.
    buffer = _calculate_threshold(
        forecast_val, # Predictions are based on deviation from forecast
        props.get("buffer_pct"),
        props.get("abs_buffer"),
        props.get("default_buffer", INDICATOR_CONFIG["Default"]["default_buffer"])
    )
    strong_buffer = buffer * NUANCED_MULTIPLIER # For "Strongly" Bullish/Bearish

    analysis_points.append(f"\nScenario for a '{desired_outcome}' outcome for {currency} (Forecast: {forecast_val:.2f}{unit}):")
    if desired_outcome == "Bullish":
        if props["type"] == "inverted": # Lower actual is bullish
            mild_beat_target = forecast_val - buffer
            strong_beat_target = forecast_val - strong_buffer
            analysis_points.extend([
                f"  - Condition: Actual significantly < Forecast.",
                f"  - Mildly Bullish: Actual release around or slightly below {mild_beat_target:.2f}{unit}.",
                f"  - Strongly Bullish: Actual release at or significantly below {strong_beat_target:.2f}{unit}."
            ])
        else: # Normal type: Higher actual is bullish
            mild_beat_target = forecast_val + buffer
            strong_beat_target = forecast_val + strong_buffer
            analysis_points.extend([
                f"  - Condition: Actual significantly > Forecast.",
                f"  - Mildly Bullish: Actual release around or slightly above {mild_beat_target:.2f}{unit}.",
                f"  - Strongly Bullish: Actual release at or significantly above {strong_beat_target:.2f}{unit}."
            ])
    elif desired_outcome == "Bearish":
        if props["type"] == "inverted": # Higher actual is bearish
            mild_miss_target = forecast_val + buffer
            strong_miss_target = forecast_val + strong_buffer
            analysis_points.extend([
                f"  - Condition: Actual significantly > Forecast.",
                f"  - Mildly Bearish: Actual release around or slightly above {mild_miss_target:.2f}{unit}.",
                f"  - Strongly Bearish: Actual release at or significantly above {strong_miss_target:.2f}{unit}."
            ])
        else: # Normal type: Lower actual is bearish
            mild_miss_target = forecast_val - buffer
            strong_miss_target = forecast_val - strong_buffer
            analysis_points.extend([
                f"  - Condition: Actual significantly < Forecast.",
                f"  - Mildly Bearish: Actual release around or slightly below {mild_miss_target:.2f}{unit}.",
                f"  - Strongly Bearish: Actual release at or significantly below {strong_miss_target:.2f}{unit}."
            ])
    elif desired_outcome == "Consolidating":
        lower_bound_neutral = forecast_val - buffer
        upper_bound_neutral = forecast_val + buffer
        analysis_points.extend([
            f"  - Condition: Actual ≈ Forecast.",
            f"  - Expected Range: Actual release roughly between {lower_bound_neutral:.2f}{unit} and {upper_bound_neutral:.2f}{unit}."
        ])
    else:
        analysis_points.append("  - Invalid desired outcome selected.")

    # Context vs Previous
    if previous is not None and not np.isnan(previous): # Check if previous is NaN or None
        try:
            prev_val = float(previous)
            analysis_points.append(f"\nContext vs. Previous ({prev_val:.2f}{unit}):")
            fcst_outlook_vs_prev = infer_market_outlook_from_data(prev_val, forecast_val, event_name)
            analysis_points.append(f"  - The current forecast ({forecast_val:.2f}{unit}) itself suggests a {fcst_outlook_vs_prev.lower()} bias for {currency} compared to the previous value.")
            if desired_outcome == "Bullish":
                if (props["type"] == "inverted" and forecast_val > prev_val) or \
                   (props["type"] != "inverted" and forecast_val < prev_val): # Forecast is already bearish vs previous
                    analysis_points.append(f"  - Achieving a bullish outcome would require the actual to strongly reverse the trend implied by forecast vs. previous.")
            elif desired_outcome == "Bearish":
                if (props["type"] == "inverted" and forecast_val < prev_val) or \
                   (props["type"] != "inverted" and forecast_val > prev_val): # Forecast is already bullish vs previous
                    analysis_points.append(f"  - Achieving a bearish outcome would require the actual to strongly reverse the trend implied by forecast vs. previous.")
        except ValueError:
            analysis_points.append(f"  - Previous value '{previous}' is invalid for comparison.")
    else:
        analysis_points.append("\nPrevious value is not available for additional context.")
    return analysis_points


def classify_actual_release(actual_value, forecast_value, previous_value, event_name, currency):
    """
    Classifies an 'Actual' data release against 'Forecast' and 'Previous' values.
    Returns a classification string (e.g., "Strongly Bullish") and an explanation string.
    """
    props = get_indicator_properties(event_name)
    unit = props.get("unit", "")

    if props["type"] == "qualitative":
        return "Qualitative", f"'{event_name}' is a qualitative event. Its impact is subjective and based on the perceived tone and message. Use the 'AI Sentiment Analysis' feature for an interpretation."

    # Validate inputs
    if actual_value is None or np.isnan(actual_value):
        return "Indeterminate", "Actual value is missing or invalid. Cannot classify."
    if forecast_value is None or np.isnan(forecast_value):
        # If forecast is missing, we can still compare actual to previous, but it's a weaker signal
        # For now, let's require forecast for a full classification.
        # Alternative: classify based on Actual vs Previous if Forecast is missing.
        return "Indeterminate", "Forecast value is missing or invalid. Cannot classify against forecast."

    try:
        actual = float(actual_value)
        forecast = float(forecast_value)
    except ValueError:
        return "Error", "Invalid numeric input for Actual or Forecast values."

    # Calculate buffer thresholds based on forecast value
    buffer = _calculate_threshold(
        forecast,
        props.get("buffer_pct"),
        props.get("abs_buffer"),
        props.get("default_buffer", INDICATOR_CONFIG["Default"]["default_buffer"])
    )
    strong_buffer = buffer * NUANCED_MULTIPLIER
    deviation = actual - forecast
    classification = "Neutral/In-Line"
    is_inverted = props["type"] == "inverted"

    # Classification logic based on deviation from forecast
    if is_inverted: # Lower actual is better/bullish
        if deviation <= -strong_buffer: classification = "Strongly Bullish"
        elif deviation < -buffer: classification = "Mildly Bullish" # deviation is negative, beyond buffer
        elif deviation >= strong_buffer: classification = "Strongly Bearish"
        elif deviation > buffer: classification = "Mildly Bearish" # deviation is positive, beyond buffer
    else: # Normal type: Higher actual is better/bullish
        if deviation >= strong_buffer: classification = "Strongly Bullish"
        elif deviation > buffer: classification = "Mildly Bullish"
        elif deviation <= -strong_buffer: classification = "Strongly Bearish"
        elif deviation < -buffer: classification = "Mildly Bearish"
    
    # Construct explanation
    prev_text_part = ""
    if previous_value is not None and not np.isnan(previous_value):
        try:
            prev_f = float(previous_value)
            prev_text_part = f", Previous: {prev_f:.2f}{unit}"
        except ValueError:
            prev_text_part = f", Previous: {previous_value} (invalid)"

    explanation_lines = [
        f"Event: '{event_name}' ({currency}, Type: {props['type']})",
        f"Data: Actual: {actual:.2f}{unit}, Forecast: {forecast:.2f}{unit}{prev_text_part}",
        f"Analysis: Deviation (Actual - Forecast): {deviation:.2f}{unit}. Buffer for Neutral: ±{buffer:.2f}{unit} (Strong Deviation beyond ±{strong_buffer:.2f}{unit}).",
        f"Outcome for {currency}: {classification}"
    ]

    reason_map = {
        "Strongly Bullish": f"Actual significantly {'better (lower)' if is_inverted else 'better (higher)'} than forecast (deviation beyond ±{strong_buffer:.2f}{unit}).",
        "Mildly Bullish": f"Actual moderately {'better (lower)' if is_inverted else 'better (higher)'} than forecast (deviation beyond ±{buffer:.2f}{unit} but within ±{strong_buffer:.2f}{unit}).",
        "Strongly Bearish": f"Actual significantly {'worse (higher)' if is_inverted else 'worse (lower)'} than forecast (deviation beyond ±{strong_buffer:.2f}{unit}).",
        "Mildly Bearish": f"Actual moderately {'worse (higher)' if is_inverted else 'worse (lower)'} than forecast (deviation beyond ±{buffer:.2f}{unit} but within ±{strong_buffer:.2f}{unit}).",
        "Neutral/In-Line": f"Actual release is within the expected neutral buffer (±{buffer:.2f}{unit}) around the forecast."
    }
    explanation_lines.append(f"  Reason: {reason_map.get(classification, 'Classification reason not defined.')}")

    # Comparison with Previous value, if available and valid
    if previous_value is not None and not np.isnan(previous_value) and pd.notna(actual): # check actual is not nan
        try:
            prev_f = float(previous_value)
            actual_vs_prev_dev = actual - prev_f
            
            # Determine if actual is better/worse/similar to previous
            # Using the same 'buffer' calculated from forecast for consistency in deviation magnitude,
            # though significance vs previous might have different psychological thresholds.
            # For simplicity, we use the same buffer.
            prev_outlook_detail = "similar to"
            if is_inverted: # lower is better
                if actual_vs_prev_dev < -buffer: prev_outlook_detail = "significantly better than"
                elif actual_vs_prev_dev < 0 : prev_outlook_detail = "better than"
                elif actual_vs_prev_dev > buffer: prev_outlook_detail = "significantly worse than"
                elif actual_vs_prev_dev > 0: prev_outlook_detail = "worse than"
            else: # higher is better
                if actual_vs_prev_dev > buffer: prev_outlook_detail = "significantly better than"
                elif actual_vs_prev_dev > 0: prev_outlook_detail = "better than"
                elif actual_vs_prev_dev < -buffer: prev_outlook_detail = "significantly worse than"
                elif actual_vs_prev_dev < 0: prev_outlook_detail = "worse than"
            
            explanation_lines.append(f"  Vs Previous: Actual ({actual:.2f}{unit}) is {prev_outlook_detail} Previous ({prev_f:.2f}{unit}). This comparison provides additional market context.")
        except ValueError:
            explanation_lines.append(f"  Vs Previous: Previous value '{previous_value}' is invalid for direct comparison.")
        
    return classification, "\n".join(explanation_lines)


# strategy_engine.py
import numpy as np
import streamlit as st # Used only for potential error logging if needed

# --- Indicator Configuration ---
NUANCED_MULTIPLIER = 2.0

INDICATOR_CONFIG = {
    "Non-Farm Employment Change": {
        "type": "normal", "unit": "K",
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05,
        "abs_significance": 30.0, "abs_buffer": 20.0,
        "description": "Measures the change in the number of employed people during the previous month, excluding the farming industry. Higher is generally better for the currency."
    },
    "Employment Change": {
        "type": "normal", "unit": "K",
        "significance_threshold_pct": 0.15, "buffer_pct": 0.10,
        "abs_significance": 7.0, "abs_buffer": 5.0,
        "description": "Measures the change in the number of employed people. Higher is generally better for the currency."
    },
    "Unemployment Rate": {
        "type": "inverted", "unit": "%",
        "significance_threshold_pct": 0.03, "buffer_pct": 0.02,
        "abs_significance": 0.1, "abs_buffer": 0.1,
        "description": "Measures the percentage of the total labor force that is unemployed but actively seeking employment and willing to work. Lower is generally better for the currency."
    },
    "GDP m/m": {
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.20, "buffer_pct": 0.10,
        "abs_significance": 0.1, "abs_buffer": 0.1,
        "description": "Measures the month-over-month change in the inflation-adjusted value of all goods and services produced by the economy. Higher is generally better."
    },
     "GDP q/q": {
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.15, "buffer_pct": 0.10,
        "abs_significance": 0.2, "abs_buffer": 0.1,
        "description": "Measures the quarter-over-quarter change in the inflation-adjusted value of all goods and services produced by the economy. Higher is generally better."
    },
    "Core CPI m/m": {
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05,
        "abs_significance": 0.1, "abs_buffer": 0.1,
        "description": "Measures the month-over-month change in the price of goods and services, excluding food and energy. Higher can indicate inflationary pressure, potentially leading to currency strengthening via rate hike expectations."
    },
     "CPI m/m": {
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05,
        "abs_significance": 0.1, "abs_buffer": 0.1,
        "description": "Measures the month-over-month change in the price of goods and services. Higher can indicate inflationary pressure, potentially leading to currency strengthening."
    },
    "Policy Rate": {
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.05, "buffer_pct": 0.02,
        "abs_significance": 0.15, "abs_buffer": 0.10,
        "description": "The interest rate at which commercial banks can borrow money from the central bank. Higher rates typically strengthen a currency."
    },
    "Retail Sales m/m": {
        "type": "normal", "unit": "%",
        "significance_threshold_pct": 0.15, "buffer_pct": 0.10,
        "abs_significance": 0.3, "abs_buffer": 0.2,
        "description": "Measures the month-over-month change in the total value of sales at the retail level. Higher indicates stronger consumer spending."
    },
    "PMI": {
        "type": "normal", "unit": "",
        "significance_threshold_pct": 0.02, "buffer_pct": 0.01,
        "abs_significance": 1.0, "abs_buffer": 0.8,
        "description": "Purchasing Managers' Index; an indicator of economic health for manufacturing or services sectors. Above 50 indicates expansion, below 50 indicates contraction."
    },
    "ECB President Speaks": {"type": "qualitative", "unit": "", "description": "Speeches by the ECB President can cause significant volatility based on their tone regarding monetary policy for the Eurozone."},
    "FOMC Press Conference": {"type": "qualitative", "unit": "", "description": "Press conference following the Federal Open Market Committee (FOMC) meeting, providing insights into US monetary policy and its outlook."},
    "BoE Gov Speaks": {"type": "qualitative", "unit": "", "description": "Speeches by the Bank of England Governor, influencing GBP through monetary policy signals."},
    "BoJ Gov Speaks": {"type": "qualitative", "unit": "", "description": "Speeches by the Bank of Japan Governor, impacting JPY via monetary policy stance."},
    "Default": {
        "type": "normal", "unit": "",
        "significance_threshold_pct": 0.10, "buffer_pct": 0.05,
        "default_significance": 0.1, "default_buffer": 0.1,
        "description": "Default interpretation rules for unclassified economic indicators."
    }
}

def get_indicator_properties(event_name):
    """Fetches properties, prioritizing more specific matches."""
    best_match_key = "Default"
    max_match_len = 0
    event_name_lower = event_name.lower()
    for key in INDICATOR_CONFIG:
        key_lower = key.lower()
        if key_lower in event_name_lower:
            if len(key_lower) > max_match_len:
                max_match_len = len(key_lower)
                best_match_key = key
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
    if abs_threshold is not None:
        return abs_threshold
    if value is not None and not np.isnan(value) and value != 0 and pct_threshold is not None:
        return abs(value * pct_threshold)
    return default_abs_threshold

def infer_market_outlook_from_data(previous, forecast, event_name):
    """Infers a general market bias based on Forecast vs. Previous."""
    props = get_indicator_properties(event_name)
    if props["type"] == "qualitative": return "Subjective (Qualitative)"
    if forecast is None or np.isnan(forecast) or previous is None or np.isnan(previous): return "Consolidating (Insufficient Data)"
    try: prev_val, fcst_val = float(previous), float(forecast)
    except ValueError: return "Consolidating (Invalid Data)"
    
    significance_threshold = _calculate_threshold(prev_val, props.get("significance_threshold_pct"), props.get("abs_significance"), props.get("default_significance", INDICATOR_CONFIG["Default"]["default_significance"]))
    deviation = fcst_val - prev_val
    if abs(deviation) < significance_threshold: return "Consolidating"
    
    is_bullish_deviation = deviation > 0
    return ("Bullish" if not is_bullish_deviation else "Bearish") if props["type"] == "inverted" else ("Bullish" if is_bullish_deviation else "Bearish")

def predict_actual_condition_for_outcome(previous, forecast, desired_outcome, currency, event_name):
    """
    Predicts conditions for a desired outcome. Returns a list of plain text strings.
    """
    props = get_indicator_properties(event_name)
    unit = props.get("unit", "")
    indicator_nature_desc = "lower is generally better" if props['type'] == 'inverted' else "higher is generally better"
    analysis_points = []

    # Section headers are now plain text
    analysis_points.append(f"Event Context for {currency} ({event_name}):")
    analysis_points.append(f"Indicator Type: {props['type']} ({indicator_nature_desc} for {currency}).")
    if props.get('description'):
        analysis_points.append(f"Event Focus: {props['description']}")

    if props["type"] == "qualitative":
        analysis_points.append(f"For a '{desired_outcome}' outcome for {currency} from '{event_name}':") # Plain section header
        if desired_outcome == "Bullish":
            analysis_points.extend([
                f"Rhetoric Needed: Clearly hawkish statements suggesting tighter monetary policy or a surprisingly optimistic economic outlook.",
                f"Key Signals: Hints of accelerated rate hikes, reduction in asset purchases, or strong confidence in achieving inflation targets (if applicable).",
                f"Market Expectation: The perceived hawkishness must exceed current market expectations to drive {currency} significantly higher."
            ])
        elif desired_outcome == "Bearish":
            analysis_points.extend([
                f"Rhetoric Needed: Clearly dovish statements suggesting looser monetary policy or a surprisingly pessimistic economic outlook.",
                f"Key Signals: Hints of potential rate cuts, increase in asset purchases, or concerns about economic growth/inflation outlook.",
                f"Market Expectation: The perceived dovishness must exceed current market expectations to drive {currency} significantly lower."
            ])
        else: # Consolidating
            analysis_points.extend([
                f"Rhetoric Needed: Balanced statements, reaffirming current policy stance without new strong signals, or comments fully aligned with market consensus.",
                f"Key Signals: Emphasis on data-dependency, no change in forward guidance, or a mix of cautious optimism and acknowledged risks.",
                f"Market Expectation: Speech offers no major surprises, leading to limited net directional impact on {currency}."
            ])
        analysis_points.append(f"Note: Qualitative event impacts are highly subjective. Consider using the 'AI Sentiment Analysis' feature for deeper insights.")
        return analysis_points

    if forecast is None or np.isnan(forecast):
        analysis_points.append(f"Quantitative interpretation for '{event_name}' ({currency}) is not possible: 'Forecast' data is unavailable.")
        return analysis_points
    try: forecast_val = float(forecast)
    except ValueError:
        analysis_points.append(f"Error: Forecast value '{forecast}' for '{event_name}' ({currency}) is invalid.")
        return analysis_points

    buffer = _calculate_threshold(forecast_val, props.get("buffer_pct"), props.get("abs_buffer"), props.get("default_buffer", INDICATOR_CONFIG["Default"]["default_buffer"]))
    strong_buffer = buffer * NUANCED_MULTIPLIER

    analysis_points.append(f"Scenario for a '{desired_outcome}' outcome for {currency} (Forecast: {forecast_val:.2f}{unit}):") # Plain section header
    if desired_outcome == "Bullish":
        if props["type"] == "inverted":
            mild_beat, strong_beat = forecast_val - buffer, forecast_val - strong_buffer
            analysis_points.extend([f"Condition: Actual < {forecast_val:.2f}{unit}.", f"Mildly Bullish: Actual around {mild_beat:.2f}{unit} (or lower).", f"Strongly Bullish: Actual ≤ {strong_beat:.2f}{unit}."])
        else:
            mild_beat, strong_beat = forecast_val + buffer, forecast_val + strong_buffer
            analysis_points.extend([f"Condition: Actual > {forecast_val:.2f}{unit}.", f"Mildly Bullish: Actual around {mild_beat:.2f}{unit} (or higher).", f"Strongly Bullish: Actual ≥ {strong_beat:.2f}{unit}."])
    elif desired_outcome == "Bearish":
        if props["type"] == "inverted":
            mild_miss, strong_miss = forecast_val + buffer, forecast_val + strong_buffer
            analysis_points.extend([f"Condition: Actual > {forecast_val:.2f}{unit}.", f"Mildly Bearish: Actual around {mild_miss:.2f}{unit} (or higher).", f"Strongly Bearish: Actual ≥ {strong_miss:.2f}{unit}."])
        else:
            mild_miss, strong_miss = forecast_val - buffer, forecast_val - strong_buffer
            analysis_points.extend([f"Condition: Actual < {forecast_val:.2f}{unit}.", f"Mildly Bearish: Actual around {mild_miss:.2f}{unit} (or lower).", f"Strongly Bearish: Actual ≤ {strong_miss:.2f}{unit}."])
    elif desired_outcome == "Consolidating":
        lower_b, upper_b = forecast_val - buffer, forecast_val + buffer
        analysis_points.extend([f"Condition: Actual ≈ {forecast_val:.2f}{unit}.", f"Expected Range: Approx. {lower_b:.2f}{unit} to {upper_b:.2f}{unit}."])
    else: analysis_points.append("Invalid desired outcome.")

    if previous is not None and not np.isnan(previous):
        try:
            prev_val = float(previous)
            analysis_points.append(f"Context vs. Previous ({prev_val:.2f}{unit}):") # Plain section header
            fcst_outlook = infer_market_outlook_from_data(prev_val, forecast_val, event_name)
            # Sub-points are also plain
            analysis_points.append(f"  The current forecast ({forecast_val:.2f}{unit}) suggests a {fcst_outlook.lower()} bias compared to the previous value.")
            if desired_outcome == "Bullish" and ( (props["type"] == "inverted" and forecast_val > prev_val) or (props["type"] == "normal" and forecast_val < prev_val) ):
                analysis_points.append("  Achieving a bullish outcome would require the actual to reverse or significantly outperform the current forecast-previous trend.")
            elif desired_outcome == "Bearish" and ( (props["type"] == "inverted" and forecast_val < prev_val) or (props["type"] == "normal" and forecast_val > prev_val) ):
                analysis_points.append("  Achieving a bearish outcome would require the actual to reverse or significantly underperform the current forecast-previous trend.")
        except ValueError: analysis_points.append(f"  Previous value '{previous}' is invalid for comparison.")
    else: analysis_points.append("Previous value is not available for additional context.")
    return analysis_points

def classify_actual_release(actual_value, forecast_value, previous_value, event_name, currency):
    """Classifies an 'Actual' release. Returns (classification_str, explanation_str with plain text)."""
    props = get_indicator_properties(event_name); unit = props.get("unit", "")
    if props["type"] == "qualitative":
        return "Qualitative", f"'{event_name}' is qualitative. Impact is subjective. Use 'AI Sentiment Analysis' for an interpretation based on perceived tone."
    if actual_value is None or np.isnan(actual_value): return "Indeterminate", "Actual value missing."
    if forecast_value is None or np.isnan(forecast_value): return "Indeterminate", "Forecast value missing."
    try: actual, forecast = float(actual_value), float(forecast_value)
    except ValueError: return "Error", "Invalid numeric input."

    buffer = _calculate_threshold(forecast, props.get("buffer_pct"), props.get("abs_buffer"), props.get("default_buffer", INDICATOR_CONFIG["Default"]["default_buffer"]))
    strong_buffer = buffer * NUANCED_MULTIPLIER
    deviation = actual - forecast
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
    
    prev_text = ""; explanation_lines = []
    if previous_value is not None and not np.isnan(previous_value):
        try: prev_text = f", Previous: {float(previous_value):.2f}{unit}"
        except ValueError: prev_text = f", Previous: {previous_value} (invalid)"

    # Explanation lines are now plain text, bolding removed.
    explanation_lines.extend([
        f"Event: '{event_name}' ({currency}, Type: {props['type']})",
        f"Data: Actual: {actual:.2f}{unit}, Forecast: {forecast:.2f}{unit}{prev_text}",
        f"Analysis: Deviation (Actual - Fcst): {deviation:.2f}{unit}. Buffer: ±{buffer:.2f}{unit} (Strong: ±{strong_buffer:.2f}{unit}).",
        f"Outcome for {currency}: {classification}" # Classification itself can be bolded in app.py if desired
    ])
    reason_map = {
        "Strongly Bullish": f"Actual significantly {'better (below)' if is_inverted else 'better (above)'} than forecast (beyond strong threshold).",
        "Mildly Bullish": f"Actual moderately {'better (below)' if is_inverted else 'better (above)'} than forecast (beyond buffer).",
        "Strongly Bearish": f"Actual significantly {'worse (above)' if is_inverted else 'worse (below)'} than forecast (beyond strong threshold).",
        "Mildly Bearish": f"Actual moderately {'worse (above)' if is_inverted else 'worse (below)'} than forecast (beyond buffer).",
        "Neutral/In-Line": f"Actual within expected buffer (±{buffer:.2f}{unit}) around forecast."
    }
    explanation_lines.append(f"  Reason: {reason_map.get(classification)}")

    if previous_value is not None and not np.isnan(previous_value) and pd.notna(actual):
        try:
            prev_f = float(previous_value); actual_vs_prev_dev = actual - prev_f
            prev_outlook = "better" if (is_inverted and actual < prev_f) or (not is_inverted and actual > prev_f) else "worse"
            if abs(actual_vs_prev_dev) < buffer: prev_outlook = "similar to"
            explanation_lines.append(f"  Vs Previous: Actual ({actual:.2f}{unit}) is {prev_outlook} Previous ({prev_f:.2f}{unit}).")
        except ValueError: pass
        
    return classification, "\n".join(explanation_lines)

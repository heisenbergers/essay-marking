import streamlit as st
import os
import json
import re
import pandas as pd
import numpy as np # <-- Added import

@st.cache_data # Cache the loaded prompts
def load_prompts(folder="context"):
    """Loads all .txt files from the specified folder as prompts."""
    prompts = {}
    if os.path.exists(folder) and os.path.isdir(folder):
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                prompt_name = os.path.splitext(filename)[0]
                prompt_name = re.sub(r'[_-]', ' ', prompt_name).title() # Clean up name
                filepath = os.path.join(folder, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        prompts[prompt_name] = f.read()
                except Exception as e:
                    st.warning(f"Could not load prompt '{filename}': {e}")
    else:
        # Handle case where folder doesn't exist locally
        # st.warning(f"Prompt folder '{folder}' not found.") # Optional warning
        pass
    if not prompts:
         # Provide a fallback minimal prompt if none are loaded
         prompts["Default Fallback"] = """Score the following text on a scale of 1-10. Respond in JSON format: {"score": N, "reason": "Your reasoning"}."""

    return prompts


def _extract_first_number(text: str) -> str:
    """Helper function to extract the first number (int or float) from text."""
    if not isinstance(text, str):
        return ""
    # Regex to find the first standalone number (positive or negative, int or float)
    # Allows optional leading/trailing whitespace around the number itself
    match = re.search(r'(?<![.\w])(-?\d+(?:\.\d+)?)(?![.\w])', text)
    # Fallback: search even if attached to words, but prefer standalone
    if not match:
        match = re.search(r'(-?\d+(?:\.\d+)?)', text)

    return match.group(1) if match else ""

# --- REFACTORED PARSING FUNCTION ---
def parse_llm_response(response_text: str, expected_format: str = "json_score_reason"):
    """
    Parse the LLM API response based on the expected format.

    Args:
        response_text (str): The raw text response from the LLM.
        expected_format (str): The expected format ('json_score_reason', 'integer_score', 'raw_text').

    Returns:
        tuple: (score, reason) - Score is the extracted numeric score (as str) or raw text.
               Reason is the extracted reason (as str) or empty/raw text.
    """
    score = ""
    reason = ""
    cleaned_text = response_text # Default to original if cleaning/parsing fails

    if not isinstance(response_text, str) or not response_text.strip():
        return score, reason # Return empty strings for empty/invalid input

    # Always clean potential markdown fences first
    cleaned_text = re.sub(r'^```(json|JSON|)\s*|\s*```$', '', response_text, flags=re.MULTILINE).strip()

    # --- Format Handling ---
    if expected_format == "json_score_reason":
        try:
            data = json.loads(cleaned_text)
            if isinstance(data, dict):
                score_val = data.get("score", "")
                reason_val = data.get("reason", "")
                # Check if score looks numeric-like
                if isinstance(score_val, (int, float)) or (isinstance(score_val, str) and re.match(r'^\s*-?\d+(\.\d+)?\s*$', str(score_val).strip())):
                    score = str(score_val).strip()
                else:
                    score = "" # Treat non-numeric score in JSON as empty score
                reason = str(reason_val).strip() # Keep reason regardless of score
            elif isinstance(data, (int, float)) or (isinstance(data, str) and re.match(r'^\s*-?\d+(\.\d+)?\s*$', str(data).strip())):
                 # Handle case where API returned just a number when JSON was expected
                 score = str(data).strip()
                 reason = "" # No reason provided
            else:
                # Valid JSON, but not a dict or number - treat score as missing
                 score = ""
                 reason = cleaned_text # Use the cleaned text as fallback reason
        except json.JSONDecodeError:
            # JSON failed, try regex fallback for score/reason within the expected JSON structure
            score_match = re.search(r'"score"\s*:\s*("?(-?\d+(?:\.\d+)?)"?|\d+)', cleaned_text, re.IGNORECASE | re.DOTALL)
            reason_match = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned_text, re.IGNORECASE | re.DOTALL) # Handle escaped quotes
            if not reason_match:
                reason_match_broad = re.search(r'"reason"\s*:\s*([\s\S]+)', cleaned_text, re.IGNORECASE | re.DOTALL)
                if reason_match_broad:
                    # Try to intelligently capture until the end or a likely closing brace/quote
                    potential_reason = reason_match_broad.group(1).strip().rstrip('"}').strip()
                    reason = potential_reason

            score = score_match.group(2) if score_match else ""
            if not reason and reason_match: # If strict match worked
                 reason = reason_match.group(1).strip()

            # If score still missing, try finding any number as final fallback for this format
            if not score:
                score = _extract_first_number(cleaned_text)
            if not reason: # Use cleaned_text if no reason found
                reason = cleaned_text

    elif expected_format == "integer_score":
        # Try to extract the first number found in the response
        score = _extract_first_number(cleaned_text)
        reason = "" # Reason is not expected for this format

    elif expected_format == "raw_text":
        # The score *is* the raw text, reason is empty
        score = cleaned_text # Use cleaned text as the primary score output
        reason = ""

    else: # Fallback for unknown format
        st.warning(f"Unknown expected_format '{expected_format}'. Treating as raw_text.")
        score = cleaned_text
        reason = ""

    # Ensure score and reason are strings
    return str(score).strip(), str(reason).strip()


def clean_scores(scores: pd.Series) -> list:
    """Clean and standardize score values before numeric conversion."""
    cleaned = []
    # Using .astype(str) and then iterating might be slightly cleaner
    # but iterating directly handles mixed types initially.
    for s in scores:
        if isinstance(s, (int, float)):
            cleaned.append(s) # Already numeric
        elif isinstance(s, str):
            # Remove common text like "score:", whitespace, trailing periods
            s_cleaned = re.sub(r'(?i)score:?\s*', '', s).strip().rstrip('.').strip()
            # Handle potential ranges (e.g., "7-8") - take the first number
            match = re.search(r'^\s*(-?\d+(?:\.\d+)?).*', s_cleaned) # Extract number even if text follows
            if match:
                 cleaned.append(match.group(1))
            else:
                 cleaned.append(s_cleaned) # Keep original if no number found at start
        else:
            cleaned.append(str(s) if s is not None else "") # Convert other types to string
    return cleaned

def try_convert_to_numeric(series: pd.Series, column_name: str = "Scores") -> pd.Series:
    """
    Try to convert a Pandas Series to numeric, providing detailed warnings for failures.

    Args:
        series (pd.Series): The series to convert.
        column_name (str): The name of the column for display in warnings.

    Returns:
        pd.Series: The converted series with non-numeric values as NaN.
    """
    if not isinstance(series, pd.Series):
        st.error(f"Input error for '{column_name}': try_convert_to_numeric expects a Pandas Series.")
        return pd.Series([np.nan] * len(series)) # Return NaNs of appropriate length

    # First, clean the potential score values
    cleaned_list = clean_scores(series) # Returns a list

    # Attempt conversion using the list
    numeric_result = pd.to_numeric(cleaned_list, errors='coerce') # Can return Series or ndarray

    # --- Ensure 'converted' is a Pandas Series ---
    if isinstance(numeric_result, np.ndarray):
        # Convert ndarray back to a Series, preserving the original index and name
        converted = pd.Series(numeric_result, index=series.index, name=series.name)
    elif isinstance(numeric_result, pd.Series):
        # If it's already a Series, ensure index and name match original
        converted = numeric_result
        converted.index = series.index
        converted.name = series.name
    else:
        # Handle unexpected types, perhaps coerce to Series with NaNs
        st.warning(f"Unexpected type returned by pd.to_numeric for '{column_name}': {type(numeric_result)}. Attempting conversion.")
        try:
            converted = pd.Series(numeric_result, index=series.index, name=series.name)
        except Exception: # Broad exception if Series creation fails
             converted = pd.Series([np.nan] * len(series), index=series.index, name=series.name)

    # --- Identify and report errors using the guaranteed Series 'converted' ---
    nan_mask = converted.isna()
    original_failed = series[nan_mask]

    non_numeric_examples = []
    # Iterate through only the values that failed conversion
    for idx, orig_val in original_failed.items():
        # Check if original value was actually non-empty before flagging
        orig_is_empty_str = isinstance(orig_val, str) and not orig_val.strip()
        if not orig_is_empty_str and orig_val is not None:
            # Find the corresponding 'cleaned' value
            try:
                loc = series.index.get_loc(idx) # Get integer position from original index
                clean_val = cleaned_list[loc] if loc < len(cleaned_list) else "[Cleaned value lookup error]"
            except Exception:
                clean_val = "[Cleaned value lookup error]"

            orig_display = str(orig_val)[:100] + ('...' if len(str(orig_val)) > 100 else '')
            clean_display = str(clean_val)[:100] + ('...' if len(str(clean_val)) > 100 else '')
            non_numeric_examples.append(f"Row index '{idx}': Original='{orig_display}', Cleaned='{clean_display}' -> Failed")

    if non_numeric_examples:
        st.warning(f"Could not convert {len(non_numeric_examples)} non-empty values in column '{column_name}' to numeric.")
        # Limit the number of examples shown in the expander
        examples_to_show = non_numeric_examples[:30]
        with st.expander(f"Show unconverted values for '{column_name}' (Max 30 shown)"):
            st.code('\n'.join(examples_to_show), language=None) # Use st.code for better formatting
            if len(non_numeric_examples) > 30:
                 st.caption(f"... and {len(non_numeric_examples) - 30} more.")
        st.info("Non-numeric scores will be excluded from calculations like ICC.")

    return converted # Returns pd.Series
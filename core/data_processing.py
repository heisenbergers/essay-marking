import streamlit as st
import pandas as pd
import time
import json
import re

# Assuming api_handler.py is in the same directory
from .api_handler import chatGPT, get_openai_client # Keep get_openai_client import if used elsewhere, but not directly needed here now
from openai import APIError, AuthenticationError, RateLimitError # Import specific errors if handling them here

# Assuming helpers.py is in ../utils/
from utils.helpers import parse_response_for_default

@st.cache_data # Cache data loading based on file content
def load_data(uploaded_file):
    """Loads data from uploaded CSV or Excel file."""
    # ... (load_data function remains the same) ...
    if uploaded_file is None:
        return None
    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
        elif file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload CSV or Excel.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def process_dataframe(df: pd.DataFrame, context: str, use_default_context: bool, model: str, response_column: str, api_key: str):
    """
    Process the dataframe by sending each response to the GPT API using the provided API key.

    Args:
        df (pd.DataFrame): The dataframe containing responses.
        context (str): The prompt/context for the GPT model.
        use_default_context (bool): Whether the default context format (JSON score/reason) is used.
        model (str): The name of the GPT model to use.
        response_column (str): The name of the column containing responses.
        api_key (str): The user's validated OpenAI API key for the session.

    Returns:
        pd.DataFrame: The dataframe with added GPT scores and potentially reasons.
    """
    if response_column not in df.columns:
        st.error(f"The uploaded file must contain the selected response column: '{response_column}'")
        return df # Return original df if column is missing

    if not api_key:
         st.error("API Key is missing. Cannot process. Please configure in Step 1.")
         # Add error markers to prevent proceeding without key
         df["gpt_score_raw"] = ["ERROR: Missing API Key"] * len(df)
         if use_default_context:
             df["gpt_score"] = ["ERROR"] * len(df)
             df["gpt_reason"] = ["ERROR"] * len(df)
         else:
              df["gpt_score"] = ["ERROR"] * len(df) # Use gpt_score even if not default when erroring
         return df

    # --- Initialize Progress ---
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(df)
    results = [] # Store tuples of (raw_response, score, reason) - might not be needed if updating df directly

    # Create a copy to avoid modifying the original if passed directly from cache
    result_df = df.copy()

    # --- Add result columns if they don't exist ---
    if "gpt_score_raw" not in result_df.columns:
        result_df["gpt_score_raw"] = ""
    if use_default_context:
        if "gpt_score" not in result_df.columns:
            result_df["gpt_score"] = ""
        if "gpt_reason" not in result_df.columns:
            result_df["gpt_reason"] = ""
    else: # If not default context, the main output might go into gpt_score or gpt_score_raw
         if "gpt_score" not in result_df.columns:
             result_df["gpt_score"] = ""
         # Ensure gpt_reason exists even if not used by default parse, might hold errors
         if "gpt_reason" not in result_df.columns:
             result_df["gpt_reason"] = ""

    processed_count = 0
    stop_processing = False # Flag to stop if a critical error occurs

    try:
        for i, row in result_df.iterrows():
            if stop_processing:
                result_df.loc[i, "gpt_score_raw"] = "Skipped: Processing stopped due to critical error."
                result_df.loc[i, "gpt_score"] = "ERROR"
                result_df.loc[i, "gpt_reason"] = "Processing stopped."
                continue # Skip remaining rows

            prompt_text = row[response_column]
            status_text.text(f"Processing item {i+1} of {total}...")

            # Basic check for valid prompt
            if not isinstance(prompt_text, str) or not prompt_text.strip():
                raw_response = "Skipped: Empty prompt"
                score = ""
                reason = ""
            else:
                try:
                    # Call API (retry is handled within chatGPT)
                    # Pass the session's api_key
                    raw_response = chatGPT(prompt_text, context, model=model, api_key=api_key)

                    # Parse response
                    if use_default_context:
                        score, reason = parse_response_for_default(raw_response)
                    else:
                        score = raw_response # The entire response is the 'score'
                        reason = "" # No separate reason field for custom prompts unless parsed

                except ValueError as ve: # Catch specific non-retryable errors from chatGPT
                     error_message = f"ERROR: {ve}"
                     raw_response = error_message
                     score = "ERROR"
                     reason = error_message
                     st.error(f"API Error on item {i+1}: {ve}")
                     # Option to stop processing on critical errors like invalid key or model not found
                     if "Invalid API key" in str(ve) or "Model" in str(ve) and "not found" in str(ve):
                          st.warning("Stopping processing due to critical API error.")
                          stop_processing = True
                     # Allow continuing for rate limit errors (if they somehow get past retry)
                     elif "Rate limit" in str(ve):
                          if not st.checkbox("Continue processing despite rate limit errors?", key=f"continue_error_{i}", value=True):
                              stop_processing = True
                     else: # Other ValueErrors
                          if not st.checkbox("Continue processing despite errors?", key=f"continue_error_{i}", value=True):
                              stop_processing = True
                except APIError as ae: # Catch retryable errors that might have exhausted retries
                     error_message = f"ERROR: API Error after retries - {ae}"
                     raw_response = error_message
                     score = "ERROR"
                     reason = error_message
                     st.error(f"API Error on item {i+1} (retries exhausted): {ae}")
                     if not st.checkbox("Continue processing despite API errors?", key=f"continue_error_{i}", value=True):
                         stop_processing = True
                except Exception as e: # Catch unexpected errors
                     error_message = f"ERROR: Unexpected error - {e}"
                     raw_response = error_message
                     score = "ERROR"
                     reason = error_message
                     st.error(f"Unexpected error on item {i+1}: {e}")
                     if not st.checkbox("Continue processing despite unexpected errors?", key=f"continue_error_{i}", value=True):
                         stop_processing = True

            # Update dataframe directly
            result_df.loc[i, "gpt_score_raw"] = raw_response
            if use_default_context:
                result_df.loc[i, "gpt_score"] = score
                result_df.loc[i, "gpt_reason"] = reason
            else:
                 result_df.loc[i, "gpt_score"] = score # Store raw response as score
                 result_df.loc[i, "gpt_reason"] = reason # Keep reason empty unless error

            processed_count += 1
            # Only update progress if not stopped
            if not stop_processing:
                progress_bar.progress(processed_count / total)

            # Optional: Update session state periodically for intermediate saving feel
            if processed_count % 5 == 0:
                 st.session_state["processed_df"] = result_df.copy()


    except Exception as e:
        status_text.error(f"Processing loop stopped unexpectedly: {e}")
        # Ensure partial results are saved
        st.session_state["processed_df"] = result_df
        return result_df # Return partially processed results
    finally:
        final_message = f"Processing finished. {processed_count} out of {total} items attempted."
        if stop_processing:
             final_message += " Processing stopped early due to critical errors."
        status_text.text(final_message)
        progress_bar.empty() # Remove progress bar after completion or error

    # Final save to session state
    st.session_state["processed_df"] = result_df
    return result_df


def clean_scores(scores):
    """Clean and standardize score values before numeric conversion."""
    # ... (clean_scores function remains the same) ...
    cleaned = []
    for s in scores:
        if isinstance(s, (int, float)):
            cleaned.append(s) # Already numeric
        elif isinstance(s, str):
            # Remove common text like "score:", whitespace, trailing periods
            s_cleaned = re.sub(r'(?i)score:?\s*', '', s).strip().rstrip('.').strip()
            # Handle potential ranges (e.g., "7-8") - take the first number or average?
            # For now, let's try taking the first valid number found
            match = re.search(r'^\s*(\d+(?:\.\d+)?)', s_cleaned)
            if match:
                 cleaned.append(match.group(1))
            else:
                 cleaned.append(s_cleaned) # Keep original if no number found at start
        else:
            cleaned.append(str(s) if s is not None else "") # Convert other types to string
    return cleaned

def try_convert_to_numeric(series, column_name="Scores"):
    """Try to convert a series to numeric, providing detailed warnings."""
    # ... (try_convert_to_numeric function remains the same) ...
     # First, clean the potential score values
    cleaned_series = clean_scores(series)
    original_series = series # Keep original for comparison

    # Attempt conversion
    converted = pd.to_numeric(cleaned_series, errors='coerce')

    # Identify and report errors
    nan_indices = converted.isna()
    original_non_numeric = original_series[nan_indices]
    cleaned_non_numeric = pd.Series(cleaned_series)[nan_indices]

    non_numeric_examples = []
    for i, (orig, clean, conv) in enumerate(zip(original_series, cleaned_series, converted)):
        # Check if original was not empty string or just whitespace before considering it a conversion failure
        if pd.isna(conv) and isinstance(orig, str) and orig.strip():
             non_numeric_examples.append(f"Row {series.index[i]+1}: Original='{orig}', Cleaned='{clean}' -> Failed")
        elif pd.isna(conv) and not isinstance(orig, str) and orig is not None: # Handle non-string, non-None original values
              non_numeric_examples.append(f"Row {series.index[i]+1}: Original='{orig}' (Type: {type(orig).__name__}), Cleaned='{clean}' -> Failed")


    if non_numeric_examples:
        st.warning(f"Could not convert {len(non_numeric_examples)} non-empty values in column '{column_name}' to numeric.")
        with st.expander(f"Show unconverted values for '{column_name}'"):
            st.write(non_numeric_examples[:20]) # Show first 20 examples
            if len(non_numeric_examples) > 20:
                 st.caption("... and more.")
        st.info("Non-numeric scores will be excluded from calculations like ICC.")

    return converted
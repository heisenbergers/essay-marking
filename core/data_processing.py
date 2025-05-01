import streamlit as st
import pandas as pd
import time
import json
import re

# Assuming api_handler.py is in the same directory
from .api_handler import chatGPT, get_openai_client

# Assuming helpers.py is in ../utils/
from utils.helpers import parse_response_for_default

@st.cache_data # Cache data loading based on file content
def load_data(uploaded_file):
    """Loads data from uploaded CSV or Excel file."""
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

def process_dataframe(df: pd.DataFrame, context: str, use_default_context: bool, model: str, response_column: str):
    """
    Process the dataframe by sending each response to the GPT API.

    Args:
        df (pd.DataFrame): The dataframe containing responses.
        context (str): The prompt/context for the GPT model.
        use_default_context (bool): Whether the default context format (JSON score/reason) is used.
        model (str): The name of the GPT model to use.
        response_column (str): The name of the column containing responses.

    Returns:
        pd.DataFrame: The dataframe with added GPT scores and potentially reasons.
    """
    if response_column not in df.columns:
        st.error(f"The uploaded file must contain the selected response column: '{response_column}'")
        return df # Return original df if column is missing

    client = get_openai_client()
    if client is None:
        st.error("OpenAI client could not be initialized. Cannot process.")
        # Return original df to avoid losing data
        df["gpt_score_raw"] = ["ERROR: OpenAI client init failed"] * len(df)
        if use_default_context:
            df["gpt_score"] = ["ERROR"] * len(df)
            df["gpt_reason"] = ["ERROR"] * len(df)
        else:
             df["gpt_score"] = ["ERROR"] * len(df)
        return df

    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(df)
    results = [] # Store tuples of (raw_response, score, reason)

    # Create a copy to avoid modifying the original if passed directly from cache
    result_df = df.copy()

    # Add result columns if they don't exist
    if "gpt_score_raw" not in result_df.columns:
        result_df["gpt_score_raw"] = ""
    if use_default_context:
        if "gpt_score" not in result_df.columns:
            result_df["gpt_score"] = ""
        if "gpt_reason" not in result_df.columns:
            result_df["gpt_reason"] = ""
    else:
         if "gpt_score" not in result_df.columns: # If not default, raw score goes here
             result_df["gpt_score"] = ""

    processed_count = 0
    try:
        for i, row in result_df.iterrows():
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
                    raw_response = chatGPT(prompt_text, context, model=model, client=client)

                    # Parse response
                    if use_default_context:
                        score, reason = parse_response_for_default(raw_response)
                    else:
                        score = raw_response # The entire response is the 'score'
                        reason = "" # No separate reason field for custom prompts unless parsed

                except ValueError as ve: # Catch specific errors from chatGPT
                     raw_response = f"ERROR: {ve}"
                     score = "ERROR"
                     reason = f"ERROR: {ve}"
                     st.error(f"API Error on item {i+1}: {ve}")
                     # Option to stop processing or continue
                     if not st.checkbox("Continue processing despite errors?", key=f"continue_error_{i}", value=True):
                         raise # Re-raise to stop the loop
                except Exception as e:
                     raw_response = f"ERROR: Unexpected error - {e}"
                     score = "ERROR"
                     reason = f"ERROR: {e}"
                     st.error(f"Unexpected error on item {i+1}: {e}")
                     if not st.checkbox("Continue processing despite errors?", key=f"continue_error_{i}", value=True):
                         raise # Re-raise to stop the loop

            # Update dataframe directly
            result_df.loc[i, "gpt_score_raw"] = raw_response
            if use_default_context:
                result_df.loc[i, "gpt_score"] = score
                result_df.loc[i, "gpt_reason"] = reason
            else:
                 result_df.loc[i, "gpt_score"] = score # Store raw response as score

            processed_count += 1
            progress_bar.progress(processed_count / total)

            # Optional: Update session state periodically for intermediate saving feel
            if processed_count % 5 == 0:
                 st.session_state["processed_df"] = result_df.copy()


    except Exception as e:
        status_text.error(f"Processing stopped due to error: {e}")
        # Ensure partial results are saved
        st.session_state["processed_df"] = result_df
        return result_df # Return partially processed results
    finally:
        status_text.text(f"Processing finished. {processed_count} items processed.")
        progress_bar.empty() # Remove progress bar after completion or error

    # Final save to session state
    st.session_state["processed_df"] = result_df
    return result_df


def clean_scores(scores):
    """Clean and standardize score values before numeric conversion."""
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
        if pd.isna(conv) and str(orig).strip(): # Check if original was not empty
             non_numeric_examples.append(f"Row {series.index[i]+1}: Original='{orig}', Cleaned='{clean}' -> Failed")

    if non_numeric_examples:
        st.warning(f"Could not convert {len(non_numeric_examples)} values in column '{column_name}' to numeric.")
        with st.expander(f"Show unconverted values for '{column_name}'"):
            st.write(non_numeric_examples[:20]) # Show first 20 examples
            if len(non_numeric_examples) > 20:
                 st.caption("... and more.")
        st.info("Non-numeric scores will be excluded from calculations like ICC.")

    return converted
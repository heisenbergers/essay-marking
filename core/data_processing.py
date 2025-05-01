import streamlit as st
import pandas as pd
import time
import json
import re

# --- MODIFIED IMPORTS ---
from .api_handler import get_llm_client # Use the factory
from utils.helpers import parse_llm_response, try_convert_to_numeric, clean_scores
# Import specific exceptions if needed for detailed handling (optional)
# from openai import APIError as OpenAIAPIError, AuthenticationError as OpenAIAuthenticationError
# from anthropic import APIError as AnthropicAPIError, AuthenticationError as AnthropicAuthenticationError
# from google.api_core import exceptions as GoogleAPIErrors
# --- END MODIFIED IMPORTS ---

@st.cache_data # Cache data loading based on file content
def load_data(uploaded_file):
    """Loads data from uploaded CSV or Excel file."""
    if uploaded_file is None: return None
    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension in ["xlsx", "xls"]: df = pd.read_excel(uploaded_file)
        elif file_extension == "csv": df = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload CSV or Excel.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def process_dataframe(
    df: pd.DataFrame,
    context: str,
    expected_format: str,
    provider: str, # Add provider
    model: str,
    response_column: str,
    api_key: str
):
    """
    Process the dataframe by sending each response to the selected LLM provider's API
    and parsing based on the expected_format.
    """
    if response_column not in df.columns:
        st.error(f"The uploaded file must contain the selected response column: '{response_column}'")
        return df

    if not api_key:
         st.error(f"API Key for {provider} is missing. Cannot process. Please configure in Step 1.")
         df["gpt_score_raw"] = [f"ERROR: Missing {provider} API Key"] * len(df)
         df["gpt_score"] = ["ERROR"] * len(df)
         df["gpt_reason"] = ["ERROR"] * len(df)
         return df

    # --- Get LLM Client using Factory ---
    try:
        # Use _key argument for cache_resource based on provider and key
        llm_client = get_llm_client(provider=provider, api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize {provider.capitalize()} client: {e}")
        st.info("Please verify the API key in Step 1.")
        df["gpt_score_raw"] = [f"ERROR: Client Init Failed ({provider})"] * len(df)
        df["gpt_score"] = ["ERROR"] * len(df)
        df["gpt_reason"] = ["ERROR"] * len(df)
        return df
    # --- End Client Init ---

    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(df)
    result_df = df.copy()

    # Ensure result columns exist
    if "gpt_score_raw" not in result_df.columns: result_df["gpt_score_raw"] = pd.NA
    if "gpt_score" not in result_df.columns: result_df["gpt_score"] = pd.NA
    if "gpt_reason" not in result_df.columns: result_df["gpt_reason"] = pd.NA
    # Ensure columns are object type to handle mixed types (str, NA)
    result_df["gpt_score_raw"] = result_df["gpt_score_raw"].astype(object)
    result_df["gpt_score"] = result_df["gpt_score"].astype(object)
    result_df["gpt_reason"] = result_df["gpt_reason"].astype(object)


    processed_count = 0
    error_count = 0
    stop_processing = False

    try:
        for i, row in result_df.iterrows():
            if stop_processing:
                result_df.loc[i, "gpt_score_raw"] = "Skipped: Processing stopped."
                result_df.loc[i, "gpt_score"] = "ERROR"
                result_df.loc[i, "gpt_reason"] = "Processing stopped."
                continue

            prompt_text = row[response_column]
            # Ensure prompt_text is a string before proceeding
            if pd.isna(prompt_text):
                prompt_text = ""
            else:
                prompt_text = str(prompt_text)

            status_text.text(f"Processing item {i+1} of {total} using {provider}/{model}...")

            raw_response = ""
            score = ""
            reason = ""

            if not prompt_text.strip():
                raw_response = "Skipped: Empty prompt"
                # score, reason remain ""
            else:
                try:
                    # Call the generate method on the obtained client instance
                    raw_response = llm_client.generate(
                        prompt=prompt_text,
                        context=context,
                        model_name=model
                    )

                    # Parse response using the selected format
                    score, reason = parse_llm_response(raw_response, expected_format)

                except ValueError as ve: # Specific errors raised by our clients/parsing/api_handler
                     error_message = f"ERROR ({provider}): {ve}"
                     raw_response = error_message
                     score = "ERROR"
                     reason = error_message
                     st.error(f"Error on item {i+1}: {ve}")
                     error_count += 1
                     # Stop on critical config errors
                     if "Invalid API key" in str(ve) or "Model" in str(ve) and "not found" in str(ve) or "Failed to configure" in str(ve) or "Permission Denied" in str(ve):
                          st.warning(f"Stopping processing due to critical {provider} configuration error.")
                          stop_processing = True
                     elif "Rate limit" in str(ve):
                          if not st.checkbox(f"Continue processing despite {provider} rate limit errors?", key=f"continue_error_{i}", value=True): stop_processing = True
                     else: # Other ValueErrors
                          if not st.checkbox(f"Continue processing despite {provider} errors?", key=f"continue_error_{i}", value=True): stop_processing = True
                except Exception as e: # Catch potential API errors from different SDKs or unexpected issues
                     error_type = type(e).__name__
                     error_message = f"ERROR ({provider} - {error_type}): {e}"
                     raw_response = error_message
                     score = "ERROR"
                     reason = error_message
                     st.error(f"Unexpected error on item {i+1} ({provider}): {e}")
                     error_count += 1
                     if not st.checkbox(f"Continue processing despite {provider} errors ({error_type})?", key=f"continue_error_{i}", value=True):
                         stop_processing = True

            # Update dataframe (use .loc to avoid SettingWithCopyWarning)
            result_df.loc[i, "gpt_score_raw"] = raw_response
            result_df.loc[i, "gpt_score"] = score
            result_df.loc[i, "gpt_reason"] = reason

            processed_count += 1
            if not stop_processing: progress_bar.progress(processed_count / total)
            # Periodically update session state (consider doing less frequently for large datasets)
            if processed_count % 20 == 0: st.session_state["processed_df"] = result_df.copy()

    except Exception as e:
        status_text.error(f"Processing loop stopped unexpectedly: {e}")
    finally:
        # Final message and cleanup
        final_message = f"Processing finished. {processed_count} out of {total} items attempted."
        if error_count > 0: final_message += f" Encountered {error_count} errors."
        if stop_processing: final_message += " Processing stopped early due to critical errors."
        status_text.text(final_message)
        progress_bar.empty()

    # Final save to session state
    st.session_state["processed_df"] = result_df
    return result_df
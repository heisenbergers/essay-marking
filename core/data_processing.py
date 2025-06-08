import streamlit as st
import pandas as pd
import time
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Dict, Any, List, Optional # Added Optional
import logging # Optional: for better logging

# --- Tenacity for Retries ---
from tenacity import retry, stop_after_attempt, wait_random_exponential, RetryError

# --- Local Imports ---
from .api_handler import get_llm_client, LLMClient
from utils.helpers import parse_llm_response

# --- Optional: Configure Logging ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

@st.cache_data # Cache data loading based on file content
def load_data(uploaded_file):
    """Loads data from uploaded CSV or Excel file."""
    if uploaded_file is None: return None
    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
        elif file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload CSV or Excel.")
            return None
        # Basic cleaning: remove unnamed columns often created by Excel exports
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        # logger.error(f"Error reading file: {e}", exc_info=True)
        return None

# --- Custom Exception for Retryable Parsing Failures ---
class ParsingRetryableError(Exception):
    """Custom exception to signal a retry due to parsing failure."""
    def __init__(self, message, raw_response_snippet=""):
        super().__init__(message)
        self.raw_response_snippet = raw_response_snippet

# --- Helper function for a single API call and parse attempt (with tenacity retry) ---
@retry(
    stop=stop_after_attempt(3),  # Retry up to 2 additional times (total 3 attempts)
    wait=wait_random_exponential(multiplier=1, max=10), # Wait 1s, then 2s, up to 10s
    retry=lambda retry_state: isinstance(retry_state.outcome.exception(), ParsingRetryableError),
    reraise=True # Reraise the ParsingRetryableError if all attempts fail
)
def _attempt_api_call_and_parse(
    llm_client: LLMClient,
    prompt_text: str,
    context: str,
    model: str,
    expected_format: str,
    index_for_log: str = "" # Optional: for more detailed logging if needed
) -> Tuple[str, str, str]:
    """
    Makes one attempt to call the LLM API and parse the response.
    Raises ParsingRetryableError if parsing fails based on expected_format and a retry is desired.
    Returns (raw_response, score, reason).
    """
    # logger.info(f"Index {index_for_log}: Attempting API call. Format: '{expected_format}'. Prompt: '{prompt_text[:50]}...'")

    raw_response_content = llm_client.generate(
        prompt=prompt_text,
        context=context,
        model_name=model
        # Potentially pass other kwargs like temperature if they are added to process_dataframe
    )

    # Check for direct error messages from llm_client.generate()
    # These are not parsing failures of an API response but errors from the client (e.g., safety block).
    # These should not trigger a ParsingRetryableError.
    if isinstance(raw_response_content, str) and \
       (raw_response_content.startswith("ERROR:") or raw_response_content.startswith("Skipped:")):
        # Return this error directly; it will be handled by _process_single_row as a final state.
        return raw_response_content, "ERROR", raw_response_content

    parsed_score, parsed_reason = parse_llm_response(raw_response_content, expected_format)

    # Determine if parsing truly failed for retry purposes
    parsing_failed_for_retry = False
    if expected_format in ["json_score_reason", "integer_score"]:
        if not parsed_score.strip(): # If score is empty or only whitespace
            parsing_failed_for_retry = True
    elif expected_format == "raw_text":
        if not parsed_score.strip(): # If raw_text output (which is the score) is empty
            parsing_failed_for_retry = True
            # Note: For raw_text, if an API legitimately returns an empty string for a valid prompt,
            # this will still trigger retries. This behavior might need refinement if it becomes an issue.

    if parsing_failed_for_retry:
        # logger.warning(f"Index {index_for_log}: Parsing failed for '{expected_format}'. Raw: '{raw_response_content[:100]}...'. Raising to retry.")
        raise ParsingRetryableError(
            f"Parsing failed for format '{expected_format}'.",
            raw_response_snippet=raw_response_content[:200] # Store a snippet for logging if all retries fail
        )

    return raw_response_content, parsed_score, parsed_reason


def _process_single_row(
    index: Any,
    row_data: pd.Series,
    response_column: str,
    context: str,
    provider: str, # Added provider to check for deepseek
    model: str,
    expected_format: str,
    llm_client: LLMClient # Pass the initialized client
) -> Tuple[Any, str, str, str, Optional[str]]: # Added chain_of_thought string
    """Processes a single row, making the API call (with retries on parsing failure) and parsing."""
    prompt_text = row_data.get(response_column, "")

    # Ensure prompt_text is a string before proceeding
    if pd.isna(prompt_text) or not isinstance(prompt_text, str):
        prompt_text = ""

    # Handle empty prompt text directly - no API call or retry needed
    if not prompt_text.strip():
        return index, "Skipped: Empty prompt", "", "", None # score, reason, and chain_of_thought are empty/None

    final_raw_response = "Not Processed" # Default value if processing fails catastrophically
    final_score = "ERROR"
    final_reason = "Processing did not complete due to an unexpected issue."
    final_chain_of_thought: Optional[str] = None


    try:
        # _attempt_api_call_and_parse is now decorated with tenacity's @retry
        # It will handle retries for ParsingRetryableError internally
        final_raw_response, final_score, final_reason = _attempt_api_call_and_parse(
            llm_client, prompt_text, context, model, expected_format, str(index)
        )
        # If DeepSeek Reasoner, try to get reasoning content
        if provider == "deepseek" and model == "deepseek-reasoner":
            final_chain_of_thought = llm_client.get_last_reasoning_content()


    except RetryError as e: # This catches ParsingRetryableError after all retries are exhausted
        # logger.error(f"Index {index}: All retries failed for parsing. Last raw response snippet: '{e.last_attempt.exception().raw_response_snippet}'. Original error: {str(e.last_attempt.exception())}")
        final_raw_response = f"ERROR (Parsing - All Retries Failed): {e.last_attempt.exception().raw_response_snippet if hasattr(e.last_attempt.exception(), 'raw_response_snippet') else str(e.last_attempt.exception())}"
        final_score = "ERROR"
        final_reason = final_raw_response # Use the detailed error as the reason
        final_chain_of_thought = None
    except ValueError as ve: # Catches non-retryable errors from llm_client.generate (e.g., auth, model not found)
                            # or errors from _attempt_api_call_and_parse if they are not ParsingRetryableError.
        # logger.warning(f"Index {index}: ValueError during processing: {ve}")
        final_raw_response = f"ERROR ({llm_client.__class__.__name__} or API Call): {ve}"
        final_score = "ERROR"
        final_reason = final_raw_response
        final_chain_of_thought = None
    except Exception as e: # Catches any other unexpected errors during the process.
        # logger.error(f"Index {index}: Unexpected error during processing: {e}", exc_info=True)
        error_type = type(e).__name__
        final_raw_response = f"ERROR (Unexpected - {error_type}): {e}"
        final_score = "ERROR"
        final_reason = final_raw_response
        final_chain_of_thought = None

    return index, final_raw_response, final_score, final_reason, final_chain_of_thought


def process_dataframe(
    df: pd.DataFrame,
    context: str,
    expected_format: str,
    provider: str,
    model: str,
    response_column: str,
    api_key: str,
    max_workers: int = 10, # Number of parallel threads (configurable)
    stop_on_error_threshold: int = -1 # Stop if this many errors occur (-1 for never)
) -> pd.DataFrame:
    """
    Process the dataframe by sending responses to the LLM API in parallel and parsing based on the expected_format.
    Includes retry logic for parsing failures of API responses.
    """
    if response_column not in df.columns:
        st.error(f"The uploaded file must contain the selected response column: '{response_column}'")
        return df
    if not api_key:
        st.error(f"API Key for {provider} is missing. Cannot process. Please configure in Step 1.")
        # Add error columns immediately
        df["gpt_score_raw"] = f"ERROR: Missing {provider} API Key"
        df["gpt_score"] = "ERROR"
        df["gpt_reason"] = f"ERROR: Missing {provider} API Key"
        df["chain_of_thought"] = f"ERROR: Missing {provider} API Key"
        return df

    # --- Get LLM Client using Factory ---
    try:
        # Use _key argument for cache_resource based on provider and key
        llm_client = get_llm_client(provider=provider, api_key=api_key)
    except Exception as e: # Error already shown by get_llm_client, just add columns and return
        st.info("Processing cannot proceed due to client initialization failure.")
        df["gpt_score_raw"] = f"ERROR: Client init failed for {provider}"
        df["gpt_score"] = "ERROR"
        df["gpt_reason"] = f"ERROR: Client init failed for {provider}"
        df["chain_of_thought"] = f"ERROR: Client init failed for {provider}"
        return df


    result_df = df.copy()
    # Initialize new columns with pd.NA for proper dtype handling later
    result_df["gpt_score_raw"] = pd.NA
    result_df["gpt_score"] = pd.NA
    result_df["gpt_reason"] = pd.NA
    result_df["chain_of_thought"] = pd.NA # New column

    total = len(df)
    if total == 0:
        st.warning("Input DataFrame is empty. Nothing to process.")
        return result_df

    progress_bar = st.progress(0.0)
    status_text = st.empty() # Placeholder for dynamic text updates
    error_count = 0
    processed_count = 0

    results: Dict[Any, Dict[str, Optional[str]]] = {} # Store results keyed by original index, allow None for CoT

    start_time = time.time()
    status_text.text(f"Starting processing for {total} items using {provider}/{model}...")

    # Use ThreadPoolExecutor for parallel API calls
    actual_max_workers = min(max_workers, total) if total > 0 else 1
    if actual_max_workers != max_workers:
        st.info(f"Using {actual_max_workers} parallel workers (adjusted from {max_workers}).")

    futures = []
    try:
        with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
            # Submit all tasks
            for index, row in df.iterrows():
                futures.append(executor.submit(
                    _process_single_row, # This now calls the retry-enabled helper
                    index=index,
                    row_data=row,
                    response_column=response_column,
                    context=context,
                    provider=provider, # Pass provider
                    model=model,
                    expected_format=expected_format,
                    llm_client=llm_client
                ))

            # Process completed futures as they finish
            for future in as_completed(futures):
                try:
                    idx, raw, score, reason, cot = future.result()
                    results[idx] = {"gpt_score_raw": raw, "gpt_score": score, "gpt_reason": reason, "chain_of_thought": cot}

                    # Increment error count if result indicates an error
                    if score == "ERROR" or (isinstance(raw, str) and raw.startswith("ERROR:")):
                        error_count += 1

                    # Check if error threshold is reached
                    if stop_on_error_threshold >= 0 and error_count >= stop_on_error_threshold: # Corrected condition
                        st.warning(f"Stopping processing: Error threshold ({stop_on_error_threshold}) reached with {error_count} errors.")
                        # Cancel remaining futures if possible (Python 3.9+)
                        # for f in futures:
                        # if not f.done():
                        # f.cancel()
                        # executor.shutdown(wait=False, cancel_futures=True) # Preferred for 3.9+
                        break # Exit the loop processing completed futures

                except Exception as exc: # This might catch errors from future.result() if a future was cancelled
                                         # or if an exception propagated from _process_single_row that wasn't handled there
                    st.error(f"Error retrieving result from a processing task: {exc}")
                    # logger.error(f"Error retrieving result from future: {exc}", exc_info=True)
                    error_count += 1
                    # Since we don't have 'idx' here, we can't easily mark a specific row.
                    # This indicates a more systemic issue with the future handling itself.

                processed_count += 1
                progress = processed_count / total if total > 0 else 1.0
                progress_bar.progress(progress)
                elapsed_time = time.time() - start_time
                est_total_time = (elapsed_time / processed_count * total) if processed_count > 0 else 0
                remaining_time = est_total_time - elapsed_time if est_total_time > elapsed_time else 0.0

                status_text.text(
                    f"Processed: {processed_count}/{total} ({progress:.1%}) | "
                    f"Errors: {error_count} | "
                    f"Elapsed: {elapsed_time:.1f}s | "
                    f"Est. Remain: {remaining_time:.1f}s"
                )
                if processed_count % 20 == 0 or processed_count == total: # Update less frequently
                    temp_df_for_state = result_df.copy()
                    for r_idx, res_data in results.items():
                        if r_idx in temp_df_for_state.index:
                            temp_df_for_state.loc[r_idx, ["gpt_score_raw", "gpt_score", "gpt_reason", "chain_of_thought"]] = res_data.values()
                    st.session_state["processed_df"] = temp_df_for_state
                    # logger.info(f"Interim update to session_state['processed_df'] at {processed_count} items.")


    except Exception as e: # Error with ThreadPoolExecutor itself or managing futures
        st.error(f"Processing loop encountered a major error: {e}")
        # logger.error(f"ThreadPoolExecutor or future management error: {e}", exc_info=True)
    finally:
        progress_bar.progress(1.0)
        final_message_parts = [f"Processing attempt finished. {processed_count}/{total} items were processed."]
        if error_count > 0:
            final_message_parts.append(f"Encountered {error_count} errors during processing.")
        if stop_on_error_threshold >= 0 and error_count >= stop_on_error_threshold:
            final_message_parts.append("Processing was stopped early due to exceeding the error threshold.")

        status_text.info(" ".join(final_message_parts)) # Use info or success based on outcome
        # logger.info(" ".join(final_message_parts))


    # --- Populate DataFrame with all results collected ---
    # Create series from the results dictionary for efficient update
    raw_scores_series = pd.Series({idx: data["gpt_score_raw"] for idx, data in results.items()}, name="gpt_score_raw")
    scores_series = pd.Series({idx: data["gpt_score"] for idx, data in results.items()}, name="gpt_score")
    reasons_series = pd.Series({idx: data["gpt_reason"] for idx, data in results.items()}, name="gpt_reason")
    chain_of_thoughts_series = pd.Series({idx: data.get("chain_of_thought") for idx, data in results.items()}, name="chain_of_thought")


    # Update the result_df. Ensure index alignment.
    result_df["gpt_score_raw"] = result_df["gpt_score_raw"].astype(object)
    result_df["gpt_score"] = result_df["gpt_score"].astype(object)
    result_df["gpt_reason"] = result_df["gpt_reason"].astype(object)
    result_df["chain_of_thought"] = result_df["chain_of_thought"].astype(object) # Ensure object type for CoT

    result_df.update(raw_scores_series)
    result_df.update(scores_series)
    result_df.update(reasons_series)
    result_df.update(chain_of_thoughts_series)


    # Fill any rows that might not have been processed if loop was exited early
    # These would retain their initial pd.NA or overwritten if processing started for them but failed to complete
    # Marking explicitly if not in 'results' dict
    for col_name in ["gpt_score_raw", "gpt_score", "gpt_reason", "chain_of_thought"]:
        mask_not_processed = ~result_df.index.isin(results.keys())
        if col_name in result_df.columns:
             result_df.loc[mask_not_processed, col_name] = result_df.loc[mask_not_processed, col_name].fillna("Skipped/Not Processed")
        else: # Should not happen if columns initialized correctly
             result_df[col_name] = pd.NA
             result_df.loc[mask_not_processed, col_name] = "Skipped/Not Processed"


    st.session_state["processed_df"] = result_df
    st.session_state["processing_complete"] = True
    return result_df
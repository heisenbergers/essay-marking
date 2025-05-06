import streamlit as st
import pandas as pd
import time
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Dict, Any, List
import logging # Optional: for better logging

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

def _process_single_row(
    index: Any,
    row_data: pd.Series,
    response_column: str,
    context: str,
    model: str,
    expected_format: str,
    llm_client: LLMClient # Pass the initialized client
) -> Tuple[Any, str, str, str]:
    """Processes a single row, making the API call and parsing."""
    prompt_text = row_data.get(response_column, "")
    # Ensure prompt_text is a string before proceeding
    if pd.isna(prompt_text) or not isinstance(prompt_text, str):
        prompt_text = ""

    raw_response = "Skipped: Error in input data"
    score = "ERROR"
    reason = "Input data error"

    if not prompt_text.strip():
        raw_response = "Skipped: Empty prompt"
        score = "" # Treat as missing score
        reason = "" # No reason for empty prompt
    else:
        try:
            # Call the generate method on the obtained client instance
            raw_response = llm_client.generate(
                prompt=prompt_text,
                context=context,
                model_name=model
                # Add other kwargs like temperature if needed, passed from process_dataframe
            )

            # Check if raw_response indicates an error returned from generate() itself
            if isinstance(raw_response, str) and raw_response.startswith("ERROR:") or raw_response.startswith("Skipped:"):
                 score = "ERROR"
                 reason = raw_response # Keep the error message as the reason
            else:
                 # Parse response using the selected format only if generate didn't return an error string
                 score, reason = parse_llm_response(raw_response, expected_format)

        except ValueError as ve: # Specific errors raised by our clients/parsing/api_handler
             error_message = f"ERROR ({llm_client.__class__.__name__}): {ve}"
             raw_response = error_message
             score = "ERROR"
             reason = error_message
             # logger.warning(f"ValueError processing index {index}: {ve}")
        except Exception as e: # Catch other unexpected issues during generation/parsing
             error_type = type(e).__name__
             error_message = f"ERROR (Unexpected - {error_type}): {e}"
             raw_response = error_message
             score = "ERROR"
             reason = error_message
             # logger.error(f"Unexpected error processing index {index}: {e}", exc_info=True)

    return index, raw_response, score, reason


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
    Process the dataframe by sending responses to the LLM API in parallel
    and parsing based on the expected_format.
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
         return df

    # --- Get LLM Client using Factory ---
    try:
        # Use _key argument for cache_resource based on provider and key
        llm_client = get_llm_client(provider=provider, api_key=api_key)
    except Exception as e:
        # Error already shown by get_llm_client, just add columns and return
        st.info("Processing cannot proceed due to client initialization failure.")
        df["gpt_score_raw"] = f"ERROR: Client Init Failed ({provider}) - {e}"
        df["gpt_score"] = "ERROR"
        df["gpt_reason"] = f"Client Init Failed: {e}"
        return df

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    total = len(df)
    result_df = df.copy() # Start with a copy

    # Initialize result columns if they don't exist, ensuring object dtype
    for col in ["gpt_score_raw", "gpt_score", "gpt_reason"]:
        if col not in result_df.columns:
            result_df[col] = pd.NA
        result_df[col] = result_df[col].astype(object)

    processed_count = 0
    error_count = 0
    results: Dict[Any, Dict[str, str]] = {} # Store results keyed by original index

    start_time = time.time()
    status_text.text(f"Starting processing for {total} items using {provider}/{model}...")

    # Use ThreadPoolExecutor for parallel API calls
    # Adjust max_workers based on API rate limits and system resources
    actual_max_workers = min(max_workers, total) if total > 0 else 1
    if actual_max_workers != max_workers:
         st.info(f"Using {actual_max_workers} parallel workers (adjusted from {max_workers}).")

    futures = []
    try:
        with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
            # Submit all tasks
            for index, row in df.iterrows():
                futures.append(executor.submit(
                    _process_single_row,
                    index=index,
                    row_data=row,
                    response_column=response_column,
                    context=context,
                    model=model,
                    expected_format=expected_format,
                    llm_client=llm_client
                ))

            # Process completed futures as they finish
            for future in as_completed(futures):
                try:
                    idx, raw, score, reason = future.result()
                    results[idx] = {"gpt_score_raw": raw, "gpt_score": score, "gpt_reason": reason}

                    # Increment error count if result indicates an error
                    if score == "ERROR" or (isinstance(raw, str) and raw.startswith("ERROR:")):
                         error_count += 1
                         # Check if error threshold is reached
                         if stop_on_error_threshold > 0 and error_count >= stop_on_error_threshold:
                              st.warning(f"Stopping processing: Error threshold ({stop_on_error_threshold}) reached.")
                              # Optionally cancel remaining futures (can be complex/unreliable)
                              # executor.shutdown(wait=False, cancel_futures=True) # Requires Python 3.9+
                              break # Exit the loop processing completed futures

                except Exception as exc:
                     # Handle exceptions from the _process_single_row function itself
                     # This should ideally not happen if _process_single_row catches its errors
                     st.error(f"Internal error processing future: {exc}")
                     # logger.error(f"Internal error processing future: {exc}", exc_info=True)
                     error_count += 1
                     # We don't know the index here easily without modifying the future handling

                processed_count += 1
                progress = processed_count / total
                progress_bar.progress(progress)
                elapsed_time = time.time() - start_time
                est_total_time = (elapsed_time / processed_count * total) if processed_count > 0 else 0
                remaining_time = est_total_time - elapsed_time
                status_text.text(
                    f"Processed: {processed_count}/{total} ({progress:.1%}) | "
                    f"Errors: {error_count} | "
                    f"Elapsed: {elapsed_time:.1f}s | "
                    f"Est. Remain: {remaining_time:.1f}s"
                )

                # Update session state less frequently or maybe not at all during loop?
                # For responsiveness, update periodically but maybe less often.
                if processed_count % 50 == 0 or processed_count == total:
                     # Update the result_df with current results before saving to state
                     temp_df = result_df.copy()
                     for idx, res_data in results.items():
                         if idx in temp_df.index:
                            temp_df.loc[idx, ["gpt_score_raw", "gpt_score", "gpt_reason"]] = res_data.values()
                     st.session_state["processed_df"] = temp_df
                     # logger.info(f"Updated session state at {processed_count} items.")

    except Exception as e:
        st.error(f"Processing loop stopped unexpectedly: {e}")
        # logger.error(f"Processing loop stopped unexpectedly: {e}", exc_info=True)
    finally:
        # Ensure progress bar and status text are updated/cleared correctly
        progress_bar.progress(1.0)
        final_message = f"Processing finished. {processed_count}/{total} items processed."
        if error_count > 0: final_message += f" Encountered {error_count} errors."
        # Add info about any remaining/cancelled tasks if executor was shut down early
        # final_message += " Processing may have been stopped early."
        status_text.success(final_message) # Use success styling
        # logger.info(final_message)

    # --- Populate DataFrame with results ---
    # Ensure all collected results are mapped back correctly
    raw_scores, scores, reasons = [], [], []
    for index in result_df.index:
        if index in results:
            res = results[index]
            raw_scores.append(res.get("gpt_score_raw", pd.NA))
            scores.append(res.get("gpt_score", pd.NA))
            reasons.append(res.get("gpt_reason", pd.NA))
        else:
            # Row was not processed (e.g., stopped early or error retrieving future)
            # Keep existing value or mark as 'Not Processed'
            raw_scores.append(result_df.loc[index, "gpt_score_raw"] if "gpt_score_raw" in result_df.columns and pd.notna(result_df.loc[index, "gpt_score_raw"]) else "Not Processed")
            scores.append(result_df.loc[index, "gpt_score"] if "gpt_score" in result_df.columns and pd.notna(result_df.loc[index, "gpt_score"]) else "ERROR")
            reasons.append(result_df.loc[index, "gpt_reason"] if "gpt_reason" in result_df.columns and pd.notna(result_df.loc[index, "gpt_reason"]) else "Not Processed")

    result_df["gpt_score_raw"] = raw_scores
    result_df["gpt_score"] = scores
    result_df["gpt_reason"] = reasons

    # Final save to session state
    st.session_state["processed_df"] = result_df.copy() # Save the final complete DF
    return result_df
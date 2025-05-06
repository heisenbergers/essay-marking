import streamlit as st
import pandas as pd
from core.data_processing import process_dataframe # Refactored version
from ui.components import download_link
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
from utils.helpers import try_convert_to_numeric # Uses updated version

# --- Constants ---
# Max workers for parallel processing - adjust based on typical API limits / machine resources
DEFAULT_MAX_WORKERS = 10
# Stop processing if more than this % of rows encounter errors
DEFAULT_ERROR_THRESHOLD_PERCENT = 25

def render_process_screen():
    """Render the processing screen UI elements."""
    st.header("2. Process Responses with LLM")

    # --- Retrieve state variables ---
    selected_provider = st.session_state.get("selected_provider")
    provider_key_name = f"{selected_provider}_api_key" if selected_provider else None
    api_key = st.session_state.get(provider_key_name) if provider_key_name else None
    api_verified = st.session_state.get("api_key_verified", {}).get(selected_provider, False)
    response_col = st.session_state.get("response_column")
    raw_df = st.session_state.get("raw_df")
    chosen_model = st.session_state.get("chosen_model")
    context = st.session_state.get("context")
    output_format = st.session_state.get("prompt_output_format")
    processing_complete = st.session_state.get("processing_complete", False)
    processed_df = st.session_state.get("processed_df") # Get potentially partial df

    # --- Verify Prerequisites ---
    if not selected_provider or not api_key or not api_verified:
         st.error(f"API Key for {selected_provider.capitalize() if selected_provider else 'provider'} not verified. Please return to Setup.")
         if st.button("⬅️ Back to Setup", key="proc_back_setup_key"): st.session_state["current_step"] = "setup"; st.rerun()
         st.stop()
    if not response_col or raw_df is None:
        st.error("Setup incomplete (missing data or response column). Please return to Setup.")
        if st.button("⬅️ Back to Setup", key="proc_back_setup_data"): st.session_state["current_step"] = "setup"; st.rerun()
        st.stop()
    if not chosen_model:
        st.error("No model selected. Please return to Setup.")
        if st.button("⬅️ Back to Setup", key="proc_back_setup_model"): st.session_state["current_step"] = "setup"; st.rerun()
        st.stop()
    if not context:
        st.error("No prompt context defined. Please return to Setup.")
        if st.button("⬅️ Back to Setup", key="proc_back_setup_context"): st.session_state["current_step"] = "setup"; st.rerun()
        st.stop()
    if not output_format:
         st.error("Prompt output format not selected. Please return to Setup.")
         if st.button("⬅️ Back to Setup", key="proc_back_setup_format"): st.session_state["current_step"] = "setup"; st.rerun()
         st.stop()

    # --- Display Settings Summary ---
    st.subheader("Processing Configuration")
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    with col_sum1:
        provider_display = selected_provider.capitalize()
        st.metric("Provider / Model", f"{provider_display} / {chosen_model}")
    with col_sum2:
        format_map = {"json_score_reason": "JSON", "integer_score": "Integer", "raw_text": "Raw Text"}
        format_display = format_map.get(output_format, output_format)
        st.metric("Output Format", format_display)
    with col_sum3:
        st.metric("Response Column", response_col)

    total_rows = len(raw_df)
    st.write(f"Dataset has **{total_rows}** rows to process.")

    # --- Processing Control ---
    if not processing_complete:
        st.markdown("---")
        st.subheader("Start Processing")
        provider_display = selected_provider.capitalize()
        st.markdown(f"""
        <div class="info-box" style="border-left-color: #0d6efd; background-color: #59616e; padding: 10px; margin-bottom: 15px;">
            <p style="margin-bottom:0;"><strong>Note:</strong> Processing involves parallel calls to the <strong>{provider_display}</strong> API ({chosen_model}). This may incur costs and take time. Ensure your API key is active and has sufficient quota/rate limits.</p>
        </div>
        """, unsafe_allow_html=True)

        # Options for processing
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
             process_subset = st.checkbox("Process only a subset (first N rows)", value=False, key="process_subset_check")
             subset_size = total_rows # Default to all
             if process_subset:
                 max_subset = total_rows
                 default_subset = min(10, max_subset) if max_subset > 0 else 1
                 if max_subset > 0:
                     subset_size = st.number_input(
                         f"Number of rows to process:",
                         min_value=1, max_value=max_subset, value=default_subset, step=10, key="subset_size_input"
                     )
                 else:
                     st.warning("No rows available to process.")
                     subset_size = 0

        with col_opt2:
             max_workers = st.number_input("Max Parallel Workers", min_value=1, max_value=50, value=DEFAULT_MAX_WORKERS, step=1, key="max_workers_input", help="Number of API calls to make concurrently. Adjust based on API rate limits.")
             error_threshold_percent = st.number_input("Stop if Error % Exceeds", min_value=-1, max_value=100, value=DEFAULT_ERROR_THRESHOLD_PERCENT, step=5, key="error_threshold_input", help="Stop processing if the percentage of rows with errors exceeds this value. Set to -1 to never stop automatically.")

        df_to_process = raw_df.head(subset_size) if process_subset else raw_df
        rows_to_process_count = len(df_to_process)
        stop_threshold_count = int(rows_to_process_count * (error_threshold_percent / 100.0)) if error_threshold_percent >= 0 else -1

        if rows_to_process_count > 0:
            st.info(f"Will process **{rows_to_process_count}** rows using up to **{max_workers}** parallel workers. Will stop if **{stop_threshold_count if stop_threshold_count >= 0 else 'N/A'}** errors occur.")
            start_processing = st.button(f"🚀 Start Processing {rows_to_process_count} Rows", type="primary", use_container_width=True, key="start_processing_btn")

            if start_processing:
                st.session_state["processing_complete"] = False # Ensure flag is reset
                st.session_state["processed_df"] = None # Clear previous results
                st.session_state["analysis_computed"] = False # Reset analysis flag too

                try:
                    # Start processing (now parallel)
                    # Spinner is automatically handled by Streamlit during button execution usually
                    processed_df_result = process_dataframe(
                        df=df_to_process.copy(), # Pass a copy to avoid modifying original raw_df
                        context=context,
                        expected_format=output_format,
                        provider=selected_provider,
                        model=chosen_model,
                        response_column=response_col,
                        api_key=api_key,
                        max_workers=max_workers,
                        stop_on_error_threshold=stop_threshold_count
                    )
                    st.session_state["processed_df"] = processed_df_result # Store results (potentially partial)
                    st.session_state["processing_complete"] = True # Mark as complete

                    # Check for errors after processing finishes
                    error_cols = ["gpt_score_raw", "gpt_score", "gpt_reason"]
                    has_errors = False
                    final_error_count = 0
                    if processed_df_result is not None:
                        # Check specific 'ERROR' marker in score column or error messages
                        error_mask = (processed_df_result['gpt_score'] == "ERROR") | \
                                     processed_df_result['gpt_score_raw'].astype(str).str.startswith("ERROR:") | \
                                     processed_df_result['gpt_reason'].astype(str).str.startswith("ERROR:")
                        final_error_count = error_mask.sum()
                        has_errors = final_error_count > 0

                    if has_errors:
                        st.warning(f"Processing finished with {final_error_count} errors. Check results table and download for details.")
                    else:
                        st.success("✅ Processing completed successfully!")

                    time.sleep(0.5) # Brief pause
                    st.rerun() # Rerun to display results section

                except Exception as e:
                    st.error(f"An unexpected error occurred during processing: {e}")
                    # Store whatever might be in processed_df already
                    st.session_state["processed_df"] = st.session_state.get("processed_df", None)
                    st.session_state["processing_complete"] = True # Mark complete even on failure to show partial results


        else:
            st.warning("No rows selected or available for processing.")

    # --- Display Results ---
    if processing_complete:
        # Use processed_df directly from state now
        if processed_df is not None and not processed_df.empty:
            st.markdown("---")
            st.header("📊 Processing Results")

            # Display errors summary if any occurred
            error_mask = (processed_df['gpt_score'] == "ERROR") | \
                         processed_df['gpt_score_raw'].astype(str).str.startswith("ERROR:") | \
                         processed_df['gpt_reason'].astype(str).str.startswith("ERROR:")
            num_errors = error_mask.sum()
            if num_errors > 0:
                 st.warning(f"Found {num_errors} errors during processing.")
                 with st.expander("Show Rows with Errors"):
                      error_df = processed_df[error_mask]
                      cols_to_show_errors = [response_col, 'gpt_score_raw', 'gpt_score', 'gpt_reason']
                      cols_exist_errors = [c for c in cols_to_show_errors if c in error_df.columns]
                      st.dataframe(error_df[cols_exist_errors], use_container_width=True)


            st.subheader("Results Preview (First 20 Rows Processed)")
            # Determine columns to show based on format and existence
            display_cols = [response_col] # Always show response col if exists
            if "gpt_score_raw" in processed_df.columns: display_cols.append("gpt_score_raw")
            if "gpt_score" in processed_df.columns: display_cols.append("gpt_score")
            if "gpt_reason" in processed_df.columns and output_format == "json_score_reason":
                # Only show reason column if it's expected to contain parsed reasons
                display_cols.append("gpt_reason")

            # Ensure display_cols only contains existing columns and remove duplicates
            display_cols_exist = sorted(list(set(col for col in display_cols if col in processed_df.columns)), key = lambda x: (x!=response_col, x=='gpt_score_raw', x))

            st.dataframe(processed_df[display_cols_exist].head(20), use_container_width=True)

            # --- Basic Score Statistics & Distribution (if applicable) ---
            score_stats_calculated = False
            if 'gpt_score' in processed_df.columns:
                st.subheader("LLM Score Analysis Preview")
                # Use the updated helper function
                temp_numeric_scores = try_convert_to_numeric(processed_df['gpt_score'].copy(), "LLM Score (Preview)") # Pass copy
                numeric_scores = temp_numeric_scores.dropna()

                if not numeric_scores.empty:
                    cols_stats = st.columns(4)
                    try: cols_stats[0].metric("Mean", f"{numeric_scores.mean():.2f}")
                    except Exception: cols_stats[0].metric("Mean", "N/A")
                    try: cols_stats[1].metric("Median", f"{numeric_scores.median():.2f}")
                    except Exception: cols_stats[1].metric("Median", "N/A")
                    try: cols_stats[2].metric("Std Dev", f"{numeric_scores.std():.2f}")
                    except Exception: cols_stats[2].metric("Std Dev", "N/A")
                    try: cols_stats[3].metric("Numeric Count", f"{len(numeric_scores)} / {len(temp_numeric_scores)}")
                    except Exception: cols_stats[3].metric("Numeric Count", "N/A")

                    st.write("Distribution of Parsed Numeric LLM Scores:")
                    try:
                        fig, ax = plt.subplots(figsize=(10, 3)) # Smaller plot
                        sns.histplot(numeric_scores, kde=True, ax=ax, bins=15, stat="density")
                        ax.set_xlabel("Score")
                        ax.set_ylabel("Density")
                        # ax.set_title("Distribution of Parsed LLM Scores") # Title implicit
                        st.pyplot(fig)
                        plt.close(fig) # Close plot
                        score_stats_calculated = True
                    except Exception as plot_e:
                         st.warning(f"Could not generate score distribution plot. Error: {plot_e}")

                else:
                    st.info("No valid numeric scores found in the 'gpt_score' column for statistical preview.")

            if not score_stats_calculated and 'gpt_score' in processed_df.columns:
                 st.info("Numeric score statistics and distribution will be available in the Analysis step after full conversion attempt.")

            # --- Download Results ---
            st.subheader("Download Processed Data")
            st.markdown("Download the full results including original data, LLM raw output, and parsed scores/reasons.")
            try:
                download_button = download_link(
                    processed_df,
                    f"llm_processed_{selected_provider}_{chosen_model}.csv", # More descriptive filename
                    "Download Processed Data as CSV"
                )
                st.markdown(download_button, unsafe_allow_html=True)
            except Exception as dl_e:
                st.error(f"Could not generate download link: {dl_e}")

        else:
            st.warning("Processed data not found or is empty. Please run processing.")

    # --- Navigation ---
    st.divider()
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("⬅️ Back to Setup", use_container_width=True, key="proc_back_setup_nav"):
            st.session_state["current_step"] = "setup"
            st.rerun()
    with col_nav2:
        # Enable Analysis button only if processing is done AND IRR was requested
        irr_requested = st.session_state.get("compute_irr", False)
        # Check if processed_df exists and has results needed for analysis
        analysis_possible = processing_complete and processed_df is not None and not processed_df.empty and 'gpt_score' in processed_df.columns

        can_analyze = analysis_possible and irr_requested
        analysis_disabled_reason = ""
        if not processing_complete: analysis_disabled_reason = "Processing must run first."
        elif processed_df is None or processed_df.empty: analysis_disabled_reason = "No processed data available."
        elif not irr_requested: analysis_disabled_reason = "IRR Comparison was not enabled in Setup."
        elif 'gpt_score' not in processed_df.columns: analysis_disabled_reason = "'gpt_score' column missing from results."


        if st.button("➡️ Proceed to Analysis", use_container_width=True, disabled=not can_analyze, type="primary", help=analysis_disabled_reason if not can_analyze else "Analyze LLM scores against manual scores."):
            if can_analyze:
                 # Ensure necessary data is available before switching
                 if 'gpt_score' in processed_df.columns and st.session_state.get("manual_columns"):
                     st.session_state["current_step"] = "analysis"
                     st.rerun()
                 else:
                     st.error("Cannot proceed to analysis. Required columns ('gpt_score', manual columns) missing.")
            # No else needed as button is disabled if not can_analyze
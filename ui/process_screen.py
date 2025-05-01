import streamlit as st
import pandas as pd
from core.data_processing import process_dataframe
from ui.components import download_link
import matplotlib.pyplot as plt
import seaborn as sns
import time # Import time

def render_process_screen():
    """Render the processing screen UI elements."""
    st.header("2. Process Responses with LLM")

    # --- Retrieve state variables ---
    selected_provider = st.session_state.get("selected_provider")
    api_key_name = f"{selected_provider}_api_key" if selected_provider else None
    api_key = st.session_state.get(api_key_name) if api_key_name else None
    api_verified = st.session_state.get("api_key_verified", {}).get(selected_provider, False)
    response_col = st.session_state.get("response_column")
    raw_df = st.session_state.get("raw_df")
    chosen_model = st.session_state.get("chosen_model")
    context = st.session_state.get("context")
    output_format = st.session_state.get("prompt_output_format")

    # --- Verify Prerequisites ---
    if not selected_provider or not api_key or not api_verified:
         st.error(f"API Key for {selected_provider.capitalize() if selected_provider else 'provider'} not verified. Please go back to Step 1.")
         if st.button("⬅️ Back to Setup", key="proc_back_setup_key"):
             st.session_state["current_step"] = "setup"
             st.rerun()
         st.stop() # Stop rendering further if key not ready

    if not response_col or raw_df is None:
        st.error("Setup not complete (missing data or column selection). Please go back to Step 1.")
        if st.button("⬅️ Back to Setup", key="proc_back_setup_data"):
            st.session_state["current_step"] = "setup"
            st.rerun()
        st.stop()
    if not chosen_model:
        st.error("No model selected. Please go back to Step 1.")
        if st.button("⬅️ Back to Setup", key="proc_back_setup_model"):
            st.session_state["current_step"] = "setup"
            st.rerun()
        st.stop()
    if not context:
        st.error("No prompt context defined. Please go back to Step 1.")
        if st.button("⬅️ Back to Setup", key="proc_back_setup_context"):
            st.session_state["current_step"] = "setup"
            st.rerun()
        st.stop()
    if not output_format:
         st.error("Prompt output format not selected. Please go back to Step 1.")
         if st.button("⬅️ Back to Setup", key="proc_back_setup_format"):
             st.session_state["current_step"] = "setup"
             st.rerun()
         st.stop()


    # --- Display Settings Summary ---
    st.subheader("Processing Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        provider_display = selected_provider.capitalize()
        st.metric("Provider / Model", f"{provider_display} / {chosen_model}")
    with col2:
        format_display = output_format.replace('_', ' ').title()
        st.metric("Output Format", format_display)
    with col3:
        st.metric("Response Column", response_col)

    total_rows = len(raw_df)

    # --- Processing Control ---
    if not st.session_state.get("processing_complete", False):
        st.info(f"Ready to process **{total_rows}** responses using the configured settings.")
        st.markdown(f"""
        <div class="warning-box" style="background-color: #fff3cd; border-left: 6px solid #ffc107; padding: 10px; margin-bottom: 15px;">
            <p style="margin-bottom:0;"><strong>⚠️ Note:</strong> Processing involves calls to the <strong>{provider_display}</strong> API which may incur costs and take time depending on the number of responses and model used. Ensure your API key is active and has sufficient quota.</p>
        </div>
        """, unsafe_allow_html=True)

        # Option to process a subset
        process_all = st.toggle("Process all responses", value=True, key="process_all_toggle")
        subset_size = 10
        df_to_process = raw_df # Default to all

        if not process_all:
             max_subset = min(total_rows, 100)
             default_subset = min(10, max_subset) if max_subset > 0 else 1
             if max_subset > 0:
                 subset_size = st.slider(f"Number of responses to process (max {max_subset}):", min_value=1, max_value=max_subset, value=default_subset)
                 df_to_process = raw_df.head(subset_size).copy()
                 st.write(f"Processing the first {subset_size} responses.")
             else:
                 st.warning("No rows available to process.")
                 st.stop()
        else:
             df_to_process = raw_df.copy()
             st.write(f"Processing all {total_rows} responses.")

        start_processing = st.button("🚀 Start Processing", type="primary", use_container_width=True, key="start_processing_btn")

        if start_processing:
            if df_to_process.empty:
                 st.warning("No data selected for processing.")
            else:
                try:
                    with st.spinner(f"🤖 Calling {provider_display}... Please wait... This may take a while."):
                        processed_df_result = process_dataframe(
                            df=df_to_process,
                            context=context,
                            expected_format=output_format,
                            provider=selected_provider,
                            model=chosen_model,
                            response_column=response_col,
                            api_key=api_key
                        )
                    st.session_state["processed_df"] = processed_df_result # Store results (even partial)
                    st.session_state["processing_complete"] = True # Mark as complete (even if errors occurred)

                    # Check for errors after processing finishes
                    error_cols = ["gpt_score_raw", "gpt_score", "gpt_reason"]
                    has_errors = False
                    if processed_df_result is not None:
                        for col in error_cols:
                            if col in processed_df_result.columns and processed_df_result[col].astype(str).str.contains("ERROR", case=False, na=False).any():
                                has_errors = True
                                break

                    if has_errors:
                        st.warning("Processing finished, but some errors occurred. Check results table and download for details.")
                    else:
                        st.success("✅ Processing completed successfully!")

                    time.sleep(1) # Pause briefly
                    st.rerun() # Rerun to display results section

                except Exception as e:
                    st.error(f"An unexpected error stopped the processing initiation: {e}")
                    # Store whatever might be in processed_df already (if any partial update happened)
                    st.session_state["processed_df"] = st.session_state.get("processed_df", None)


    # --- Display Results ---
    if st.session_state.get("processing_complete", False):
        processed_df = st.session_state.get("processed_df") # Get potentially updated df

        if processed_df is not None and not processed_df.empty:
            st.success("✅ LLM Processing has run.")
            st.subheader("Results Preview")

            # Determine columns to show based on context type
            display_cols = [response_col, "gpt_score_raw"] # Always show response and raw output
            if "gpt_score" in processed_df.columns: display_cols.append("gpt_score")
            if "gpt_reason" in processed_df.columns and output_format == "json_score_reason":
                # Only show reason column if it's expected to contain parsed reasons
                display_cols.append("gpt_reason")

            # Ensure display_cols only contains existing columns and remove duplicates
            display_cols = sorted(list(set(col for col in display_cols if col in processed_df.columns)), key = lambda x: (x!=response_col, x=='gpt_score_raw', x))

            st.dataframe(processed_df[display_cols].head(20), use_container_width=True) # Show more rows

            # --- Basic Score Statistics & Distribution (if applicable) ---
            score_stats_calculated = False
            numeric_score_col = 'gpt_score_numeric' # This column is added during analysis step now

            # Display stats based on 'gpt_score' if possible BEFORE analysis step adds numeric col
            if 'gpt_score' in processed_df.columns:
                try:
                    from core.data_processing import try_convert_to_numeric # Use helper directly
                    temp_numeric_scores = try_convert_to_numeric(processed_df['gpt_score'], "LLM Score (Preview)")
                    numeric_scores = temp_numeric_scores.dropna()

                    if not numeric_scores.empty:
                        st.subheader("LLM Score Statistics Preview (Numeric Only)")
                        cols = st.columns(4)
                        cols[0].metric("Mean", f"{numeric_scores.mean():.2f}")
                        cols[1].metric("Median", f"{numeric_scores.median():.2f}")
                        cols[2].metric("Std Dev", f"{numeric_scores.std():.2f}")
                        cols[3].metric("Count", f"{len(numeric_scores)}")

                        st.subheader("LLM Score Distribution Preview")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        sns.histplot(numeric_scores, kde=True, ax=ax, bins=15)
                        ax.set_xlabel("Score")
                        ax.set_ylabel("Frequency")
                        ax.set_title("Distribution of Parsed LLM Scores")
                        st.pyplot(fig)
                        plt.close(fig) # Close plot
                        score_stats_calculated = True
                except Exception as e:
                    st.info(f"Could not generate score preview statistics from 'gpt_score' column. Error: {e}")


            if not score_stats_calculated:
                 st.info("Numeric score statistics and distribution will be available in the Analysis step after conversion.")


            # --- Download Results ---
            st.subheader("Download Processed Data")
            st.markdown("Download the full results including original data, LLM raw output, and parsed scores/reasons.")
            st.markdown(
                download_link(processed_df, "llm_processed_results.csv"),
                unsafe_allow_html=True
            )

        else:
            st.warning("Processed data not found or is empty in session state.")

    # --- Navigation ---
    st.divider()
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("⬅️ Back to Setup", use_container_width=True, key="proc_back_setup_nav"):
            st.session_state["current_step"] = "setup"
            st.rerun()
    with col_nav2:
        # Enable Analysis button only if processing is done AND IRR was requested
        processing_done = st.session_state.get("processing_complete", False)
        irr_requested = st.session_state.get("compute_irr", False)
        can_analyze = processing_done and irr_requested
        analysis_disabled_reason = ""
        if not processing_done: analysis_disabled_reason = "Processing must be completed first."
        elif not irr_requested: analysis_disabled_reason = "Comparison with manual scores must be enabled in Setup."

        if st.button("➡️ Proceed to Analysis", use_container_width=True, disabled=not can_analyze, type="primary", help=analysis_disabled_reason if not can_analyze else None):
            if can_analyze:
                 st.session_state["current_step"] = "analysis"
                 st.rerun()
            # No else needed as button is disabled
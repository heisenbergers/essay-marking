import streamlit as st
import pandas as pd
from core.data_processing import process_dataframe
from ui.components import download_link
import matplotlib.pyplot as plt
import seaborn as sns

def render_process_screen():
    """Render the processing screen UI elements."""
    st.header("2. Process Responses with GPT")

    # --- Verify Prerequisites ---
    # Check if API key is verified first
    if not st.session_state.get("api_key_verified", False):
         st.error("API Key not verified. Please go back to Step 1 and provide/verify your key.")
         if st.button("⬅️ Back to Setup"):
             st.session_state["current_step"] = "setup"
             st.rerun()
         return

    if not st.session_state.get("response_column") or st.session_state.get("raw_df") is None:
        st.error("Setup not complete (missing data or column selection). Please go back to Step 1.")
        if st.button("⬅️ Back to Setup"):
            st.session_state["current_step"] = "setup"
            st.rerun()
        return
    if not st.session_state.get("chosen_model"):
        st.error("No model selected. Please go back to Step 1.")
        if st.button("⬅️ Back to Setup"):
            st.session_state["current_step"] = "setup"
            st.rerun()
        return
    if not st.session_state.get("context"):
        st.error("No prompt context defined. Please go back to Step 1.")
        if st.button("⬅️ Back to Setup"):
            st.session_state["current_step"] = "setup"
            st.rerun()
        return

    # Get the API key from session state
    user_api_key = st.session_state.get("user_api_key")
    if not user_api_key: # Should be caught by api_key_verified check, but safety first
        st.error("API Key missing from session. Please return to Step 1.")
        st.stop()


    # --- Display Settings Summary ---
    st.subheader("Processing Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", st.session_state.get('chosen_model', 'N/A'))
    with col2:
        prompt_type = "Default" if st.session_state.get('use_default_context') else "Custom"
        st.metric("Prompt Type", prompt_type)
    with col3:
        st.metric("Response Column", st.session_state.get('response_column', 'N/A'))

    raw_df = st.session_state["raw_df"]
    response_col = st.session_state["response_column"]
    total_rows = len(raw_df)

    # --- Processing Control ---
    if not st.session_state.get("processing_complete", False):
        st.info(f"Ready to process **{total_rows}** responses using the configured settings.")
        st.markdown("""
        <div class="warning-box" style="background-color: #fff3cd; border-left: 6px solid #ffc107; padding: 10px; margin-bottom: 15px;">
            <p style="margin-bottom:0;"><strong>⚠️ Note:</strong> Processing involves calls to the OpenAI API which may incur costs and take time depending on the number of responses and model used. Ensure your API key is active and has sufficient quota.</p>
        </div>
        """, unsafe_allow_html=True)

        # Option to process a subset (useful for large files or testing)
        process_all = st.toggle("Process all responses", value=True)
        subset_size = 10 # Default subset size
        if not process_all:
             max_subset = min(total_rows, 100) # Limit subset testing size
             subset_size = st.slider(f"Number of responses to process (max {max_subset}):", min_value=1, max_value=max_subset, value=min(10, max_subset))
             df_to_process = raw_df.head(subset_size).copy()
             st.write(f"Processing the first {subset_size} responses.")
        else:
             df_to_process = raw_df.copy()
             st.write(f"Processing all {total_rows} responses.")

        start_processing = st.button("🚀 Start Processing", type="primary", use_container_width=True)

        if start_processing:
            try:
                with st.spinner("🤖 Calling GPT... Please wait..."):
                    # Pass the user's API key to the processing function
                    processed_df = process_dataframe(
                        df=df_to_process,
                        context=st.session_state["context"],
                        use_default_context=st.session_state["use_default_context"],
                        model=st.session_state["chosen_model"],
                        response_column=response_col,
                        api_key=user_api_key # Pass the key here
                    )
                st.session_state["processed_df"] = processed_df # Store potentially partial results too
                st.session_state["processing_complete"] = True # Mark as complete (even if errors occurred)

                # Check if errors occurred during processing by looking for "ERROR" markers
                error_cols = ["gpt_score_raw", "gpt_score", "gpt_reason"]
                has_errors = False
                for col in error_cols:
                     # Check if column exists and has "ERROR" (case-insensitive)
                     if col in processed_df.columns and processed_df[col].astype(str).str.contains("ERROR", case=False, na=False).any():
                          has_errors = True
                          break

                if has_errors:
                    st.warning("Processing finished, but some errors occurred. Check results table.")
                else:
                    st.success("✅ Processing completed successfully!")

                st.rerun() # Rerun to display results section

            except Exception as e:
                st.error(f"An unexpected error stopped processing: {e}")
                # Keep partial results if any in st.session_state["processed_df"]

    # --- Display Results ---
    # ... (Rest of the display logic remains the same) ...
    if st.session_state.get("processing_complete", False):
        st.success("✅ Processing has run.")
        processed_df = st.session_state.get("processed_df")

        if processed_df is not None:
            st.subheader("Results Preview")

            # Determine columns to show based on context type
            display_cols = [response_col]
            # Always show raw response if exists
            if "gpt_score_raw" in processed_df.columns: display_cols.append("gpt_score_raw")

            if st.session_state["use_default_context"]:
                if "gpt_score" in processed_df.columns: display_cols.append("gpt_score")
                if "gpt_reason" in processed_df.columns: display_cols.append("gpt_reason")
            else: # Custom context - main result is usually in gpt_score_raw, but show gpt_score if it exists too
                 if "gpt_score" in processed_df.columns and "gpt_score" not in display_cols:
                    display_cols.append("gpt_score")

            # Ensure display_cols only contains existing columns
            display_cols = [col for col in display_cols if col in processed_df.columns]

            st.dataframe(processed_df[display_cols].head(10), use_container_width=True)

            # --- Basic Score Statistics & Distribution (if applicable) ---
            score_stats_calculated = False
            numeric_score_col = 'gpt_score_numeric' # Defined in analysis prep or data_processing
            # Ensure numeric conversion happened if not already done (e.g., if analysis step skipped)
            # Check if 'gpt_score' or 'gpt_score_raw' exists before trying conversion
            score_source_col = None
            if 'gpt_score' in processed_df.columns:
                score_source_col = 'gpt_score'
            elif 'gpt_score_raw' in processed_df.columns: # Fallback for custom prompts if gpt_score doesn't exist
                score_source_col = 'gpt_score_raw'

            if score_source_col and numeric_score_col not in processed_df.columns:
                 from core.data_processing import try_convert_to_numeric
                 # Determine which column to convert based on context type (more robust)
                 col_to_convert = 'gpt_score' if st.session_state["use_default_context"] else score_source_col
                 if col_to_convert in processed_df.columns:
                     processed_df[numeric_score_col] = try_convert_to_numeric(processed_df[col_to_convert], "GPT Score")
                     st.session_state["processed_df"] = processed_df # Save back with numeric col

            if numeric_score_col in processed_df.columns:
                 numeric_scores = processed_df[numeric_score_col].dropna()
                 if not numeric_scores.empty:
                    st.subheader("GPT Score Statistics (Numeric Only)")
                    cols = st.columns(4)
                    cols[0].metric("Mean", f"{numeric_scores.mean():.2f}")
                    cols[1].metric("Median", f"{numeric_scores.median():.2f}")
                    cols[2].metric("Std Dev", f"{numeric_scores.std():.2f}")
                    cols[3].metric("Count", f"{len(numeric_scores)}")

                    st.subheader("GPT Score Distribution")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.histplot(numeric_scores, kde=True, ax=ax, bins=15)
                    ax.set_xlabel("Score")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Distribution of GPT Scores")
                    st.pyplot(fig)
                    plt.close(fig) # Close plot to free memory
                    score_stats_calculated = True

            if not score_stats_calculated:
                 st.info("Could not calculate numeric score statistics. Ensure scores can be converted to numbers (conversion attempted from 'gpt_score' or 'gpt_score_raw'). Check error messages during processing or analysis.")


            # --- Download Results ---
            st.subheader("Download Processed Data")
            st.markdown(
                download_link(processed_df, "gpt_processed_results.csv"),
                unsafe_allow_html=True
            )

        else:
            st.warning("Processed data not found in session state.")

    # --- Navigation ---
    st.divider()
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("⬅️ Back to Setup", use_container_width=True):
            st.session_state["current_step"] = "setup"
            st.rerun()
    with col_nav2:
        # Enable Analysis button only if processing is done AND IRR was requested
        can_analyze = st.session_state.get("processing_complete", False) and st.session_state.get("compute_irr", False)
        analysis_disabled_reason = ""
        if not st.session_state.get("processing_complete", False):
             analysis_disabled_reason = "Processing must be completed first."
        elif not st.session_state.get("compute_irr", False):
             analysis_disabled_reason = "Comparison with manual scores must be enabled in Setup."


        if st.button("➡️ Proceed to Analysis", use_container_width=True, disabled=not can_analyze, type="primary", help=analysis_disabled_reason if not can_analyze else None):
            if can_analyze:
                 st.session_state["current_step"] = "analysis"
                 st.rerun()
            # No else needed as button is disabled
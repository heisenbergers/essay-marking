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
    if not st.session_state.get("response_column") or not st.session_state.get("raw_df") is not None:
        st.error("Setup not complete. Please go back to Step 1 and upload data/select columns.")
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
            <p style="margin-bottom:0;"><strong>⚠️ Note:</strong> Processing involves calls to the OpenAI API which may incur costs and take time depending on the number of responses and model used. Ensure your API key (configured in Secrets) is active and has sufficient quota.</p>
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
                    processed_df = process_dataframe(
                        df=df_to_process,
                        context=st.session_state["context"],
                        use_default_context=st.session_state["use_default_context"],
                        model=st.session_state["chosen_model"],
                        response_column=response_col
                    )
                st.session_state["processed_df"] = processed_df # Store potentially partial results too
                st.session_state["processing_complete"] = True # Mark as complete (even if errors occurred)

                # Check if errors occurred during processing by looking for "ERROR" markers
                error_cols = ["gpt_score_raw", "gpt_score", "gpt_reason"]
                has_errors = False
                for col in error_cols:
                     if col in processed_df.columns and processed_df[col].astype(str).str.contains("ERROR").any():
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
    if st.session_state.get("processing_complete", False):
        st.success("✅ Processing has run.")
        processed_df = st.session_state.get("processed_df")

        if processed_df is not None:
            st.subheader("Results Preview")

            # Determine columns to show based on context type
            display_cols = [response_col]
            if st.session_state["use_default_context"]:
                if "gpt_score" in processed_df.columns: display_cols.append("gpt_score")
                if "gpt_reason" in processed_df.columns: display_cols.append("gpt_reason")
                if "gpt_score_raw" in processed_df.columns: display_cols.append("gpt_score_raw")
            else: # Custom context - main result is in gpt_score or gpt_score_raw
                 score_col = "gpt_score" if "gpt_score" in processed_df.columns else "gpt_score_raw"
                 if score_col in processed_df.columns:
                    display_cols.append(score_col)


            st.dataframe(processed_df[display_cols].head(10), use_container_width=True)

            # --- Basic Score Statistics & Distribution (if applicable) ---
            score_stats_calculated = False
            numeric_score_col = 'gpt_score_numeric' # Defined in analysis prep or data_processing
            # Ensure numeric conversion happened if not already done (e.g., if analysis step skipped)
            if 'gpt_score' in processed_df.columns and numeric_score_col not in processed_df.columns:
                 from core.data_processing import try_convert_to_numeric
                 score_col_to_convert = 'gpt_score'
                 if not st.session_state["use_default_context"]:
                     score_col_to_convert = 'gpt_score_raw' if 'gpt_score_raw' in processed_df.columns else 'gpt_score'

                 if score_col_to_convert in processed_df.columns:
                     processed_df[numeric_score_col] = try_convert_to_numeric(processed_df[score_col_to_convert], "GPT Score")
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
                    score_stats_calculated = True

            if not score_stats_calculated:
                 st.info("Could not calculate score statistics. Ensure scores are numeric (this may happen automatically in the Analysis step).")


            # --- Download Results ---
            st.subheader("Download Processed Data")
            st.markdown(
                download_link(processed_df, "gpt_processed_results.csv"),
                unsafe_allow_html=True
            )

            # --- Option to Edit (Consider removing/simplifying for Cloud) ---
            # Edits might be lost easily on Streamlit Cloud. Download/re-upload is safer.
            # with st.expander("Edit Results (Experimental)", expanded=False):
            #     st.warning("Edits made here are temporary for this session.")
            #     edited_df = st.data_editor(processed_df.copy(), use_container_width=True, num_rows="dynamic")
            #     if st.button("Apply Edits Temporarily"):
            #         st.session_state["processed_df"] = edited_df
            #         st.success("Temporary edits applied. Download the data to save them permanently.")
            #         st.rerun()

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
        if st.button("➡️ Proceed to Analysis", use_container_width=True, disabled=not can_analyze, type="primary"):
            if can_analyze:
                 st.session_state["current_step"] = "analysis"
                 st.rerun()
            else:
                 st.info("Analysis requires processing to be complete and 'Compare with manual scores' to be enabled in Setup.")
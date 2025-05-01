import streamlit as st
import pandas as pd
from core.analysis import perform_icc_computation, interpret_icc, create_icc_visualization
from ui.components import download_link
import matplotlib.pyplot as plt # Keep imports for potential direct plotting if needed
import seaborn as sns
from scipy import stats
import traceback # For displaying errors

def render_analysis_screen():
    """Render the analysis screen UI elements."""
    st.header("3. Analyze Results (Inter-Rater Reliability)")

    # --- Prerequisites Check ---
    if not st.session_state.get("processing_complete", False):
        st.error("Processing not complete. Please go back to Step 2.")
        if st.button("⬅️ Back to Processing", key="analysis_back_proc_err"): st.session_state["current_step"] = "process"; st.rerun()
        st.stop()
    if not st.session_state.get("compute_irr", False):
        st.warning("IRR comparison was not enabled in Setup (Step 1).")
        st.info("Enable 'Compare with manual scores?' in Step 1 and re-process if you want to perform IRR analysis.")
        if st.button("⬅️ Back to Setup", key="analysis_back_setup_noirr"): st.session_state["current_step"] = "setup"; st.rerun()
        st.stop()
    if not st.session_state.get("manual_columns"):
        st.error("No manual score columns selected in Setup (Step 1). Cannot perform IRR.")
        if st.button("⬅️ Back to Setup", key="analysis_back_setup_nocols"): st.session_state["current_step"] = "setup"; st.rerun()
        st.stop()

    processed_df = st.session_state.get("processed_df")
    if processed_df is None or processed_df.empty:
        st.error("Processed data is missing or empty. Please go back to Step 2.")
        if st.button("⬅️ Back to Processing", key="analysis_back_proc_nodata"): st.session_state["current_step"] = "process"; st.rerun()
        st.stop()

    # --- Always Show Download Link for Processed Data ---
    try:
        st.subheader("Download Processed Data")
        st.markdown("Download the results including raw LLM outputs, parsed scores/reasons, and original data.")
        st.markdown(
            download_link(processed_df, "llm_processed_results.csv", "Download Processed CSV"),
            unsafe_allow_html=True
        )
        st.divider()
    except Exception as e:
        st.warning(f"Could not generate download link for processed data: {e}")
    # --- End Download Section ---

    response_column = st.session_state.get("response_column", "N/A") # Provide default
    manual_columns = st.session_state.get("manual_columns", [])

    st.info(f"Comparing LLM scores with manual scores from column(s): **{', '.join(manual_columns)}**")

    # --- Data Alignment Verification ---
    st.subheader("Verify Data Alignment")
    st.markdown("Ensure the LLM scores correspond to the correct manual scores within the processed data.")
    try:
        with st.expander("Show Sample Data for Verification"):
             cols_to_show = [response_column] if response_column in processed_df.columns else []
             if 'gpt_score_raw' in processed_df.columns: cols_to_show.append('gpt_score_raw')
             if 'gpt_score' in processed_df.columns: cols_to_show.append('gpt_score')
             if 'gpt_reason' in processed_df.columns: cols_to_show.append('gpt_reason')
             # Add only existing manual columns
             cols_to_show.extend([mc for mc in manual_columns if mc in processed_df.columns])
             # Ensure uniqueness and existence
             cols_to_show = sorted(list(set(col for col in cols_to_show if col in processed_df.columns)), key = lambda x: (x!=response_column, x))
             st.dataframe(processed_df[cols_to_show].head(), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not display sample data for verification: {e}")

    alignment_verified = st.checkbox(
        "✓ I confirm the data seems correctly aligned.",
        value=st.session_state.get("alignment_verified", False),
        key="alignment_checkbox"
    )
    st.session_state["alignment_verified"] = alignment_verified
    if not alignment_verified:
        st.warning("Please verify the data alignment before computing reliability.")
        st.stop()


    # --- Compute ICC ---
    st.subheader("Compute Reliability")
    compute_btn = st.button("📈 Compute ICC(3,1)", type="primary", use_container_width=True, key="compute_icc_btn")

    # Perform computation if button clicked OR if results already exist in state
    # Get values from state or set to None
    icc_value = st.session_state.get("icc_value", None)
    analysis_computed = st.session_state.get("analysis_computed", False)
    analyzed_df = st.session_state.get("analyzed_df", None) # DF with numeric cols
    ratings_matrix_df = st.session_state.get("ratings_matrix_for_viz", None) # Filtered DF for ICC

    if compute_btn:
        analysis_computed = False # Reset flag on button press
        icc_value = None
        analyzed_df = None
        ratings_matrix_df = None
        with st.spinner("Calculating ICC and preparing analysis..."):
            try:
                # Pass the current processed_df
                icc_value_result, df_with_numeric, ratings_df_result = perform_icc_computation(
                    df=processed_df.copy(), # Pass a copy
                    response_column=response_column,
                    manual_columns=manual_columns
                )
                # Store results in session state
                st.session_state["icc_value"] = icc_value_result
                st.session_state["analyzed_df"] = df_with_numeric # Store df with numeric cols added
                st.session_state["ratings_matrix_for_viz"] = ratings_df_result # Store filtered df for viz
                st.session_state["analysis_computed"] = True # Flag that analysis ran
                # Update local variables for immediate display
                icc_value = icc_value_result
                analyzed_df = df_with_numeric
                ratings_matrix_df = ratings_df_result
                analysis_computed = True
            except Exception as e:
                 st.error(f"Failed during ICC computation process: {e}")
                 st.expander("Show Computation Error Traceback").error(traceback.format_exc())
                 st.session_state["analysis_computed"] = True # Mark as computed, even if failed
                 st.session_state["icc_value"] = None # Ensure ICC is None on failure


    # --- Display Results (Conditional) ---
    if analysis_computed:
        st.markdown(interpret_icc(icc_value), unsafe_allow_html=True) # Show interpretation regardless of value

        # Check if ICC calculation was successful *and* we have the necessary data for plots
        if icc_value is not None and not pd.isna(icc_value) and ratings_matrix_df is not None and not ratings_matrix_df.empty:
            try:
                gpt_numeric_col = 'gpt_score_numeric'
                if gpt_numeric_col not in ratings_matrix_df.columns:
                     raise ValueError("'gpt_score_numeric' column missing from analysis data.")

                gpt_ratings_numeric = ratings_matrix_df[gpt_numeric_col]
                numeric_manual_cols = [col for col in ratings_matrix_df.columns if col != gpt_numeric_col]

                manual_ratings_for_viz = None
                manual_label = "Manual Score"
                if len(numeric_manual_cols) > 1:
                    manual_ratings_for_viz = ratings_matrix_df[numeric_manual_cols].mean(axis=1)
                    manual_label = "Average Manual Score"
                elif len(numeric_manual_cols) == 1:
                    manual_ratings_for_viz = ratings_matrix_df[numeric_manual_cols[0]]
                    # Derive original name carefully
                    original_col_name = numeric_manual_cols[0].replace('_numeric', '')
                    manual_label = original_col_name if original_col_name in manual_columns else "Manual Score"
                else:
                     st.warning("Could not identify manual score column(s) in the final ratings data for plotting.")

                if manual_ratings_for_viz is not None:
                    st.subheader("Visual Analysis")
                    try:
                        viz_buffer = create_icc_visualization(manual_ratings_for_viz, gpt_ratings_numeric, icc_value)
                        if viz_buffer:
                            st.image(viz_buffer)
                        # No else needed, create_icc_visualization handles internal warnings
                    except Exception as e:
                        st.warning(f"An error occurred during visualization generation: {e}")
                        st.expander("Show Visualization Error Traceback").error(traceback.format_exc())

                    # --- Detailed Analysis Tabs ---
                    st.subheader("Detailed Analysis")
                    tab1, tab2, tab3 = st.tabs(["Statistics", "Score Comparison", "Recommendations"])

                    with tab1:
                        st.markdown("#### Key Metrics")
                        cols_metrics = st.columns(3)
                        cols_metrics[0].metric("ICC(3,1)", f"{icc_value:.3f}")
                        try:
                             # Ensure no NaNs before correlation calculation
                             combined = pd.DataFrame({'manual': manual_ratings_for_viz, 'llm': gpt_ratings_numeric}).dropna()
                             if len(combined) >= 2:
                                 corr, p_val = stats.pearsonr(combined['manual'], combined['llm'])
                                 cols_metrics[1].metric("Pearson Correlation", f"{corr:.3f}", help=f"p-value: {p_val:.3g}")
                             else:
                                 cols_metrics[1].metric("Pearson Correlation", "N/A", help="Not enough data points.")
                        except Exception as e:
                             cols_metrics[1].metric("Pearson Correlation", "Error")
                             st.caption(f"Corr Error: {e}")
                        cols_metrics[2].metric("Analyzed Subjects", f"{len(gpt_ratings_numeric)}")

                        st.markdown("#### Agreement within Threshold")
                        try:
                             diffs = abs(gpt_ratings_numeric - manual_ratings_for_viz)
                             if len(diffs) > 0:
                                 agree_within_1 = (diffs <= 1).mean() * 100
                                 agree_within_2 = (diffs <= 2).mean() * 100
                             else: agree_within_1, agree_within_2 = 0, 0
                             cols_agree = st.columns(2)
                             cols_agree[0].metric("Agreement ≤ 1 point", f"{agree_within_1:.1f}%")
                             cols_agree[1].metric("Agreement ≤ 2 points", f"{agree_within_2:.1f}%")
                        except Exception as e:
                             st.warning(f"Could not calculate agreement percentages: {e}")

                        st.markdown("#### Rating Statistics (for analyzed subjects)")
                        col_stats1, col_stats2 = st.columns(2)
                        with col_stats1:
                             st.write(f"**{manual_label}**")
                             try: st.dataframe(manual_ratings_for_viz.describe().to_frame().T, use_container_width=True)
                             except Exception: st.caption("Could not display stats.")
                        with col_stats2:
                             st.write("**LLM Scores**")
                             try: st.dataframe(gpt_ratings_numeric.describe().to_frame().T, use_container_width=True)
                             except Exception: st.caption("Could not display stats.")

                    with tab2:
                        try:
                            st.markdown("#### Score Distributions")
                            fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
                            sns.histplot(manual_ratings_for_viz, alpha=0.5, label=manual_label, kde=True, ax=ax_dist, color='blue')
                            sns.histplot(gpt_ratings_numeric, alpha=0.5, label="LLM", kde=True, ax=ax_dist, color='orange')
                            ax_dist.set_xlabel("Score")
                            ax_dist.set_ylabel("Density") # Using Density from KDE
                            ax_dist.set_title("Distribution Comparison")
                            ax_dist.legend()
                            st.pyplot(fig_dist)
                            plt.close(fig_dist) # Close plot

                            st.markdown("#### Residual Analysis (LLM - Manual)")
                            residuals = gpt_ratings_numeric - manual_ratings_for_viz
                            fig_res, (ax_res1, ax_res2) = plt.subplots(1, 2, figsize=(12, 5))
                            sns.scatterplot(x=manual_ratings_for_viz, y=residuals, alpha=0.6, ax=ax_res1)
                            ax_res1.axhline(y=0, color='r', linestyle='--')
                            ax_res1.set_xlabel(manual_label)
                            ax_res1.set_ylabel("Residual (LLM - Manual)")
                            ax_res1.set_title("Residual Plot")
                            ax_res1.grid(True, linestyle=':', alpha=0.6)
                            sns.histplot(residuals, kde=True, ax=ax_res2)
                            ax_res2.set_xlabel("Residual Value")
                            ax_res2.set_ylabel("Frequency")
                            ax_res2.set_title("Residual Distribution")
                            ax_res2.grid(True, linestyle=':', alpha=0.6)
                            plt.tight_layout()
                            st.pyplot(fig_res)
                            plt.close(fig_res) # Close plot
                        except Exception as e:
                            st.warning(f"Could not generate comparison plots: {e}")
                            st.expander("Show Plotting Error Traceback").error(traceback.format_exc())

                    with tab3:
                         # Recommendations based on ICC value - should be safe
                         st.markdown("#### Recommendations Based on ICC(3,1)")
                         # Use appropriate box styles from components.py if defined, otherwise simple markdown
                         rec_style_map = {
                             "Poor": "error-box", "Moderate": "warning-box",
                             "Good": "success-box", "Excellent": "info-box"
                         }
                         rec_border_map = {
                             "Poor": "#dc3545", "Moderate": "#ffc107",
                             "Good": "#198754", "Excellent": "#0d6efd"
                         }
                         rec_cat = ""
                         rec_text = ""
                         if icc_value < 0.5: rec_cat = "Poor"; rec_text = """<strong>Low Reliability (Poor):</strong> Consistency is low. Consider:<ul><li>Refining the prompt (clearer instructions, examples).</li><li>Reviewing manual scores for consistency.</li><li>Trying a different LLM model/provider.</li><li>Checking data quality.</li></ul>"""
                         elif icc_value < 0.75: rec_cat = "Moderate"; rec_text = """<strong>Moderate Reliability:</strong> Some consistency, room for improvement. Consider:<ul><li>Prompt tuning, focusing on areas of disagreement (see Residual Analysis).</li><li>Reviewing edge cases where scores diverge significantly.</li><li>Enhancing the scoring rubric within the prompt.</li></ul>"""
                         elif icc_value < 0.9: rec_cat = "Good"; rec_text = """<strong>Good Reliability:</strong> Good consistency. Often acceptable. Minor prompt tuning might further improve alignment if desired."""
                         else: rec_cat = "Excellent"; rec_text = """<strong>Excellent Reliability:</strong> High consistency. The current configuration seems effective. Consider saving the prompt and model settings."""

                         rec_style = rec_style_map.get(rec_cat, "info-box")
                         rec_border = rec_border_map.get(rec_cat, "gray")
                         st.markdown(f"<div class='{rec_style}' style='border-left-color: {rec_border};'>{rec_text}</div>", unsafe_allow_html=True)


                else: # Manual ratings couldn't be prepared
                     st.warning("Could not prepare manual ratings for detailed analysis plots.")

            except Exception as detail_e: # Catch errors during detailed analysis prep
                 st.error(f"An error occurred preparing data for detailed analysis: {detail_e}")
                 st.expander("Show Detail Prep Error Traceback").error(traceback.format_exc())

        # --- Download Link for Analysis Data (DF with numeric columns) ---
        # Offer download if analysis was computed, even if ICC failed but numeric cols were added
        if analyzed_df is not None:
            st.divider()
            st.subheader("Download Analysis Data")
            st.markdown("Includes original data, LLM outputs, and added numeric columns used for analysis.")
            try:
                st.markdown(
                    download_link(analyzed_df, "llm_analysis_results.csv", "Download Analysis CSV"),
                    unsafe_allow_html=True
                )
            except Exception as dl_e:
                st.warning(f"Could not generate analysis download link: {dl_e}")
        # --- End Analysis Download Link ---

        elif icc_value is None or pd.isna(icc_value): # ICC failed specifically
            st.warning("ICC calculation did not yield a valid result. Detailed analysis and plots cannot be displayed. Check warnings above or download the *processed* data for details.")
        else: # Other data prep failure (e.g., ratings_matrix_df empty)
             st.warning("Could not prepare necessary data for detailed analysis and plots. Download the *processed* data for details.")


    # --- Navigation ---
    st.divider()
    if st.button("⬅️ Back to Processing", use_container_width=True, key="analysis_back_proc_nav"):
        st.session_state["current_step"] = "process"
        # Keep analysis state so user can see results if they come back
        st.rerun()
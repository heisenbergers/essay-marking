import streamlit as st
import pandas as pd
from core.analysis import perform_icc_computation, interpret_icc, create_icc_visualization
from ui.components import download_link
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import traceback
import numpy as np # Ensure numpy is imported

def render_analysis_screen():
    """Render the analysis screen UI elements."""
    st.header("3. Analyze Results (Inter-Rater Reliability)")

    # --- Retrieve state ---
    processing_complete = st.session_state.get("processing_complete", False)
    compute_irr = st.session_state.get("compute_irr", False)
    manual_columns = st.session_state.get("manual_columns", [])
    processed_df = st.session_state.get("processed_df")
    response_column = st.session_state.get("response_column", "N/A")

    # --- Prerequisites Check ---
    if not processing_complete:
        st.error("Processing not complete. Please go back to Step 2.")
        if st.button("⬅️ Back to Processing", key="analysis_back_proc_err1"): st.session_state["current_step"] = "process"; st.rerun()
        st.stop()
    if processed_df is None or processed_df.empty:
        st.error("Processed data is missing or empty. Please run processing in Step 2.")
        if st.button("⬅️ Back to Processing", key="analysis_back_proc_nodata"): st.session_state["current_step"] = "process"; st.rerun()
        st.stop()
    if 'gpt_score' not in processed_df.columns:
         st.error("The required 'gpt_score' column is missing from the processed data. Cannot perform analysis. Please check processing step/results.")
         if st.button("⬅️ Back to Processing", key="analysis_back_proc_nogptscore"): st.session_state["current_step"] = "process"; st.rerun()
         st.stop()
    if not compute_irr:
        st.warning("IRR comparison was not enabled in Setup (Step 1).")
        st.info("Enable 'Compare with manual scores?' in Step 1 and re-process if you want to perform IRR analysis.")
        if st.button("⬅️ Back to Setup", key="analysis_back_setup_noirr"): st.session_state["current_step"] = "setup"; st.rerun()
        st.stop()
    if not manual_columns:
        st.error("No manual score columns selected in Setup (Step 1). Cannot perform IRR.")
        if st.button("⬅️ Back to Setup", key="analysis_back_setup_nocols"): st.session_state["current_step"] = "setup"; st.rerun()
        st.stop()


    # --- Display Info ---
    st.info(f"Comparing LLM scores ('gpt_score') with manual scores from: **{', '.join(manual_columns)}**")


    # --- Data Alignment Verification ---
    # Use a separate flag for this specific verification step
    alignment_step_verified = st.session_state.get("alignment_step_verified", False)
    if not alignment_step_verified:
        st.subheader("Verify Data Alignment")
        st.markdown("Ensure the LLM scores correspond to the correct manual scores within the processed data.")
        try:
            with st.expander("Show Sample Data for Verification (First 10 Rows)"):
                 cols_to_show = [response_column] if response_column in processed_df.columns else []
                 if 'gpt_score' in processed_df.columns: cols_to_show.append('gpt_score')
                 # Add only existing manual columns
                 cols_to_show.extend([mc for mc in manual_columns if mc in processed_df.columns])
                 # Ensure uniqueness and existence
                 cols_to_show_exist = sorted(list(set(col for col in cols_to_show if col in processed_df.columns)), key = lambda x: (x!=response_column, x=='gpt_score', x))
                 st.dataframe(processed_df[cols_to_show_exist].head(10), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display sample data for verification: {e}")

        # Checkbox to proceed
        if st.checkbox("✓ I confirm the data seems correctly aligned.", key="alignment_confirm_check"):
            st.session_state["alignment_step_verified"] = True
            st.rerun() # Rerun to show the compute button
        else:
            st.warning("Please verify the data alignment before computing reliability.")
            st.stop() # Stop further rendering until verified


    # --- Compute ICC ---
    st.subheader("Compute Reliability")
    st.markdown("Click the button below to calculate the Intraclass Correlation Coefficient (ICC) comparing the numeric LLM scores ('gpt_score') against the selected manual score column(s).")

    compute_btn = st.button("📈 Compute ICC(3,1) and Analyze", type="primary", use_container_width=True, key="compute_icc_btn")

    # --- Get analysis results from state (if already computed) ---
    analysis_computed = st.session_state.get("analysis_computed", False)
    icc_value = st.session_state.get("icc_value", None)
    analyzed_df = st.session_state.get("analyzed_df", None) # DF with numeric cols
    ratings_matrix_df = st.session_state.get("ratings_matrix_for_viz", None) # Filtered DF for ICC

    if compute_btn:
        # Reset previous analysis state on new computation
        analysis_computed = False
        icc_value = None
        analyzed_df = None
        ratings_matrix_df = None
        st.session_state["analysis_computed"] = False
        st.session_state["icc_value"] = None
        st.session_state["analyzed_df"] = None
        st.session_state["ratings_matrix_for_viz"] = None

        with st.spinner("Converting scores, calculating ICC, and preparing analysis..."):
            try:
                # Pass the current processed_df
                icc_value_result, df_with_numeric, ratings_df_result = perform_icc_computation(
                    df=processed_df.copy(), # Pass a copy
                    manual_columns=manual_columns
                    # response_column is not needed by perform_icc_computation
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
                st.success("Analysis computation complete.")
            except Exception as e:
                 st.error(f"Failed during ICC computation process: {e}")
                 st.expander("Show Computation Error Traceback").error(traceback.format_exc())
                 # Still mark as computed to prevent re-running automatically on rerun
                 st.session_state["analysis_computed"] = True
                 st.session_state["icc_value"] = None # Ensure ICC is None on failure


    # --- Display Results (Conditional on analysis_computed) ---
    if analysis_computed:
        st.markdown("---")
        st.header("📊 Analysis Results")

        # Show interpretation regardless of value (handles None/NaN)
        st.markdown(interpret_icc(icc_value), unsafe_allow_html=True)

        # --- Detailed Analysis (only if ICC calculation succeeded) ---
        if icc_value is not None and not pd.isna(icc_value) and ratings_matrix_df is not None and not ratings_matrix_df.empty:
            try:
                gpt_numeric_col = 'gpt_score_numeric'
                if gpt_numeric_col not in ratings_matrix_df.columns:
                     raise ValueError("'gpt_score_numeric' column missing from final analysis data.")

                gpt_ratings_numeric = ratings_matrix_df[gpt_numeric_col]
                # Identify numeric manual columns used in the ratings_matrix_df
                numeric_manual_cols_in_matrix = [col for col in ratings_matrix_df.columns if col != gpt_numeric_col]

                manual_ratings_for_viz = None
                manual_label = "Manual Score"
                if len(numeric_manual_cols_in_matrix) > 1:
                    # Calculate row-wise mean for average manual score
                    manual_ratings_for_viz = ratings_matrix_df[numeric_manual_cols_in_matrix].mean(axis=1)
                    manual_label = "Average Manual Score"
                    st.caption(f"Using average of columns: {', '.join(numeric_manual_cols_in_matrix)}")
                elif len(numeric_manual_cols_in_matrix) == 1:
                    manual_col_name = numeric_manual_cols_in_matrix[0]
                    manual_ratings_for_viz = ratings_matrix_df[manual_col_name]
                    # Try to derive original name more reliably
                    original_col_name = manual_col_name.replace('_numeric', '')
                    manual_label = original_col_name if original_col_name in manual_columns else "Manual Score"
                else:
                     st.warning("Could not identify manual score column(s) in the final ratings data for plotting.")

                # --- Proceed only if we have both manual and LLM numeric scores ---
                if manual_ratings_for_viz is not None and not manual_ratings_for_viz.empty:
                    st.subheader("Visual Analysis")
                    try:
                        # Pass Series directly to visualization function
                        viz_buffer = create_icc_visualization(manual_ratings_for_viz, gpt_ratings_numeric, icc_value)
                        if viz_buffer:
                            st.image(viz_buffer, caption="Manual vs LLM Ratings & Bland-Altman Plot")
                        else:
                             st.warning("Visualization could not be generated.")
                        # No need to close figure here, create_icc_visualization handles it
                    except Exception as e:
                        st.warning(f"An error occurred during visualization generation: {e}")
                        st.expander("Show Visualization Error Traceback").error(traceback.format_exc())

                    # --- Detailed Analysis Tabs ---
                    st.subheader("Detailed Analysis")
                    tab1, tab2, tab3 = st.tabs(["Statistics", "Score Comparison Plots", "Recommendations"])

                    with tab1:
                        st.markdown("#### Key Metrics")
                        cols_metrics = st.columns(3)
                        cols_metrics[0].metric("ICC(3,1)", f"{icc_value:.3f}")
                        try:
                             # Ensure no NaNs before correlation calculation (should be clean already)
                             if len(manual_ratings_for_viz) >= 2 and len(gpt_ratings_numeric) == len(manual_ratings_for_viz):
                                 corr, p_val = stats.pearsonr(manual_ratings_for_viz, gpt_ratings_numeric)
                                 cols_metrics[1].metric("Pearson Correlation", f"{corr:.3f}", help=f"p-value: {p_val:.3g}")
                             else:
                                 cols_metrics[1].metric("Pearson Correlation", "N/A", help="Not enough paired data points.")
                        except Exception as e:
                             cols_metrics[1].metric("Pearson Correlation", "Error")
                             st.caption(f"Corr Error: {e}")
                        cols_metrics[2].metric("Analyzed Subjects", f"{len(gpt_ratings_numeric)}", help="Number of rows with valid numeric scores for both LLM and all manual columns.")

                        st.markdown("#### Agreement within Threshold")
                        try:
                             diffs = abs(gpt_ratings_numeric - manual_ratings_for_viz)
                             if len(diffs) > 0:
                                 agree_within_0 = (diffs == 0).mean() * 100 # Exact agreement
                                 agree_within_1 = (diffs <= 1).mean() * 100 # Agreement within 1 point
                                 agree_within_2 = (diffs <= 2).mean() * 100
                             else: agree_within_0, agree_within_1, agree_within_2 = 0, 0, 0
                             cols_agree = st.columns(3)
                             cols_agree[0].metric("Exact Agreement", f"{agree_within_0:.1f}%")
                             cols_agree[1].metric("Agreement ≤ 1 point", f"{agree_within_1:.1f}%")
                             cols_agree[2].metric("Agreement ≤ 2 points", f"{agree_within_2:.1f}%")
                        except Exception as e:
                             st.warning(f"Could not calculate agreement percentages: {e}")

                        st.markdown("#### Rating Statistics (for analyzed subjects)")
                        try:
                            stats_df = pd.DataFrame({
                                manual_label: manual_ratings_for_viz.describe(),
                                "LLM Score": gpt_ratings_numeric.describe()
                            })
                            st.dataframe(stats_df.T, use_container_width=True) # Transpose for better layout
                        except Exception as e:
                            st.caption(f"Could not display descriptive statistics: {e}")


                    with tab2:
                        try:
                            st.markdown("#### Score Distributions")
                            fig_dist, ax_dist = plt.subplots(figsize=(10, 4)) # Slightly smaller
                            sns.histplot(manual_ratings_for_viz, alpha=0.6, label=manual_label, kde=True, ax=ax_dist, color='blue', stat="density", bins=15)
                            sns.histplot(gpt_ratings_numeric, alpha=0.6, label="LLM Score", kde=True, ax=ax_dist, color='orange', stat="density", bins=15)
                            ax_dist.set_xlabel("Score")
                            ax_dist.set_ylabel("Density")
                            ax_dist.set_title("Score Distribution Comparison")
                            ax_dist.legend()
                            st.pyplot(fig_dist)
                            plt.close(fig_dist) # Close plot

                            st.markdown("#### Residual Analysis (LLM - Manual)")
                            residuals = gpt_ratings_numeric - manual_ratings_for_viz
                            fig_res, ax_res = plt.subplots(figsize=(10, 4)) # Single plot might be clearer
                            sns.residplot(x=manual_ratings_for_viz, y=gpt_ratings_numeric, lowess=True, ax=ax_res, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red', 'lw': 1})
                            # sns.scatterplot(x=manual_ratings_for_viz, y=residuals, alpha=0.6, ax=ax_res)
                            # ax_res.axhline(y=0, color='r', linestyle='--')
                            ax_res.set_xlabel(manual_label)
                            ax_res.set_ylabel("Residual (LLM - Manual)")
                            ax_res.set_title("Residual Plot vs. Manual Score")
                            ax_res.grid(True, linestyle=':', alpha=0.6)
                            st.pyplot(fig_res)
                            plt.close(fig_res) # Close plot
                        except Exception as e:
                            st.warning(f"Could not generate comparison plots: {e}")
                            st.expander("Show Plotting Error Traceback").error(traceback.format_exc())

                    with tab3:
                         st.markdown("#### Recommendations Based on ICC(3,1)")
                         rec_style_map = {
                             "Poor": "error", "Moderate": "warning",
                             "Good": "success", "Excellent": "info"
                         }
                         rec_text = ""
                         rec_type = "info" # Default
                         if icc_value < 0.5: rec_type="Poor"; rec_text = """**Low Reliability:** Consistency is low. Strongly consider:
                                                                        * Refining the prompt (clearer instructions, examples, scoring criteria).
                                                                        * Reviewing manual scores for consistency issues (are human raters agreeing?).
                                                                        * Trying a different LLM model/provider (some may follow instructions better).
                                                                        * Checking data quality and if the task is suitable for current LLM capabilities.
                                                                        """
                         elif icc_value < 0.75: rec_type="Moderate"; rec_text = """**Moderate Reliability:** Some consistency, but significant room for improvement. Consider:
                                                                                * Prompt tuning, focusing on areas of disagreement identified in Residual Analysis.
                                                                                * Reviewing edge cases where scores diverge significantly.
                                                                                * Enhancing the scoring rubric details within the prompt.
                                                                                * Checking if LLM exhibits specific biases (e.g., always scoring high/low).
                                                                                """
                         elif icc_value < 0.9: rec_type="Good"; rec_text = """**Good Reliability:** Acceptable consistency for many use cases. Minor prompt tuning or model adjustments might improve alignment further if needed. Focus on understanding any remaining systematic differences shown in plots."""
                         else: rec_type="Excellent"; rec_text = """**Excellent Reliability:** High consistency between LLM and manual scores. The current configuration seems effective. Consider saving the prompt and model settings as a reliable baseline."""

                         # Use Streamlit alert types
                         alert_func = getattr(st, rec_style_map.get(rec_type, "info"))
                         alert_func(body=rec_text, icon="💡" if rec_type in ["Good", "Excellent"] else ("⚠️" if rec_type=="Moderate" else "🔥"))


                else: # Manual ratings couldn't be prepared or were empty
                     st.warning("Manual ratings could not be prepared for detailed analysis plots (check numeric conversion warnings).")

            except Exception as detail_e: # Catch errors during detailed analysis prep/plotting
                 st.error(f"An error occurred during detailed analysis generation: {detail_e}")
                 st.expander("Show Detail Prep Error Traceback").error(traceback.format_exc())

        # --- Fallback Messages if Analysis Failed ---
        elif icc_value is None or pd.isna(icc_value):
            st.warning("ICC calculation failed or yielded an invalid result (NaN). Detailed analysis cannot be displayed.")
            st.info("Check warnings during the 'Compute ICC' step above, specifically regarding numeric conversion or insufficient valid data points.")
        else: # Other data prep failure (e.g., ratings_matrix_df empty)
             st.warning("Could not prepare necessary data (e.g., valid ratings matrix) for detailed analysis and plots.")

        # --- Download Link for Analysis Data ---
        # Offer download if analysis was computed, even if ICC failed but numeric cols were added
        if analyzed_df is not None:
            st.divider()
            st.subheader("Download Analysis Data")
            st.markdown("Includes original data, LLM outputs, and added numeric columns used for analysis (rows with NaN scores might be excluded from ICC calculation but included here).")
            try:
                 download_button_analysis = download_link(
                     analyzed_df,
                     "llm_analysis_results_with_numeric.csv",
                     "Download Analysis Data CSV"
                 )
                 st.markdown(download_button_analysis, unsafe_allow_html=True)
            except Exception as dl_e:
                st.warning(f"Could not generate analysis download link: {dl_e}")
        # --- End Analysis Download Link ---

    # --- Navigation ---
    st.divider()
    if st.button("⬅️ Back to Processing", use_container_width=True, key="analysis_back_proc_nav"):
        # Navigate back but keep analysis state if user returns
        st.session_state["current_step"] = "process"
        st.rerun()
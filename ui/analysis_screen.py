import streamlit as st
import pandas as pd
from core.analysis import perform_icc_computation, interpret_icc, create_icc_visualization
from ui.components import download_link
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def render_analysis_screen():
    """Render the analysis screen UI elements."""
    st.header("3. Analyze Results (Inter-Rater Reliability)")

    # --- Verify Prerequisites ---
    if not st.session_state.get("processing_complete", False):
        st.error("Processing not complete. Please go back to Step 2.")
        if st.button("⬅️ Back to Processing"):
            st.session_state["current_step"] = "process"
            st.rerun()
        return
    if not st.session_state.get("compute_irr", False):
        st.warning("IRR comparison was not enabled in Setup (Step 1).")
        st.info("Enable 'Compare with manual scores?' in Step 1 and re-process if you want to perform IRR analysis.")
        if st.button("⬅️ Back to Setup"):
             st.session_state["current_step"] = "setup"
             st.rerun()
        return
    if not st.session_state.get("manual_columns"):
        st.error("No manual score columns selected in Setup (Step 1). Cannot perform IRR.")
        if st.button("⬅️ Back to Setup"):
             st.session_state["current_step"] = "setup"
             st.rerun()
        return

    processed_df = st.session_state.get("processed_df")
    if processed_df is None:
        st.error("Processed data is missing. Please go back to Step 2.")
        if st.button("⬅️ Back to Processing"):
            st.session_state["current_step"] = "process"
            st.rerun()
        return

    response_column = st.session_state.get("response_column")
    manual_columns = st.session_state.get("manual_columns")
    use_default_context = st.session_state.get("use_default_context")

    st.info(f"Comparing GPT scores with manual scores from column(s): **{', '.join(manual_columns)}**")

    # --- Data Alignment Verification (User Confirmation) ---
    # This is less critical now with a single file, but good to remind users
    st.subheader("Verify Data")
    st.markdown("Ensure the GPT scores correspond to the correct manual scores within the processed data.")
    with st.expander("Show Sample Data for Verification"):
         cols_to_show = [response_column]
         gpt_score_col = 'gpt_score' if use_default_context else ('gpt_score_raw' if 'gpt_score_raw' in processed_df.columns else 'gpt_score')
         if gpt_score_col in processed_df.columns:
              cols_to_show.append(gpt_score_col)
         cols_to_show.extend(manual_columns)
         st.dataframe(processed_df[cols_to_show].head(), use_container_width=True)

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
    compute_btn = st.button("📈 Compute ICC(3,1)", type="primary", use_container_width=True)

    # Store result in session state to avoid recomputing on every interaction
    if compute_btn:
        with st.spinner("Calculating ICC and preparing analysis..."):
            icc_value, df_with_numeric = perform_icc_computation(
                df=processed_df.copy(), # Pass a copy to avoid modifying state directly until end
                use_default_context=use_default_context,
                response_column=response_column,
                manual_columns=manual_columns
            )
            st.session_state["icc_value"] = icc_value
            st.session_state["processed_df"] = df_with_numeric # Update df with numeric cols
            st.session_state["analysis_computed"] = True # Flag that analysis ran
            st.rerun() # Rerun to display results

    # --- Display Results ---
    if st.session_state.get("analysis_computed"):
        icc_value = st.session_state.get("icc_value")
        st.markdown(interpret_icc(icc_value), unsafe_allow_html=True)

        # Retrieve the dataframe with numeric columns added during computation
        analyzed_df = st.session_state.get("processed_df")
        ratings_matrix_df = st.session_state.get('ratings_matrix_for_viz') # Get df used for ICC calc

        if analyzed_df is None or ratings_matrix_df is None:
             st.error("Analysis data is missing after computation. Please try again.")
             st.stop()


        if icc_value is not None and not pd.isna(icc_value):
            # Prepare data for visualization (using the same filtered data as ICC)
            gpt_ratings_numeric = ratings_matrix_df['gpt_score_numeric']
            numeric_manual_cols = [col for col in ratings_matrix_df.columns if col != 'gpt_score_numeric']

            # Use average manual score if multiple raters, else the single manual score column
            if len(numeric_manual_cols) > 1:
                manual_ratings_for_viz = ratings_matrix_df[numeric_manual_cols].mean(axis=1)
                manual_label = "Average Manual Score"
            elif len(numeric_manual_cols) == 1:
                manual_ratings_for_viz = ratings_matrix_df[numeric_manual_cols[0]]
                manual_label = manual_columns[0] # Original name
            else: # Should not happen if ICC calculation succeeded
                manual_ratings_for_viz = None
                manual_label = "Manual Score"

            if manual_ratings_for_viz is not None:
                st.subheader("Visual Analysis")
                try:
                    viz_buffer = create_icc_visualization(manual_ratings_for_viz, gpt_ratings_numeric, icc_value)
                    if viz_buffer:
                        st.image(viz_buffer)
                except Exception as e:
                    st.warning(f"Could not generate visualizations: {e}")

                # --- Detailed Analysis Tabs ---
                st.subheader("Detailed Analysis")
                tab1, tab2, tab3 = st.tabs(["Statistics", "Score Comparison", "Recommendations"])

                with tab1:
                    st.markdown("#### Key Metrics")
                    cols_metrics = st.columns(3)
                    cols_metrics[0].metric("ICC(3,1)", f"{icc_value:.3f}")

                    # Pearson Correlation
                    try:
                         corr, p_val = stats.pearsonr(manual_ratings_for_viz, gpt_ratings_numeric)
                         cols_metrics[1].metric("Pearson Correlation", f"{corr:.3f}", help=f"p-value: {p_val:.3g}")
                    except Exception:
                         cols_metrics[1].metric("Pearson Correlation", "Error")

                    cols_metrics[2].metric("Analyzed Subjects", f"{len(gpt_ratings_numeric)}")


                    st.markdown("#### Agreement within Threshold")
                    diffs = abs(gpt_ratings_numeric - manual_ratings_for_viz)
                    try:
                         agree_within_1 = (diffs <= 1).mean() * 100
                         agree_within_2 = (diffs <= 2).mean() * 100
                         cols_agree = st.columns(2)
                         cols_agree[0].metric("Agreement ≤ 1 point", f"{agree_within_1:.1f}%")
                         cols_agree[1].metric("Agreement ≤ 2 points", f"{agree_within_2:.1f}%")
                    except Exception:
                         st.warning("Could not calculate agreement percentages.")

                    st.markdown("#### Rating Statistics (for analyzed subjects)")
                    col_stats1, col_stats2 = st.columns(2)
                    with col_stats1:
                         st.write(f"**{manual_label}**")
                         st.dataframe(manual_ratings_for_viz.describe().to_frame().T, use_container_width=True)
                    with col_stats2:
                         st.write("**GPT Scores**")
                         st.dataframe(gpt_ratings_numeric.describe().to_frame().T, use_container_width=True)

                with tab2:
                    st.markdown("#### Score Distributions")
                    fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
                    sns.histplot(manual_ratings_for_viz, alpha=0.5, label=manual_label, kde=True, ax=ax_dist, color='blue')
                    sns.histplot(gpt_ratings_numeric, alpha=0.5, label="GPT", kde=True, ax=ax_dist, color='orange')
                    ax_dist.set_xlabel("Score")
                    ax_dist.set_ylabel("Density" if ax_dist.get_yaxis().get_scale() == 'linear' else "Frequency") # Check if KDE is on Y
                    ax_dist.set_title("Distribution Comparison")
                    ax_dist.legend()
                    st.pyplot(fig_dist)

                    st.markdown("#### Residual Analysis (GPT - Manual)")
                    residuals = gpt_ratings_numeric - manual_ratings_for_viz
                    fig_res, (ax_res1, ax_res2) = plt.subplots(1, 2, figsize=(12, 5))

                    # Residual plot vs Manual Score
                    sns.scatterplot(x=manual_ratings_for_viz, y=residuals, alpha=0.6, ax=ax_res1)
                    ax_res1.axhline(y=0, color='r', linestyle='--')
                    ax_res1.set_xlabel(manual_label)
                    ax_res1.set_ylabel("Residual (GPT - Manual)")
                    ax_res1.set_title("Residual Plot")
                    ax_res1.grid(True, linestyle=':', alpha=0.6)

                    # Residual Distribution
                    sns.histplot(residuals, kde=True, ax=ax_res2)
                    ax_res2.set_xlabel("Residual Value")
                    ax_res2.set_ylabel("Frequency")
                    ax_res2.set_title("Residual Distribution")
                    ax_res2.grid(True, linestyle=':', alpha=0.6)

                    plt.tight_layout()
                    st.pyplot(fig_res)

                with tab3:
                    st.markdown("#### Recommendations Based on ICC(3,1)")
                    if icc_value < 0.5:
                        st.markdown("""
                        <div class="warning-box" style="border-left-color: #dc3545; background-color: #f8d7da;">
                            <p><strong>Low Reliability (Poor):</strong> The consistency between GPT and manual scores is low.</p>
                            <p>Consider:</p>
                            <ul>
                                <li><strong>Refining the Prompt:</strong> Make instructions clearer, add specific examples, or adjust the scoring scale definition.</li>
                                <li><strong>Reviewing Manual Scores:</strong> Check for inconsistencies or ambiguity in human scoring.</li>
                                <li><strong>Model Choice:</strong> A different GPT model might perform better.</li>
                                <li><strong>Data Quality:</strong> Ensure responses are suitable for the task.</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    elif icc_value < 0.75:
                         st.markdown("""
                        <div class="warning-box" style="border-left-color: #ffc107; background-color: #fff3cd;">
                            <p><strong>Moderate Reliability:</strong> There's some consistency, but room for improvement.</p>
                             <p>Consider:</p>
                            <ul>
                                <li><strong>Prompt Tuning:</strong> Minor adjustments to the prompt, focusing on areas where scores differ most (see Residual Analysis).</li>
                                <li><strong>Reviewing Edge Cases:</strong> Examine responses where GPT and manual scores diverge significantly.</li>
                                <li><strong>Enhancing Rubric in Prompt:</strong> Ensure the prompt clearly reflects all aspects of the manual scoring rubric.</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    elif icc_value < 0.9:
                         st.markdown("""
                        <div class="success-box" style="border-left-color: #198754; background-color: #d1e7dd;">
                            <p><strong>Good Reliability:</strong> GPT scores show good consistency with manual scores.</p>
                            <p>This level is often acceptable for many applications. Minor prompt tuning might further improve alignment if desired.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                         st.markdown("""
                        <div class="success-box" style="border-left-color: #0d6efd; background-color: #cfe2ff;">
                            <p><strong>Excellent Reliability:</strong> High consistency between GPT and manual scores.</p>
                            <p>The current configuration seems effective. Consider saving the prompt and model settings for future use.</p>
                        </div>
                        """, unsafe_allow_html=True)


            # --- Download Combined Results ---
            st.subheader("Download Analysis Data")
            st.markdown(
                download_link(analyzed_df, "gpt_analysis_results.csv"),
                unsafe_allow_html=True,
                help="Includes original data, GPT scores/reasons, numeric scores, and manual scores."
            )

        else:
            st.warning("ICC calculation did not yield a valid result. Cannot display detailed analysis.")

    # --- Navigation ---
    st.divider()
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("⬅️ Back to Processing", use_container_width=True):
            st.session_state["current_step"] = "process"
            # Clear analysis state? Maybe not, allow viewing old results
            # st.session_state["analysis_computed"] = False
            # st.session_state["icc_value"] = None
            st.rerun()
    # with col_nav2: # No 'next' step from analysis currently
    #     st.write("") # Placeholder
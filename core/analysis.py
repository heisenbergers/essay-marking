import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import traceback
from typing import Tuple, Optional, List

# Ensure helper is imported correctly relative to this file's location if needed
# If helpers.py is in utils/, and analysis.py is in core/
from utils.helpers import try_convert_to_numeric # Adjusted path assuming standard structure

# --- ICC Computation ---
def compute_icc(ratings_matrix: np.ndarray, icc_type: str = 'ICC(3,1)') -> float:
    """
    Compute Intraclass Correlation Coefficient (ICC) using ANOVA.
    Focuses on ICC(3,1) - Two-way mixed, consistency, single rater/measurement.

    Args:
        ratings_matrix (np.ndarray): 2D array (subjects x raters) of numeric ratings.
                                     NaN values are NOT handled here, must be dropped beforehand.
        icc_type (str): The type of ICC to compute (only 'ICC(3,1)' implemented).

    Returns:
        float: Computed ICC value, or np.nan if calculation is not possible.
    """
    if not isinstance(ratings_matrix, np.ndarray) or ratings_matrix.ndim != 2:
        st.error("ICC Error: Input must be a 2D numpy array.")
        return np.nan

    n, k = ratings_matrix.shape  # n=subjects, k=raters

    if n < 2 or k < 2:
        st.warning(f"ICC Warning: Need >= 2 subjects and >= 2 raters (got n={n}, k={k}).")
        return np.nan

    # Check for NaN values - should be handled before this function
    if np.isnan(ratings_matrix).any():
        st.error("ICC Error: NaN values found in input matrix. Check numeric conversion and filtering.")
        return np.nan

    try:
        # === ANOVA Calculations ===
        # Total Sum of Squares (SST)
        grand_mean = np.mean(ratings_matrix)
        ss_total = np.sum((ratings_matrix - grand_mean) ** 2)

        # Between-Subjects Sum of Squares (SSB / SSR)
        ss_subjects = k * np.sum((np.mean(ratings_matrix, axis=1) - grand_mean) ** 2)

        # Between-Raters Sum of Squares (SSW / SSC) - Not directly used in ICC(3,1) but useful for context
        # ss_raters = n * np.sum((np.mean(ratings_matrix, axis=0) - grand_mean) ** 2)

        # Within-Subjects Sum of Squares (SSE / Residual)
        # SSE = SST - SSB - SSW (ensure non-negative)
        # An alternative way: calculate SSW subject-wise and sum
        ss_error = np.sum((ratings_matrix - np.mean(ratings_matrix, axis=1, keepdims=True) - np.mean(ratings_matrix, axis=0, keepdims=True) + grand_mean)**2)
        ss_error = max(0, ss_error) # Ensure non-negative

        # Degrees of Freedom
        df_subjects = n - 1
        # df_raters = k - 1
        df_error = (n - 1) * (k - 1)

        # === Mean Squares ===
        # Check for valid degrees of freedom
        if df_subjects <= 0:
            st.warning("ICC Warning: Not enough subjects (n-1 <= 0).")
            return np.nan
        if df_error <= 0:
            st.warning("ICC Warning: Not enough error degrees of freedom ((n-1)(k-1) <= 0).")
            return np.nan

        ms_subjects = ss_subjects / df_subjects
        ms_error = ss_error / df_error

        # === Compute ICC ===
        if icc_type == 'ICC(3,1)':
            # Formula: (BMS - EMS) / (BMS + (k-1)*EMS)
            # BMS = Between Mean Square (Subjects) = ms_subjects
            # EMS = Error Mean Square = ms_error
            denominator = ms_subjects + (k - 1) * ms_error
            if denominator <= 1e-9:  # Avoid division by zero or near-zero
                st.warning(
                    f"ICC Warning: Denominator near zero ({denominator:.2e}). Cannot calculate reliable ICC(3,1). "
                    "May indicate no variance between subjects or extremely high error variance relative to subject variance."
                )
                # Return 0 if BMS <= EMS (no consistency), otherwise NaN as unstable
                return 0.0 if ms_subjects <= ms_error else np.nan

            icc_value = (ms_subjects - ms_error) / denominator
            # Clamp value between 0 and 1 (theoretically ICC(3,1) can be < 0, but often clipped)
            # Returning the raw value might be more informative in edge cases.
            # For practical interpretation, often capped at 0.
            return max(0.0, icc_value)
            # return icc_value # Alternative: return raw value even if < 0

        else:
            st.error(f"ICC Error: ICC type '{icc_type}' not implemented.")
            return np.nan

    except Exception as e:
        st.error(f"Unexpected error during ICC calculation: {e}")
        st.expander("Show ICC Calculation Traceback").error(traceback.format_exc())
        return np.nan

# --- Interpretation ---
def interpret_icc(icc_value: Optional[float]) -> str:
    """Interpret the ICC value with explanation (based on Koo & Li, 2016)."""
    if icc_value is None or pd.isna(icc_value):
        return "<div class='alert alert-warning' role='alert'>ICC value could not be calculated. Check numeric conversion and data filtering steps.</div>"

    # Ensure float for comparison, handle potential non-numeric input defensively
    try:
        icc_f = float(icc_value)
    except (ValueError, TypeError):
        return "<div class='alert alert-danger' role='alert'>Invalid ICC value provided for interpretation.</div>"


    if icc_f < 0.5:
        category, alert_class = "Poor reliability", "alert-danger" # Red
    elif icc_f < 0.75:
        category, alert_class = "Moderate reliability", "alert-warning" # Yellow/Orange
    elif icc_f < 0.9:
        category, alert_class = "Good reliability", "alert-success" # Green
    else:
        category, alert_class = "Excellent reliability", "alert-info" # Blue

    explanation = f"""
    <div class="alert {alert_class}" role="alert">
      <h4 class="alert-heading">ICC(3,1) Result: {icc_f:.3f} ({category})</h4>
      <p>Interpretation based on Koo & Li (2016): &lt; 0.50: Poor | 0.50 - 0.75: Moderate | 0.75 - 0.90: Good | &ge; 0.90: Excellent.</p>
      <hr>
      <p class="mb-0" style="font-size: 0.9em;"><i>Note: Acceptable reliability depends on context. ICC(3,1) measures consistency for single raters (e.g., LLM vs Manual Average/Single) assuming raters are fixed and subjects are random.</i></p>
    </div>
    """
    return explanation

# --- Visualization ---
def create_icc_visualization(manual_ratings: pd.Series, llm_ratings: pd.Series, icc_value: Optional[float]) -> Optional[BytesIO]:
    """
    Create visualization (Scatter plot and Bland-Altman) for ICC results.

    Args:
        manual_ratings (pd.Series): Numeric manual scores (can be single column or average).
        llm_ratings (pd.Series): Numeric LLM scores.
        icc_value (Optional[float]): The calculated ICC value for display.

    Returns:
        Optional[BytesIO]: Buffer containing the plot image, or None if plotting fails.
    """
    fig = None # Initialize fig to None
    try:
        # Ensure inputs are Series and drop NaNs if any remain (should be done before)
        manual_valid = pd.Series(manual_ratings).dropna()
        llm_valid = pd.Series(llm_ratings).dropna()

        # Align data based on index after dropping NaNs separately
        common_index = manual_valid.index.intersection(llm_valid.index)
        if len(common_index) < 2:
            st.warning("Not enough valid paired data points (< 2) after alignment to create visualizations.")
            return None

        manual_aligned = manual_valid.loc[common_index]
        llm_aligned = llm_valid.loc[common_index]

        # --- Create Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        icc_display = f"{icc_value:.3f}" if icc_value is not None and not pd.isna(icc_value) else "N/A"
        fig.suptitle(f"Rater Consistency Analysis (ICC = {icc_display})", fontsize=14, y=1.0) # Adjust y

        # 1. Scatter plot
        ax1 = axes[0]
        sns.regplot(x=manual_aligned, y=llm_aligned, ax=ax1, scatter_kws={'alpha': 0.5, 's': 30}, line_kws={'color': 'blue', 'alpha': 0.7})
        ax1.set_xlabel(f"Manual Ratings ({manual_aligned.name or 'Manual'})") # Use series name if available
        ax1.set_ylabel(f"LLM Ratings ({llm_aligned.name or 'LLM'})")
        ax1.set_title("Manual vs. LLM Ratings")
        # Add y=x line
        min_val = min(manual_aligned.min(), llm_aligned.min())
        max_val = max(manual_aligned.max(), llm_aligned.max())
        lims = [min_val - (max_val-min_val)*0.05, max_val + (max_val-min_val)*0.05] # Add padding
        ax1.plot(lims, lims, 'r--', alpha=0.7, label="Perfect Agreement (y=x)")
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)
        # ax1.axis('equal') # Can distort view if ranges differ significantly
        # ax1.set_aspect('equal', adjustable='box')


        # 2. Bland-Altman plot
        ax2 = axes[1]
        differences = llm_aligned - manual_aligned
        averages = (manual_aligned + llm_aligned) / 2
        mean_diff = differences.mean()
        std_diff = differences.std()
        limit_of_agreement_upper = mean_diff + 1.96 * std_diff
        limit_of_agreement_lower = mean_diff - 1.96 * std_diff

        sns.scatterplot(x=averages, y=differences, alpha=0.6, ax=ax2, s=30)
        ax2.axhline(mean_diff, color='r', linestyle='-', linewidth=1.5, label=f"Mean Diff: {mean_diff:.2f}")
        # Only draw limits if std_diff is meaningful and limits are distinct from mean
        if not pd.isna(std_diff) and std_diff > 1e-6:
            ax2.axhline(limit_of_agreement_upper, color='grey', linestyle='--', linewidth=1, label=f"+1.96 SD ({limit_of_agreement_upper:.2f})")
            ax2.axhline(limit_of_agreement_lower, color='grey', linestyle='--', linewidth=1, label=f"-1.96 SD ({limit_of_agreement_lower:.2f})")
        else:
             st.info("Limits of agreement (LoA) not shown on Bland-Altman plot (std dev near zero or NaN).")

        ax2.set_xlabel("Average of Ratings [(Manual + LLM) / 2]")
        ax2.set_ylabel("Difference (LLM - Manual)")
        ax2.set_title("Bland-Altman Plot (Agreement)")
        ax2.legend(fontsize='small')
        ax2.grid(True, linestyle=':', alpha=0.6)

        # --- Save and Return Plot ---
        plt.tight_layout(rect=(0, 0.03, 1, 0.95)) # Adjust layout
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100) # Adjust DPI if needed
        img_buffer.seek(0)
        return img_buffer

    except Exception as e:
        st.warning(f"Could not generate visualizations: {e}")
        st.expander("Show Visualization Error Traceback").error(traceback.format_exc())
        return None
    finally:
         # Ensure plot is closed to free memory, even if errors occurred
        if fig is not None:
            plt.close(fig)


# --- Main Analysis Orchestration ---
def perform_icc_computation(
    df: pd.DataFrame,
    manual_columns: List[str]
) -> Tuple[Optional[float], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Perform ICC computation after converting scores to numeric.

    Args:
        df (pd.DataFrame): DF containing 'gpt_score' and manual score columns.
        manual_columns (List[str]): List of column names containing manual scores.

    Returns:
        tuple: (icc_value, df_with_numeric, ratings_df_for_icc)
               - icc_value: Computed ICC(3,1) value or None/NaN on failure.
               - df_with_numeric: Original df updated with numeric columns.
               - ratings_df_for_icc: Filtered df used for ICC (NaNs dropped).
    """
    if df is None or df.empty:
        st.error("Analysis Error: No processed data available.")
        return None, df, None
    if not manual_columns:
        st.error("Analysis Error: No manual score columns selected.")
        return None, df, None
    if "gpt_score" not in df.columns:
        st.error("Analysis Error: 'gpt_score' column missing from processed data.")
        return None, df, None

    df_analysis = df.copy()
    icc_value_result = None
    ratings_df_result = None
    all_numeric_cols_added = []

    try:
        # 1. Prepare LLM Scores
        gpt_numeric_col = 'gpt_score_numeric'
        df_analysis[gpt_numeric_col] = try_convert_to_numeric(df_analysis['gpt_score'], column_name="LLM Score")
        all_numeric_cols_added.append(gpt_numeric_col)

        # 2. Prepare Manual Scores
        numeric_manual_cols = []
        conversion_warnings = 0
        for col in manual_columns:
            if col not in df_analysis.columns:
                st.warning(f"Analysis Warning: Manual score column '{col}' not found in the data. Skipping.")
                conversion_warnings += 1
                continue # Skip this column

            numeric_col_name = f"{col}_numeric"
            df_analysis[numeric_col_name] = try_convert_to_numeric(df_analysis[col], column_name=f"Manual Score ({col})")

            # Check if conversion failed for all non-NaN original values
            original_not_nan = df_analysis[col].notna().sum()
            converted_nan = df_analysis[numeric_col_name].isna().sum()
            original_values_present = df_analysis[col].dropna().ne("").any() # Check if there were non-empty strings/values

            if original_values_present and df_analysis[numeric_col_name].isnull().all():
                st.warning(f"Analysis Warning: Could not convert any values in manual column '{col}' to numeric.")
                conversion_warnings += 1
                # Keep column in df_analysis, but don't use for ICC
            else:
                numeric_manual_cols.append(numeric_col_name)
                all_numeric_cols_added.append(numeric_col_name)

        if not numeric_manual_cols:
            st.error("Analysis Error: No usable numeric manual score columns found after conversion attempts.")
            return None, df_analysis, None # Return df with attempted conversions

        # 3. Combine and Filter Data for ICC
        cols_for_icc = [gpt_numeric_col] + numeric_manual_cols
        ratings_df_for_icc = df_analysis[cols_for_icc].copy()

        original_rows = len(ratings_df_for_icc)
        ratings_df_for_icc.dropna(inplace=True) # Drop rows with NaN in *any* selected column
        valid_rows = len(ratings_df_for_icc)

        if valid_rows < original_rows:
            st.warning(
                f"Removed {original_rows - valid_rows} rows due to missing/non-numeric scores "
                f"in required columns ({', '.join(cols_for_icc)}) before calculating ICC."
            )

        if valid_rows < 2:
            st.error(
                f"Not enough subjects ({valid_rows}) with complete numeric ratings across all "
                f"required columns ({', '.join(cols_for_icc)}) to compute ICC (minimum 2 required)."
            )
            return None, df_analysis, None

        # 4. Compute ICC
        ratings_matrix = ratings_df_for_icc.to_numpy()
        icc_value_result = compute_icc(ratings_matrix, icc_type='ICC(3,1)')
        ratings_df_result = ratings_df_for_icc # Keep the df used for calculation

    except Exception as e:
        st.error(f"An unexpected error occurred during analysis preparation: {e}")
        st.expander("Show Analysis Prep Traceback").error(traceback.format_exc())
        return None, df_analysis, None # Return df with attempted conversions

    # Return results
    return icc_value_result, df_analysis, ratings_df_result
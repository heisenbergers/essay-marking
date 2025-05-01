import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO # Use BytesIO instead of tempfile
import scipy.stats as stats
import traceback # Import traceback for detailed error info

from .data_processing import try_convert_to_numeric # Keep using this helper


def compute_icc(ratings_matrix, icc_type='ICC(3,1)'):
    """
    Compute Intraclass Correlation Coefficient (ICC) using various models.
    Simplified implementation focusing on ICC(3,1). Uses ANOVA approach.
    Handles potential NaN values and division by zero more gracefully.
    """
    if not isinstance(ratings_matrix, np.ndarray) or ratings_matrix.ndim != 2:
         st.error("Invalid input: ratings_matrix must be a 2D numpy array.")
         return np.nan

    n, k = ratings_matrix.shape # n=subjects, k=raters

    if n < 2 or k < 2:
        st.warning(f"Need at least 2 subjects and 2 raters for ICC calculation (got {n} subjects, {k} raters).")
        return np.nan

    # Check for NaN values - should ideally be handled before this function
    if np.isnan(ratings_matrix).any():
        st.error("NaN values found in ratings matrix for ICC calculation. Check numeric conversion.")
        return np.nan

    try:
        # Calculate Sums of Squares
        grand_mean = np.mean(ratings_matrix)
        ss_total = np.sum((ratings_matrix - grand_mean)**2)
        ss_subjects = k * np.sum((np.mean(ratings_matrix, axis=1) - grand_mean)**2)
        ss_raters = n * np.sum((np.mean(ratings_matrix, axis=0) - grand_mean)**2)
        # Ensure ss_error is not negative due to floating point inaccuracies
        ss_error = max(0, ss_total - ss_subjects - ss_raters)

        # Calculate Mean Squares
        df_subjects = n - 1
        df_raters = k - 1
        df_error = (n - 1) * (k - 1)

        if df_subjects <= 0:
             st.warning("Not enough degrees of freedom for subjects (n-1 <= 0).")
             return np.nan
        if df_error <= 0:
            # If df_error is 0 (e.g., only 2 raters and n subjects, or n=1),
            # we might not be able to calculate ICC(3,1) which needs MS_error.
            # Different ICC forms handle this differently. For ICC(3,1), return NaN.
            st.warning("Not enough degrees of freedom for error ((n-1)(k-1) <= 0).")
            return np.nan

        ms_subjects = ss_subjects / df_subjects
        # ms_raters = ss_raters / df_raters # Not needed for ICC(3,1)
        ms_error = ss_error / df_error

        # Compute ICC based on type
        if icc_type == 'ICC(3,1)':
            # Two-way mixed, consistency, single rater/measurement
            # ICC = (BMS - EMS) / (BMS + (k-1)*EMS)
            denominator = ms_subjects + (k - 1) * ms_error
            if denominator <= 0: # Avoid division by zero or negative denominator
                 st.warning(f"ICC denominator non-positive ({denominator:.2f}). Cannot calculate reliable ICC(3,1). May indicate no variance between subjects or high error variance.")
                 # Return 0 if BMS <= EMS, otherwise NaN as it's problematic
                 return 0.0 if ms_subjects <= ms_error else np.nan

            icc_value = (ms_subjects - ms_error) / denominator
            # Clamp value between 0 and 1, handle potential floating point issues or negative estimates
            return max(0.0, min(1.0, icc_value if not pd.isna(icc_value) else np.nan))

        else:
            st.error(f"ICC type '{icc_type}' not implemented.")
            return np.nan

    except Exception as e:
        st.error(f"Unexpected error during ICC calculation: {e}")
        st.expander("Show ICC Calculation Traceback").error(traceback.format_exc())
        return np.nan


def interpret_icc(icc_value):
    """Interpret the ICC value with explanation (based on Koo & Li, 2016)."""
    if icc_value is None or pd.isna(icc_value):
         return "<div class='info-box' style='border-left-color: gray;'>ICC value could not be calculated. Check if scores could be converted to numeric and if enough valid data points were available.</div>"

    icc_value = float(icc_value) # Ensure it's float for comparison

    if icc_value < 0.5: category, color = "Poor reliability", "#dc3545" # Red
    elif icc_value < 0.75: category, color = "Moderate reliability", "#ffc107" # Yellow/Orange
    elif icc_value < 0.9: category, color = "Good reliability", "#198754" # Green
    else: category, color = "Excellent reliability", "#0d6efd" # Blue

    explanation = f"""
    <div style="padding: 10px; border-radius: 5px; border: 1px solid {color}; margin: 10px 0; background-color: {color}20;">
    <h4 style="color: {color}; margin-top:0; margin-bottom: 5px;">ICC(3,1) Result: {icc_value:.3f} ({category})</h4>
    <p style="font-size: 0.9em; margin-bottom: 5px;">Interpretation based on Koo & Li (2016):</p>
    <ul style="margin-bottom: 5px; font-size: 0.9em;">
        <li>&lt; 0.50: Poor</li><li>0.50 - 0.75: Moderate</li><li>0.75 - 0.90: Good</li><li>&ge; 0.90: Excellent</li>
    </ul>
    <p style="font-size: 0.85em; margin-bottom:0; color: #555;"><i>Note: Acceptable reliability depends on context. ICC(3,1) measures consistency for single raters (LLM vs Manual Average/Single) assuming raters are fixed and subjects are random.</i></p>
    </div>
    """
    return explanation

def create_icc_visualization(manual_ratings: pd.Series, llm_ratings: pd.Series, icc_value):
    """Create visualization (Scatter plot and Bland-Altman) for ICC results."""
    # Wrap the entire plotting in try-except for graceful failure
    try:
        manual = pd.to_numeric(manual_ratings, errors='coerce')
        llm = pd.to_numeric(llm_ratings, errors='coerce')
        valid_mask = ~(manual.isna() | llm.isna())
        manual_valid = manual[valid_mask]
        llm_valid = llm[valid_mask]

        if len(manual_valid) < 2:
            st.warning("Not enough valid paired data points (< 2) to create visualizations.")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5)) # Slightly taller figure
        icc_display = f"{icc_value:.3f}" if icc_value is not None and not pd.isna(icc_value) else "N/A"
        fig.suptitle(f"Consistency Analysis (ICC = {icc_display})", fontsize=14, y=0.98) # Adjust title position

        # --- Scatter plot ---
        ax1 = axes[0]
        sns.regplot(x=manual_valid, y=llm_valid, ax=ax1, scatter_kws={'alpha':0.5}, line_kws={'color':'blue', 'alpha':0.7})
        ax1.set_xlabel("Manual Ratings")
        ax1.set_ylabel("LLM Ratings")
        ax1.set_title("Manual vs. LLM Ratings")
        # Add perfect agreement line (y=x)
        min_val = min(manual_valid.min(), llm_valid.min()) if not manual_valid.empty else 0
        max_val = max(manual_valid.max(), llm_valid.max()) if not manual_valid.empty else 1
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label="Perfect Agreement (y=x)")
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)
        # Try to make axes roughly equal if range is similar
        ax1.axis('equal')
        ax1.autoscale(True) # Re-apply autoscale after setting equal aspect

        # --- Bland-Altman plot ---
        ax2 = axes[1]
        differences = llm_valid - manual_valid
        averages = (manual_valid + llm_valid) / 2
        mean_diff = differences.mean()
        std_diff = differences.std()
        limit_of_agreement_upper = mean_diff + 1.96 * std_diff
        limit_of_agreement_lower = mean_diff - 1.96 * std_diff

        sns.scatterplot(x=averages, y=differences, alpha=0.6, ax=ax2)
        ax2.axhline(mean_diff, color='r', linestyle='-', label=f"Mean Diff: {mean_diff:.2f}")
        # Only draw limits if std_diff is meaningful
        if not pd.isna(std_diff) and std_diff > 1e-6: # Avoid drawing if std dev is tiny/zero
            ax2.axhline(limit_of_agreement_upper, color='grey', linestyle='--', label=f"+1.96 SD ({limit_of_agreement_upper:.2f})")
            ax2.axhline(limit_of_agreement_lower, color='grey', linestyle='--', label=f"-1.96 SD ({limit_of_agreement_lower:.2f})")
        else:
             st.info("Limits of agreement not shown on Bland-Altman (std dev near zero or NaN).")


        ax2.set_xlabel("Average of Ratings [(Manual + LLM) / 2]")
        ax2.set_ylabel("Difference (LLM - Manual)")
        ax2.set_title("Bland-Altman Plot")
        ax2.legend(fontsize='small')
        ax2.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout further
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=120) # Slightly higher DPI
        plt.close(fig) # Close the plot to free memory
        img_buffer.seek(0)
        return img_buffer

    except Exception as e:
        st.warning(f"Could not generate visualizations: {e}")
        st.expander("Show Visualization Error Traceback").error(traceback.format_exc())
        return None # Return None if plotting fails

def perform_icc_computation(df: pd.DataFrame, response_column: str, manual_columns: list):
    """
    Perform the ICC computation using the single dataframe. Handles numeric conversion internally.

    Args:
        df (pd.DataFrame): The dataframe containing response, LLM score, and manual score columns.
                           MUST contain 'gpt_score' (parsed/extracted score).
        response_column (str): Name of the student response column (for reference/display).
        manual_columns (list): List of column names containing manual scores.

    Returns:
        tuple: (icc_value, df_with_numeric, ratings_df_for_icc)
               - icc_value (float or None): Computed ICC value.
               - df_with_numeric (pd.DataFrame): Original df possibly updated with numeric columns.
               - ratings_df_for_icc (pd.DataFrame or None): Filtered df used for ICC calculation (for viz).
    """
    if df is None or df.empty:
        st.error("No processed data available for analysis.")
        return None, df, None

    if not manual_columns:
        st.error("No manual score columns selected for IRR analysis.")
        return None, df, None

    # Work on a copy to add numeric columns without modifying the original state directly yet
    df_analysis = df.copy()
    icc_value_result = None
    ratings_df_result = None

    try:
        # --- 1. Prepare LLM Scores ---
        gpt_numeric_col = 'gpt_score_numeric'
        if "gpt_score" not in df_analysis.columns:
             st.error("Column 'gpt_score' not found in processed data. This column should contain the extracted score.")
             return None, df, None # Return original df
        df_analysis[gpt_numeric_col] = try_convert_to_numeric(df_analysis['gpt_score'], column_name="LLM Score")

        # --- 2. Prepare Manual Scores ---
        numeric_manual_cols = []
        manual_conversion_failed = False
        for col in manual_columns:
            if col not in df_analysis.columns:
                st.warning(f"Manual score column '{col}' not found in the data.")
                manual_conversion_failed = True # Treat as failure if column missing
                continue
            numeric_col_name = f"{col}_numeric"
            df_analysis[numeric_col_name] = try_convert_to_numeric(df_analysis[col], column_name=f"Manual Score ({col})")
            if df_analysis[numeric_col_name].isnull().all():
                 st.warning(f"Could not convert any values in manual column '{col}' to numeric.")
                 manual_conversion_failed = True # Treat as failure if all NaN
            else:
                 numeric_manual_cols.append(numeric_col_name)

        if not numeric_manual_cols:
             st.error("Could not find or convert any specified manual score columns to usable numeric data.")
             return None, df_analysis, None # Return df with attempted conversions

        # --- 3. Combine and Filter Data for ICC ---
        all_numeric_cols = [gpt_numeric_col] + numeric_manual_cols
        ratings_df_for_icc = df_analysis[all_numeric_cols].copy()

        original_rows = len(ratings_df_for_icc)
        ratings_df_for_icc.dropna(inplace=True)
        valid_rows = len(ratings_df_for_icc)

        if valid_rows < original_rows:
            st.warning(f"Removed {original_rows - valid_rows} rows due to missing/non-numeric scores in required columns before calculating ICC.")

        if valid_rows < 2:
            st.error(f"Not enough valid subjects ({valid_rows}) with complete ratings across all selected columns to compute ICC (minimum 2 required).")
            return None, df_analysis, None # Return df with attempted conversions

        # --- 4. Compute ICC ---
        ratings_matrix = ratings_df_for_icc.to_numpy()
        icc_value_result = compute_icc(ratings_matrix, icc_type='ICC(3,1)') # Using ICC(3,1)
        ratings_df_result = ratings_df_for_icc # Keep the df used for calculation

    except Exception as e:
        st.error(f"An unexpected error occurred during analysis preparation: {str(e)}")
        st.expander("Show Analysis Prep Traceback").error(traceback.format_exc())
        return None, df_analysis, None # Return df with attempted conversions

    # Return results, including the df with added numeric columns
    return icc_value_result, df_analysis, ratings_df_result
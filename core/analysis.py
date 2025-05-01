import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tempfile import NamedTemporaryFile
import scipy.stats as stats # Using scipy for stats

# Assuming data_processing.py is in the same directory
from .data_processing import try_convert_to_numeric


def compute_icc(ratings_matrix, icc_type='ICC(3,1)'):
    """
    Compute Intraclass Correlation Coefficient (ICC) using various models.
    Simplified implementation focusing on ICC(3,1).
    Uses ANOVA approach.

    Args:
        ratings_matrix (np.ndarray): A matrix where rows are subjects and columns are raters.
                                     Should contain numeric data only. NaN values should be handled before calling.
        icc_type (str): The type of ICC to compute. Currently supports 'ICC(3,1)'.

    Returns:
        float: The calculated ICC value, or np.nan if computation fails.
    """
    n, k = ratings_matrix.shape # n=subjects, k=raters

    if n < 2 or k < 2:
        st.warning(f"Need at least 2 subjects and 2 raters for ICC calculation (got {n} subjects, {k} raters).")
        return np.nan

    # Check for NaN values - should ideally be handled before this function
    if np.isnan(ratings_matrix).any():
        st.error("NaN values found in ratings matrix for ICC calculation.")
        return np.nan

    # Calculate Sums of Squares
    grand_mean = np.mean(ratings_matrix)
    ss_total = np.sum((ratings_matrix - grand_mean)**2)
    ss_subjects = k * np.sum((np.mean(ratings_matrix, axis=1) - grand_mean)**2)
    ss_raters = n * np.sum((np.mean(ratings_matrix, axis=0) - grand_mean)**2)
    ss_error = ss_total - ss_subjects - ss_raters

    # Calculate Mean Squares
    df_subjects = n - 1
    df_raters = k - 1
    df_error = (n - 1) * (k - 1)

    if df_subjects <= 0 or df_error <= 0:
         st.warning("Not enough degrees of freedom for ICC calculation.")
         return np.nan

    ms_subjects = ss_subjects / df_subjects
    # ms_raters = ss_raters / df_raters # Not needed for ICC(3,1)
    ms_error = ss_error / df_error

    # Compute ICC based on type
    if icc_type == 'ICC(3,1)':
        # Two-way mixed, consistency, single rater/measurement
        # ICC = (BMS - EMS) / (BMS + (k-1)*EMS)
        if ms_subjects < ms_error: # Handle case where variance between subjects is less than error variance
             return 0.0
        if (ms_subjects + (k - 1) * ms_error) == 0: # Avoid division by zero
             return np.nan

        icc_value = (ms_subjects - ms_error) / (ms_subjects + (k - 1) * ms_error)
        return max(0.0, min(1.0, icc_value)) # Ensure value is between 0 and 1

    else:
        st.error(f"ICC type '{icc_type}' not implemented.")
        return np.nan


def interpret_icc(icc_value):
    """Interpret the ICC value with explanation (based on Koo & Li, 2016)."""
    if icc_value is None or pd.isna(icc_value):
         return "<div style='padding: 10px; border-radius: 5px; border: 1px solid gray; margin: 10px 0;'><p>ICC value could not be calculated.</p></div>"

    if icc_value < 0.5:
        category = "Poor reliability"
        color = "#dc3545" # Red
    elif icc_value < 0.75:
        category = "Moderate reliability"
        color = "#ffc107" # Yellow/Orange
    elif icc_value < 0.9:
        category = "Good reliability"
        color = "#198754" # Green
    else:
        category = "Excellent reliability"
        color = "#0d6efd" # Blue

    explanation = f"""
    <div style="padding: 10px; border-radius: 5px; border: 1px solid {color}; margin: 10px 0; background-color: {color}20;">
    <h3 style="color: {color}; margin-top:0;">ICC(3,1): {icc_value:.3f}</h3>
    <p>This suggests <strong style="color: {color};">{category}</strong> based on common guidelines (e.g., Koo & Li, 2016):</p>
    <ul style="margin-bottom: 5px;">
        <li><strong>&lt; 0.50:</strong> Poor</li>
        <li><strong>0.50 - 0.75:</strong> Moderate</li>
        <li><strong>0.75 - 0.90:</strong> Good</li>
        <li><strong>&ge; 0.90:</strong> Excellent</li>
    </ul>
    <p style="font-size: 0.9em; margin-bottom:0;"><i>Note: Acceptable reliability depends on the context and field. ICC(3,1) measures consistency for single raters assuming raters are fixed and subjects are random.</i></p>
    </div>
    """
    return explanation

def create_icc_visualization(manual_ratings: pd.Series, gpt_ratings: pd.Series, icc_value):
    """Create visualization (Scatter plot and Bland-Altman) for ICC results."""
    # Ensure data is numeric and drop NaNs for plotting
    manual = pd.to_numeric(manual_ratings, errors='coerce')
    gpt = pd.to_numeric(gpt_ratings, errors='coerce')
    valid_mask = ~(manual.isna() | gpt.isna())
    manual_valid = manual[valid_mask]
    gpt_valid = gpt[valid_mask]

    if len(manual_valid) < 2:
        st.warning("Not enough valid paired data points (< 2) to create visualizations.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Consistency Analysis (ICC = {icc_value:.3f})", fontsize=14)

    # --- Scatter plot ---
    ax1 = axes[0]
    sns.scatterplot(x=manual_valid, y=gpt_valid, alpha=0.6, ax=ax1)
    ax1.set_xlabel("Manual Ratings")
    ax1.set_ylabel("GPT Ratings")
    ax1.set_title("Manual vs. GPT Ratings")

    # Add perfect agreement line (y=x)
    min_val = min(manual_valid.min(), gpt_valid.min())
    max_val = max(manual_valid.max(), gpt_valid.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label="Perfect Agreement (y=x)")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Bland-Altman plot ---
    ax2 = axes[1]
    differences = gpt_valid - manual_valid
    averages = (manual_valid + gpt_valid) / 2
    mean_diff = differences.mean()
    std_diff = differences.std()
    limit_of_agreement = 1.96 * std_diff

    sns.scatterplot(x=averages, y=differences, alpha=0.6, ax=ax2)
    ax2.axhline(mean_diff, color='r', linestyle='-', label=f"Mean Diff: {mean_diff:.2f}")
    ax2.axhline(mean_diff + limit_of_agreement, color='grey', linestyle='--', label=f"+1.96 SD ({mean_diff + limit_of_agreement:.2f})")
    ax2.axhline(mean_diff - limit_of_agreement, color='grey', linestyle='--', label=f"-1.96 SD ({mean_diff - limit_of_agreement:.2f})")

    ax2.set_xlabel("Average of Ratings [(Manual + GPT) / 2]")
    ax2.set_ylabel("Difference (GPT - Manual)")
    ax2.set_title("Bland-Altman Plot")
    ax2.legend(fontsize='small')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # Save figure to a temporary file for display
    # Using BytesIO is generally better for web apps than temp files
    from io import BytesIO
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100)
    plt.close(fig) # Close the plot to free memory
    img_buffer.seek(0)
    return img_buffer


def perform_icc_computation(df: pd.DataFrame, use_default_context: bool, response_column: str, manual_columns: list):
    """
    Perform the ICC computation using the single dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing response, GPT score, and manual score columns.
        use_default_context (bool): If true, assumes 'gpt_score' and 'gpt_reason' exist. Otherwise uses 'gpt_score_raw'.
        response_column (str): Name of the student response column (for reference/display).
        manual_columns (list): List of column names containing manual scores.

    Returns:
        float or None: The computed ICC value, or None if computation failed.
                       Also modifies the input df to add cleaned numeric columns.
    """
    if df is None or df.empty:
        st.error("No data available for analysis.")
        return None, df

    if not manual_columns:
        st.error("No manual score columns selected for IRR analysis.")
        return None, df

    # --- 1. Prepare GPT Scores ---
    gpt_score_col_name = "gpt_score"
    if use_default_context:
        if "gpt_score" not in df.columns:
             st.error("Column 'gpt_score' not found in processed data (expected for default context).")
             return None, df
        # Use the dedicated score column, attempt conversion
        df['gpt_score_numeric'] = try_convert_to_numeric(df['gpt_score'], column_name="GPT Score")
    else:
        # Use the raw score column if not default context, attempt conversion
        score_source_col = "gpt_score_raw" if "gpt_score_raw" in df.columns else "gpt_score"
        if score_source_col not in df.columns:
             st.error(f"Column '{score_source_col}' not found in processed data.")
             return None, df
        df['gpt_score_numeric'] = try_convert_to_numeric(df[score_source_col], column_name="GPT Score (from Raw)")
        gpt_score_col_name = score_source_col # For referencing original gpt score

    # --- 2. Prepare Manual Scores ---
    numeric_manual_cols = []
    for col in manual_columns:
        if col not in df.columns:
            st.warning(f"Manual score column '{col}' not found in the uploaded data.")
            continue
        numeric_col_name = f"{col}_numeric"
        df[numeric_col_name] = try_convert_to_numeric(df[col], column_name=f"Manual Score ({col})")
        if not df[numeric_col_name].isnull().all(): # Only include if some values were converted
             numeric_manual_cols.append(numeric_col_name)

    if not numeric_manual_cols:
         st.error("Could not find or convert any specified manual score columns to numeric data.")
         return None, df

    # --- 3. Combine and Filter Data for ICC ---
    all_numeric_cols = ['gpt_score_numeric'] + numeric_manual_cols
    ratings_df_for_icc = df[all_numeric_cols].copy()

    # Drop rows where *any* of the required ratings are missing/non-numeric
    original_rows = len(ratings_df_for_icc)
    ratings_df_for_icc.dropna(inplace=True)
    valid_rows = len(ratings_df_for_icc)

    if valid_rows < original_rows:
        st.warning(f"Removed {original_rows - valid_rows} rows due to missing/non-numeric scores in required columns.")

    if valid_rows < 2:
        st.error(f"Not enough valid subjects ({valid_rows}) with complete ratings across all selected columns to compute ICC (minimum 2 required).")
        return None, df

    # --- 4. Compute ICC ---
    try:
        ratings_matrix = ratings_df_for_icc.to_numpy()
        icc_value = compute_icc(ratings_matrix, icc_type='ICC(3,1)') # Using ICC(3,1)
        st.session_state['ratings_matrix_for_viz'] = ratings_df_for_icc # Save for visualization
        return icc_value, df # Return computed value and modified df

    except Exception as e:
        st.error(f"Error computing ICC: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, df
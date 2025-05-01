import streamlit as st
import base64
import pandas as pd

def apply_custom_css():
    """Applies custom CSS styling to the Streamlit app."""
    st.markdown("""
    <style>
    /* General */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px; /* Slightly wider */
    }
    h1, h2, h3 {
        margin-bottom: 0.75rem;
        font-weight: 600; /* Slightly bolder headers */
    }
    h1 { margin-top: 0rem; }
    h2 { margin-top: 1.5rem; }
    h3 { margin-top: 1rem; font-size: 1.25rem; } /* Smaller h3 */

    /* Buttons */
    .stButton > button {
        border-radius: 5px;
        padding: 0.4rem 0.8rem; /* Adjusted padding */
        font-weight: 500;
    }
    /* Primary button style */
    div.stButton > button:first-child[kind="primary"] {
        background-color: #0d6efd; /* Bootstrap primary blue */
        color: white;
        border: 1px solid #0d6efd;
    }
    div.stButton > button:first-child[kind="primary"]:hover {
        background-color: #0b5ed7;
        border-color: #0a58ca;
    }
    /* Secondary button style */
    div.stButton > button:first-child[kind="secondary"] {
        background-color: #6c757d; /* Bootstrap secondary gray */
        color: white;
        border: 1px solid #6c757d;
    }
     div.stButton > button:first-child[kind="secondary"]:hover {
        background-color: #5c636a;
        border-color: #565e64;
    }
     /* Disabled button style */
    div.stButton > button:disabled {
         background-color: #e9ecef;
         color: #adb5bd;
         border-color: #ced4da;
         opacity: 0.65;
     }


    /* Custom Download Link Button */
    .download-link {
        display: inline-block;
        background-color: #198754; /* Bootstrap success green */
        color: white !important; /* Ensure text is white */
        padding: 8px 16px;
        text-decoration: none;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: 500;
        border: 1px solid #198754;
        transition: background-color 0.2s ease;
    }
    .download-link:hover {
        background-color: #157347;
        color: white !important;
        text-decoration: none;
    }

    /* Info/Warning/Success Boxes */
    .info-box, .warning-box, .success-box, .error-box {
        padding: 12px 16px;
        margin: 12px 0;
        border-radius: 5px;
        border-left-width: 6px;
        border-left-style: solid;
    }
    .info-box { background-color: #cfe2ff; border-left-color: #0d6efd; } /* Blue */
    .warning-box { background-color: #fff3cd; border-left-color: #ffc107; } /* Yellow */
    .success-box { background-color: #d1e7dd; border-left-color: #198754; } /* Green */
    .error-box { background-color: #f8d7da; border-left-color: #dc3545; } /* Red */

    /* Expander styling */
    div[data-testid="stExpander"] {
        border: 1px solid #dee2e6; /* Lighter border */
        border-radius: 5px;
        margin-bottom: 1rem;
    }
     div[data-testid="stExpander"] > div:first-child { /* Header */
         background-color: #f8f9fa; /* Light background for header */
     }

     /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #dee2e6;
        border-radius: 5px;
    }

    /* Footer */
     .footer {
         text-align: center;
         margin-top: 40px;
         padding-top: 20px;
         border-top: 1px solid #ddd;
         color: #777;
         font-size: 0.9em;
     }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Renders the main application header and introductory text."""
    st.title("📊 GPT Essay Scoring Tool")
    st.markdown("""
    Use Large Language Models (like GPT-4o) to automatically score student responses based on your criteria,
    and optionally compare results against manual scores using Inter-Rater Reliability (ICC).
    """, unsafe_allow_html=True)

    # Only show detailed instructions on the setup screen
    if st.session_state.get("current_step", "setup") == "setup":
        with st.expander("📘 How to Use", expanded=False):
            st.markdown("""
            1.  **Configure Secrets:** Ensure your OpenAI API key is added to Streamlit Secrets by the app deployer (if running on Cloud).
            2.  **Configure Prompt:** Select a default prompt template or create your own custom instructions for the AI.
            3.  **Select Model:** Choose the desired OpenAI model (e.g., `gpt-4o-mini`, `gpt-4o`).
            4.  **Upload Data:** Upload a CSV or Excel file containing student responses. If comparing scores, this file *must also* include the manual scores in separate columns.
            5.  **Select Columns:** Specify which column holds the student responses. If comparing, select the column(s) with manual scores.
            6.  **Process:** Navigate to the 'Process' step and click 'Start Processing'.
            7.  **Analyze (Optional):** If manual scores were provided and comparison was enabled, go to the 'Analysis' step to view ICC results and visualizations.
            8.  **Download:** Download the processed data and analysis results.

            **💡 Tip:** Start with a small subset of responses to test your prompt and configuration before processing a large dataset.
            """)
    st.divider()


def download_link(df: pd.DataFrame, filename: str, link_text: str = "Download Results as CSV"):
    """Generates a download link for a Pandas DataFrame."""
    try:
        csv = df.to_csv(index=False, encoding='utf-8')
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">{link_text}</a>'
        return href
    except Exception as e:
        st.error(f"Error creating download link: {e}")
        return ""

def render_footer():
     st.markdown("""
     <div class="footer">
         GPT Scoring Tool | Built with Streamlit | Version 3.1
     </div>
     """, unsafe_allow_html=True)
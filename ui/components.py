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
    st.title("📊 GPT Essay Scoring Tool") #
    st.markdown("""
    Use Large Language Models (like GPT-4o) to automatically score student responses based on your criteria,
    and optionally compare results against manual scores using Inter-Rater Reliability (ICC).
    """, unsafe_allow_html=True) #

    # Only show detailed instructions on the setup screen
    if st.session_state.get("current_step", "setup") == "setup": #
        with st.expander("📘 How to Use", expanded=False): #
            st.markdown("""
            This tool helps you use Artificial Intelligence (AI) like GPT-4o to automatically score student essays or responses. You can set your own grading instructions and even compare the AI's scores with manual scores.

            Here’s a step-by-step guide to get you started:

            ### 1. Understanding API Keys (Your Key to Using AI)

            **What is an API Key?**

            Think of an API key as a secret password that allows computer programs to talk to each other. In this case, it lets the GPT Essay Scoring Tool connect to AI services (like OpenAI) to get the essays scored. Each user has their own unique key. It's important to keep your API key private, just like a password, because it's linked to your account and usage.

            **How to Get an API Key (Example: OpenAI)**

            To use AI models from companies like OpenAI, you'll need to get an API key from them. Here’s how you can get an OpenAI API key:

            * **Create an Account:** If you don’t have one already, go to the [OpenAI website](https://openai.com/) and sign up. You'll need to verify your email address.
            * **Go to the API Section:** Once you're logged in, you need to find the API section. The direct link is often [platform.openai.com](https://platform.openai.com/).
            * **Find API Keys:** In your account dashboard on the OpenAI platform, look for a menu item on the left typically labeled "API keys".
            * **Create a New Key:** You should see an option like "Create new secret key". Click on it.
            * **Name Your Key (Optional but Recommended):** Give your key a name that helps you remember what it's for (e.g., "EssayScoringToolKey").
            * **Copy and Save Your Key:** OpenAI will show you your new API key. **This is the only time you will see the full key, so copy it immediately and save it in a safe, private place** (like a password manager or a secure note). If you lose it, you'll have to create a new one.
            * **Set Up Billing:** You have to charge the wallet with some funds in order for the API key to be active.

            **Important Notes on API Keys:**
            * **Keep them Secret:** Do not share your API keys publicly.
            * **Different Providers, Different Keys:** If you want to use models from other AI providers (like Gemini or Claude), you'll need to get an API key from each of them separately, following their specific instructions. This tool will ask for the relevant key based on the provider you select.

            ### 2. Configure the Scoring Tool

            Once you have your API key, you can start setting up the tool:

            * **Enter Your API Key:** In the "Setup" section of this tool, choose your AI Provider (e.g., OpenAI). Then, enter the API key you saved. The tool will try to verify it.
            * **Write Your Grading Instructions (Prompt):**
                * Think about how you want the AI to grade. What criteria should it use? What's the scoring scale (e.g., 1-10)? What should the AI focus on?
                * You can select a pre-written prompt template or write your own "Custom Prompt." Clearly explain the task to the AI. For example: "You are a teaching assistant. Grade this essay from 1 to 10 based on clarity, argument strength, and use of evidence. Provide a score and a brief reason for the score."
            * **Select the AI Model:** Choose the specific AI model you want to use (e.g., `gpt-4o-mini`, `gpt-4o`). Different models have different capabilities and costs.
            * **Choose Output Format:** Tell the tool how you expect the AI to give you the score.
                * `JSON (Score and Reason)`: Good if you want a score and a written explanation. The AI should reply like: `{"score": 8, "reason": "The essay was well-structured..."}`
                * `Integer Score Only`: If you just need a number (e.g., `8`).
                * `Raw Text`: If you want the AI's full text response without specific formatting.

            ### 3. Upload Your Data

            * **Prepare Your File:** Your student responses should be in a CSV or Excel file.
                * Make sure there's a column that contains the actual text of each student's response.
                * **If comparing with manual scores:** Your file *must also* have one or more columns containing the scores given by human graders.
            * **Upload the File:** Use the "Upload CSV or Excel file" button in the "Setup" section.
            * **Select Columns:**
                * Tell the tool which column has the student responses.
                * **If comparing:** Enable the "Compare LLM scores with manual scores?" option and then select the column(s) that have your manual scores.

            ### 4. Process the Responses

            * **Go to the "Process" Step:** Once everything is set up, navigate to the "Process Responses with LLM" section.
            * **Review Settings:** Check your chosen provider, model, and response column.
            * **Optional: Process a Subset:** If you have a large file, you can choose to process only the first few rows to test your setup and prompt.
            * **Start Processing:** Click the "Start Processing" button. The tool will send the responses to the AI one by one (or in small batches) to get them scored. This might take some time and may incur costs depending on your API provider and usage.

            ### 5. Analyze Results (If Comparing with Manual Scores)

            * **Go to the "Analyze" Step:** If you uploaded manual scores and enabled the comparison, go to the "Analyze Results" section after processing is complete.
            * **View Reliability:** The tool can calculate a statistic called the Intraclass Correlation Coefficient (ICC). This tells you how consistent the AI's scores are with the manual scores.
                * You'll see the ICC value and an interpretation (e.g., Poor, Moderate, Good, Excellent reliability).
                * Visual charts like scatter plots can also help you see how the scores compare.
            * **Recommendations:** Based on the ICC, you might get suggestions on how to improve the AI's grading (e.g., by refining your prompt).

            ### 6. Download Your Results

            * After processing (and analysis, if done), you can download a file containing:
                * Your original data.
                * The AI's raw output.
                * The parsed scores (and reasons, if you chose that format).
                * Any analysis data if you compared scores.

            **💡 Tip:** Always start with a small number of responses (e.g., 5-10) to test your prompt and settings before processing a large dataset. This helps you fine-tune your instructions to the AI and ensures you're getting the kind of results you expect.
            """)
    st.divider() #


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
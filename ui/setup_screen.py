import streamlit as st
import pandas as pd
from utils.helpers import load_prompts
from core.data_processing import load_data

def render_setup_screen():
    """Render the setup screen UI elements."""
    st.header("1. Setup Configuration")

    # --- Check for API Key (via Secrets) ---
    api_key_status = "Not Set"
    try:
        if st.secrets.get("openai_api_key"):
            api_key_status = "✅ Configured in Secrets"
            st.success(api_key_status)
        else:
            api_key_status = "❌ Missing in Secrets"
            st.error("OpenAI API key (`openai_api_key`) is missing in Streamlit Secrets.")
            st.info("Please ask the app deployer to add the key via the Streamlit Cloud dashboard.")
            st.stop() # Stop execution if key is missing
    except Exception as e:
        # Secrets might not exist locally, provide guidance
        api_key_status = f"❓ Error checking Secrets ({type(e).__name__}). May work on Cloud."
        st.warning(api_key_status)
        st.info("Secrets are typically configured when deploying to Streamlit Community Cloud.")
        # Allow proceeding locally for testing UI, but API calls will fail later
        # Consider adding a local input for testing if needed:
        # local_key = st.text_input("Enter API key for local testing (optional)", type="password")
        # if local_key: st.session_state['local_api_key'] = local_key

    # --- Prompt Configuration ---
    st.subheader("Prompt Configuration")
    default_prompts = load_prompts() # Load from context/ folder
    prompt_options = ["Custom Prompt"] + list(default_prompts.keys()) # Custom first

    # Determine default selection
    current_prompt_name = st.session_state.get("selected_prompt_name", prompt_options[1] if len(prompt_options) > 1 else prompt_options[0])
    if current_prompt_name not in prompt_options: # Handle case where saved name is no longer valid
         current_prompt_name = prompt_options[1] if len(prompt_options) > 1 else prompt_options[0]

    selected_prompt_name = st.selectbox(
        "Select Prompt Template:",
        options=prompt_options,
        index=prompt_options.index(current_prompt_name),
        key="prompt_select"
    )
    st.session_state["selected_prompt_name"] = selected_prompt_name

    # Display text area for the prompt
    prompt_changed = False
    if selected_prompt_name == "Custom Prompt":
        st.session_state["prompt_choice"] = "Use custom prompt"
        prompt_label = "Enter Custom Prompt:"
        # Use existing custom context if available, otherwise empty
        default_value = st.session_state.get("custom_context", "")
        current_context_input = st.text_area(prompt_label, value=default_value, height=250, key="custom_prompt_area")
        if current_context_input != default_value:
             st.session_state["custom_context"] = current_context_input # Save custom prompt separately
             st.session_state["context"] = current_context_input # Main context used by app
             prompt_changed = True

    else: # A default prompt is selected
        st.session_state["prompt_choice"] = "Use default prompt"
        prompt_label = f"Selected Prompt: {selected_prompt_name} (Editable for this session)"
        # Use current session context if it matches the selected name's content *or* if it's been edited
        # Otherwise, load the default content for the selected name
        default_text = default_prompts.get(selected_prompt_name, "")
        current_session_context = st.session_state.get("context", default_text)

        # If the selected prompt name changed OR if the current context IS the default (meaning not edited yet)
        # load the default. Otherwise, keep the edited version.
        display_value = default_text if current_session_context == default_prompts.get(st.session_state.get("_last_selected_prompt"), "") else current_session_context
        # Correctly load default when selection changes
        if selected_prompt_name != st.session_state.get("_last_selected_prompt"):
             display_value = default_text

        current_context_input = st.text_area(prompt_label, value=display_value, height=250, key="default_prompt_area")

        # Update context if text area changes
        if current_context_input != display_value:
            st.session_state["context"] = current_context_input
            prompt_changed = True
        # If selection changed, update context to the new default
        elif selected_prompt_name != st.session_state.get("_last_selected_prompt"):
             st.session_state["context"] = default_text
             prompt_changed = True # Flag that context might need update

        # Track the last selected prompt name to manage edits vs defaults
        st.session_state["_last_selected_prompt"] = selected_prompt_name

        # Reset button
        if st.button("Reset to Default Template"):
            st.session_state["context"] = default_prompts.get(selected_prompt_name, "")
            st.rerun()

    # Ensure context is always set
    if "context" not in st.session_state or prompt_changed:
         st.session_state["context"] = current_context_input

    st.session_state["use_default_context"] = (selected_prompt_name != "Custom Prompt")

    # --- Model Selection ---
    st.subheader("Model Selection")
    model_options = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "Custom Model"] # Add more as needed
    # Ensure current choice is valid, default to gpt-4o if not
    current_model = st.session_state.get("model_choice", "gpt-4o")
    if current_model not in model_options:
        current_model = "gpt-4o"

    model_choice = st.selectbox(
        "Select GPT Model:",
        model_options,
        index=model_options.index(current_model),
        key="model_select",
        help="Select the OpenAI model to use for scoring."
    )
    st.session_state["model_choice"] = model_choice

    chosen_model = ""
    if model_choice == "Custom Model":
        custom_model = st.text_input(
            "Enter Custom Model Name:",
            value=st.session_state.get("custom_model", ""),
            key="custom_model_input",
            help="Enter a valid OpenAI model identifier (e.g., fine-tuned model ID)."
        )
        st.session_state["custom_model"] = custom_model
        chosen_model = custom_model.strip() if custom_model else ""
    else:
        chosen_model = model_choice
        st.session_state["custom_model"] = "" # Clear custom model if standard is selected

    st.session_state["chosen_model"] = chosen_model
    if chosen_model:
        st.write(f"Using model: `{chosen_model}`")
    else:
        st.warning("Please select or enter a model name.")


    # --- Data Upload and Configuration ---
    st.subheader("Data Upload & Column Selection")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file with responses **and** manual scores (if comparing)",
        type=["csv", "xlsx", "xls"],
        key="file_uploader",
        help="The file should contain a column with student responses. If computing IRR, it must also contain columns with the corresponding manual scores."
    )

    # Process uploaded file
    if uploaded_file:
        # Check if it's a new file upload
        if uploaded_file != st.session_state.get("uploaded_file_obj"):
             st.session_state["uploaded_file_obj"] = uploaded_file # Store file object
             # Load data and clear previous results if file changes
             df = load_data(uploaded_file)
             st.session_state["raw_df"] = df # Store the initially loaded data
             st.session_state["processed_df"] = None # Clear previous processing results
             st.session_state["processing_complete"] = False
             st.session_state["response_column"] = None # Reset selections
             st.session_state["manual_columns"] = []
             st.session_state["compute_irr"] = False
             st.session_state["alignment_verified"] = False
             st.session_state["icc_value"] = None
             st.info("New file uploaded. Please re-select columns.")
        else:
             # Keep existing dataframe if file object hasn't changed
             df = st.session_state.get("raw_df")

        if df is not None:
            st.success(f"Successfully loaded `{uploaded_file.name}` with {len(df)} rows.")
            st.dataframe(df.head(), use_container_width=True)

            # --- Column Selection ---
            st.markdown("**Select Columns:**")
            available_columns = df.columns.tolist()

            # Response Column
            resp_col_index = 0
            if st.session_state.get("response_column") in available_columns:
                 resp_col_index = available_columns.index(st.session_state["response_column"])
            elif "response" in available_columns: # Default guess
                 resp_col_index = available_columns.index("response")

            response_column = st.selectbox(
                "1. Column containing student responses:",
                options=available_columns,
                index=resp_col_index,
                key="response_column_select"
            )
            st.session_state["response_column"] = response_column

            # --- IRR Configuration ---
            st.markdown("**Inter-Rater Reliability (Optional):**")
            compute_irr = st.toggle(
                 "Compare GPT scores with manual scores?",
                 value=st.session_state.get("compute_irr", False),
                 key="compute_irr_toggle",
                 help="Enable this to calculate ICC between GPT and manual scores from your file."
            )
            st.session_state["compute_irr"] = compute_irr

            if compute_irr:
                st.info("Ensure the manual score columns exist in the uploaded file.")
                # Manual Score Columns (using multiselect for single/multiple)
                # Filter out the selected response column from options
                manual_col_options = [col for col in available_columns if col != response_column]

                # Try to preserve selection if options are still valid
                current_selection = st.session_state.get("manual_columns", [])
                valid_selection = [col for col in current_selection if col in manual_col_options]

                manual_columns = st.multiselect(
                    "2. Column(s) containing manual scores:",
                    options=manual_col_options,
                    default=valid_selection,
                    key="manual_columns_select",
                    help="Select one or more columns from your file that contain the human scores."
                )
                st.session_state["manual_columns"] = manual_columns
                if not manual_columns:
                     st.warning("Please select at least one manual score column to compute IRR.")

            # --- Validation and Proceed Button ---
            can_proceed = True
            error_messages = []

            if not chosen_model:
                 error_messages.append("Model is not selected.")
                 can_proceed = False
            if not st.session_state.get("context", "").strip():
                 error_messages.append("Prompt context is empty.")
                 can_proceed = False
            if not response_column:
                 error_messages.append("Response column is not selected.")
                 can_proceed = False
            if compute_irr and not manual_columns:
                 error_messages.append("Manual score column(s) must be selected when IRR is enabled.")
                 can_proceed = False

            if not can_proceed:
                 st.error("Please resolve the following issues before proceeding:\n- " + "\n- ".join(error_messages))
            else:
                 st.success("Setup complete. Ready to process.")
                 if st.button("➡️ Proceed to Processing", type="primary", use_container_width=True):
                     # Mark setup as complete? Optional flag if needed
                     # st.session_state["setup_complete"] = True
                     st.session_state["current_step"] = "process"
                     st.rerun()

    else:
        st.info("Upload a file to begin configuration.")
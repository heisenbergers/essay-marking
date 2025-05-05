import streamlit as st
import pandas as pd
from utils.helpers import load_prompts
from core.data_processing import load_data
from core.api_handler import get_llm_client # Import factory
import time # Added missing import for time.sleep

# --- Define Available Models ---
AVAILABLE_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    "gemini": ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.0-pro"],
    "claude": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
    # Add other providers and their models here
}

def render_setup_screen():
    """Render the setup screen UI elements."""
    st.header("1. Setup Configuration")

    # --- Provider Selection ---
    st.subheader("Select LLM Provider")
    provider_options = list(AVAILABLE_MODELS.keys())
    default_provider_index = 0
    if st.session_state.get("selected_provider") in provider_options:
        default_provider_index = provider_options.index(st.session_state.selected_provider)

    selected_provider = st.selectbox(
        "Choose the AI Provider:",
        options=provider_options,
        index=default_provider_index,
        key="provider_select"
    )
    # Update state only if selection changes to avoid unnecessary reruns
    if selected_provider != st.session_state.get("selected_provider"):
        st.session_state["selected_provider"] = selected_provider
        # Reset model choice when provider changes
        st.session_state["model_choice"] = AVAILABLE_MODELS.get(selected_provider, [""])[0]
        st.session_state["chosen_model"] = st.session_state["model_choice"]
        st.rerun() # Rerun to update model list and key input

    # --- API Key Input (Conditional based on provider) ---
    st.subheader(f"{selected_provider.capitalize()} API Key")

    provider_key_name = f"{selected_provider}_api_key" # e.g., openai_api_key
    is_verified = st.session_state.get("api_key_verified", {}).get(selected_provider, False)
    current_key = st.session_state.get(provider_key_name, "")

    api_key_col1, api_key_col2 = st.columns([3, 1])

    with api_key_col1:
        key_label_map = {
            "openai": "OpenAI API Key",
            "gemini": "Google AI Studio API Key",
            "claude": "Anthropic API Key"
        }
        key_label = key_label_map.get(selected_provider, f"{selected_provider.capitalize()} API Key")

        provided_key = st.text_input(
            f"Enter your {key_label}:",
            type="password",
            key=f"{provider_key_name}_input",
            help="Your key is used only for this session and is not saved.",
            value=current_key if not is_verified else ("*" * 10 if current_key else ""), # Mask if verified
            disabled=is_verified
        )

    with api_key_col2:
        st.write("") # Align button vertically
        st.write("")
        if is_verified and current_key:
            if st.button(f"Change Key", key=f"change_key_{selected_provider}"):
                st.session_state["api_key_verified"][selected_provider] = False
                st.session_state[provider_key_name] = ""
                st.rerun()
        else:
            if st.button("Verify Key", key=f"verify_key_{selected_provider}"):
                actual_key_to_verify = provided_key if provided_key else current_key
                if actual_key_to_verify:
                    with st.spinner(f"Verifying {selected_provider.capitalize()} key..."):
                        try:
                            # Validate key using the factory and the specific provider's test
                            client = get_llm_client(provider=selected_provider, api_key=actual_key_to_verify)
                            # test_connection is called during init or explicitly if needed
                            # client.test_connection()
                            st.session_state[provider_key_name] = actual_key_to_verify
                            st.session_state.api_key_verified[selected_provider] = True
                            st.success(f"{selected_provider.capitalize()} API Key verified.")
                            time.sleep(1) # Brief pause for user to see success
                            st.rerun()
                        except ValueError as ve: # Catch specific errors from get_llm_client/test
                            st.error(f"Verification failed: {ve}")
                            st.session_state[provider_key_name] = "" # Clear invalid key attempt
                            st.session_state.api_key_verified[selected_provider] = False
                        except Exception as e: # Catch unexpected errors
                            st.error(f"Verification error: {e}")
                            st.session_state[provider_key_name] = ""
                            st.session_state.api_key_verified[selected_provider] = False
                else:
                    st.warning("Please enter an API key to verify.")

    # --- Model Selection (Conditional based on provider) ---
    st.subheader("Model Selection")
    provider_models = AVAILABLE_MODELS.get(selected_provider, [])
    model_options = provider_models + ["Custom Model"]

    # Get current choice, default to first in list if invalid for current provider
    current_model_choice = st.session_state.get("model_choice", model_options[0])
    if current_model_choice not in model_options:
        current_model_choice = model_options[0]

    try:
        model_index = model_options.index(current_model_choice)
    except ValueError:
        model_index = 0 # Default to first option if current choice is somehow invalid

    model_choice_selected = st.selectbox(
        f"Select {selected_provider.capitalize()} Model:",
        model_options,
        index=model_index,
        key="model_select",
        help=f"Select the {selected_provider} model to use for scoring."
    )
    # Update state only if selection changes
    if model_choice_selected != st.session_state.get("model_choice"):
        st.session_state["model_choice"] = model_choice_selected
        st.rerun() # Rerun to update chosen_model state if needed

    chosen_model_final = ""
    if st.session_state.model_choice == "Custom Model":
        custom_model = st.text_input(
            "Enter Custom Model Name/ID:",
            value=st.session_state.get("custom_model", ""),
            key="custom_model_input",
            help="Enter a valid model identifier for the selected provider."
        )
        # Update state only if input changes
        if custom_model != st.session_state.get("custom_model"):
            st.session_state["custom_model"] = custom_model
            st.rerun()
        chosen_model_final = st.session_state.custom_model.strip()
    else:
        chosen_model_final = st.session_state.model_choice
        if st.session_state.get("custom_model"): # Clear custom model if standard is selected
            st.session_state["custom_model"] = ""
            # No rerun needed here as chosen_model_final is already set

    # Update the final chosen model used for processing
    st.session_state["chosen_model"] = chosen_model_final

    if chosen_model_final:
        st.write(f"Using Provider: `{selected_provider}`, Model: `{chosen_model_final}`")
    else:
        st.warning("Please select or enter a model name.")


    # --- Prompt Configuration ---
    st.subheader("Prompt Configuration")
    default_prompts = load_prompts()
    prompt_options = ["Custom Prompt"] + list(default_prompts.keys())
    # Default to first available template if current selection invalid
    current_prompt_name = st.session_state.get("selected_prompt_name", prompt_options[1] if len(prompt_options) > 1 else prompt_options[0])
    if current_prompt_name not in prompt_options:
        current_prompt_name = prompt_options[1] if len(prompt_options) > 1 else prompt_options[0]

    selected_prompt_name = st.selectbox(
        "Select Prompt Template:",
        options=prompt_options,
        index=prompt_options.index(current_prompt_name),
        key="prompt_select"
    )

    # Update selected prompt name state
    if selected_prompt_name != st.session_state.get("selected_prompt_name"):
        st.session_state["selected_prompt_name"] = selected_prompt_name
        # If selection changes, reset context to the new default (unless it was custom)
        if selected_prompt_name != "Custom Prompt":
             st.session_state["context"] = default_prompts.get(selected_prompt_name, "")
             # Reset custom context when a template is chosen
             st.session_state["custom_context"] = ""
        st.rerun()


    # Display text area for the prompt
    prompt_changed = False
    if st.session_state.selected_prompt_name == "Custom Prompt":
        # st.session_state["prompt_choice"] = "Use custom prompt" # Less needed now
        prompt_label = "Enter Custom Prompt:"
        # Use existing custom context if available, otherwise empty
        default_value = st.session_state.get("custom_context", "")
        current_context_input = st.text_area(prompt_label, value=default_value, height=250, key="custom_prompt_area")
        if current_context_input != default_value:
             st.session_state["custom_context"] = current_context_input # Save custom prompt separately
             st.session_state["context"] = current_context_input # Main context used by app
             prompt_changed = True # Flag that context was updated

    else: # A default prompt is selected
        # st.session_state["prompt_choice"] = "Use default prompt" # Less needed now
        prompt_label = f"Selected Prompt: {st.session_state.selected_prompt_name} (Editable for this session)"
        # Use current session context if it's different from the stored default (means it was edited)
        default_text = default_prompts.get(st.session_state.selected_prompt_name, "")
        current_session_context = st.session_state.get("context", default_text)

        # Display current context, allow editing
        current_context_input = st.text_area(
            prompt_label,
            value=current_session_context, # Show current potentially edited value
            height=250,
            key="default_prompt_area"
        )

        # Update context if text area changes
        if current_context_input != current_session_context:
            st.session_state["context"] = current_context_input
            prompt_changed = True

        # Reset button for templates
        if st.button("Reset to Default Template"):
            st.session_state["context"] = default_prompts.get(st.session_state.selected_prompt_name, "")
            st.rerun()

    # Ensure context is always set if changed
    if prompt_changed:
         st.session_state["context"] = current_context_input # Make sure state reflects latest input


    # --- Prompt Output Format Selection ---
    st.markdown("**Expected Output Format:**")
    st.caption("Select the format you expect the model to return based on your prompt.")
    format_options = {
        "JSON (Score and Reason)": "json_score_reason",
        "Integer Score Only": "integer_score",
        "Raw Text": "raw_text"
    }
    # Get the display name from the current state value
    current_format_value = st.session_state.get("prompt_output_format", "json_score_reason")
    # Handle case where format might be invalid somehow
    current_format_display = next((k for k, v in format_options.items() if v == current_format_value), list(format_options.keys())[0])

    selected_format_display = st.selectbox(
        "Select Format:",
        options=list(format_options.keys()),
        index=list(format_options.keys()).index(current_format_display),
        key="prompt_format_select",
        help="""
        - 'JSON': Expects `{"score": N, "reason": "..."}`. Will attempt to parse score/reason.
        - 'Integer Score Only': Expects only a number. Will attempt to extract the number as the score.
        - 'Raw Text': Takes the entire model output as the result (no parsing).
        """
    )
    # Update state if format selection changes
    new_format_value = format_options[selected_format_display]
    if new_format_value != st.session_state.get("prompt_output_format"):
        st.session_state["prompt_output_format"] = new_format_value
        st.rerun() # Rerun if format changes


    # --- Data Upload and Configuration ---
    st.subheader("Data Upload & Column Selection")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file with responses **and** manual scores (if comparing)",
        type=["csv", "xlsx", "xls"],
        key="file_uploader",
        help="The file should contain a column with responses. If computing IRR, include columns with manual scores."
    )

    # Process uploaded file
    if uploaded_file:
        if uploaded_file != st.session_state.get("uploaded_file_obj"):
             st.session_state["uploaded_file_obj"] = uploaded_file
             with st.spinner("Loading data..."):
                 df = load_data(uploaded_file)
             st.session_state["raw_df"] = df
             # Reset downstream state on new file upload
             st.session_state["processed_df"] = None
             st.session_state["analyzed_df"] = None
             st.session_state["processing_complete"] = False
             st.session_state["analysis_computed"] = False
             st.session_state["response_column"] = None
             st.session_state["manual_columns"] = []
             st.session_state["compute_irr"] = False
             st.session_state["alignment_verified"] = False
             st.session_state["icc_value"] = None
             st.session_state["ratings_matrix_for_viz"] = None
             st.info("New file uploaded. Please re-select columns below.")
             st.rerun() # Rerun immediately after loading new data and resetting state
        else:
             # Keep existing dataframe if file object hasn't changed
             df = st.session_state.get("raw_df")

        if df is not None:
            st.success(f"Successfully loaded `{uploaded_file.name}` with {len(df)} rows.")
            st.dataframe(df.head(), use_container_width=True)

            # --- Column Selection ---
            st.markdown("**Select Columns:**")
            available_columns = [""] + df.columns.tolist() # Add empty option

            # Response Column
            default_resp_col = None
            if st.session_state.get("response_column") in df.columns.tolist():
                default_resp_col = st.session_state.response_column
            elif "response" in df.columns.tolist(): # Auto-select 'response' if exists
                 default_resp_col = "response"

            response_column = st.selectbox(
                "1. Column containing text responses to score:",
                options=available_columns,
                index=available_columns.index(default_resp_col) if default_resp_col else 0,
                key="response_column_select"
            )
            # Update state only on change
            if response_column != st.session_state.get("response_column"):
                st.session_state["response_column"] = response_column if response_column else None
                st.rerun()


            # --- IRR Configuration ---
            st.markdown("**Inter-Rater Reliability (Optional):**")
            compute_irr = st.toggle(
                 "Compare LLM scores with manual scores?",
                 value=st.session_state.get("compute_irr", False),
                 key="compute_irr_toggle",
                 help="Enable this to calculate ICC between LLM and manual scores from your file."
            )
            # Update state only on change
            if compute_irr != st.session_state.get("compute_irr"):
                st.session_state["compute_irr"] = compute_irr
                st.rerun()


            if st.session_state.compute_irr:
                st.info("Ensure the manual score columns exist in the uploaded file.")
                # Manual Score Columns (using multiselect for single/multiple)
                # Filter out the selected response column from options
                manual_col_options = [col for col in df.columns.tolist() if col != st.session_state.response_column]

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
                 # Update state only on change
                if manual_columns != st.session_state.get("manual_columns"):
                    st.session_state["manual_columns"] = manual_columns
                    st.rerun()

                if not st.session_state.manual_columns:
                     st.warning("Please select at least one manual score column to compute IRR.")

            # --- Validation and Proceed Button ---
            st.divider()
            can_proceed = True
            error_messages = []

            # Check API key verification for the *selected* provider
            if not st.session_state.get("api_key_verified", {}).get(selected_provider, False):
                 error_messages.append(f"{selected_provider.capitalize()} API Key is not provided or verified.")
                 can_proceed = False

            if not st.session_state.chosen_model:
                 error_messages.append("Model is not selected or entered.")
                 can_proceed = False
            if not st.session_state.get("context", "").strip():
                 error_messages.append("Prompt context is empty.")
                 can_proceed = False
            if not st.session_state.response_column:
                 error_messages.append("Response column is not selected.")
                 can_proceed = False
            if st.session_state.compute_irr and not st.session_state.manual_columns:
                 error_messages.append("Manual score column(s) must be selected when IRR is enabled.")
                 can_proceed = False

            if not can_proceed:
                 st.error("Please resolve the following issues before proceeding:\n- " + "\n- ".join(error_messages))
                 # Disable button explicitly
                 st.button("➡️ Proceed to Processing", type="primary", use_container_width=True, disabled=True)
            else:
                 st.success("Setup complete. Ready to process.")
                 if st.button("➡️ Proceed to Processing", type="primary", use_container_width=True):
                     st.session_state["current_step"] = "process"
                     st.rerun()

    else:
        st.info("Upload a file to begin configuration.")
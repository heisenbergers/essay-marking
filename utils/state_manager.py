import streamlit as st
import uuid
from utils.helpers import load_prompts # Need this for default context

def get_default_context():
    """Gets the context from the first loaded default prompt."""
    prompts = load_prompts()
    if prompts:
        return next(iter(prompts.values()))
    else:
        return """Score the following text on a scale of 1-10. Respond in JSON format: {"score": N, "reason": "Your reasoning"}."""

def init_session_state():
    """Initializes the session state with default values."""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

    if "initialized" not in st.session_state:
        st.session_state["initialized"] = True

        # --- Core Workflow State ---
        st.session_state["current_step"] = "setup"
        st.session_state["processing_complete"] = False
        st.session_state["analysis_computed"] = False
        st.session_state["error_message"] = None

        # --- Data State ---
        st.session_state["uploaded_file_obj"] = None
        st.session_state["raw_df"] = None
        st.session_state["processed_df"] = None # Holds results after LLM processing
        st.session_state["analyzed_df"] = None # Holds df after analysis steps (e.g., numeric conversion)
        st.session_state["icc_value"] = None
        st.session_state["ratings_matrix_for_viz"] = None

        # --- Configuration State ---
        # API Keys (Store separately)
        st.session_state["openai_api_key"] = None
        st.session_state["gemini_api_key"] = None
        st.session_state["anthropic_api_key"] = None
        st.session_state["openrouter_api_key"] = None
        st.session_state["deepseek_api_key"] = None # Added DeepSeek key state

        # Track verification per provider
        st.session_state["api_key_verified"] = {
            "openai": False,
            "gemini": False,
            "claude": False,
            "openrouter": False,
            "deepseek": False # Added DeepSeek verification state
        }

        # Model Selection
        st.session_state["selected_provider"] = "openai" # Default provider
        st.session_state["model_choice"] = "gpt-4o" # Default model for OpenAI
        st.session_state["custom_model"] = "" # Specific custom model ID
        st.session_state["chosen_model"] = "gpt-4o" # Final model string used in API call
        st.session_state["_last_selected_provider"] = "openai" # Helper for UI model reset

        # Prompt related state
        default_context = get_default_context()
        st.session_state["context"] = default_context
        st.session_state["custom_context"] = ""
        st.session_state["prompt_choice"] = "Use default prompt"
        # Initialize selected_prompt_name safely
        loaded_prompts = load_prompts()
        st.session_state["selected_prompt_name"] = next(iter(loaded_prompts.keys()), "Default Fallback") if loaded_prompts else "Default Fallback"

        st.session_state["prompt_output_format"] = "json_score_reason" # Default format
        st.session_state["_last_selected_prompt"] = st.session_state["selected_prompt_name"]


        # Column Selections
        st.session_state["response_column"] = None
        st.session_state["manual_columns"] = []

        # IRR related
        st.session_state["compute_irr"] = False
        st.session_state["alignment_step_verified"] = False # For analysis screen data check


def reset_session(preserve_keys=None):
    """
    Resets the session state, optionally preserving some API keys and their verification.
    All other states are reset to their initial defaults.
    """
    if preserve_keys is None:
        preserve_keys = []

    # Store values to preserve
    preserved_values = {}
    for key in preserve_keys:
        if f"{key}_api_key" in st.session_state:
            preserved_values[f"{key}_api_key"] = st.session_state[f"{key}_api_key"]
        if "api_key_verified" in st.session_state and key in st.session_state.api_key_verified:
            if "api_key_verified" not in preserved_values:
                preserved_values["api_key_verified"] = {}
            preserved_values["api_key_verified"][key] = st.session_state.api_key_verified[key]

    # Clear all session state keys
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        try:
            del st.session_state[key]
        except KeyError:
            pass # Key might have been deleted already

    # Restore preserved values
    for key, value in preserved_values.items():
        if value is not None:
            st.session_state[key] = value

    # Re-initialize defaults for non-preserved keys
    st.session_state["initialized"] = False # Force re-initialization
    init_session_state() # Ensure session ID is consistent if it was cleared
    st.session_state["current_step"] = "setup" # Reset to setup step
    st.success("Session reset. Please reconfigure your setup.")
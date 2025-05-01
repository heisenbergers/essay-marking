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
        st.session_state["google_api_key"] = None
        st.session_state["anthropic_api_key"] = None
        # Track verification per provider
        st.session_state["api_key_verified"] = {"openai": False, "gemini": False, "claude": False}

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
        st.session_state["selected_prompt_name"] = next(iter(load_prompts().keys()), "Default Fallback")
        st.session_state["prompt_output_format"] = "json_score_reason" # Default format
        # st.session_state["use_default_context"] = True # Less relevant now, format is explicit
        st.session_state["_last_selected_prompt"] = st.session_state["selected_prompt_name"]

        # Column Selections
        st.session_state["response_column"] = None
        st.session_state["manual_columns"] = []

        # IRR related
        st.session_state["compute_irr"] = False
        st.session_state["alignment_verified"] = False


def reset_session():
    """Resets the session state for a new analysis run."""
    current_session_id = st.session_state.get("session_id", str(uuid.uuid4()))

    # List of keys to preserve (e.g., API keys, potentially provider/model choice)
    # For now, let's preserve API keys but reset other selections
    keys_to_preserve = ['session_id', 'openai_api_key', 'google_api_key', 'anthropic_api_key', 'api_key_verified']
    preserved_values = {key: st.session_state.get(key) for key in keys_to_preserve}

    # Clear all keys
    keys_to_clear = [k for k in st.session_state.keys()]
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
    init_session_state()

    # Ensure session ID is consistent if it existed
    st.session_state["session_id"] = current_session_id

    st.success("Session reset. API keys preserved. Ready for new analysis.")
    st.rerun()
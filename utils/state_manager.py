import streamlit as st
import uuid
from utils.helpers import load_prompts # Need this for default context

def get_default_context():
    """Gets the context from the first loaded default prompt."""
    prompts = load_prompts()
    if prompts:
        # Return the content of the first prompt found
        return next(iter(prompts.values()))
    else:
        # Fallback if load_prompts returns empty
        return """Score the following text on a scale of 1-10. Respond in JSON format: {"score": N, "reason": "Your reasoning"}."""

def init_session_state():
    """Initializes the session state with default values."""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

    # Initialize or reset state variables if session restarts or first load
    # Using a flag to prevent re-initialization on simple reruns
    if "initialized" not in st.session_state:
        st.session_state["initialized"] = True # Mark as initialized

        # --- Core Workflow State ---
        st.session_state["current_step"] = "setup" # setup, process, analysis
        st.session_state["processing_complete"] = False
        st.session_state["analysis_computed"] = False
        st.session_state["error_message"] = None

        # --- Data State ---
        st.session_state["uploaded_file_obj"] = None # Store the actual file object
        st.session_state["raw_df"] = None # Dataframe as loaded initially
        st.session_state["processed_df"] = None # Dataframe after GPT processing + numeric cols
        st.session_state["icc_value"] = None
        st.session_state["ratings_matrix_for_viz"] = None # Data used for ICC/viz

        # --- Configuration State ---
        st.session_state["user_api_key"] = None # ADDED: To store the user's key for the session
        st.session_state["api_key_verified"] = False # ADDED: Flag to track if the session key is valid
        st.session_state["model_choice"] = "gpt-4o" # Default model
        st.session_state["custom_model"] = ""
        st.session_state["chosen_model"] = "gpt-4o"

        # Prompt related state
        default_context = get_default_context()
        st.session_state["context"] = default_context
        st.session_state["custom_context"] = "" # Store custom prompt separately
        st.session_state["prompt_choice"] = "Use default prompt" # Tracks if default or custom radio is selected
        st.session_state["selected_prompt_name"] = next(iter(load_prompts().keys()), "Default Fallback") # Name of selected template
        st.session_state["use_default_context"] = True # Flag derived from prompt_choice
        st.session_state["_last_selected_prompt"] = st.session_state["selected_prompt_name"] # Helper for edits

        # Column Selections
        st.session_state["response_column"] = None
        st.session_state["manual_columns"] = []

        # IRR related
        st.session_state["compute_irr"] = False
        st.session_state["alignment_verified"] = False

        # Add any other default states needed here


def reset_session():
    """Resets the session state for a new analysis run, keeping essential config."""
    # Store values to potentially keep (like chosen model, maybe prompt settings)
    # For now, just resetting most things for a clean slate
    current_session_id = st.session_state.get("session_id", str(uuid.uuid4()))

    # Clear all keys except the 'initialized' flag and session_id
    keys_to_clear = [k for k in st.session_state.keys() if k not in ['initialized', 'session_id']]
    for key in keys_to_clear:
        del st.session_state[key]

    # Re-initialize defaults
    st.session_state["initialized"] = False # Force re-initialization
    init_session_state()

    # Restore session ID
    st.session_state["session_id"] = current_session_id
    st.success("Session reset. Ready for new analysis.")
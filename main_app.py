import streamlit as st

# Import utility functions
from utils.state_manager import init_session_state
from utils.helpers import load_prompts # Needed early for state init

# Import UI components
from ui.sidebar import render_sidebar
from ui.components import apply_custom_css, render_header, render_footer
from ui.setup_screen import render_setup_screen
from ui.process_screen import render_process_screen
from ui.analysis_screen import render_analysis_screen

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="GPT Scoring Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main function to run the Streamlit application."""

    # --- Initialize State & Apply Styling ---
    # Load prompts early if needed by state initialization
    _ = load_prompts() # Load prompts into cache
    init_session_state()
    apply_custom_css()

    # --- Render UI Elements ---
    render_sidebar() # Render sidebar first

    # --- Main Content Area ---
    render_header() # Render the main header

    # --- Error Display ---
    if st.session_state.get("error_message"):
        st.error(f"Error: {st.session_state['error_message']}")
        if st.button("Clear Error", key="clear_error_btn"):
            st.session_state["error_message"] = None
            st.rerun()

    # --- Render Current Step Screen ---
    current_step = st.session_state.get("current_step", "setup")

    if current_step == "setup":
        render_setup_screen()
    elif current_step == "process":
        render_process_screen()
    elif current_step == "analysis":
        render_analysis_screen()
    else: # Fallback for unknown step
        st.error("Invalid application step.")
        st.session_state["current_step"] = "setup"
        st.rerun()

    # --- Footer ---
    render_footer()


if __name__ == "__main__":
    main()
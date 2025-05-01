import streamlit as st
from utils.state_manager import reset_session

def render_sidebar():
    """Render the sidebar navigation and controls."""
    with st.sidebar:
        st.title("📊 GPT Scorer")
        st.caption("v3.1 Refactored") # Example version

        # Navigation / Step Indicator
        current_step = st.session_state.get("current_step", "setup")
        steps = ["setup", "process", "analysis"]
        step_names = ["1. Setup", "2. Process", "3. Analysis"]

        # Determine which steps are completed/accessible
        setup_done = st.session_state.get("setup_ Mcomplete", False) # Flag set after setup
        process_done = st.session_state.get("processing_complete", False) # Existing flag
        compute_irr = st.session_state.get("compute_irr", False)

        st.subheader("Workflow")
        for i, (step, name) in enumerate(zip(steps, step_names)):
            is_current = (step == current_step)
            # Determine if step is accessible
            if step == "setup":
                disabled = False
            elif step == "process":
                 # Must have completed setup (e.g., uploaded file and selected columns)
                 disabled = not st.session_state.get("uploaded_file") or not st.session_state.get("response_column")
            elif step == "analysis":
                # Must have completed processing and requested IRR
                disabled = not process_done or not compute_irr
            else: # Future steps?
                disabled = True

            button_type = "primary" if is_current else "secondary"
            if st.button(
                name,
                key=f"nav_{step}",
                use_container_width=True,
                type=button_type,
                disabled=disabled,
                help="Complete previous steps to enable." if disabled else f"Go to {name}"
            ):
                 # Only allow navigation if not disabled
                 if not disabled:
                     st.session_state["current_step"] = step
                     st.rerun() # Rerun to update the main page display

        st.divider()

        # Action Buttons
        st.subheader("Actions")
        if st.button("🔄 Start New Analysis", type="secondary", use_container_width=True, help="Resets data and settings."):
            reset_session() # Keep API key status, reset data
            st.rerun()

        # Add session state download/upload later if implementing persistence

        st.divider()
        st.caption("© 2025 - Built with Streamlit")
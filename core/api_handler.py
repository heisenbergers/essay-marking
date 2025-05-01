import streamlit as st
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import time # Added for potential sleeps in retry

SUPPORTED_SYSTEM_MODELS = ["gpt-4", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini"]

@st.cache_resource # Cache the client for efficiency
def get_openai_client():
    """Initializes and returns the OpenAI client using secrets."""
    try:
        api_key = st.secrets["openai_api_key"]
        if not api_key:
            st.error("OpenAI API key is missing in Streamlit Secrets.")
            return None
        client = OpenAI(api_key=api_key)
        # Test the client connection (optional, but good practice)
        client.models.list()
        return client
    except KeyError:
        st.error("OpenAI API key (`openai_api_key`) not found in Streamlit Secrets.")
        st.info("Please add your OpenAI API key to the Streamlit Cloud dashboard.")
        return None
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return None

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chatGPT(prompt: str, context: str, model: str = "gpt-4o", client: OpenAI = None):
    """
    Call OpenAI API with retry logic and better error handling.

    Args:
        prompt (str): The user prompt/response to be scored.
        context (str): The system context/instructions for the model.
        model (str): The OpenAI model name.
        client (OpenAI): An initialized OpenAI client instance.

    Returns:
        str: The content of the chat completion.

    Raises:
        ValueError: If API key is invalid, rate limit exceeded, model not found, or other API errors.
        Exception: For unexpected errors during the API call.
    """
    if client is None:
        raise ValueError("OpenAI client is not initialized. Check API key in secrets.")

    if not isinstance(prompt, str) or not prompt.strip():
        # Don't call API for empty prompts
        return ""

    if model in SUPPORTED_SYSTEM_MODELS:
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ]
        temperature = 0.3
    else:
        # Append context to user prompt if 'system' role is unsupported or it's a custom model where we are unsure
        st.warning(f"Model '{model}' might not optimally use the 'system' role. Combining context with user prompt.")
        messages = [
            {"role": "user", "content": f"{context}\n\n---\n\nUser Response:\n{prompt}"}
        ]
        temperature = 0.7 # Slightly higher temp for potentially less structured models

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=4000, # Adjust as needed, consider cost
            stream=False
        )
        content = response.choices[0].message.content
        return content if content else "" # Return empty string if response is None or empty

    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            # Re-raise specifically for the app to know it's an auth issue
            raise ValueError("Invalid API key configured in Streamlit Secrets.")
        elif "rate limit" in error_msg.lower():
            raise ValueError("OpenAI Rate limit exceeded. Please wait and try again, or check your usage tier.")
        elif "model" in error_msg.lower() and "does not exist" in error_msg.lower():
            raise ValueError(f"Model '{model}' not found. Please check the model name.")
        else:
            # Catch-all for other API errors
            st.error(f"An unexpected error occurred calling the OpenAI API: {error_msg}")
            raise # Re-raise the original exception after logging
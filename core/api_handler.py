# core/api_handler.py

import streamlit as st
from openai import OpenAI, APIError, AuthenticationError # Import specific exceptions
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)
import time

SUPPORTED_SYSTEM_MODELS = ["gpt-4", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini"]

# Cache the client *based on the API key* for the session
@st.cache_resource
def get_openai_client(api_key: str):
    """Initializes and returns the OpenAI client using the provided API key."""
    if not api_key:
        raise ValueError("API key is missing.")
    try:
        client = OpenAI(api_key=api_key)
        # Test the client connection (optional, but good practice)
        # This call will raise AuthenticationError if the key is invalid
        # REMOVED limit=1 argument as it's not supported in all versions
        client.models.list() # <--- MODIFIED LINE
        return client
    except AuthenticationError:
        st.error("Invalid OpenAI API key provided.")
        raise
    except APIError as e:
        st.error(f"Failed to initialize OpenAI client due to API error: {e}")
        raise
    except Exception as e:
        st.error(f"An unexpected error occurred initializing OpenAI client: {e}")
        raise

# Define specific API errors to retry on (e.g., rate limits, server errors)
retryable_errors = (APIError,) # Add RateLimitError etc. if needed from openai exceptions

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(retryable_errors) # Only retry specific errors
)
def chatGPT(prompt: str, context: str, model: str = "gpt-4o", api_key: str = None):
    """
    Call OpenAI API with retry logic and better error handling.
    Uses the API key provided for the session.

    Args:
        prompt (str): The user prompt/response to be scored.
        context (str): The system context/instructions for the model.
        model (str): The OpenAI model name.
        api_key (str): The user's API key for the current session.

    Returns:
        str: The content of the chat completion.

    Raises:
        ValueError: If API key is missing or invalid, model not found, or other non-retryable API errors.
        APIError: If retryable API errors persist after retries.
        Exception: For unexpected errors during the API call.
    """
    # ... (rest of chatGPT function remains the same) ...
    if not api_key:
        raise ValueError("API Key not found in session state.")

    if not isinstance(prompt, str) or not prompt.strip():
        return ""

    try:
        client = get_openai_client(api_key=api_key)

        if model in SUPPORTED_SYSTEM_MODELS:
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": prompt}
            ]
            temperature = 0.3
        else:
            st.warning(f"Model '{model}' might not optimally use the 'system' role. Combining context with user prompt.")
            messages = [
                {"role": "user", "content": f"{context}\n\n---\n\nUser Response:\n{prompt}"}
            ]
            temperature = 0.7

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=4000,
            stream=False
        )
        content = response.choices[0].message.content
        return content if content else ""

    except AuthenticationError:
        st.error("Invalid API key. Please check and re-enter your key in Step 1.")
        raise ValueError("Invalid API key.")
    except APIError as e:
         error_msg = str(e)
         if "rate limit" in error_msg.lower():
             st.error("OpenAI Rate limit exceeded. Please wait and try again, check your usage tier, or reduce processing.")
             raise ValueError("Rate limit exceeded.")
         elif "model" in error_msg.lower() and ("not found" in error_msg.lower() or "does not exist" in error_msg.lower()):
             st.error(f"Model '{model}' not found or unavailable with your key. Please check the model name or your key's permissions.")
             raise ValueError(f"Model '{model}' not found.")
         else:
             st.error(f"An API error occurred: {error_msg}")
             raise
    except Exception as e:
        st.error(f"An unexpected error occurred calling the OpenAI API: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        raise
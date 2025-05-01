import streamlit as st
import time
from abc import ABC, abstractmethod # Use Abstract Base Classes

# --- OpenAI ---
from openai import OpenAI, APIError, AuthenticationError as OpenAIAuthenticationError
from tenacity import retry as openai_retry, stop_after_attempt as openai_stop, wait_random_exponential as openai_wait, retry_if_exception_type as openai_retry_if

# --- Gemini ---
import google.generativeai as genai
from google.api_core import exceptions as GoogleAPIErrors # Specific Gemini errors

# --- Claude ---
from anthropic import Anthropic, APIError as AnthropicAPIError, AuthenticationError as AnthropicAuthenticationError
from tenacity import retry as anthropic_retry, stop_after_attempt as anthropic_stop, wait_random_exponential as anthropic_wait, retry_if_exception_type as anthropic_retry_if

# --- Abstract Base Class ---
class LLMClient(ABC):
    """Abstract Base Class for LLM API clients."""
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required.")
        self.api_key = api_key
        self._initialize_client() # Initialize in constructor

    @abstractmethod
    def _initialize_client(self):
        """Initialize the specific API client."""
        pass

    @abstractmethod
    def generate(self, prompt: str, context: str, model_name: str, **kwargs) -> str:
        """Generate text using the LLM."""
        pass

    @abstractmethod
    def test_connection(self):
        """Test if the API key and connection are valid."""
        pass

# --- OpenAI Implementation ---
# Define specific API errors to retry on for OpenAI
openai_retryable_errors = (APIError,)

class OpenAIClient(LLMClient):
    """Client for OpenAI API."""
    SUPPORTED_SYSTEM_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"] # Add others like gpt-4 etc.

    def _initialize_client(self):
        try:
            self.client = OpenAI(api_key=self.api_key)
            self.test_connection() # Test on init
        except OpenAIAuthenticationError:
            # Don't show streamlit error here, just raise
            raise ValueError("Invalid OpenAI API key provided.")
        except Exception as e:
            # Don't show streamlit error here, just raise
            raise ValueError(f"Failed to initialize OpenAI client: {e}")

    def test_connection(self):
        try:
            self.client.models.list() # Simple call to test auth
        except OpenAIAuthenticationError as e:
             raise ValueError(f"OpenAI Authentication Error: {e}")
        except APIError as e:
            raise ValueError(f"OpenAI API Error during connection test: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error during OpenAI connection test: {e}")

    @openai_retry(
        wait=openai_wait(min=1, max=30), # Shorter max wait for retries
        stop=openai_stop(5), # Fewer attempts
        retry=openai_retry_if(openai_retry_if(openai_retryable_errors))
    )
    def generate(self, prompt: str, context: str, model_name: str, **kwargs) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            return ""

        try:
            if model_name in self.SUPPORTED_SYSTEM_MODELS:
                 messages = [{"role": "system", "content": context},{"role": "user", "content": prompt}]
                 temperature = kwargs.get("temperature", 0.3)
            else:
                 st.warning(f"Model '{model_name}' might not optimally use the 'system' role. Combining context.")
                 messages = [{"role": "user", "content": f"{context}\n\n---\n\nUser Response:\n{prompt}"}]
                 temperature = kwargs.get("temperature", 0.7)

            response = self.client.chat.completions.create(
                model=model_name, messages=messages, temperature=temperature,
                max_tokens=kwargs.get("max_tokens", 4000), stream=False )
            content = response.choices[0].message.content
            return content if content else ""

        except OpenAIAuthenticationError: # Should be caught by init, but possible if key revoked mid-session
            raise ValueError("Invalid OpenAI API key.")
        except APIError as e:
             error_msg = str(e)
             if "rate limit" in error_msg.lower(): raise ValueError("OpenAI Rate limit exceeded.")
             elif "model" in error_msg.lower() and ("not found" in error_msg.lower() or "does not exist" in error_msg.lower()):
                 raise ValueError(f"OpenAI Model '{model_name}' not found or unavailable.")
             else: raise ValueError(f"OpenAI API error: {error_msg}") # Raise as ValueError for consistent handling
        except Exception as e:
            raise Exception(f"Unexpected error calling OpenAI API: {e}") # Raise generic for unexpected

# --- Gemini Implementation ---
class GeminiClient(LLMClient):
    """Client for Google Gemini API."""
    def _initialize_client(self):
        try:
            genai.configure(api_key=self.api_key)
            # Client is implicit, test connection now
            self.test_connection()
        except Exception as e:
             raise ValueError(f"Failed to configure Gemini: {e}")

    def test_connection(self):
         try:
             models = genai.list_models()
             if not any('generateContent' in m.supported_generation_methods for m in models):
                  raise ValueError("No text generation models found for your Gemini API key.")
         except GoogleAPIErrors.PermissionDenied:
              raise ValueError("Invalid Google Gemini API Key (Permission Denied).")
         except GoogleAPIErrors.Unauthenticated:
              raise ValueError("Google Gemini API Key not valid or not configured correctly.")
         except Exception as e:
             raise ValueError(f"Error testing Gemini connection: {e}")

    # Add Gemini-specific retry logic if desired
    def generate(self, prompt: str, context: str, model_name: str, **kwargs) -> str:
         if not isinstance(prompt, str) or not prompt.strip():
            return ""
         try:
            model_instance = genai.GenerativeModel(model_name)
            full_prompt = f"{context}\n\n---\n\nUser Response:\n{prompt}"

            generation_config = genai.types.GenerationConfig(
                max_output_tokens=kwargs.get("max_tokens", 4000),
                temperature=kwargs.get("temperature", 0.5) )

            response = model_instance.generate_content(
                 full_prompt, generation_config=generation_config, stream=False)

            if not response.candidates:
                 block_reason = response.prompt_feedback.block_reason.name if response.prompt_feedback else "Unknown"
                 return f"ERROR: Response blocked by safety filter ({block_reason})"

            return response.text
         except GoogleAPIErrors.PermissionDenied:
              raise ValueError("Invalid Google Gemini API Key (Permission Denied).")
         except GoogleAPIErrors.InvalidArgument as e:
              raise ValueError(f"Invalid argument for Gemini model '{model_name}': {e}")
         except GoogleAPIErrors.ResourceExhausted as e: # Rate limiting
              raise ValueError(f"Gemini Rate limit exceeded: {e}")
         except Exception as e:
            raise Exception(f"Unexpected error calling Google Gemini API: {e}")

# --- Claude Implementation ---
# Define specific API errors to retry on for Anthropic
anthropic_retryable_errors = (AnthropicAPIError,) # Add RateLimitError from anthropic.types if needed

class ClaudeClient(LLMClient):
    """Client for Anthropic Claude API."""
    def _initialize_client(self):
        try:
            self.client = Anthropic(api_key=self.api_key)
            self.test_connection() # Test on init
        except AnthropicAuthenticationError:
             raise ValueError("Invalid Anthropic API key provided.")
        except Exception as e:
            raise ValueError(f"Failed to initialize Anthropic client: {e}")

    def test_connection(self):
         try:
             # Make a minimal call to test authentication
             self.client.count_tokens("test")
         except AnthropicAuthenticationError as e:
             raise ValueError(f"Anthropic Authentication Error: {e}")
         except AnthropicAPIError as e:
             raise ValueError(f"Anthropic API Error during connection test: {e}")
         except Exception as e:
             raise ValueError(f"Unexpected error during Anthropic connection test: {e}")

    @anthropic_retry(
        wait=anthropic_wait(min=1, max=30),
        stop=anthropic_stop(5),
        retry=anthropic_retry_if(anthropic_retry_if(anthropic_retryable_errors))
    )
    def generate(self, prompt: str, context: str, model_name: str, **kwargs) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            return ""
        try:
            response = self.client.messages.create(
                model=model_name, system=context,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 4000),
                temperature=kwargs.get("temperature", 0.5) )

            if response.content and isinstance(response.content, list) and len(response.content) > 0:
                text_content = response.content[0].text
                return text_content
            else: return ""

        except AnthropicAuthenticationError:
            raise ValueError("Invalid Anthropic API key.")
        except AnthropicAPIError as e:
            # Handle specific Anthropic errors like rate limits, model not found etc.
            error_msg = str(e)
            if "rate limit" in error_msg.lower(): raise ValueError("Anthropic Rate limit exceeded.")
            elif "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                 raise ValueError(f"Anthropic Model '{model_name}' not found or unavailable.")
            else: raise ValueError(f"Anthropic API error: {error_msg}")
        except Exception as e:
            raise Exception(f"Unexpected error calling Anthropic Claude API: {e}")

# --- Factory Function ---
# Cache resource based on provider and API key to reuse client instances within a session
@st.cache_resource(max_entries=5) # Cache a few clients
def get_llm_client(provider: str, api_key: str) -> LLMClient:
    """Factory function to get an instance of the appropriate LLM client."""
    try:
        if provider == "openai":
            return OpenAIClient(api_key=api_key)
        elif provider == "gemini":
            return GeminiClient(api_key=api_key)
        elif provider == "claude":
            return ClaudeClient(api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    except ValueError as e: # Catch init errors and re-raise for UI
         raise ValueError(f"Failed to create client for {provider}: {e}")
    except Exception as e: # Catch unexpected init errors
         raise Exception(f"Unexpected error creating client for {provider}: {e}")
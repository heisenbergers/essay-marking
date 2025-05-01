# utils/helpers.py

import streamlit as st
import os
import json
import re
import pandas as pd # Import pandas here if needed, e.g., for parse_response

@st.cache_data # Cache the loaded prompts
def load_prompts(folder="context"):
    """Loads all .txt files from the specified folder as prompts."""
    # ... (load_prompts function remains the same) ...
    prompts = {}
    if os.path.exists(folder) and os.path.isdir(folder):
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                prompt_name = os.path.splitext(filename)[0]
                prompt_name = re.sub(r'[_-]', ' ', prompt_name).title() # Clean up name
                filepath = os.path.join(folder, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        prompts[prompt_name] = f.read()
                except Exception as e:
                    st.warning(f"Could not load prompt '{filename}': {e}")
    else:
        pass
    if not prompts:
         prompts["Default Fallback"] = """Score the following text on a scale of 1-10. Respond in JSON format: {"score": N, "reason": "Your reasoning"}."""
    return prompts


def parse_response_for_default(response_text: str):
    """
    Parse the API response expecting the default JSON format {"score": N, "reason": "..."}.
    Includes robust fallback mechanisms.
    """
    score = ""
    reason = ""

    if not isinstance(response_text, str) or not response_text.strip():
        return score, reason # Return empty strings for empty/invalid input

    try:
        # 1. Try parsing directly as JSON
        # Clean potential markdown code block fences
        cleaned_text = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE).strip()
        data = json.loads(cleaned_text)

        # --- MODIFICATION START ---
        # Check if the parsed data is a dictionary before using .get()
        if isinstance(data, dict):
            score_val = data.get("score", "")
            reason_val = data.get("reason", "")
            # Ensure score is somewhat reasonable (e.g., primarily numeric like) before assigning
            if isinstance(score_val, (int, float)) or (isinstance(score_val, str) and re.match(r'^\s*-?\d+(\.\d+)?\s*$', score_val.strip())):
                score = str(score_val).strip() # Standardize output as strings
                reason = str(reason_val).strip()
            else:
                 # Score exists in dict but isn't numeric-like, reset score, keep text as reason
                 score = ""
                 reason = cleaned_text # Use cleaned text as reason if score format is unexpected
        elif isinstance(data, (int, float)) or (isinstance(data, str) and re.match(r'^\s*-?\d+(\.\d+)?\s*$', str(data).strip())):
             # Handle case where API returned just a number
             score = str(data).strip()
             reason = "" # No reason provided if only a number came back
        else:
             # Parsed JSON is not a dict or a number, treat as failure for this block
             score = ""
             reason = cleaned_text # Use the cleaned text as fallback reason
        # --- MODIFICATION END ---

    except json.JSONDecodeError:
        # 2. JSON failed, try regex extraction (more robust)
        # Use the cleaned_text from the try block if available, else use original response_text
        text_to_search = cleaned_text if 'cleaned_text' in locals() else response_text

        score_match = re.search(r'"score"\s*:\s*("?(-?\d+(?:\.\d+)?)"?|\d+)', text_to_search, re.IGNORECASE)
        # Look for reason, capture content within quotes if possible, otherwise be more lenient
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text_to_search, re.IGNORECASE)
        if not reason_match: # Try finding reason without strict quotes
            reason_match = re.search(r'"reason"\s*:\s*([\s\S]+)', text_to_search, re.IGNORECASE)
            # Clean up potential trailing characters if match is broad
            if reason_match:
                 potential_reason = reason_match.group(1).strip().rstrip('"}').strip()
                 reason = potential_reason


        score = score_match.group(2) if score_match else "" # Group 2 captures the number
        if not reason and reason_match: # If reason wasn't set via strict quotes but lenient match found
            potential_reason = reason_match.group(1).strip().rstrip('"}').strip()
            reason = potential_reason
        elif not reason: # If no reason found yet
             reason = text_to_search # Fallback: use original text if reason can't be isolated

    # 3. Regex failed or only partial success, final fallback for score
    if not score: # If score is still missing after JSON and regex
        # Try finding *any* number, preferring integers 1-10 if possible
        text_to_search = cleaned_text if 'cleaned_text' in locals() else response_text
        numbers = re.findall(r'\b(-?\d+(?:\.\d+)?)\b', text_to_search)
        if numbers:
            # Look for numbers between 1 and 10 specifically
            try:
                potential_scores = [n for n in numbers if 1 <= float(n) <= 10]
                if potential_scores:
                    score = potential_scores[0] # Take the first plausible score
                else:
                    score = numbers[0] # Take the first number found otherwise
            except (ValueError, TypeError):
                 score = numbers[0] # Fallback if float conversion fails
        else:
             score = "" # No number found

    # Final fallback for reason if still empty
    if not reason:
        reason = cleaned_text if 'cleaned_text' in locals() and cleaned_text else response_text # Use cleaned or original

    # Return standardized strings
    return str(score).strip(), str(reason).strip()

# --- Add other general utility functions below as needed ---
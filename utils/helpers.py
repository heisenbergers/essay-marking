import streamlit as st
import os
import json
import re
import pandas as pd # Import pandas here if needed, e.g., for parse_response

@st.cache_data # Cache the loaded prompts
def load_prompts(folder="context"):
    """Loads all .txt files from the specified folder as prompts."""
    prompts = {}
    if os.path.exists(folder) and os.path.isdir(folder):
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                # Use filename without extension as key, replace underscores/hyphens
                prompt_name = os.path.splitext(filename)[0]
                prompt_name = re.sub(r'[_-]', ' ', prompt_name).title() # Clean up name
                filepath = os.path.join(folder, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        prompts[prompt_name] = f.read()
                except Exception as e:
                    st.warning(f"Could not load prompt '{filename}': {e}")
    else:
        # Handle case where folder doesn't exist locally (might during dev)
        # st.warning(f"Prompt folder '{folder}' not found.") # Optional warning
        pass
    if not prompts:
         # Provide a fallback minimal prompt if none are loaded
         prompts["Default Fallback"] = """Score the following text on a scale of 1-10. Respond in JSON format: {"score": N, "reason": "Your reasoning"}."""

    return prompts


def parse_response_for_default(response_text: str):
    """
    Parse the API response expecting the default JSON format {"score": N, "reason": "..."}.
    Includes robust fallback mechanisms.
    """
    if not isinstance(response_text, str) or not response_text.strip():
        return "", "" # Return empty strings for empty/invalid input

    try:
        # 1. Try parsing directly as JSON
        # Clean potential markdown code block fences
        cleaned_text = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE).strip()
        data = json.loads(cleaned_text)
        score = data.get("score", "")
        reason = data.get("reason", "")
        # Ensure score is somewhat reasonable (e.g., primarily numeric) before returning
        if isinstance(score, (int, float)) or (isinstance(score, str) and re.match(r'^\s*\d+(\.\d+)?\s*$', score)):
            return str(score), str(reason) # Standardize output as strings
        else:
             # Score exists but isn't numeric, treat as failure for this block
             score = "" # Reset score
             reason = cleaned_text # Keep the original text as reason if score fails

    except json.JSONDecodeError:
        # 2. JSON failed, try regex extraction (more robust)
        score_match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?|\d+)', cleaned_text, re.IGNORECASE)
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', cleaned_text, re.IGNORECASE) # Extracts content within quotes

        score = score_match.group(1) if score_match else ""
        reason = reason_match.group(1) if reason_match else ""

        # If reason regex failed, try a broader search
        if not reason and score: # Only if score was found
             reason_match_broader = re.search(r'reason["\':\s]+([\s\S]+)', cleaned_text, re.IGNORECASE)
             if reason_match_broader:
                  # Get everything after "reason" might be too broad, try capturing until next likely field or end
                  potential_reason = reason_match_broader.group(1).strip().rstrip('"}').strip()
                  reason = potential_reason
             else:
                 reason = cleaned_text # Fallback: use original text if reason can't be isolated


    # 3. Regex failed or only partial success, final fallback
    if not score: # If score is still missing after JSON and regex
        # Try finding *any* number, preferring integers 1-10 if possible
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', cleaned_text)
        if numbers:
            # Look for numbers between 1 and 10 specifically
            potential_scores = [n for n in numbers if 1 <= float(n) <= 10]
            if potential_scores:
                score = potential_scores[0] # Take the first plausible score
            else:
                score = numbers[0] # Take the first number found otherwise
        else:
             score = "" # No number found

    # If reason is still empty, use the whole text
    if not reason:
        reason = cleaned_text if cleaned_text else response_text # Use cleaned or original

    # Return standardized strings
    return str(score).strip(), str(reason).strip()

# --- Add other general utility functions below as needed ---
"""
This module provides utility functions for cleaning, formatting, and parsing text responses,
particularly for handling conversation transcripts and responses from language models.

Functions:
    clean_hc_response(response_text: str) -> str
        Removes formatting tokens and normalizes whitespace and punctuation in a response text.

    format_chatml(messages: list) -> str
        Formats a list of message objects into a simple conversation transcript.

    format_structured_prompt(messages: list) -> str
        Formats the conversation history into a prompt instructing the model to return its answer as JSON.

    parse_structured_response(response_text: str) -> str
        Attempts to parse a JSON-formatted response to extract the 'response' field, falling back to cleaning the text if parsing fails.

    clean_output(text: str) -> str
        Removes formatting tokens, normalizes spacing and punctuation, and ensures proper punctuation at the end of the text.

    extract_json(text: str) -> str
        Extracts the first JSON-like substring found in the provided text. Returns an empty JSON object string if none is found.
    
    get_list_docs() -> list
        Returns a list of file names found in the specified ingestion directory.
"""

import re
import json
import os

def clean_hc_response(response_text: str) -> str:
    """
    Clean the response text by removing formatting tokens and normalizing whitespace and punctuation.

    This function:
      - Removes tokens of the form "<|im_start|>" and "<|im_end|>".
      - Collapses multiple consecutive spaces into a single space.
      - Corrects spacing before punctuation marks.

    Parameters:
        response_text (str): The raw response text containing formatting tokens.

    Returns:
        str: The cleaned response text.
    """
    # Remove the tokens and replace them with a single space.
    response_text = re.sub(r"<\|im_start\|>\s*", "", response_text)
    response_text = re.sub(r"\s*<\|im_end\|>", "", response_text)
    # Normalize whitespace by collapsing multiple spaces into one.
    response_text = " ".join(response_text.split())
    # Remove any extra spaces before punctuation.
    response_text = re.sub(r'\s+([,?.!])', r'\1', response_text)
    return response_text.strip()

def format_chatml(messages: list) -> str:
    """
    Format a list of message objects into a conversation transcript.

    Iterates over the provided messages and prefixes each message content with
    a role identifier ("System", "User", or "Assistant") based on its class name.

    Parameters:
        messages (list): A list of message objects. Expected classes include
                         SystemMessage, HumanMessage, and AIMessage.

    Returns:
        str: A formatted conversation transcript.
    """
    formatted = []
    for msg in messages:
        # Determine the role based on the message class.
        if msg.__class__.__name__ == "SystemMessage":
            formatted.append("System: " + msg.content)
        elif msg.__class__.__name__ == "HumanMessage":
            formatted.append("User: " + msg.content)
        elif msg.__class__.__name__ == "AIMessage":
            formatted.append("Assistant: " + msg.content)
        else:
            # Fallback to a generic label if the class is unrecognized.
            formatted.append("User: " + str(msg))
    return "\n".join(formatted)

def format_structured_prompt(messages: list) -> str:
    """
    Format the conversation history into a prompt instructing the model to return its answer as JSON.

    The function builds a base prompt that identifies the AI as "Dolphin, a helpful AI concierge assistant",
    appends the conversation history (each message prefixed by its role in lowercase), and ends with a prompt for the assistant.

    Parameters:
        messages (list): A list of message objects, each with a 'content' attribute.
    
    Returns:
        str: A complete prompt string instructing the model to respond with a JSON object.
    """
    base_prompt = (
        "You are Dolphin, a helpful AI concierge assistant. "
        "Answer the user's query and return your answer as a JSON object with a single key 'response'.\n\n"
    )
    conversation = ""
    for msg in messages:
        # Extract the role by removing "Message" from the class name and converting to lowercase.
        role = msg.__class__.__name__.replace("Message", "").lower()
        conversation += f"{role}: {msg.content}\n"
    # Append a prompt for the assistant to provide its response.
    conversation += "assistant: "
    return base_prompt + conversation

def parse_structured_response(response_text: str) -> str:
    """
    Parse a JSON-formatted response text to extract the 'response' field.

    Attempts to decode the provided text as JSON. If successful, returns the value associated
    with the 'response' key. If parsing fails, falls back to cleaning the response text.

    Parameters:
        response_text (str): The raw response text which may include JSON formatting.

    Returns:
        str: The extracted 'response' field, or a cleaned version of the response text if parsing fails.
    """
    try:
        data = json.loads(response_text)
        return data.get("response", "").strip()
    except Exception:
        return clean_hc_response(response_text)

def clean_output(text: str) -> str:
    """
    Clean the output text by removing formatting tokens, normalizing spacing, and ensuring proper punctuation.

    This function:
      - Removes tokens "<|im_start|>" and "<|im_end|>".
      - Removes stray double quotes.
      - Collapses multiple spaces into a single space.
      - Ensures the text ends with proper punctuation (adds a period if necessary).

    Parameters:
        text (str): The raw text output.

    Returns:
        str: The cleaned and formatted text.
    """
    # Remove formatting tokens.
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    # Remove any double quotes.
    # text = text.replace('"', '').replace('\\"', '')
    # Normalize whitespace.
    text = " ".join(text.split())
    # Append a period if the text doesn't end with proper punctuation.
    if text and text[-1] not in ".!?":
        text += "."
    return text.strip()

def extract_json(text: str) -> str:
    """
    Extract the first JSON-like substring from the provided text.

    Uses a regular expression to search for a substring that resembles a JSON object.
    If found, returns that JSON string; otherwise, returns an empty JSON object ("{}").

    Parameters:
        text (str): The input text that may contain a JSON substring.

    Returns:
        str: The extracted JSON substring, or "{}" if no JSON-like substring is found.
    """
    match = re.search(r'({.*})', text, re.DOTALL)
    if match:
        return match.group(1)
    return "{}"

def get_list_docs(folder_path: str = "my_furhat_backend/ingestion") -> list:
    """
    Get a list of document file names from the ingestion directory.

    Scans the "my_furhat_backend/ingestion" directory and returns the names of all files found.

    Returns:
        list: A list of file names.
    """
    # List file names without extension using os.scandir and os.path.splitext
    return [os.path.splitext(entry.name)[0] 
                    for entry in os.scandir(folder_path) 
                    if entry.is_file()]

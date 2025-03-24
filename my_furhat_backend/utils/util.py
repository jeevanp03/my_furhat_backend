"""
This module provides utility functions for cleaning, formatting, parsing, and summarizing text responses,
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
    
    get_list_docs(folder_path: str = "my_furhat_backend/ingestion") -> list
        Returns a list of file names (without extensions) found in the specified ingestion directory.

    summarize_text(text: str) -> str
        Summarizes the input text using a pre-trained summarization model.

    get_document_context(document: str) -> str
        Retrieves the context of a document from the DocumentAgent based on the document's title.

    generate_followup_prompt(summary: str) -> str
        Generates a follow-up prompt for the assistant based on the provided summary.
    
    classify_text(content: str, docs: list) -> dict
        Ranks a list of documents based on their similarity to the provided content using a text classifier.
"""

import re
import json
import os

# Import necessary classes and instances from the backend.
from my_furhat_backend.models.llm_factory import HuggingFaceLLM
from my_furhat_backend.models.classifier import TextClassifier

# Instantiate a summarizer using a pre-trained model for text summarization.
summarizer = HuggingFaceLLM(model_id="sshleifer/distilbart-cnn-12-6", task="summarization")
text_classifier = TextClassifier()


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
    # Remove the "<|im_start|>" token along with any trailing whitespace.
    response_text = re.sub(r"<\|im_start\|>\s*", "", response_text)
    # Remove the "<|im_end|>" token along with any leading whitespace.
    response_text = re.sub(r"\s*<\|im_end\|>", "", response_text)
    # Collapse multiple spaces into one.
    response_text = " ".join(response_text.split())
    # Remove any extra spaces before punctuation marks (e.g., commas, periods, question marks, exclamation points).
    response_text = re.sub(r'\s+([,?.!])', r'\1', response_text)
    # Return the cleaned and trimmed response text.
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
    formatted = []  # Initialize a list to hold each formatted message.
    for msg in messages:
        # Determine the role of the message based on its class name.
        if msg.__class__.__name__ == "SystemMessage":
            formatted.append("System: " + msg.content)
        elif msg.__class__.__name__ == "HumanMessage":
            formatted.append("User: " + msg.content)
        elif msg.__class__.__name__ == "AIMessage":
            formatted.append("Assistant: " + msg.content)
        else:
            # Fallback: if the message type is unrecognized, default to "User" label.
            formatted.append("User: " + str(msg))
    # Join all messages with newline characters to create a complete transcript.
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
    # Define the base prompt that sets the assistant's identity and instructions.
    base_prompt = (
        "You are Dolphin, a helpful AI concierge assistant. "
        "Answer the user's query and return your answer as a JSON object with a single key 'response'.\n\n"
    )
    conversation = ""
    for msg in messages:
        # Extract the role by removing "Message" from the class name and converting it to lowercase.
        role = msg.__class__.__name__.replace("Message", "").lower()
        # Append the role and content to the conversation transcript.
        conversation += f"{role}: {msg.content}\n"
    # Append the assistant's prompt to indicate where the response should begin.
    conversation += "assistant: "
    # Combine the base prompt and the conversation transcript.
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
        # Attempt to parse the response text as JSON.
        data = json.loads(response_text)
        # Return the value of the 'response' field if available.
        return data.get("response", "").strip()
    except Exception:
        # Fallback: if JSON parsing fails, clean the raw response text.
        return clean_hc_response(response_text)


def clean_output(text: str) -> str:
    """
    Clean the output text by removing formatting tokens, normalizing spacing, and ensuring proper punctuation.

    This function:
      - Removes tokens "<|im_start|>" and "<|im_end|>".
      - Normalizes whitespace.
      - Appends proper punctuation at the end if missing.

    Parameters:
        text (str): The raw text output.

    Returns:
        str: The cleaned and formatted text.
    """
    # Remove formatting tokens.
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    # (Optional) Remove stray double quotes; this is currently commented out.
    # text = text.replace('"', '').replace('\\"', '')
    # Normalize whitespace by collapsing multiple spaces into one.
    text = " ".join(text.split())
    # If the text does not end with a punctuation mark, append a period.
    if text and text[-1] not in ".!?":
        text += "."
    # Return the cleaned and trimmed text.
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
    # Use a regex pattern to locate a JSON-like structure within the text.
    match = re.search(r'({.*})', text, re.DOTALL)
    if match:
        # Return the matched JSON substring.
        return match.group(1)
    # Return an empty JSON object if no match is found.
    return "{}"


def get_list_docs(folder_path: str = "my_furhat_backend/ingestion") -> list:
    """
    Get a list of document file names from the ingestion directory.

    Scans the specified directory (default: "my_furhat_backend/ingestion") and returns the names of all files found.
    The file extensions are removed from the file names.

    Parameters:
        folder_path (str): The directory path to scan for document files. Defaults to "my_furhat_backend/ingestion".

    Returns:
        list: A list of file names (without extensions).
    """
    # Iterate over entries in the directory and include only files, stripping their extensions.
    return [
        os.path.splitext(entry.name)[0]
        for entry in os.scandir(folder_path)
        if entry.is_file()
    ]


def summarize_text(text: str) -> str:
    """
    Summarize the input text using a pre-trained summarization model.

    This function creates a prompt by appending a summarization instruction to the input text
    and then queries the summarizer model to generate a concise summary.

    Parameters:
        text (str): The input text to be summarized.

    Returns:
        str: The summarized version of the input text.
    """
    # Build a prompt by combining an instruction with the provided text.
    prompt = (
        "Summarize the retrieved document context to provide a concise response: " + text
    )
    # Query the summarizer model and return its output.
    return summarizer.query(prompt)


def get_document_context(document: str) -> str:
    from my_furhat_backend.agents.document_agent import rag_instance
    """
    Retrieve the context of a document from the DocumentAgent.

    Generates a prompt instructing the retrieval system to extract the main context and themes from the document.
    The function then queries the retrieval system (rag_instance) to obtain the document context.

    Parameters:
        document (str): The name or identifier of the document to retrieve context for.

    Returns:
        str: The context of the specified document.
    """
    # Construct a prompt that asks for the document's overarching context and main themes.
    prompt = (
        "Extract the overarching context and main themes from the document titled "
        f"'{document}'. Focus on summarizing the key topics, narrative, and any major findings "
        "without including extraneous details."
    )
    # Retrieve and return the document context using the retrieval instance.
    return rag_instance.retrieve_similar(prompt)


def generate_followup_prompt(summary: str) -> str:
    from my_furhat_backend.agents.document_agent import chatbot
    """
    Generate a follow-up prompt for the assistant based on the provided summary.

    Constructs a prompt that instructs the assistant to generate a follow-up question or response,
    ensuring the conversation remains engaging and contextually relevant.

    Parameters:
        summary (str): The summarized text to generate a follow-up prompt.

    Returns:
        str: A follow-up prompt for the assistant based on the summary.
    """
    # Build a prompt using the summary to encourage further engagement from the user.
    prompt = (
        "Based on the summary provided, generate a follow-up question or response that "
        "engages the user and encourages further interaction. Be sure to maintain context "
        "and relevance to the user's query. The summary is as follows: " + summary
    )
    # Query the chatbot instance with the generated prompt and return the result.
    return chatbot.query(prompt)


def classify_text(content: str, docs: list) -> dict:
    """
    Rank a list of documents based on their similarity to the provided content.

    This function ranks the documents by their similarity to the input content using a text classifier.
    The classifier assigns a score to each document, with higher scores indicating greater relevance.

    Parameters:
        content (str): The content to be used for ranking the documents.
        docs (list): A list of document titles or identifiers to be ranked.

    Returns:
        dict: A dictionary mapping document titles to their corresponding relevance scores.
    """
    return text_classifier.classify(content, docs)
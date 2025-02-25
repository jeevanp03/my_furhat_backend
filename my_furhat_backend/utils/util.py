import re
import json

def clean_hc_response(response_text: str) -> str:
    # Remove the tokens and replace them with a single space.
    response_text = re.sub(r"<\|im_start\|>\s*", "", response_text)
    response_text = re.sub(r"\s*<\|im_end\|>", "", response_text)
    # Normalize whitespace: collapse multiple spaces into one.
    response_text = " ".join(response_text.split())
    # Fix spacing before punctuation.
    response_text = re.sub(r'\s+([,?.!])', r'\1', response_text)
    return response_text.strip()

def format_chatml(messages: list) -> str:
    # Produce a simple conversation transcript.
    formatted = []
    for msg in messages:
        if msg.__class__.__name__ == "SystemMessage":
            formatted.append("System: " + msg.content)
        elif msg.__class__.__name__ == "HumanMessage":
            formatted.append("User: " + msg.content)
        elif msg.__class__.__name__ == "AIMessage":
            formatted.append("Assistant: " + msg.content)
        else:
            formatted.append("User: " + str(msg))
    return "\n".join(formatted)

def format_structured_prompt(messages: list) -> str:
    """
    Format the conversation history into a prompt instructing the model to return its answer as JSON.
    """
    base_prompt = (
        "You are Dolphin, a helpful AI concierge assistant. "
        "Answer the user's query and return your answer as a JSON object with a single key 'response'.\n\n"
    )
    conversation = ""
    for msg in messages:
        role = msg.__class__.__name__.replace("Message", "").lower()
        conversation += f"{role}: {msg.content}\n"
    conversation += "assistant: "
    return base_prompt + conversation

def parse_structured_response(response_text: str) -> str:
    """
    Try to parse the response as JSON and extract the 'response' field.
    If parsing fails, fallback to cleaning the response.
    """
    try:
        data = json.loads(response_text)
        return data.get("response", "").strip()
    except Exception:
        return clean_hc_response(response_text)

def clean_output(text: str) -> str:
    """
    Remove formatting tokens and normalize spacing and punctuation.
    """
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    text = " ".join(text.split())
    # Optionally, ensure the output ends with proper punctuation.
    if text and text[-1] not in ".!?":
        text += "."
    return text.strip()

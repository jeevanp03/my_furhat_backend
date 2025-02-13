from langgraph import Node
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage
from my_furhat_backend.models.llm import HuggingFaceLLM
import re

def clean_llm_response(response_text: str) -> str:
    """
    Removes unwanted tokens and extra whitespace from the LLM response.
    """
    # Remove any tokens like <|im_start|> and <|im_end|> and their contents.
    response_text = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", response_text, flags=re.DOTALL)
    # Strip leading/trailing whitespace.
    return response_text.strip()

class ChatbotNode(Node):
    def __init__(self, 
                 model_id: str = "bartowski/Mistral-Small-24B-Instruct-2501-GGUF",
                 file_name: str = "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf",
                 **kwargs):
        """
        Initialize the ChatbotNode by instantiating the HuggingFaceLLM.
        """
        self.llm = HuggingFaceLLM(model_id=model_id, file_name=file_name, **kwargs)
    
    def run(self, state: dict) -> dict:
        """
        Processes the conversation state:
          - Combines all messages to form a prompt.
          - Invokes the LLM to generate a response.
          - Cleans the response.
          - Appends the AI's message to the state.
        
        Args:
            state (dict): Expected to have a "messages" key (a list of message objects).
        
        Returns:
            dict: The updated state with the new AIMessage appended.
        """
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("No messages found in state.")

        # Combine conversation history into one prompt for context.
        prompt = "\n".join(message.content for message in messages)
        
        # Generate a response using the LLM.
        response = self.llm.query(prompt)
        
        # Normalize the response based on its type.
        if isinstance(response, AIMessage):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        elif isinstance(response, dict):
            response_text = response.get("content", "")
        else:
            response_text = str(response)
        
        # Clean the response text.
        response_text = clean_llm_response(response_text)
        print(f"Cleaned LLM Response: {response_text}")  # Debug output

        # Append the new AIMessage to the conversation state.
        messages.append(AIMessage(content=response_text))
        state["messages"] = messages
        
        return state

# Example usage (if run as a script):
if __name__ == "__main__":
    # For testing, create a dummy state with a couple of messages.
    dummy_state = {
        "messages": [
            SystemMessage(content="You are a helpful AI concierge assistant."),
            HumanMessage(content="Find me a good restaurant nearby.")
        ]
    }
    chatbot_node = ChatbotNode()
    new_state = chatbot_node.run(dummy_state)
    print("Final State Messages:")
    for msg in new_state["messages"]:
        print(msg.content)

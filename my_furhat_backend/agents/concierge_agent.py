from langgraph.graph import StateGraph, START, END
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from typing_extensions import TypedDict, Annotated, List
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from my_furhat_backend.models.chatbot_factory import Chatbot_HuggingFace, Chatbot_LlamaCpp
from my_furhat_backend.models.llm_factory import llm_hc_instance, llm_llama_instance
from my_furhat_backend.utils.util import get_location_data, clean_output

from my_furhat_backend.llm_tools.tools import tools as all_tools

# Define our custom state.
class State(TypedDict):
    messages: Annotated[List[BaseMessage], "add_messages"]

def aggregate_poi_context(state: dict) -> str:
    parts = []
    fs_results = state.get("foursquare_results", [])
    if fs_results:
        fs_names = ", ".join(item.get("name", "Unnamed") for item in fs_results[:3])
        parts.append(f"Foursquare: {fs_names}")
    # You can add OSM/Overpass if needed.
    return "\n".join(parts)

def chatbot_node_hc(state: State) -> State:
    from my_furhat_backend.models.chatbot_factory import Chatbot_HuggingFace
    chatbot = Chatbot_HuggingFace(model_instance=llm_hc_instance)
    return chatbot.chatbot(state)

# Build the LangGraph state graph.
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node_hc)

# Create a ToolNode for the remaining tools.
tool_node = ToolNode(tools=all_tools)
graph_builder.add_node("tools", tool_node)

# Add conditional edges so that if the chatbot issues a tool call, control flows to the tool node.
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

graph_builder.set_entry_point("chatbot")
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def run_conversation_streaming():
    config = {"configurable": {"thread_id": "1"}}
    print("Welcome to Mistrial, your AI concierge assistant.")
    query = input("Enter your query (or 'exit' to quit): ")
    if query.strip().lower() == "exit":
        return
    
    # Get location data.
    location = get_location_data()
    latitude = location.get("latitude", 0.0)
    longitude = location.get("longitude", 0.0)
    print(f"Location data: latitude {latitude}, longitude {longitude}")
    
    # Updated system prompt instructing the use of the tool.
    system_prompt = SystemMessage(content=(
        f"You are Mistrial, a sophisticated, friendly, and highly knowledgeable concierge AI assistant. "
        "Your mission is to deliver tailored, local recommendations and valuable information to guests and visitors. "
        "You have been provided with the current location as follows: "
        f"latitude {latitude} and longitude {longitude}. Always use this location data to inform your responses. "
        "If you need to fetch additional local information or perform a lookup, output a tool call using the exact format: "
        'TOOL: {"name": "restaurants_search", "args": {"query": "<your query>", "latitude": <latitude>, "longitude": <longitude>}}. '
        "Ensure your responses are clear, professional, and engaging."
    ))
    
    state: State = {
        "messages": [
            system_prompt,
            HumanMessage(content=query)
        ]
    }
    
    print("Initial state set. Streaming response from Mistrial...")
    events = graph.stream(state, config, stream_mode="values")
    final_response = ""
    for event in events:
        final_response = clean_output(event["messages"][-1].content)
    print("Mistrial:", final_response)
    
    # Add additional context from POI results if available.
    poi_context = aggregate_poi_context(state)
    if poi_context:
        state["messages"].append(HumanMessage(content=f"Context: {poi_context}"))
        events = graph.stream(state, config, stream_mode="values")
        for event in events:
            final_response = clean_output(event["messages"][-1].content)
        print("Mistrial:", final_response)
    
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        state["messages"].append(HumanMessage(content=user_input))
        events = graph.stream(state, config, stream_mode="values")
        for event in events:
            final_response = clean_output(event["messages"][-1].content)
        print("Mistrial:", final_response)

if __name__ == "__main__":
    run_conversation_streaming()

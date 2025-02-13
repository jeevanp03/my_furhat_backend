from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from typing_extensions import TypedDict

# Import your node classes:
from my_furhat_backend.nodes.ipdata_node import IPDataNode
from my_furhat_backend.nodes.foursquare_node import FoursquareNode
from my_furhat_backend.nodes.osm_node import OSMNode
from my_furhat_backend.nodes.overpass_node import OverpassNode
# from agent.nodes.tavily_search_node import TavilySearchNode  # Your web search tool node
from my_furhat_backend.nodes.chatbot_node import ChatbotNode

# Define the state type. The state is a dictionary that holds a list of messages.
class State(TypedDict):
    messages: "Annotated[List[AIMessage | HumanMessage | SystemMessage], add_messages]"

def aggregate_poi_context(fs_data: dict, osm_data: dict, op_data: dict, tavily_data: dict) -> str:
    """Aggregate outputs from multiple POI nodes into a single context string."""
    parts = []
    
    if fs_data.get("foursquare_results"):
        fs_names = ", ".join(item.get("name", "Unnamed") for item in fs_data["foursquare_results"][:3])
        parts.append(f"Foursquare: {fs_names}")
    
    if osm_data.get("osm_results"):
        osm_names = ", ".join(poi.get("display_name", "Unnamed") for poi in osm_data["osm_results"][:3])
        parts.append(f"OSM: {osm_names}")
    
    if op_data.get("overpass_results"):
        elements = op_data.get("overpass_results", {}).get("elements", [])
        op_names = ", ".join(
            el.get("tags", {}).get("name") or el.get("tags", {}).get("amenity", "Unnamed")
            for el in elements[:3]
        )
        parts.append(f"Overpass: {op_names}")
    
    # if tavily_data.get("tavily_results"):
    #     tavily_names = ", ".join(result.get("title", "Unnamed") for result in tavily_data["tavily_results"][:3])
    #     parts.append(f"Tavily Search: {tavily_names}")
    
    return "\n".join(parts)

class ConciergeAgent:
    def __init__(self):
        # Create a state graph with an initial state structure.
        self.graph = StateGraph(State)
        # Add our nodes to the graph.
        self.graph.add_node("ipdata", IPDataNode())
        self.graph.add_node("foursquare", FoursquareNode())
        self.graph.add_node("osm", OSMNode())
        self.graph.add_node("overpass", OverpassNode())
        # self.graph.add_node("tavily", TavilySearchNode())
        self.graph.add_node("chatbot", ChatbotNode())
        # In this simple design, we'll set the chatbot node as the final step.
        self.graph.set_entry_point("chatbot")
    
    def handle_query(self, query: str, ip: str = None) -> str:
        # Initialize the conversation state.
        state: State = {
            "messages": [
                SystemMessage(content="You are a helpful AI concierge assistant."), 
                HumanMessage(content=query)
            ]
        }
        
        # Step 1: Get user location using IPData node.
        location_data = self.graph.run_node("ipdata", {"ip": ip})
        lat = location_data.get("latitude")
        lon = location_data.get("longitude")
        if not lat or not lon:
            return "Could not determine your location."
        
        # Prepare input data for POI nodes.
        poi_input = {"latitude": lat, "longitude": lon, "query": query}
        fs_data = self.graph.run_node("foursquare", poi_input)
        osm_data = self.graph.run_node("osm", poi_input)
        op_data = self.graph.run_node("overpass", poi_input)
        tavily_data = self.graph.run_node("tavily", {"query": query})
        
        # Aggregate POI context.
        poi_context = aggregate_poi_context(fs_data, osm_data, op_data, tavily_data)
        
        # Add a message with the aggregated context to the state.
        state["messages"].append(HumanMessage(content=f"Context: {poi_context}"))
        
        # Now, run the chatbot node with the complete state.
        final_state = self.graph.run_node("chatbot", state)
        
        # Extract the final response (assumed to be the last message in state).
        if final_state.get("messages"):
            return final_state["messages"][-1].content
        else:
            return "Sorry, no response generated."

if __name__ == "__main__":
    agent = ConciergeAgent()
    user_query = "Find me a good restaurant nearby"
    response = agent.handle_query(user_query, ip="")  # Optionally, pass a test IP.
    print("Agent Response:")
    print(response)

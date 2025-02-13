from langgraph import Node
from my_furhat_backend.api_clients.osm_client import OSMClient

class OSMNode(Node):
    def __init__(self):
        self.osm_client = OSMClient()

    def run(self, input_data: dict) -> dict:
        lat = input_data.get("latitude")
        lon = input_data.get("longitude")
        query = input_data.get("query", "restaurant")
        results = self.osm_client.search_pois(lat, lon, query)
        return {"osm_results": results}
from langgraph import Node
from my_furhat_backend.api_clients.overpass_client import OverpassClient

class OverpassNode(Node):
    def __init__(self):
        self.overpass_client = OverpassClient()

    def run(self, input_data: dict) -> dict:
        lat = input_data.get("latitude")
        lon = input_data.get("longitude")
        query = input_data.get("query", "restaurant")
        results = self.overpass_client.search_pois(lat, lon, query)
        return {"overpass_results": results}
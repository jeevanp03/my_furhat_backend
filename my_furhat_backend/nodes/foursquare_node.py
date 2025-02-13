from langgraph import Node
from my_furhat_backend.api_clients.foursquare_client import FoursquareClient
import os

class FoursquareNode(Node):
    def __init__(self):
        self.fs_client = FoursquareClient()

    def run(self, input_data: dict) -> dict:
        lat = input_data.get("latitude")
        lon = input_data.get("longitude")
        query = input_data.get("query", "restaurant")
        ll = f"{lat},{lon}"
        results = self.fs_client.search_places(ll, query)
        return {"foursquare_results": results.get("results", [])}
from my_furhat_backend.api_clients.overpass_client import OverpassClient

class OverpassNode:
    def __init__(self):
        self.overpass_client = OverpassClient()
    
    def __call__(self, state: dict) -> dict:
        lat = state.get("latitude")
        lon = state.get("longitude")
        query = state.get("query", "restaurant")
        if lat is None or lon is None:
            state["overpass_results"] = {}
        else:
            results = self.overpass_client.search_pois(lat, lon, query)
            state["overpass_results"] = results
        return state

from my_furhat_backend.api_clients.osm_client import OSMClient

class OSMNode:
    def __init__(self):
        self.osm_client = OSMClient()
    
    def __call__(self, state: dict) -> dict:
        lat = state.get("latitude")
        lon = state.get("longitude")
        query = state.get("query", "restaurant")
        if lat is None or lon is None:
            state["osm_results"] = []
        else:
            results = self.osm_client.search_pois(lat, lon, query)
            state["osm_results"] = results
        return state

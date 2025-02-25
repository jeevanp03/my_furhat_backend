from my_furhat_backend.api_clients.foursquare_client import FoursquareClient

class FoursquareNode:
    def __init__(self):
        self.fs_client = FoursquareClient()
    
    def __call__(self, state: dict) -> dict:
        lat = state.get("latitude")
        lon = state.get("longitude")
        query = state.get("query", "restaurant")
        if lat is None or lon is None:
            state["foursquare_results"] = []
        else:
            ll = f"{lat},{lon}"
            results = self.fs_client.search_places(ll, query)
            state["foursquare_results"] = results.get("results", [])
        return state

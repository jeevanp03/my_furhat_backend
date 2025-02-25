from my_furhat_backend.api_clients.ipdata_client import IPDataClient

class IPDataNode:
    def __init__(self):
        self.ip_client = IPDataClient()
    
    def __call__(self, state: dict) -> dict:
        data = self.ip_client.lookup()
        # Assuming data is a dict or an object with latitude/longitude attributes
        state["latitude"] = data.latitude or getattr(data, "latitude", None)
        state["longitude"] = data.longitude or getattr(data, "longitude", None)
        return state

from langgraph import Node
from my_furhat_backend.api_clients.ipdata_client import IPDataClient

class IPDataNode(Node):
    def __init__(self):
        self.ip_client = IPDataClient()

    def run(self) -> dict:
        data = self.ip_client.lookup()
        return {"latitude": data.latitude, "longitude": data.longitude}
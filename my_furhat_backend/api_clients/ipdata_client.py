import ipdata
from my_furhat_backend.config.settings import config

class IPDataClient:
    def __init__(self):
        """
        Initialize the IPDataClient.
        If no API key is provided, the client will look for the 'IPDATA_API_KEY' environment variable.
        """
        self.api_key = config["IP_KEY"]
        if not self.api_key:
            raise ValueError("IPDATA_API_KEY is not set. Provide an API key or set the environment variable.")
        # Set the API key in the ipdata package
        ipdata.api_key = self.api_key

    def lookup(self) -> dict:
        """
        Look up information for the provided IP address.
        If no IP is provided, it will use the default behavior (typically your current public IP).
        
        Args:
            ip (str): The IP address to look up (optional).
        
        Returns:
            dict: A dictionary of the lookup data.
        """
        return ipdata.lookup()

if __name__ == "__main__":
    client = IPDataClient()
    print(client.lookup().country_name)
    print(client.lookup().city)
    print(client.lookup().latitude)
    print(client.lookup().longitude)
    
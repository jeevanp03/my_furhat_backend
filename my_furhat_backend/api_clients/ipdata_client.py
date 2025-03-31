import ipdata
from my_furhat_backend.config.settings import config

class IPDataClient:
    """
    Client for interacting with the ipdata API to perform IP address lookups.

    This client initializes the ipdata package with an API key and provides a method to
    look up geographical and network information about an IP address.
    """

    def __init__(self):
        """
        Initialize the IPDataClient.

        Retrieves the API key from the configuration. If no API key is found, it raises a ValueError.
        The API key is then set for the ipdata package to enable authenticated requests.
        """
        # Retrieve the API key from the configuration settings.
        self.api_key = config["IP_KEY"]
        if not self.api_key:
            raise ValueError("IPDATA_API_KEY is not set. Provide an API key or set the environment variable.")
        # Set the API key in the ipdata package to authenticate API requests.
        ipdata.api_key = self.api_key

    def lookup(self) -> dict:
        """
        Look up information for an IP address.

        Uses ipdata's default lookup behavior, which typically retrieves information for the caller's public IP.
        This method does not require an explicit IP address parameter; if needed, the ipdata package's
        configuration can be modified externally to lookup a specific IP.

        Returns:
            dict: A dictionary containing the lookup data, such as location, network, and organization details.
        """
        # Perform the IP lookup using the ipdata package and return the resulting data.
        return ipdata.lookup()

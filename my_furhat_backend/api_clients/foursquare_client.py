import requests
from my_furhat_backend.config.settings import config

class FoursquareClient:
    """
    Client for interacting with the Foursquare Places API.

    This client is responsible for constructing requests to the Foursquare API to search for
    places (venues) near a given location based on search queries.
    """

    def __init__(self):
        """
        Initialize the FoursquareClient.

        Retrieves the API key from configuration settings and sets the base URL and headers for requests.
        Raises a ValueError if the API key is not provided.
        """
        # Retrieve the Foursquare API key from configuration.
        self.api_key = config["FSQ_KEY"]
        if not self.api_key:
            raise ValueError("FOURSQUARE_API_KEY is not set. Provide an API key or set the environment variable.")
        
        # Set the base URL for the Foursquare Places API.
        self.base_url = "https://api.foursquare.com/v3/places"
        
        # Set the headers required for Foursquare API requests.
        self.headers = {
            "Accept": "application/json",
            "Authorization": self.api_key
        }
    
    def search_places(self, ll: str, query: str, limit: int = 10) -> dict:
        """
        Search for places near a given location using Foursquare's Places API.

        Constructs and sends a GET request to the Foursquare search endpoint using the provided location,
        query, and result limit. Returns the JSON response containing the search results.

        Parameters:
            ll (str): A comma-separated latitude,longitude string (e.g., "40.7128,-74.0060").
            query (str): Search term such as "restaurant", "museum", etc.
            limit (int): Maximum number of results to return (default is 10).

        Returns:
            dict: Parsed JSON response from the Foursquare API containing the list of places.

        Raises:
            requests.HTTPError: If the HTTP request fails.
        """
        # Construct the endpoint URL for the search.
        endpoint = f"{self.base_url}/search"
        
        # Define the query parameters for the request.
        params = {
            "ll": ll,
            "query": query,
            "limit": limit
        }
        
        # Send the GET request to the Foursquare API.
        response = requests.get(endpoint, headers=self.headers, params=params)
        
        # Raise an exception if the request was unsuccessful.
        response.raise_for_status()
        
        # Return the parsed JSON response.
        return response.json()

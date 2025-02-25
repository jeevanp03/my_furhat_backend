import requests
from my_furhat_backend.config.settings import config

class FoursquareClient:
    def __init__(self):
        self.api_key = config["FSQ_KEY"]
        if not self.api_key:
            raise ValueError("FOURSQUARE_API_KEY is not set. Provide an API key or set the environment variable.")
        self.base_url = "https://api.foursquare.com/v3/places"
        self.headers = {
            "Accept": "application/json",
            "Authorization": self.api_key
        }
    
    def search_places(self, ll: str, query: str, limit: int = 10) -> dict:
        """
        Search for places near a given location.
        
        Args:
            ll (str): A comma-separated latitude,longitude string (e.g., "40.7128,-74.0060").
            query (str): Search term such as "restaurant".
            limit (int): Maximum number of results.
            
        Returns:
            dict: JSON response from Foursquare.
        """
        endpoint = f"{self.base_url}/search"
        params = {
            "ll": ll,
            "query": query,
            "limit": limit
        }
        response = requests.get(endpoint, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
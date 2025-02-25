import requests

class OSMClient:
    def __init__(self, base_url: str = "https://nominatim.openstreetmap.org"):
        self.base_url = base_url

    def search_pois(self, lat: float, lon: float, query: str, limit: int = 10) -> list:
        """
        Search for points of interest using Nominatim.
        
        Args:
            lat (float): Latitude of the location.
            lon (float): Longitude of the location.
            query (str): The search term (e.g., "restaurant", "museum").
            limit (int): Maximum number of results.
        
        Returns:
            list: List of POI dictionaries.
        """
        url = f"{self.base_url}/search"
        params = {
            "q": query,
            "format": "json",
            "addressdetails": 1,
            "limit": limit,
            "lat": lat,
            "lon": lon
        }
        headers = {
            "User-Agent": "my-concierge-agent/1.0 (contact@example.com)"
        }
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()

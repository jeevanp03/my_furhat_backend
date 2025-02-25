import requests

class OverpassClient:
    def __init__(self, base_url: str = "https://overpass-api.de/api/interpreter"):
        self.base_url = base_url

    def search_pois(self, lat: float, lon: float, query: str, radius: int = 1000, limit: int = 10) -> dict:
        """
        Search for points of interest using the Overpass API.
        
        Args:
            lat (float): Latitude.
            lon (float): Longitude.
            query (str): The type of POI, e.g., "restaurant".
            radius (int): Search radius in meters.
            limit (int): Limit on the number of results (implemented in query logic).
            
        Returns:
            dict: Parsed JSON response.
        """
        # Construct an Overpass QL query.
        overpass_query = f"""
        [out:json][timeout:25];
        (
          node["amenity"="{query}"](around:{radius},{lat},{lon});
          way["amenity"="{query}"](around:{radius},{lat},{lon});
          relation["amenity"="{query}"](around:{radius},{lat},{lon});
        );
        out center {limit};
        """
        response = requests.post(self.base_url, data={'data': overpass_query})
        response.raise_for_status()
        return response.json()


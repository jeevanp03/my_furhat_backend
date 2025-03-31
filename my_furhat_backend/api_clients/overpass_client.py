import requests

class OverpassClient:
    """
    Client for interacting with the Overpass API to search for points of interest (POIs).

    This client constructs and sends Overpass QL queries to the Overpass API endpoint.
    """

    def __init__(self, base_url: str = "https://overpass-api.de/api/interpreter"):
        """
        Initialize the OverpassClient with the specified base URL.

        Parameters:
            base_url (str): The base URL of the Overpass API endpoint.
                            Default is "https://overpass-api.de/api/interpreter".
        """
        self.base_url = base_url

    def search_pois(self, lat: float, lon: float, query: str, radius: int = 1000, limit: int = 10) -> dict:
        """
        Search for points of interest (POIs) using the Overpass API.

        Constructs an Overpass QL query to search for nodes, ways, and relations that match
        the specified query (e.g., "restaurant") within a given radius around the specified coordinates.

        Parameters:
            lat (float): Latitude coordinate of the search center.
            lon (float): Longitude coordinate of the search center.
            query (str): The type of POI to search for (e.g., "restaurant").
            radius (int): Search radius in meters. Default is 1000 meters.
            limit (int): Limit on the number of results (implemented in query logic). Default is 10.

        Returns:
            dict: Parsed JSON response from the Overpass API containing the search results.

        Raises:
            requests.HTTPError: If the HTTP request to the Overpass API fails.
        """
        # Construct an Overpass QL query to search for nodes, ways, and relations with the specified amenity.
        overpass_query = f"""
        [out:json][timeout:25];
        (
          node["amenity"="{query}"](around:{radius},{lat},{lon});
          way["amenity"="{query}"](around:{radius},{lat},{lon});
          relation["amenity"="{query}"](around:{radius},{lat},{lon});
        );
        out center {limit};
        """
        # Send the POST request with the query as data to the Overpass API endpoint.
        response = requests.post(self.base_url, data={'data': overpass_query})
        # Raise an HTTPError if the request was unsuccessful.
        response.raise_for_status()
        # Return the parsed JSON response.
        return response.json()

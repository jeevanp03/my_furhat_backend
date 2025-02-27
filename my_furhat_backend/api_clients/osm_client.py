import requests

class OSMClient:
    """
    A client for interacting with the OpenStreetMap Nominatim API to search for points of interest (POIs).

    This client constructs requests to the Nominatim search endpoint and returns the parsed JSON response.
    """

    def __init__(self, base_url: str = "https://nominatim.openstreetmap.org"):
        """
        Initialize the OSMClient with a base URL.

        Parameters:
            base_url (str): The base URL of the Nominatim service.
                            Default is "https://nominatim.openstreetmap.org".
        """
        self.base_url = base_url

    def search_pois(self, lat: float, lon: float, query: str, limit: int = 10) -> list:
        """
        Search for points of interest (POIs) using the Nominatim API.

        Constructs and sends a GET request to the Nominatim search endpoint with the specified parameters,
        and returns a list of POIs that match the query.

        Parameters:
            lat (float): Latitude of the location.
            lon (float): Longitude of the location.
            query (str): The search term, e.g., "restaurant", "museum".
            limit (int): Maximum number of results to return. Default is 10.

        Returns:
            list: A list of dictionaries, each representing a POI with details as provided by Nominatim.

        Raises:
            requests.HTTPError: If the HTTP request fails.
        """
        # Build the URL for the Nominatim search endpoint.
        url = f"{self.base_url}/search"
        
        # Set up query parameters required by Nominatim.
        params = {
            "q": query,            # The search term.
            "format": "json",      # Request response in JSON format.
            "addressdetails": 1,   # Include detailed address information in the response.
            "limit": limit,        # Limit on the number of results.
            "lat": lat,            # Latitude coordinate.
            "lon": lon             # Longitude coordinate.
        }
        
        # Define custom headers to include a User-Agent as required by Nominatim's usage policy.
        headers = {
            "User-Agent": "my-concierge-agent/1.0 (contact@example.com)"
        }
        
        # Send the GET request to the Nominatim search endpoint.
        response = requests.get(url, params=params, headers=headers)
        
        # Raise an exception if the HTTP request returned an unsuccessful status code.
        response.raise_for_status()
        
        # Return the JSON response parsed into a Python list.
        return response.json()

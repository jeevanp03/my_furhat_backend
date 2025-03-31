from langchain_core.tools import tool
from pydantic import BaseModel, Field
from my_furhat_backend.api_clients.foursquare_client import FoursquareClient
from my_furhat_backend.api_clients.osm_client import OSMClient
from my_furhat_backend.api_clients.overpass_client import OverpassClient

# Input schema for the Foursquare tool using Pydantic.
class FoursquareInput(BaseModel):
    lat: float = Field(..., description="Latitude coordinate")
    lon: float = Field(..., description="Longitude coordinate")
    query: str = Field(..., description="Search query for places")
    tool_call_id: str = Field(None, description="Optional tool call identifier")

@tool("foursquare_tool", args_schema=FoursquareInput)
def foursquare_tool(lat: float, lon: float, query: str, tool_call_id: str = None) -> dict:
    """
    Search for venues using the Foursquare API.

    This tool uses the FoursquareClient to search for places near the specified coordinates
    that match the given search query.

    Parameters:
        lat (float): Latitude coordinate.
        lon (float): Longitude coordinate.
        query (str): Search query string to filter venues.
        tool_call_id (str, optional): An optional identifier for tracking the tool call.

    Returns:
        dict: A dictionary with a key "foursquare_results" containing the list of search results.
    """
    # Instantiate the Foursquare API client.
    client = FoursquareClient()
    # Format latitude and longitude into a comma-separated string as expected by the API.
    ll = f"{lat},{lon}"
    # Perform the search and extract the 'results' from the returned data.
    results = client.search_places(ll, query)
    return {"foursquare_results": results.get("results", [])}


# Input schema for the OSM (OpenStreetMap) tool using Pydantic.
class OSMInput(BaseModel):
    lat: float = Field(..., description="Latitude coordinate")
    lon: float = Field(..., description="Longitude coordinate")
    query: str = Field(..., description="Search query for POIs")
    tool_call_id: str = Field(None, description="Optional tool call identifier")

@tool("osm_tool", args_schema=OSMInput)
def osm_tool(lat: float, lon: float, query: str, tool_call_id: str = None) -> list:
    """
    Search for Points of Interest (POIs) using OpenStreetMap's Nominatim service.

    This tool utilizes the OSMClient to query for POIs based on the provided location and search query.

    Parameters:
        lat (float): Latitude coordinate.
        lon (float): Longitude coordinate.
        query (str): Search query string to filter POIs.
        tool_call_id (str, optional): An optional identifier for tracking the tool call.

    Returns:
        list: A list of POIs found by the search.
    """
    # Instantiate the OSM API client.
    client = OSMClient()
    # Perform the search for POIs using the given parameters.
    return client.search_pois(lat, lon, query)


# Input schema for the Overpass tool using Pydantic.
class OverpassInput(BaseModel):
    lat: float = Field(..., description="Latitude coordinate")
    lon: float = Field(..., description="Longitude coordinate")
    query: str = Field(..., description="Search query for POIs")
    tool_call_id: str = Field(None, description="Optional tool call identifier")

@tool("overpass_tool", args_schema=OverpassInput)
def overpass_tool(lat: float, lon: float, query: str, tool_call_id: str = None) -> dict:
    """
    Search for Points of Interest (POIs) using the Overpass API.

    This tool uses the OverpassClient to query for POIs around the specified location that match the given query.

    Parameters:
        lat (float): Latitude coordinate.
        lon (float): Longitude coordinate.
        query (str): Search query string to filter POIs.
        tool_call_id (str, optional): An optional identifier for tracking the tool call.

    Returns:
        dict: A dictionary containing the search results.
    """
    # Instantiate the Overpass API client.
    client = OverpassClient()
    # Perform the POI search using the client.
    return client.search_pois(lat, lon, query)


# Input schema for restaurant search using Foursquare, defined using Pydantic.
class RestaurantsSearchInput(BaseModel):
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    query: str = Field(..., description="Search query for restaurants")
    tool_call_id: str = Field(None, description="Optional tool call identifier")

@tool("restaurants_search", args_schema=RestaurantsSearchInput)
def restaurants_search(latitude: float, longitude: float, query: str, tool_call_id: str = None) -> dict:
    """
    Search for restaurants using the Foursquare API.

    This tool is similar to the foursquare_tool but is tailored for restaurant searches.
    It uses the FoursquareClient to perform the search based on the provided coordinates and query.

    Parameters:
        latitude (float): Latitude coordinate.
        longitude (float): Longitude coordinate.
        query (str): Search query string to filter restaurants.
        tool_call_id (str, optional): An optional identifier for tracking the tool call.

    Returns:
        dict: A dictionary with a key "foursquare_results" containing the list of restaurant search results.
    """
    # Instantiate the Foursquare API client.
    client = FoursquareClient()
    # Format the coordinates into a comma-separated string.
    ll = f"{latitude},{longitude}"
    # Perform the search and extract the 'results' from the returned data.
    results = client.search_places(ll, query)
    return {"foursquare_results": results.get("results", [])}

# List of tools provided by this module.
tools = [foursquare_tool, osm_tool, overpass_tool, restaurants_search]

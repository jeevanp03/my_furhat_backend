from langchain_core.tools import tool
from pydantic import BaseModel, Field
from my_furhat_backend.api_clients.foursquare_client import FoursquareClient
from my_furhat_backend.api_clients.osm_client import OSMClient
from my_furhat_backend.api_clients.overpass_client import OverpassClient

class FoursquareInput(BaseModel):
    lat: float = Field(..., description="Latitude coordinate")
    lon: float = Field(..., description="Longitude coordinate")
    query: str = Field(..., description="Search query for places")
    tool_call_id: str = Field(None, description="Optional tool call identifier")

@tool("foursquare_tool", args_schema=FoursquareInput)
def foursquare_tool(lat: float, lon: float, query: str, tool_call_id: str = None) -> dict:
    """
    Search for venues using Foursquare.
    """
    client = FoursquareClient()
    ll = f"{lat},{lon}"
    results = client.search_places(ll, query)
    return {"foursquare_results": results.get("results", [])}


class OSMInput(BaseModel):
    lat: float = Field(..., description="Latitude coordinate")
    lon: float = Field(..., description="Longitude coordinate")
    query: str = Field(..., description="Search query for POIs")
    tool_call_id: str = Field(None, description="Optional tool call identifier")

@tool("osm_tool", args_schema=OSMInput)
def osm_tool(lat: float, lon: float, query: str, tool_call_id: str = None) -> list:
    """
    Search for POIs using OSM's Nominatim.
    """
    client = OSMClient()
    return client.search_pois(lat, lon, query)


class OverpassInput(BaseModel):
    lat: float = Field(..., description="Latitude coordinate")
    lon: float = Field(..., description="Longitude coordinate")
    query: str = Field(..., description="Search query for POIs")
    tool_call_id: str = Field(None, description="Optional tool call identifier")

@tool("overpass_tool", args_schema=OverpassInput)
def overpass_tool(lat: float, lon: float, query: str, tool_call_id: str = None) -> dict:
    """
    Search for POIs using the Overpass API.
    """
    client = OverpassClient()
    return client.search_pois(lat, lon, query)

class RestaurantsSearchInput(BaseModel):
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    query: str = Field(..., description="Search query for restaurants")
    tool_call_id: str = Field(None, description="Optional tool call identifier")

@tool("restaurants_search", args_schema=RestaurantsSearchInput)
def restaurants_search(latitude: float, longitude: float, query: str, tool_call_id: str = None) -> dict:
    """
    Search for venues using Foursquare.
    """
    client = FoursquareClient()
    ll = f"{latitude},{longitude}"
    results = client.search_places(ll, query)
    return {"foursquare_results": results.get("results", [])}

tools = [restaurants_search]
# tools = [foursquare_tool, osm_tool, overpass_tool]

__all__ = ["tools"]


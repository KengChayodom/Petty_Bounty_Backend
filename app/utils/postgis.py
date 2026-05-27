"""
PostGIS utility functions for geospatial data handling.

This module provides helper functions for converting between
different coordinate formats and creating PostGIS spatial objects.
"""
from typing import Union


def create_postgis_point(latitude: float, longitude: float) -> str:
    """
    Convert latitude and longitude to PostGIS POINT format with SRID.

    Note: PostGIS uses longitude first (x, y) in POINT format.
    This function handles the conversion automatically.

    Args:
        latitude: Latitude coordinate (-90 to 90)
        longitude: Longitude coordinate (-180 to 180)

    Returns:
        PostGIS POINT string with SRID: "ST_SetSRID(ST_MakePoint(lng, lat), 4326)"

    Example:
        >>> create_postgis_point(13.7563, 100.5018)
        "ST_SetSRID(ST_MakePoint(100.5018,13.7563),4326)"
    """
    return f"POINT({longitude} {latitude})"


def create_postgis_point_from_dict(coords: dict) -> str:
    """
    Convert a dictionary with lat/lng keys to PostGIS POINT format.

    Args:
        coords: Dictionary with 'latitude' and 'longitude' keys

    Returns:
        PostGIS POINT string with SRID

    Raises:
        KeyError: If required keys are missing
        ValueError: If coordinates are out of valid range
    """
    latitude = coords.get('latitude') or coords.get('lat')
    longitude = coords.get('longitude') or coords.get('lng')

    if latitude is None:
        raise KeyError("latitude or lat key required")
    if longitude is None:
        raise KeyError("longitude or lng key required")

    # Validate ranges
    if not -90 <= latitude <= 90:
        raise ValueError(f"Latitude must be between -90 and 90, got {latitude}")
    if not -180 <= longitude <= 180:
        raise ValueError(f"Longitude must be between -180 and 180, got {longitude}")

    return create_postgis_point(latitude, longitude)


def coordinates_from_point(point_wkt: str) -> tuple[float, float]:
    """
    Extract latitude and longitude from a PostGIS POINT string.

    Args:
        point_wkt: PostGIS POINT string in format "POINT(lng lat)"

    Returns:
        Tuple of (latitude, longitude)

    Example:
        >>> coordinates_from_point("POINT(100.5018 13.7563)")
        (13.7563, 100.5018)
    """
    # Remove "POINT(" and ")" then split by space
    coords = point_wkt.replace("POINT(", "").replace(")", "").split()

    if len(coords) != 2:
        raise ValueError(f"Invalid POINT format: {point_wkt}")

    longitude = float(coords[0])
    latitude = float(coords[1])

    return latitude, longitude


def format_geojson_point(latitude: float, longitude: float) -> dict:
    """
    Convert lat/lng to GeoJSON Point format.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate

    Returns:
        GeoJSON Point dictionary

    Example:
        >>> format_geojson_point(13.7563, 100.5018)
        {"type": "Point", "coordinates": [100.5018, 13.7563]}
    """
    return {
        "type": "Point",
        "coordinates": [longitude, latitude]  # GeoJSON uses [lng, lat]
    }


def calculate_distance_meters(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate the Haversine distance between two points in meters.

    This is a simple approximation. For production use,
    consider using PostGIS's ST_Distance function instead.

    Args:
        lat1: First point latitude
        lon1: First point longitude
        lat2: Second point latitude
        lon2: Second point longitude

    Returns:
        Distance in meters
    """
    import math

    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon1_rad = math.radians(lon1)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2) ** 2 +
        math.cos(lat1_rad) * math.cos(lat2_rad) *
        math.sin(dlon / 2) ** 2
    )

    c = 2 * math.asin(math.sqrt(a))

    # Earth's radius in meters
    earth_radius = 6371000

    return earth_radius * c

"""
Unit tests for app/utils/postgis.py.

These functions format the WKT that every sighting/missing-pet INSERT writes
to the geography column, so a silent error here corrupts spatial matching for
the whole product. Boundary Value Analysis is applied to the lat/lon ranges:
the poles (+-90), the antimeridian (+-180), the equator/prime-meridian (0),
and just-outside values.
"""
import math

import pytest

from app.utils.postgis import (
    calculate_distance_meters,
    coordinates_from_point,
    create_postgis_point,
    create_postgis_point_from_dict,
    format_geojson_point,
)


# --------------------------------------------------------------------------- #
# create_postgis_point — emits WKT "POINT(lng lat)" (lng first).
#
# NOTE: the function's docstring claims it returns an ST_SetSRID(...) string;
# the implementation actually returns plain WKT. These tests pin the *real*
# behaviour (what callers depend on), not the stale docstring.
# --------------------------------------------------------------------------- #
class TestCreatePostgisPoint:
    def test_emits_lng_then_lat_in_wkt(self):
        # Bangkok: lat 13.7563, lng 100.5018 -> lng must come first in WKT.
        assert create_postgis_point(13.7563, 100.5018) == "POINT(100.5018 13.7563)"

    def test_negative_southern_western_hemisphere(self):
        assert create_postgis_point(-33.8688, -70.6693) == "POINT(-70.6693 -33.8688)"

    def test_null_island_zero_zero(self):
        # Guards against a regression that drops a 0.0 coordinate.
        assert create_postgis_point(0.0, 0.0) == "POINT(0.0 0.0)"

    @pytest.mark.parametrize(
        "lat,lng,expected",
        [
            (90.0, 180.0, "POINT(180.0 90.0)"),     # NE extreme
            (-90.0, -180.0, "POINT(-180.0 -90.0)"),  # SW extreme
        ],
    )
    def test_extreme_corners(self, lat, lng, expected):
        assert create_postgis_point(lat, lng) == expected


# --------------------------------------------------------------------------- #
# create_postgis_point_from_dict — validates ranges and accepts key aliases.
# --------------------------------------------------------------------------- #
class TestCreatePostgisPointFromDict:
    def test_long_form_keys(self):
        out = create_postgis_point_from_dict({"latitude": 13.75, "longitude": 100.5})
        assert out == "POINT(100.5 13.75)"

    def test_short_alias_keys(self):
        out = create_postgis_point_from_dict({"lat": 13.75, "lng": 100.5})
        assert out == "POINT(100.5 13.75)"

    def test_missing_latitude_raises_keyerror(self):
        with pytest.raises(KeyError):
            create_postgis_point_from_dict({"longitude": 100.5})

    def test_missing_longitude_raises_keyerror(self):
        with pytest.raises(KeyError):
            create_postgis_point_from_dict({"latitude": 13.75})

    # --- range BVA -------------------------------------------------------- #
    @pytest.mark.parametrize("lat", [90.0, -90.0])
    def test_latitude_at_pole_is_valid(self, lat):
        # boundary value: exactly +-90 must be accepted
        create_postgis_point_from_dict({"latitude": lat, "longitude": 10.0})

    @pytest.mark.parametrize("lat", [90.0001, -90.0001])
    def test_latitude_just_past_pole_rejected(self, lat):
        with pytest.raises(ValueError):
            create_postgis_point_from_dict({"latitude": lat, "longitude": 10.0})

    @pytest.mark.parametrize("lng", [180.0, -180.0])
    def test_longitude_at_antimeridian_is_valid(self, lng):
        create_postgis_point_from_dict({"latitude": 10.0, "longitude": lng})

    @pytest.mark.parametrize("lng", [180.0001, -180.0001])
    def test_longitude_just_past_antimeridian_rejected(self, lng):
        with pytest.raises(ValueError):
            create_postgis_point_from_dict({"latitude": 10.0, "longitude": lng})

    # --- regression: zero coordinates must NOT be treated as "missing" ---- #
    # `coords.get('latitude') or coords.get('lat')` short-circuits on the
    # falsy 0.0, so a sighting on the equator or prime meridian used to raise
    # a spurious KeyError. These pin the fix.
    def test_zero_latitude_equator_is_accepted(self):
        out = create_postgis_point_from_dict({"latitude": 0.0, "longitude": 100.5})
        assert out == "POINT(100.5 0.0)"

    def test_zero_longitude_prime_meridian_is_accepted(self):
        out = create_postgis_point_from_dict({"latitude": 51.4779, "longitude": 0.0})
        assert out == "POINT(0.0 51.4779)"

    def test_null_island_via_dict_is_accepted(self):
        out = create_postgis_point_from_dict({"latitude": 0.0, "longitude": 0.0})
        assert out == "POINT(0.0 0.0)"


# --------------------------------------------------------------------------- #
# coordinates_from_point — inverse of create_postgis_point.
# --------------------------------------------------------------------------- #
class TestCoordinatesFromPoint:
    def test_parses_lat_lng_in_correct_order(self):
        lat, lng = coordinates_from_point("POINT(100.5018 13.7563)")
        assert (lat, lng) == (13.7563, 100.5018)

    def test_round_trips_with_create(self):
        wkt = create_postgis_point(-33.8688, 151.2093)
        lat, lng = coordinates_from_point(wkt)
        assert lat == pytest.approx(-33.8688)
        assert lng == pytest.approx(151.2093)

    @pytest.mark.parametrize("bad", ["POINT(1 2 3)", "POINT(1)", "POINT()"])
    def test_malformed_point_raises_valueerror(self, bad):
        with pytest.raises(ValueError):
            coordinates_from_point(bad)


# --------------------------------------------------------------------------- #
# format_geojson_point — GeoJSON also uses [lng, lat] ordering.
# --------------------------------------------------------------------------- #
class TestFormatGeojsonPoint:
    def test_geojson_shape_and_lng_lat_order(self):
        assert format_geojson_point(13.7563, 100.5018) == {
            "type": "Point",
            "coordinates": [100.5018, 13.7563],
        }


# --------------------------------------------------------------------------- #
# calculate_distance_meters — Haversine.
# --------------------------------------------------------------------------- #
class TestCalculateDistanceMeters:
    def test_identical_points_is_zero(self):
        assert calculate_distance_meters(13.75, 100.5, 13.75, 100.5) == pytest.approx(0.0)

    def test_is_symmetric(self):
        a = calculate_distance_meters(13.75, 100.5, 18.78, 98.99)
        b = calculate_distance_meters(18.78, 98.99, 13.75, 100.5)
        assert a == pytest.approx(b)

    def test_one_degree_latitude_is_about_111km(self):
        # 1 deg of latitude ~= 111.19 km on a sphere of r=6371 km.
        d = calculate_distance_meters(0.0, 0.0, 1.0, 0.0)
        assert d == pytest.approx(111_195, rel=0.01)

    def test_known_city_pair_bangkok_chiangmai(self):
        # Bangkok -> Chiang Mai great-circle distance is ~580 km.
        d = calculate_distance_meters(13.7563, 100.5018, 18.7883, 98.9853)
        assert d == pytest.approx(580_000, rel=0.05)
        assert not math.isnan(d)

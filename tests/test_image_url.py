"""
The photo-URL allowlist: which addresses the server will download for a caller.

The defect it closes: `image_url` was a bare `str` that went straight into an
HTTP GET made by the server, so any signed-in account could aim the server at an
address only the server can reach and read the outcome from the error text.

Two layers are tested. The pure function carries the rule and gets the boundary
cases. The schema cases prove the rule is actually wired to every request that
carries a photo URL, which the pure function cannot show: it would stay green
if a fourth schema forgot to use it.
"""
import pytest
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.missing_pets import MissingPetCreate
from app.schemas.sightings import (
    AnalyzeRequest,
    SightingCreate,
    TargetedSightingCreate,
)
from app.utils.image_url import validate_image_url

HOST = "proj.supabase.co"
GOOD = f"https://{HOST}/storage/v1/object/public/pet-images/a.jpg"


class TestValidateImageUrl:
    def test_the_projects_own_storage_is_accepted_and_returned_unchanged(self):
        assert validate_image_url(GOOD, {HOST}) == GOOD

    def test_case_and_port_do_not_matter(self):
        url = "HTTPS://PROJ.Supabase.CO:8443/x.jpg"
        assert validate_image_url(url, {HOST}) == url

    @pytest.mark.parametrize("url", [
        "http://169.254.169.254/latest/meta-data/",   # cloud metadata
        "http://localhost:8000/admin/reports",        # the API itself
        "http://10.0.0.5/internal.png",               # private network
        "https://example.com/x.jpg",                  # any other host
    ])
    def test_an_address_outside_the_allowlist_is_refused(self, url):
        with pytest.raises(ValueError):
            validate_image_url(url, {HOST})

    def test_credentials_in_the_url_cannot_smuggle_another_host(self):
        """The host of this URL is evil.example, whatever comes before the @."""
        with pytest.raises(ValueError):
            validate_image_url(f"https://{HOST}@evil.example/x.jpg", {HOST})

    def test_a_lookalike_subdomain_is_not_the_allowed_host(self):
        with pytest.raises(ValueError):
            validate_image_url(f"https://{HOST}.evil.example/x.jpg", {HOST})

    @pytest.mark.parametrize("url", [
        "file:///etc/passwd",
        "ftp://proj.supabase.co/x.jpg",
        "gopher://proj.supabase.co/x",
        "//proj.supabase.co/x.jpg",
        "proj.supabase.co/x.jpg",
    ])
    def test_only_http_and_https_are_fetched(self, url):
        with pytest.raises(ValueError):
            validate_image_url(url, {HOST})

    @pytest.mark.parametrize("url", ["", "   ", None])
    def test_an_empty_value_is_refused(self, url):
        with pytest.raises(ValueError):
            validate_image_url(url, {HOST})

    def test_no_configured_hosts_refuses_everything(self):
        """A missing configuration must close the door, not open it."""
        with pytest.raises(ValueError):
            validate_image_url(GOOD, set())
        with pytest.raises(ValueError):
            validate_image_url(GOOD, None)


@pytest.fixture
def enforced(monkeypatch):
    """Turn the rule back on, against a known project host."""
    monkeypatch.setattr(settings, "IMAGE_URL_ALLOWLIST_ENFORCED", True)
    monkeypatch.setattr(settings, "SUPABASE_URL", f"https://{HOST}")


_SIGHTING = dict(hunter_id="h1", latitude=13.7, longitude=100.5,
                 detected_species="Cat")
_PET = dict(pet_name="Mochi", species="Cat", characteristics={"c": "x"},
            bounty_amount=100, latitude=13.7, longitude=100.5,
            last_seen_time="2026-09-01T10:00:00Z")

# One row per request schema that carries a photo URL the server downloads.
_SCHEMAS = [
    pytest.param(AnalyzeRequest, {}, id="analyze"),
    pytest.param(SightingCreate, _SIGHTING, id="discovery-sighting"),
    pytest.param(TargetedSightingCreate,
                 {**_SIGHTING, "target_pet_id": "p1"}, id="targeted-sighting"),
    pytest.param(MissingPetCreate, _PET, id="missing-pet"),
]


@pytest.mark.usefixtures("enforced")
class TestEveryRequestThatCarriesAPhotoUsesTheRule:
    @pytest.mark.parametrize("schema, rest", _SCHEMAS)
    def test_the_projects_storage_is_accepted(self, schema, rest):
        assert schema(image_url=GOOD, **rest).image_url == GOOD

    @pytest.mark.parametrize("schema, rest", _SCHEMAS)
    def test_an_internal_address_is_a_validation_error(self, schema, rest):
        with pytest.raises(ValidationError):
            schema(image_url="http://169.254.169.254/latest/meta-data/", **rest)


class TestTheProjectHostIsAlwaysAllowed:
    def test_extra_hosts_come_from_the_environment(self, monkeypatch):
        monkeypatch.setattr(settings, "SUPABASE_URL", f"https://{HOST}")
        monkeypatch.setenv("IMAGE_ALLOWED_HOSTS", " Cdn.Example.com , ,other.io")
        assert settings.image_allowed_hosts() == {
            HOST, "cdn.example.com", "other.io",
        }

    def test_a_local_stack_url_yields_its_host(self, monkeypatch):
        monkeypatch.setattr(settings, "SUPABASE_URL", "http://127.0.0.1:54321")
        monkeypatch.delenv("IMAGE_ALLOWED_HOSTS", raising=False)
        assert settings.image_allowed_hosts() == {"127.0.0.1"}

"""Field types shared by more than one request schema."""
from typing import Annotated

from pydantic import AfterValidator

from app.core.config import settings
from app.utils.image_url import validate_image_url


def _check_image_url(value: str) -> str:
    if not settings.IMAGE_URL_ALLOWLIST_ENFORCED:
        return value
    return validate_image_url(value, settings.image_allowed_hosts())


# A photo URL the server will download. One definition for every request that
# carries one, so the rule cannot be applied to three endpoints and forgotten on
# the fourth. A failure is a 422 from Pydantic before any handler runs.
ImageUrl = Annotated[str, AfterValidator(_check_image_url)]

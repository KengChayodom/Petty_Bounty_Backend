"""Which image URLs the server is willing to fetch on a caller's behalf.

Every AI endpoint downloads the photo it is given by URL. Left unchecked, that
lets any signed-in account point the server at an address only the server can
reach (a cloud metadata endpoint, an internal service) and read the outcome from
the error that comes back. The photos this product handles are uploaded to its
own storage, so the rule is to accept only that host.

Pure on purpose: no settings, no I/O. The caller supplies the allowed hosts.
"""
from urllib.parse import urlsplit

_ALLOWED_SCHEMES = ("http", "https")


def validate_image_url(url: str, allowed_hosts) -> str:
    """Return `url` unchanged if the server may fetch it, else raise ValueError.

    Compares the parsed hostname, never a substring, so
    `https://good.example@evil.example/` (whose host is evil.example) and
    `https://good.example.evil.example/` are both refused. Case and port are
    ignored. An empty `allowed_hosts` refuses everything: a missing
    configuration must close the door, not open it.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("image_url is required")
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname
    except ValueError:
        raise ValueError("image_url is not a valid URL") from None

    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        raise ValueError("image_url must be an http or https URL")
    if not host:
        raise ValueError("image_url has no host")

    permitted = {h.strip().lower() for h in (allowed_hosts or ()) if h and h.strip()}
    if host.lower() not in permitted:
        raise ValueError("image_url must point at this project's image storage")
    return url

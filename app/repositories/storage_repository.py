"""The object-storage port (POST /upload/pet-image).

Owned by this codebase, like the table ports beside it: UploadService depends
on it, tests double it. It is the reason the storage SDK no longer appears in a
route. Sync on purpose, because supabase-py's storage client is blocking.
"""
from typing import Protocol


class StorageRepository(Protocol):
    def upload_public(self, path: str, data: bytes, content_type: str) -> str:
        """Store `data` at `path` and return the object's public URL."""
        ...

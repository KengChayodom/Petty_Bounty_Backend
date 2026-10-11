"""
Photo upload for lost-pet reports and profile pictures.

The file checks (name, extension, content type, size) live here rather than in
the route, so the route only translates HTTP. Storage is reached through the
StorageRepository port, so this service holds zero supabase-py calls.
"""
import asyncio
import uuid

from app.repositories.storage_repository import StorageRepository

# Tuples, not sets, so the "Allowed: ..." message lists them in a fixed order.
ALLOWED_EXTENSIONS = ("jpg", "jpeg", "png", "webp")
ALLOWED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/webp")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def validate_image_upload(
    filename: str | None, content_type: str | None, size: int,
) -> str:
    """Check an uploaded photo and return its lower-case extension.

    Raises:
        ValueError: the file has no name, an unsupported extension or content
            type, or is over the size limit (the API answers 400).
    """
    if not filename:
        raise ValueError("No filename provided")

    extension = filename.split(".")[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError(
            f"Invalid content type. Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}"
        )
    if size > MAX_UPLOAD_BYTES:
        raise ValueError("File size exceeds 10MB limit")
    return extension


class UploadService:
    def __init__(self, storage: StorageRepository):
        self.storage = storage

    async def upload_pet_image(
        self, filename: str | None, content_type: str | None, data: bytes,
    ) -> dict:
        """MD-17 — store a validated photo under a fresh name.

        Returns `{image_url, filename}`.

        Raises:
            ValueError: the file failed validation. Nothing is written.
        """
        extension = validate_image_upload(filename, content_type, len(data))
        unique_filename = f"{uuid.uuid4()}.{extension}"

        # The storage client is synchronous, so calling it inline would block
        # the event loop for the whole upload, a stall that grows with the
        # user's file size and connection.
        public_url = await asyncio.to_thread(
            self.storage.upload_public, unique_filename, data, content_type,
        )
        return {"image_url": public_url, "filename": unique_filename}

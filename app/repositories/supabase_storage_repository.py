"""Supabase Storage adapter for StorageRepository (POST /upload/pet-image)."""

# The one public bucket the API writes to. Lost-pet photos and profile photos
# both land here.
PET_IMAGES_BUCKET = "pet-images"


class SupabaseStorageRepository:
    def __init__(self, db, bucket: str = PET_IMAGES_BUCKET):
        self._db = db
        self._bucket = bucket

    def upload_public(self, path: str, data: bytes, content_type: str) -> str:
        bucket = self._db.storage.from_(self._bucket)
        bucket.upload(
            path=path,
            file=data,
            file_options={"content-type": content_type},
        )
        return bucket.get_public_url(path)

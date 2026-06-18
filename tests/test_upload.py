"""
Route unit tests for POST /upload/pet-image  (UTC-14, MD-17, SRS-28 / UD-07).

Boundary rule: auth is overridden; the Supabase Storage surface
(`storage.from_(bucket).upload(...) / .get_public_url(...)`) is a purpose-built
double so we can assert the exact bytes/content-type handed to storage and that
bad files are rejected BEFORE any storage write. The 10 MB ceiling, the
extension allow-list, and the content-type allow-list are all validated.
"""
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import upload as upload_api
from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client


# --------------------------------------------------------------------------- #
# Storage double — records upload/get_public_url calls; can simulate a failure.
# --------------------------------------------------------------------------- #
class _Storage:
    def __init__(self, upload_raises=False):
        self._upload_raises = upload_raises
        self.bucket = None
        self.upload_calls = []      # [(bucket, path, file_bytes, file_options), ...]
        self.public_calls = []      # [(bucket, name), ...]

    def from_(self, bucket):
        self.bucket = bucket
        return self

    def upload(self, path, file, file_options):
        self.upload_calls.append((self.bucket, path, file, file_options))
        if self._upload_raises:
            raise Exception("storage backend unavailable")
        return SimpleNamespace(path=path)

    def get_public_url(self, name):
        self.public_calls.append((self.bucket, name))
        return f"https://storage.example/object/pet-images/{name}"


class _DB:
    def __init__(self, storage):
        self.storage = storage


def _client(storage):
    app = FastAPI()
    app.include_router(upload_api.router)
    app.dependency_overrides[get_current_user_id] = lambda: "jwt-user-1111"
    app.dependency_overrides[get_supabase_client] = lambda: _DB(storage)
    return TestClient(app)


class TestUploadPetImage:
    def test_valid_jpeg_uploads(self):
        storage = _Storage()
        client = _client(storage)
        content = b"\xff\xd8\xff\xe0fake-jpeg-bytes"

        r = client.post(
            "/upload/pet-image",
            files={"file": ("cat.jpg", content, "image/jpeg")},
        )

        assert r.status_code == 200
        data = r.json()["data"]
        assert data["filename"].endswith(".jpg")
        assert data["filename"] in data["image_url"]
        # Exactly one write, to the pet-images bucket, with the exact bytes
        # and content-type the client sent.
        assert len(storage.upload_calls) == 1
        bucket, path, file_bytes, opts = storage.upload_calls[0]
        assert bucket == "pet-images"
        assert path.endswith(".jpg")
        assert file_bytes == content
        assert opts == {"content-type": "image/jpeg"}

    def test_disallowed_extension_rejected_before_write(self):
        for name, ctype in (("clip.gif", "image/gif"), ("doc.pdf", "application/pdf")):
            storage = _Storage()
            r = _client(storage).post(
                "/upload/pet-image",
                files={"file": (name, b"x", ctype)},
            )
            assert r.status_code == 400, name
            assert storage.upload_calls == []

    def test_allowed_extension_bad_content_type_rejected(self):
        storage = _Storage()
        r = _client(storage).post(
            "/upload/pet-image",
            files={"file": ("cat.jpg", b"x", "image/gif")},
        )
        assert r.status_code == 400
        assert storage.upload_calls == []

    def test_file_over_10mb_rejected(self):
        storage = _Storage()
        oversized = b"\x00" * (10 * 1024 * 1024 + 1)
        r = _client(storage).post(
            "/upload/pet-image",
            files={"file": ("big.png", oversized, "image/png")},
        )
        assert r.status_code == 400
        assert storage.upload_calls == []

    def test_storage_failure_is_500(self):
        storage = _Storage(upload_raises=True)
        r = _client(storage).post(
            "/upload/pet-image",
            files={"file": ("cat.png", b"data", "image/png")},
        )
        assert r.status_code == 500

-- 2026-09-30 — Storage: uploads need a login, and have a size and a type limit.
--
-- STATUS: PROPOSED. NOT APPLIED. NOT TESTED against a database: the local test
-- container has no `storage` schema, so read it against the live one before applying.
--
-- WHY. Read from the live catalog on 2026-09-30: the three buckets (pet-images,
-- profile-images, slip-images) are public, have no file_size_limit and no
-- allowed_mime_types, and the INSERT policies apply to role `public`, which includes
-- someone who is not signed in. Anyone holding the anon key can therefore upload
-- files of any type and size to a public URL on the project's own domain.
-- `sighting_repository.dart` already carries a note that these limits are missing.
--
-- LIMITS. 10 MB and jpeg, png and webp match what /upload/pet-image enforces and
-- what the AI pipeline can decode. A photo picked in HEIC format is already
-- unreadable to the backend today, so it is not allowed here either.
--
-- ROLLBACK:
--   UPDATE storage.buckets SET file_size_limit = NULL, allowed_mime_types = NULL
--    WHERE id IN ('pet-images', 'profile-images', 'slip-images');
--   and recreate the three dropped policies with `TO public`.

BEGIN;

UPDATE storage.buckets
   SET file_size_limit = 10485760,
       allowed_mime_types = ARRAY['image/jpeg', 'image/png', 'image/webp']
 WHERE id IN ('pet-images', 'profile-images', 'slip-images');

DROP POLICY IF EXISTS "Allow Upload x0itpi_0" ON storage.objects;   -- pet-images
DROP POLICY IF EXISTS "Allow Upload vejz8c_0" ON storage.objects;   -- profile-images
DROP POLICY IF EXISTS "Allow Upload g4ljtx_0" ON storage.objects;   -- slip-images

CREATE POLICY "Signed-in users upload photos" ON storage.objects
    FOR INSERT TO authenticated
    WITH CHECK (bucket_id IN ('pet-images', 'profile-images', 'slip-images'));

COMMIT;

BEGIN;

DROP FUNCTION IF EXISTS get_missing_pet_by_id(uuid);

CREATE FUNCTION get_missing_pet_by_id(p_pet_id uuid)
RETURNS TABLE (
    id                uuid,
    owner_id          uuid,
    pet_name          character varying,
    species           character varying,
    characteristics   jsonb,
    bounty_amount     numeric,
    latitude          double precision,
    longitude         double precision,
    last_seen_time    timestamp with time zone,
    image_url         text,
    status            character varying,
    created_at        timestamp with time zone,
    expires_at        timestamp with time zone,
    primary_color_hex character varying,
    pattern_id        character varying,
    owner_display_name character varying,
    owner_phone       character varying,
    owner_profile_image_url text
) LANGUAGE plpgsql STABLE AS $$
BEGIN
    RETURN QUERY
    SELECT
        mp.id,
        mp.owner_id,
        mp.pet_name,
        mp.species::character varying,
        mp.characteristics,
        mp.bounty_amount,
        ST_Y(mp.last_seen_location::geometry) AS latitude,
        ST_X(mp.last_seen_location::geometry) AS longitude,
        mp.last_seen_time,
        mp.image_url,
        mp.status::character varying,
        mp.created_at,
        mp.expires_at,
        mp.primary_color_hex,
        mp.pattern_id,
        u.display_name,
        u.phone,
        u.profile_image_url
    FROM public.missing_pets mp
    LEFT JOIN public.users u ON u.id = mp.owner_id
    WHERE mp.id = p_pet_id;
END;
$$;

COMMIT;

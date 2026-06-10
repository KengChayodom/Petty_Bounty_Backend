# Test Audit — `tests/test_sighting_service.py`

**Scope:** `TestProcessAndSaveCacheHit`, `TestProcessAndSaveCacheMiss`, and the
shared `wire_save_path` helper (the cache-MISS round and the HIT tests it
refactored).
**Date:** 2026-06-04 (updated after fixes #1–#2)
**Verdict:** substance is sound — **12/12 targeted mutations now killed**
(was 10/12; M8 & M9 closed by fixes #1–#2), async-mock wiring verified
mechanically, no vacuous assertions. Three reachable branch gaps remain
(resilience/error paths), documented below.

---

## Method

A **controlled mutation harness** applied 12 mutations to a copy of
`app/services/sighting_service.py`, each run against `tests/test_sighting_service.py`,
restoring the original after every run.

- `mutmut 3.x` was attempted first but its copy-and-run stats pass doesn't honor
  a scoped runner and chokes on this repo's `--integration` / testcontainers
  conftest plumbing → used a transparent targeted harness instead.
- One result (M10) was initially a false `SURVIVED` due to `__pycache__` mtime
  staleness (file rewritten within the same mtime-second → stale `.pyc`). Re-run
  with `PYTHONDONTWRITEBYTECODE=1` and cross-checked by hand.

---

## Per-test table

| Test | Exercises | Asserts (effect vs call) | Guards against (real bug) | Can it fail? (mutation) |
|---|---|---|---|---|
| **HIT-1** `…persists_user_species_and_cached_vector_not_yolo` | HIT branch `:96→97-100` (read cached vector), payload build `:124-133`, insert `:136`; short-circuits `:101-119` | **effect:** insert payload `detected_species` / `feature_vector` / `sighting_status` (reads what was *sent*). **call:** `assert_not_called` ×3 | Persisting YOLO's cached guess instead of user's species; not reusing cached vector; wrong status; running the pipeline on a hit | ✅ kills **M3** (species→const) & **M5** (status const) |
| **HIT-2** `…strips_feature_vector_…and_returns_matches` | HIT branch + response shaping `:177` pop, `:178` return; matches threaded from `:151` | **effect:** `feature_vector` absent from response. **passthrough:** `matches` echoed | Leaking the 512-d vector to clients; dropping matches from the response | ✅ kills **M7** (stop stripping) |
| **HIT-3** `…persists_matches_into_sighting_matches` *(fix #1)* | save-tail persist call-site `:165-167` | **effect:** `sighting_matches` upsert payload `[{sighting_id, missing_pet_id, similarity_score}]` | Returning matches but never writing them through to `sighting_matches` | ✅ kills **M8** (invert persist guard) |
| **HIT-4** `…target_pet_id_is_threaded_into_insert_payload` *(fix #2)* | payload `target_pet_id` True branch `:134-135` | **effect:** insert payload `initial_target_pet_id == "target-pet-1"` | Dropping the explicit target when the hunter reports against a specific pet | ✅ kills **M9** (invert target branch) |
| **HIT-5** `…initial_target_pet_id_omitted_when_no_target` *(fix #2)* | payload None branch `:134-135` | **effect:** `initial_target_pet_id` key absent (not even `None`) | Writing a spurious `initial_target_pet_id=None` for stray reports | ✅ kills **M9** (other direction) |
| **MISS-1** `…reruns_pipeline_and_inserts_recomputed_vector` | MISS block `:101-119` (download `:106`, yolo `:107`, isolate `:113`, unpack `:118`, clip `:119`), insert | **call/dataflow:** `assert_awaited_once_with` ×3. **effect:** insert `feature_vector==recomputed`, species=="Dog", stripped | Dropped `await`; wrong tuple element fed to CLIP; cached-vs-recomputed vector; YOLO species overriding user | ✅ kills **M1**, **M2**, **M11** (+M3) |
| **MISS-2** `…isolate_runs_without_species_constraint` | MISS up to isolate `:113` (runs full path, asserts only the call) | **call only:** `isolate_subject.assert_called_once_with(img, results)` | Constraining the re-run with `expected_species=` (breaks seed/live vector parity) | ✅ adding the kwarg breaks the call assertion — **weakest test: no effect assertion** |
| **MISS-3** `…yolo_finds_nothing_raises_and_skips_insert` | MISS failure path `:113-117` (iso None → raise `:115`) | **effect:** raises `ValueError(match=…)`; insert payload `is None` (no row). **call:** `clip_encode.assert_not_awaited` | Not raising on YOLO miss; encoding a non-existent subject; inserting a row anyway | ✅ kills **M6** (flip `is None`) |
| **helper** `wire_save_path` | n/a — programs `fake_db` echoes | n/a | n/a | n/a |

---

## Mutation results (real — 12 killed / 0 survived after fixes #1–#2)

| Mutant | Verdict |
|---|---|
| M1 drop `await download_image` · M2 swap isolated-frame index · M3 species→`"Cat"` · M4 `>=`→`>` · M5 status const · M6 flip `iso is None` · M7 stop stripping vector · M10 invert `if threshold` · M11 drop `await clip_encode` · M12 invert insert-failed guard | **KILLED** ✅ |
| **M8** invert `if matches:` (persist guard) | **KILLED** ✅ (was SURVIVED — closed by fix #1: `test_persists_matches_into_sighting_matches`) |
| **M9** invert `if sighting.target_pet_id:` | **KILLED** ✅ (was SURVIVED — closed by fix #2: `test_target_pet_id_is_threaded_into_insert_payload` + `test_initial_target_pet_id_omitted_when_no_target`) |

---

## Risk-area findings

### Vacuous assertions
None are `assert True` or assert-on-empty. **MISS-2 is the one soft spot:** it
drives the entire hot path (insert / match / persist all execute) but asserts
*only* the `isolate_subject` call — no persisted effect. Not decorative (it
guards the `expected_species` parity bug) but call-only. The
`assert_not_called` / `assert_awaited_once_with` assertions elsewhere are
call-based but each is paired with an effect assertion, so they are not vacuous.

### Async-mock claim — verified mechanically
`_make_ai` uses `AsyncMock` for exactly the three awaited coroutines
(`download_image:264`, `run_yolo_seg:265`, `clip_encode:267`) and a plain
`MagicMock` for the synchronous `isolate_subject:266`. Mutating the test both
ways:

- `isolate_subject` → `AsyncMock`: **3 MISS tests fail** (coroutine can't be
  tuple-unpacked at `:118`).
- `clip_encode` → plain `MagicMock`: **3 MISS tests fail** (`await` on a
  non-awaitable + missing `assert_awaited`).

A mismatch is **loudly caught, not swallowed.**

### Over-mocking
Clean. Mocked surfaces are the two real boundaries (Supabase client, AI
manager). `AnalyzeCache` is the **real** object, so the HIT/MISS branch decision
is genuine logic under test. Assertions read `payload_for(...)` = what the
service *sent to the boundary* (a real effect), not "a mock was called." We test
the service, not our fakes.

### Isolation / `wire_save_path` leak
`fake_db` is function-scoped (fresh per test). `wire_save_path` mutates only
that instance — no module/global state, no leak.

**Real fragility:** HIT and MISS use the *same* `image_url`; the MISS tests'
"cache is empty" precondition depends entirely on the autouse
`_clean_analyze_cache` fixture. The suite passing proves it works today (HIT runs
first, sets cache, teardown clears, MISS sees empty), but if that fixture were
removed you'd get a silent order-dependent failure. MISS tests would be more
robust using a distinct `image_url` or an explicit
`assert AnalyzeCache.get(url) is None`. Separately, `wire_save_path` hardcodes
`"detected_species":"Dog"` in its echoes regardless of the sighting (MISS-2 uses
`"Bird"`) — harmless now (nothing asserts the echo), but a trap for a future
test.

### Branch gaps in the MISS / hot path
(The branch figure includes wholly-untested methods; these are the *reachable*
gaps in `process_and_save_sighting`.)

- ~~**M8 survivor** — `_persist_matches` invocation unverified.~~ **CLOSED (fix #1)**
  — `test_persists_matches_into_sighting_matches` asserts the `sighting_matches`
  upsert payload.
- ~~**M9 survivor** — `target_pet_id` True branch unexercised.~~ **CLOSED (fix #2)**
  — both the set and unset branches of `:134-135` are now asserted.
- **Insert-failure** `:137-138` — no positive test that empty insert data →
  `ValueError` (M12 catches it only indirectly).
- **Match-RPC-failure resilience** `:150-158` — the documented "RPC fails but
  the row is still saved, matches=[]" path is untested.
- **Persist-failure resilience** `:166-173` — "persist throws, response
  unaffected" is untested.

---

## Naming audit
Names describe **behavior + outcome**, not mechanics
(`miss_reruns_pipeline_and_inserts_recomputed_vector`,
`miss_yolo_finds_nothing_raises_and_skips_insert`,
`persists_user_species_and_cached_vector_not_yolo`). They read well in a failure
report months later. No `test_process_2`-style names. ✅

---

## "Deferred correctly" vs "can't be unit-tested"
- **Genuinely integration** (correctly deferred, now covered): `match_missing_pets`,
  `resolve_missing_pet`, `sightings_for_pet` — real pgvector/PostGIS/plpgsql.
- **Actually unit-testable — punted, not legitimately deferred:**
  - `analyze_sighting_image` `:33-85` needs **no DB** — its *coordination logic*
    (three branches: iso `None`→`not_found`, success→cache-set+shape,
    exception→reraise) is unit-testable with the same `AsyncMock` AI manager +
    real `AnalyzeCache`. **Status: DEFERRED by decision (2026-06-10).** The
    branch logic is testable and should be; the *real YOLO/CLIP detection it
    wraps* is **MANUAL** (slow, hardware-non-deterministic — pinned by
    `debug_mochi_match.py` / `test_encode_parity.py`, see SRS §5.2). We chose not
    to add the mocked-AI branch test this round to keep scope to the two
    highest-value gaps; this remains the correct critique, not closed.
  - `get_sighting_by_id` `:257-270` (select + pop + None-on-empty) — unit-testable
    with `FakeSupabase`.
  - The **Python merge logic** in `get_hunter_activity` / `get_hunter_stats`
    (`matches_by_sighting` / `awards_by_sighting` assembly) is unit-testable; only
    their PostgREST *filter* semantics need integration.

---

## Weak spots / recommended additions (priority order)
1. ~~**Kill M8** — assert the persist call-site.~~ ✅ **DONE** —
   `test_persists_matches_into_sighting_matches`.
2. ~~**Kill M9** — `target_pet_id` threading.~~ ✅ **DONE** —
   `test_target_pet_id_is_threaded_into_insert_payload` +
   `test_initial_target_pet_id_omitted_when_no_target`.
3. **Resilience branches** — match-RPC raises → row still saved + `matches==[]`
   (`:150-158`); `_persist_matches` raises → response unaffected (`:166-173`).
   Documented behaviors, currently unverified.
4. **Strengthen MISS-2** — add one effect assertion (insert
   `feature_vector==recomputed`) so it's not call-only.
5. **De-fragilize isolation** — give MISS tests a distinct `image_url` or an
   explicit empty-cache assert.
6. **Unit-test `analyze_sighting_image`** branch logic (mocked AI, 3 branches) —
   **DEFERRED by decision** this round (see SRS §5.2); real detection stays MANUAL.
7. Optional: add a `[mutmut]` / cosmic-ray CI target so this mutation check runs
   nightly rather than by hand.

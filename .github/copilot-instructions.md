
# Project Context: Petty Bounty (Frontend Repository)

## 1. Project Overview & Stack
- **Description:** A community-driven mobile application (Flutter) that helps pet owners recover lost pets through geolocation-based alerting, AI-powered image matching, and a bounty reward system.
- **Development Paradigm:** Spec-Driven Development (SDD). All generated code MUST align with the specified feature plans and the architectural rules below.
- **Framework:** Flutter (Dart) — single codebase for iOS and Android.
- **State Management:** Riverpod (via hooks_riverpod / flutter_riverpod).
- **Navigation:** GoRouter.
- **Data Models:** Freezed + json_serializable for immutable data classes.

## 2. Backend Roles (CRITICAL RULE)
- **Supabase:** Used directly by Flutter ONLY for Authentication, Object Storage (uploading pet images), and Realtime Subscriptions. Flutter MUST NOT perform direct database CRUD operations (e.g., supabase.from('table').insert()).
- **FastAPI:** Acts as the main API gateway. Handles ALL database CRUD operations, business logic, and the AI pipelines (YOLO detection and Scikit-learn ranking). FastAPI communicates with the Supabase PostgreSQL database using the Python client. Flutter must use HTTP requests to FastAPI for database reads/writes.
## 3. API Flow
- **General Data Flow (Standard Features):** For standard features (e.g., User Profiles, Missing Pets CRUD, Bounties), Flutter MUST NOT connect to the Supabase database directly. Flutter sends HTTP requests (GET/POST/PUT/DELETE) to FastAPI. FastAPI handles the business logic and queries the Supabase PostgreSQL database.
- **Complex Flow 1 (AI Detection - Sightings):** Flutter uploads photo to Supabase Storage -> Receives public URL -> POST image URL to FastAPI `/sightings/analyze` -> FastAPI runs YOLO -> Returns species + confidence.
- **Complex Flow 2 (Geospatial Save - Sightings):** User confirms data -> POST sighting data to FastAPI `/sightings/` -> FastAPI formats location as PostGIS `POINT` -> FastAPI inserts into Supabase -> Returns success.
- **Complex Flow 3 (AI Ranking - Match Finder):** POST search params to FastAPI `/rank` -> FastAPI runs `pgvector` similarity search -> Returns ranked potential matches to Flutter.

## 4. Architecture & Directory Structure (Feature-First)
You MUST strictly follow the Feature-First Clean Architecture. NEVER mix layers or create files outside of this structure.

`lib/`
├── `main.dart`          # App entry point, env initialization
└── `src/`
    ├── `core/`          # App-wide config, constants, themes (e.g., app_config.dart)
    ├── `routing/`       # GoRouter definitions
    ├── `utils/`         # Global helpers, extensions
    ├── `presentation/`  # Shared UI widgets across multiple features
    └── `features/`      # Feature modules (e.g., map, missing_pets, sightings)
        └── `{feature_name}/`
            ├── `data/`         # Repositories, Data Sources, API calls, Supabase Storage
            ├── `domain/`       # Freezed Models, Entities, Riverpod Providers
            └── `presentation/` # Screens, Widgets, UI Controllers

## 5. Coding Guidelines & Quality Standards
When generating code, you MUST adhere to the following principles:

- **Strict Layering:** `presentation` (UI) MUST NOT communicate directly with `Supabase.instance` or `http` clients.
- **Communication Flow:** UI calls -> Riverpod Provider (Domain layer) -> Repository (Data layer).
- **Navigation:** Use `context.go()` or `context.push()` via GoRouter. NEVER use standard `Navigator.push`.
- **Data Models:** Always use `@freezed` and `json_serializable` for Data Models to eliminate boilerplate.
- **Naming Conventions:**
  - Files/Folders: `snake_case`
  - Classes: `PascalCase`
  - Variables/Methods: `camelCase` (must be intention-revealing without needing extra comments).
- **Error Handling:** Always wrap HTTP requests and Supabase calls in `try/catch` blocks. Throw informative error messages (e.g., `throw Exception('Failed to upload image: $e');`).
- **Null Safety (Strict):** NEVER use the force-unwrap operator (`!`). Use explicit null checks (`if (data != null)`) or early returns.
- **Network Validation:** Always check `if (response.statusCode == 200)` before decoding JSON from the FastAPI backend.
- **Async Operations:** Always use `async/await`. Do not use `.then()`.
- **Resource Management:** Always `dispose()` controllers (e.g., `CameraController`, `TextEditingController`) in `StatefulWidget`.
- **Secrets:** No hardcoded secrets. Retrieve URLs and API keys from `.env` via `app_config.dart`.

## 6. Pre-Generation Code Review Checklist
Before outputting the final code, verify internally:
[ ] Is the Flutter UI/Data layer completely free of direct Supabase database CRUD calls (e.g., .select(), .insert())?
[ ] Is Supabase usage in Flutter strictly limited to Auth, Storage, and Realtime?
[ ] Are all database read/write requests from Flutter routed through HTTP calls to the FastAPI backend?
[ ] Are all asynchronous functions using async/await?
[ ] Are models annotated with @freezed?
[ ] Are potential errors handled with try/catch blocks?
[ ] Are explicit null checks used instead of force-unwrapping (!)?
[ ] Does the code belong to the correct layer (data, domain, or presentation) within its feature folder?

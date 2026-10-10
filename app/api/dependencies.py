"""
Composition root: the one place that turns the request's Supabase client into
the adapters and services the routes use.

Routes ask for a service (or, for a background task, a port) with `Depends`,
and nothing below this module names a `Supabase*Repository`. Swapping an
adapter therefore touches this file only, and a test can replace any link of
the chain with `app.dependency_overrides[...]`: overriding a repository
provider also reaches every service built on top of it.
"""
from fastapi import Depends

from app.core.database import get_supabase_client
from app.repositories.device_token_repository import DeviceTokenRepository
from app.repositories.missing_pet_repository import MissingPetRepository
from app.repositories.notification_repository import NotificationRepository
from app.repositories.sighting_repository import SightingRepository
from app.repositories.storage_repository import StorageRepository
from app.repositories.supabase_admin_repository import SupabaseAdminRepository
from app.repositories.supabase_bounty_repository import SupabaseBountyRepository
from app.repositories.supabase_device_token_repository import (
    SupabaseDeviceTokenRepository,
)
from app.repositories.supabase_leaderboard_repository import (
    SupabaseLeaderboardRepository,
)
from app.repositories.supabase_missing_pet_repository import (
    SupabaseMissingPetRepository,
)
from app.repositories.supabase_notification_repository import (
    SupabaseNotificationRepository,
)
from app.repositories.supabase_report_repository import SupabaseReportRepository
from app.repositories.supabase_sighting_repository import (
    SupabaseSightingRepository,
)
from app.repositories.supabase_storage_repository import (
    SupabaseStorageRepository,
)
from app.repositories.supabase_user_repository import SupabaseUserRepository
from app.repositories.user_repository import UserRepository
from app.services.admin_service import AdminService
from app.services.ai_service import AIManager
from app.services.bounty_service import BountyService
from app.services.device_service import DeviceService
from app.services.leaderboard_service import LeaderboardService
from app.services.pet_service import PetService
from app.services.report_service import ReportService
from app.services.sighting_service import SightingService
from app.services.upload_service import UploadService
from app.services.user_service import UserService


# --- repositories (ports) ------------------------------------------------ #

def get_missing_pet_repository(
    supabase=Depends(get_supabase_client),
) -> MissingPetRepository:
    return SupabaseMissingPetRepository(supabase)


def get_sighting_repository(
    supabase=Depends(get_supabase_client),
) -> SightingRepository:
    return SupabaseSightingRepository(supabase)


def get_user_repository(
    supabase=Depends(get_supabase_client),
) -> UserRepository:
    return SupabaseUserRepository(supabase)


def get_device_token_repository(
    supabase=Depends(get_supabase_client),
) -> DeviceTokenRepository:
    return SupabaseDeviceTokenRepository(supabase)


def get_notification_repository(
    supabase=Depends(get_supabase_client),
) -> NotificationRepository:
    return SupabaseNotificationRepository(supabase)


def get_storage_repository(
    supabase=Depends(get_supabase_client),
) -> StorageRepository:
    return SupabaseStorageRepository(supabase)


# --- services ------------------------------------------------------------ #

def get_pet_service(
    repo: MissingPetRepository = Depends(get_missing_pet_repository),
) -> PetService:
    return PetService(repo)


def get_sighting_service(
    repo: SightingRepository = Depends(get_sighting_repository),
) -> SightingService:
    return SightingService(repo=repo, ai_manager=AIManager)


def get_user_service(
    repo: UserRepository = Depends(get_user_repository),
) -> UserService:
    return UserService(repo)


def get_device_service(
    repo: DeviceTokenRepository = Depends(get_device_token_repository),
) -> DeviceService:
    return DeviceService(repo)


def get_upload_service(
    storage: StorageRepository = Depends(get_storage_repository),
) -> UploadService:
    return UploadService(storage)


def get_report_service(
    supabase=Depends(get_supabase_client),
    sighting_repo: SightingRepository = Depends(get_sighting_repository),
) -> ReportService:
    return ReportService(
        repo=SupabaseReportRepository(supabase),
        sighting_repo=sighting_repo,
    )


def get_admin_service(
    supabase=Depends(get_supabase_client),
    user_repo: UserRepository = Depends(get_user_repository),
) -> AdminService:
    return AdminService(
        repo=SupabaseAdminRepository(supabase),
        report_repo=SupabaseReportRepository(supabase),
        user_repo=user_repo,
    )


def get_bounty_service(supabase=Depends(get_supabase_client)) -> BountyService:
    return BountyService(SupabaseBountyRepository(supabase))


def get_leaderboard_service(
    supabase=Depends(get_supabase_client),
) -> LeaderboardService:
    return LeaderboardService(SupabaseLeaderboardRepository(supabase))

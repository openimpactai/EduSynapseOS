# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Curriculum sync background tasks.

This module provides background tasks for synchronizing curriculum data
from the Central Curriculum service to tenant databases.

Tasks:
    - sync_curriculum_for_tenant: Sync curriculum for a single tenant
    - sync_curriculum_all_tenants: Sync curriculum for all tenants
    - daily_curriculum_sync_job: Scheduler job wrapper for daily sync

Example:
    >>> from src.infrastructure.background.tasks import sync_curriculum_for_tenant
    >>> sync_curriculum_for_tenant.send("acme")
"""

import logging
from typing import Any

import dramatiq

from src.infrastructure.background.broker import Priority, Queues, setup_dramatiq
from src.infrastructure.background.tasks.base import run_async

setup_dramatiq()

logger = logging.getLogger(__name__)


@dramatiq.actor(
    queue_name=Queues.CURRICULUM,
    max_retries=3,
    time_limit=1800000,  # 30 minutes (curriculum sync can be slow)
    priority=Priority.LOW,
)
def sync_curriculum_for_tenant(
    tenant_code: str,
    framework_code: str | None = None,
) -> dict[str, Any]:
    """Sync curriculum data for a specific tenant.

    Fetches curriculum data from Central Curriculum API and upserts
    it into the tenant database.

    Uses per-tenant CC credentials if configured in tenant.settings,
    otherwise falls back to global credentials.

    Args:
        tenant_code: Tenant to sync curriculum for.
        framework_code: Optional specific framework to sync. If None, syncs all
                       frameworks the tenant has profile access to.

    Returns:
        Sync result with counts and status.
    """
    logger.info("Starting curriculum sync for tenant: %s", tenant_code)

    async def _sync() -> dict[str, Any]:
        from sqlalchemy import select
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker

        from src.core.config import get_settings
        from src.domains.curriculum.sync_service import CurriculumSyncService
        from src.infrastructure.database.tenant_manager import get_worker_db_manager
        from src.infrastructure.database.models.central.tenant import Tenant

        settings = get_settings()

        if not settings.central_curriculum.sync_enabled:
            logger.info("Curriculum sync is disabled in settings")
            return {"status": "disabled", "tenant_code": tenant_code}

        # Fetch tenant record from central DB to get CC credentials
        cc_api_key = None
        cc_api_secret = None

        try:
            central_engine = create_async_engine(
                settings.central_database.url,
                echo=False,
            )
            async_session = sessionmaker(
                central_engine, class_=AsyncSession, expire_on_commit=False
            )

            async with async_session() as central_session:
                result = await central_session.execute(
                    select(Tenant).where(Tenant.code == tenant_code)
                )
                tenant_record = result.scalar_one_or_none()

                if tenant_record and tenant_record.settings:
                    cc_settings = tenant_record.settings.get("central_curriculum", {})
                    cc_api_key = cc_settings.get("api_key")
                    cc_api_secret = cc_settings.get("api_secret")

                    if cc_api_key and cc_api_secret:
                        logger.info(
                            "Using per-tenant CC credentials for tenant: %s",
                            tenant_code,
                        )
                    else:
                        logger.info(
                            "No per-tenant CC credentials for tenant: %s, using global",
                            tenant_code,
                        )

            await central_engine.dispose()
        except Exception as e:
            logger.warning(
                "Failed to fetch tenant CC credentials: %s, using global", str(e)
            )

        tenant_db = get_worker_db_manager()
        async with tenant_db.get_session(tenant_code) as session:
            # Initialize sync service with optional per-tenant credentials
            sync_service = CurriculumSyncService(
                session,
                settings.central_curriculum,
                cc_api_key=cc_api_key,
                cc_api_secret=cc_api_secret,
            )

            try:
                if framework_code:
                    # Sync specific framework
                    result = await sync_service.sync_framework(framework_code)
                elif cc_api_key and cc_api_secret:
                    # Per-tenant mode: sync only frameworks the tenant has profiles for
                    result = await sync_service.sync_tenant_frameworks()
                else:
                    # Global mode: sync all frameworks
                    result = await sync_service.sync_all()

                return {
                    "status": "success",
                    "tenant_code": tenant_code,
                    "framework_code": framework_code,
                    "success": result.success,
                    "frameworks_synced": result.frameworks_synced,
                    "stages_synced": result.stages_synced,
                    "grades_synced": result.grades_synced,
                    "subjects_synced": result.subjects_synced,
                    "units_synced": result.units_synced,
                    "topics_synced": result.topics_synced,
                    "objectives_synced": result.objectives_synced,
                    "prerequisites_synced": result.prerequisites_synced,
                    "taxonomies_synced": result.taxonomies_synced,
                    "categories_synced": result.categories_synced,
                    "mappings_synced": result.mappings_synced,
                    "duration_seconds": result.duration_seconds,
                }
            except Exception as e:
                logger.error(
                    "Curriculum sync failed for tenant %s: %s",
                    tenant_code,
                    str(e),
                    exc_info=True,
                )
                return {
                    "status": "failed",
                    "tenant_code": tenant_code,
                    "framework_code": framework_code,
                    "error": str(e),
                }
            finally:
                await sync_service.close()

    try:
        result = run_async(_sync())
        logger.info(
            "Curriculum sync completed for tenant %s: %s",
            tenant_code,
            result.get("status"),
        )
        return result
    except Exception as e:
        logger.error(
            "Curriculum sync task failed for tenant %s: %s",
            tenant_code,
            str(e),
            exc_info=True,
        )
        return {
            "status": "failed",
            "tenant_code": tenant_code,
            "error": str(e),
        }


@dramatiq.actor(
    queue_name=Queues.CURRICULUM,
    max_retries=1,
    time_limit=600000,  # 10 minutes (just schedules tasks)
    priority=Priority.LOW,
)
def sync_curriculum_all_tenants() -> dict[str, Any]:
    """Schedule curriculum sync for all active tenants.

    Dispatches sync_curriculum_for_tenant tasks for each tenant
    with a running database container.

    Returns:
        Execution statistics.
    """
    logger.info("Starting curriculum sync for all tenants")

    async def _schedule() -> dict[str, Any]:
        from src.core.config import get_settings
        from src.infrastructure.background.scheduler import get_all_tenant_codes

        settings = get_settings()

        if not settings.central_curriculum.sync_enabled:
            logger.info("Curriculum sync is disabled in settings")
            return {"status": "disabled"}

        tenant_codes = get_all_tenant_codes()

        for tenant_code in tenant_codes:
            sync_curriculum_for_tenant.send(tenant_code)
            logger.debug("Scheduled curriculum sync for tenant: %s", tenant_code)

        return {
            "status": "scheduled",
            "tenant_count": len(tenant_codes),
        }

    try:
        result = run_async(_schedule())
        logger.info(
            "Curriculum sync scheduled for %d tenants",
            result.get("tenant_count", 0),
        )
        return result
    except Exception as e:
        logger.error(
            "Failed to schedule curriculum sync: %s",
            str(e),
            exc_info=True,
        )
        return {"status": "failed", "error": str(e)}


@dramatiq.actor(
    queue_name=Queues.CURRICULUM,
    max_retries=1,
    time_limit=600000,  # 10 minutes
    priority=Priority.NORMAL,
)
def daily_curriculum_sync_job() -> dict[str, Any]:
    """Scheduler job: Run daily curriculum sync.

    This actor is called by APScheduler cron job at the configured time.
    Dispatches sync tasks for all tenants.

    Returns:
        Execution statistics.
    """
    logger.info("Daily curriculum sync job triggered")

    async def _execute() -> dict[str, Any]:
        from src.core.config import get_settings

        settings = get_settings()

        if not settings.central_curriculum.sync_enabled:
            logger.info("Curriculum sync is disabled")
            return {"status": "disabled"}

        # Dispatch to all tenants
        sync_curriculum_all_tenants.send()

        return {"status": "dispatched"}

    try:
        result = run_async(_execute())
        logger.info("Daily curriculum sync job completed: %s", result.get("status"))
        return result
    except Exception as e:
        logger.error("Daily curriculum sync job failed: %s", e, exc_info=True)
        return {"status": "failed", "error": str(e)}


def get_curriculum_actors() -> list:
    """Get all curriculum sync actors.

    Returns:
        List of curriculum sync actor functions.
    """
    return [
        sync_curriculum_for_tenant,
        sync_curriculum_all_tenants,
        daily_curriculum_sync_job,
    ]

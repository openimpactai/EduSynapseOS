# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Scheduler for periodic Dramatiq tasks.

Uses APScheduler for cron-style job scheduling integrated with Dramatiq actors.

Example:
    from src.infrastructure.background.scheduler import get_scheduler

    scheduler = get_scheduler()

    # Add cron job (runs daily at 00:05)
    scheduler.add_cron_task(
        name="Daily Analytics Summary",
        actor_name="aggregate_daily_analytics",
        cron_expression="5 0 * * *",
    )

    # Add interval job (runs every 4 hours)
    scheduler.add_interval_task(
        name="Sync Due Reviews",
        actor_name="sync_due_reviews",
        hours=4,
    )

    # Start scheduler
    await scheduler.start()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# TENANT ITERATION HELPERS
# =============================================================================


def get_all_tenant_codes() -> list[str]:
    """Get all running tenant codes for scheduled jobs.

    Uses Docker container list via TenantDatabaseManager.
    Only returns tenants with running containers.

    Returns:
        List of tenant codes with running containers.
    """
    try:
        from src.infrastructure.database.tenant_manager import get_worker_db_manager
        from src.infrastructure.docker import ContainerStatus

        tenant_db = get_worker_db_manager()
        containers = tenant_db.list_tenants()

        return [
            c.tenant_code
            for c in containers
            if c.status == ContainerStatus.RUNNING
        ]
    except Exception as e:
        logger.error("Failed to get tenant codes: %s", e)
        return []


async def get_active_students_for_tenant(tenant_code: str) -> list[str]:
    """Get active student IDs for a tenant.

    Active = has activity in last 7 days (based on AnalyticsEvent).

    Args:
        tenant_code: Tenant code.

    Returns:
        List of active student UUIDs as strings.
    """
    try:
        from sqlalchemy import and_, distinct, select

        from src.infrastructure.database.models.tenant.analytics import AnalyticsEvent
        from src.infrastructure.database.tenant_manager import get_worker_db_manager

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)

        tenant_db = get_worker_db_manager()
        async with tenant_db.get_session(tenant_code) as session:
            result = await session.execute(
                select(distinct(AnalyticsEvent.student_id)).where(
                    and_(
                        AnalyticsEvent.student_id.isnot(None),
                        AnalyticsEvent.occurred_at >= cutoff,
                    )
                )
            )
            return [str(row[0]) for row in result.fetchall()]
    except Exception as e:
        logger.error("Failed to get active students for %s: %s", tenant_code, e)
        return []


async def get_recently_active_students(tenant_code: str) -> list[str]:
    """Get students with activity in the last 4 hours.

    Based on PracticeSession updates.

    Args:
        tenant_code: Tenant code.

    Returns:
        List of student UUIDs.
    """
    try:
        from sqlalchemy import distinct, select

        from src.infrastructure.database.models.tenant.practice import PracticeSession
        from src.infrastructure.database.tenant_manager import get_worker_db_manager

        cutoff = datetime.now(timezone.utc) - timedelta(hours=4)

        tenant_db = get_worker_db_manager()
        async with tenant_db.get_session(tenant_code) as session:
            result = await session.execute(
                select(distinct(PracticeSession.student_id)).where(
                    PracticeSession.updated_at >= cutoff
                )
            )
            return [str(row[0]) for row in result.fetchall()]
    except Exception as e:
        logger.error("Failed to get recent students for %s: %s", tenant_code, e)
        return []


# =============================================================================
# DIAGNOSTIC JOB EXECUTORS
# =============================================================================


async def execute_daily_diagnostic_scans() -> dict[str, Any]:
    """Execute daily diagnostic scans for all active students.

    Runs for each tenant, sending individual scan tasks to queue.
    Uses run_diagnostic_scan actor with scan_type="targeted".

    Returns:
        Execution statistics.
    """
    from src.infrastructure.background.tasks import run_diagnostic_scan

    logger.info("Starting daily diagnostic scans")

    total_scans = 0
    tenant_codes = get_all_tenant_codes()

    for tenant_code in tenant_codes:
        try:
            student_ids = await get_active_students_for_tenant(tenant_code)

            for student_id in student_ids:
                run_diagnostic_scan.send(
                    tenant_code,
                    student_id,
                    scan_type="targeted",
                    trigger_reason="scheduled_daily",
                )
                total_scans += 1

        except Exception as e:
            logger.error(
                "Failed to schedule scans for tenant %s: %s",
                tenant_code,
                e,
            )

    logger.info(
        "Scheduled %d daily diagnostic scans across %d tenants",
        total_scans,
        len(tenant_codes),
    )

    return {
        "total_scans": total_scans,
        "tenant_count": len(tenant_codes),
    }


async def execute_weekly_batch_scans() -> dict[str, Any]:
    """Execute weekly comprehensive batch scans.

    Runs full scans for each tenant using batch actor.

    Returns:
        Execution statistics.
    """
    from src.infrastructure.background.tasks import run_batch_diagnostic_scans

    logger.info("Starting weekly batch diagnostic scans")

    tenant_codes = get_all_tenant_codes()

    for tenant_code in tenant_codes:
        try:
            run_batch_diagnostic_scans.send(
                tenant_code,
                student_ids=None,
                class_id=None,
                scan_type="full",
            )
            logger.debug("Scheduled batch scan for tenant: %s", tenant_code)

        except Exception as e:
            logger.error(
                "Failed to schedule batch scan for tenant %s: %s",
                tenant_code,
                e,
            )

    logger.info("Scheduled weekly batch scans for %d tenants", len(tenant_codes))

    return {
        "tenant_count": len(tenant_codes),
    }


async def execute_threshold_checks() -> dict[str, Any]:
    """Execute periodic threshold checks for recently active students.

    Checks students with recent session activity for threshold breaches.

    Returns:
        Execution statistics.
    """
    from src.infrastructure.background.tasks import check_diagnostic_thresholds

    logger.info("Starting periodic threshold checks")

    total_checks = 0
    tenant_codes = get_all_tenant_codes()

    for tenant_code in tenant_codes:
        try:
            student_ids = await get_recently_active_students(tenant_code)

            for student_id in student_ids:
                check_diagnostic_thresholds.send(
                    tenant_code,
                    student_id,
                )
                total_checks += 1

        except Exception as e:
            logger.error(
                "Failed to schedule threshold checks for tenant %s: %s",
                tenant_code,
                e,
            )

    logger.info(
        "Scheduled %d threshold checks across %d tenants",
        total_checks,
        len(tenant_codes),
    )

    return {
        "total_checks": total_checks,
        "tenant_count": len(tenant_codes),
    }

# APScheduler imports (optional but recommended)
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    AsyncIOScheduler = None  # type: ignore
    CronTrigger = None  # type: ignore
    IntervalTrigger = None  # type: ignore


@dataclass
class ScheduledTask:
    """Configuration for a scheduled Dramatiq task.

    Attributes:
        id: Unique task identifier.
        name: Human-readable task name.
        actor_name: Name of the Dramatiq actor to call.
        args: Positional arguments for the actor.
        kwargs: Keyword arguments for the actor.
        enabled: Whether the task is enabled.
        last_run: Last run timestamp.
        next_run: Next scheduled run.
        run_count: Total number of runs.
        error_count: Number of failed runs.
    """

    name: str
    actor_name: str
    args: tuple = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
    enabled: bool = True
    last_run: datetime | None = None
    next_run: datetime | None = None
    run_count: int = 0
    error_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "actor_name": self.actor_name,
            "enabled": self.enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count,
            "error_count": self.error_count,
        }


class DramatiqScheduler:
    """Scheduler for periodic Dramatiq task execution.

    Integrates APScheduler with Dramatiq actors for cron-style
    and interval-based job scheduling.

    Features:
    - Cron expression support
    - Interval-based scheduling
    - Job management (add, remove, enable, disable)
    - Statistics tracking

    Attributes:
        _scheduler: APScheduler instance.
        _tasks: Dictionary of scheduled tasks.
        _running: Whether scheduler is running.
    """

    def __init__(self) -> None:
        """Initialize the scheduler."""
        self._scheduler: AsyncIOScheduler | None = None
        self._tasks: dict[str, ScheduledTask] = {}
        self._running = False

        if not APSCHEDULER_AVAILABLE:
            logger.warning(
                "APScheduler not available. Install with: pip install apscheduler"
            )

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    def _get_actor(self, actor_name: str) -> Callable[..., Any] | None:
        """Get a Dramatiq actor by name.

        Args:
            actor_name: Name of the actor.

        Returns:
            Actor function or None.
        """
        # Import tasks module to get actors
        try:
            from src.infrastructure.background import tasks

            return getattr(tasks, actor_name, None)
        except ImportError:
            logger.error("Failed to import tasks module")
            return None

    def add_cron_task(
        self,
        name: str,
        actor_name: str,
        cron_expression: str,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> ScheduledTask:
        """Add a cron-scheduled task.

        Args:
            name: Task name.
            actor_name: Dramatiq actor to call.
            cron_expression: Cron expression (minute hour day month weekday).
            args: Actor arguments.
            kwargs: Actor keyword arguments.
            enabled: Whether task is enabled.

        Returns:
            Created ScheduledTask.

        Raises:
            ValueError: If APScheduler not available or invalid cron.
        """
        if not APSCHEDULER_AVAILABLE:
            raise ValueError("APScheduler not available")

        task = ScheduledTask(
            name=name,
            actor_name=actor_name,
            args=args,
            kwargs=kwargs or {},
            enabled=enabled,
        )

        self._tasks[task.id] = task

        if self._scheduler and enabled:
            parts = cron_expression.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid cron expression: {cron_expression}")

            trigger = CronTrigger(
                minute=parts[0],
                hour=parts[1],
                day=parts[2],
                month=parts[3],
                day_of_week=parts[4],
            )

            self._scheduler.add_job(
                self._execute_task,
                trigger=trigger,
                args=[task.id],
                id=task.id,
                name=name,
            )

        logger.info("Added cron task: %s (%s)", name, cron_expression)
        return task

    def add_interval_task(
        self,
        name: str,
        actor_name: str,
        seconds: int = 0,
        minutes: int = 0,
        hours: int = 0,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        enabled: bool = True,
        start_immediately: bool = False,
    ) -> ScheduledTask:
        """Add an interval-scheduled task.

        Args:
            name: Task name.
            actor_name: Dramatiq actor to call.
            seconds: Interval seconds.
            minutes: Interval minutes.
            hours: Interval hours.
            args: Actor arguments.
            kwargs: Actor keyword arguments.
            enabled: Whether task is enabled.
            start_immediately: Run immediately on start.

        Returns:
            Created ScheduledTask.

        Raises:
            ValueError: If APScheduler not available.
        """
        if not APSCHEDULER_AVAILABLE:
            raise ValueError("APScheduler not available")

        task = ScheduledTask(
            name=name,
            actor_name=actor_name,
            args=args,
            kwargs=kwargs or {},
            enabled=enabled,
        )

        self._tasks[task.id] = task

        if self._scheduler and enabled:
            trigger = IntervalTrigger(
                seconds=seconds,
                minutes=minutes,
                hours=hours,
            )

            next_run = datetime.now(timezone.utc) if start_immediately else None

            self._scheduler.add_job(
                self._execute_task,
                trigger=trigger,
                args=[task.id],
                id=task.id,
                name=name,
                next_run_time=next_run,
            )

        logger.info(
            "Added interval task: %s (every %dh %dm %ds)",
            name,
            hours,
            minutes,
            seconds,
        )
        return task

    async def _execute_task(self, task_id: str) -> None:
        """Execute a scheduled task.

        Args:
            task_id: ID of the task to execute.
        """
        task = self._tasks.get(task_id)
        if not task or not task.enabled:
            return

        logger.debug("Executing scheduled task: %s", task.name)

        try:
            actor = self._get_actor(task.actor_name)
            if actor is None:
                raise ValueError(f"Actor not found: {task.actor_name}")

            # Send the task to Dramatiq
            actor.send(*task.args, **task.kwargs)

            task.last_run = datetime.now(timezone.utc)
            task.run_count += 1

            logger.debug("Scheduled task %s sent to queue", task.name)

        except Exception as e:
            task.error_count += 1
            logger.error("Scheduled task %s failed: %s", task.name, str(e))

    def remove_task(self, task_id: str) -> bool:
        """Remove a scheduled task.

        Args:
            task_id: Task ID to remove.

        Returns:
            True if removed.
        """
        if task_id not in self._tasks:
            return False

        if self._scheduler:
            try:
                self._scheduler.remove_job(task_id)
            except Exception:
                pass

        del self._tasks[task_id]
        logger.info("Removed scheduled task: %s", task_id)
        return True

    def enable_task(self, task_id: str) -> bool:
        """Enable a scheduled task.

        Args:
            task_id: Task ID to enable.

        Returns:
            True if enabled.
        """
        task = self._tasks.get(task_id)
        if not task:
            return False

        task.enabled = True
        if self._scheduler:
            try:
                self._scheduler.resume_job(task_id)
            except Exception:
                pass

        return True

    def disable_task(self, task_id: str) -> bool:
        """Disable a scheduled task.

        Args:
            task_id: Task ID to disable.

        Returns:
            True if disabled.
        """
        task = self._tasks.get(task_id)
        if not task:
            return False

        task.enabled = False
        if self._scheduler:
            try:
                self._scheduler.pause_job(task_id)
            except Exception:
                pass

        return True

    def get_task(self, task_id: str) -> ScheduledTask | None:
        """Get a scheduled task by ID.

        Args:
            task_id: Task ID.

        Returns:
            ScheduledTask or None.
        """
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[ScheduledTask]:
        """List all scheduled tasks.

        Returns:
            List of scheduled tasks.
        """
        return list(self._tasks.values())

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        if not APSCHEDULER_AVAILABLE:
            logger.warning("Cannot start scheduler: APScheduler not available")
            return

        self._scheduler = AsyncIOScheduler()
        self._scheduler.start()
        self._running = True

        logger.info("Dramatiq scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return

        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None

        self._running = False
        logger.info("Dramatiq scheduler stopped")

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Statistics dictionary.
        """
        return {
            "is_running": self._running,
            "task_count": len(self._tasks),
            "enabled_count": sum(1 for t in self._tasks.values() if t.enabled),
            "total_runs": sum(t.run_count for t in self._tasks.values()),
            "total_errors": sum(t.error_count for t in self._tasks.values()),
            "tasks": [t.to_dict() for t in self._tasks.values()],
        }


# Singleton instance
_scheduler: DramatiqScheduler | None = None


def get_scheduler() -> DramatiqScheduler:
    """Get the singleton scheduler instance.

    Returns:
        DramatiqScheduler instance.
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = DramatiqScheduler()
    return _scheduler


async def start_scheduler() -> DramatiqScheduler:
    """Start the scheduler and register default jobs.

    Returns:
        Started scheduler instance.
    """
    scheduler = get_scheduler()
    await scheduler.start()

    # Register default periodic jobs
    if APSCHEDULER_AVAILABLE and scheduler.is_running:
        # Daily analytics summary at 00:05
        scheduler.add_cron_task(
            name="Daily Analytics Summary",
            actor_name="aggregate_daily_analytics",
            cron_expression="5 0 * * *",
        )

        # Mastery snapshot at 01:00
        scheduler.add_cron_task(
            name="Daily Mastery Snapshot",
            actor_name="create_mastery_snapshots",
            cron_expression="0 1 * * *",
        )

        # Sync due reviews every 4 hours
        scheduler.add_interval_task(
            name="Sync Due Reviews",
            actor_name="sync_due_reviews",
            hours=4,
        )

        # Memory cleanup weekly on Sunday at 03:00
        scheduler.add_cron_task(
            name="Weekly Memory Cleanup",
            actor_name="cleanup_old_memories",
            cron_expression="0 3 * * 0",
        )

        # =================================================================
        # DIAGNOSTIC SCHEDULED JOBS (Phase 4)
        # =================================================================

        # Daily diagnostic scans at 02:00
        scheduler.add_cron_task(
            name="Daily Diagnostic Scans",
            actor_name="daily_diagnostic_scan_job",
            cron_expression="0 2 * * *",
        )

        # Weekly batch diagnostic scans on Sunday at 04:00
        scheduler.add_cron_task(
            name="Weekly Batch Diagnostic Scans",
            actor_name="weekly_batch_scan_job",
            cron_expression="0 4 * * 0",
        )

        # Threshold checks every 4 hours
        scheduler.add_interval_task(
            name="Diagnostic Threshold Checks",
            actor_name="threshold_check_job",
            hours=4,
        )

        # =================================================================
        # CURRICULUM SYNC SCHEDULED JOBS
        # =================================================================

        # Daily curriculum sync at 03:30 (after diagnostic scans)
        scheduler.add_cron_task(
            name="Daily Curriculum Sync",
            actor_name="daily_curriculum_sync_job",
            cron_expression="30 3 * * *",
        )

        logger.info("Registered %d default scheduled tasks", len(scheduler.list_tasks()))

    return scheduler


async def stop_scheduler() -> None:
    """Stop the scheduler."""
    global _scheduler
    if _scheduler is not None:
        await _scheduler.stop()
        _scheduler = None

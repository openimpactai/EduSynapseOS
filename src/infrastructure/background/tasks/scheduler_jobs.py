# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Scheduler job wrapper actors for EduSynapseOS.

These actors wrap async scheduler functions to be called by APScheduler.
Each actor dispatches the actual work to other diagnostic actors.

Actors:
    - daily_diagnostic_scan_job: Runs daily at 02:00, schedules targeted scans
    - weekly_batch_scan_job: Runs Sunday at 04:00, schedules full batch scans
    - threshold_check_job: Runs every 4 hours, checks thresholds for active students
"""

import logging
from typing import Any

import dramatiq

from src.infrastructure.background.broker import Priority, Queues, setup_dramatiq
from src.infrastructure.background.tasks.base import run_async

setup_dramatiq()

logger = logging.getLogger(__name__)


@dramatiq.actor(
    queue_name=Queues.DIAGNOSTICS,
    max_retries=1,
    time_limit=600000,  # 10 minutes
    priority=Priority.NORMAL,
)
def daily_diagnostic_scan_job() -> dict[str, Any]:
    """Scheduler job: Run daily diagnostic scans.

    This actor is called by APScheduler cron job at 02:00 daily.
    Iterates over all tenants and dispatches targeted diagnostic scans
    for students who were active in the last 7 days.

    Returns:
        Execution statistics with scan counts.
    """
    logger.info("Daily diagnostic scan job triggered")

    async def _execute() -> dict[str, Any]:
        from src.infrastructure.background.scheduler import execute_daily_diagnostic_scans

        return await execute_daily_diagnostic_scans()

    try:
        result = run_async(_execute())
        logger.info(
            "Daily diagnostic scan job completed: %d scans scheduled",
            result.get("total_scans", 0),
        )
        return result
    except Exception as e:
        logger.error("Daily diagnostic scan job failed: %s", e, exc_info=True)
        return {"status": "failed", "error": str(e)}


@dramatiq.actor(
    queue_name=Queues.DIAGNOSTICS,
    max_retries=1,
    time_limit=600000,  # 10 minutes
    priority=Priority.LOW,
)
def weekly_batch_scan_job() -> dict[str, Any]:
    """Scheduler job: Run weekly batch scans.

    This actor is called by APScheduler cron job on Sundays at 04:00.
    Dispatches full batch diagnostic scans for all tenants.

    Returns:
        Execution statistics.
    """
    logger.info("Weekly batch scan job triggered")

    async def _execute() -> dict[str, Any]:
        from src.infrastructure.background.scheduler import execute_weekly_batch_scans

        return await execute_weekly_batch_scans()

    try:
        result = run_async(_execute())
        logger.info(
            "Weekly batch scan job completed: %d tenants scheduled",
            result.get("tenant_count", 0),
        )
        return result
    except Exception as e:
        logger.error("Weekly batch scan job failed: %s", e, exc_info=True)
        return {"status": "failed", "error": str(e)}


@dramatiq.actor(
    queue_name=Queues.DIAGNOSTICS,
    max_retries=1,
    time_limit=300000,  # 5 minutes
    priority=Priority.DIAGNOSTIC,
)
def threshold_check_job() -> dict[str, Any]:
    """Scheduler job: Run periodic threshold checks.

    This actor is called by APScheduler interval job every 4 hours.
    Checks diagnostic thresholds for students who had activity
    in the last 4 hours.

    Returns:
        Execution statistics with check counts.
    """
    logger.info("Threshold check job triggered")

    async def _execute() -> dict[str, Any]:
        from src.infrastructure.background.scheduler import execute_threshold_checks

        return await execute_threshold_checks()

    try:
        result = run_async(_execute())
        logger.info(
            "Threshold check job completed: %d checks scheduled",
            result.get("total_checks", 0),
        )
        return result
    except Exception as e:
        logger.error("Threshold check job failed: %s", e, exc_info=True)
        return {"status": "failed", "error": str(e)}


def get_scheduler_job_actors() -> list:
    """Get all scheduler job actors.

    Returns:
        List of scheduler job actor functions.
    """
    return [
        daily_diagnostic_scan_job,
        weekly_batch_scan_job,
        threshold_check_job,
    ]

# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Review (spaced repetition) background tasks for EduSynapseOS.

Tasks for managing FSRS-based spaced repetition review scheduling.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import dramatiq

from src.infrastructure.background.broker import Priority, Queues, setup_dramatiq
from src.infrastructure.background.tasks.base import run_async

# Setup broker before defining actors
setup_dramatiq()

logger = logging.getLogger(__name__)


@dramatiq.actor(
    queue_name=Queues.REVIEW,
    max_retries=1,
    time_limit=300000,  # 5 minutes
    priority=Priority.NORMAL,
)
def sync_due_reviews(
    tenant_code: str,
    hours_ahead: int = 4,
) -> dict[str, Any]:
    """Sync review items that are due or will be due soon.

    Finds all review items that are due within the specified hours.
    Can be used for sending notifications or pre-fetching data.

    Args:
        tenant_code: Tenant code.
        hours_ahead: Hours ahead to look for due items.

    Returns:
        Sync result with counts.
    """

    async def _sync() -> dict[str, Any]:
        from sqlalchemy import and_, select

        from src.infrastructure.database.models.tenant.review import ReviewItem
        from src.infrastructure.database.tenant_manager import get_worker_db_manager

        due_before = datetime.now(timezone.utc) + timedelta(hours=hours_ahead)

        try:
            tenant_db = get_worker_db_manager()
            async with tenant_db.get_session(tenant_code) as session:
                # Find due review items
                stmt = (
                    select(ReviewItem)
                    .where(
                        and_(
                            ReviewItem.due_date <= due_before,
                            ReviewItem.state != "suspended",
                        )
                    )
                )

                result = await session.execute(stmt)
                due_items = result.scalars().all()

                # Group by student for potential notification
                students_with_due: dict[str, int] = {}
                for item in due_items:
                    student_key = str(item.student_id)
                    students_with_due[student_key] = (
                        students_with_due.get(student_key, 0) + 1
                    )

                logger.info(
                    "Due reviews synced for %s: %d items due for %d students",
                    tenant_code,
                    len(due_items),
                    len(students_with_due),
                )

                return {
                    "tenant_code": tenant_code,
                    "due_items": len(due_items),
                    "students_affected": len(students_with_due),
                    "due_before": due_before.isoformat(),
                    "students_summary": students_with_due,
                }

        except Exception as e:
            logger.error("Failed to sync due reviews: %s", str(e), exc_info=True)
            return {
                "tenant_code": tenant_code,
                "error": str(e),
            }

    return run_async(_sync())


def get_review_actors() -> list:
    """Get all review actors.

    Returns:
        List of review actor functions.
    """
    return [
        sync_due_reviews,
    ]

# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Background task actors for EduSynapseOS.

This module provides all Dramatiq actors organized by domain:
- Analytics: Event processing, daily aggregation
- Memory: Learning events, cleanup, mastery snapshots
- Review: Spaced repetition syncing
- Diagnostics: Learning difficulty detection
- Curriculum: Curriculum sync from Central Curriculum
- Learning Analysis: Session analysis, pattern detection, report generation
- Scheduler Jobs: Periodic scheduled triggers

Usage:
    from src.infrastructure.background.tasks import (
        process_analytics_event,
        aggregate_daily_analytics,
        get_all_actors,
    )

    # Send a task
    process_analytics_event.send("engagement", "student-id", {...})

    # Get all actors for worker registration
    actors = get_all_actors()

Running Workers:
    dramatiq src.infrastructure.background.tasks --processes 2 --threads 4
"""

# Import all actors from submodules
from src.infrastructure.background.tasks.analytics import (
    aggregate_daily_analytics,
    get_analytics_actors,
    process_analytics_event,
)
from src.infrastructure.background.tasks.memory import (
    cleanup_old_memories,
    create_mastery_snapshots,
    get_memory_actors,
    record_learning_event,
)
from src.infrastructure.background.tasks.review import (
    get_review_actors,
    sync_due_reviews,
)
from src.infrastructure.background.tasks.diagnostics import (
    assess_risk_score,
    check_diagnostic_thresholds,
    generate_diagnostic_report,
    get_diagnostic_actors,
    run_batch_diagnostic_scans,
    run_diagnostic_scan,
)
from src.infrastructure.background.tasks.curriculum_sync import (
    daily_curriculum_sync_job,
    get_curriculum_actors,
    sync_curriculum_all_tenants,
    sync_curriculum_for_tenant,
)
from src.infrastructure.background.tasks.scheduler_jobs import (
    daily_diagnostic_scan_job,
    get_scheduler_job_actors,
    threshold_check_job,
    weekly_batch_scan_job,
)
from src.infrastructure.background.tasks.learning_analysis import (
    analyze_learning_session,
    detect_learning_patterns,
    generate_student_report,
    get_learning_analysis_actors,
)

# Re-export run_async for convenience
from src.infrastructure.background.tasks.base import run_async

__all__ = [
    # Analytics
    "process_analytics_event",
    "aggregate_daily_analytics",
    # Memory
    "record_learning_event",
    "cleanup_old_memories",
    "create_mastery_snapshots",
    # Review
    "sync_due_reviews",
    # Diagnostics
    "run_diagnostic_scan",
    "run_batch_diagnostic_scans",
    "assess_risk_score",
    "generate_diagnostic_report",
    "check_diagnostic_thresholds",
    # Curriculum
    "sync_curriculum_for_tenant",
    "sync_curriculum_all_tenants",
    "daily_curriculum_sync_job",
    # Learning Analysis
    "analyze_learning_session",
    "detect_learning_patterns",
    "generate_student_report",
    # Scheduler Jobs
    "daily_diagnostic_scan_job",
    "weekly_batch_scan_job",
    "threshold_check_job",
    # Utilities
    "run_async",
    "get_all_actors",
]


def get_all_actors() -> list:
    """Get list of all defined actors.

    Returns:
        List of all Dramatiq actors from all domains.
    """
    actors = []
    actors.extend(get_analytics_actors())
    actors.extend(get_memory_actors())
    actors.extend(get_review_actors())
    actors.extend(get_diagnostic_actors())
    actors.extend(get_curriculum_actors())
    actors.extend(get_learning_analysis_actors())
    actors.extend(get_scheduler_job_actors())
    return actors

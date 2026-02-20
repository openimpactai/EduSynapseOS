# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Analytics domain services.

This module provides services for learning analytics:
- Event tracking and storage
- Data aggregation (daily, weekly, mastery snapshots)
- Dashboard and report generation

The analytics system processes events from practice sessions
and conversations to provide insights on student learning
progress and engagement.

Integration with Background Tasks:
- Events are processed asynchronously via process_analytics_event task
- Daily aggregation runs via aggregate_daily_analytics task
- Mastery snapshots are created via create_mastery_snapshots task

Usage:
    # Event tracking
    from src.domains.analytics import EventTracker, track_event

    tracker = EventTracker(tenant_code="acme")
    await tracker.track_event("student.answer.evaluated", student_id, data)

    # Or use convenience function
    await track_event("acme", "student.answer.evaluated", student_id, data)

    # Dashboard and reports
    from src.domains.analytics import AnalyticsService

    service = AnalyticsService(db, "acme")
    dashboard = await service.get_dashboard(student_id)

    # Aggregation
    from src.domains.analytics import AnalyticsAggregator

    aggregator = AnalyticsAggregator("acme")
    await aggregator.aggregate_daily_async()
"""

from src.domains.analytics.aggregator import (
    AggregationResult,
    AnalyticsAggregator,
    WeeklyAggregation,
)
from src.domains.analytics.events import EventTracker, track_event
from src.domains.analytics.service import (
    AnalyticsService,
    ClassAnalytics,
    DashboardStats,
    ProgressReport,
    StudentDashboard,
    StudentSummary,
    TopicProgress,
    WeakArea,
)

__all__ = [
    # Event tracking
    "EventTracker",
    "track_event",
    # Aggregation
    "AnalyticsAggregator",
    "AggregationResult",
    "WeeklyAggregation",
    # Service
    "AnalyticsService",
    "StudentDashboard",
    "DashboardStats",
    "TopicProgress",
    "WeakArea",
    "ProgressReport",
    "ClassAnalytics",
    "StudentSummary",
]

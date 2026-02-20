# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Learning analysis background tasks.

This module provides background tasks for analyzing learning sessions,
detecting learning patterns, and generating progress reports.

Tasks:
    - analyze_learning_session: Analyze a completed learning session for insights
    - detect_learning_patterns: Weekly pattern detection for all students
    - generate_student_report: Generate progress reports for students

These tasks are triggered:
    - analyze_learning_session: After each learning session completion
    - detect_learning_patterns: Weekly via scheduler
    - generate_student_report: Weekly via scheduler or on-demand
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import dramatiq

from src.infrastructure.background.broker import Priority, Queues, setup_dramatiq
from src.infrastructure.background.middleware import get_current_tenant
from src.infrastructure.background.tasks.base import run_async

# Setup broker before defining actors
setup_dramatiq()

logger = logging.getLogger(__name__)


@dramatiq.actor(
    queue_name=Queues.LEARNING_ANALYSIS,
    max_retries=2,
    time_limit=120000,  # 2 minutes
    priority=Priority.ANALYSIS,
)
def analyze_learning_session(
    tenant_code: str,
    student_id: str,
    session_id: str,
    topic_full_code: str,
    session_data: dict[str, Any],
) -> dict[str, Any]:
    """Analyze a completed learning session and extract insights.

    This task runs after a learning session completes to:
    1. Generate a summary of the session
    2. Extract key learnings and breakthroughs
    3. Identify effective teaching strategies
    4. Update procedural memory with patterns
    5. Store insights as episodic memories

    Args:
        tenant_code: Tenant identifier.
        student_id: Student identifier.
        session_id: Learning session identifier.
        topic_full_code: Topic full code.
        session_data: Session data containing:
            - conversation_history: List of turns
            - understanding_progress: Final understanding score
            - understanding_verified: Whether AI-verified
            - mode_transitions: Number of mode changes
            - comprehension_evaluations: List of evaluation results
            - total_duration_seconds: Session duration

    Returns:
        Analysis results with insights and recommendations.
    """

    async def _analyze() -> dict[str, Any]:
        from src.core.agents import AgentFactory
        from src.core.memory.manager import MemoryManager
        from src.infrastructure.database.tenant_manager import get_worker_db_manager

        logger.info(
            "Analyzing learning session: tenant=%s, student=%s, session=%s",
            tenant_code,
            student_id,
            session_id,
        )

        try:
            # Get dependencies
            tenant_db = get_worker_db_manager()

            # Initialize memory manager
            from src.core.intelligence.embeddings import get_embedding_service
            from src.infrastructure.vectors import get_qdrant_client

            embedding_service = get_embedding_service()
            qdrant_client = await get_qdrant_client()

            memory_manager = MemoryManager(
                tenant_db_manager=tenant_db,
                embedding_service=embedding_service,
                qdrant_client=qdrant_client,
            )

            # Extract session metrics
            understanding_progress = session_data.get("understanding_progress", 0.0)
            understanding_verified = session_data.get("understanding_verified", False)
            mode_transitions = session_data.get("mode_transitions", 0)
            comprehension_evaluations = session_data.get("comprehension_evaluations", [])
            conversation_history = session_data.get("conversation_history", [])
            duration_seconds = session_data.get("total_duration_seconds", 0)

            # Generate session summary
            summary = _generate_session_summary(
                topic_full_code=topic_full_code,
                understanding_progress=understanding_progress,
                understanding_verified=understanding_verified,
                mode_transitions=mode_transitions,
                comprehension_evaluations=comprehension_evaluations,
                turn_count=len(conversation_history),
                duration_seconds=duration_seconds,
            )

            # Extract key learnings from comprehension evaluations
            key_learnings = _extract_key_learnings(comprehension_evaluations)

            # Identify effective strategies
            effective_strategies = _identify_effective_strategies(
                mode_transitions=mode_transitions,
                comprehension_evaluations=comprehension_evaluations,
                understanding_verified=understanding_verified,
            )

            # Record session insights as episodic memory
            from uuid import UUID

            await memory_manager.episodic.store(
                tenant_code=tenant_code,
                student_id=UUID(student_id),
                event_type="session_completion",
                summary=summary,
                details={
                    "session_id": session_id,
                    "understanding_progress": understanding_progress,
                    "understanding_verified": understanding_verified,
                    "key_learnings": key_learnings,
                    "effective_strategies": effective_strategies,
                    "duration_seconds": duration_seconds,
                    "mode_transitions": mode_transitions,
                    "workflow_type": "learning_tutor",
                    "reportable": True,
                },
                importance=0.7 if understanding_verified else 0.5,
                session_id=UUID(session_id),
                topic_full_code=topic_full_code,
            )

            # Update procedural memory with effective mode observations
            for strategy in effective_strategies:
                await memory_manager.record_procedural_observation(
                    tenant_code=tenant_code,
                    student_id=student_id,
                    observation={
                        "session_type": "learning_tutor",
                        "learning_mode": strategy.get("mode"),
                        "understanding_progress": understanding_progress,
                        "time_of_day": _get_time_of_day(),
                    },
                    topic_full_code=topic_full_code,
                )

            logger.info(
                "Session analysis complete: session=%s, learnings=%d, strategies=%d",
                session_id,
                len(key_learnings),
                len(effective_strategies),
            )

            return {
                "success": True,
                "session_id": session_id,
                "summary": summary,
                "key_learnings": key_learnings,
                "effective_strategies": effective_strategies,
                "understanding_verified": understanding_verified,
            }

        except Exception as e:
            logger.error(
                "Failed to analyze learning session: %s",
                str(e),
                exc_info=True,
            )
            return {
                "success": False,
                "session_id": session_id,
                "error": str(e),
            }

    return run_async(_analyze())


@dramatiq.actor(
    queue_name=Queues.LEARNING_ANALYSIS,
    max_retries=1,
    time_limit=600000,  # 10 minutes
    priority=Priority.LOW,
)
def detect_learning_patterns(
    tenant_code: str,
    student_id: str | None = None,
) -> dict[str, Any]:
    """Detect learning patterns for students.

    This task runs weekly to analyze accumulated learning data
    and detect patterns that can improve future teaching.

    Patterns detected:
    - Optimal learning times
    - Effective learning modes by subject
    - Common misconception areas
    - Learning style preferences

    Args:
        tenant_code: Tenant identifier.
        student_id: Optional specific student. If None, processes all students.

    Returns:
        Pattern detection results.
    """

    async def _detect() -> dict[str, Any]:
        from src.infrastructure.database.tenant_manager import get_worker_db_manager

        logger.info(
            "Detecting learning patterns: tenant=%s, student=%s",
            tenant_code,
            student_id or "all",
        )

        try:
            tenant_db = get_worker_db_manager()

            # Initialize memory manager
            from src.core.intelligence.embeddings import get_embedding_service
            from src.core.memory.manager import MemoryManager
            from src.infrastructure.vectors import get_qdrant_client

            embedding_service = get_embedding_service()
            qdrant_client = await get_qdrant_client()

            memory_manager = MemoryManager(
                tenant_db_manager=tenant_db,
                embedding_service=embedding_service,
                qdrant_client=qdrant_client,
            )

            students_processed = 0
            patterns_found = 0

            if student_id:
                # Process single student
                patterns = await memory_manager.procedural.get_learning_patterns(
                    tenant_code=tenant_code,
                    student_id=student_id,
                )
                if patterns:
                    patterns_found = 1
                    students_processed = 1
            else:
                # Process all students with recent activity
                # This would need a method to get all active students
                # For now, just log that we would process all students
                logger.info("Would process all active students (not implemented yet)")
                students_processed = 0
                patterns_found = 0

            logger.info(
                "Pattern detection complete: students=%d, patterns=%d",
                students_processed,
                patterns_found,
            )

            return {
                "success": True,
                "tenant_code": tenant_code,
                "students_processed": students_processed,
                "patterns_found": patterns_found,
            }

        except Exception as e:
            logger.error(
                "Failed to detect learning patterns: %s",
                str(e),
                exc_info=True,
            )
            return {
                "success": False,
                "tenant_code": tenant_code,
                "error": str(e),
            }

    return run_async(_detect())


@dramatiq.actor(
    queue_name=Queues.REPORTS,
    max_retries=2,
    time_limit=300000,  # 5 minutes
    priority=Priority.ANALYSIS,
)
def generate_student_report(
    tenant_code: str,
    student_id: str,
    report_type: str = "weekly",
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Generate a progress report for a student.

    This task generates detailed progress reports using the
    report_generator agent and memory data.

    Args:
        tenant_code: Tenant identifier.
        student_id: Student identifier.
        report_type: Type of report (weekly, monthly, parent_report).
        date_from: Start date (ISO format).
        date_to: End date (ISO format).

    Returns:
        Generated report data.
    """

    async def _generate() -> dict[str, Any]:
        from src.core.agents import AgentFactory
        from src.core.memory.manager import MemoryManager
        from src.infrastructure.database.tenant_manager import get_worker_db_manager

        logger.info(
            "Generating %s report: tenant=%s, student=%s",
            report_type,
            tenant_code,
            student_id,
        )

        try:
            # Parse dates
            if date_from:
                from_dt = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
            else:
                from_dt = datetime.now(timezone.utc) - timedelta(days=7)

            if date_to:
                to_dt = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
            else:
                to_dt = datetime.now(timezone.utc)

            # Get dependencies
            tenant_db = get_worker_db_manager()

            # Initialize memory manager
            from src.core.intelligence.embeddings import get_embedding_service
            from src.infrastructure.vectors import get_qdrant_client

            embedding_service = get_embedding_service()
            qdrant_client = await get_qdrant_client()

            memory_manager = MemoryManager(
                tenant_db_manager=tenant_db,
                embedding_service=embedding_service,
                qdrant_client=qdrant_client,
            )

            # Get report data from memory manager
            report_data = await memory_manager.get_student_progress_report_data(
                tenant_code=tenant_code,
                student_id=student_id,
                date_from=from_dt,
                date_to=to_dt,
            )

            # Generate report using agent (optional - could use AI for narrative)
            # For now, just return the structured data
            report = {
                "report_id": str(uuid4()),
                "student_id": student_id,
                "report_type": report_type,
                "period": {
                    "from": from_dt.isoformat(),
                    "to": to_dt.isoformat(),
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "data": report_data,
            }

            logger.info(
                "Report generated: type=%s, student=%s, events=%d",
                report_type,
                student_id,
                report_data.get("summary_stats", {}).get("total_events", 0),
            )

            return {
                "success": True,
                "report": report,
            }

        except Exception as e:
            logger.error(
                "Failed to generate report: %s",
                str(e),
                exc_info=True,
            )
            return {
                "success": False,
                "student_id": student_id,
                "error": str(e),
            }

    return run_async(_generate())


def _generate_session_summary(
    topic_full_code: str,
    understanding_progress: float,
    understanding_verified: bool,
    mode_transitions: int,
    comprehension_evaluations: list[dict],
    turn_count: int,
    duration_seconds: int,
) -> str:
    """Generate a human-readable session summary.

    Args:
        topic_full_code: Topic code.
        understanding_progress: Final understanding score.
        understanding_verified: Whether verified by AI.
        mode_transitions: Number of mode changes.
        comprehension_evaluations: List of evaluations.
        turn_count: Number of conversation turns.
        duration_seconds: Session duration.

    Returns:
        Summary string.
    """
    # Extract topic name from code
    topic_parts = topic_full_code.split(".")
    topic_name = topic_parts[-1] if topic_parts else "Unknown"

    duration_minutes = duration_seconds // 60

    # Determine outcome
    if understanding_verified and understanding_progress >= 0.8:
        outcome = "achieved verified understanding"
    elif understanding_progress >= 0.7:
        outcome = "showed good progress"
    elif understanding_progress >= 0.5:
        outcome = "made moderate progress"
    else:
        outcome = "engaged with the material"

    # Count verified comprehension checks
    verified_checks = len([
        e for e in comprehension_evaluations
        if e.get("verified", False)
    ])

    summary = f"Learning session on {topic_name}: Student {outcome} "
    summary += f"(progress: {understanding_progress:.0%}). "
    summary += f"Session lasted {duration_minutes} minutes with {turn_count} exchanges."

    if verified_checks > 0:
        summary += f" Understanding was verified {verified_checks} time(s)."

    if mode_transitions > 0:
        summary += f" Teaching mode adjusted {mode_transitions} time(s)."

    return summary


def _extract_key_learnings(comprehension_evaluations: list[dict]) -> list[str]:
    """Extract key learnings from comprehension evaluations.

    Args:
        comprehension_evaluations: List of evaluation results.

    Returns:
        List of key learning points.
    """
    learnings = []

    for evaluation in comprehension_evaluations:
        concepts_understood = evaluation.get("concepts_understood", [])
        if concepts_understood:
            learnings.extend(concepts_understood[:3])  # Top 3 per evaluation

    # Deduplicate
    return list(dict.fromkeys(learnings))[:10]


def _identify_effective_strategies(
    mode_transitions: int,
    comprehension_evaluations: list[dict],
    understanding_verified: bool,
) -> list[dict[str, Any]]:
    """Identify what teaching strategies were effective.

    Args:
        mode_transitions: Number of mode changes.
        comprehension_evaluations: List of evaluation results.
        understanding_verified: Whether understanding was verified.

    Returns:
        List of effective strategies.
    """
    strategies = []

    # If understanding was verified with few transitions, current approach worked
    if understanding_verified and mode_transitions <= 2:
        strategies.append({
            "type": "mode_stability",
            "description": "Minimal mode changes led to verified understanding",
            "effectiveness": 0.8,
        })

    # Check for successful comprehension checks
    successful_checks = [
        e for e in comprehension_evaluations
        if e.get("understanding_score", 0) >= 0.7
    ]

    if successful_checks:
        strategies.append({
            "type": "comprehension_check",
            "description": "Comprehension checks helped verify understanding",
            "effectiveness": 0.7,
        })

    return strategies


def _get_time_of_day() -> str:
    """Get current time of day category.

    Returns:
        Time of day: morning, afternoon, evening, or night.
    """
    hour = datetime.now(timezone.utc).hour

    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


def get_learning_analysis_actors() -> list:
    """Get all learning analysis actors.

    Returns:
        List of learning analysis actor functions.
    """
    return [
        analyze_learning_session,
        detect_learning_patterns,
        generate_student_report,
    ]

# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Diagnostic trigger handlers for event-based scans.

Listens to specific events and conditionally triggers diagnostic
scans when certain thresholds are met. Rate limiting prevents
excessive scanning for the same student.

Trigger Conditions:
    - ANSWER_EVALUATED: accuracy < 40% (3+ answers) -> Targeted scan
    - MISCONCEPTION_DETECTED: Always -> Related detector scan
    - STRUGGLING_DETECTED: Always -> Threshold check
    - SESSION_COMPLETED: accuracy < 60% -> Threshold check

These handlers run alongside the analytics handlers in the EventBus.
Each handler checks rate limits before dispatching diagnostic tasks.

Session Accuracy Tracking:
    Uses Redis for session accuracy tracking to support multiple
    API workers sharing the same state. Each session's accuracy
    is tracked in a Redis hash with automatic TTL-based cleanup.
"""

import logging
from typing import Any

from src.infrastructure.events.bus import EventData

logger = logging.getLogger(__name__)


# =============================================================================
# SESSION ACCURACY TRACKING (Redis-based)
# =============================================================================

# Redis key prefix for session accuracy tracking
ACCURACY_KEY_PREFIX = "diagnostic:session_accuracy"

# TTL for session accuracy keys (2 hours in seconds)
ACCURACY_TTL_SECONDS = 7200


async def _get_redis_client():
    """Get Redis client if available.

    Returns:
        RedisClient instance or None if not available.
    """
    try:
        from src.infrastructure.cache import get_redis
        return get_redis()
    except Exception as e:
        logger.warning("Redis not available for session accuracy tracking: %s", str(e))
        return None


def _accuracy_key(tenant_code: str, session_id: str) -> str:
    """Build Redis key for session accuracy tracking.

    Args:
        tenant_code: Tenant code.
        session_id: Session identifier.

    Returns:
        Redis key string.
    """
    return f"{ACCURACY_KEY_PREFIX}:{tenant_code}:{session_id}"


async def _track_answer_accuracy(
    tenant_code: str,
    session_id: str,
    is_correct: bool,
) -> tuple[int, int] | None:
    """Track answer accuracy in Redis.

    Uses Redis hash with HINCRBY for atomic counter updates.
    Returns current totals after the update.

    Args:
        tenant_code: Tenant code.
        session_id: Session identifier.
        is_correct: Whether the answer was correct.

    Returns:
        Tuple of (correct_count, total_count) or None if Redis unavailable.
    """
    redis = await _get_redis_client()
    if redis is None:
        return None

    try:
        key = _accuracy_key(tenant_code, session_id)
        redis_client = redis._ensure_connected()

        # Atomic increment of total
        total = await redis_client.hincrby(key, "total", 1)

        # Atomic increment of correct if answer is correct
        if is_correct:
            correct = await redis_client.hincrby(key, "correct", 1)
        else:
            # Get current correct count
            correct_str = await redis_client.hget(key, "correct")
            correct = int(correct_str) if correct_str else 0

        # Set TTL on first access (EXPIRE only if TTL not already set)
        ttl = await redis_client.ttl(key)
        if ttl == -1:  # No TTL set
            await redis_client.expire(key, ACCURACY_TTL_SECONDS)

        logger.debug(
            "Session accuracy tracked: session=%s, correct=%d, total=%d",
            session_id,
            correct,
            total,
        )

        return (correct, total)

    except Exception as e:
        logger.warning("Failed to track session accuracy in Redis: %s", str(e))
        return None


async def _get_session_accuracy(
    tenant_code: str,
    session_id: str,
) -> tuple[int, int] | None:
    """Get session accuracy from Redis.

    Args:
        tenant_code: Tenant code.
        session_id: Session identifier.

    Returns:
        Tuple of (correct_count, total_count) or None if not found/unavailable.
    """
    redis = await _get_redis_client()
    if redis is None:
        return None

    try:
        key = _accuracy_key(tenant_code, session_id)
        redis_client = redis._ensure_connected()

        correct_str = await redis_client.hget(key, "correct")
        total_str = await redis_client.hget(key, "total")

        if total_str is None:
            return None

        correct = int(correct_str) if correct_str else 0
        total = int(total_str)

        return (correct, total)

    except Exception as e:
        logger.warning("Failed to get session accuracy from Redis: %s", str(e))
        return None


async def _delete_session_accuracy(tenant_code: str, session_id: str) -> None:
    """Delete session accuracy from Redis.

    Called when session completes to clean up.

    Args:
        tenant_code: Tenant code.
        session_id: Session identifier.
    """
    redis = await _get_redis_client()
    if redis is None:
        return

    try:
        key = _accuracy_key(tenant_code, session_id)
        redis_client = redis._ensure_connected()
        await redis_client.delete(key)

        logger.debug("Session accuracy deleted: session=%s", session_id)

    except Exception as e:
        logger.warning("Failed to delete session accuracy from Redis: %s", str(e))


# =============================================================================
# EVENT HANDLERS
# =============================================================================


async def on_answer_evaluated(event: EventData) -> None:
    """Handle answer evaluation for diagnostic triggers.

    Tracks session accuracy in Redis and triggers targeted scan if:
    - At least 3 answers submitted in session
    - Accuracy drops below 40%

    Args:
        event: Answer evaluated event from EventBus.
    """
    from src.infrastructure.background.tasks import run_diagnostic_scan
    from src.infrastructure.background.utils import diagnostic_scan_limiter

    payload = event.payload
    tenant_code = event.tenant_code or payload.get("tenant_code")
    student_id = payload.get("student_id") or payload.get("user_id")
    session_id = payload.get("session_id")
    is_correct = payload.get("is_correct", False)

    if not all([tenant_code, student_id, session_id]):
        logger.debug(
            "Missing required fields for answer evaluation: tenant=%s, student=%s, session=%s",
            tenant_code,
            student_id,
            session_id,
        )
        return

    # Track accuracy in Redis (shared across all workers)
    result = await _track_answer_accuracy(tenant_code, session_id, is_correct)
    if result is None:
        logger.warning("Could not track session accuracy - Redis unavailable")
        return

    correct, total = result

    # Check trigger conditions
    if total < 3:
        return  # Not enough data yet

    accuracy = correct / total

    if accuracy < 0.40:
        # Check rate limit before dispatching
        entity_id = f"{tenant_code}:{student_id}"
        if not diagnostic_scan_limiter.allow(entity_id):
            logger.debug(
                "Diagnostic scan rate limited for student %s",
                student_id,
            )
            return

        logger.info(
            "Triggering targeted scan: student=%s, accuracy=%.1f%% (%d/%d), session=%s",
            student_id,
            accuracy * 100,
            correct,
            total,
            session_id,
        )

        run_diagnostic_scan.send(
            tenant_code,
            str(student_id),
            scan_type="targeted",
            trigger_reason="low_accuracy",
            indicator_types=None,  # All indicators
        )


async def on_misconception_detected(event: EventData) -> None:
    """Handle misconception detection for diagnostic triggers.

    Triggers a targeted scan focused on indicators related to
    the detected misconception type.

    Args:
        event: Misconception detected event from EventBus.
    """
    from src.infrastructure.background.tasks import run_diagnostic_scan
    from src.infrastructure.background.utils import diagnostic_scan_limiter

    payload = event.payload
    tenant_code = event.tenant_code or payload.get("tenant_code")
    student_id = payload.get("student_id") or payload.get("user_id")
    misconception_type = payload.get("misconception_type")
    subject = payload.get("subject")

    if not all([tenant_code, student_id]):
        return

    # Check rate limit
    entity_id = f"{tenant_code}:{student_id}"
    if not diagnostic_scan_limiter.allow(entity_id):
        logger.debug(
            "Diagnostic scan rate limited for student %s (misconception)",
            student_id,
        )
        return

    # Map misconception to relevant detector indicators
    indicator_types = _map_misconception_to_indicators(misconception_type, subject)

    logger.info(
        "Triggering targeted scan for misconception: student=%s, type=%s, indicators=%s",
        student_id,
        misconception_type,
        indicator_types,
    )

    run_diagnostic_scan.send(
        tenant_code,
        str(student_id),
        scan_type="targeted",
        trigger_reason=f"misconception_{misconception_type or 'unknown'}",
        indicator_types=indicator_types,
    )


def _map_misconception_to_indicators(
    misconception_type: str | None,
    subject: str | None,
) -> list[str] | None:
    """Map misconception type to relevant detector indicators.

    Uses pattern matching to determine which learning difficulty
    indicators might be related to the misconception.

    Args:
        misconception_type: Type of misconception detected.
        subject: Subject area (math, reading, etc.).

    Returns:
        List of indicator types to check, or None for all indicators.
    """
    # Mapping of misconception patterns to indicators
    mappings = {
        # Math-related misconceptions
        "calculation": ["dyscalculia"],
        "number_sense": ["dyscalculia"],
        "fraction": ["dyscalculia"],
        "algebra": ["dyscalculia", "attention"],
        "arithmetic": ["dyscalculia"],
        "geometry": ["dyscalculia", "visual_processing"],
        # Reading-related misconceptions
        "reading_comprehension": ["dyslexia"],
        "word_recognition": ["dyslexia"],
        "spelling": ["dyslexia"],
        "phonological": ["dyslexia", "auditory_processing"],
        "decoding": ["dyslexia"],
        # Attention-related
        "focus": ["attention"],
        "sequence": ["attention", "dyscalculia"],
        "instruction": ["attention", "auditory_processing"],
        "careless": ["attention"],
        # Processing-related
        "visual": ["visual_processing"],
        "auditory": ["auditory_processing"],
        "spatial": ["visual_processing", "dyscalculia"],
    }

    if misconception_type:
        misconception_lower = misconception_type.lower()
        for pattern, indicators in mappings.items():
            if pattern in misconception_lower:
                return indicators

    # Subject-based fallback
    subject_mappings = {
        "math": ["dyscalculia", "attention"],
        "mathematics": ["dyscalculia", "attention"],
        "reading": ["dyslexia", "attention"],
        "language": ["dyslexia", "auditory_processing"],
        "english": ["dyslexia", "auditory_processing"],
        "science": ["attention", "visual_processing"],
    }

    if subject:
        subject_lower = subject.lower()
        for subj, indicators in subject_mappings.items():
            if subj in subject_lower:
                return indicators

    return None  # Scan all indicators


async def on_struggling_detected(event: EventData) -> None:
    """Handle struggling detection for diagnostic triggers.

    Triggers a threshold check when system detects student is struggling.
    Uses more lenient rate limiting than full scans.

    Args:
        event: Struggling detected event from EventBus.
    """
    from src.infrastructure.background.tasks import check_diagnostic_thresholds
    from src.infrastructure.background.utils import threshold_check_limiter

    payload = event.payload
    tenant_code = event.tenant_code or payload.get("tenant_code")
    student_id = payload.get("student_id") or payload.get("user_id")
    session_accuracy = payload.get("accuracy")

    if not all([tenant_code, student_id]):
        return

    # Check rate limit (more lenient for threshold checks)
    entity_id = f"{tenant_code}:{student_id}"
    if not threshold_check_limiter.allow(entity_id):
        logger.debug(
            "Threshold check rate limited for student %s",
            student_id,
        )
        return

    logger.info(
        "Triggering threshold check: student=%s, accuracy=%s",
        student_id,
        session_accuracy,
    )

    check_diagnostic_thresholds.send(
        tenant_code,
        str(student_id),
        session_accuracy=session_accuracy,
    )


async def on_session_completed(event: EventData) -> None:
    """Handle session completion for diagnostic triggers.

    Runs a threshold check if session had low performance (< 60% accuracy).
    Cleans up session accuracy cache after processing.

    Args:
        event: Session completed event from EventBus.
    """
    from src.infrastructure.background.tasks import check_diagnostic_thresholds
    from src.infrastructure.background.utils import threshold_check_limiter

    payload = event.payload
    tenant_code = event.tenant_code or payload.get("tenant_code")
    student_id = payload.get("student_id") or payload.get("user_id")
    session_id = payload.get("session_id")

    if not all([tenant_code, student_id]):
        return

    # Get session accuracy from Redis
    result = await _get_session_accuracy(tenant_code, session_id) if session_id else None

    # Cleanup Redis entry
    if session_id:
        await _delete_session_accuracy(tenant_code, session_id)

    if not result or result[1] < 3:
        return  # Not enough data for meaningful check

    correct, total = result
    accuracy = correct / total

    # Only trigger check for low accuracy sessions
    if accuracy >= 0.60:
        return  # Good performance, no check needed

    # Check rate limit
    entity_id = f"{tenant_code}:{student_id}"
    if not threshold_check_limiter.allow(entity_id):
        return

    logger.info(
        "Session completed with low accuracy: student=%s, accuracy=%.1f%% (%d/%d)",
        student_id,
        accuracy * 100,
        correct,
        total,
    )

    check_diagnostic_thresholds.send(
        tenant_code,
        str(student_id),
        session_accuracy=accuracy,
    )


# =============================================================================
# NEW WORKFLOW DIAGNOSTIC TRIGGERS
# =============================================================================


async def on_learning_tutor_understanding_updated(event: EventData) -> None:
    """Handle learning tutor understanding update for diagnostic triggers.

    Triggers a targeted scan if:
    - At least 5 conversational turns in session
    - Understanding score drops below 0.3

    Args:
        event: Understanding updated event from EventBus.
    """
    from src.infrastructure.background.tasks import run_diagnostic_scan
    from src.infrastructure.background.utils import diagnostic_scan_limiter

    payload = event.payload
    tenant_code = event.tenant_code or payload.get("tenant_code")
    student_id = payload.get("student_id") or payload.get("user_id")
    understanding_score = payload.get("understanding_score", 1.0)
    turn_count = payload.get("turn_count", 0)

    if not all([tenant_code, student_id]):
        return

    # Check trigger conditions
    if turn_count < 5:
        return  # Not enough data yet

    if understanding_score >= 0.3:
        return  # Understanding is acceptable

    # Check rate limit
    entity_id = f"{tenant_code}:{student_id}"
    if not diagnostic_scan_limiter.allow(entity_id):
        logger.debug(
            "Diagnostic scan rate limited for student %s (low understanding)",
            student_id,
        )
        return

    logger.info(
        "Triggering targeted scan for low understanding: student=%s, score=%.2f, turns=%d",
        student_id,
        understanding_score,
        turn_count,
    )

    run_diagnostic_scan.send(
        tenant_code,
        str(student_id),
        scan_type="targeted",
        trigger_reason="low_understanding",
        indicator_types=None,  # All indicators
    )


async def on_gaming_mistake_detected(event: EventData) -> None:
    """Handle gaming mistake detection for diagnostic triggers.

    Triggers an attention-focused scan if:
    - 3 or more consecutive mistakes detected

    This is particularly useful for detecting attention-related issues
    (like ADHD patterns) during gaming sessions.

    Args:
        event: Mistake detected event from EventBus.
    """
    from src.infrastructure.background.tasks import run_diagnostic_scan
    from src.infrastructure.background.utils import diagnostic_scan_limiter

    payload = event.payload
    tenant_code = event.tenant_code or payload.get("tenant_code")
    student_id = payload.get("student_id") or payload.get("user_id")
    consecutive_mistakes = payload.get("consecutive_mistakes", 0)

    if not all([tenant_code, student_id]):
        return

    # Only trigger on 3+ consecutive mistakes
    if consecutive_mistakes < 3:
        return

    # Check rate limit
    entity_id = f"{tenant_code}:{student_id}"
    if not diagnostic_scan_limiter.allow(entity_id):
        logger.debug(
            "Diagnostic scan rate limited for student %s (gaming mistakes)",
            student_id,
        )
        return

    logger.info(
        "Triggering attention scan for consecutive mistakes: student=%s, mistakes=%d",
        student_id,
        consecutive_mistakes,
    )

    # Focus on attention-related indicators for gaming
    run_diagnostic_scan.send(
        tenant_code,
        str(student_id),
        scan_type="targeted",
        trigger_reason="consecutive_gaming_mistakes",
        indicator_types=["attention"],
    )


async def on_practice_helper_mode_escalated(event: EventData) -> None:
    """Handle practice helper mode escalation for diagnostic triggers.

    Triggers a threshold check if:
    - 2 or more mode escalations in the session

    Mode escalation is a strong struggle signal indicating the student
    needs progressively more help to understand a concept.

    Args:
        event: Mode escalated event from EventBus.
    """
    from src.infrastructure.background.tasks import check_diagnostic_thresholds
    from src.infrastructure.background.utils import threshold_check_limiter

    payload = event.payload
    tenant_code = event.tenant_code or payload.get("tenant_code")
    student_id = payload.get("student_id") or payload.get("user_id")
    escalation_count = payload.get("escalation_count", 0)

    if not all([tenant_code, student_id]):
        return

    # Only trigger on 2+ escalations
    if escalation_count < 2:
        return

    # Check rate limit
    entity_id = f"{tenant_code}:{student_id}"
    if not threshold_check_limiter.allow(entity_id):
        logger.debug(
            "Threshold check rate limited for student %s (mode escalation)",
            student_id,
        )
        return

    logger.info(
        "Triggering threshold check for mode escalation: student=%s, escalations=%d",
        student_id,
        escalation_count,
    )

    check_diagnostic_thresholds.send(
        tenant_code,
        str(student_id),
        escalation_count=escalation_count,
    )

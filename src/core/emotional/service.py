# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Emotional State Service - Core service for emotional intelligence.

This service manages emotional signals and state for students:
- Recording emotional signals analyzed by LLM agents
- Calculating current emotional state
- Providing EmotionalContext to consumers
- Tracking emotional history and trends

The service is the central point for all emotional intelligence operations
and is used by:
- Chat workflow (records sentiment signals via emotional_analyzer agent)
- Educational Theories (consumes EmotionalContext)
- Proactive Monitors (consumes EmotionalContext)
- Companion Agent (consumes and produces signals)

Note: All emotional analysis is performed by LLM through the
emotional_analyzer agent with MessageAnalysisCapability.
"""

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.emotional.constants import (
    STATE_TO_ACTIONS,
    EmotionalIntensity,
    EmotionalSignalSource,
    EmotionalState,
    EmotionalThresholds,
)
from src.core.emotional.context import (
    EmotionalContext,
    EmotionalSignalData,
    ParentMoodInput,
)
from src.core.emotional.signals import (
    analyze_trend,
    calculate_current_state,
)
from src.infrastructure.database.models.tenant.emotional import EmotionalSignal
from src.infrastructure.database.models.tenant.student_note import StudentNote

logger = logging.getLogger(__name__)


class EmotionalStateService:
    """Central service for emotional intelligence operations.

    Manages emotional signals from LLM-based analysis:
    1. Recording pre-analyzed signals from emotional_analyzer agent
    2. Calculating current emotional state from signal history
    3. Building EmotionalContext for consumers
    4. Tracking trends and history

    All emotional analysis is performed by the emotional_analyzer agent
    through the MessageAnalysisCapability. This service only stores
    and aggregates the results.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialize the emotional state service.

        Args:
            db: Async database session.
        """
        self._db = db

    async def record_analyzed_signal(
        self,
        student_id: UUID,
        source: str | EmotionalSignalSource,
        emotional_state: str,
        intensity: str,
        confidence: float,
        triggers: list[str] | None = None,
        activity_id: str | None = None,
        activity_type: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> EmotionalSignal:
        """Record a signal with pre-analyzed emotional data.

        Used when emotional analysis has already been performed by an agent
        (e.g., emotional_analyzer agent in TutoringWorkflow). This method
        simply records the already-analyzed data.

        Args:
            student_id: The student's ID.
            source: Signal source (chat, etc.).
            emotional_state: Pre-detected emotional state string.
            intensity: Pre-detected intensity string (low, moderate, high).
            confidence: Confidence score (0.0-1.0).
            triggers: Emotional triggers identified.
            activity_id: ID of the activity.
            activity_type: Type of activity.
            context: Additional context.

        Returns:
            The created EmotionalSignal record.
        """
        # Convert source to string if enum
        if isinstance(source, EmotionalSignalSource):
            source = source.value

        # Convert intensity string to decimal value
        intensity_map = {
            "high": 0.85,
            "moderate": 0.55,
            "low": 0.25,
        }
        intensity_value = intensity_map.get(intensity.lower(), 0.5) if isinstance(intensity, str) else intensity

        # Create signal record with pre-analyzed data
        signal = EmotionalSignal(
            id=uuid4(),
            student_id=student_id,
            source=source,
            signal_type="analyzed_message",
            raw_value={"analysis_source": "emotional_analyzer_agent"},
            detected_emotion=emotional_state if emotional_state != "neutral" else None,
            emotion_intensity=intensity_value if emotional_state != "neutral" else None,
            emotion_confidence=confidence if emotional_state != "neutral" else None,
            activity_id=activity_id,
            activity_type=activity_type,
            trigger_context={
                **(context or {}),
                "triggers": triggers or [],
            },
            processing_method="llm_agent",
        )

        self._db.add(signal)
        await self._db.flush()

        logger.debug(
            "Recorded analyzed signal: student=%s, state=%s, intensity=%s",
            student_id,
            emotional_state,
            intensity,
        )

        return signal

    async def get_current_state(
        self,
        student_id: UUID,
        window_minutes: int = EmotionalThresholds.SIGNAL_AGGREGATION_WINDOW,
    ) -> EmotionalContext:
        """Get the current emotional context for a student.

        Retrieves recent signals, calculates current state, and builds
        the complete EmotionalContext for consumers.

        Args:
            student_id: The student's ID.
            window_minutes: Time window for signal aggregation.

        Returns:
            EmotionalContext with current state and trends.
        """
        # Get recent signals from database
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)

        stmt = (
            select(EmotionalSignal)
            .where(
                EmotionalSignal.student_id == student_id,
                EmotionalSignal.created_at >= cutoff,
            )
            .order_by(EmotionalSignal.created_at.asc())
        )

        result = await self._db.execute(stmt)
        db_signals = result.scalars().all()

        # Convert to EmotionalSignalData
        signals = [
            EmotionalSignalData(
                signal_type=s.signal_type,
                source=s.source,
                value=s.raw_value.get("value") if isinstance(s.raw_value, dict) else s.raw_value,
                detected_emotion=EmotionalState(s.detected_emotion) if s.detected_emotion else None,
                confidence=s.emotion_confidence or 0.5,
                activity_id=s.activity_id,
                activity_type=s.activity_type,
                context=s.context or {},
                created_at=s.created_at,
            )
            for s in db_signals
        ]

        # Calculate current state
        current_state, intensity, confidence, triggers = calculate_current_state(
            signals, window_minutes
        )

        # Analyze trend
        trend = analyze_trend(signals, window_minutes)

        # Get parent mood if available
        parent_mood = await self._get_parent_mood(student_id)

        # Get recommended actions
        actions = STATE_TO_ACTIONS.get(current_state, [])

        # Build context
        return EmotionalContext(
            student_id=student_id,
            current_state=current_state,
            intensity=intensity,
            confidence=confidence,
            triggers=triggers,
            trend=trend,
            recent_signals=signals[-10:],  # Last 10 signals
            recent_signals_count=len(signals),
            parent_mood=parent_mood,
            recommended_actions=actions,
            updated_at=datetime.utcnow(),
        )

    async def _get_parent_mood(self, student_id: UUID) -> ParentMoodInput | None:
        """Get today's parent-reported mood for a student.

        Args:
            student_id: The student's ID.

        Returns:
            ParentMoodInput if available, None otherwise.
        """
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        stmt = (
            select(StudentNote)
            .where(
                StudentNote.student_id == student_id,
                StudentNote.note_type == "daily_mood",
                StudentNote.source_type == "parent",
                StudentNote.created_at >= today_start,
            )
            .order_by(StudentNote.created_at.desc())
            .limit(1)
        )

        result = await self._db.execute(stmt)
        note = result.scalar_one_or_none()

        if note:
            return ParentMoodInput(
                mood=note.content[:50] if note.content else "unknown",
                note=note.title,
                reported_at=note.created_at,
                valid_until=note.valid_until,
            )

        return None

    async def get_signals_for_session(
        self,
        student_id: UUID,
        session_id: str,
    ) -> list[EmotionalSignal]:
        """Get all signals for a specific session.

        Args:
            student_id: The student's ID.
            session_id: The session ID to filter by.

        Returns:
            List of EmotionalSignal records for the session.
        """
        stmt = (
            select(EmotionalSignal)
            .where(
                EmotionalSignal.student_id == student_id,
                EmotionalSignal.activity_id == session_id,
            )
            .order_by(EmotionalSignal.created_at.asc())
        )

        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def get_emotional_history(
        self,
        student_id: UUID,
        days: int = 7,
    ) -> list[dict[str, Any]]:
        """Get emotional history for a student over multiple days.

        Returns aggregated daily emotional summaries.

        Args:
            student_id: The student's ID.
            days: Number of days to look back.

        Returns:
            List of daily summaries with dominant emotion and signal counts.
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        stmt = (
            select(EmotionalSignal)
            .where(
                EmotionalSignal.student_id == student_id,
                EmotionalSignal.created_at >= cutoff,
            )
            .order_by(EmotionalSignal.created_at.asc())
        )

        result = await self._db.execute(stmt)
        db_signals = result.scalars().all()

        # Group by day
        daily_data: dict[str, list[EmotionalSignal]] = {}
        for signal in db_signals:
            day_key = signal.created_at.strftime("%Y-%m-%d")
            if day_key not in daily_data:
                daily_data[day_key] = []
            daily_data[day_key].append(signal)

        # Calculate daily summaries
        summaries = []
        for day, day_signals in sorted(daily_data.items()):
            emotion_counts: dict[str, int] = {}
            for s in day_signals:
                if s.detected_emotion:
                    emotion_counts[s.detected_emotion] = emotion_counts.get(s.detected_emotion, 0) + 1

            dominant = max(emotion_counts.keys(), key=lambda e: emotion_counts[e]) if emotion_counts else "neutral"

            summaries.append({
                "date": day,
                "dominant_emotion": dominant,
                "signal_count": len(day_signals),
                "emotion_counts": emotion_counts,
            })

        return summaries

    async def has_recent_distress(
        self,
        student_id: UUID,
        window_minutes: int = 15,
        threshold_count: int = 3,
    ) -> bool:
        """Check if student has shown recent distress signals.

        Used by ProactiveService to determine if alerts are needed.

        Args:
            student_id: The student's ID.
            window_minutes: Time window to check.
            threshold_count: Minimum negative signals for distress.

        Returns:
            True if student shows distress pattern.
        """
        context = await self.get_current_state(student_id, window_minutes)

        # Check for high frustration or anxiety
        if context.current_state in (EmotionalState.FRUSTRATED, EmotionalState.ANXIOUS):
            if context.intensity in (EmotionalIntensity.MODERATE, EmotionalIntensity.HIGH):
                # Count recent negative signals
                negative_count = sum(
                    1 for s in context.recent_signals
                    if s.detected_emotion in (
                        EmotionalState.FRUSTRATED,
                        EmotionalState.ANXIOUS,
                        EmotionalState.CONFUSED,
                    )
                )
                return negative_count >= threshold_count

        return False

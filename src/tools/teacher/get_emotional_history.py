# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get emotional history tool for teachers.

Returns the emotional signal history for a specific student,
showing emotional patterns and trends over time.
"""

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import func, select

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.tools.teacher.helpers import verify_teacher_has_student_access

logger = logging.getLogger(__name__)


class GetEmotionalHistoryTool(BaseTool):
    """Tool to get student emotional history."""

    @property
    def name(self) -> str:
        return "get_emotional_history"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_emotional_history",
                "description": "Get the emotional signal history for a student, showing emotional patterns and trends over time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "student_id": {
                            "type": "string",
                            "description": "The UUID of the student",
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of days to include in history (default: 14)",
                        },
                        "source": {
                            "type": "string",
                            "enum": ["practice", "game", "chat", "creative", "study", "parent_input"],
                            "description": "Optional: Filter by signal source",
                        },
                    },
                    "required": ["student_id"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the get_emotional_history tool.

        Args:
            params: Tool parameters (student_id, days, source).
            context: Tool context with user_id (teacher).

        Returns:
            ToolResult with emotional history.
        """
        if not context.is_teacher:
            return ToolResult(
                success=False,
                error="This tool is only available for teachers.",
            )

        if not context.session:
            return ToolResult(
                success=False,
                error="Database session not available.",
            )

        student_id_str = params.get("student_id")
        if not student_id_str:
            return ToolResult(
                success=False,
                error="student_id is required.",
            )

        try:
            student_id = UUID(student_id_str)
        except ValueError:
            return ToolResult(
                success=False,
                error="Invalid student_id format.",
            )

        days = params.get("days", 14)
        source = params.get("source")
        teacher_id = context.user_id

        try:
            # Verify teacher has access to this student
            has_access = await verify_teacher_has_student_access(
                context.session, teacher_id, student_id
            )
            if not has_access:
                return ToolResult(
                    success=False,
                    error="You don't have access to this student.",
                )

            from src.infrastructure.database.models.tenant.emotional import EmotionalSignal
            from src.infrastructure.database.models.tenant.user import User

            # Get student info
            student_query = (
                select(User.first_name, User.last_name)
                .where(User.id == str(student_id))
            )
            student_result = await context.session.execute(student_query)
            student_row = student_result.first()

            if not student_row:
                return ToolResult(
                    success=False,
                    error="Student not found.",
                )

            student_name = f"{student_row.first_name} {student_row.last_name}"

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Get recent emotional signals
            signals_query = (
                select(EmotionalSignal)
                .where(EmotionalSignal.student_id == str(student_id))
                .where(EmotionalSignal.created_at >= start_date)
                .where(EmotionalSignal.detected_emotion.isnot(None))
            )

            if source:
                signals_query = signals_query.where(EmotionalSignal.source == source)

            signals_query = signals_query.order_by(EmotionalSignal.created_at.desc()).limit(100)

            signals_result = await context.session.execute(signals_query)
            signals = signals_result.scalars().all()

            # Build signal list
            signal_list = []
            for signal in signals:
                signal_list.append({
                    "id": str(signal.id),
                    "source": signal.source,
                    "signal_type": signal.signal_type,
                    "detected_emotion": signal.detected_emotion,
                    "emotion_intensity": float(signal.emotion_intensity) if signal.emotion_intensity else None,
                    "intensity_level": signal.intensity_level,
                    "activity_type": signal.activity_type,
                    "created_at": signal.created_at.isoformat() if signal.created_at else None,
                })

            # Calculate emotion distribution
            emotion_counts: dict[str, int] = {}
            intensity_sum: dict[str, float] = {}
            for signal in signals:
                emotion = signal.detected_emotion
                if emotion:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    if signal.emotion_intensity:
                        if emotion not in intensity_sum:
                            intensity_sum[emotion] = 0
                        intensity_sum[emotion] += float(signal.emotion_intensity)

            # Calculate averages and percentages
            total_signals = len(signals)
            emotion_distribution = []
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                avg_intensity = intensity_sum.get(emotion, 0) / count if count > 0 else 0
                emotion_distribution.append({
                    "emotion": emotion,
                    "count": count,
                    "percentage": round(count / total_signals * 100, 1) if total_signals > 0 else 0,
                    "avg_intensity": round(avg_intensity, 2),
                })

            # Identify patterns
            negative_emotions = {"frustrated", "anxious", "stressed", "tired", "sad", "confused", "bored"}
            positive_emotions = {"happy", "confident", "excited", "proud", "curious", "engaged"}

            negative_count = sum(
                ec["count"] for ec in emotion_distribution
                if ec["emotion"].lower() in negative_emotions
            )
            positive_count = sum(
                ec["count"] for ec in emotion_distribution
                if ec["emotion"].lower() in positive_emotions
            )

            emotional_balance = "neutral"
            if total_signals > 0:
                negative_ratio = negative_count / total_signals
                positive_ratio = positive_count / total_signals
                if negative_ratio > 0.5:
                    emotional_balance = "concerning"
                elif positive_ratio > 0.6:
                    emotional_balance = "positive"

            # Get source breakdown
            source_stats_query = (
                select(
                    EmotionalSignal.source,
                    func.count(EmotionalSignal.id).label("count"),
                )
                .where(EmotionalSignal.student_id == str(student_id))
                .where(EmotionalSignal.created_at >= start_date)
                .where(EmotionalSignal.detected_emotion.isnot(None))
                .group_by(EmotionalSignal.source)
            )

            source_result = await context.session.execute(source_stats_query)
            source_breakdown = {row.source: row.count for row in source_result.all()}

            # Build message
            if not signals:
                message = f"No emotional data recorded for {student_name} in the last {days} days."
            else:
                message = f"{student_name} - Emotional History ({days} days):\n"
                message += f"- Total signals: {total_signals}\n"
                message += f"- Emotional balance: {emotional_balance}\n"

                if emotion_distribution:
                    top_emotion = emotion_distribution[0]
                    message += f"- Most common: {top_emotion['emotion']} ({top_emotion['percentage']}%)\n"

                if emotional_balance == "concerning":
                    message += f"- Concern: {negative_count} negative signals ({round(negative_count/total_signals*100)}%)\n"

            logger.info(
                "get_emotional_history: teacher=%s, student=%s, signals=%d, balance=%s",
                teacher_id,
                student_id,
                total_signals,
                emotional_balance,
            )

            return ToolResult(
                success=True,
                data={
                    "message": message,
                    "student_id": str(student_id),
                    "student_name": student_name,
                    "period_days": days,
                    "total_signals": total_signals,
                    "emotional_balance": emotional_balance,
                    "emotion_distribution": emotion_distribution,
                    "source_breakdown": source_breakdown,
                    "recent_signals": signal_list[:20],
                    "negative_count": negative_count,
                    "positive_count": positive_count,
                },
                passthrough_data={
                    "student_name": student_name,
                    "emotional_balance": emotional_balance,
                    "emotion_distribution": emotion_distribution[:3],
                },
            )

        except Exception as e:
            logger.exception("get_emotional_history failed")
            return ToolResult(
                success=False,
                error=f"Failed to get emotional history: {str(e)}",
            )

# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Mastery Learning theory implementation.

Mastery Learning, developed by Benjamin Bloom, is based on the principle
that all students can achieve high levels of learning given:
- Sufficient time
- Appropriate instruction
- Clear mastery criteria

Key principles:
- Set clear mastery criteria (typically 80-90% proficiency)
- Students must demonstrate mastery before advancing
- Provide corrective instruction for those who don't reach mastery
- Allow additional time and practice as needed

This implementation determines:
- Whether a student should advance to new content
- Whether to provide corrective instruction
- Appropriate practice intensity
"""

from src.core.educational.theories.base import (
    BaseTheory,
    StudentContext,
    TheoryRecommendation,
)


class MasteryTheory(BaseTheory):
    """Mastery Learning progression calculator.

    Determines whether a student has achieved mastery and
    should advance, or needs additional practice.

    Configuration parameters:
        mastery_threshold: Required mastery level to advance (default: 0.80)
        min_attempts: Minimum attempts before advancement (default: 5)
        min_accuracy: Minimum accuracy required (default: 0.75)
        streak_requirement: Required consecutive correct answers (default: 3)
        time_factor: Whether time spent affects decision (default: True)
        min_time_seconds: Minimum time on topic before advancing (default: 120)
        prerequisite_weight: Weight of prerequisite mastery (default: 0.3)
    """

    @property
    def name(self) -> str:
        """Return the theory name."""
        return "mastery"

    def calculate(self, context: StudentContext) -> TheoryRecommendation:
        """Calculate mastery status and advancement recommendation.

        Evaluates multiple criteria to determine if the student
        has demonstrated sufficient mastery to advance.

        Args:
            context: Student's current learning context

        Returns:
            Recommendation with advance_to_next and difficulty
        """
        threshold = self.get_param("mastery_threshold", 0.80)
        min_attempts = self.get_param("min_attempts", 5)
        min_accuracy = self.get_param("min_accuracy", 0.75)
        streak_req = self.get_param("streak_requirement", 3)
        time_factor = self.get_param("time_factor", True)
        min_time = self.get_param("min_time_seconds", 120)
        prereq_weight = self.get_param("prerequisite_weight", 0.3)

        criteria_met = self._evaluate_criteria(
            context, threshold, min_attempts, min_accuracy,
            streak_req, time_factor, min_time
        )
        prereq_ready = self._check_prerequisites(context, prereq_weight)
        should_advance = criteria_met["all_met"] and prereq_ready
        practice_intensity = self._calculate_practice_intensity(
            context, criteria_met, threshold
        )
        difficulty = self._calculate_difficulty(
            context, should_advance, criteria_met
        )
        confidence = self._calculate_confidence(context, criteria_met)
        rationale = self._build_rationale(
            context, should_advance, criteria_met, prereq_ready
        )

        return self._create_recommendation(
            advance_to_next=should_advance,
            difficulty=difficulty,
            confidence=confidence,
            rationale=rationale,
            extra={
                "criteria_met": criteria_met,
                "prerequisites_ready": prereq_ready,
                "practice_intensity": practice_intensity,
                "mastery_gap": max(0, threshold - self._get_mastery(context)),
                "threshold": threshold,
            },
        )

    def _get_mastery(self, context: StudentContext) -> float:
        """Get current topic mastery level.

        Args:
            context: Student context

        Returns:
            Mastery level 0-1
        """
        if context.current_topic:
            return context.current_topic.mastery_level
        return context.overall_mastery

    def _evaluate_criteria(
        self,
        context: StudentContext,
        threshold: float,
        min_attempts: int,
        min_accuracy: float,
        streak_req: int,
        time_factor: bool,
        min_time: int,
    ) -> dict[str, bool | float]:
        """Evaluate all mastery criteria.

        Args:
            context: Student context
            threshold: Mastery threshold
            min_attempts: Minimum attempts
            min_accuracy: Minimum accuracy
            streak_req: Required streak
            time_factor: Whether time matters
            min_time: Minimum time

        Returns:
            Dict of criteria status
        """
        criteria: dict[str, bool | float] = {}
        mastery = self._get_mastery(context)
        criteria["mastery_met"] = mastery >= threshold
        criteria["mastery_level"] = mastery

        if context.current_topic:
            attempts = context.current_topic.attempts_total
            criteria["attempts_met"] = attempts >= min_attempts
            criteria["attempts_count"] = attempts

            accuracy = context.current_topic.accuracy
            criteria["accuracy_met"] = accuracy is not None and accuracy >= min_accuracy
            criteria["accuracy_level"] = accuracy if accuracy else 0.0

            time_spent = context.current_topic.time_spent_seconds
            criteria["time_met"] = not time_factor or time_spent >= min_time
            criteria["time_spent"] = time_spent
        else:
            criteria["attempts_met"] = False
            criteria["attempts_count"] = 0
            criteria["accuracy_met"] = False
            criteria["accuracy_level"] = 0.0
            criteria["time_met"] = not time_factor
            criteria["time_spent"] = 0

        streak = context.recent_performance.consecutive_correct
        criteria["streak_met"] = streak >= streak_req
        criteria["streak_count"] = streak
        criteria["all_met"] = all([
            criteria["mastery_met"],
            criteria["attempts_met"],
            criteria["accuracy_met"],
            criteria["streak_met"],
            criteria["time_met"],
        ])

        return criteria

    def _check_prerequisites(
        self, context: StudentContext, prereq_weight: float
    ) -> bool:
        """Check if prerequisites are sufficiently mastered.

        Args:
            context: Student context
            prereq_weight: Weight of prerequisites

        Returns:
            Whether prerequisites are ready
        """
        if not context.related_topics:
            return True

        prereq_threshold = 0.6 * prereq_weight + 0.4
        mastered_count = sum(
            1 for topic in context.related_topics
            if topic.mastery_level >= prereq_threshold
        )
        required_ratio = 0.7
        if len(context.related_topics) > 0:
            ratio = mastered_count / len(context.related_topics)
            return ratio >= required_ratio

        return True

    def _calculate_practice_intensity(
        self,
        context: StudentContext,
        criteria: dict[str, bool | float],
        threshold: float,
    ) -> str:
        """Calculate recommended practice intensity.

        Args:
            context: Student context
            criteria: Evaluated criteria
            threshold: Mastery threshold

        Returns:
            Practice intensity: 'low', 'medium', 'high', 'intensive'
        """
        mastery = float(criteria.get("mastery_level", 0.0))
        gap = threshold - mastery

        if criteria.get("all_met", False):
            return "low"

        if gap > 0.4:
            return "intensive"

        if gap > 0.2:
            return "high"

        if gap > 0.1:
            return "medium"

        return "low"

    def _calculate_difficulty(
        self,
        context: StudentContext,
        should_advance: bool,
        criteria: dict[str, bool | float],
    ) -> float:
        """Calculate appropriate difficulty.

        Args:
            context: Student context
            should_advance: Whether advancing
            criteria: Evaluated criteria

        Returns:
            Recommended difficulty 0-1
        """
        mastery = float(criteria.get("mastery_level", 0.0))

        if should_advance:
            return min(0.7, mastery + 0.15)

        if criteria.get("mastery_met", False):
            return mastery + 0.05

        if context.recent_performance.consecutive_incorrect >= 3:
            return max(0.2, mastery - 0.1)

        return mastery + 0.05

    def _calculate_confidence(
        self, context: StudentContext, criteria: dict[str, bool | float]
    ) -> float:
        """Calculate confidence in the recommendation.

        Args:
            context: Student context
            criteria: Evaluated criteria

        Returns:
            Confidence score 0-1
        """
        confidence = 0.5
        attempts = int(criteria.get("attempts_count", 0))
        if attempts >= 20:
            confidence += 0.3
        elif attempts >= 10:
            confidence += 0.2
        elif attempts >= 5:
            confidence += 0.1

        if criteria.get("streak_met", False):
            confidence += 0.1

        accuracy = float(criteria.get("accuracy_level", 0.0))
        if accuracy > 0.9:
            confidence += 0.1

        return min(1.0, confidence)

    def _build_rationale(
        self,
        context: StudentContext,
        should_advance: bool,
        criteria: dict[str, bool | float],
        prereq_ready: bool,
    ) -> str:
        """Build explanation for the recommendation.

        Args:
            context: Student context
            should_advance: Whether advancing
            criteria: Evaluated criteria
            prereq_ready: Prerequisites status

        Returns:
            Human-readable rationale
        """
        mastery = float(criteria.get("mastery_level", 0.0))
        parts = []

        if should_advance:
            parts.append(
                f"Ready to advance (mastery: {mastery:.2f}, "
                f"all criteria met)"
            )
        else:
            unmet = []
            if not criteria.get("mastery_met", False):
                unmet.append(f"mastery {mastery:.2f} < threshold")
            if not criteria.get("attempts_met", False):
                unmet.append(
                    f"attempts {criteria.get('attempts_count', 0)} < minimum"
                )
            if not criteria.get("accuracy_met", False):
                acc = float(criteria.get("accuracy_level", 0.0))
                unmet.append(f"accuracy {acc:.2f} < minimum")
            if not criteria.get("streak_met", False):
                unmet.append(
                    f"streak {criteria.get('streak_count', 0)} < required"
                )
            if not prereq_ready:
                unmet.append("prerequisites not mastered")

            parts.append(f"Not ready: {', '.join(unmet)}")

        return " | ".join(parts)

    @classmethod
    def get_corrective_strategies(cls, mastery_gap: float) -> list[str]:
        """Get corrective instruction strategies based on gap.

        Args:
            mastery_gap: Gap between current and target mastery

        Returns:
            List of corrective strategies
        """
        if mastery_gap > 0.4:
            return [
                "Review foundational concepts",
                "Provide extensive worked examples",
                "Use alternative explanations",
                "Break content into smaller units",
                "Consider prerequisite review",
            ]

        if mastery_gap > 0.2:
            return [
                "Provide additional practice problems",
                "Offer targeted feedback",
                "Use visual aids and diagrams",
                "Review common misconceptions",
            ]

        if mastery_gap > 0.1:
            return [
                "Focused practice on weak areas",
                "Peer discussion or tutoring",
                "Quick review activities",
            ]

        return [
            "Light review to reinforce learning",
            "Challenge problems for enrichment",
        ]

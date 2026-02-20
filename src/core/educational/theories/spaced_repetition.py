# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Spaced Repetition (FSRS) theory implementation.

Spaced repetition is a learning technique that involves reviewing
material at increasing intervals to optimize long-term retention.
This implementation uses the FSRS (Free Spaced Repetition Scheduler)
algorithm via the fsrs library.

FSRS-6 features:
- Optimal interval calculation based on memory model
- Difficulty and stability parameters per item
- Review rating: Again, Hard, Good, Easy
- Adaptive scheduling based on performance

This implementation:
- Calculates next review times for items
- Determines review priority based on due status
- Adjusts difficulty based on student responses
- Integrates with the semantic memory layer (FSRS parameters)
"""

from datetime import datetime, timedelta, timezone
from enum import IntEnum

from fsrs import Card, Rating, Scheduler, State

from src.core.educational.theories.base import (
    BaseTheory,
    StudentContext,
    TheoryRecommendation,
)


class ReviewRating(IntEnum):
    """Review quality ratings mapped to FSRS Rating."""

    AGAIN = 1
    HARD = 2
    GOOD = 3
    EASY = 4


class SpacedRepetitionTheory(BaseTheory):
    """Spaced Repetition (FSRS) scheduler and calculator.

    Manages review scheduling using the FSRS algorithm for
    optimal long-term retention.

    Configuration parameters:
        enable_fuzz: Add randomness to intervals (default: True)
        request_retention: Target retention rate (default: 0.9)
        maximum_interval: Max days between reviews (default: 365)
        w: FSRS weights (default: FSRS defaults)
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize with FSRS scheduler."""
        super().__init__(*args, **kwargs)
        self._scheduler = self._create_scheduler()

    @property
    def name(self) -> str:
        """Return the theory name."""
        return "spaced_repetition"

    def _create_scheduler(self) -> Scheduler:
        """Create FSRS Scheduler instance with configuration.

        Returns:
            Configured FSRS Scheduler
        """
        enable_fuzz = self.get_param("enable_fuzz", True)
        desired_retention = self.get_param("request_retention", 0.9)
        maximum_interval = self.get_param("maximum_interval", 365)
        custom_weights = self.get_param("w")

        kwargs = {
            "enable_fuzzing": enable_fuzz,
            "desired_retention": desired_retention,
            "maximum_interval": maximum_interval,
        }

        if custom_weights and len(custom_weights) == 21:
            kwargs["parameters"] = custom_weights

        return Scheduler(**kwargs)

    def calculate(self, context: StudentContext) -> TheoryRecommendation:
        """Calculate spaced repetition recommendation.

        Determines review priority and next review timing based
        on due items and current performance.

        Args:
            context: Student's current learning context

        Returns:
            Recommendation with next_review_at and priority info
        """
        due_count = context.fsrs_due_items_count
        review_needed = due_count > 0
        next_review_at = None
        priority = self._calculate_priority(context)

        if context.current_topic:
            next_review_at = self._estimate_next_review(context)

        difficulty = self._calculate_difficulty_recommendation(context)
        confidence = self._calculate_confidence(context)
        rationale = self._build_rationale(
            context, due_count, review_needed, priority
        )

        return self._create_recommendation(
            next_review_at=next_review_at,
            difficulty=difficulty,
            confidence=confidence,
            rationale=rationale,
            extra={
                "due_items_count": due_count,
                "review_needed": review_needed,
                "priority": priority,
                "estimated_review_time_minutes": self._estimate_review_time(due_count),
            },
        )

    def schedule_review(
        self,
        stability: float | None,
        difficulty: float | None,
        state: str,
        step: int,
        last_review: datetime | None,
        rating: ReviewRating,
    ) -> dict:
        """Schedule next review using FSRS algorithm.

        Takes current card parameters and rating to calculate
        the next review state.

        Args:
            stability: Current stability parameter (or None for new)
            difficulty: Current difficulty parameter (or None for new)
            state: Current state ('new', 'learning', 'review', 'relearning')
            step: Learning step count
            last_review: Last review datetime
            rating: Review quality rating

        Returns:
            Dict with updated card parameters and next review time
        """
        card = Card()

        if stability is not None:
            card.stability = stability
        if difficulty is not None:
            card.difficulty = difficulty

        # Note: fsrs library doesn't have State.New
        # New cards start in Learning state
        state_map = {
            "new": State.Learning,
            "learning": State.Learning,
            "review": State.Review,
            "relearning": State.Relearning,
        }
        card.state = state_map.get(state, State.Learning)
        card.step = step

        if last_review:
            card.last_review = last_review

        fsrs_rating = Rating(rating.value)
        now = datetime.now(timezone.utc)
        updated_card, review_log = self._scheduler.review_card(card, fsrs_rating, now)

        state_name_map = {
            State.Learning: "learning",
            State.Review: "review",
            State.Relearning: "relearning",
        }

        return {
            "stability": updated_card.stability,
            "difficulty": updated_card.difficulty,
            "state": state_name_map.get(updated_card.state, "new"),
            "step": updated_card.step,
            "due": updated_card.due,
            "last_review": updated_card.last_review,
            "review_log": {
                "rating": rating.value,
                "review_time": now.isoformat(),
            },
        }

    def create_new_card(self) -> dict:
        """Create a new FSRS card with default parameters.

        Returns:
            Dict with initial card parameters for a new item.
            New cards start in 'learning' state per fsrs library.
        """
        card = Card()
        return {
            "stability": card.stability,
            "difficulty": card.difficulty,
            "state": "learning",
            "step": card.step,
            "due": card.due,
            "last_review": None,
        }

    def get_retrievability(
        self, stability: float, elapsed_days: int
    ) -> float:
        """Calculate current memory retrievability.

        Uses the FSRS forgetting curve formula.

        Args:
            stability: Card stability parameter
            elapsed_days: Days since last review

        Returns:
            Retrievability probability 0-1
        """
        if stability <= 0:
            return 0.0
        factor = 19 / 81
        return (1 + factor * elapsed_days / stability) ** (-1)

    def _calculate_priority(self, context: StudentContext) -> str:
        """Calculate review priority level.

        Args:
            context: Student context

        Returns:
            Priority level: 'critical', 'high', 'medium', 'low', 'none'
        """
        due_count = context.fsrs_due_items_count

        if due_count == 0:
            return "none"
        if due_count >= 20:
            return "critical"
        if due_count >= 10:
            return "high"
        if due_count >= 5:
            return "medium"
        return "low"

    def _estimate_next_review(self, context: StudentContext) -> datetime | None:
        """Estimate next review time for current topic.

        Uses performance to estimate when review will be needed.

        Args:
            context: Student context

        Returns:
            Estimated next review datetime or None
        """
        if not context.current_topic:
            return None

        mastery = context.current_topic.mastery_level
        streak = context.current_topic.current_streak

        if mastery >= 0.9 and streak >= 5:
            days = 7
        elif mastery >= 0.7:
            days = 3
        elif mastery >= 0.5:
            days = 1
        else:
            hours = 4
            return datetime.now(timezone.utc) + timedelta(hours=hours)

        return datetime.now(timezone.utc) + timedelta(days=days)

    def _calculate_difficulty_recommendation(
        self, context: StudentContext
    ) -> float:
        """Calculate difficulty for review items.

        Reviews should use moderate difficulty to test retention.

        Args:
            context: Student context

        Returns:
            Recommended difficulty 0-1
        """
        if context.fsrs_due_items_count > 0:
            return 0.5

        if context.current_topic:
            return min(0.6, context.current_topic.mastery_level + 0.1)

        return 0.5

    def _estimate_review_time(self, due_count: int) -> int:
        """Estimate time needed for reviews.

        Args:
            due_count: Number of due items

        Returns:
            Estimated minutes
        """
        avg_time_per_item = 1.5
        return max(5, int(due_count * avg_time_per_item))

    def _calculate_confidence(self, context: StudentContext) -> float:
        """Calculate confidence in the recommendation.

        Args:
            context: Student context

        Returns:
            Confidence score 0-1
        """
        confidence = 0.7

        if context.current_topic:
            attempts = context.current_topic.attempts_total
            if attempts >= 20:
                confidence += 0.2
            elif attempts >= 10:
                confidence += 0.1

        return min(1.0, confidence)

    def _build_rationale(
        self,
        context: StudentContext,
        due_count: int,
        review_needed: bool,
        priority: str,
    ) -> str:
        """Build explanation for the recommendation.

        Args:
            context: Student context
            due_count: Number of due items
            review_needed: Whether review is needed
            priority: Priority level

        Returns:
            Human-readable rationale
        """
        if not review_needed:
            return "No items due for review"

        parts = [f"{due_count} items due for review | Priority: {priority}"]

        if context.current_topic:
            mastery = context.current_topic.mastery_level
            parts.append(f"Current topic mastery: {mastery:.2f}")

        estimated_time = self._estimate_review_time(due_count)
        parts.append(f"Estimated time: {estimated_time} minutes")

        return " | ".join(parts)

    @classmethod
    def rating_from_performance(
        cls,
        correct: bool,
        response_time_seconds: float | None,
        used_hint: bool,
        difficulty: float,
    ) -> ReviewRating:
        """Determine FSRS rating from response performance.

        Maps response characteristics to FSRS rating.

        Args:
            correct: Whether answer was correct
            response_time_seconds: Time to respond
            used_hint: Whether hint was used
            difficulty: Question difficulty

        Returns:
            Appropriate ReviewRating
        """
        if not correct:
            return ReviewRating.AGAIN

        if used_hint:
            return ReviewRating.HARD

        expected_time = 30 + (difficulty * 60)

        if response_time_seconds is None:
            return ReviewRating.GOOD

        if response_time_seconds < expected_time * 0.5:
            return ReviewRating.EASY

        if response_time_seconds > expected_time * 2:
            return ReviewRating.HARD

        return ReviewRating.GOOD

    @classmethod
    def get_interval_description(cls, days: int) -> str:
        """Get human-readable interval description.

        Args:
            days: Interval in days

        Returns:
            Human-readable description
        """
        if days < 1:
            hours = int(days * 24)
            if hours < 1:
                return "less than an hour"
            return f"{hours} hour{'s' if hours != 1 else ''}"

        if days == 1:
            return "1 day"

        if days < 7:
            return f"{days} days"

        if days < 30:
            weeks = days // 7
            return f"{weeks} week{'s' if weeks != 1 else ''}"

        if days < 365:
            months = days // 30
            return f"{months} month{'s' if months != 1 else ''}"

        years = days // 365
        return f"{years} year{'s' if years != 1 else ''}"

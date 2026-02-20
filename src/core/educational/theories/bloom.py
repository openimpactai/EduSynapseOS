# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bloom's Taxonomy theory implementation.

Bloom's Taxonomy is a hierarchical classification of cognitive skills
used in education. The revised taxonomy (Anderson & Krathwohl, 2001)
defines six levels of cognitive complexity:

1. Remember - Recall facts and basic concepts
2. Understand - Explain ideas or concepts
3. Apply - Use information in new situations
4. Analyze - Draw connections among ideas
5. Evaluate - Justify a decision or course of action
6. Create - Produce new or original work

This implementation determines the appropriate cognitive level
based on the student's mastery and progression through a topic.
"""

from src.core.educational.theories.base import (
    BaseTheory,
    BloomLevel,
    StudentContext,
    TheoryRecommendation,
)


class BloomTheory(BaseTheory):
    """Bloom's Taxonomy cognitive level calculator.

    Determines the appropriate cognitive complexity level for
    questions and activities based on student mastery.

    Configuration parameters:
        level_thresholds: Dict mapping bloom levels to mastery thresholds
        streak_boost: Whether to boost level after consecutive successes (default: True)
        streak_count: Consecutive correct answers to trigger boost (default: 3)
        allow_skip: Whether to allow skipping levels (default: False)
        min_attempts_per_level: Minimum attempts before advancing (default: 3)
    """

    LEVEL_ORDER = [
        BloomLevel.REMEMBER,
        BloomLevel.UNDERSTAND,
        BloomLevel.APPLY,
        BloomLevel.ANALYZE,
        BloomLevel.EVALUATE,
        BloomLevel.CREATE,
    ]

    DEFAULT_THRESHOLDS = {
        BloomLevel.REMEMBER: 0.0,
        BloomLevel.UNDERSTAND: 0.3,
        BloomLevel.APPLY: 0.5,
        BloomLevel.ANALYZE: 0.65,
        BloomLevel.EVALUATE: 0.8,
        BloomLevel.CREATE: 0.9,
    }

    @property
    def name(self) -> str:
        """Return the theory name."""
        return "bloom"

    def calculate(self, context: StudentContext) -> TheoryRecommendation:
        """Calculate appropriate Bloom's level for the student.

        Progression through levels is based on demonstrated mastery,
        with adjustments for recent performance.

        Args:
            context: Student's current learning context

        Returns:
            Recommendation with bloom_level
        """
        thresholds = self._get_thresholds()
        streak_boost = self.get_param("streak_boost", True)
        streak_count = self.get_param("streak_count", 3)
        allow_skip = self.get_param("allow_skip", False)

        mastery = self._get_effective_mastery(context)
        base_level = self._determine_base_level(mastery, thresholds)
        final_level = base_level
        boost_applied = False

        if (
            streak_boost
            and context.recent_performance.consecutive_correct >= streak_count
            and base_level != BloomLevel.CREATE
        ):
            next_level = self._get_next_level(base_level)
            if next_level:
                if allow_skip or self._is_adjacent(base_level, next_level):
                    final_level = next_level
                    boost_applied = True

        confidence = self._calculate_confidence(context, final_level)
        rationale = self._build_rationale(
            mastery, base_level, final_level, boost_applied, thresholds
        )

        return self._create_recommendation(
            bloom_level=final_level,
            confidence=confidence,
            rationale=rationale,
            extra={
                "mastery": mastery,
                "base_level": base_level.value,
                "boost_applied": boost_applied,
                "level_index": self.LEVEL_ORDER.index(final_level),
                "thresholds": {k.value: v for k, v in thresholds.items()},
            },
        )

    def _get_thresholds(self) -> dict[BloomLevel, float]:
        """Get level thresholds from config or defaults.

        Returns:
            Dict mapping BloomLevel to mastery threshold
        """
        config_thresholds = self.get_param("level_thresholds", {})
        thresholds = dict(self.DEFAULT_THRESHOLDS)

        for level_str, threshold in config_thresholds.items():
            try:
                level = BloomLevel(level_str)
                thresholds[level] = threshold
            except ValueError:
                continue

        return thresholds

    def _get_effective_mastery(self, context: StudentContext) -> float:
        """Get effective mastery level for Bloom calculation.

        Args:
            context: Student context

        Returns:
            Effective mastery level 0-1
        """
        if context.current_topic and context.current_topic.mastery_level > 0:
            return context.current_topic.mastery_level
        return context.overall_mastery

    def _determine_base_level(
        self, mastery: float, thresholds: dict[BloomLevel, float]
    ) -> BloomLevel:
        """Determine the base Bloom level from mastery.

        Finds the highest level whose threshold is met.

        Args:
            mastery: Current mastery level
            thresholds: Level thresholds

        Returns:
            Appropriate BloomLevel
        """
        current_level = BloomLevel.REMEMBER

        for level in self.LEVEL_ORDER:
            if mastery >= thresholds[level]:
                current_level = level
            else:
                break

        return current_level

    def _get_next_level(self, current: BloomLevel) -> BloomLevel | None:
        """Get the next level in the hierarchy.

        Args:
            current: Current level

        Returns:
            Next level or None if at highest
        """
        current_index = self.LEVEL_ORDER.index(current)
        if current_index < len(self.LEVEL_ORDER) - 1:
            return self.LEVEL_ORDER[current_index + 1]
        return None

    def _is_adjacent(self, level1: BloomLevel, level2: BloomLevel) -> bool:
        """Check if two levels are adjacent in the hierarchy.

        Args:
            level1: First level
            level2: Second level

        Returns:
            True if levels are adjacent
        """
        idx1 = self.LEVEL_ORDER.index(level1)
        idx2 = self.LEVEL_ORDER.index(level2)
        return abs(idx1 - idx2) == 1

    def _calculate_confidence(
        self, context: StudentContext, level: BloomLevel
    ) -> float:
        """Calculate confidence in the level recommendation.

        Args:
            context: Student context
            level: Recommended level

        Returns:
            Confidence score 0-1
        """
        confidence = 0.6

        if context.current_topic:
            attempts = context.current_topic.attempts_total
            if attempts >= 15:
                confidence += 0.25
            elif attempts >= 8:
                confidence += 0.15
            elif attempts >= 3:
                confidence += 0.05

        level_idx = self.LEVEL_ORDER.index(level)
        if level_idx >= 4:
            confidence -= 0.1

        return min(1.0, max(0.3, confidence))

    def _build_rationale(
        self,
        mastery: float,
        base_level: BloomLevel,
        final_level: BloomLevel,
        boost_applied: bool,
        thresholds: dict[BloomLevel, float],
    ) -> str:
        """Build explanation for the recommendation.

        Args:
            mastery: Current mastery
            base_level: Base level from mastery
            final_level: Final recommended level
            boost_applied: Whether streak boost was applied
            thresholds: Level thresholds

        Returns:
            Human-readable rationale
        """
        parts = [
            f"Mastery {mastery:.2f} corresponds to '{base_level.value}' level"
        ]

        if boost_applied:
            parts.append(
                f"Boosted to '{final_level.value}' due to consecutive correct answers"
            )

        next_level = self._get_next_level(final_level)
        if next_level:
            next_threshold = thresholds[next_level]
            gap = next_threshold - mastery
            if gap > 0:
                parts.append(
                    f"Need {gap:.2f} more mastery to reach '{next_level.value}'"
                )

        return " | ".join(parts)

    @classmethod
    def get_level_description(cls, level: BloomLevel) -> str:
        """Get a description of what a Bloom level entails.

        Args:
            level: The Bloom level

        Returns:
            Description of the cognitive level
        """
        descriptions = {
            BloomLevel.REMEMBER: "Recall facts, terms, basic concepts, and answers",
            BloomLevel.UNDERSTAND: "Demonstrate understanding of ideas by explaining, summarizing",
            BloomLevel.APPLY: "Use knowledge in new situations, solve problems",
            BloomLevel.ANALYZE: "Break information into parts, identify patterns and relationships",
            BloomLevel.EVALUATE: "Justify decisions, critique, assess value",
            BloomLevel.CREATE: "Produce original work, design solutions, synthesize ideas",
        }
        return descriptions.get(level, "Unknown level")

    @classmethod
    def get_question_verbs(cls, level: BloomLevel) -> list[str]:
        """Get action verbs appropriate for a Bloom level.

        Useful for generating questions at the right cognitive level.

        Args:
            level: The Bloom level

        Returns:
            List of action verbs for that level
        """
        verbs = {
            BloomLevel.REMEMBER: [
                "define", "list", "name", "identify", "recall", "state", "match"
            ],
            BloomLevel.UNDERSTAND: [
                "explain", "describe", "summarize", "classify", "compare", "interpret"
            ],
            BloomLevel.APPLY: [
                "apply", "solve", "use", "demonstrate", "calculate", "implement"
            ],
            BloomLevel.ANALYZE: [
                "analyze", "differentiate", "examine", "compare", "contrast", "organize"
            ],
            BloomLevel.EVALUATE: [
                "evaluate", "judge", "justify", "critique", "assess", "argue"
            ],
            BloomLevel.CREATE: [
                "create", "design", "construct", "develop", "formulate", "compose"
            ],
        }
        return verbs.get(level, [])

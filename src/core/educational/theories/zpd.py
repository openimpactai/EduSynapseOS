# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Zone of Proximal Development (ZPD) theory implementation.

The ZPD theory, developed by Lev Vygotsky, identifies the optimal
difficulty zone for learning - challenging enough to promote growth
but not so difficult as to cause frustration.

Key concepts:
- Zone of Actual Development: What the learner can do independently
- Zone of Proximal Development: What the learner can do with guidance
- Zone of Frustration: What is beyond current capability

This implementation calculates the optimal difficulty based on:
- Current mastery level (semantic memory)
- Recent performance (episodic memory)
- Success/failure streaks
- Historical difficulty preferences (procedural memory)
"""

from src.core.educational.theories.base import (
    BaseTheory,
    ScaffoldLevel,
    StudentContext,
    TheoryRecommendation,
)


class ZPDTheory(BaseTheory):
    """Zone of Proximal Development theory calculator.

    Determines optimal difficulty targeting the learner's ZPD,
    where learning is most effective.

    Configuration parameters:
        zpd_lower_bound: Lower bound of ZPD relative to mastery (default: 0.1)
        zpd_upper_bound: Upper bound of ZPD relative to mastery (default: 0.3)
        streak_adjustment: Adjustment per consecutive correct/incorrect (default: 0.02)
        min_difficulty: Minimum difficulty floor (default: 0.1)
        max_difficulty: Maximum difficulty ceiling (default: 0.95)
        frustration_threshold: Consecutive failures before reducing difficulty (default: 3)
        confidence_threshold: Consecutive successes before increasing difficulty (default: 5)
    """

    @property
    def name(self) -> str:
        """Return the theory name."""
        return "zpd"

    def calculate(self, context: StudentContext) -> TheoryRecommendation:
        """Calculate optimal difficulty based on ZPD.

        The ZPD is calculated as a range slightly above the student's
        current mastery level, adjusted by recent performance and
        diagnostic indicators.

        Diagnostic adjustments:
        - Any indicator >= 0.5: Widens comfort zone (reduces difficulty)
        - HIGH risk (>= 0.7): Lowers frustration threshold for earlier intervention
        - Ensures student stays within productive struggle zone

        Args:
            context: Student's current learning context

        Returns:
            Recommendation with difficulty and scaffold level
        """
        zpd_lower = self.get_param("zpd_lower_bound", 0.1)
        zpd_upper = self.get_param("zpd_upper_bound", 0.3)
        streak_adj = self.get_param("streak_adjustment", 0.02)
        min_diff = self.get_param("min_difficulty", 0.1)
        max_diff = self.get_param("max_difficulty", 0.95)
        frustration_thresh = self.get_param("frustration_threshold", 3)
        confidence_thresh = self.get_param("confidence_threshold", 5)

        base_mastery = self._get_effective_mastery(context)
        zpd_center = base_mastery + (zpd_lower + zpd_upper) / 2
        difficulty = zpd_center
        streak_modifier = 0.0
        rationale_parts = []
        diagnostic_adjustments: list[str] = []

        # Apply diagnostic-aware adjustments if available
        diagnostic_modifier, adjusted_frustration_thresh = self._apply_diagnostic_adjustment(
            context, frustration_thresh, diagnostic_adjustments
        )
        difficulty += diagnostic_modifier

        if context.recent_performance.consecutive_incorrect >= adjusted_frustration_thresh:
            frustration_reduction = min(
                0.15, context.recent_performance.consecutive_incorrect * streak_adj
            )
            difficulty -= frustration_reduction
            streak_modifier = -frustration_reduction
            rationale_parts.append(
                f"Reduced difficulty due to {context.recent_performance.consecutive_incorrect} "
                f"consecutive errors (frustration zone detected)"
            )

        elif context.recent_performance.consecutive_correct >= confidence_thresh:
            confidence_increase = min(
                0.10, (context.recent_performance.consecutive_correct - confidence_thresh + 1) * streak_adj
            )
            difficulty += confidence_increase
            streak_modifier = confidence_increase
            rationale_parts.append(
                f"Increased difficulty due to {context.recent_performance.consecutive_correct} "
                f"consecutive correct answers (confidence zone)"
            )

        difficulty = max(min_diff, min(max_diff, difficulty))
        scaffold_level = self._calculate_scaffold_level(context, difficulty, base_mastery)
        confidence = self._calculate_confidence(context)

        if not rationale_parts:
            rationale_parts.append(
                f"Targeting ZPD at {difficulty:.2f} based on mastery level {base_mastery:.2f}"
            )

        # Add diagnostic adjustments to rationale if any were made
        if diagnostic_adjustments:
            rationale_parts.append(f"Diagnostic: {', '.join(diagnostic_adjustments)}")

        return self._create_recommendation(
            difficulty=difficulty,
            scaffold_level=scaffold_level,
            confidence=confidence,
            rationale=" | ".join(rationale_parts),
            extra={
                "base_mastery": base_mastery,
                "zpd_range": [base_mastery + zpd_lower, base_mastery + zpd_upper],
                "streak_modifier": streak_modifier,
                "diagnostic_modifier": diagnostic_modifier,
                "adjusted_frustration_threshold": adjusted_frustration_thresh,
                "zone": self._identify_zone(difficulty, base_mastery, zpd_lower, zpd_upper),
                "diagnostic_adjustments": diagnostic_adjustments,
            },
        )

    def _apply_diagnostic_adjustment(
        self,
        context: StudentContext,
        base_frustration_thresh: int,
        adjustments: list[str],
    ) -> tuple[float, int]:
        """Apply diagnostic-based adjustments to ZPD calculation.

        Reduces difficulty and lowers frustration threshold for students
        with learning difficulty indicators. Does nothing if no diagnostic
        data is available (new students).

        Args:
            context: Student context with optional diagnostic data.
            base_frustration_thresh: Default frustration threshold.
            adjustments: List to append adjustment descriptions to.

        Returns:
            Tuple of (difficulty_modifier, adjusted_frustration_threshold).
        """
        # No diagnostic data available - use defaults
        # This is normal for new students or when diagnostic feature is disabled
        if not context.diagnostic:
            return 0.0, base_frustration_thresh

        diag = context.diagnostic
        difficulty_modifier = 0.0
        frustration_thresh = base_frustration_thresh

        # Any elevated indicator: widen comfort zone (reduce difficulty)
        if diag.has_elevated_risk:
            if diag.has_high_risk:
                # HIGH risk: significant difficulty reduction
                difficulty_modifier = -0.10
                # Also lower frustration threshold for earlier intervention
                frustration_thresh = max(1, base_frustration_thresh - 2)
                adjustments.append(
                    f"high_risk ({diag.max_risk:.2f}): difficulty-0.10, thresh={frustration_thresh}"
                )
            else:
                # ELEVATED risk: moderate difficulty reduction
                difficulty_modifier = -0.05
                # Slightly lower frustration threshold
                frustration_thresh = max(2, base_frustration_thresh - 1)
                adjustments.append(
                    f"elevated_risk ({diag.max_risk:.2f}): difficulty-0.05, thresh={frustration_thresh}"
                )

        # Attention issues: further reduce difficulty for focus
        if diag.is_risk_elevated(diag.attention_risk):
            attention_modifier = -0.03
            difficulty_modifier += attention_modifier
            adjustments.append(f"attention ({diag.attention_risk:.2f}): difficulty{attention_modifier}")

        return difficulty_modifier, frustration_thresh

    def _get_effective_mastery(self, context: StudentContext) -> float:
        """Get effective mastery level for ZPD calculation.

        Considers current topic mastery with fallback to overall mastery.

        Args:
            context: Student context

        Returns:
            Effective mastery level 0-1
        """
        if context.current_topic and context.current_topic.mastery_level > 0:
            return context.current_topic.mastery_level
        return context.overall_mastery

    def _calculate_scaffold_level(
        self, context: StudentContext, difficulty: float, mastery: float
    ) -> ScaffoldLevel:
        """Calculate appropriate scaffolding level.

        Higher scaffolding when difficulty is much above mastery,
        lower when close to or below mastery.

        Args:
            context: Student context
            difficulty: Target difficulty
            mastery: Current mastery

        Returns:
            Appropriate scaffold level
        """
        gap = difficulty - mastery

        if gap > 0.3 or context.recent_performance.consecutive_incorrect >= 2:
            return ScaffoldLevel.HIGH

        if gap > 0.2 or context.hint_dependency > 0.5:
            return ScaffoldLevel.MODERATE

        if gap > 0.1:
            return ScaffoldLevel.LOW

        return ScaffoldLevel.MINIMAL

    def _calculate_confidence(self, context: StudentContext) -> float:
        """Calculate confidence in this recommendation.

        Higher confidence when we have more data about the student.

        Args:
            context: Student context

        Returns:
            Confidence score 0-1
        """
        confidence = 0.5

        if context.current_topic:
            attempts = context.current_topic.attempts_total
            if attempts >= 20:
                confidence += 0.3
            elif attempts >= 10:
                confidence += 0.2
            elif attempts >= 5:
                confidence += 0.1

        recent_total = (
            context.recent_performance.last_n_correct +
            context.recent_performance.last_n_incorrect
        )
        if recent_total >= 10:
            confidence += 0.2
        elif recent_total >= 5:
            confidence += 0.1

        return min(1.0, confidence)

    def _identify_zone(
        self, difficulty: float, mastery: float, zpd_lower: float, zpd_upper: float
    ) -> str:
        """Identify which zone the difficulty falls into.

        Args:
            difficulty: Target difficulty
            mastery: Current mastery
            zpd_lower: ZPD lower bound offset
            zpd_upper: ZPD upper bound offset

        Returns:
            Zone name: 'comfort', 'zpd', or 'frustration'
        """
        if difficulty < mastery + zpd_lower:
            return "comfort"
        if difficulty <= mastery + zpd_upper:
            return "zpd"
        return "frustration"

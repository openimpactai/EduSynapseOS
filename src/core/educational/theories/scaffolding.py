# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Scaffolding theory implementation.

Scaffolding is an instructional technique where teachers provide
successive levels of temporary support that help students reach
higher levels of understanding. As competence grows, support is
gradually removed (fading).

Key principles:
- Provide appropriate support based on current ability
- Gradually fade support as competence increases
- Maintain student in the zone of productive struggle
- Offer hints before answers, guidance before solutions

This implementation determines the appropriate level of support
based on student performance and hint usage patterns.
"""

from src.core.educational.theories.base import (
    BaseTheory,
    ScaffoldLevel,
    StudentContext,
    TheoryRecommendation,
)


class ScaffoldingTheory(BaseTheory):
    """Scaffolding support level calculator.

    Determines the appropriate level of instructional support
    based on student performance and learning patterns.

    Configuration parameters:
        base_scaffold: Default scaffold level for new topics (default: 3)
        fade_rate: How quickly to reduce support with success (default: 0.15)
        increase_rate: How quickly to increase support with failure (default: 0.25)
        hint_dependency_weight: Weight of hint usage in calculation (default: 0.3)
        min_attempts_for_fade: Minimum attempts before fading (default: 3)
        struggle_threshold: Consecutive errors to increase support (default: 2)
        success_threshold: Consecutive correct to decrease support (default: 3)
    """

    @property
    def name(self) -> str:
        """Return the theory name."""
        return "scaffolding"

    def calculate(self, context: StudentContext) -> TheoryRecommendation:
        """Calculate appropriate scaffolding level.

        Balances support level based on current performance,
        hint dependency, mastery progress, and diagnostic indicators.

        Diagnostic adjustments:
        - Any indicator >= 0.5: Increases support level
        - Attention risk >= 0.5: Recommends shorter task chunks
        - Multiple elevated indicators: Significantly increases support

        Args:
            context: Student's current learning context

        Returns:
            Recommendation with scaffold_level and hints_enabled
        """
        base_scaffold = self.get_param("base_scaffold", 3)
        fade_rate = self.get_param("fade_rate", 0.15)
        increase_rate = self.get_param("increase_rate", 0.25)
        hint_weight = self.get_param("hint_dependency_weight", 0.3)
        min_attempts = self.get_param("min_attempts_for_fade", 3)
        struggle_thresh = self.get_param("struggle_threshold", 2)
        success_thresh = self.get_param("success_threshold", 3)

        current_level = self._calculate_base_level(context, base_scaffold)
        adjusted_level = self._apply_performance_adjustment(
            context, current_level, fade_rate, increase_rate,
            struggle_thresh, success_thresh, min_attempts
        )
        hint_adjusted_level = self._apply_hint_dependency_adjustment(
            context, adjusted_level, hint_weight
        )

        # Apply diagnostic-aware adjustments if available
        diagnostic_adjustments: list[str] = []
        extra_recommendations: dict = {}
        final_level = self._apply_diagnostic_adjustment(
            context, hint_adjusted_level, diagnostic_adjustments, extra_recommendations
        )

        scaffold_level = self._score_to_level(final_level)
        hints_enabled = self._should_enable_hints(context, scaffold_level)
        confidence = self._calculate_confidence(context)
        rationale = self._build_rationale(
            context, current_level, final_level, scaffold_level, hints_enabled
        )

        # Add diagnostic adjustments to rationale if any were made
        if diagnostic_adjustments:
            rationale = f"{rationale} | Diagnostic: {', '.join(diagnostic_adjustments)}"

        extra = {
            "base_level_score": current_level,
            "adjusted_level_score": final_level,
            "hint_dependency": context.hint_dependency,
            "consecutive_correct": context.recent_performance.consecutive_correct,
            "consecutive_incorrect": context.recent_performance.consecutive_incorrect,
            "fading_eligible": self._is_fading_eligible(context, min_attempts),
            "diagnostic_adjustments": diagnostic_adjustments,
        }
        extra.update(extra_recommendations)

        return self._create_recommendation(
            scaffold_level=scaffold_level,
            hints_enabled=hints_enabled,
            confidence=confidence,
            rationale=rationale,
            extra=extra,
        )

    def _apply_diagnostic_adjustment(
        self,
        context: StudentContext,
        current: float,
        adjustments: list[str],
        extra: dict,
    ) -> float:
        """Apply diagnostic-based adjustments to scaffold level.

        Increases support level based on learning difficulty indicators.
        Does nothing if no diagnostic data is available (new students).

        Args:
            context: Student context with optional diagnostic data.
            current: Current scaffold score.
            adjustments: List to append adjustment descriptions to.
            extra: Dict to add extra recommendations to.

        Returns:
            Adjusted scaffold score.
        """
        # No diagnostic data available - use original score
        # This is normal for new students or when diagnostic feature is disabled
        if not context.diagnostic:
            return current

        diag = context.diagnostic
        adjusted = current

        # Count elevated indicators for cumulative effect
        elevated_count = sum([
            1 if diag.is_risk_elevated(diag.dyslexia_risk) else 0,
            1 if diag.is_risk_elevated(diag.dyscalculia_risk) else 0,
            1 if diag.is_risk_elevated(diag.attention_risk) else 0,
            1 if diag.is_risk_elevated(diag.auditory_risk) else 0,
            1 if diag.is_risk_elevated(diag.visual_risk) else 0,
        ])

        # Any elevated indicator: increase support
        if diag.has_elevated_risk:
            if diag.has_high_risk:
                # HIGH risk: significant support increase
                increase = 1.5
                adjustments.append(f"high_risk ({diag.max_risk:.2f}): +{increase}")
            else:
                # ELEVATED risk: moderate support increase
                increase = 0.75
                adjustments.append(f"elevated_risk ({diag.max_risk:.2f}): +{increase}")

            adjusted = min(5.0, adjusted + increase)

            # Multiple elevated indicators: additional support
            if elevated_count >= 2:
                additional = 0.25 * (elevated_count - 1)
                adjusted = min(5.0, adjusted + additional)
                adjustments.append(f"multiple_indicators ({elevated_count}): +{additional}")

        # Attention risk: recommend shorter tasks
        if diag.is_risk_elevated(diag.attention_risk):
            extra["shorter_tasks_recommended"] = True
            if diag.is_risk_high(diag.attention_risk):
                extra["recommended_task_duration_minutes"] = 5
                extra["break_frequency_minutes"] = 10
            else:
                extra["recommended_task_duration_minutes"] = 10
                extra["break_frequency_minutes"] = 15
            adjustments.append(f"attention ({diag.attention_risk:.2f}): shorter tasks")

        # Dyscalculia: recommend visual supports for math
        if diag.is_risk_elevated(diag.dyscalculia_risk):
            extra["visual_math_aids_recommended"] = True
            extra["step_by_step_math_recommended"] = True

        return adjusted

    def _calculate_base_level(
        self, context: StudentContext, default: int
    ) -> float:
        """Calculate base scaffold level from mastery.

        Higher mastery = lower scaffold need.

        Args:
            context: Student context
            default: Default level for new students

        Returns:
            Base scaffold score (1-5 scale)
        """
        mastery = 0.0
        if context.current_topic:
            mastery = context.current_topic.mastery_level
        else:
            mastery = context.overall_mastery

        inverted_mastery = 1.0 - mastery
        base = 1.0 + (inverted_mastery * 4.0)
        return max(1.0, min(5.0, base))

    def _apply_performance_adjustment(
        self,
        context: StudentContext,
        current: float,
        fade_rate: float,
        increase_rate: float,
        struggle_thresh: int,
        success_thresh: int,
        min_attempts: int,
    ) -> float:
        """Adjust scaffold based on recent performance.

        Args:
            context: Student context
            current: Current scaffold score
            fade_rate: Rate of support reduction
            increase_rate: Rate of support increase
            struggle_thresh: Errors before increasing
            success_thresh: Successes before decreasing
            min_attempts: Minimum attempts for adjustment

        Returns:
            Adjusted scaffold score
        """
        adjusted = current

        if context.recent_performance.consecutive_incorrect >= struggle_thresh:
            increase = increase_rate * context.recent_performance.consecutive_incorrect
            adjusted = min(5.0, current + increase)

        elif context.recent_performance.consecutive_correct >= success_thresh:
            if self._is_fading_eligible(context, min_attempts):
                decrease = fade_rate * (
                    context.recent_performance.consecutive_correct - success_thresh + 1
                )
                adjusted = max(1.0, current - decrease)

        return adjusted

    def _apply_hint_dependency_adjustment(
        self, context: StudentContext, current: float, hint_weight: float
    ) -> float:
        """Adjust scaffold based on hint usage patterns.

        Students who frequently use hints may need more support.

        Args:
            context: Student context
            current: Current scaffold score
            hint_weight: Weight of hint dependency

        Returns:
            Final scaffold score
        """
        if context.hint_dependency > 0.7:
            increase = hint_weight * (context.hint_dependency - 0.3)
            return min(5.0, current + increase)

        elif context.hint_dependency < 0.2 and current > 2.0:
            decrease = hint_weight * (0.2 - context.hint_dependency)
            return max(1.0, current - decrease)

        return current

    def _score_to_level(self, score: float) -> ScaffoldLevel:
        """Convert numeric score to ScaffoldLevel enum.

        Args:
            score: Scaffold score 1-5

        Returns:
            Corresponding ScaffoldLevel
        """
        rounded = round(score)
        level_map = {
            1: ScaffoldLevel.MINIMAL,
            2: ScaffoldLevel.LOW,
            3: ScaffoldLevel.MODERATE,
            4: ScaffoldLevel.HIGH,
            5: ScaffoldLevel.MAXIMUM,
        }
        return level_map.get(rounded, ScaffoldLevel.MODERATE)

    def _should_enable_hints(
        self, context: StudentContext, level: ScaffoldLevel
    ) -> bool:
        """Determine if hints should be enabled.

        Args:
            context: Student context
            level: Current scaffold level

        Returns:
            Whether to enable hints
        """
        if level.value >= ScaffoldLevel.MODERATE.value:
            return True

        if context.recent_performance.consecutive_incorrect >= 1:
            return True

        if context.hint_dependency > 0.5:
            return True

        return False

    def _is_fading_eligible(
        self, context: StudentContext, min_attempts: int
    ) -> bool:
        """Check if student is eligible for support fading.

        Args:
            context: Student context
            min_attempts: Minimum attempts required

        Returns:
            Whether fading is appropriate
        """
        if not context.current_topic:
            return False

        if context.current_topic.attempts_total < min_attempts:
            return False

        if context.current_topic.mastery_level < 0.3:
            return False

        return True

    def _calculate_confidence(self, context: StudentContext) -> float:
        """Calculate confidence in scaffold recommendation.

        Args:
            context: Student context

        Returns:
            Confidence score 0-1
        """
        confidence = 0.5

        if context.current_topic:
            attempts = context.current_topic.attempts_total
            if attempts >= 15:
                confidence += 0.3
            elif attempts >= 8:
                confidence += 0.2
            elif attempts >= 3:
                confidence += 0.1

        recent_total = (
            context.recent_performance.last_n_correct +
            context.recent_performance.last_n_incorrect
        )
        if recent_total >= 5:
            confidence += 0.1

        return min(1.0, confidence)

    def _build_rationale(
        self,
        context: StudentContext,
        base_score: float,
        final_score: float,
        level: ScaffoldLevel,
        hints_enabled: bool,
    ) -> str:
        """Build explanation for the recommendation.

        Args:
            context: Student context
            base_score: Initial scaffold score
            final_score: Adjusted scaffold score
            level: Final scaffold level
            hints_enabled: Whether hints are enabled

        Returns:
            Human-readable rationale
        """
        parts = [f"Scaffold level: {level.name} (score: {final_score:.2f})"]

        if abs(final_score - base_score) > 0.1:
            direction = "increased" if final_score > base_score else "decreased"
            parts.append(f"Support {direction} from base {base_score:.2f}")

        if context.recent_performance.consecutive_incorrect >= 2:
            parts.append(
                f"Struggling ({context.recent_performance.consecutive_incorrect} errors)"
            )
        elif context.recent_performance.consecutive_correct >= 3:
            parts.append(
                f"Succeeding ({context.recent_performance.consecutive_correct} correct)"
            )

        if hints_enabled:
            parts.append("Hints enabled")

        return " | ".join(parts)

    @classmethod
    def get_support_strategies(cls, level: ScaffoldLevel) -> list[str]:
        """Get instructional strategies for a scaffold level.

        Args:
            level: The scaffold level

        Returns:
            List of support strategies
        """
        strategies = {
            ScaffoldLevel.MINIMAL: [
                "Let student work independently",
                "Only intervene if stuck for extended time",
                "Provide minimal prompts only if requested",
            ],
            ScaffoldLevel.LOW: [
                "Offer occasional check-ins",
                "Provide light prompts when hesitation detected",
                "Encourage self-correction",
            ],
            ScaffoldLevel.MODERATE: [
                "Break problem into steps",
                "Provide hints before full solutions",
                "Ask guiding questions",
                "Offer worked examples",
            ],
            ScaffoldLevel.HIGH: [
                "Provide step-by-step guidance",
                "Offer multiple hints",
                "Model problem-solving process",
                "Use think-aloud explanations",
            ],
            ScaffoldLevel.MAXIMUM: [
                "Provide extensive worked examples",
                "Offer immediate corrective feedback",
                "Use multiple modalities",
                "Break into smallest possible steps",
                "Consider prerequisite review",
            ],
        }
        return strategies.get(level, [])

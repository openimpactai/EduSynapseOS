# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Socratic Method theory implementation.

The Socratic Method is a form of cooperative argumentative dialogue
that stimulates critical thinking and illuminates ideas through
asking and answering questions.

Key principles:
- Use questioning to lead students to discover answers themselves
- Build understanding through dialogue rather than direct instruction
- Challenge assumptions and explore underlying beliefs
- Progress from simple to complex questions

This implementation determines:
- Guide vs. tell ratio (how much questioning vs. direct instruction)
- Questioning style (direct, guided, exploratory, challenging)
- When to switch between modes based on student responses
"""

from src.core.educational.theories.base import (
    BaseTheory,
    QuestioningStyle,
    StudentContext,
    TheoryRecommendation,
)


class SocraticTheory(BaseTheory):
    """Socratic Method questioning style calculator.

    Determines the appropriate balance between guiding questions
    and direct instruction based on student context.

    Configuration parameters:
        base_guide_ratio: Default guide vs tell ratio (default: 0.7)
        frustration_threshold: Errors before reducing questioning (default: 3)
        mastery_threshold: Mastery level for more challenging questions (default: 0.6)
        confidence_boost: Streak length to increase questioning (default: 2)
        min_guide_ratio: Minimum guide ratio (default: 0.3)
        max_guide_ratio: Maximum guide ratio (default: 0.95)
    """

    @property
    def name(self) -> str:
        """Return the theory name."""
        return "socratic"

    def calculate(self, context: StudentContext) -> TheoryRecommendation:
        """Calculate Socratic questioning approach.

        Balances guiding questions with direct instruction based
        on student's current state, performance, and diagnostic indicators.

        Diagnostic adjustments:
        - Any indicator >= 0.5: Uses more guided/direct style (less exploratory)
        - HIGH risk (>= 0.7): Reduces guide ratio for more direct instruction
        - Ensures questioning style is appropriate for learning needs

        Args:
            context: Student's current learning context

        Returns:
            Recommendation with guide_vs_tell_ratio and questioning_style
        """
        base_ratio = self.get_param("base_guide_ratio", 0.7)
        frustration_thresh = self.get_param("frustration_threshold", 3)
        mastery_thresh = self.get_param("mastery_threshold", 0.6)
        confidence_boost = self.get_param("confidence_boost", 2)
        min_ratio = self.get_param("min_guide_ratio", 0.3)
        max_ratio = self.get_param("max_guide_ratio", 0.95)

        mastery = self._get_mastery(context)
        guide_ratio = self._calculate_guide_ratio(
            context, base_ratio, frustration_thresh,
            confidence_boost, min_ratio, max_ratio
        )

        # Apply diagnostic-aware adjustments if available
        diagnostic_adjustments: list[str] = []
        guide_ratio = self._apply_diagnostic_adjustment(
            context, guide_ratio, min_ratio, diagnostic_adjustments
        )

        questioning_style = self._determine_questioning_style(
            context, mastery, mastery_thresh, guide_ratio
        )

        # Override style for high risk students
        questioning_style = self._adjust_style_for_diagnostic(
            context, questioning_style, diagnostic_adjustments
        )

        question_types = self._get_recommended_question_types(
            questioning_style, mastery
        )
        confidence = self._calculate_confidence(context)
        rationale = self._build_rationale(
            context, guide_ratio, questioning_style, mastery
        )

        # Add diagnostic adjustments to rationale if any were made
        if diagnostic_adjustments:
            rationale = f"{rationale} | Diagnostic: {', '.join(diagnostic_adjustments)}"

        return self._create_recommendation(
            guide_vs_tell_ratio=guide_ratio,
            questioning_style=questioning_style,
            confidence=confidence,
            rationale=rationale,
            extra={
                "mastery": mastery,
                "recommended_question_types": question_types,
                "dialogue_mode": self._get_dialogue_mode(guide_ratio),
                "diagnostic_adjustments": diagnostic_adjustments,
            },
        )

    def _apply_diagnostic_adjustment(
        self,
        context: StudentContext,
        guide_ratio: float,
        min_ratio: float,
        adjustments: list[str],
    ) -> float:
        """Apply diagnostic-based adjustments to guide ratio.

        Reduces guide ratio (more direct instruction) for students with
        learning difficulty indicators. Does nothing if no diagnostic
        data is available (new students).

        Args:
            context: Student context with optional diagnostic data.
            guide_ratio: Current guide ratio.
            min_ratio: Minimum guide ratio.
            adjustments: List to append adjustment descriptions to.

        Returns:
            Adjusted guide ratio.
        """
        # No diagnostic data available - use original ratio
        # This is normal for new students or when diagnostic feature is disabled
        if not context.diagnostic:
            return guide_ratio

        diag = context.diagnostic
        adjusted = guide_ratio

        # Any elevated indicator: reduce guide ratio (more direct instruction)
        if diag.has_elevated_risk:
            if diag.has_high_risk:
                # HIGH risk: more direct instruction needed
                reduction = 0.15
                adjustments.append(f"high_risk ({diag.max_risk:.2f}): guide_ratio-{reduction}")
            else:
                # ELEVATED risk: somewhat more direct
                reduction = 0.10
                adjustments.append(f"elevated_risk ({diag.max_risk:.2f}): guide_ratio-{reduction}")

            adjusted = max(min_ratio, adjusted - reduction)

        return adjusted

    def _adjust_style_for_diagnostic(
        self,
        context: StudentContext,
        current_style: QuestioningStyle,
        adjustments: list[str],
    ) -> QuestioningStyle:
        """Adjust questioning style based on diagnostic indicators.

        Prevents using challenging/exploratory styles for students with
        significant learning difficulty indicators.

        Args:
            context: Student context with optional diagnostic data.
            current_style: Currently selected style.
            adjustments: List to append adjustment descriptions to.

        Returns:
            Adjusted questioning style.
        """
        # No diagnostic data available - use current style
        if not context.diagnostic:
            return current_style

        diag = context.diagnostic

        # HIGH risk: avoid challenging style, prefer guided/direct
        if diag.has_high_risk:
            if current_style == QuestioningStyle.CHALLENGING:
                adjustments.append("style: CHALLENGING→GUIDED (high risk)")
                return QuestioningStyle.GUIDED
            if current_style == QuestioningStyle.EXPLORATORY:
                adjustments.append("style: EXPLORATORY→GUIDED (high risk)")
                return QuestioningStyle.GUIDED

        # ELEVATED risk: avoid challenging style
        elif diag.has_elevated_risk:
            if current_style == QuestioningStyle.CHALLENGING:
                adjustments.append("style: CHALLENGING→EXPLORATORY (elevated risk)")
                return QuestioningStyle.EXPLORATORY

        return current_style

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

    def _calculate_guide_ratio(
        self,
        context: StudentContext,
        base_ratio: float,
        frustration_thresh: int,
        confidence_boost: int,
        min_ratio: float,
        max_ratio: float,
    ) -> float:
        """Calculate the guide vs. tell ratio.

        Higher ratio = more guiding questions, less direct instruction.

        Args:
            context: Student context
            base_ratio: Starting ratio
            frustration_thresh: Errors before reducing
            confidence_boost: Successes before increasing
            min_ratio: Minimum ratio
            max_ratio: Maximum ratio

        Returns:
            Guide ratio 0-1
        """
        ratio = base_ratio

        if context.recent_performance.consecutive_incorrect >= frustration_thresh:
            reduction = 0.1 * (
                context.recent_performance.consecutive_incorrect - frustration_thresh + 1
            )
            ratio = max(min_ratio, ratio - reduction)

        elif context.recent_performance.consecutive_correct >= confidence_boost:
            increase = 0.05 * (
                context.recent_performance.consecutive_correct - confidence_boost + 1
            )
            ratio = min(max_ratio, ratio + increase)

        if context.hint_dependency > 0.6:
            ratio = max(min_ratio, ratio - 0.15)

        return ratio

    def _determine_questioning_style(
        self,
        context: StudentContext,
        mastery: float,
        mastery_thresh: float,
        guide_ratio: float,
    ) -> QuestioningStyle:
        """Determine the appropriate questioning style.

        Args:
            context: Student context
            mastery: Current mastery
            mastery_thresh: Threshold for advanced questioning
            guide_ratio: Current guide ratio

        Returns:
            Appropriate QuestioningStyle
        """
        if context.recent_performance.consecutive_incorrect >= 3:
            return QuestioningStyle.DIRECT

        if guide_ratio < 0.4:
            return QuestioningStyle.DIRECT

        if mastery >= mastery_thresh:
            if context.recent_performance.consecutive_correct >= 3:
                return QuestioningStyle.CHALLENGING
            return QuestioningStyle.EXPLORATORY

        if context.recent_performance.consecutive_correct >= 2:
            return QuestioningStyle.EXPLORATORY

        return QuestioningStyle.GUIDED

    def _get_recommended_question_types(
        self, style: QuestioningStyle, mastery: float
    ) -> list[str]:
        """Get question types appropriate for the style.

        Args:
            style: Questioning style
            mastery: Current mastery

        Returns:
            List of recommended question types
        """
        question_types = {
            QuestioningStyle.DIRECT: [
                "clarification",
                "yes_no",
                "fill_in_blank",
                "recognition",
            ],
            QuestioningStyle.GUIDED: [
                "what_if",
                "how_would_you",
                "can_you_explain",
                "step_by_step",
            ],
            QuestioningStyle.EXPLORATORY: [
                "why",
                "what_are_alternatives",
                "compare_contrast",
                "predict_outcome",
            ],
            QuestioningStyle.CHALLENGING: [
                "counter_example",
                "devil_advocate",
                "implications",
                "synthesis",
                "evaluation",
            ],
        }

        base_types = question_types.get(style, ["clarification"])

        if mastery > 0.8:
            base_types.extend(["meta_cognitive", "self_assessment"])

        return base_types

    def _get_dialogue_mode(self, guide_ratio: float) -> str:
        """Get the dialogue mode description.

        Args:
            guide_ratio: Current guide ratio

        Returns:
            Dialogue mode name
        """
        if guide_ratio >= 0.8:
            return "pure_socratic"
        if guide_ratio >= 0.6:
            return "guided_discovery"
        if guide_ratio >= 0.4:
            return "mixed"
        return "direct_instruction"

    def _calculate_confidence(self, context: StudentContext) -> float:
        """Calculate confidence in the recommendation.

        Args:
            context: Student context

        Returns:
            Confidence score 0-1
        """
        confidence = 0.6

        if context.current_topic:
            attempts = context.current_topic.attempts_total
            if attempts >= 10:
                confidence += 0.2
            elif attempts >= 5:
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
        guide_ratio: float,
        style: QuestioningStyle,
        mastery: float,
    ) -> str:
        """Build explanation for the recommendation.

        Args:
            context: Student context
            guide_ratio: Guide ratio
            style: Questioning style
            mastery: Current mastery

        Returns:
            Human-readable rationale
        """
        mode = self._get_dialogue_mode(guide_ratio)
        parts = [
            f"Style: {style.value} | Mode: {mode} | Guide ratio: {guide_ratio:.2f}"
        ]

        if context.recent_performance.consecutive_incorrect >= 3:
            parts.append("Switching to direct mode due to struggles")
        elif context.recent_performance.consecutive_correct >= 3:
            parts.append("Increasing questioning due to success")

        if mastery >= 0.6:
            parts.append(f"Higher mastery ({mastery:.2f}) enables deeper questioning")

        return " | ".join(parts)

    @classmethod
    def get_question_starters(cls, style: QuestioningStyle) -> list[str]:
        """Get question starter phrases for a style.

        Useful for generating questions.

        Args:
            style: Questioning style

        Returns:
            List of question starters
        """
        starters = {
            QuestioningStyle.DIRECT: [
                "What is...",
                "Can you recall...",
                "Do you remember...",
                "Is it true that...",
            ],
            QuestioningStyle.GUIDED: [
                "How would you...",
                "What do you think happens when...",
                "Can you explain why...",
                "What's the next step in...",
            ],
            QuestioningStyle.EXPLORATORY: [
                "Why do you think...",
                "What would happen if...",
                "How does this compare to...",
                "What are some other ways to...",
            ],
            QuestioningStyle.CHALLENGING: [
                "Can you think of a counterexample...",
                "What if I told you that...",
                "How would you defend...",
                "What are the implications of...",
            ],
        }
        return starters.get(style, ["What...", "How...", "Why..."])

    @classmethod
    def get_follow_up_strategies(cls, style: QuestioningStyle) -> list[str]:
        """Get follow-up strategies for a questioning style.

        Args:
            style: Questioning style

        Returns:
            List of follow-up strategies
        """
        strategies = {
            QuestioningStyle.DIRECT: [
                "Provide immediate feedback",
                "Offer correct answer if needed",
                "Give brief explanation",
            ],
            QuestioningStyle.GUIDED: [
                "Ask probing follow-up questions",
                "Request elaboration on answers",
                "Connect to prior knowledge",
            ],
            QuestioningStyle.EXPLORATORY: [
                "Encourage multiple perspectives",
                "Ask for evidence and reasoning",
                "Explore connections to other topics",
            ],
            QuestioningStyle.CHALLENGING: [
                "Present opposing viewpoints",
                "Push for deeper analysis",
                "Encourage self-questioning",
            ],
        }
        return strategies.get(style, [])

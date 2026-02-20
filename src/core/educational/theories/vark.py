# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""VARK learning styles theory implementation.

VARK is a learning styles model developed by Neil Fleming that
categorizes learners into four main types:

- Visual (V): Prefer diagrams, charts, graphs, maps
- Auditory (A): Prefer listening, discussion, verbal explanation
- Reading/Writing (R): Prefer text, lists, definitions
- Kinesthetic (K): Prefer hands-on experience, examples, practice

Most learners are multimodal, using a combination of styles.
This implementation recommends content format based on the
student's VARK profile from procedural memory.
"""

from src.core.educational.theories.base import (
    BaseTheory,
    ContentFormat,
    StudentContext,
    TheoryRecommendation,
    VARKScores,
)


class VARKTheory(BaseTheory):
    """VARK learning style theory calculator.

    Recommends content format based on the student's
    learning style preferences.

    Configuration parameters:
        multimodal_threshold: Max score to be considered multimodal (default: 0.35)
        adaptation_strength: How strongly to weight VARK (default: 0.8)
        fallback_format: Default format for new students (default: "multimodal")
        topic_overrides: Dict of topic_full_code -> preferred format overrides
    """

    @property
    def name(self) -> str:
        """Return the theory name."""
        return "vark"

    def calculate(self, context: StudentContext) -> TheoryRecommendation:
        """Calculate recommended content format based on VARK.

        Uses the student's VARK profile to recommend the most
        effective content delivery format. When diagnostic indicators
        are present, adjusts recommendations to accommodate learning
        difficulty patterns.

        Diagnostic adjustments:
        - Dyslexia risk >= 0.5: Reduces reading preference, boosts visual/auditory
        - Auditory risk >= 0.5: Reduces auditory preference, boosts visual
        - Visual risk >= 0.5: Reduces visual preference, boosts auditory/kinesthetic

        Args:
            context: Student's current learning context

        Returns:
            Recommendation with content_format
        """
        multimodal_thresh = self.get_param("multimodal_threshold", 0.35)
        adaptation_strength = self.get_param("adaptation_strength", 0.8)
        fallback_format_str = self.get_param("fallback_format", "multimodal")

        try:
            fallback_format = ContentFormat(fallback_format_str)
        except ValueError:
            fallback_format = ContentFormat.MULTIMODAL

        # Get base VARK scores
        vark = context.vark_scores

        # Apply diagnostic-aware adjustments if available
        diagnostic_adjustments: list[str] = []
        adjusted_vark = self._apply_diagnostic_adjustments(
            context, vark, diagnostic_adjustments
        )

        recommended_format = self._determine_format(adjusted_vark, multimodal_thresh)
        secondary_formats = self._get_secondary_formats(adjusted_vark, recommended_format)
        confidence = self._calculate_confidence(context, adjusted_vark, adaptation_strength)
        rationale = self._build_rationale(adjusted_vark, recommended_format, secondary_formats)

        # Add diagnostic adjustments to rationale if any were made
        if diagnostic_adjustments:
            rationale = f"{rationale} | Diagnostic adjustments: {', '.join(diagnostic_adjustments)}"

        if recommended_format == ContentFormat.MULTIMODAL and confidence < 0.5:
            recommended_format = fallback_format
            rationale = f"Using fallback format '{fallback_format.value}' due to low confidence"

        return self._create_recommendation(
            content_format=recommended_format,
            confidence=confidence,
            rationale=rationale,
            extra={
                "vark_scores": {
                    "visual": adjusted_vark.visual,
                    "auditory": adjusted_vark.auditory,
                    "reading": adjusted_vark.reading,
                    "kinesthetic": adjusted_vark.kinesthetic,
                },
                "original_vark_scores": {
                    "visual": vark.visual,
                    "auditory": vark.auditory,
                    "reading": vark.reading,
                    "kinesthetic": vark.kinesthetic,
                },
                "secondary_formats": [f.value for f in secondary_formats],
                "is_multimodal": recommended_format == ContentFormat.MULTIMODAL,
                "diagnostic_adjustments": diagnostic_adjustments,
            },
        )

    def _apply_diagnostic_adjustments(
        self,
        context: StudentContext,
        vark: VARKScores,
        adjustments: list[str],
    ) -> VARKScores:
        """Apply diagnostic-based adjustments to VARK scores.

        Modifies VARK preferences based on learning difficulty indicators
        to provide more appropriate content formats. Does nothing if no
        diagnostic data is available (new students use default behavior).

        Args:
            context: Student context with optional diagnostic data.
            vark: Original VARK scores.
            adjustments: List to append adjustment descriptions to.

        Returns:
            Adjusted VARKScores (or original if no adjustments needed).
        """
        # No diagnostic data available - use original scores
        # This is normal for new students or when diagnostic feature is disabled
        if not context.diagnostic:
            return vark

        diag = context.diagnostic

        # Start with original values
        visual = vark.visual
        auditory = vark.auditory
        reading = vark.reading
        kinesthetic = vark.kinesthetic

        # Dyslexia: reading difficulties - reduce reading, boost visual/auditory
        if diag.is_risk_elevated(diag.dyslexia_risk):
            reduction = 0.3 if diag.is_risk_high(diag.dyslexia_risk) else 0.2
            reading = max(0.0, reading - reduction)
            visual = min(1.0, visual + reduction * 0.5)
            auditory = min(1.0, auditory + reduction * 0.5)
            adjustments.append(
                f"dyslexia ({diag.dyslexia_risk:.2f}): reading-{reduction:.1f}"
            )

        # Auditory processing: reduce auditory, boost visual/kinesthetic
        if diag.is_risk_elevated(diag.auditory_risk):
            reduction = 0.3 if diag.is_risk_high(diag.auditory_risk) else 0.2
            auditory = max(0.0, auditory - reduction)
            visual = min(1.0, visual + reduction * 0.6)
            kinesthetic = min(1.0, kinesthetic + reduction * 0.4)
            adjustments.append(
                f"auditory ({diag.auditory_risk:.2f}): auditory-{reduction:.1f}"
            )

        # Visual processing: reduce visual, boost auditory/kinesthetic
        if diag.is_risk_elevated(diag.visual_risk):
            reduction = 0.3 if diag.is_risk_high(diag.visual_risk) else 0.2
            visual = max(0.0, visual - reduction)
            auditory = min(1.0, auditory + reduction * 0.6)
            kinesthetic = min(1.0, kinesthetic + reduction * 0.4)
            adjustments.append(
                f"visual ({diag.visual_risk:.2f}): visual-{reduction:.1f}"
            )

        return VARKScores(
            visual=visual,
            auditory=auditory,
            reading=reading,
            kinesthetic=kinesthetic,
        )

    def _determine_format(
        self, vark: VARKScores, multimodal_threshold: float
    ) -> ContentFormat:
        """Determine the primary content format.

        Args:
            vark: VARK score profile
            multimodal_threshold: Threshold for multimodal classification

        Returns:
            Recommended ContentFormat
        """
        scores = {
            ContentFormat.VISUAL: vark.visual,
            ContentFormat.AUDITORY: vark.auditory,
            ContentFormat.READING: vark.reading,
            ContentFormat.KINESTHETIC: vark.kinesthetic,
        }

        max_format = max(scores, key=lambda k: scores[k])
        max_score = scores[max_format]

        if max_score < multimodal_threshold:
            return ContentFormat.MULTIMODAL

        second_highest = sorted(scores.values(), reverse=True)[1]
        if max_score - second_highest < 0.1:
            return ContentFormat.MULTIMODAL

        return max_format

    def _get_secondary_formats(
        self, vark: VARKScores, primary: ContentFormat
    ) -> list[ContentFormat]:
        """Get secondary format recommendations.

        Args:
            vark: VARK score profile
            primary: Primary recommended format

        Returns:
            List of secondary formats above threshold
        """
        scores = {
            ContentFormat.VISUAL: vark.visual,
            ContentFormat.AUDITORY: vark.auditory,
            ContentFormat.READING: vark.reading,
            ContentFormat.KINESTHETIC: vark.kinesthetic,
        }

        secondary = []
        threshold = 0.2

        for fmt, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if fmt != primary and score >= threshold:
                secondary.append(fmt)
                if len(secondary) >= 2:
                    break

        return secondary

    def _calculate_confidence(
        self, context: StudentContext, vark: VARKScores, adaptation_strength: float
    ) -> float:
        """Calculate confidence in the format recommendation.

        Higher confidence when VARK profile is well-established
        with clear preferences.

        Args:
            context: Student context
            vark: VARK score profile
            adaptation_strength: Configuration weight

        Returns:
            Confidence score 0-1
        """
        scores = [vark.visual, vark.auditory, vark.reading, vark.kinesthetic]
        max_score = max(scores)
        score_variance = max(scores) - min(scores)

        confidence = 0.5

        if score_variance > 0.3:
            confidence += 0.25
        elif score_variance > 0.15:
            confidence += 0.1

        if max_score > 0.5:
            confidence += 0.15

        if context.current_topic and context.current_topic.attempts_total >= 10:
            confidence += 0.1

        return min(1.0, confidence * adaptation_strength + (1 - adaptation_strength) * 0.5)

    def _build_rationale(
        self,
        vark: VARKScores,
        primary: ContentFormat,
        secondary: list[ContentFormat],
    ) -> str:
        """Build explanation for the recommendation.

        Args:
            vark: VARK scores
            primary: Primary format
            secondary: Secondary formats

        Returns:
            Human-readable rationale
        """
        scores_str = (
            f"V:{vark.visual:.2f} A:{vark.auditory:.2f} "
            f"R:{vark.reading:.2f} K:{vark.kinesthetic:.2f}"
        )

        if primary == ContentFormat.MULTIMODAL:
            return f"Multimodal learner ({scores_str}), mix content types"

        parts = [f"Primary style: {primary.value} ({scores_str})"]

        if secondary:
            secondary_str = ", ".join(f.value for f in secondary)
            parts.append(f"Secondary: {secondary_str}")

        return " | ".join(parts)

    @classmethod
    def get_format_strategies(cls, format_type: ContentFormat) -> list[str]:
        """Get teaching strategies for a content format.

        Useful for generating appropriate content.

        Args:
            format_type: The content format

        Returns:
            List of teaching strategies
        """
        strategies = {
            ContentFormat.VISUAL: [
                "Use diagrams and flowcharts",
                "Include graphs and charts",
                "Provide visual representations",
                "Use color coding",
                "Show spatial relationships",
            ],
            ContentFormat.AUDITORY: [
                "Explain concepts verbally",
                "Use discussion and dialogue",
                "Include mnemonics and rhymes",
                "Encourage talking through problems",
                "Provide verbal summaries",
            ],
            ContentFormat.READING: [
                "Provide written explanations",
                "Use lists and definitions",
                "Include reading materials",
                "Encourage note-taking",
                "Offer written examples",
            ],
            ContentFormat.KINESTHETIC: [
                "Use hands-on examples",
                "Provide practice problems",
                "Include real-world applications",
                "Encourage learning by doing",
                "Use simulations and experiments",
            ],
            ContentFormat.MULTIMODAL: [
                "Combine multiple formats",
                "Offer content variety",
                "Allow format switching",
                "Use integrated approaches",
                "Provide multiple representations",
            ],
        }
        return strategies.get(format_type, [])

    @classmethod
    def get_content_types(cls, format_type: ContentFormat) -> list[str]:
        """Get appropriate content types for a format.

        Args:
            format_type: The content format

        Returns:
            List of content type suggestions
        """
        content_types = {
            ContentFormat.VISUAL: [
                "diagram", "chart", "graph", "infographic",
                "flowchart", "mind_map", "illustration",
            ],
            ContentFormat.AUDITORY: [
                "explanation", "discussion", "verbal_walkthrough",
                "audio_example", "narration",
            ],
            ContentFormat.READING: [
                "text", "definition", "written_example",
                "article", "documentation",
            ],
            ContentFormat.KINESTHETIC: [
                "practice_problem", "interactive_example",
                "hands_on_exercise", "simulation", "experiment",
            ],
            ContentFormat.MULTIMODAL: [
                "mixed", "interactive", "multimedia",
            ],
        }
        return content_types.get(format_type, ["text"])

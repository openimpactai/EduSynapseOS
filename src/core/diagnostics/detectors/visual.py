# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Visual processing difficulty indicator detector.

This module provides detection of potential visual processing
difficulty indicators based on student performance patterns.

IMPORTANT: This detector identifies INDICATORS only, not diagnoses.
Professional evaluation by qualified specialists is required for diagnosis.
"""

from collections import defaultdict

from src.core.diagnostics.detectors.base import (
    BaseDetector,
    DetectorResult,
    Evidence,
    IndicatorType,
    StudentData,
    ThresholdLevel,
)


class VisualDetector(BaseDetector):
    """Detector for potential visual processing difficulty indicators.

    Analyzes student performance patterns to identify signs that may
    indicate visual processing difficulties.

    Indicators analyzed:
    - Performance on visual vs. text-based questions
    - Response time patterns on visual content
    - Spatial reasoning performance
    - Pattern recognition difficulties

    IMPORTANT: High scores indicate need for professional evaluation,
    not a diagnosis of visual processing disorder.
    """

    # Weights for different indicator categories
    WEIGHTS = {
        "visual_vs_text": 0.30,
        "visual_response_time": 0.25,
        "spatial_performance": 0.25,
        "pattern_recognition": 0.20,
    }

    # Keywords indicating visual content
    VISUAL_KEYWORDS = [
        "graph", "chart", "diagram", "image", "picture", "figure",
        "grafik", "diyagram", "şekil", "resim", "görsel",
        "geometry", "geometri", "shape", "şekil", "spatial",
    ]

    # Keywords indicating spatial/geometry content
    SPATIAL_KEYWORDS = [
        "geometry", "geometri", "shape", "angle", "açı",
        "triangle", "üçgen", "square", "kare", "circle", "daire",
        "area", "alan", "perimeter", "çevre", "volume", "hacim",
        "coordinate", "koordinat", "graph", "grafik",
    ]

    @property
    def indicator_type(self) -> IndicatorType:
        """Return visual indicator type."""
        return IndicatorType.VISUAL

    @property
    def name(self) -> str:
        """Return detector name."""
        return "Visual Processing Indicator Detector"

    @property
    def description(self) -> str:
        """Return detector description."""
        return (
            "Analyzes performance patterns for potential visual processing indicators "
            "including visual vs. text performance and spatial reasoning difficulties."
        )

    async def analyze(self, student_data: StudentData) -> DetectorResult:
        """Analyze student data for visual processing indicators.

        Args:
            student_data: Aggregated student data.

        Returns:
            DetectorResult with risk score and evidence.
        """
        if not self.has_sufficient_data(student_data):
            self.logger.debug(
                "Insufficient data for visual analysis",
                extra={"student_id": student_data.student_id},
            )
            return DetectorResult.no_data(self.indicator_type)

        evidence_list: list[Evidence] = []
        component_scores: dict[str, float] = {}

        # Analyze visual vs text performance
        visual_text_score, visual_text_evidence = (
            self._analyze_visual_vs_text_performance(student_data)
        )
        component_scores["visual_vs_text"] = visual_text_score
        evidence_list.extend(visual_text_evidence)

        # Analyze visual content response times
        response_time_score, response_time_evidence = (
            self._analyze_visual_response_time(student_data)
        )
        component_scores["visual_response_time"] = response_time_score
        evidence_list.extend(response_time_evidence)

        # Analyze spatial reasoning performance
        spatial_score, spatial_evidence = self._analyze_spatial_performance(
            student_data
        )
        component_scores["spatial_performance"] = spatial_score
        evidence_list.extend(spatial_evidence)

        # Analyze pattern recognition
        pattern_score, pattern_evidence = self._analyze_pattern_recognition(
            student_data
        )
        component_scores["pattern_recognition"] = pattern_score
        evidence_list.extend(pattern_evidence)

        # Calculate weighted risk score
        risk_score = sum(
            component_scores[k] * self.WEIGHTS[k]
            for k in component_scores
            if k in self.WEIGHTS
        )
        risk_score = min(1.0, max(0.0, risk_score))

        # Calculate confidence
        confidence = self.calculate_confidence(
            sample_size=student_data.total_answers,
            evidence_count=len(evidence_list),
            evidence_consistency=self._calculate_evidence_consistency(component_scores),
        )

        # Determine threshold level
        threshold_level = self.calculate_threshold_level(risk_score)

        # Generate summary
        analysis_summary = self._generate_summary(
            component_scores, evidence_list, threshold_level
        )

        self.logger.info(
            "Visual analysis complete",
            extra={
                "student_id": student_data.student_id,
                "risk_score": risk_score,
                "threshold_level": threshold_level.value,
                "evidence_count": len(evidence_list),
            },
        )

        return DetectorResult(
            indicator_type=self.indicator_type,
            risk_score=round(risk_score, 2),
            confidence=confidence,
            threshold_level=threshold_level,
            evidence=evidence_list,
            sample_size=student_data.total_answers,
            analysis_summary=analysis_summary,
        )

    def _analyze_visual_vs_text_performance(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze performance difference between visual and text content.

        Students with visual processing issues may perform worse
        on visual content compared to text-based content.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []

        visual_correct = 0
        visual_total = 0
        text_correct = 0
        text_total = 0

        for answer in student_data.student_answers:
            is_visual = self._is_visual_question(answer)

            evaluation = next(
                (e for e in student_data.evaluation_results
                 if e.answer_id == answer.id),
                None
            )

            if evaluation:
                if is_visual:
                    visual_total += 1
                    if evaluation.is_correct:
                        visual_correct += 1
                else:
                    text_total += 1
                    if evaluation.is_correct:
                        text_correct += 1

        if visual_total < 5 or text_total < 5:
            return 0.0, []

        visual_accuracy = visual_correct / visual_total
        text_accuracy = text_correct / text_total

        # Check for significant performance gap (text better than visual)
        performance_gap = text_accuracy - visual_accuracy

        if performance_gap > 0.2:  # Text is 20%+ better
            evidence_list.append(
                self.create_evidence(
                    category="visual_vs_text",
                    description=f"Text performance {performance_gap:.1%} higher than visual",
                    data={
                        "visual_accuracy": visual_accuracy,
                        "text_accuracy": text_accuracy,
                        "gap": performance_gap,
                        "visual_sample": visual_total,
                        "text_sample": text_total,
                    },
                    weight=0.9,
                )
            )

        # Score based on performance gap
        if performance_gap <= 0.1:
            score = 0.0
        elif performance_gap <= 0.2:
            score = 0.3
        elif performance_gap <= 0.3:
            score = 0.6
        else:
            score = min(1.0, 0.6 + (performance_gap - 0.3) * 2)

        return score, evidence_list

    def _analyze_visual_response_time(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze response time patterns on visual content.

        Significantly longer times on visual content might indicate
        visual processing difficulties.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []

        visual_times: list[float] = []
        text_times: list[float] = []

        for answer in student_data.student_answers:
            if answer.time_spent_seconds <= 0:
                continue

            is_visual = self._is_visual_question(answer)

            if is_visual:
                visual_times.append(float(answer.time_spent_seconds))
            else:
                text_times.append(float(answer.time_spent_seconds))

        if len(visual_times) < 5 or len(text_times) < 5:
            return 0.0, []

        avg_visual_time = sum(visual_times) / len(visual_times)
        avg_text_time = sum(text_times) / len(text_times)

        if avg_text_time == 0:
            return 0.0, []

        time_ratio = avg_visual_time / avg_text_time

        # Visual taking significantly longer is an indicator
        if time_ratio > 1.5:
            evidence_list.append(
                self.create_evidence(
                    category="visual_response_time",
                    description=f"Visual questions take {time_ratio:.1f}x longer than text",
                    data={
                        "avg_visual_time": avg_visual_time,
                        "avg_text_time": avg_text_time,
                        "ratio": time_ratio,
                        "visual_sample": len(visual_times),
                        "text_sample": len(text_times),
                    },
                    weight=0.7,
                )
            )

        # Score based on time ratio
        if time_ratio <= 1.2:
            score = 0.0
        elif time_ratio <= 1.5:
            score = 0.3
        elif time_ratio <= 2.0:
            score = 0.6
        else:
            score = min(1.0, 0.6 + (time_ratio - 2.0) * 0.2)

        return score, evidence_list

    def _analyze_spatial_performance(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze performance on spatial reasoning questions.

        Difficulty with geometry, shapes, and spatial concepts
        can indicate visual processing issues.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []

        # Check semantic memories for spatial/geometry topics
        spatial_memories = [
            mem for mem in student_data.semantic_memories
            if self._is_spatial_topic(mem.entity_type)
        ]

        other_memories = [
            mem for mem in student_data.semantic_memories
            if not self._is_spatial_topic(mem.entity_type)
        ]

        if len(spatial_memories) < 2 or len(other_memories) < 5:
            return 0.0, []

        # Calculate average mastery for spatial vs. other topics
        spatial_mastery = sum(
            float(m.mastery_level) for m in spatial_memories
        ) / len(spatial_memories)

        other_mastery = sum(
            float(m.mastery_level) for m in other_memories
        ) / len(other_memories)

        # Calculate accuracy
        spatial_attempts = sum(m.attempts_total for m in spatial_memories)
        spatial_correct = sum(m.attempts_correct for m in spatial_memories)
        spatial_accuracy = spatial_correct / spatial_attempts if spatial_attempts > 0 else 0

        other_attempts = sum(m.attempts_total for m in other_memories)
        other_correct = sum(m.attempts_correct for m in other_memories)
        other_accuracy = other_correct / other_attempts if other_attempts > 0 else 0

        # Performance gap
        accuracy_gap = other_accuracy - spatial_accuracy
        mastery_gap = other_mastery - spatial_mastery

        if accuracy_gap > 0.15 or mastery_gap > 0.2:
            evidence_list.append(
                self.create_evidence(
                    category="spatial_performance",
                    description=f"Spatial reasoning performance {accuracy_gap:.1%} below average",
                    data={
                        "spatial_accuracy": spatial_accuracy,
                        "other_accuracy": other_accuracy,
                        "accuracy_gap": accuracy_gap,
                        "spatial_mastery": spatial_mastery,
                        "other_mastery": other_mastery,
                    },
                    weight=0.8,
                )
            )

        # Combined score
        combined_gap = (accuracy_gap + mastery_gap) / 2
        if combined_gap <= 0.1:
            score = 0.0
        elif combined_gap <= 0.2:
            score = 0.3
        elif combined_gap <= 0.3:
            score = 0.6
        else:
            score = min(1.0, 0.6 + (combined_gap - 0.3) * 2)

        return score, evidence_list

    def _analyze_pattern_recognition(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze pattern recognition performance.

        Difficulty recognizing visual patterns can indicate
        visual processing issues.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []

        # Look for episodic memories indicating pattern struggles
        pattern_struggles = [
            mem for mem in student_data.episodic_memories
            if self._is_pattern_related_event(mem)
        ]

        pattern_successes = [
            mem for mem in student_data.episodic_memories
            if mem.event_type == "success" and self._is_pattern_related_event(mem)
        ]

        if not pattern_struggles and not pattern_successes:
            return 0.0, []

        total_pattern_events = len(pattern_struggles) + len(pattern_successes)
        struggle_rate = len(pattern_struggles) / max(1, total_pattern_events)

        if struggle_rate > 0.5:  # More struggles than successes with patterns
            evidence_list.append(
                self.create_evidence(
                    category="pattern_recognition",
                    description=f"Pattern recognition struggles in {struggle_rate:.1%} of related events",
                    data={
                        "struggle_count": len(pattern_struggles),
                        "success_count": len(pattern_successes),
                        "rate": struggle_rate,
                    },
                    weight=0.6,
                )
            )

        score = min(1.0, struggle_rate * 1.5)
        return score, evidence_list

    def _is_visual_question(self, answer) -> bool:
        """Determine if question is visual-based.

        Args:
            answer: Student answer object.

        Returns:
            True if question is visual-based.
        """
        # Check display_hint from the question if available
        if hasattr(answer, 'question') and answer.question:
            display_hint = getattr(answer.question, 'display_hint', '')
            if display_hint in ['image', 'graph', 'diagram', 'chart']:
                return True

            # Check question content for visual keywords
            content = getattr(answer.question, 'content', '')
            if content and any(kw in content.lower() for kw in self.VISUAL_KEYWORDS):
                return True

        return False

    def _is_spatial_topic(self, entity_type: str) -> bool:
        """Check if entity type is spatial/geometry related.

        Args:
            entity_type: Entity type string.

        Returns:
            True if spatial/geometry related.
        """
        entity_lower = entity_type.lower()
        return any(kw in entity_lower for kw in self.SPATIAL_KEYWORDS)

    def _is_pattern_related_event(self, memory) -> bool:
        """Check if episodic memory is related to pattern recognition.

        Args:
            memory: Episodic memory instance.

        Returns:
            True if pattern-related.
        """
        pattern_keywords = [
            "pattern", "sequence", "series", "desen", "örüntü",
            "shape", "şekil", "matching", "eşleştirme",
        ]

        summary_lower = memory.summary.lower()
        return any(kw in summary_lower for kw in pattern_keywords)

    def _calculate_evidence_consistency(
        self,
        component_scores: dict[str, float],
    ) -> float:
        """Calculate how consistent the evidence is across components."""
        if not component_scores:
            return 0.0

        scores = list(component_scores.values())
        elevated_count = sum(1 for s in scores if s >= 0.3)
        high_count = sum(1 for s in scores if s >= 0.5)

        if high_count >= 3:
            return 1.0
        elif high_count >= 2 or elevated_count >= 4:
            return 0.8
        elif high_count >= 1 or elevated_count >= 3:
            return 0.6
        elif elevated_count >= 2:
            return 0.4
        elif elevated_count >= 1:
            return 0.2

        return 0.1

    def _generate_summary(
        self,
        component_scores: dict[str, float],
        evidence_list: list[Evidence],
        threshold_level: ThresholdLevel,
    ) -> str:
        """Generate analysis summary."""
        if threshold_level == ThresholdLevel.LOW:
            return "No significant visual processing indicators detected."

        elevated_components = [
            k.replace("_", " ") for k, v in component_scores.items() if v >= 0.3
        ]

        if threshold_level == ThresholdLevel.HIGH:
            return (
                f"Elevated visual processing indicators in {len(elevated_components)} areas: "
                f"{', '.join(elevated_components)}. "
                "Professional evaluation recommended."
            )
        elif threshold_level == ThresholdLevel.ELEVATED:
            return (
                f"Moderate visual indicators in: {', '.join(elevated_components)}. "
                "Continued monitoring suggested."
            )
        else:
            return (
                f"Mild indicators in: {', '.join(elevated_components)}. "
                "Continue regular monitoring."
            )

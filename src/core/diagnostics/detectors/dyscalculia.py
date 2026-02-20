# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dyscalculia indicator detector.

This module provides detection of potential dyscalculia indicators
based on student math performance patterns and behaviors.

IMPORTANT: This detector identifies INDICATORS only, not diagnoses.
Professional evaluation by qualified specialists is required for diagnosis.
"""

import re
from collections import defaultdict

from src.core.diagnostics.detectors.base import (
    BaseDetector,
    DetectorResult,
    Evidence,
    IndicatorType,
    StudentData,
    ThresholdLevel,
)


class DyscalculiaDetector(BaseDetector):
    """Detector for potential dyscalculia indicators.

    Analyzes student math performance to identify signs that may
    indicate numerical processing difficulties associated with dyscalculia.

    Indicators analyzed:
    - Number reversal patterns (12↔21, 6↔9)
    - Basic arithmetic difficulties
    - Place value confusion
    - Inconsistent math performance
    - Math-specific anxiety patterns (avoiding math, much slower)
    - Number sense issues

    IMPORTANT: High scores indicate need for professional evaluation,
    not a diagnosis of dyscalculia.
    """

    # Number reversal pairs
    NUMBER_REVERSALS = [
        ("6", "9"),
        ("2", "5"),  # Can look similar when handwritten
    ]

    # Weights for different indicator categories
    WEIGHTS = {
        "number_reversal": 0.20,
        "arithmetic_difficulty": 0.25,
        "place_value_confusion": 0.20,
        "math_performance_inconsistency": 0.15,
        "math_avoidance": 0.20,
    }

    # Math-related topic keywords
    MATH_KEYWORDS = [
        "math", "matematik", "algebra", "cebir",
        "arithmetic", "aritmetik", "geometry", "geometri",
        "number", "sayı", "fraction", "kesir",
        "decimal", "ondalık", "percent", "yüzde",
        "equation", "denklem", "calculation", "hesaplama",
    ]

    @property
    def indicator_type(self) -> IndicatorType:
        """Return dyscalculia indicator type."""
        return IndicatorType.DYSCALCULIA

    @property
    def name(self) -> str:
        """Return detector name."""
        return "Dyscalculia Indicator Detector"

    @property
    def description(self) -> str:
        """Return detector description."""
        return (
            "Analyzes math performance patterns for potential dyscalculia indicators "
            "including number reversals, arithmetic difficulties, and math avoidance."
        )

    async def analyze(self, student_data: StudentData) -> DetectorResult:
        """Analyze student data for dyscalculia indicators.

        Args:
            student_data: Aggregated student data.

        Returns:
            DetectorResult with risk score and evidence.
        """
        if not self.has_sufficient_data(student_data):
            self.logger.debug(
                "Insufficient data for dyscalculia analysis",
                extra={"student_id": student_data.student_id},
            )
            return DetectorResult.no_data(self.indicator_type)

        evidence_list: list[Evidence] = []
        component_scores: dict[str, float] = {}

        # Analyze number reversals
        reversal_score, reversal_evidence = self._analyze_number_reversals(
            student_data
        )
        component_scores["number_reversal"] = reversal_score
        evidence_list.extend(reversal_evidence)

        # Analyze arithmetic difficulties
        arithmetic_score, arithmetic_evidence = self._analyze_arithmetic_difficulty(
            student_data
        )
        component_scores["arithmetic_difficulty"] = arithmetic_score
        evidence_list.extend(arithmetic_evidence)

        # Analyze place value confusion
        place_value_score, place_value_evidence = self._analyze_place_value_confusion(
            student_data
        )
        component_scores["place_value_confusion"] = place_value_score
        evidence_list.extend(place_value_evidence)

        # Analyze math performance inconsistency
        inconsistency_score, inconsistency_evidence = (
            self._analyze_math_performance_inconsistency(student_data)
        )
        component_scores["math_performance_inconsistency"] = inconsistency_score
        evidence_list.extend(inconsistency_evidence)

        # Analyze math avoidance patterns
        avoidance_score, avoidance_evidence = self._analyze_math_avoidance(
            student_data
        )
        component_scores["math_avoidance"] = avoidance_score
        evidence_list.extend(avoidance_evidence)

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
            "Dyscalculia analysis complete",
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

    def _analyze_number_reversals(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze answers for number reversal patterns.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []
        reversal_instances: list[dict] = []
        total_numeric_answers = 0

        for answer in student_data.student_answers:
            answer_value = self._extract_numeric_value(answer.answer)
            if answer_value is None:
                continue

            total_numeric_answers += 1

            # Check for digit reversal patterns
            answer_str = str(abs(int(answer_value))) if answer_value == int(answer_value) else str(answer_value)

            # Check two-digit reversal (12 vs 21)
            if len(answer_str) == 2 and answer_str[0] != answer_str[1]:
                reversed_str = answer_str[::-1]
                # Store as potential reversal for pattern analysis
                reversal_instances.append({
                    "original": answer_str,
                    "reversed": reversed_str,
                })

            # Check for 6/9 confusion
            if "6" in answer_str or "9" in answer_str:
                # Count occurrences - asymmetric usage might indicate confusion
                count_6 = answer_str.count("6")
                count_9 = answer_str.count("9")
                if count_6 > 0 and count_9 == 0:
                    reversal_instances.append({
                        "pattern": "6_only",
                        "value": answer_str,
                    })

        if total_numeric_answers == 0:
            return 0.0, []

        # Analyze reversal patterns
        reversal_rate = len(reversal_instances) / total_numeric_answers

        if reversal_rate > 0.1:  # More than 10% show reversal patterns
            evidence_list.append(
                self.create_evidence(
                    category="number_reversal",
                    description=f"Number reversal patterns detected in {reversal_rate:.1%} of answers",
                    data={
                        "reversal_count": len(reversal_instances),
                        "total_numeric": total_numeric_answers,
                        "rate": reversal_rate,
                        "examples": reversal_instances[:3],
                    },
                    weight=0.8,
                )
            )

        score = min(1.0, reversal_rate * 5)  # Scale: 20% = max score
        return score, evidence_list

    def _analyze_arithmetic_difficulty(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze basic arithmetic performance.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []

        # Analyze semantic memories for math topics
        math_memories = [
            mem for mem in student_data.semantic_memories
            if self._is_math_related(mem.entity_type)
        ]

        if not math_memories:
            return 0.0, []

        # Calculate average math mastery
        total_mastery = sum(float(mem.mastery_level) for mem in math_memories)
        avg_mastery = total_mastery / len(math_memories)

        # Calculate accuracy from attempts
        total_attempts = sum(mem.attempts_total for mem in math_memories)
        total_correct = sum(mem.attempts_correct for mem in math_memories)
        math_accuracy = total_correct / total_attempts if total_attempts > 0 else 0

        # Find struggling topics
        struggling_topics = [
            mem for mem in math_memories
            if float(mem.mastery_level) < 0.4 and mem.attempts_total >= 5
        ]

        # Evidence for low math performance
        if avg_mastery < 0.4 or math_accuracy < 0.5:
            evidence_list.append(
                self.create_evidence(
                    category="arithmetic_difficulty",
                    description=f"Below-average math performance: {math_accuracy:.1%} accuracy",
                    data={
                        "avg_mastery": avg_mastery,
                        "accuracy": math_accuracy,
                        "total_attempts": total_attempts,
                        "struggling_topic_count": len(struggling_topics),
                    },
                    weight=0.9,
                )
            )

        # Score based on performance
        if math_accuracy >= 0.7:
            score = 0.0
        elif math_accuracy >= 0.5:
            score = 0.3
        elif math_accuracy >= 0.3:
            score = 0.6
        else:
            score = min(1.0, 0.7 + (0.3 - math_accuracy))

        return score, evidence_list

    def _analyze_place_value_confusion(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze for place value understanding issues.

        Place value confusion is a common dyscalculia indicator where
        students confuse tens/ones, hundreds/tens, etc.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []
        place_value_errors = 0
        total_multi_digit_answers = 0

        for answer in student_data.student_answers:
            answer_value = self._extract_numeric_value(answer.answer)
            if answer_value is None or abs(answer_value) < 10:
                continue

            total_multi_digit_answers += 1

            # Check for answers that are off by a factor of 10
            # This would require knowing the correct answer, which we get from evaluation
            evaluation = next(
                (e for e in student_data.evaluation_results
                 if e.answer_id == answer.id),
                None
            )

            if evaluation and not evaluation.is_correct:
                # Check misconceptions for place value issues
                misconceptions = evaluation.misconceptions or []
                for misconception in misconceptions:
                    if isinstance(misconception, dict):
                        misc_type = misconception.get("type", "")
                        if "place_value" in misc_type.lower() or "decimal" in misc_type.lower():
                            place_value_errors += 1

        if total_multi_digit_answers == 0:
            return 0.0, []

        error_rate = place_value_errors / total_multi_digit_answers

        if error_rate > 0.15:  # More than 15% place value errors
            evidence_list.append(
                self.create_evidence(
                    category="place_value_confusion",
                    description=f"Place value errors in {error_rate:.1%} of multi-digit answers",
                    data={
                        "error_count": place_value_errors,
                        "total_multi_digit": total_multi_digit_answers,
                        "rate": error_rate,
                    },
                    weight=0.7,
                )
            )

        score = min(1.0, error_rate * 4)  # Scale: 25% = max score
        return score, evidence_list

    def _analyze_math_performance_inconsistency(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze math performance consistency over time.

        Dyscalculia can cause highly variable performance even on
        similar problems.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []

        # Group sessions by math vs non-math
        math_sessions = []
        other_sessions = []

        for session in student_data.practice_sessions:
            if session.questions_answered == 0:
                continue

            session_accuracy = session.questions_correct / session.questions_answered
            is_math = self._is_math_session(session)

            if is_math:
                math_sessions.append(session_accuracy)
            else:
                other_sessions.append(session_accuracy)

        if len(math_sessions) < 3:
            return 0.0, []

        # Calculate variability
        math_variance = self._calculate_variance(math_sessions)
        math_cv = self._calculate_coefficient_of_variation(math_sessions)

        # Compare with other subjects if available
        other_cv = 0.0
        if len(other_sessions) >= 3:
            other_cv = self._calculate_coefficient_of_variation(other_sessions)

        # High variability in math compared to other subjects is a potential indicator
        relative_variability = math_cv - other_cv if other_cv > 0 else math_cv

        if math_cv > 0.3:  # CV > 30% indicates high variability
            evidence_list.append(
                self.create_evidence(
                    category="math_performance_inconsistency",
                    description=f"High math performance variability (CV: {math_cv:.1%})",
                    data={
                        "math_cv": math_cv,
                        "other_cv": other_cv,
                        "math_session_count": len(math_sessions),
                        "variance": math_variance,
                    },
                    weight=0.6,
                )
            )

        # Score based on variability
        if math_cv <= 0.2:
            score = 0.0
        elif math_cv <= 0.3:
            score = 0.3
        elif math_cv <= 0.5:
            score = 0.6
        else:
            score = min(1.0, 0.6 + (math_cv - 0.5))

        return score, evidence_list

    def _analyze_math_avoidance(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze for math avoidance patterns.

        Students with dyscalculia often avoid math or show anxiety-related
        behaviors like abandoning math sessions.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []

        math_sessions = [s for s in student_data.practice_sessions if self._is_math_session(s)]
        other_sessions = [s for s in student_data.practice_sessions if not self._is_math_session(s)]

        if not math_sessions:
            return 0.0, []

        # Check for abandoned math sessions
        math_abandoned = sum(1 for s in math_sessions if s.status == "abandoned")
        other_abandoned = sum(1 for s in other_sessions if s.status == "abandoned") if other_sessions else 0

        math_abandon_rate = math_abandoned / len(math_sessions)
        other_abandon_rate = other_abandoned / len(other_sessions) if other_sessions else 0

        # Check for shorter math sessions
        math_durations = [s.time_spent_seconds for s in math_sessions if s.time_spent_seconds > 0]
        other_durations = [s.time_spent_seconds for s in other_sessions if s.time_spent_seconds > 0]

        avg_math_duration = sum(math_durations) / len(math_durations) if math_durations else 0
        avg_other_duration = sum(other_durations) / len(other_durations) if other_durations else 0

        avoidance_indicators = 0

        # Higher abandonment rate for math
        if math_abandon_rate > other_abandon_rate + 0.1:
            avoidance_indicators += 1
            evidence_list.append(
                self.create_evidence(
                    category="math_avoidance",
                    description=f"Higher math session abandonment: {math_abandon_rate:.1%} vs {other_abandon_rate:.1%}",
                    data={
                        "math_abandon_rate": math_abandon_rate,
                        "other_abandon_rate": other_abandon_rate,
                    },
                    weight=0.7,
                )
            )

        # Shorter math sessions
        if avg_other_duration > 0 and avg_math_duration < avg_other_duration * 0.7:
            avoidance_indicators += 1
            evidence_list.append(
                self.create_evidence(
                    category="math_avoidance",
                    description=f"Math sessions 30%+ shorter than other subjects",
                    data={
                        "avg_math_duration": avg_math_duration,
                        "avg_other_duration": avg_other_duration,
                        "ratio": avg_math_duration / avg_other_duration if avg_other_duration > 0 else 0,
                    },
                    weight=0.6,
                )
            )

        # Score based on avoidance indicators
        score = min(1.0, avoidance_indicators * 0.4 + math_abandon_rate)
        return score, evidence_list

    def _extract_numeric_value(self, answer_data: dict) -> float | None:
        """Extract numeric value from answer data.

        Args:
            answer_data: Answer data dictionary.

        Returns:
            Numeric value or None.
        """
        if isinstance(answer_data, (int, float)):
            return float(answer_data)

        if isinstance(answer_data, str):
            try:
                # Remove common formatting
                cleaned = answer_data.strip().replace(",", ".").replace(" ", "")
                return float(cleaned)
            except ValueError:
                return None

        if isinstance(answer_data, dict):
            for key in ["value", "answer", "number", "result"]:
                if key in answer_data:
                    return self._extract_numeric_value(answer_data[key])

        return None

    def _is_math_related(self, entity_type: str) -> bool:
        """Check if entity type is math-related.

        Args:
            entity_type: Entity type string.

        Returns:
            True if math-related.
        """
        entity_lower = entity_type.lower()
        return any(kw in entity_lower for kw in self.MATH_KEYWORDS)

    def _is_math_session(self, session) -> bool:
        """Check if session is math-related.

        Args:
            session: Practice session.

        Returns:
            True if math-related.
        """
        # Check topic subject code for math (e.g., "MAT")
        if hasattr(session, "topic_subject_code") and session.topic_subject_code:
            subject_lower = session.topic_subject_code.lower()
            if any(kw in subject_lower for kw in ["mat", "math"]):
                return True

        # Check topic full code
        if hasattr(session, "topic_full_code") and session.topic_full_code:
            topic_lower = session.topic_full_code.lower()
            if any(kw in topic_lower for kw in ["mat", "math"]):
                return True

        # Fallback to session_type
        if session.session_type:
            return "math" in session.session_type.lower()

        return False

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
            return "No significant dyscalculia indicators detected."

        elevated_components = [
            k.replace("_", " ") for k, v in component_scores.items() if v >= 0.3
        ]

        if threshold_level == ThresholdLevel.HIGH:
            return (
                f"Elevated math difficulty indicators in {len(elevated_components)} areas: "
                f"{', '.join(elevated_components)}. "
                "Professional evaluation recommended."
            )
        elif threshold_level == ThresholdLevel.ELEVATED:
            return (
                f"Moderate math difficulty indicators in: {', '.join(elevated_components)}. "
                "Continued monitoring suggested."
            )
        else:
            return (
                f"Mild indicators in: {', '.join(elevated_components)}. "
                "Continue regular monitoring."
            )

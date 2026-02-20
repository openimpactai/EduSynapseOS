# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Attention difficulty indicator detector.

This module provides detection of potential attention-related learning
difficulty indicators based on student behavior patterns.

IMPORTANT: This detector identifies INDICATORS only, not diagnoses.
Professional evaluation by qualified specialists is required for diagnosis.
"""

from collections import defaultdict
from datetime import datetime, timedelta

from src.core.diagnostics.detectors.base import (
    BaseDetector,
    DetectorResult,
    Evidence,
    IndicatorType,
    StudentData,
    ThresholdLevel,
)


class AttentionDetector(BaseDetector):
    """Detector for potential attention difficulty indicators.

    Analyzes student behavior patterns to identify signs that may
    indicate attention-related learning difficulties.

    Indicators analyzed:
    - Session duration patterns (very short sessions)
    - Answer time variability (highly inconsistent response times)
    - Session abandonment rates
    - Performance inconsistency within sessions
    - Response time patterns suggesting distraction

    IMPORTANT: High scores indicate need for professional evaluation,
    not a diagnosis of attention disorders.
    """

    # Thresholds for session analysis
    MIN_SESSION_DURATION_SECONDS = 120  # 2 minutes minimum expected
    SHORT_SESSION_THRESHOLD = 300  # 5 minutes considered short

    # Weights for different indicator categories
    WEIGHTS = {
        "session_duration": 0.25,
        "answer_time_variability": 0.25,
        "session_abandonment": 0.20,
        "within_session_inconsistency": 0.15,
        "distraction_patterns": 0.15,
    }

    @property
    def indicator_type(self) -> IndicatorType:
        """Return attention indicator type."""
        return IndicatorType.ATTENTION

    @property
    def name(self) -> str:
        """Return detector name."""
        return "Attention Difficulty Indicator Detector"

    @property
    def description(self) -> str:
        """Return detector description."""
        return (
            "Analyzes behavioral patterns for potential attention-related indicators "
            "including session duration, response time variability, and abandonment rates."
        )

    async def analyze(self, student_data: StudentData) -> DetectorResult:
        """Analyze student data for attention indicators.

        Args:
            student_data: Aggregated student data.

        Returns:
            DetectorResult with risk score and evidence.
        """
        if not self.has_sufficient_data(student_data):
            self.logger.debug(
                "Insufficient data for attention analysis",
                extra={"student_id": student_data.student_id},
            )
            return DetectorResult.no_data(self.indicator_type)

        evidence_list: list[Evidence] = []
        component_scores: dict[str, float] = {}

        # Analyze session duration patterns
        duration_score, duration_evidence = self._analyze_session_duration(
            student_data
        )
        component_scores["session_duration"] = duration_score
        evidence_list.extend(duration_evidence)

        # Analyze answer time variability
        variability_score, variability_evidence = self._analyze_answer_time_variability(
            student_data
        )
        component_scores["answer_time_variability"] = variability_score
        evidence_list.extend(variability_evidence)

        # Analyze session abandonment
        abandonment_score, abandonment_evidence = self._analyze_session_abandonment(
            student_data
        )
        component_scores["session_abandonment"] = abandonment_score
        evidence_list.extend(abandonment_evidence)

        # Analyze within-session inconsistency
        inconsistency_score, inconsistency_evidence = (
            self._analyze_within_session_inconsistency(student_data)
        )
        component_scores["within_session_inconsistency"] = inconsistency_score
        evidence_list.extend(inconsistency_evidence)

        # Analyze distraction patterns
        distraction_score, distraction_evidence = self._analyze_distraction_patterns(
            student_data
        )
        component_scores["distraction_patterns"] = distraction_score
        evidence_list.extend(distraction_evidence)

        # Calculate weighted risk score
        risk_score = sum(
            component_scores[k] * self.WEIGHTS[k]
            for k in component_scores
            if k in self.WEIGHTS
        )
        risk_score = min(1.0, max(0.0, risk_score))

        # Calculate confidence
        confidence = self.calculate_confidence(
            sample_size=student_data.total_sessions,
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
            "Attention analysis complete",
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
            sample_size=student_data.total_sessions,
            analysis_summary=analysis_summary,
        )

    def _analyze_session_duration(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze session duration patterns.

        Very short sessions may indicate difficulty maintaining focus.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []

        sessions = [
            s for s in student_data.practice_sessions
            if s.time_spent_seconds > 0
        ]

        if len(sessions) < 3:
            return 0.0, []

        durations = [s.time_spent_seconds for s in sessions]
        avg_duration = sum(durations) / len(durations)

        # Count short sessions
        very_short = sum(1 for d in durations if d < self.MIN_SESSION_DURATION_SECONDS)
        short = sum(1 for d in durations if d < self.SHORT_SESSION_THRESHOLD)

        very_short_rate = very_short / len(sessions)
        short_rate = short / len(sessions)

        # Check for consistently short sessions
        if very_short_rate > 0.3:  # More than 30% very short
            evidence_list.append(
                self.create_evidence(
                    category="session_duration",
                    description=f"{very_short_rate:.1%} of sessions under 2 minutes",
                    data={
                        "very_short_count": very_short,
                        "total_sessions": len(sessions),
                        "rate": very_short_rate,
                        "avg_duration": avg_duration,
                    },
                    weight=0.9,
                )
            )
        elif short_rate > 0.5:  # More than 50% short
            evidence_list.append(
                self.create_evidence(
                    category="session_duration",
                    description=f"{short_rate:.1%} of sessions under 5 minutes",
                    data={
                        "short_count": short,
                        "total_sessions": len(sessions),
                        "rate": short_rate,
                        "avg_duration": avg_duration,
                    },
                    weight=0.7,
                )
            )

        # Score based on short session rates
        if very_short_rate > 0.5:
            score = 1.0
        elif very_short_rate > 0.3:
            score = 0.7
        elif short_rate > 0.6:
            score = 0.5
        elif short_rate > 0.4:
            score = 0.3
        else:
            score = 0.0

        return score, evidence_list

    def _analyze_answer_time_variability(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze variability in answer times.

        High variability in response times may indicate attention fluctuations.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []

        answer_times = [
            float(a.time_spent_seconds)
            for a in student_data.student_answers
            if a.time_spent_seconds > 0
        ]

        if len(answer_times) < 10:
            return 0.0, []

        # Calculate coefficient of variation
        cv = self._calculate_coefficient_of_variation(answer_times)

        # Calculate median and detect outliers
        sorted_times = sorted(answer_times)
        median_time = sorted_times[len(sorted_times) // 2]

        # Count extreme outliers (>3x or <0.2x median)
        high_outliers = sum(1 for t in answer_times if t > median_time * 3)
        low_outliers = sum(1 for t in answer_times if t < median_time * 0.2)
        total_outliers = high_outliers + low_outliers
        outlier_rate = total_outliers / len(answer_times)

        if cv > 0.8:  # Very high variability
            evidence_list.append(
                self.create_evidence(
                    category="answer_time_variability",
                    description=f"Very high response time variability (CV: {cv:.1%})",
                    data={
                        "coefficient_of_variation": cv,
                        "median_time": median_time,
                        "answer_count": len(answer_times),
                        "outlier_rate": outlier_rate,
                    },
                    weight=0.8,
                )
            )
        elif cv > 0.5:  # High variability
            evidence_list.append(
                self.create_evidence(
                    category="answer_time_variability",
                    description=f"High response time variability (CV: {cv:.1%})",
                    data={
                        "coefficient_of_variation": cv,
                        "median_time": median_time,
                    },
                    weight=0.6,
                )
            )

        # Score based on variability
        if cv > 1.0:
            score = 1.0
        elif cv > 0.8:
            score = 0.7
        elif cv > 0.5:
            score = 0.4
        elif cv > 0.3:
            score = 0.2
        else:
            score = 0.0

        return score, evidence_list

    def _analyze_session_abandonment(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze session abandonment patterns.

        High abandonment rates may indicate difficulty maintaining focus.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []

        sessions = student_data.practice_sessions
        if len(sessions) < 5:
            return 0.0, []

        abandoned = sum(1 for s in sessions if s.status == "abandoned")
        paused = sum(1 for s in sessions if s.status == "paused")
        incomplete_rate = (abandoned + paused) / len(sessions)
        abandon_rate = abandoned / len(sessions)

        # Also check for sessions with very few questions answered
        low_completion = sum(
            1 for s in sessions
            if s.questions_total > 0 and s.questions_answered / s.questions_total < 0.5
        )
        low_completion_rate = low_completion / len(sessions)

        if abandon_rate > 0.3:  # More than 30% abandoned
            evidence_list.append(
                self.create_evidence(
                    category="session_abandonment",
                    description=f"High session abandonment rate: {abandon_rate:.1%}",
                    data={
                        "abandoned_count": abandoned,
                        "total_sessions": len(sessions),
                        "rate": abandon_rate,
                    },
                    weight=0.9,
                )
            )
        elif incomplete_rate > 0.4:  # More than 40% incomplete
            evidence_list.append(
                self.create_evidence(
                    category="session_abandonment",
                    description=f"High incomplete session rate: {incomplete_rate:.1%}",
                    data={
                        "abandoned": abandoned,
                        "paused": paused,
                        "total_sessions": len(sessions),
                    },
                    weight=0.7,
                )
            )

        if low_completion_rate > 0.3:
            evidence_list.append(
                self.create_evidence(
                    category="session_abandonment",
                    description=f"{low_completion_rate:.1%} of sessions less than half completed",
                    data={
                        "low_completion_count": low_completion,
                        "rate": low_completion_rate,
                    },
                    weight=0.6,
                )
            )

        # Combined score
        score = min(1.0, abandon_rate * 2 + incomplete_rate * 0.5 + low_completion_rate)
        return score, evidence_list

    def _analyze_within_session_inconsistency(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze performance consistency within sessions.

        Attention difficulties can cause performance to degrade
        as sessions progress.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []
        sessions_with_decline = 0
        analyzed_sessions = 0

        for session in student_data.practice_sessions:
            # Get answers for this session
            session_answers = [
                a for a in student_data.student_answers
                if hasattr(a, 'question') and hasattr(a.question, 'session_id')
                and a.question.session_id == session.id
            ]

            if len(session_answers) < 6:  # Need enough questions to analyze
                continue

            analyzed_sessions += 1

            # Split into first half and second half
            half = len(session_answers) // 2
            first_half = session_answers[:half]
            second_half = session_answers[half:]

            # Get evaluations for each half
            first_correct = sum(
                1 for a in first_half
                if any(e.answer_id == a.id and e.is_correct
                       for e in student_data.evaluation_results)
            )
            second_correct = sum(
                1 for a in second_half
                if any(e.answer_id == a.id and e.is_correct
                       for e in student_data.evaluation_results)
            )

            first_accuracy = first_correct / len(first_half) if first_half else 0
            second_accuracy = second_correct / len(second_half) if second_half else 0

            # Check for significant decline
            if first_accuracy > 0.5 and second_accuracy < first_accuracy - 0.2:
                sessions_with_decline += 1

        if analyzed_sessions < 3:
            return 0.0, []

        decline_rate = sessions_with_decline / analyzed_sessions

        if decline_rate > 0.4:  # More than 40% of sessions show decline
            evidence_list.append(
                self.create_evidence(
                    category="within_session_inconsistency",
                    description=f"Performance decline in {decline_rate:.1%} of sessions",
                    data={
                        "sessions_with_decline": sessions_with_decline,
                        "analyzed_sessions": analyzed_sessions,
                        "rate": decline_rate,
                    },
                    weight=0.7,
                )
            )

        score = min(1.0, decline_rate * 2)
        return score, evidence_list

    def _analyze_distraction_patterns(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze patterns suggesting distraction.

        Look for very long pauses between answers that might indicate
        the student was distracted.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []

        answer_times = [
            float(a.time_spent_seconds)
            for a in student_data.student_answers
            if a.time_spent_seconds > 0
        ]

        if len(answer_times) < 10:
            return 0.0, []

        # Calculate median and identify potential distraction episodes
        sorted_times = sorted(answer_times)
        median_time = sorted_times[len(sorted_times) // 2]

        # Very long responses (>5x median) might indicate distraction
        distraction_threshold = max(median_time * 5, 180)  # At least 3 minutes
        potential_distractions = sum(
            1 for t in answer_times
            if t > distraction_threshold
        )

        distraction_rate = potential_distractions / len(answer_times)

        if distraction_rate > 0.1:  # More than 10% potential distractions
            evidence_list.append(
                self.create_evidence(
                    category="distraction_patterns",
                    description=f"Potential distraction episodes in {distraction_rate:.1%} of responses",
                    data={
                        "distraction_count": potential_distractions,
                        "total_answers": len(answer_times),
                        "threshold_seconds": distraction_threshold,
                        "median_time": median_time,
                    },
                    weight=0.6,
                )
            )

        score = min(1.0, distraction_rate * 5)
        return score, evidence_list

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
            return "No significant attention difficulty indicators detected."

        elevated_components = [
            k.replace("_", " ") for k, v in component_scores.items() if v >= 0.3
        ]

        if threshold_level == ThresholdLevel.HIGH:
            return (
                f"Elevated attention difficulty indicators in {len(elevated_components)} areas: "
                f"{', '.join(elevated_components)}. "
                "Professional evaluation recommended."
            )
        elif threshold_level == ThresholdLevel.ELEVATED:
            return (
                f"Moderate attention indicators in: {', '.join(elevated_components)}. "
                "Continued monitoring suggested."
            )
        else:
            return (
                f"Mild indicators in: {', '.join(elevated_components)}. "
                "Continue regular monitoring."
            )

# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Auditory processing difficulty indicator detector.

This module provides detection of potential auditory processing
difficulty indicators based on student performance patterns.

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


class AuditoryDetector(BaseDetector):
    """Detector for potential auditory processing difficulty indicators.

    Analyzes student performance patterns to identify signs that may
    indicate auditory processing difficulties.

    Indicators analyzed:
    - Phonetic confusion patterns (sound-alike errors)
    - Performance differences on verbal vs. visual content
    - Difficulty following sequential verbal instructions
    - Homophones and similar-sounding word confusions

    IMPORTANT: High scores indicate need for professional evaluation,
    not a diagnosis of auditory processing disorder.
    """

    # Common phonetic confusion pairs (sounds that are often confused)
    PHONETIC_CONFUSIONS = [
        # Voiced/unvoiced pairs
        ("b", "p"),
        ("d", "t"),
        ("g", "k"),
        ("v", "f"),
        ("z", "s"),
        # Similar sounds
        ("m", "n"),
        ("sh", "ch"),
        ("th", "f"),
        ("th", "s"),
    ]

    # Common homophones (words that sound alike but are spelled differently)
    HOMOPHONES = [
        ("their", "there", "they're"),
        ("your", "you're"),
        ("its", "it's"),
        ("to", "too", "two"),
        ("hear", "here"),
        ("know", "no"),
        ("write", "right"),
        ("break", "brake"),
        ("piece", "peace"),
        ("flour", "flower"),
    ]

    # Weights for different indicator categories
    WEIGHTS = {
        "phonetic_confusion": 0.30,
        "homophone_confusion": 0.25,
        "verbal_vs_visual": 0.25,
        "sequential_instruction": 0.20,
    }

    @property
    def indicator_type(self) -> IndicatorType:
        """Return auditory indicator type."""
        return IndicatorType.AUDITORY

    @property
    def name(self) -> str:
        """Return detector name."""
        return "Auditory Processing Indicator Detector"

    @property
    def description(self) -> str:
        """Return detector description."""
        return (
            "Analyzes performance patterns for potential auditory processing indicators "
            "including phonetic confusions and verbal vs. visual performance differences."
        )

    async def analyze(self, student_data: StudentData) -> DetectorResult:
        """Analyze student data for auditory processing indicators.

        Args:
            student_data: Aggregated student data.

        Returns:
            DetectorResult with risk score and evidence.
        """
        if not self.has_sufficient_data(student_data):
            self.logger.debug(
                "Insufficient data for auditory analysis",
                extra={"student_id": student_data.student_id},
            )
            return DetectorResult.no_data(self.indicator_type)

        evidence_list: list[Evidence] = []
        component_scores: dict[str, float] = {}

        # Analyze phonetic confusions
        phonetic_score, phonetic_evidence = self._analyze_phonetic_confusion(
            student_data
        )
        component_scores["phonetic_confusion"] = phonetic_score
        evidence_list.extend(phonetic_evidence)

        # Analyze homophone confusions
        homophone_score, homophone_evidence = self._analyze_homophone_confusion(
            student_data
        )
        component_scores["homophone_confusion"] = homophone_score
        evidence_list.extend(homophone_evidence)

        # Analyze verbal vs visual performance
        verbal_visual_score, verbal_visual_evidence = (
            self._analyze_verbal_vs_visual_performance(student_data)
        )
        component_scores["verbal_vs_visual"] = verbal_visual_score
        evidence_list.extend(verbal_visual_evidence)

        # Analyze sequential instruction following
        sequential_score, sequential_evidence = (
            self._analyze_sequential_instruction_performance(student_data)
        )
        component_scores["sequential_instruction"] = sequential_score
        evidence_list.extend(sequential_evidence)

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
            "Auditory analysis complete",
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

    def _analyze_phonetic_confusion(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze answers for phonetic confusion patterns.

        Phonetic confusions occur when students substitute sounds that
        are acoustically similar.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []
        confusion_instances: list[dict] = []
        total_text_answers = 0

        for answer in student_data.student_answers:
            answer_text = self._extract_answer_text(answer.answer)
            if not answer_text or len(answer_text) < 3:
                continue

            total_text_answers += 1

            # Check for phonetic confusion patterns
            for sound1, sound2 in self.PHONETIC_CONFUSIONS:
                # Look for potential substitutions
                instances = self._find_phonetic_substitutions(
                    answer_text, sound1, sound2
                )
                if instances:
                    confusion_instances.extend(instances)

        if total_text_answers == 0:
            return 0.0, []

        confusion_rate = len(confusion_instances) / total_text_answers

        if confusion_rate > 0.1:  # More than 10% show confusion patterns
            evidence_list.append(
                self.create_evidence(
                    category="phonetic_confusion",
                    description=f"Phonetic confusion patterns in {confusion_rate:.1%} of answers",
                    data={
                        "confusion_count": len(confusion_instances),
                        "total_text_answers": total_text_answers,
                        "rate": confusion_rate,
                        "examples": confusion_instances[:5],
                    },
                    weight=0.8,
                )
            )

        score = min(1.0, confusion_rate * 5)  # Scale: 20% = max score
        return score, evidence_list

    def _analyze_homophone_confusion(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze answers for homophone confusions.

        Homophone errors (their/there/they're) can indicate difficulty
        processing sounds into correct spellings.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []
        homophone_errors: list[dict] = []
        total_relevant_answers = 0

        for answer in student_data.student_answers:
            answer_text = self._extract_answer_text(answer.answer)
            if not answer_text:
                continue

            # Check each homophone group
            for homophone_group in self.HOMOPHONES:
                used_homophones = [
                    h for h in homophone_group
                    if h.lower() in answer_text.lower()
                ]

                if used_homophones:
                    total_relevant_answers += 1

                    # Check if word is used incorrectly by looking at evaluations
                    evaluation = next(
                        (e for e in student_data.evaluation_results
                         if e.answer_id == answer.id),
                        None
                    )

                    if evaluation and not evaluation.is_correct:
                        # Check if misconception relates to homophone
                        misconceptions = evaluation.misconceptions or []
                        for misc in misconceptions:
                            if isinstance(misc, dict):
                                misc_text = str(misc.get("description", "")).lower()
                                if any(h in misc_text for h in homophone_group):
                                    homophone_errors.append({
                                        "group": homophone_group,
                                        "used": used_homophones,
                                    })

        if total_relevant_answers == 0:
            return 0.0, []

        error_rate = len(homophone_errors) / max(1, total_relevant_answers)

        if error_rate > 0.15:  # More than 15% homophone errors
            evidence_list.append(
                self.create_evidence(
                    category="homophone_confusion",
                    description=f"Homophone confusion in {error_rate:.1%} of relevant answers",
                    data={
                        "error_count": len(homophone_errors),
                        "total_relevant": total_relevant_answers,
                        "rate": error_rate,
                    },
                    weight=0.7,
                )
            )

        score = min(1.0, error_rate * 4)  # Scale: 25% = max score
        return score, evidence_list

    def _analyze_verbal_vs_visual_performance(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze performance difference between verbal and visual content.

        Students with auditory processing issues may perform better
        on visual content than on text/verbal content.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []

        verbal_correct = 0
        verbal_total = 0
        visual_correct = 0
        visual_total = 0

        for answer in student_data.student_answers:
            # Determine if question was verbal/text-heavy or visual
            is_verbal = self._is_verbal_question(answer)

            evaluation = next(
                (e for e in student_data.evaluation_results
                 if e.answer_id == answer.id),
                None
            )

            if evaluation:
                if is_verbal:
                    verbal_total += 1
                    if evaluation.is_correct:
                        verbal_correct += 1
                else:
                    visual_total += 1
                    if evaluation.is_correct:
                        visual_correct += 1

        if verbal_total < 5 or visual_total < 5:
            return 0.0, []

        verbal_accuracy = verbal_correct / verbal_total
        visual_accuracy = visual_correct / visual_total

        # Check for significant performance gap
        performance_gap = visual_accuracy - verbal_accuracy

        if performance_gap > 0.2:  # Visual is 20%+ better
            evidence_list.append(
                self.create_evidence(
                    category="verbal_vs_visual",
                    description=f"Visual performance {performance_gap:.1%} higher than verbal",
                    data={
                        "verbal_accuracy": verbal_accuracy,
                        "visual_accuracy": visual_accuracy,
                        "gap": performance_gap,
                        "verbal_sample": verbal_total,
                        "visual_sample": visual_total,
                    },
                    weight=0.8,
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

    def _analyze_sequential_instruction_performance(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze performance on multi-step questions.

        Difficulty following sequential instructions is an auditory
        processing indicator.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []

        single_step_correct = 0
        single_step_total = 0
        multi_step_correct = 0
        multi_step_total = 0

        for answer in student_data.student_answers:
            # Determine if question has multiple steps
            is_multi_step = self._is_multi_step_question(answer)

            evaluation = next(
                (e for e in student_data.evaluation_results
                 if e.answer_id == answer.id),
                None
            )

            if evaluation:
                if is_multi_step:
                    multi_step_total += 1
                    if evaluation.is_correct:
                        multi_step_correct += 1
                else:
                    single_step_total += 1
                    if evaluation.is_correct:
                        single_step_correct += 1

        if multi_step_total < 5 or single_step_total < 5:
            return 0.0, []

        single_accuracy = single_step_correct / single_step_total
        multi_accuracy = multi_step_correct / multi_step_total

        # Check for significant drop in multi-step performance
        performance_drop = single_accuracy - multi_accuracy

        if performance_drop > 0.2:  # Multi-step is 20%+ worse
            evidence_list.append(
                self.create_evidence(
                    category="sequential_instruction",
                    description=f"Multi-step performance {performance_drop:.1%} lower than single-step",
                    data={
                        "single_step_accuracy": single_accuracy,
                        "multi_step_accuracy": multi_accuracy,
                        "drop": performance_drop,
                        "single_sample": single_step_total,
                        "multi_sample": multi_step_total,
                    },
                    weight=0.7,
                )
            )

        # Score based on performance drop
        if performance_drop <= 0.1:
            score = 0.0
        elif performance_drop <= 0.2:
            score = 0.3
        elif performance_drop <= 0.35:
            score = 0.6
        else:
            score = min(1.0, 0.6 + (performance_drop - 0.35) * 2)

        return score, evidence_list

    def _find_phonetic_substitutions(
        self,
        text: str,
        sound1: str,
        sound2: str,
    ) -> list[dict]:
        """Find potential phonetic substitution patterns.

        Args:
            text: Text to analyze.
            sound1: First sound in the pair.
            sound2: Second sound in the pair.

        Returns:
            List of potential substitution instances.
        """
        instances = []
        text_lower = text.lower()

        # Check for common substitution patterns
        # This is a heuristic - real detection would need more context

        # Count occurrences of each sound
        count1 = text_lower.count(sound1)
        count2 = text_lower.count(sound2)

        # Asymmetric usage in long text might indicate substitution
        if len(text) > 50:
            if count1 > 0 and count2 == 0:
                instances.append({
                    "pattern": f"Only {sound1} used, never {sound2}",
                    "sound1": sound1,
                    "sound2": sound2,
                })
            elif count2 > 0 and count1 == 0:
                instances.append({
                    "pattern": f"Only {sound2} used, never {sound1}",
                    "sound1": sound1,
                    "sound2": sound2,
                })

        return instances

    def _extract_answer_text(self, answer_data: dict) -> str:
        """Extract text content from answer data."""
        if isinstance(answer_data, str):
            return answer_data

        if isinstance(answer_data, dict):
            for key in ["text", "answer", "content", "response", "value"]:
                if key in answer_data and isinstance(answer_data[key], str):
                    return answer_data[key]

        return ""

    def _is_verbal_question(self, answer) -> bool:
        """Determine if question is verbal/text-heavy.

        Args:
            answer: Student answer object.

        Returns:
            True if question is verbal/text-based.
        """
        # This is a heuristic - in real implementation would check question metadata
        answer_text = self._extract_answer_text(answer.answer)

        # Long text answers suggest text-heavy questions
        if answer_text and len(answer_text) > 30:
            return True

        # Check if answer contains mostly text (not numbers)
        if answer_text:
            text_ratio = sum(c.isalpha() for c in answer_text) / max(1, len(answer_text))
            return text_ratio > 0.6

        return False

    def _is_multi_step_question(self, answer) -> bool:
        """Determine if question requires multiple steps.

        Args:
            answer: Student answer object.

        Returns:
            True if question is multi-step.
        """
        # This is a heuristic - would check question metadata in real implementation
        answer_text = self._extract_answer_text(answer.answer)

        # Multiple parts in answer might indicate multi-step
        if answer_text:
            # Check for numbered steps, semicolons, "and then", etc.
            multi_step_indicators = [
                r"\d+\)", r"\d+\.", r";", r"\band\s+then\b",
                r"\bfirst\b.*\bthen\b", r"\bstep\s+\d+",
            ]
            for pattern in multi_step_indicators:
                if re.search(pattern, answer_text, re.IGNORECASE):
                    return True

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
            return "No significant auditory processing indicators detected."

        elevated_components = [
            k.replace("_", " ") for k, v in component_scores.items() if v >= 0.3
        ]

        if threshold_level == ThresholdLevel.HIGH:
            return (
                f"Elevated auditory processing indicators in {len(elevated_components)} areas: "
                f"{', '.join(elevated_components)}. "
                "Professional evaluation recommended."
            )
        elif threshold_level == ThresholdLevel.ELEVATED:
            return (
                f"Moderate auditory indicators in: {', '.join(elevated_components)}. "
                "Continued monitoring suggested."
            )
        else:
            return (
                f"Mild indicators in: {', '.join(elevated_components)}. "
                "Continue regular monitoring."
            )

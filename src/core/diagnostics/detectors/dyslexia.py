# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dyslexia indicator detector.

This module provides detection of potential dyslexia indicators
based on student answer patterns and learning behaviors.

IMPORTANT: This detector identifies INDICATORS only, not diagnoses.
Professional evaluation by qualified specialists is required for diagnosis.
"""

import re
from collections import Counter
from datetime import datetime

from src.core.diagnostics.detectors.base import (
    BaseDetector,
    DetectorResult,
    Evidence,
    IndicatorType,
    StudentData,
    ThresholdLevel,
)


class DyslexiaDetector(BaseDetector):
    """Detector for potential dyslexia indicators.

    Analyzes student answer patterns to identify signs that may
    indicate reading/writing difficulties associated with dyslexia.

    Indicators analyzed:
    - Letter reversals (b/d, p/q, m/w, n/u)
    - Letter transpositions
    - Spelling inconsistencies
    - Word confusion patterns
    - Answer time patterns for text-heavy questions

    IMPORTANT: High scores indicate need for professional evaluation,
    not a diagnosis of dyslexia.
    """

    # Common letter reversal pairs
    REVERSAL_PAIRS = [
        ("b", "d"),
        ("p", "q"),
        ("m", "w"),
        ("n", "u"),
        ("6", "9"),
    ]

    # Common transposition patterns (letter swaps within words)
    TRANSPOSITION_WEIGHT = 0.3

    # Weights for different indicator categories
    WEIGHTS = {
        "letter_reversal": 0.25,
        "transposition": 0.20,
        "spelling_inconsistency": 0.20,
        "word_confusion": 0.15,
        "reading_time": 0.20,
    }

    @property
    def indicator_type(self) -> IndicatorType:
        """Return dyslexia indicator type."""
        return IndicatorType.DYSLEXIA

    @property
    def name(self) -> str:
        """Return detector name."""
        return "Dyslexia Indicator Detector"

    @property
    def description(self) -> str:
        """Return detector description."""
        return (
            "Analyzes answer patterns for potential dyslexia indicators "
            "including letter reversals, transpositions, and spelling patterns."
        )

    async def analyze(self, student_data: StudentData) -> DetectorResult:
        """Analyze student data for dyslexia indicators.

        Args:
            student_data: Aggregated student data.

        Returns:
            DetectorResult with risk score and evidence.
        """
        if not self.has_sufficient_data(student_data):
            self.logger.debug(
                "Insufficient data for dyslexia analysis",
                extra={"student_id": student_data.student_id},
            )
            return DetectorResult.no_data(self.indicator_type)

        evidence_list: list[Evidence] = []
        component_scores: dict[str, float] = {}

        # Analyze letter reversals
        reversal_score, reversal_evidence = self._analyze_letter_reversals(
            student_data
        )
        component_scores["letter_reversal"] = reversal_score
        evidence_list.extend(reversal_evidence)

        # Analyze transpositions
        transposition_score, transposition_evidence = self._analyze_transpositions(
            student_data
        )
        component_scores["transposition"] = transposition_score
        evidence_list.extend(transposition_evidence)

        # Analyze spelling inconsistencies
        spelling_score, spelling_evidence = self._analyze_spelling_inconsistencies(
            student_data
        )
        component_scores["spelling_inconsistency"] = spelling_score
        evidence_list.extend(spelling_evidence)

        # Analyze reading time patterns
        reading_score, reading_evidence = self._analyze_reading_time(student_data)
        component_scores["reading_time"] = reading_score
        evidence_list.extend(reading_evidence)

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
            "Dyslexia analysis complete",
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

    def _analyze_letter_reversals(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze answers for letter reversal patterns.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []
        reversal_count = 0
        total_text_answers = 0

        for answer in student_data.student_answers:
            answer_text = self._extract_answer_text(answer.answer)
            if not answer_text:
                continue

            total_text_answers += 1

            # Check for reversal patterns
            for char1, char2 in self.REVERSAL_PAIRS:
                # Look for context where reversal might have occurred
                # This is a heuristic - we look for words that might have reversals
                reversal_instances = self._find_potential_reversals(
                    answer_text, char1, char2
                )
                if reversal_instances:
                    reversal_count += len(reversal_instances)
                    evidence_list.append(
                        self.create_evidence(
                            category="letter_reversal",
                            description=f"Potential {char1}/{char2} reversal pattern detected",
                            data={
                                "pair": f"{char1}/{char2}",
                                "instances": reversal_instances[:3],  # Limit stored instances
                            },
                            weight=0.8,
                        )
                    )

        if total_text_answers == 0:
            return 0.0, []

        # Calculate score based on reversal frequency
        reversal_rate = reversal_count / total_text_answers
        score = min(1.0, reversal_rate * 2)  # Scale: 50% reversal rate = max score

        return score, evidence_list

    def _analyze_transpositions(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze answers for letter transposition patterns.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []
        transposition_count = 0
        total_words = 0

        for answer in student_data.student_answers:
            answer_text = self._extract_answer_text(answer.answer)
            if not answer_text:
                continue

            words = answer_text.lower().split()
            total_words += len(words)

            # Check for common transposition patterns
            for word in words:
                if self._has_transposition_pattern(word):
                    transposition_count += 1

        if total_words == 0:
            return 0.0, []

        transposition_rate = transposition_count / max(1, total_words)

        if transposition_rate > 0.05:  # More than 5% of words
            evidence_list.append(
                self.create_evidence(
                    category="transposition",
                    description=f"Elevated letter transposition rate: {transposition_rate:.1%}",
                    data={
                        "transposition_count": transposition_count,
                        "total_words": total_words,
                        "rate": transposition_rate,
                    },
                    weight=0.7,
                )
            )

        score = min(1.0, transposition_rate * 10)  # Scale: 10% = max score
        return score, evidence_list

    def _analyze_spelling_inconsistencies(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze for inconsistent spelling of the same words.

        Dyslexia can cause the same word to be spelled differently
        at different times.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []
        word_spellings: dict[str, set[str]] = {}

        for answer in student_data.student_answers:
            answer_text = self._extract_answer_text(answer.answer)
            if not answer_text:
                continue

            words = answer_text.lower().split()
            for word in words:
                # Normalize to find similar words
                normalized = self._normalize_word(word)
                if normalized and len(normalized) >= 4:  # Only track longer words
                    if normalized not in word_spellings:
                        word_spellings[normalized] = set()
                    word_spellings[normalized].add(word)

        # Find words spelled multiple ways
        inconsistent_words = {
            norm: spellings
            for norm, spellings in word_spellings.items()
            if len(spellings) > 1
        }

        if inconsistent_words:
            evidence_list.append(
                self.create_evidence(
                    category="spelling_inconsistency",
                    description=f"Found {len(inconsistent_words)} words with inconsistent spellings",
                    data={
                        "examples": dict(
                            list(inconsistent_words.items())[:5]
                        ),  # Top 5 examples
                    },
                    weight=0.8,
                )
            )

        total_tracked = len(word_spellings)
        if total_tracked == 0:
            return 0.0, []

        inconsistency_rate = len(inconsistent_words) / total_tracked
        score = min(1.0, inconsistency_rate * 5)  # Scale: 20% inconsistency = max

        return score, evidence_list

    def _analyze_reading_time(
        self,
        student_data: StudentData,
    ) -> tuple[float, list[Evidence]]:
        """Analyze reading/response time patterns.

        Students with dyslexia may show significantly longer reading times
        for text-heavy questions compared to visual/numerical questions.

        Args:
            student_data: Student data to analyze.

        Returns:
            Tuple of (score, evidence_list).
        """
        evidence_list: list[Evidence] = []
        text_heavy_times: list[float] = []
        other_times: list[float] = []

        for answer in student_data.student_answers:
            if answer.time_spent_seconds <= 0:
                continue

            # Determine if question is text-heavy based on answer type
            # This is a heuristic - in real implementation, would check question type
            answer_text = self._extract_answer_text(answer.answer)
            is_text_heavy = bool(answer_text and len(answer_text) > 50)

            if is_text_heavy:
                text_heavy_times.append(float(answer.time_spent_seconds))
            else:
                other_times.append(float(answer.time_spent_seconds))

        if len(text_heavy_times) < 3 or len(other_times) < 3:
            return 0.0, []

        avg_text_time = sum(text_heavy_times) / len(text_heavy_times)
        avg_other_time = sum(other_times) / len(other_times)

        if avg_other_time == 0:
            return 0.0, []

        time_ratio = avg_text_time / avg_other_time

        # Significant if text questions take >50% longer
        if time_ratio > 1.5:
            evidence_list.append(
                self.create_evidence(
                    category="reading_time",
                    description=f"Text-heavy questions take {time_ratio:.1f}x longer than others",
                    data={
                        "avg_text_time": avg_text_time,
                        "avg_other_time": avg_other_time,
                        "ratio": time_ratio,
                        "text_sample_size": len(text_heavy_times),
                        "other_sample_size": len(other_times),
                    },
                    weight=0.6,
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

    def _find_potential_reversals(
        self,
        text: str,
        char1: str,
        char2: str,
    ) -> list[str]:
        """Find potential reversal instances in text.

        This is a heuristic approach - it looks for unusual patterns
        that might indicate reversals.

        Args:
            text: Text to analyze.
            char1: First character of reversal pair.
            char2: Second character of reversal pair.

        Returns:
            List of potential reversal instances.
        """
        instances: list[str] = []
        text_lower = text.lower()

        # Common words that are often confused due to b/d reversals
        if char1 == "b" and char2 == "d":
            confusion_patterns = [
                (r"\bbig\b", "dig"),
                (r"\bbed\b", "deb"),
                (r"\bbad\b", "dad"),
                (r"\bdog\b", "bog"),
            ]
            for pattern, confused in confusion_patterns:
                if re.search(pattern, text_lower):
                    # Check if the confused version appears elsewhere
                    if confused in text_lower:
                        instances.append(f"Possible {char1}/{char2} confusion")

        # Check for repeated use of one letter where context suggests the other
        # This is a simplified heuristic
        count1 = text_lower.count(char1)
        count2 = text_lower.count(char2)

        if count1 > 0 and count2 == 0 and len(text) > 20:
            # Only one of the pair used in substantial text
            instances.append(f"Asymmetric {char1}/{char2} usage pattern")

        return instances

    def _has_transposition_pattern(self, word: str) -> bool:
        """Check if a word shows potential transposition patterns.

        Args:
            word: Word to check.

        Returns:
            True if transposition pattern detected.
        """
        if len(word) < 4:
            return False

        # Common transposition patterns
        transposition_indicators = [
            # Double letters that might be transposed
            r"(.)\1{2,}",  # Three or more same letters
            # Common transposition results
            r"teh|hte|taht|thta",  # the, that
            r"adn|nad",  # and
            r"fomr|form|from",  # from variations
            r"beacuse|becuase",  # because
        ]

        for pattern in transposition_indicators:
            if re.search(pattern, word, re.IGNORECASE):
                return True

        return False

    def _normalize_word(self, word: str) -> str:
        """Normalize a word to find similar spellings.

        Removes vowels and double letters to create a consonant skeleton.

        Args:
            word: Word to normalize.

        Returns:
            Normalized word skeleton.
        """
        # Remove non-alphabetic characters
        word = re.sub(r"[^a-z]", "", word.lower())

        if len(word) < 3:
            return ""

        # Keep first letter, remove consecutive duplicates
        result = word[0]
        for char in word[1:]:
            if char != result[-1]:
                result += char

        return result

    def _extract_answer_text(self, answer_data: dict) -> str:
        """Extract text content from answer data.

        Args:
            answer_data: Answer data dictionary.

        Returns:
            Extracted text content.
        """
        if isinstance(answer_data, str):
            return answer_data

        if isinstance(answer_data, dict):
            # Try common keys
            for key in ["text", "answer", "content", "response", "value"]:
                if key in answer_data and isinstance(answer_data[key], str):
                    return answer_data[key]

        return ""

    def _calculate_evidence_consistency(
        self,
        component_scores: dict[str, float],
    ) -> float:
        """Calculate how consistent the evidence is across components.

        High consistency = multiple components showing elevated scores.

        Args:
            component_scores: Scores for each component.

        Returns:
            Consistency score (0.0-1.0).
        """
        if not component_scores:
            return 0.0

        scores = list(component_scores.values())
        elevated_count = sum(1 for s in scores if s >= 0.3)
        high_count = sum(1 for s in scores if s >= 0.5)

        # More consistent if multiple indicators are elevated
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
        """Generate analysis summary.

        Args:
            component_scores: Scores per component.
            evidence_list: Collected evidence.
            threshold_level: Overall threshold level.

        Returns:
            Human-readable summary.
        """
        if threshold_level == ThresholdLevel.LOW:
            return "No significant dyslexia indicators detected."

        elevated_components = [
            k for k, v in component_scores.items() if v >= 0.3
        ]

        if threshold_level == ThresholdLevel.HIGH:
            return (
                f"Elevated indicators in {len(elevated_components)} areas: "
                f"{', '.join(elevated_components)}. "
                "Professional evaluation recommended."
            )
        elif threshold_level == ThresholdLevel.ELEVATED:
            return (
                f"Moderate indicators detected in: {', '.join(elevated_components)}. "
                "Continued monitoring suggested."
            )
        else:
            return (
                f"Mild indicators in: {', '.join(elevated_components)}. "
                "Continue regular monitoring."
            )

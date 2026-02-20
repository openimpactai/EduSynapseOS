# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check Age Appropriateness Tool.

Verifies content is appropriate for target grade level.
"""

import logging
import re
from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class CheckAgeAppropriatenessTool(BaseTool):
    """Check age appropriateness of content.

    Evaluates vocabulary level, concept complexity, and
    cultural appropriateness for the target grade level.

    Example usage by agent:
        - "Is this vocabulary appropriate for grade 3?"
        - "Check if content is suitable for middle school"
        - "Verify age appropriateness"
    """

    # Grade level vocabulary guidelines
    GRADE_GUIDELINES = {
        "1-3": {
            "max_word_length": 8,
            "max_sentence_length": 12,
            "max_syllables": 3,
            "concepts": ["concrete", "familiar", "observable"],
            "avoid": ["abstract", "technical", "complex"],
        },
        "4-6": {
            "max_word_length": 10,
            "max_sentence_length": 18,
            "max_syllables": 4,
            "concepts": ["some_abstract", "cause_effect", "comparison"],
            "avoid": ["highly_technical", "philosophical"],
        },
        "7-9": {
            "max_word_length": 12,
            "max_sentence_length": 25,
            "max_syllables": 5,
            "concepts": ["abstract", "analysis", "inference"],
            "avoid": ["advanced_technical", "mature_themes"],
        },
        "10-12": {
            "max_word_length": 15,
            "max_sentence_length": 30,
            "max_syllables": 6,
            "concepts": ["complex", "critical_thinking", "synthesis"],
            "avoid": ["inappropriate_content"],
        },
    }

    @property
    def name(self) -> str:
        return "check_age_appropriateness"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "check_age_appropriateness",
                "description": (
                    "Check if content is appropriate for the target grade level. "
                    "Evaluates vocabulary, complexity, and cultural appropriateness."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content_type": {
                            "type": "string",
                            "description": "Type of content being checked",
                        },
                        "ai_content": {
                            "type": "object",
                            "description": "Content to evaluate",
                        },
                        "grade_level": {
                            "type": "integer",
                            "description": "Target grade level (1-12)",
                        },
                        "language": {
                            "type": "string",
                            "description": "Content language code",
                        },
                    },
                    "required": ["content_type", "ai_content", "grade_level"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute age appropriateness check."""
        content_type = params.get("content_type", "")
        ai_content = params.get("ai_content", {})
        grade_level = params.get("grade_level", 5)
        language = params.get("language", context.language or "en")

        if not ai_content:
            return ToolResult(
                success=False,
                data={"message": "Content is required"},
                error="Missing required parameter: ai_content",
            )

        try:
            # Get grade band
            grade_band = self._get_grade_band(grade_level)
            guidelines = self.GRADE_GUIDELINES[grade_band]

            # Extract all text from content
            text_content = self._extract_text(ai_content)

            # Analyze vocabulary
            vocab_analysis = self._analyze_vocabulary(text_content, guidelines)

            # Analyze sentence complexity
            sentence_analysis = self._analyze_sentences(text_content, guidelines)

            # Check for inappropriate content
            content_flags = self._check_content_flags(text_content)

            # Calculate overall score
            vocab_score = vocab_analysis["score"]
            sentence_score = sentence_analysis["score"]
            content_score = 100 if not content_flags else max(0, 100 - len(content_flags) * 25)

            overall_score = (vocab_score * 0.4 + sentence_score * 0.4 + content_score * 0.2)

            # Compile issues
            issues = []
            for word in vocab_analysis.get("difficult_words", [])[:5]:
                issues.append({
                    "type": "vocabulary",
                    "severity": "minor",
                    "description": f"Word '{word}' may be too complex for grade {grade_level}",
                    "suggestion": f"Consider using simpler alternative",
                })

            if sentence_analysis.get("long_sentences"):
                issues.append({
                    "type": "complexity",
                    "severity": "minor",
                    "description": f"Found {sentence_analysis['long_sentences']} long sentences",
                    "suggestion": "Break into shorter sentences for clarity",
                })

            for flag in content_flags:
                issues.append({
                    "type": "content",
                    "severity": "major",
                    "description": flag,
                    "suggestion": "Review and modify content",
                })

            # Determine recommendation
            if overall_score >= 80:
                recommendation = "appropriate"
            elif overall_score >= 60:
                recommendation = "needs_minor_adjustments"
            else:
                recommendation = "needs_revision"

            logger.info(
                "Age appropriateness check: grade=%d, score=%.0f%%",
                grade_level,
                overall_score,
            )

            return ToolResult(
                success=True,
                data={
                    "appropriateness_score": overall_score,
                    "recommendation": recommendation,
                    "grade_level": grade_level,
                    "grade_band": grade_band,
                    "vocabulary_score": vocab_score,
                    "sentence_score": sentence_score,
                    "content_score": content_score,
                    "issues": issues,
                    "difficult_words": vocab_analysis.get("difficult_words", [])[:10],
                    "statistics": {
                        "avg_word_length": vocab_analysis.get("avg_word_length", 0),
                        "avg_sentence_length": sentence_analysis.get("avg_length", 0),
                        "total_words": vocab_analysis.get("total_words", 0),
                    },
                    "message": (
                        f"Age appropriateness: {overall_score:.0f}% ({recommendation}). "
                        f"Vocabulary: {vocab_score:.0f}%, Sentences: {sentence_score:.0f}%"
                    ),
                },
            )

        except Exception as e:
            logger.exception("Error checking age appropriateness")
            return ToolResult(
                success=False,
                data={"message": f"Failed to check appropriateness: {e}"},
                error=str(e),
            )

    def _get_grade_band(self, grade: int) -> str:
        """Get grade band from grade level."""
        if grade <= 3:
            return "1-3"
        elif grade <= 6:
            return "4-6"
        elif grade <= 9:
            return "7-9"
        else:
            return "10-12"

    def _extract_text(self, ai_content: dict) -> str:
        """Extract all text from content."""
        texts = []

        # Questions
        for q in ai_content.get("questions", []):
            texts.append(q.get("question", ""))
            texts.extend(q.get("answers", []))
            texts.append(q.get("explanation", ""))

        # Cards
        for c in ai_content.get("cards", []):
            texts.append(c.get("term", ""))
            texts.append(c.get("definition", ""))
            texts.append(c.get("example", ""))

        # Statements
        for s in ai_content.get("statements", []):
            texts.append(s.get("statement", ""))
            texts.append(s.get("explanation", ""))

        # Panels
        for p in ai_content.get("panels", []):
            texts.append(p.get("title", ""))
            texts.append(p.get("content", ""))

        # General
        texts.append(ai_content.get("title", ""))
        texts.append(ai_content.get("description", ""))

        return " ".join(filter(None, texts))

    def _analyze_vocabulary(self, text: str, guidelines: dict) -> dict:
        """Analyze vocabulary complexity."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        if not words:
            return {"score": 100, "difficult_words": [], "avg_word_length": 0, "total_words": 0}

        max_length = guidelines["max_word_length"]

        # Find difficult words
        difficult_words = [w for w in set(words) if len(w) > max_length]

        # Calculate average word length
        avg_length = sum(len(w) for w in words) / len(words)

        # Calculate score
        difficult_ratio = len(difficult_words) / len(set(words)) if words else 0
        score = max(0, 100 - difficult_ratio * 200)

        # Adjust for average length
        if avg_length > max_length * 0.8:
            score = max(0, score - 10)

        return {
            "score": score,
            "difficult_words": difficult_words,
            "avg_word_length": round(avg_length, 1),
            "total_words": len(words),
        }

    def _analyze_sentences(self, text: str, guidelines: dict) -> dict:
        """Analyze sentence complexity."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return {"score": 100, "avg_length": 0, "long_sentences": 0}

        max_length = guidelines["max_sentence_length"]

        # Calculate lengths
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)

        # Count long sentences
        long_sentences = sum(1 for l in lengths if l > max_length)

        # Calculate score
        long_ratio = long_sentences / len(sentences) if sentences else 0
        score = max(0, 100 - long_ratio * 150)

        # Adjust for average
        if avg_length > max_length * 0.8:
            score = max(0, score - 10)

        return {
            "score": score,
            "avg_length": round(avg_length, 1),
            "long_sentences": long_sentences,
        }

    def _check_content_flags(self, text: str) -> list[str]:
        """Check for inappropriate content flags."""
        flags = []
        text_lower = text.lower()

        # Check for potentially inappropriate terms
        # This is a simplified check - production would use more sophisticated filtering
        sensitive_patterns = [
            (r'\b(death|die|kill|murder)\b', "Contains potentially disturbing content"),
            (r'\b(hate|racist|discrimination)\b', "Contains potentially sensitive topic"),
            (r'\b(drug|alcohol|smoking)\b', "Contains substance-related content"),
        ]

        for pattern, message in sensitive_patterns:
            if re.search(pattern, text_lower):
                flags.append(message)

        return flags

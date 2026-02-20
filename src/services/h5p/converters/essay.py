# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Essay H5P Converter.

Converts AI-generated essay prompts to H5P.Essay format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class EssayConverter(BaseH5PConverter):
    """Converter for H5P.Essay content type.

    AI Input Format:
        {
            "title": "Essay Question",
            "prompt": "Explain the process of photosynthesis and its importance to life on Earth.",
            "sampleOutline": "Discuss sunlight, water, CO2, glucose, and oxygen production",
            "keywords": [
                {
                    "keyword": "sunlight",
                    "alternatives": ["light", "solar energy"],
                    "points": 2,
                    "importance": "required"
                },
                {
                    "keyword": "chlorophyll",
                    "alternatives": ["green pigment"],
                    "points": 1,
                    "importance": "optional"
                }
            ],
            "minimumLength": 100,
            "feedbackRanges": [
                {"from": 0, "to": 50, "feedback": "Try to include more key concepts."},
                {"from": 51, "to": 80, "feedback": "Good job! You covered most concepts."},
                {"from": 81, "to": 100, "feedback": "Excellent! Comprehensive answer."}
            ]
        }

    Keywords are used for automatic scoring based on presence in the essay.
    """

    @property
    def content_type(self) -> str:
        return "essay"

    @property
    def library(self) -> str:
        return "H5P.Essay 1.5"

    @property
    def category(self) -> str:
        return "assessment"

    @property
    def bloom_levels(self) -> list[str]:
        return ["create", "evaluate"]

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "prompt" not in ai_content:
            raise H5PValidationError(
                message="Missing 'prompt' field",
                content_type=self.content_type,
            )

        if "keywords" not in ai_content:
            raise H5PValidationError(
                message="Missing 'keywords' field",
                content_type=self.content_type,
            )

        keywords = ai_content.get("keywords", [])
        if not keywords:
            raise H5PValidationError(
                message="At least one keyword is required",
                content_type=self.content_type,
            )

        for i, kw in enumerate(keywords):
            if "keyword" not in kw:
                raise H5PValidationError(
                    message=f"Keyword {i+1} missing 'keyword' field",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P Essay format."""
        prompt = ai_content.get("prompt", "")
        sample_outline = ai_content.get("sampleOutline", "")
        keywords = ai_content.get("keywords", [])
        min_length = ai_content.get("minimumLength", 50)
        feedback_ranges = ai_content.get("feedbackRanges", [])

        if not prompt:
            raise H5PValidationError(
                message="Prompt is required",
                content_type=self.content_type,
            )

        # Convert keywords to H5P format
        h5p_keywords = []
        for kw in keywords:
            keyword_text = kw.get("keyword", "")
            alternatives = kw.get("alternatives", [])
            points = kw.get("points", 1)
            importance = kw.get("importance", "optional")

            # H5P expects alternatives as pipe-separated in keyword
            keyword_with_alts = keyword_text
            if alternatives:
                keyword_with_alts = keyword_text + "|" + "|".join(alternatives)

            h5p_keyword = {
                "keyword": keyword_with_alts,
                "options": {
                    "points": points,
                    "forgiveMistakes": True,
                    "caseSensitive": False,
                },
            }

            # Add importance feedback
            if importance == "required":
                h5p_keyword["options"]["feedbackIncluded"] = self._get_keyword_included_feedback(keyword_text, language)
                h5p_keyword["options"]["feedbackMissed"] = self._get_keyword_missed_feedback(keyword_text, language)

            h5p_keywords.append(h5p_keyword)

        # Build overall feedback
        if feedback_ranges:
            overall_feedback = []
            for fr in feedback_ranges:
                overall_feedback.append({
                    "from": fr.get("from", 0),
                    "to": fr.get("to", 100),
                    "feedback": fr.get("feedback", ""),
                })
        else:
            overall_feedback = self.get_overall_feedback(language)

        h5p_params = {
            "taskDescription": f"<p>{prompt}</p>",
            "keywords": h5p_keywords,
            "behaviour": self.get_default_behavior(min_length),
            "overallFeedback": overall_feedback,
            "l10n": self._get_l10n_strings(language),
            "a11y": self._get_a11y_strings(language),
        }

        if sample_outline:
            l10n_all = self.get_l10n(language)
            h5p_params["solution"] = {
                "introduction": l10n_all.get("sampleIntro", "Sample answer should include:"),
                "sample": f"<p>{sample_outline}</p>",
            }

        return h5p_params

    def get_default_behavior(self, min_length: int = 50) -> dict[str, Any]:
        """Get default behavior for Essay."""
        return {
            "enableRetry": True,
            "enableSolutionsButton": True,
            "enableCheckButton": True,
            "ignoreScoring": False,
            "pointsHost": 1,
            "minimumLength": min_length,
            "inputFieldSize": "medium",
            "showPopup": True,
        }

    def _get_keyword_included_feedback(self, keyword: str, language: str) -> str:
        template = self.get_l10n(language).get("keywordIncluded", "You correctly included '@keyword'.")
        return template.replace("@keyword", keyword)

    def _get_keyword_missed_feedback(self, keyword: str, language: str) -> str:
        template = self.get_l10n(language).get("keywordMissed", "Consider including '@keyword'.")
        return template.replace("@keyword", keyword)

    def _get_l10n_strings(self, language: str) -> dict[str, str]:
        """Get l10n strings from locale files."""
        l10n = self.get_l10n(language)
        keys = [
            "checkAnswer", "tryAgain", "showSolution", "submitAnswer",
            "remainingChars", "yourAnswer", "feedbackHeader", "solutionTitle",
            "notEnoughChars", "overallFeedback",
        ]
        return {k: l10n[k] for k in keys if k in l10n}

    def _get_a11y_strings(self, language: str) -> dict[str, str]:
        """Get accessibility strings from locale files."""
        l10n = self.get_l10n(language)
        return {
            "yourScore": l10n.get("a11yYourScore", "Your score: @score out of @total"),
            "checkAnswerAnnouncement": l10n.get("a11yCheckAnnouncement", "Your answer was checked"),
            "retryAnnouncement": l10n.get("a11yRetryAnnouncement", "You are retrying"),
        }

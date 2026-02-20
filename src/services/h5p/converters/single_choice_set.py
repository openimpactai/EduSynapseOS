# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Single Choice Set H5P Converter.

Converts AI-generated rapid-fire single choice questions to H5P.SingleChoiceSet format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class SingleChoiceSetConverter(BaseH5PConverter):
    """Converter for H5P.SingleChoiceSet content type.

    AI Input Format:
        {
            "title": "Quick Quiz",
            "questions": [
                {
                    "question": "What is 2+2?",
                    "answers": ["4", "5", "3", "6"]  // First is always correct
                }
            ]
        }

    NOTE: First answer is ALWAYS the correct one. H5P shuffles automatically.
    """

    @property
    def content_type(self) -> str:
        return "single-choice-set"

    @property
    def library(self) -> str:
        return "H5P.SingleChoiceSet 1.11"

    @property
    def category(self) -> str:
        return "assessment"

    @property
    def bloom_levels(self) -> list[str]:
        return ["remember", "understand"]

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "questions" not in ai_content:
            raise H5PValidationError(
                message="Missing 'questions' field",
                content_type=self.content_type,
            )

        for i, q in enumerate(ai_content.get("questions", [])):
            if "question" not in q:
                raise H5PValidationError(
                    message=f"Question {i+1} missing 'question' field",
                    content_type=self.content_type,
                )
            if "answers" not in q or len(q["answers"]) < 2:
                raise H5PValidationError(
                    message=f"Question {i+1} needs at least 2 answers",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P SingleChoiceSet format."""
        questions = ai_content.get("questions", [])

        if not questions:
            raise H5PValidationError(
                message="No questions provided",
                content_type=self.content_type,
            )

        # Build choices array
        choices = []
        for q in questions:
            answers = []
            for answer in q.get("answers", []):
                answers.append(self.wrap_html(answer))

            choices.append({
                "question": self.wrap_html(q.get("question", "")),
                "answers": answers,
            })

        h5p_params = {
            "choices": choices,
            "overallFeedback": self.get_overall_feedback(language),
            "behaviour": self.get_default_behavior(),
            "l10n": self.get_l10n(language),
        }

        return h5p_params

    def get_default_behavior(self) -> dict[str, Any]:
        """Get default behavior for SingleChoiceSet."""
        return {
            "autoContinue": True,
            "timeoutCorrect": 2000,
            "timeoutWrong": 3000,
            "soundEffectsEnabled": True,
            "enableRetry": True,
            "enableSolutionsButton": True,
            "passPercentage": 100,
        }


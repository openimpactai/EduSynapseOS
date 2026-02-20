# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Summary H5P Converter.

Converts AI-generated summary content to H5P.Summary format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class SummaryConverter(BaseH5PConverter):
    """Converter for H5P.Summary content type.

    AI Input Format:
        {
            "title": "Summary Title",
            "introduction": "Read the text and select the correct statements",
            "panels": [
                {
                    "statements": [
                        {"text": "Correct statement 1", "correct": true},
                        {"text": "Incorrect statement 1", "correct": false},
                        {"text": "Incorrect statement 2", "correct": false}
                    ],
                    "tip": "Think about the main idea"
                }
            ],
            "summary": "Great job! You've identified the key points."
        }

    Each panel has multiple statements, only one is correct per panel.
    User progresses through panels by selecting correct statements.
    """

    @property
    def content_type(self) -> str:
        return "summary"

    @property
    def library(self) -> str:
        return "H5P.Summary 1.10"

    @property
    def category(self) -> str:
        return "learning"

    @property
    def bloom_levels(self) -> list[str]:
        return ["analyze"]

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "panels" not in ai_content:
            raise H5PValidationError(
                message="Missing 'panels' field",
                content_type=self.content_type,
            )

        panels = ai_content.get("panels", [])
        if not panels:
            raise H5PValidationError(
                message="At least one panel is required",
                content_type=self.content_type,
            )

        for i, panel in enumerate(panels):
            statements = panel.get("statements", [])
            if len(statements) < 2:
                raise H5PValidationError(
                    message=f"Panel {i+1} must have at least 2 statements",
                    content_type=self.content_type,
                )

            # Check there's exactly one correct statement
            correct_count = sum(1 for s in statements if s.get("correct"))
            if correct_count != 1:
                raise H5PValidationError(
                    message=f"Panel {i+1} must have exactly 1 correct statement",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P Summary format."""
        panels = ai_content.get("panels", [])
        l10n = self.get_l10n(language)
        introduction = ai_content.get("introduction", l10n["defaultIntro"])
        summary_text = ai_content.get("summary", "")

        if not panels:
            raise H5PValidationError(
                message="No panels provided",
                content_type=self.content_type,
            )

        # Convert panels to H5P format
        h5p_summaries = []
        for panel in panels:
            statements = panel.get("statements", [])
            tip = panel.get("tip")

            # H5P expects the correct statement first in the summary array
            correct_statement = None
            wrong_statements = []

            for s in statements:
                if s.get("correct"):
                    correct_statement = s.get("text", "")
                else:
                    wrong_statements.append(s.get("text", ""))

            h5p_summary = {
                "summary": [correct_statement] + wrong_statements,
            }
            if tip:
                h5p_summary["tip"] = tip

            h5p_summaries.append(h5p_summary)

        h5p_params = {
            "intro": introduction,
            "summaries": h5p_summaries,
            "solvedLabel": l10n["solved"],
            "scoreLabel": l10n["scoreLabel"],
            "resultLabel": l10n["result"],
            "labelCorrect": l10n["correct"],
            "labelIncorrect": l10n["incorrect"],
            "alternativeIncorrectLabel": l10n["altIncorrect"],
            "labelCorrectAnswers": l10n["correctAnswers"],
            "tipButtonLabel": l10n["tip"],
            "scoreBarLabel": l10n["scoreBarLabel"],
            "progressText": l10n["progressText"],
            "overallFeedback": self.get_overall_feedback(language),
        }

        if summary_text:
            h5p_params["overallFeedback"] = [{"from": 0, "to": 100, "feedback": summary_text}]

        return h5p_params


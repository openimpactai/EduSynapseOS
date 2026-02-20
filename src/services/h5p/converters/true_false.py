# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""True/False H5P Converter.

Converts AI-generated true/false statements to H5P.TrueFalse format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class TrueFalseConverter(BaseH5PConverter):
    """Converter for H5P.TrueFalse content type.

    AI Input Format:
        {
            "title": "Quiz Title",
            "statements": [
                {
                    "statement": "The Earth is flat.",
                    "isTrue": false,
                    "explanation": "The Earth is actually round.",
                    "wrongFeedback": "That's incorrect."
                }
            ]
        }

    H5P Semantics mapping:
        - question ← statement (HTML wrapped)
        - correct ← "true"/"false" string
        - behaviour.feedbackOnCorrect ← explanation (shown when user answers correctly)
        - behaviour.feedbackOnWrong ← wrongFeedback (shown when user answers incorrectly)
        - l10n ← UI strings (13 fields)
        - media ← optional image (only included if image_url present)
        - confirmCheck/confirmRetry ← dialog strings
    """

    @property
    def content_type(self) -> str:
        return "true-false"

    @property
    def library(self) -> str:
        return "H5P.TrueFalse 1.8"

    @property
    def category(self) -> str:
        return "assessment"

    @property
    def bloom_levels(self) -> list[str]:
        return ["remember"]

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "statements" not in ai_content:
            raise H5PValidationError(
                message="Missing 'statements' field",
                content_type=self.content_type,
            )

        for i, s in enumerate(ai_content.get("statements", [])):
            if "statement" not in s:
                raise H5PValidationError(
                    message=f"Statement {i+1} missing 'statement' field",
                    content_type=self.content_type,
                )
            if "isTrue" not in s:
                raise H5PValidationError(
                    message=f"Statement {i+1} missing 'isTrue' field",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P TrueFalse format."""
        statements = ai_content.get("statements", [])

        if not statements:
            raise H5PValidationError(
                message="No statements provided",
                content_type=self.content_type,
            )

        s = statements[0]

        is_true = s.get("isTrue", False)
        explanation = s.get("explanation", "")
        wrong_feedback = s.get("wrongFeedback", "")

        # Determine feedback based on correctness
        if is_true:
            feedback_correct = explanation or self._default_correct_feedback(language)
            feedback_wrong = wrong_feedback or self._default_wrong_feedback(language, "false")
        else:
            feedback_correct = explanation or self._default_correct_feedback(language)
            feedback_wrong = wrong_feedback or self._default_wrong_feedback(language, "true")

        h5p_params: dict[str, Any] = {
            "question": self.wrap_html(s.get("statement", "")),
            "correct": "true" if is_true else "false",
            "behaviour": {
                "enableRetry": True,
                "enableSolutionsButton": True,
                "enableCheckButton": True,
                "confirmCheckDialog": False,
                "confirmRetryDialog": False,
                "autoCheck": False,
                "feedbackOnCorrect": feedback_correct,
                "feedbackOnWrong": feedback_wrong,
            },
            "l10n": self._get_l10n_strings(language),
            "confirmCheck": self._get_confirm_check(language),
            "confirmRetry": self._get_confirm_retry(language),
        }

        # Add media (image/video/audio) if available
        media = self.build_media(s)
        if media:
            h5p_params["media"] = media

        return h5p_params

    def _get_l10n_strings(self, language: str) -> dict[str, str]:
        """Get l10n strings matching H5P.TrueFalse 1.8 semantics."""
        l10n = self.get_l10n(language)
        keys = [
            "trueText", "falseText", "score", "checkAnswer", "submitAnswer",
            "showSolutionButton", "tryAgain", "wrongAnswerMessage",
            "correctAnswerMessage", "scoreBarLabel", "a11yCheck",
            "a11yShowSolution", "a11yRetry",
        ]
        return {k: l10n[k] for k in keys if k in l10n}

    def _get_confirm_check(self, language: str) -> dict[str, str]:
        return self.get_l10n(language).get("confirmCheck", self.get_l10n("en")["confirmCheck"])

    def _get_confirm_retry(self, language: str) -> dict[str, str]:
        return self.get_l10n(language).get("confirmRetry", self.get_l10n("en")["confirmRetry"])

    def _default_correct_feedback(self, language: str) -> str:
        return self.get_l10n(language).get("correctFeedback", "Correct!")

    def _default_wrong_feedback(self, language: str, expected: str) -> str:
        prefix = self.get_l10n(language).get("wrongFeedbackPrefix", "Incorrect. The correct answer was")
        return f"{prefix} '{expected.upper()}'."

    def convert_single_params(
        self,
        params: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert a single statement's params to H5P format.

        Used by QuestionSetConverter to delegate conversion.
        Accepts: {statement, isTrue, explanation, wrongFeedback, image_url, image_alt, ...}
        """
        return self.convert({"statements": [params]}, language)

    def convert_multiple(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> list[dict[str, Any]]:
        """Convert multiple statements to individual H5P params."""
        statements = ai_content.get("statements", [])
        results = []

        for s in statements:
            single_content = {"statements": [s]}
            h5p_params = self.convert(single_content, language)
            results.append(h5p_params)

        return results

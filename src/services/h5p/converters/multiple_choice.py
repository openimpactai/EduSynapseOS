# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multiple Choice H5P Converter.

Converts AI-generated multiple choice questions to H5P.MultiChoice format.
"""

from typing import Any
from uuid import uuid4

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class MultipleChoiceConverter(BaseH5PConverter):
    """Converter for H5P.MultiChoice content type.

    Transforms AI-generated multiple choice questions into the
    H5P MultiChoice format with proper feedback and behavior settings.

    AI Input Format:
        {
            "title": "Quiz Title",
            "questions": [
                {
                    "question": "Question text",
                    "answers": ["A", "B", "C", "D"],
                    "correctIndex": 0,
                    "explanation": "Why A is correct",
                    "distractorFeedback": ["B feedback", "C feedback", "D feedback"]
                }
            ]
        }
    """

    @property
    def content_type(self) -> str:
        return "multiple-choice"

    @property
    def library(self) -> str:
        return "H5P.MultiChoice 1.16"

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
                message="Missing 'questions' field in AI content",
                content_type=self.content_type,
                validation_errors=["questions field is required"],
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
            if "correctIndex" not in q:
                raise H5PValidationError(
                    message=f"Question {i+1} missing 'correctIndex' field",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P MultiChoice format.

        For single question, returns H5P.MultiChoice params.
        For multiple questions, this converter handles one question at a time.
        Use QuestionSet converter for multiple questions.
        """
        questions = ai_content.get("questions", [])

        if not questions:
            raise H5PValidationError(
                message="No questions provided",
                content_type=self.content_type,
            )

        # Convert first question (for single question scenarios)
        # For multiple questions, use QuestionSet
        q = questions[0]

        # Build answers array
        answers = []
        correct_index = q.get("correctIndex", 0)
        distractor_feedback = q.get("distractorFeedback", [])
        explanation = q.get("explanation", "")

        for i, answer_text in enumerate(q.get("answers", [])):
            is_correct = i == correct_index

            # Get feedback for this answer
            if is_correct:
                chosen_feedback = explanation or self._get_correct_feedback(language)
                not_chosen_feedback = ""
            else:
                # Get distractor feedback (adjust index since correct is not in list)
                dist_index = i if i < correct_index else i - 1
                if dist_index < len(distractor_feedback):
                    chosen_feedback = distractor_feedback[dist_index]
                else:
                    chosen_feedback = self._get_incorrect_feedback(language)
                not_chosen_feedback = ""

            answers.append({
                "text": self.wrap_html(answer_text, "div"),
                "correct": is_correct,
                "tipsAndFeedback": {
                    "tip": "",
                    "chosenFeedback": self.wrap_html(chosen_feedback, "div"),
                    "notChosenFeedback": self.wrap_html(not_chosen_feedback, "div"),
                },
            })

        # Build H5P params
        h5p_params: dict[str, Any] = {
            "question": self.wrap_html(q.get("question", "")),
            "answers": answers,
            "behaviour": self.get_default_behavior(),
            "UI": self._get_ui(language),
            "overallFeedback": self.get_overall_feedback(language),
        }

        # Add media (image/video/audio) if available
        media = self.build_media(q)
        if media:
            h5p_params["media"] = media

        return h5p_params

    def get_default_behavior(self) -> dict[str, Any]:
        """Get default behavior for MultiChoice."""
        return {
            "enableRetry": True,
            "enableSolutionsButton": True,
            "enableCheckButton": True,
            "type": "auto",
            "singlePoint": False,
            "randomAnswers": True,
            "showSolutionsRequiresInput": True,
            "autoCheck": False,
            "passPercentage": 100,
            "confirmCheckDialog": False,
            "confirmRetryDialog": False,
        }

    def _get_ui(self, language: str = "en") -> dict[str, str]:
        """Get localized UI strings from locale files."""
        l10n = self.get_l10n(language)
        # Filter to only UI keys (exclude correctFeedback/incorrectFeedback)
        ui_keys = [
            "checkAnswerButton", "submitAnswerButton", "showSolutionButton",
            "tryAgainButton", "tipsLabel", "scoreBarLabel", "tipAvailable",
            "feedbackAvailable", "readFeedback", "wrongAnswer", "correctAnswer",
            "shouldCheck", "shouldNotCheck", "noInput",
        ]
        return {k: l10n[k] for k in ui_keys if k in l10n}

    def _get_correct_feedback(self, language: str) -> str:
        """Get default correct feedback."""
        return self.get_l10n(language).get("correctFeedback", "Correct!")

    def _get_incorrect_feedback(self, language: str) -> str:
        """Get default incorrect feedback."""
        return self.get_l10n(language).get("incorrectFeedback", "Incorrect. Try again.")

    def convert_single_params(
        self,
        params: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert a single question's params to H5P format.

        Used by QuestionSetConverter to delegate conversion.
        Accepts the same params as a single item in the 'questions' array.
        """
        return self.convert({"questions": [params]}, language)

    def convert_multiple(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> list[dict[str, Any]]:
        """Convert multiple questions to individual H5P params.

        Returns a list of H5P params, one for each question.
        Useful for building QuestionSet content.
        """
        questions = ai_content.get("questions", [])
        results = []

        for q in questions:
            single_content = {"questions": [q]}
            h5p_params = self.convert(single_content, language)
            results.append(h5p_params)

        return results

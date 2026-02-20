# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Arithmetic Quiz H5P Converter.

Converts AI-generated arithmetic quiz parameters to H5P.ArithmeticQuiz format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class ArithmeticQuizConverter(BaseH5PConverter):
    """Converter for H5P.ArithmeticQuiz content type.

    AI Input Format:
        {
            "quizType": "arithmetic",
            "operations": ["addition", "subtraction"],
            "maxNumber": 20,
            "questionCount": 10
        }

    Supports: addition, subtraction, multiplication, division, arithmetic (mixed).
    """

    VALID_QUIZ_TYPES = {"addition", "subtraction", "multiplication", "division", "arithmetic"}
    VALID_ARITHMETIC_TYPES = {"addition", "subtraction", "multiplication", "division"}

    @property
    def content_type(self) -> str:
        return "arithmetic-quiz"

    @property
    def library(self) -> str:
        return "H5P.ArithmeticQuiz 1.1"

    @property
    def category(self) -> str:
        return "assessment"

    @property
    def bloom_levels(self) -> list[str]:
        return ["apply"]

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        quiz_type = ai_content.get("quizType", "arithmetic")
        if quiz_type not in self.VALID_QUIZ_TYPES:
            raise H5PValidationError(
                message=f"Invalid quizType: {quiz_type}. Valid types: {self.VALID_QUIZ_TYPES}",
                content_type=self.content_type,
            )

        max_number = ai_content.get("maxNumber", 10)
        if not isinstance(max_number, (int, float)) or max_number < 1:
            raise H5PValidationError(
                message="maxNumber must be a positive number",
                content_type=self.content_type,
            )

        question_count = ai_content.get("questionCount", 10)
        if not isinstance(question_count, (int, float)) or question_count < 1:
            raise H5PValidationError(
                message="questionCount must be a positive number",
                content_type=self.content_type,
            )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P ArithmeticQuiz format."""
        quiz_type = ai_content.get("quizType", "arithmetic")
        operations = ai_content.get("operations", [])
        max_number = int(ai_content.get("maxNumber", 10))
        question_count = int(ai_content.get("questionCount", 10))

        # H5P semantics: arithmeticType only accepts single operation
        # (addition, subtraction, multiplication, division).
        # "arithmetic" is NOT a valid arithmeticType — it's only valid for quizType.
        # When mixed operations requested, default arithmeticType to "addition".
        if quiz_type == "arithmetic":
            # Mixed operations — pick first valid operation or default to addition
            arithmetic_type = "addition"
            if operations:
                for op in operations:
                    if op in self.VALID_ARITHMETIC_TYPES:
                        arithmetic_type = op
                        break
            h5p_params = {
                "quizType": "arithmetic",
                "arithmeticType": arithmetic_type,
                "maxQuestions": question_count,
                "useFractions": False,
            }
        elif quiz_type in self.VALID_ARITHMETIC_TYPES:
            # Single operation mode
            h5p_params = {
                "quizType": "arithmetic",
                "arithmeticType": quiz_type,
                "maxQuestions": question_count,
                "useFractions": False,
            }
        else:
            # Linear equation mode
            h5p_params = {
                "quizType": "linearEquation",
                "equationType": quiz_type if quiz_type in {"basic", "intermediate", "advanced"} else "intermediate",
                "useFractions": bool(ai_content.get("useFractions", False)),
                "maxQuestions": question_count,
            }

        # Set UI strings based on language
        h5p_params["l10n"] = self.get_l10n(language)

        return h5p_params


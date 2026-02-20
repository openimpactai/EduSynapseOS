# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Personality Quiz H5P Converter.

Converts AI-generated personality quiz content to H5P.PersonalityQuiz format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class PersonalityQuizConverter(BaseH5PConverter):
    """Converter for H5P.PersonalityQuiz content type.

    AI Input Format:
        {
            "title": "Learning Style Quiz",
            "description": "Discover your learning style",
            "questions": [
                {
                    "question": "How do you prefer to learn new things?",
                    "answers": [
                        {"text": "By reading about it", "personality": "visual"},
                        {"text": "By listening to explanations", "personality": "auditory"},
                        {"text": "By doing hands-on activities", "personality": "kinesthetic"}
                    ]
                }
            ],
            "personalities": [
                {
                    "id": "visual",
                    "name": "Visual Learner",
                    "description": "You learn best through images, diagrams, and written text.",
                    "image": {"url": "visual.jpg", "alt": "Visual learner icon"}
                },
                {
                    "id": "auditory",
                    "name": "Auditory Learner",
                    "description": "You learn best by listening and discussing."
                },
                {
                    "id": "kinesthetic",
                    "name": "Kinesthetic Learner",
                    "description": "You learn best through hands-on experience."
                }
            ]
        }

    Questions with answers mapped to personality types.
    """

    @property
    def content_type(self) -> str:
        return "personality-quiz"

    @property
    def library(self) -> str:
        return "H5P.PersonalityQuiz 1.0"

    @property
    def category(self) -> str:
        return "game"

    @property
    def bloom_levels(self) -> list[str]:
        return ["understand"]

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "questions" not in ai_content:
            raise H5PValidationError(
                message="Missing 'questions' field",
                content_type=self.content_type,
            )

        if "personalities" not in ai_content:
            raise H5PValidationError(
                message="Missing 'personalities' field",
                content_type=self.content_type,
            )

        questions = ai_content.get("questions", [])
        personalities = ai_content.get("personalities", [])

        if not questions:
            raise H5PValidationError(
                message="At least one question is required",
                content_type=self.content_type,
            )

        if len(personalities) < 2:
            raise H5PValidationError(
                message="At least 2 personality types are required",
                content_type=self.content_type,
            )

        # Validate personality IDs are unique
        personality_ids = [p.get("id") for p in personalities]
        if len(personality_ids) != len(set(personality_ids)):
            raise H5PValidationError(
                message="Personality IDs must be unique",
                content_type=self.content_type,
            )

        # Validate all answers reference valid personalities
        for i, q in enumerate(questions):
            answers = q.get("answers", [])
            for j, a in enumerate(answers):
                if a.get("personality") not in personality_ids:
                    raise H5PValidationError(
                        message=f"Question {i+1}, Answer {j+1} references invalid personality",
                        content_type=self.content_type,
                    )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P PersonalityQuiz format."""
        questions = ai_content.get("questions", [])
        personalities = ai_content.get("personalities", [])
        title = ai_content.get("title", "Personality Quiz")
        description = ai_content.get("description", "")

        if not questions or not personalities:
            raise H5PValidationError(
                message="Questions and personalities are required",
                content_type=self.content_type,
            )

        # Create personality ID mapping
        personality_map = {p.get("id"): i for i, p in enumerate(personalities)}

        # Convert personalities to H5P format
        h5p_personalities = []
        for p in personalities:
            h5p_personality = {
                "name": p.get("name", ""),
                "description": f"<p>{p.get('description', '')}</p>",
            }

            if p.get("image"):
                h5p_personality["image"] = {
                    "path": p["image"].get("url", ""),
                    "alt": p["image"].get("alt", p.get("name", "")),
                }

            h5p_personalities.append(h5p_personality)

        # Convert questions to H5P format
        h5p_questions = []
        for q in questions:
            question_text = q.get("question", "")
            answers = q.get("answers", [])

            h5p_answers = []
            for a in answers:
                personality_id = a.get("personality")
                personality_index = personality_map.get(personality_id, 0)

                h5p_answers.append({
                    "text": a.get("text", ""),
                    "personality": personality_index,
                })

            h5p_question = {
                "text": question_text,
                "answers": h5p_answers,
            }

            if q.get("image"):
                h5p_question["image"] = {
                    "path": q["image"].get("url", ""),
                    "alt": q["image"].get("alt", ""),
                }

            h5p_questions.append(h5p_question)

        h5p_params = {
            "title": title,
            "personalities": h5p_personalities,
            "questions": h5p_questions,
            "resultScreen": self.get_l10n(language).get("resultScreen", {}),
            "l10n": self.get_l10n(language).get("l10n", {}),
            "animation": "slide",
            "showProgressBar": True,
            "allowReset": True,
        }

        if description:
            h5p_params["startScreen"] = {
                "title": title,
                "introduction": f"<p>{description}</p>",
            }

        return h5p_params


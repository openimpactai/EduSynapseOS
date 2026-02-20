# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Course Presentation H5P Converter.

Converts AI-generated course presentation content to H5P.CoursePresentation format.
"""

from typing import Any
from uuid import uuid4

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class CoursePresentationConverter(BaseH5PConverter):
    """Converter for H5P.CoursePresentation content type.

    AI Input Format:
        {
            "title": "Introduction to Photosynthesis",
            "slides": [
                {
                    "type": "title",
                    "title": "Welcome",
                    "content": "Learn about how plants make food",
                    "speakerNotes": "Introduce the topic"
                },
                {
                    "type": "content",
                    "title": "What is Photosynthesis?",
                    "content": "<p>Photosynthesis is the process...</p>",
                    "elements": [
                        {
                            "type": "image",
                            "url": "plant.jpg",
                            "alt": "A green plant",
                            "position": {"x": 10, "y": 20, "width": 40, "height": 30}
                        }
                    ]
                },
                {
                    "type": "quiz",
                    "title": "Check Your Understanding",
                    "questionType": "multiple_choice",
                    "question": "What do plants need for photosynthesis?",
                    "answers": ["Sunlight, water, CO2", "Only water", "Only air"],
                    "correctIndex": 0
                }
            ]
        }

    Multi-slide presentation with embedded interactive elements.
    """

    ELEMENT_TYPES = {
        "text": "H5P.AdvancedText 1.1",
        "image": "H5P.Image 1.1",
        "multiple_choice": "H5P.MultiChoice 1.16",
        "true_false": "H5P.TrueFalse 1.8",
        "blanks": "H5P.Blanks 1.14",
        "drag_words": "H5P.DragText 1.10",
    }

    @property
    def content_type(self) -> str:
        return "course-presentation"

    @property
    def library(self) -> str:
        return "H5P.CoursePresentation 1.25"

    @property
    def category(self) -> str:
        return "learning"

    @property
    def bloom_levels(self) -> list[str]:
        return ["understand", "apply"]

    @property
    def ai_support(self) -> str:
        return "partial"

    @property
    def requires_media(self) -> bool:
        return True

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "slides" not in ai_content:
            raise H5PValidationError(
                message="Missing 'slides' field",
                content_type=self.content_type,
            )

        slides = ai_content.get("slides", [])
        if not slides:
            raise H5PValidationError(
                message="At least one slide is required",
                content_type=self.content_type,
            )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P CoursePresentation format."""
        slides = ai_content.get("slides", [])
        title = ai_content.get("title", "Presentation")

        if not slides:
            raise H5PValidationError(
                message="No slides provided",
                content_type=self.content_type,
            )

        # Convert each slide
        h5p_slides = []
        for slide in slides:
            h5p_slide = self._convert_slide(slide, language)
            h5p_slides.append(h5p_slide)

        h5p_params = {
            "presentation": {
                "slides": h5p_slides,
                "keywordListEnabled": True,
                "globalBackgroundSelector": {},
                "keywordListAlwaysShow": False,
                "keywordListAutoHide": False,
                "keywordListOpacity": 90,
            },
            "l10n": self.get_l10n(language),
            "override": {
                "activeSurface": False,
                "hideSummarySlide": False,
                "summarySlideSolutionButton": True,
                "summarySlideRetryButton": True,
                "enablePrintButton": False,
                "social": {},
            },
        }

        return h5p_params

    def _convert_slide(self, slide: dict[str, Any], language: str) -> dict[str, Any]:
        """Convert a single slide to H5P format."""
        slide_type = slide.get("type", "content")
        elements = []

        if slide_type == "title":
            elements = self._create_title_slide(slide, language)
        elif slide_type == "content":
            elements = self._create_content_slide(slide, language)
        elif slide_type == "quiz":
            elements = self._create_quiz_slide(slide, language)
        else:
            elements = self._create_content_slide(slide, language)

        # Add additional elements if provided
        for elem in slide.get("elements", []):
            h5p_elem = self._convert_element(elem)
            if h5p_elem:
                elements.append(h5p_elem)

        h5p_slide = {
            "elements": elements,
            "keywords": [],
        }

        if slide.get("speakerNotes"):
            h5p_slide["slideBackgroundSelector"] = {}

        return h5p_slide

    def _create_title_slide(self, slide: dict[str, Any], language: str) -> list[dict]:
        """Create elements for title slide."""
        title = slide.get("title", "")
        content = slide.get("content", "")

        elements = []

        # Title text
        if title:
            elements.append({
                "x": 10,
                "y": 30,
                "width": 80,
                "height": 20,
                "action": {
                    "library": "H5P.AdvancedText 1.1",
                    "params": {
                        "text": f"<h2 style='text-align:center'>{title}</h2>",
                    },
                    "subContentId": str(uuid4()),
                },
            })

        # Subtitle/content
        if content:
            elements.append({
                "x": 10,
                "y": 55,
                "width": 80,
                "height": 20,
                "action": {
                    "library": "H5P.AdvancedText 1.1",
                    "params": {
                        "text": f"<p style='text-align:center'>{content}</p>",
                    },
                    "subContentId": str(uuid4()),
                },
            })

        return elements

    def _create_content_slide(self, slide: dict[str, Any], language: str) -> list[dict]:
        """Create elements for content slide."""
        title = slide.get("title", "")
        content = slide.get("content", "")

        elements = []

        # Title
        if title:
            elements.append({
                "x": 5,
                "y": 5,
                "width": 90,
                "height": 15,
                "action": {
                    "library": "H5P.AdvancedText 1.1",
                    "params": {
                        "text": f"<h3>{title}</h3>",
                    },
                    "subContentId": str(uuid4()),
                },
            })

        # Content
        if content:
            # Wrap in paragraph if not HTML
            if not content.startswith("<"):
                content = f"<p>{content}</p>"

            elements.append({
                "x": 5,
                "y": 25,
                "width": 90,
                "height": 60,
                "action": {
                    "library": "H5P.AdvancedText 1.1",
                    "params": {
                        "text": content,
                    },
                    "subContentId": str(uuid4()),
                },
            })

        return elements

    def _create_quiz_slide(self, slide: dict[str, Any], language: str) -> list[dict]:
        """Create elements for quiz slide."""
        title = slide.get("title", "")
        question_type = slide.get("questionType", "multiple_choice")
        question = slide.get("question", "")
        answers = slide.get("answers", [])
        correct_idx = slide.get("correctIndex", 0)

        elements = []

        # Title
        if title:
            elements.append({
                "x": 5,
                "y": 5,
                "width": 90,
                "height": 10,
                "action": {
                    "library": "H5P.AdvancedText 1.1",
                    "params": {
                        "text": f"<h3>{title}</h3>",
                    },
                    "subContentId": str(uuid4()),
                },
            })

        # Quiz element
        if question_type == "multiple_choice":
            h5p_answers = []
            for i, answer in enumerate(answers):
                h5p_answers.append({
                    "text": f"<div>{answer}</div>",
                    "correct": i == correct_idx,
                })

            elements.append({
                "x": 5,
                "y": 20,
                "width": 90,
                "height": 70,
                "action": {
                    "library": "H5P.MultiChoice 1.16",
                    "params": {
                        "question": f"<p>{question}</p>",
                        "answers": h5p_answers,
                        "behaviour": {
                            "enableRetry": True,
                            "enableSolutionsButton": True,
                            "singlePoint": False,
                        },
                    },
                    "subContentId": str(uuid4()),
                },
            })

        elif question_type == "true_false":
            is_true = slide.get("isTrue", True)
            elements.append({
                "x": 5,
                "y": 20,
                "width": 90,
                "height": 70,
                "action": {
                    "library": "H5P.TrueFalse 1.8",
                    "params": {
                        "question": f"<p>{question}</p>",
                        "correct": "true" if is_true else "false",
                    },
                    "subContentId": str(uuid4()),
                },
            })

        return elements

    def _convert_element(self, elem: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a custom element."""
        elem_type = elem.get("type", "")
        position = elem.get("position", {})

        x = position.get("x", 0)
        y = position.get("y", 0)
        width = position.get("width", 40)
        height = position.get("height", 30)

        if elem_type == "image":
            return {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "action": {
                    "library": "H5P.Image 1.1",
                    "params": {
                        "file": {
                            "path": elem.get("url", ""),
                        },
                        "alt": elem.get("alt", ""),
                    },
                    "subContentId": str(uuid4()),
                },
            }

        elif elem_type == "text":
            content = elem.get("content", "")
            return {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "action": {
                    "library": "H5P.AdvancedText 1.1",
                    "params": {
                        "text": content if content.startswith("<") else f"<p>{content}</p>",
                    },
                    "subContentId": str(uuid4()),
                },
            }

        return None


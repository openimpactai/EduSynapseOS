# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Interactive Book H5P Converter.

Converts AI-generated interactive book content to H5P.InteractiveBook format.
"""

from typing import Any
from uuid import uuid4

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class InteractiveBookConverter(BaseH5PConverter):
    """Converter for H5P.InteractiveBook content type.

    AI Input Format:
        {
            "title": "The Water Cycle",
            "coverImage": {"url": "water_cycle.jpg", "alt": "Water cycle diagram"},
            "chapters": [
                {
                    "title": "Introduction",
                    "content": [
                        {"type": "text", "content": "<p>Water is essential...</p>"},
                        {
                            "type": "quiz",
                            "questionType": "multiple_choice",
                            "question": "What is H2O?",
                            "answers": ["Water", "Oxygen", "Hydrogen"],
                            "correctIndex": 0
                        }
                    ]
                },
                {
                    "title": "Evaporation",
                    "content": [
                        {"type": "text", "content": "<p>Evaporation occurs when...</p>"},
                        {"type": "image", "url": "evaporation.jpg", "alt": "Evaporation"}
                    ]
                }
            ],
            "showSummary": true,
            "showProgressBar": true
        }

    Multi-chapter book with embedded H5P content.
    """

    @property
    def content_type(self) -> str:
        return "interactive-book"

    @property
    def library(self) -> str:
        return "H5P.InteractiveBook 1.7"

    @property
    def category(self) -> str:
        return "learning"

    @property
    def bloom_levels(self) -> list[str]:
        return ["understand", "apply", "analyze"]

    @property
    def ai_support(self) -> str:
        return "partial"

    @property
    def requires_media(self) -> bool:
        return True

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "chapters" not in ai_content:
            raise H5PValidationError(
                message="Missing 'chapters' field",
                content_type=self.content_type,
            )

        chapters = ai_content.get("chapters", [])
        if not chapters:
            raise H5PValidationError(
                message="At least one chapter is required",
                content_type=self.content_type,
            )

        for i, ch in enumerate(chapters):
            if "title" not in ch:
                raise H5PValidationError(
                    message=f"Chapter {i+1} missing 'title'",
                    content_type=self.content_type,
                )
            if "content" not in ch:
                raise H5PValidationError(
                    message=f"Chapter {i+1} missing 'content'",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P InteractiveBook format."""
        chapters = ai_content.get("chapters", [])
        title = ai_content.get("title", "Interactive Book")
        cover_image = ai_content.get("coverImage")
        show_summary = ai_content.get("showSummary", True)

        if not chapters:
            raise H5PValidationError(
                message="No chapters provided",
                content_type=self.content_type,
            )

        # Convert chapters
        h5p_chapters = []
        for chapter in chapters:
            h5p_chapter = self._convert_chapter(chapter, language)
            h5p_chapters.append(h5p_chapter)

        h5p_params = {
            "chapters": h5p_chapters,
            "showCoverPage": True,
            "bookCover": {
                "coverImage": self._convert_cover_image(cover_image) if cover_image else None,
                "title": title,
                "author": "",
                "datePublished": "",
            },
            "behaviour": {
                "displaySummary": show_summary,
                "displaySolutionsSummary": True,
                "defaultTableOfContents": True,
                "progressIndicators": True,
                "progressAuto": True,
            },
            "l10n": self.get_l10n(language).get("l10n", {}),
            "a11y": self.get_l10n(language).get("a11y", {}),
        }

        return h5p_params

    def _convert_chapter(self, chapter: dict[str, Any], language: str) -> dict[str, Any]:
        """Convert a single chapter to H5P format."""
        title = chapter.get("title", "Chapter")
        content_items = chapter.get("content", [])

        # Build chapter content as a Column
        column_content = []
        for item in content_items:
            h5p_item = self._convert_content_item(item, language)
            if h5p_item:
                column_content.append(h5p_item)

        return {
            "params": {
                "content": column_content,
            },
            "library": "H5P.Column 1.16",
            "subContentId": str(uuid4()),
            "metadata": {
                "title": title,
                "license": "U",
            },
        }

    def _convert_content_item(self, item: dict[str, Any], language: str) -> dict[str, Any] | None:
        """Convert a content item to H5P format."""
        item_type = item.get("type", "text")

        if item_type == "text":
            return self._convert_text_item(item)
        elif item_type == "image":
            return self._convert_image_item(item)
        elif item_type == "quiz":
            return self._convert_quiz_item(item, language)
        elif item_type == "video":
            return self._convert_video_item(item)

        return None

    def _convert_text_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Convert text content item."""
        content = item.get("content", "")
        if not content.startswith("<"):
            content = f"<p>{content}</p>"

        return {
            "content": {
                "params": {
                    "text": content,
                },
                "library": "H5P.AdvancedText 1.1",
                "subContentId": str(uuid4()),
            },
        }

    def _convert_image_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Convert image content item."""
        return {
            "content": {
                "params": {
                    "file": {
                        "path": item.get("url", ""),
                    },
                    "alt": item.get("alt", ""),
                    "title": item.get("title", ""),
                },
                "library": "H5P.Image 1.1",
                "subContentId": str(uuid4()),
            },
        }

    def _convert_quiz_item(self, item: dict[str, Any], language: str) -> dict[str, Any]:
        """Convert quiz content item."""
        question_type = item.get("questionType", "multiple_choice")
        question = item.get("question", "")
        answers = item.get("answers", [])
        correct_idx = item.get("correctIndex", 0)

        if question_type == "multiple_choice":
            h5p_answers = []
            for i, answer in enumerate(answers):
                h5p_answers.append({
                    "text": f"<div>{answer}</div>",
                    "correct": i == correct_idx,
                })

            return {
                "content": {
                    "params": {
                        "question": f"<p>{question}</p>",
                        "answers": h5p_answers,
                        "behaviour": {
                            "enableRetry": True,
                            "enableSolutionsButton": True,
                        },
                    },
                    "library": "H5P.MultiChoice 1.16",
                    "subContentId": str(uuid4()),
                },
            }

        elif question_type == "true_false":
            is_true = item.get("isTrue", True)
            return {
                "content": {
                    "params": {
                        "question": f"<p>{question}</p>",
                        "correct": "true" if is_true else "false",
                        "behaviour": {
                            "enableRetry": True,
                            "enableSolutionsButton": True,
                        },
                    },
                    "library": "H5P.TrueFalse 1.8",
                    "subContentId": str(uuid4()),
                },
            }

        return self._convert_text_item({"content": question})

    def _convert_video_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Convert video content item."""
        return {
            "content": {
                "params": {
                    "sources": [
                        {
                            "path": item.get("url", ""),
                            "mime": item.get("mime", "video/mp4"),
                        }
                    ],
                    "title": item.get("title", ""),
                },
                "library": "H5P.Video 1.6",
                "subContentId": str(uuid4()),
            },
        }

    def _convert_cover_image(self, cover: dict[str, Any]) -> dict[str, Any]:
        """Convert cover image."""
        return {
            "path": cover.get("url", ""),
            "alt": cover.get("alt", "Book cover"),
        }


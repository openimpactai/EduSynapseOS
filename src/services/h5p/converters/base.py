# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Base H5P Converter.

This module defines the abstract base class for all H5P converters.
Each content type converter must inherit from this class and implement
the required methods.

The converter is responsible for:
1. Transforming AI-generated content to H5P params format
2. Validating the converted content
3. Adding required H5P metadata and behavior settings
4. Providing localization support

Example:
    class MultipleChoiceConverter(BaseH5PConverter):
        @property
        def content_type(self) -> str:
            return "multiple-choice"

        @property
        def library(self) -> str:
            return "H5P.MultiChoice 1.16"

        def convert(self, ai_content: dict, language: str = "en") -> dict:
            # Transform ai_content to H5P params
            return h5p_params
"""

import json
import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.services.h5p.exceptions import H5PConversionError, H5PValidationError

logger = logging.getLogger(__name__)

_LOCALES_DIR = Path(__file__).resolve().parents[4] / "config" / "h5p-locales"


@lru_cache(maxsize=256)
def _load_locale_file(path: str) -> dict[str, Any]:
    """Load and cache a JSON locale file."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


class BaseH5PConverter(ABC):
    """Abstract base class for H5P content converters.

    All H5P content type converters must inherit from this class
    and implement the required abstract methods.

    The converter handles the transformation from AI-generated content
    (which uses a simplified, consistent format) to H5P params format
    (which follows specific H5P schema requirements).

    Attributes:
        content_type: Unique identifier for the content type (e.g., "multiple-choice").
        library: Full H5P library identifier (e.g., "H5P.MultiChoice 1.16").
    """

    @property
    @abstractmethod
    def content_type(self) -> str:
        """Get the content type identifier.

        Returns:
            Content type string used for registry lookup (e.g., "multiple-choice").
        """
        pass

    @property
    @abstractmethod
    def library(self) -> str:
        """Get the H5P library identifier.

        Returns:
            Full H5P library string (e.g., "H5P.MultiChoice 1.16").
        """
        pass

    @property
    def category(self) -> str:
        """Get the content category.

        Returns:
            Category string (assessment, vocabulary, learning, game, media).
        """
        return "assessment"

    @property
    def ai_support(self) -> str:
        """Get the AI support level.

        Returns:
            Support level (full, partial, none).
        """
        return "full"

    @property
    def bloom_levels(self) -> list[str]:
        """Get supported Bloom's taxonomy levels.

        Returns:
            List of Bloom's levels (remember, understand, apply, etc.).
        """
        return ["remember", "understand"]

    @property
    def requires_media(self) -> bool:
        """Check if content type requires media generation.

        Returns:
            True if media is required.
        """
        return False

    @abstractmethod
    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI-generated content to H5P params format.

        This is the main conversion method that transforms the AI output
        into H5P-compatible parameters.

        Args:
            ai_content: AI-generated content in the standardized format.
            language: Language code for localization (default: "en").

        Returns:
            H5P params dictionary ready for API submission.

        Raises:
            H5PConversionError: If conversion fails.
        """
        pass

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI-generated content before conversion.

        Override this method to add content-specific validation.
        The base implementation performs basic structure checks.

        Args:
            ai_content: AI-generated content to validate.

        Returns:
            True if content is valid.

        Raises:
            H5PValidationError: If validation fails.
        """
        if not isinstance(ai_content, dict):
            raise H5PValidationError(
                message="AI content must be a dictionary",
                content_type=self.content_type,
            )
        return True

    def validate_h5p_params(self, h5p_params: dict[str, Any]) -> bool:
        """Validate converted H5P params.

        Override this method to add content-specific validation
        against the H5P schema.

        Args:
            h5p_params: Converted H5P params to validate.

        Returns:
            True if params are valid.

        Raises:
            H5PValidationError: If validation fails.
        """
        if not isinstance(h5p_params, dict):
            raise H5PValidationError(
                message="H5P params must be a dictionary",
                content_type=self.content_type,
            )
        return True

    def get_l10n(self, language: str = "en") -> dict[str, Any]:
        """Load l10n strings from JSON locale files.

        Merges: _common/{lang}.json + {content_type}/{lang}.json
        Content-type strings override common strings.
        Falls back to English if the requested language file is not found.

        Args:
            language: Language code (e.g., "en", "tr", "ar", "es", "fr").

        Returns:
            Merged dictionary of localized strings.
        """
        ct = self.content_type

        # Load common strings
        common_path = str(_LOCALES_DIR / "_common" / f"{language}.json")
        common_en_path = str(_LOCALES_DIR / "_common" / "en.json")
        common = _load_locale_file(common_en_path).copy()
        if language != "en":
            lang_common = _load_locale_file(common_path)
            if lang_common:
                common.update(lang_common)

        # Load content-type strings
        ct_en_path = str(_LOCALES_DIR / ct / "en.json")
        ct_path = str(_LOCALES_DIR / ct / f"{language}.json")
        ct_strings = _load_locale_file(ct_en_path).copy()
        if language != "en":
            lang_ct = _load_locale_file(ct_path)
            if lang_ct:
                ct_strings.update(lang_ct)

        # Merge: content-type overrides common
        merged = common
        merged.update(ct_strings)
        return merged

    def get_default_behavior(self) -> dict[str, Any]:
        """Get default behavior settings for this content type.

        Override to customize default behavior settings.

        Returns:
            Dictionary of behavior settings.
        """
        return {
            "enableRetry": True,
            "enableSolutionsButton": True,
            "enableCheckButton": True,
        }

    def get_ui_strings(self, language: str = "en") -> dict[str, str]:
        """Get localized UI strings for the content type.

        Args:
            language: Language code.

        Returns:
            Dictionary of UI string translations.
        """
        l10n = self.get_l10n(language)
        return l10n.get("ui", l10n)

    def get_overall_feedback(
        self,
        language: str = "en",
        custom_feedback: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Get overall feedback settings based on score ranges.

        Args:
            language: Language code.
            custom_feedback: Optional custom feedback entries.

        Returns:
            List of feedback entries with from, to, and feedback fields.
        """
        if custom_feedback:
            return custom_feedback

        l10n = self.get_l10n(language)
        feedback = l10n.get("overallFeedback")
        if feedback:
            return feedback

        return [
            {"from": 0, "to": 50, "feedback": "You need more practice."},
            {"from": 51, "to": 80, "feedback": "Good job! You can do even better."},
            {"from": 81, "to": 100, "feedback": "Excellent! You've mastered this topic."},
        ]

    def wrap_html(self, text: str, tag: str = "p") -> str:
        """Wrap text in HTML tag if not already wrapped.

        H5P often expects HTML-formatted text. This helper ensures
        text is properly wrapped.

        Args:
            text: Text to wrap.
            tag: HTML tag to use (default: "p").

        Returns:
            HTML-wrapped text.
        """
        if not text:
            return f"<{tag}></{tag}>"

        # Check if already wrapped
        text = text.strip()
        if text.startswith(f"<{tag}>") or text.startswith("<p>") or text.startswith("<div>"):
            return text

        return f"<{tag}>{text}</{tag}>"

    def build_media(
        self,
        item: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Build H5P media structure from item's media URL fields.

        Checks for image_url, video_url, audio_url (in priority order)
        and builds the appropriate H5P media structure.

        Args:
            item: Dict that may contain image_url, video_url, or audio_url.

        Returns:
            H5P media dict or None if no media URL found.
        """
        from uuid import uuid4

        image_url = item.get("image_url")
        video_url = item.get("video_url")
        audio_url = item.get("audio_url")

        if video_url:
            return {
                "type": {
                    "library": "H5P.Video 1.6",
                    "params": {
                        "sources": [
                            {
                                "path": video_url,
                                "mime": "video/mp4",
                            }
                        ],
                        "visuals": {
                            "fit": True,
                            "controls": True,
                        },
                    },
                    "metadata": {
                        "contentType": "Video",
                        "license": "U",
                        "title": "Generated Video",
                    },
                    "subContentId": str(uuid4()),
                },
                "disableImageZooming": False,
            }
        elif audio_url:
            return {
                "type": {
                    "library": "H5P.Audio 1.5",
                    "params": {
                        "files": [
                            {
                                "path": audio_url,
                                "mime": "audio/wav",
                            }
                        ],
                        "fitToWrapper": False,
                        "controls": True,
                        "autoplay": False,
                    },
                    "metadata": {
                        "contentType": "Audio",
                        "license": "U",
                        "title": "Generated Audio",
                    },
                    "subContentId": str(uuid4()),
                },
                "disableImageZooming": False,
            }
        elif image_url:
            return {
                "type": {
                    "library": "H5P.Image 1.1",
                    "params": {
                        "file": {
                            "path": image_url,
                            "mime": "image/png",
                        },
                        "alt": item.get("image_alt", ""),
                    },
                    "metadata": {
                        "contentType": "Image",
                        "license": "U",
                        "title": "Generated Image",
                    },
                    "subContentId": str(uuid4()),
                },
                "disableImageZooming": False,
            }

        return None

    def generate_metadata(
        self,
        title: str,
        language: str = "en",
        license_type: str = "U",
    ) -> dict[str, Any]:
        """Generate H5P content metadata.

        Args:
            title: Content title.
            language: Language code.
            license_type: License type (default: "U" for undecided).

        Returns:
            H5P metadata dictionary.
        """
        return {
            "title": title,
            "license": license_type,
            "authors": [],
            "changes": [],
            "extraTitle": title,
            "defaultLanguage": language,
        }

    def safe_convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Safely convert AI content with validation.

        This method wraps the convert() method with validation
        and error handling.

        Args:
            ai_content: AI-generated content.
            language: Language code.

        Returns:
            H5P params dictionary.

        Raises:
            H5PConversionError: If conversion fails.
            H5PValidationError: If validation fails.
        """
        try:
            # Validate input
            self.validate_ai_content(ai_content)

            # Convert
            h5p_params = self.convert(ai_content, language)

            # Validate output
            self.validate_h5p_params(h5p_params)

            return h5p_params

        except (H5PConversionError, H5PValidationError):
            raise
        except Exception as e:
            raise H5PConversionError(
                message=f"Conversion failed: {str(e)}",
                content_type=self.content_type,
                source_format="ai_content",
                details={"error": str(e)},
            ) from e

    def get_content_info(self) -> dict[str, Any]:
        """Get content type information for documentation.

        Returns:
            Dictionary with content type details.
        """
        return {
            "content_type": self.content_type,
            "library": self.library,
            "category": self.category,
            "ai_support": self.ai_support,
            "bloom_levels": self.bloom_levels,
            "requires_media": self.requires_media,
        }

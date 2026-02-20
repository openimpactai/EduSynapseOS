# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Memory Game H5P Converter.

Converts AI-generated memory game content to H5P.MemoryGame format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class MemoryGameConverter(BaseH5PConverter):
    """Converter for H5P.MemoryGame content type.

    AI Input Format:
        {
            "title": "Science Matching",
            "description": "Match the terms with their definitions",
            "pairs": [
                {
                    "card1": {"type": "text", "content": "Photosynthesis"},
                    "card2": {"type": "text", "content": "Process plants use to make food"}
                },
                {
                    "card1": {"type": "image", "url": "sun.jpg", "alt": "Sun"},
                    "card2": {"type": "text", "content": "Star"}
                }
            ]
        }

    Cards can be text or images. Pairs are matched by players.
    """

    @property
    def content_type(self) -> str:
        return "memory-game"

    @property
    def library(self) -> str:
        return "H5P.MemoryGame 1.3"

    @property
    def category(self) -> str:
        return "game"

    @property
    def bloom_levels(self) -> list[str]:
        return ["remember"]

    @property
    def ai_support(self) -> str:
        return "partial"

    @property
    def requires_media(self) -> bool:
        return True

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "pairs" not in ai_content:
            raise H5PValidationError(
                message="Missing 'pairs' field",
                content_type=self.content_type,
            )

        pairs = ai_content.get("pairs", [])
        if len(pairs) < 2:
            raise H5PValidationError(
                message="At least 2 pairs are required",
                content_type=self.content_type,
            )

        for i, pair in enumerate(pairs):
            if "card1" not in pair or "card2" not in pair:
                raise H5PValidationError(
                    message=f"Pair {i+1} missing 'card1' or 'card2'",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P MemoryGame format."""
        pairs = ai_content.get("pairs", [])
        description = ai_content.get("description", "")

        if not pairs:
            raise H5PValidationError(
                message="No pairs provided",
                content_type=self.content_type,
            )

        # Convert pairs to H5P format
        h5p_cards = []
        for pair in pairs:
            card1 = pair.get("card1", {})
            card2 = pair.get("card2", {})

            h5p_card = {
                "image": self._convert_card(card1),
                "match": self._convert_card(card2),
                "description": pair.get("description", ""),
            }
            h5p_cards.append(h5p_card)

        h5p_params = {
            "cards": h5p_cards,
            "behaviour": self.get_default_behavior(),
            "lookNFeel": self._get_look_and_feel(),
            "l10n": self.get_l10n(language),
        }

        if description:
            h5p_params["description"] = f"<p>{description}</p>"

        return h5p_params

    def _convert_card(self, card: dict[str, Any]) -> dict[str, Any]:
        """Convert a card to H5P format."""
        card_type = card.get("type", "text")

        if card_type == "image":
            return {
                "path": card.get("url", ""),
                "alt": card.get("alt", card.get("content", "")),
            }
        else:  # text
            # Memory game uses images, so wrap text in a styled div
            content = card.get("content", "")
            return {
                "path": "",  # No image
                "alt": content,
                # H5P will use alt text as fallback
            }

    def get_default_behavior(self) -> dict[str, Any]:
        """Get default behavior for MemoryGame."""
        return {
            "useGrid": True,
            "allowRetry": True,
        }

    def _get_look_and_feel(self) -> dict[str, Any]:
        """Get visual styling settings."""
        return {
            "themeColor": "#4285F4",
            "cardBack": "",  # Use default card back
        }


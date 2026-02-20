# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Branching Scenario H5P Converter.

Converts AI-generated branching scenario content to H5P.BranchingScenario format.
"""

from typing import Any
from uuid import uuid4

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class BranchingScenarioConverter(BaseH5PConverter):
    """Converter for H5P.BranchingScenario content type.

    AI Input Format:
        {
            "title": "Making Good Decisions",
            "startScreen": {
                "title": "Welcome",
                "subtitle": "Learn to make good choices",
                "image": {"url": "start.jpg", "alt": "Welcome image"}
            },
            "nodes": [
                {
                    "id": "start",
                    "type": "content",
                    "title": "The Situation",
                    "content": "<p>You find a wallet on the ground...</p>",
                    "nextContentId": "decision1"
                },
                {
                    "id": "decision1",
                    "type": "branching",
                    "title": "What do you do?",
                    "content": "<p>You need to decide...</p>",
                    "alternatives": [
                        {
                            "text": "Return the wallet",
                            "nextContentId": "good_outcome",
                            "feedback": "Great choice!"
                        },
                        {
                            "text": "Keep the wallet",
                            "nextContentId": "bad_outcome",
                            "feedback": "Think again..."
                        }
                    ]
                },
                {
                    "id": "good_outcome",
                    "type": "ending",
                    "title": "Good Ending",
                    "content": "<p>You did the right thing!</p>",
                    "score": 10
                },
                {
                    "id": "bad_outcome",
                    "type": "ending",
                    "title": "Try Again",
                    "content": "<p>That wasn't the best choice.</p>",
                    "score": 0
                }
            ],
            "endScreens": [
                {
                    "score": {"min": 8, "max": 10},
                    "title": "Excellent!",
                    "message": "You made great choices."
                },
                {
                    "score": {"min": 0, "max": 7},
                    "title": "Keep Learning",
                    "message": "You can do better!"
                }
            ]
        }

    Decision-based branching content with multiple paths and outcomes.
    """

    @property
    def content_type(self) -> str:
        return "branching-scenario"

    @property
    def library(self) -> str:
        return "H5P.BranchingScenario 1.7"

    @property
    def category(self) -> str:
        return "learning"

    @property
    def bloom_levels(self) -> list[str]:
        return ["evaluate"]

    @property
    def ai_support(self) -> str:
        return "partial"

    @property
    def requires_media(self) -> bool:
        return True

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "nodes" not in ai_content:
            raise H5PValidationError(
                message="Missing 'nodes' field",
                content_type=self.content_type,
            )

        nodes = ai_content.get("nodes", [])
        if not nodes:
            raise H5PValidationError(
                message="At least one node is required",
                content_type=self.content_type,
            )

        # Check for at least one ending
        endings = [n for n in nodes if n.get("type") == "ending"]
        if not endings:
            raise H5PValidationError(
                message="At least one ending node is required",
                content_type=self.content_type,
            )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P BranchingScenario format."""
        nodes = ai_content.get("nodes", [])
        start_screen = ai_content.get("startScreen", {})
        end_screens = ai_content.get("endScreens", [])
        title = ai_content.get("title", "Branching Scenario")

        if not nodes:
            raise H5PValidationError(
                message="No nodes provided",
                content_type=self.content_type,
            )

        # Build node ID to index mapping
        node_id_map = {}
        for i, node in enumerate(nodes):
            node_id = node.get("id", str(i))
            node_id_map[node_id] = i

        # Convert nodes
        h5p_content = []
        for i, node in enumerate(nodes):
            h5p_node = self._convert_node(node, node_id_map, language)
            h5p_content.append(h5p_node)

        # Convert end screens
        h5p_end_screens = self._convert_end_screens(end_screens, language)

        h5p_params = {
            "startScreen": self._convert_start_screen(start_screen, title, language),
            "content": h5p_content,
            "endScreens": h5p_end_screens,
            "behaviour": {
                "enableBackwardsNavigation": True,
                "forceContentFinished": False,
                "randomizeBranchingQuestions": False,
                "scoringOptionGroup": {
                    "scoringOption": "static-end-score",
                    "includeInteractionsScores": True,
                },
            },
            "l10n": self.get_l10n(language),
        }

        return h5p_params

    def _convert_node(
        self,
        node: dict[str, Any],
        node_id_map: dict[str, int],
        language: str,
    ) -> dict[str, Any]:
        """Convert a single node to H5P format."""
        node_type = node.get("type", "content")
        title = node.get("title", "")
        content = node.get("content", "")

        if node_type == "branching":
            return self._convert_branching_node(node, node_id_map, language)
        elif node_type == "ending":
            return self._convert_ending_node(node, language)
        else:  # content node
            return self._convert_content_node(node, node_id_map, language)

    def _convert_content_node(
        self,
        node: dict[str, Any],
        node_id_map: dict[str, int],
        language: str,
    ) -> dict[str, Any]:
        """Convert content node."""
        content = node.get("content", "")
        next_id = node.get("nextContentId")
        next_idx = node_id_map.get(next_id, -1) if next_id else -1

        return {
            "type": {
                "library": "H5P.AdvancedText 1.1",
                "params": {
                    "text": content if content.startswith("<") else f"<p>{content}</p>",
                },
                "subContentId": str(uuid4()),
            },
            "showContentTitle": True,
            "contentTitle": node.get("title", ""),
            "nextContentId": next_idx,
            "feedback": {},
        }

    def _convert_branching_node(
        self,
        node: dict[str, Any],
        node_id_map: dict[str, int],
        language: str,
    ) -> dict[str, Any]:
        """Convert branching (question) node."""
        content = node.get("content", "")
        alternatives = node.get("alternatives", [])

        h5p_alternatives = []
        for alt in alternatives:
            next_id = alt.get("nextContentId")
            next_idx = node_id_map.get(next_id, -1) if next_id else -1

            h5p_alternatives.append({
                "text": alt.get("text", ""),
                "nextContentId": next_idx,
                "feedback": {
                    "title": "",
                    "subtitle": "",
                    "image": {},
                    "endScreenScore": 0,
                },
            })

        return {
            "type": {
                "library": "H5P.BranchingQuestion 1.0",
                "params": {
                    "branchingQuestion": {
                        "question": content if content.startswith("<") else f"<p>{content}</p>",
                        "alternatives": h5p_alternatives,
                    },
                },
                "subContentId": str(uuid4()),
            },
            "showContentTitle": True,
            "contentTitle": node.get("title", ""),
            "feedback": {},
        }

    def _convert_ending_node(
        self,
        node: dict[str, Any],
        language: str,
    ) -> dict[str, Any]:
        """Convert ending node."""
        content = node.get("content", "")
        score = node.get("score", 0)

        return {
            "type": {
                "library": "H5P.AdvancedText 1.1",
                "params": {
                    "text": content if content.startswith("<") else f"<p>{content}</p>",
                },
                "subContentId": str(uuid4()),
            },
            "showContentTitle": True,
            "contentTitle": node.get("title", ""),
            "nextContentId": -1,  # -1 indicates ending
            "feedback": {
                "title": node.get("title", ""),
                "subtitle": "",
                "endScreenScore": score,
            },
        }

    def _convert_start_screen(
        self,
        start_screen: dict[str, Any],
        title: str,
        language: str,
    ) -> dict[str, Any]:
        """Convert start screen."""
        h5p_start = {
            "startScreenTitle": start_screen.get("title", title),
            "startScreenSubtitle": start_screen.get("subtitle", ""),
        }

        if start_screen.get("image"):
            h5p_start["startScreenImage"] = {
                "path": start_screen["image"].get("url", ""),
                "alt": start_screen["image"].get("alt", ""),
            }

        return h5p_start

    def _convert_end_screens(
        self,
        end_screens: list[dict[str, Any]],
        language: str,
    ) -> list[dict[str, Any]]:
        """Convert end screens."""
        if not end_screens:
            return [
                {
                    "endScreenTitle": self.get_l10n(language).get("defaultEndTitle", "Completed!"),
                    "endScreenSubtitle": "",
                    "contentId": -1,
                    "endScreenScore": 0,
                }
            ]

        h5p_screens = []
        for screen in end_screens:
            h5p_screen = {
                "endScreenTitle": screen.get("title", ""),
                "endScreenSubtitle": screen.get("message", ""),
                "contentId": -1,
            }

            score_range = screen.get("score", {})
            if score_range:
                h5p_screen["endScreenScore"] = score_range.get("min", 0)

            h5p_screens.append(h5p_screen)

        return h5p_screens


# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get H5P Content Types Tool.

Returns available H5P content types with their capabilities,
AI support level, and recommended use cases.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult, UIElement, UIElementOption, UIElementType

logger = logging.getLogger(__name__)

# Load content types from schema file
CONTENT_TYPES_PATH = Path(__file__).parent.parent.parent.parent.parent / "config" / "h5p-schemas" / "content-types.json"


def load_content_types() -> dict[str, Any]:
    """Load content types from schema file."""
    try:
        with open(CONTENT_TYPES_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("Content types schema not found at %s", CONTENT_TYPES_PATH)
        return {"content_types": {}, "categories": {}}


class GetH5PContentTypesTool(BaseTool):
    """Get list of supported H5P content types.

    Returns content types with their capabilities, AI support level,
    Bloom's taxonomy alignment, and recommended use cases.

    Example usage by agent:
        - "What content types can you create?"
        - "Show me quiz options"
        - "What vocabulary activities are available?"
    """

    # Content type definitions (inline for reliability)
    CONTENT_TYPES = {
        "multiple-choice": {
            "type_id": "H5P.MultiChoice",
            "name": "Multiple Choice",
            "category": "assessment",
            "ai_support": "full",
            "bloom_levels": ["remember", "understand"],
            "description": "Multiple choice questions with rich feedback",
            "use_cases": ["Knowledge check", "Quick assessment", "Review quizzes"],
            "requires_media": False,
        },
        "true-false": {
            "type_id": "H5P.TrueFalse",
            "name": "True/False",
            "category": "assessment",
            "ai_support": "full",
            "bloom_levels": ["remember"],
            "description": "Simple true or false questions",
            "use_cases": ["Quick check", "Misconception test", "Fact verification"],
            "requires_media": False,
        },
        "fill-blanks": {
            "type_id": "H5P.Blanks",
            "name": "Fill in the Blanks",
            "category": "assessment",
            "ai_support": "full",
            "bloom_levels": ["apply"],
            "description": "Text with missing words to fill in",
            "use_cases": ["Vocabulary practice", "Grammar exercises", "Concept application"],
            "requires_media": False,
        },
        "drag-words": {
            "type_id": "H5P.DragText",
            "name": "Drag the Words",
            "category": "assessment",
            "ai_support": "full",
            "bloom_levels": ["apply", "analyze"],
            "description": "Drag words into correct positions in text",
            "use_cases": ["Sentence completion", "Ordering concepts", "Term placement"],
            "requires_media": False,
        },
        "mark-words": {
            "type_id": "H5P.MarkTheWords",
            "name": "Mark the Words",
            "category": "assessment",
            "ai_support": "full",
            "bloom_levels": ["analyze"],
            "description": "Identify and mark specific words in text",
            "use_cases": ["Grammar identification", "Concept recognition", "Text analysis"],
            "requires_media": False,
        },
        "single-choice-set": {
            "type_id": "H5P.SingleChoiceSet",
            "name": "Single Choice Set",
            "category": "assessment",
            "ai_support": "full",
            "bloom_levels": ["remember", "understand"],
            "description": "Rapid-fire single choice questions",
            "use_cases": ["Quick drill", "Speed review", "Knowledge recall"],
            "requires_media": False,
        },
        "essay": {
            "type_id": "H5P.Essay",
            "name": "Essay",
            "category": "assessment",
            "ai_support": "full",
            "bloom_levels": ["create", "evaluate"],
            "description": "Essay writing with keyword-based evaluation",
            "use_cases": ["Extended response", "Critical thinking", "Written expression"],
            "requires_media": False,
        },
        "flashcards": {
            "type_id": "H5P.Flashcards",
            "name": "Flashcards",
            "category": "vocabulary",
            "ai_support": "full",
            "bloom_levels": ["remember"],
            "description": "Flip cards for term-definition memorization",
            "use_cases": ["Vocabulary learning", "Concept memorization", "Quick review"],
            "requires_media": False,
        },
        "dialog-cards": {
            "type_id": "H5P.Dialogcards",
            "name": "Dialog Cards",
            "category": "vocabulary",
            "ai_support": "full",
            "bloom_levels": ["remember", "understand"],
            "description": "Flip cards with dialog/conversation format",
            "use_cases": ["Language learning", "Q&A practice", "Concept exploration"],
            "requires_media": False,
        },
        "crossword": {
            "type_id": "H5P.Crossword",
            "name": "Crossword",
            "category": "vocabulary",
            "ai_support": "full",
            "bloom_levels": ["remember", "apply"],
            "description": "Interactive crossword puzzle",
            "use_cases": ["Vocabulary review", "Term definitions", "Fun learning"],
            "requires_media": False,
        },
        "word-search": {
            "type_id": "H5P.FindTheWords",
            "name": "Word Search",
            "category": "vocabulary",
            "ai_support": "full",
            "bloom_levels": ["remember"],
            "description": "Find hidden words in a grid",
            "use_cases": ["Vocabulary reinforcement", "Spelling practice", "Engagement activity"],
            "requires_media": False,
        },
        "summary": {
            "type_id": "H5P.Summary",
            "name": "Summary",
            "category": "learning",
            "ai_support": "full",
            "bloom_levels": ["analyze"],
            "description": "Interactive summary with key statements",
            "use_cases": ["Content review", "Key point identification", "Comprehension check"],
            "requires_media": False,
        },
        "accordion": {
            "type_id": "H5P.Accordion",
            "name": "Accordion",
            "category": "learning",
            "ai_support": "full",
            "bloom_levels": ["remember", "understand"],
            "description": "Expandable FAQ-style information panels",
            "use_cases": ["FAQ content", "Topic overview", "Reference material"],
            "requires_media": False,
        },
        "sort-paragraphs": {
            "type_id": "H5P.SortParagraphs",
            "name": "Sort Paragraphs",
            "category": "learning",
            "ai_support": "full",
            "bloom_levels": ["analyze"],
            "description": "Order paragraphs or steps correctly",
            "use_cases": ["Process ordering", "Story sequencing", "Logical flow"],
            "requires_media": False,
        },
        "arithmetic-quiz": {
            "type_id": "H5P.ArithmeticQuiz",
            "name": "Arithmetic Quiz",
            "category": "assessment",
            "ai_support": "full",
            "bloom_levels": ["apply"],
            "description": "Math calculation practice",
            "use_cases": ["Math drills", "Calculation practice", "Mental math"],
            "requires_media": False,
        },
        "cornell-notes": {
            "type_id": "H5P.CornellNotes",
            "name": "Cornell Notes",
            "category": "learning",
            "ai_support": "full",
            "bloom_levels": ["analyze"],
            "description": "Structured note-taking template",
            "use_cases": ["Note-taking", "Study guide", "Guided learning"],
            "requires_media": False,
        },
        "memory-game": {
            "type_id": "H5P.MemoryGame",
            "name": "Memory Game",
            "category": "game",
            "ai_support": "partial",
            "bloom_levels": ["remember"],
            "description": "Card matching memory game",
            "use_cases": ["Vocabulary matching", "Image-term association", "Fun review"],
            "requires_media": True,
        },
        "timeline": {
            "type_id": "H5P.Timeline",
            "name": "Timeline",
            "category": "game",
            "ai_support": "partial",
            "bloom_levels": ["understand"],
            "description": "Interactive chronological timeline",
            "use_cases": ["Historical events", "Process steps", "Chronological learning"],
            "requires_media": False,
        },
        "image-pairing": {
            "type_id": "H5P.ImagePair",
            "name": "Image Pairing",
            "category": "game",
            "ai_support": "partial",
            "bloom_levels": ["remember"],
            "description": "Match pairs of images",
            "use_cases": ["Visual matching", "Concept pairing", "Memory exercise"],
            "requires_media": True,
        },
        "image-sequencing": {
            "type_id": "H5P.ImageSequencing",
            "name": "Image Sequencing",
            "category": "game",
            "ai_support": "partial",
            "bloom_levels": ["analyze"],
            "description": "Arrange images in correct order",
            "use_cases": ["Process ordering", "Story sequencing", "Step arrangement"],
            "requires_media": True,
        },
        "chart": {
            "type_id": "H5P.Chart",
            "name": "Chart",
            "category": "media",
            "ai_support": "full",
            "bloom_levels": ["understand"],
            "description": "Interactive bar and pie charts",
            "use_cases": ["Data visualization", "Statistics display", "Comparison"],
            "requires_media": False,
        },
        "personality-quiz": {
            "type_id": "H5P.PersonalityQuiz",
            "name": "Personality Quiz",
            "category": "game",
            "ai_support": "full",
            "bloom_levels": ["understand"],
            "description": "Personality-style quiz with outcome-based results",
            "use_cases": ["Self-discovery", "Learning style assessment", "Engagement"],
            "requires_media": False,
        },
        "question-set": {
            "type_id": "H5P.QuestionSet",
            "name": "Question Set",
            "category": "assessment",
            "ai_support": "full",
            "bloom_levels": ["remember", "understand", "apply"],
            "description": "Composite quiz with multiple question types",
            "use_cases": ["Comprehensive quiz", "Mixed assessment", "Unit test"],
            "requires_media": False,
        },
        "course-presentation": {
            "type_id": "H5P.CoursePresentation",
            "name": "Course Presentation",
            "category": "learning",
            "ai_support": "partial",
            "bloom_levels": ["understand", "apply"],
            "description": "Interactive slideshow with embedded activities",
            "use_cases": ["Lesson delivery", "Interactive lecture", "Self-paced learning"],
            "requires_media": True,
        },
        "interactive-book": {
            "type_id": "H5P.InteractiveBook",
            "name": "Interactive Book",
            "category": "learning",
            "ai_support": "partial",
            "bloom_levels": ["understand", "apply", "analyze"],
            "description": "Multi-chapter book with embedded content",
            "use_cases": ["Comprehensive learning", "Self-paced course", "Reference material"],
            "requires_media": True,
        },
        "branching-scenario": {
            "type_id": "H5P.BranchingScenario",
            "name": "Branching Scenario",
            "category": "learning",
            "ai_support": "partial",
            "bloom_levels": ["evaluate"],
            "description": "Decision-based learning path",
            "use_cases": ["Decision training", "Scenario-based learning", "Soft skills"],
            "requires_media": True,
        },
        "image-hotspots": {
            "type_id": "H5P.ImageHotspots",
            "name": "Image Hotspots",
            "category": "media",
            "ai_support": "partial",
            "bloom_levels": ["understand"],
            "description": "Interactive image with clickable hotspots",
            "use_cases": ["Diagram exploration", "Labeled images", "Visual learning"],
            "requires_media": True,
        },
        "documentation-tool": {
            "type_id": "H5P.DocumentationTool",
            "name": "Documentation Tool",
            "category": "learning",
            "ai_support": "full",
            "bloom_levels": ["create"],
            "description": "Structured forms for guided documentation",
            "use_cases": ["Lab reports", "Project documentation", "Guided writing"],
            "requires_media": False,
        },
        "agamotto": {
            "type_id": "H5P.Agamotto",
            "name": "Agamotto (Image Slider)",
            "category": "media",
            "ai_support": "partial",
            "bloom_levels": ["understand"],
            "description": "Image sequence slider for process visualization",
            "use_cases": ["Process visualization", "Before/after comparison", "Step-by-step"],
            "requires_media": True,
        },
        "image-juxtaposition": {
            "type_id": "H5P.ImageJuxtaposition",
            "name": "Image Juxtaposition",
            "category": "media",
            "ai_support": "partial",
            "bloom_levels": ["analyze"],
            "description": "Before/after comparison slider",
            "use_cases": ["Comparison", "Change visualization", "Analysis"],
            "requires_media": True,
        },
        "collage": {
            "type_id": "H5P.Collage",
            "name": "Collage",
            "category": "media",
            "ai_support": "partial",
            "bloom_levels": ["understand"],
            "description": "Visual collections for visual learning",
            "use_cases": ["Image gallery", "Visual overview", "Collection display"],
            "requires_media": True,
        },
    }

    # Category definitions
    CATEGORIES = {
        "assessment": {
            "name": "Assessment",
            "description": "Quiz and test content types for evaluating learning",
            "icon": "clipboard-check",
        },
        "vocabulary": {
            "name": "Vocabulary",
            "description": "Content types for vocabulary and term learning",
            "icon": "book-open",
        },
        "learning": {
            "name": "Learning",
            "description": "Content types for teaching and presenting information",
            "icon": "graduation-cap",
        },
        "game": {
            "name": "Game",
            "description": "Game-based and interactive content types",
            "icon": "puzzle-piece",
        },
        "media": {
            "name": "Media",
            "description": "Media-rich content types with images and charts",
            "icon": "image",
        },
    }

    @property
    def name(self) -> str:
        return "get_h5p_content_types"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_h5p_content_types",
                "description": (
                    "Get list of supported H5P content types with their capabilities, "
                    "AI support level, Bloom's taxonomy alignment, and recommended use cases. "
                    "Use this to show available content types to users or filter by category."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Filter by category: assessment, vocabulary, learning, game, media, or all",
                            "enum": ["all", "assessment", "vocabulary", "learning", "game", "media"],
                        },
                        "ai_supported_only": {
                            "type": "boolean",
                            "description": "Only return content types with full AI generation support",
                        },
                        "bloom_level": {
                            "type": "string",
                            "description": "Filter by Bloom's taxonomy level",
                            "enum": ["remember", "understand", "apply", "analyze", "evaluate", "create"],
                        },
                    },
                    "required": [],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the tool to get H5P content types."""
        category = params.get("category", "all")
        ai_only = params.get("ai_supported_only", False)
        bloom_level = params.get("bloom_level")

        # Filter content types
        filtered_types = []
        for type_id, type_info in self.CONTENT_TYPES.items():
            # Category filter
            if category != "all" and type_info["category"] != category:
                continue

            # AI support filter
            if ai_only and type_info["ai_support"] != "full":
                continue

            # Bloom's level filter
            if bloom_level and bloom_level not in type_info["bloom_levels"]:
                continue

            filtered_types.append({
                "id": type_id,
                **type_info,
            })

        # Sort by category, then name
        filtered_types.sort(key=lambda x: (x["category"], x["name"]))

        # Build message for LLM
        if category != "all":
            category_name = self.CATEGORIES.get(category, {}).get("name", category)
            message = f"Found {len(filtered_types)} {category_name} content types"
        else:
            message = f"Found {len(filtered_types)} H5P content types"

        if ai_only:
            message += " with full AI support"
        if bloom_level:
            message += f" for '{bloom_level}' level"

        # Format list for message
        type_list = []
        for ct in filtered_types:
            type_list.append(
                f"- {ct['name']} ({ct['id']}): {ct['description']} "
                f"[AI: {ct['ai_support']}, Media: {'Yes' if ct['requires_media'] else 'No'}]"
            )

        message += ":\n" + "\n".join(type_list)

        # Build UI element for selection
        ui_options = []
        for ct in filtered_types:
            ui_options.append(
                UIElementOption(
                    id=ct["id"],
                    label=ct["name"],
                    description=ct["description"],
                    icon=ct["category"],
                    metadata={
                        "type_id": ct["type_id"],
                        "category": ct["category"],
                        "ai_support": ct["ai_support"],
                        "bloom_levels": ct["bloom_levels"],
                        "requires_media": ct["requires_media"],
                    },
                )
            )

        ui_element = UIElement(
            type=UIElementType.SINGLE_SELECT,
            id=f"h5p_content_type_selection_{int(time.time() * 1000)}",
            title="Select Content Type",
            options=ui_options,
            searchable=True,
            allow_text_input=False,
            placeholder="Choose a content type...",
        )

        return ToolResult(
            success=True,
            data={
                "content_types": filtered_types,
                "categories": self.CATEGORIES,
                "count": len(filtered_types),
                "message": message,
            },
            ui_element=ui_element,
            passthrough_data={
                "content_types": filtered_types,
                "filter": {
                    "category": category,
                    "ai_supported_only": ai_only,
                    "bloom_level": bloom_level,
                },
            },
        )

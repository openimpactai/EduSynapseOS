# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Recommend Content Types Tool.

Recommends H5P content types based on learning objective,
topic complexity, and target Bloom's level.
"""

import logging
import time
from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult, UIElement, UIElementOption, UIElementType

logger = logging.getLogger(__name__)


class RecommendContentTypesTool(BaseTool):
    """Recommend H5P content types based on learning context.

    Analyzes the learning objective, topic, and purpose to recommend
    the most appropriate H5P content types with rationale.
    """

    # Recommendation rules based on purpose and Bloom's level
    PURPOSE_RECOMMENDATIONS = {
        "learn": {
            "primary": ["accordion", "dialog-cards", "course-presentation", "timeline"],
            "secondary": ["flashcards", "summary"],
            "description": "Content types for introducing new concepts",
        },
        "practice": {
            "primary": ["fill-blanks", "drag-words", "multiple-choice", "single-choice-set"],
            "secondary": ["flashcards", "crossword"],
            "description": "Content types for practicing skills",
        },
        "assess": {
            "primary": ["multiple-choice", "true-false", "question-set", "essay"],
            "secondary": ["fill-blanks", "summary"],
            "description": "Content types for evaluating learning",
        },
        "review": {
            "primary": ["flashcards", "crossword", "word-search", "single-choice-set"],
            "secondary": ["memory-game", "dialog-cards"],
            "description": "Content types for reviewing material",
        },
        "engage": {
            "primary": ["memory-game", "personality-quiz", "branching-scenario"],
            "secondary": ["crossword", "word-search", "timeline"],
            "description": "Content types for engagement and fun",
        },
    }

    BLOOM_RECOMMENDATIONS = {
        "remember": ["flashcards", "true-false", "word-search", "single-choice-set"],
        "understand": ["multiple-choice", "dialog-cards", "accordion", "timeline"],
        "apply": ["fill-blanks", "drag-words", "arithmetic-quiz"],
        "analyze": ["mark-words", "sort-paragraphs", "summary", "image-juxtaposition"],
        "evaluate": ["essay", "branching-scenario"],
        "create": ["course-presentation", "interactive-book", "documentation-tool"],
    }

    CONTENT_TYPE_INFO = {
        "multiple-choice": {
            "name": "Multiple Choice",
            "strengths": ["Quick to create", "Easy to grade", "Familiar format"],
            "best_for": ["Knowledge check", "Quick assessment", "Review"],
        },
        "true-false": {
            "name": "True/False",
            "strengths": ["Very quick", "Tests misconceptions", "Binary decision"],
            "best_for": ["Fact checking", "Misconception test", "Quick review"],
        },
        "fill-blanks": {
            "name": "Fill in the Blanks",
            "strengths": ["Tests recall", "Context-based", "Active typing"],
            "best_for": ["Vocabulary", "Key terms", "Grammar"],
        },
        "drag-words": {
            "name": "Drag the Words",
            "strengths": ["Interactive", "Tests placement", "Less typing"],
            "best_for": ["Sentence completion", "Term placement", "Sequencing"],
        },
        "flashcards": {
            "name": "Flashcards",
            "strengths": ["Self-paced", "Term-definition pairs", "Memorization"],
            "best_for": ["Vocabulary learning", "Quick review", "Definitions"],
        },
        "dialog-cards": {
            "name": "Dialog Cards",
            "strengths": ["Q&A format", "Hints available", "Language learning"],
            "best_for": ["Conversation practice", "Q&A learning", "Exploration"],
        },
        "crossword": {
            "name": "Crossword",
            "strengths": ["Fun engagement", "Vocabulary focus", "Challenge"],
            "best_for": ["Vocabulary review", "Term definitions", "Engagement"],
        },
        "word-search": {
            "name": "Word Search",
            "strengths": ["Low pressure", "Fun activity", "Recognition"],
            "best_for": ["Vocabulary reinforcement", "Brain break", "Warm-up"],
        },
        "accordion": {
            "name": "Accordion",
            "strengths": ["Organized info", "FAQ format", "Self-directed"],
            "best_for": ["Reference material", "FAQ", "Topic overview"],
        },
        "summary": {
            "name": "Summary",
            "strengths": ["Tests comprehension", "Statement evaluation"],
            "best_for": ["Content review", "Comprehension check", "Analysis"],
        },
        "sort-paragraphs": {
            "name": "Sort Paragraphs",
            "strengths": ["Tests sequencing", "Logical ordering", "Process steps"],
            "best_for": ["Process ordering", "Story sequencing", "Timelines"],
        },
        "essay": {
            "name": "Essay",
            "strengths": ["Open-ended", "Deep thinking", "Written expression"],
            "best_for": ["Extended response", "Critical thinking", "Synthesis"],
        },
        "question-set": {
            "name": "Question Set",
            "strengths": ["Mixed types", "Comprehensive", "Flexible"],
            "best_for": ["Unit assessment", "Comprehensive quiz", "Varied practice"],
        },
        "memory-game": {
            "name": "Memory Game",
            "strengths": ["Fun engagement", "Visual matching", "Gamification"],
            "best_for": ["Vocabulary matching", "Image association", "Fun review"],
        },
        "timeline": {
            "name": "Timeline",
            "strengths": ["Chronological", "Visual history", "Interactive"],
            "best_for": ["Historical events", "Process steps", "Sequences"],
        },
        "personality-quiz": {
            "name": "Personality Quiz",
            "strengths": ["Engagement", "Self-discovery", "Low stakes"],
            "best_for": ["Learning styles", "Self-assessment", "Engagement"],
        },
        "course-presentation": {
            "name": "Course Presentation",
            "strengths": ["Comprehensive", "Embedded activities", "Slide-based"],
            "best_for": ["Lesson delivery", "Interactive lecture", "Self-paced"],
        },
        "interactive-book": {
            "name": "Interactive Book",
            "strengths": ["Multi-chapter", "Comprehensive", "Self-paced"],
            "best_for": ["Complete course", "Reference material", "Deep learning"],
        },
        "branching-scenario": {
            "name": "Branching Scenario",
            "strengths": ["Decision-based", "Real-world", "Consequences"],
            "best_for": ["Decision training", "Soft skills", "Scenario practice"],
        },
    }

    @property
    def name(self) -> str:
        return "recommend_content_types"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "recommend_content_types",
                "description": (
                    "Recommend H5P content types based on learning objective, "
                    "topic complexity, and purpose. Returns ranked recommendations "
                    "with rationale and suggested combinations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The topic or subject to create content for",
                        },
                        "learning_objective": {
                            "type": "string",
                            "description": "The specific learning objective or goal",
                        },
                        "bloom_level": {
                            "type": "string",
                            "description": "Target Bloom's taxonomy level",
                            "enum": ["remember", "understand", "apply", "analyze", "evaluate", "create"],
                        },
                        "grade_level": {
                            "type": "integer",
                            "description": "Target grade level (1-12)",
                        },
                        "purpose": {
                            "type": "string",
                            "description": "Purpose of the content",
                            "enum": ["learn", "practice", "assess", "review", "engage"],
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of recommendations to return",
                        },
                    },
                    "required": ["topic"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the tool to get content type recommendations."""
        topic = params.get("topic", "")
        learning_objective = params.get("learning_objective", "")
        bloom_level = params.get("bloom_level")
        grade_level = params.get("grade_level", context.grade_level or 5)
        purpose = params.get("purpose", "practice")
        count = params.get("count", 5)

        if not topic:
            return ToolResult(
                success=False,
                data={"message": "topic is required"},
                error="Missing required parameter: topic",
            )

        # Build recommendations
        recommendations = []
        scores = {}

        # Score based on purpose
        purpose_info = self.PURPOSE_RECOMMENDATIONS.get(purpose, self.PURPOSE_RECOMMENDATIONS["practice"])
        for ct in purpose_info["primary"]:
            scores[ct] = scores.get(ct, 0) + 30
        for ct in purpose_info["secondary"]:
            scores[ct] = scores.get(ct, 0) + 15

        # Score based on Bloom's level
        if bloom_level and bloom_level in self.BLOOM_RECOMMENDATIONS:
            for ct in self.BLOOM_RECOMMENDATIONS[bloom_level]:
                scores[ct] = scores.get(ct, 0) + 25

        # Adjust for grade level
        # Younger grades: favor simpler types
        if grade_level <= 3:
            for ct in ["flashcards", "true-false", "word-search", "memory-game"]:
                scores[ct] = scores.get(ct, 0) + 10
            for ct in ["essay", "branching-scenario", "course-presentation"]:
                scores[ct] = scores.get(ct, 0) - 15
        # Older grades: can handle complex types
        elif grade_level >= 9:
            for ct in ["essay", "branching-scenario", "interactive-book"]:
                scores[ct] = scores.get(ct, 0) + 10

        # Sort by score and take top recommendations
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        for content_type, score in sorted_types[:count]:
            if content_type not in self.CONTENT_TYPE_INFO:
                continue

            info = self.CONTENT_TYPE_INFO[content_type]

            # Build rationale
            rationale_parts = []
            if content_type in purpose_info["primary"]:
                rationale_parts.append(f"Primary choice for '{purpose}' purpose")
            elif content_type in purpose_info.get("secondary", []):
                rationale_parts.append(f"Good alternative for '{purpose}' purpose")

            if bloom_level and content_type in self.BLOOM_RECOMMENDATIONS.get(bloom_level, []):
                rationale_parts.append(f"Supports '{bloom_level}' level learning")

            rationale = ". ".join(rationale_parts) if rationale_parts else "General recommendation"

            recommendations.append({
                "content_type": content_type,
                "name": info["name"],
                "match_score": min(score, 100),
                "rationale": rationale,
                "strengths": info["strengths"],
                "best_for": info["best_for"],
                "suggested_count": self._suggest_count(content_type, purpose),
            })

        # Build learning path suggestion
        learning_path = self._build_learning_path(purpose, bloom_level, grade_level)

        # Build message
        message = f"Recommendations for '{topic}' ({purpose} purpose):\n\n"
        for i, rec in enumerate(recommendations, 1):
            message += f"{i}. **{rec['name']}** (Score: {rec['match_score']})\n"
            message += f"   {rec['rationale']}\n"
            message += f"   Best for: {', '.join(rec['best_for'])}\n\n"

        if learning_path:
            message += "\n**Suggested Learning Path:**\n"
            for i, step in enumerate(learning_path, 1):
                message += f"{i}. {step['name']} - {step['purpose']}\n"

        # Build UI element
        ui_options = [
            UIElementOption(
                id=rec["content_type"],
                label=rec["name"],
                description=rec["rationale"],
                metadata={
                    "match_score": rec["match_score"],
                    "best_for": rec["best_for"],
                },
            )
            for rec in recommendations
        ]

        ui_element = UIElement(
            type=UIElementType.SINGLE_SELECT,
            id=f"content_type_recommendation_{int(time.time() * 1000)}",
            title="Recommended Content Types",
            options=ui_options,
            searchable=False,
            allow_text_input=False,
        )

        return ToolResult(
            success=True,
            data={
                "recommendations": recommendations,
                "learning_path": learning_path,
                "context": {
                    "topic": topic,
                    "purpose": purpose,
                    "bloom_level": bloom_level,
                    "grade_level": grade_level,
                },
                "message": message,
            },
            ui_element=ui_element,
            passthrough_data={
                "recommendations": recommendations,
                "learning_path": learning_path,
            },
        )

    def _suggest_count(self, content_type: str, purpose: str) -> int:
        """Suggest appropriate content count based on type and purpose."""
        if content_type in ["flashcards", "dialog-cards"]:
            return 10 if purpose == "review" else 15
        if content_type in ["multiple-choice", "true-false"]:
            return 10 if purpose == "assess" else 5
        if content_type in ["crossword", "word-search"]:
            return 10  # Number of words
        if content_type == "accordion":
            return 5  # Number of panels
        if content_type in ["fill-blanks", "drag-words"]:
            return 5  # Number of exercises
        return 5

    def _build_learning_path(
        self,
        purpose: str,
        bloom_level: str | None,
        grade_level: int,
    ) -> list[dict[str, str]]:
        """Build a suggested learning path."""
        paths = {
            "learn": [
                {"content_type": "accordion", "name": "Accordion", "purpose": "Introduction & Overview"},
                {"content_type": "flashcards", "name": "Flashcards", "purpose": "Key Vocabulary"},
                {"content_type": "multiple-choice", "name": "Multiple Choice", "purpose": "Comprehension Check"},
            ],
            "practice": [
                {"content_type": "flashcards", "name": "Flashcards", "purpose": "Term Review"},
                {"content_type": "fill-blanks", "name": "Fill Blanks", "purpose": "Application Practice"},
                {"content_type": "drag-words", "name": "Drag Words", "purpose": "Advanced Practice"},
            ],
            "assess": [
                {"content_type": "true-false", "name": "True/False", "purpose": "Quick Check"},
                {"content_type": "multiple-choice", "name": "Multiple Choice", "purpose": "Core Assessment"},
                {"content_type": "essay", "name": "Essay", "purpose": "Extended Response"},
            ],
            "review": [
                {"content_type": "flashcards", "name": "Flashcards", "purpose": "Quick Recall"},
                {"content_type": "crossword", "name": "Crossword", "purpose": "Fun Review"},
                {"content_type": "single-choice-set", "name": "Single Choice Set", "purpose": "Speed Drill"},
            ],
            "engage": [
                {"content_type": "personality-quiz", "name": "Personality Quiz", "purpose": "Hook Activity"},
                {"content_type": "memory-game", "name": "Memory Game", "purpose": "Interactive Learning"},
                {"content_type": "crossword", "name": "Crossword", "purpose": "Challenge Activity"},
            ],
        }

        path = paths.get(purpose, paths["practice"])

        # Adjust for younger grades
        if grade_level <= 3:
            # Remove complex types
            path = [p for p in path if p["content_type"] not in ["essay", "branching-scenario"]]

        return path

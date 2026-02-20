# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check Bloom's Taxonomy Alignment Tool.

Verifies content aligns with target Bloom's taxonomy level.
"""

import logging
from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class CheckBloomAlignmentTool(BaseTool):
    """Check Bloom's Taxonomy alignment of content.

    Evaluates whether content questions and activities align
    with the intended Bloom's taxonomy cognitive level.

    Example usage by agent:
        - "Check if these questions test understanding"
        - "Is this content at the analyze level?"
        - "Verify Bloom's alignment for this quiz"
    """

    BLOOM_LEVELS = {
        "remember": {
            "description": "Recall facts, terms, basic concepts",
            "verbs": ["define", "list", "name", "identify", "recall", "recognize", "state", "match"],
            "question_patterns": ["what is", "who was", "when did", "name the", "list the", "which of"],
        },
        "understand": {
            "description": "Explain ideas, interpret meaning",
            "verbs": ["explain", "describe", "summarize", "interpret", "classify", "compare", "paraphrase"],
            "question_patterns": ["explain why", "describe how", "what does", "summarize", "in your own words"],
        },
        "apply": {
            "description": "Use knowledge in new situations",
            "verbs": ["apply", "demonstrate", "solve", "use", "calculate", "predict", "construct"],
            "question_patterns": ["solve", "calculate", "how would you", "demonstrate", "apply this to"],
        },
        "analyze": {
            "description": "Break down information, find patterns",
            "verbs": ["analyze", "compare", "contrast", "differentiate", "examine", "categorize", "investigate"],
            "question_patterns": ["compare and contrast", "what is the difference", "analyze", "why do you think"],
        },
        "evaluate": {
            "description": "Make judgments, defend positions",
            "verbs": ["evaluate", "judge", "critique", "justify", "assess", "argue", "defend", "prioritize"],
            "question_patterns": ["which is better", "do you agree", "evaluate", "justify", "what would you recommend"],
        },
        "create": {
            "description": "Produce original work, design solutions",
            "verbs": ["create", "design", "compose", "develop", "construct", "formulate", "generate", "plan"],
            "question_patterns": ["design", "create", "develop", "what if", "compose", "propose"],
        },
    }

    @property
    def name(self) -> str:
        return "check_bloom_alignment"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "check_bloom_alignment",
                "description": (
                    "Check if content aligns with the target Bloom's Taxonomy level. "
                    "Evaluates cognitive level of questions and activities."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content_type": {
                            "type": "string",
                            "description": "Type of content being checked",
                        },
                        "ai_content": {
                            "type": "object",
                            "description": "Content to evaluate",
                        },
                        "target_bloom_level": {
                            "type": "string",
                            "enum": ["remember", "understand", "apply", "analyze", "evaluate", "create"],
                            "description": "Expected Bloom's taxonomy level",
                        },
                    },
                    "required": ["content_type", "ai_content", "target_bloom_level"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute Bloom's alignment check."""
        content_type = params.get("content_type", "")
        ai_content = params.get("ai_content", {})
        target_level = params.get("target_bloom_level", "understand")

        if not ai_content:
            return ToolResult(
                success=False,
                data={"message": "Content is required"},
                error="Missing required parameter: ai_content",
            )

        if target_level not in self.BLOOM_LEVELS:
            return ToolResult(
                success=False,
                data={"message": f"Invalid Bloom's level: {target_level}"},
                error="Invalid target_bloom_level",
            )

        try:
            # Analyze each item in content
            analysis_results = []

            # Analyze questions
            for i, q in enumerate(ai_content.get("questions", [])):
                question_text = q.get("question", q.get("text", ""))
                detected_level = self._detect_bloom_level(question_text)
                analysis_results.append({
                    "item": f"Question {i+1}",
                    "text": question_text[:100],
                    "detected_level": detected_level,
                    "target_level": target_level,
                    "aligned": self._is_level_aligned(detected_level, target_level),
                })

            # Analyze statements
            for i, s in enumerate(ai_content.get("statements", [])):
                statement = s.get("statement", "")
                detected_level = self._detect_bloom_level(statement)
                analysis_results.append({
                    "item": f"Statement {i+1}",
                    "text": statement[:100],
                    "detected_level": detected_level,
                    "target_level": target_level,
                    "aligned": self._is_level_aligned(detected_level, target_level),
                })

            # Calculate alignment score
            if analysis_results:
                aligned_count = sum(1 for r in analysis_results if r.get("aligned", False))
                total_count = len(analysis_results)
                alignment_score = (aligned_count / total_count) * 100
            else:
                alignment_score = 100
                aligned_count = 0
                total_count = 0

            # Identify misaligned items
            misaligned = [r for r in analysis_results if not r.get("aligned", True)]

            target_info = self.BLOOM_LEVELS[target_level]

            logger.info(
                "Bloom alignment check: type=%s, target=%s, score=%.0f%%",
                content_type,
                target_level,
                alignment_score,
            )

            return ToolResult(
                success=True,
                data={
                    "alignment_score": alignment_score,
                    "target_level": target_level,
                    "target_description": target_info["description"],
                    "items_analyzed": total_count,
                    "items_aligned": aligned_count,
                    "misaligned_items": misaligned,
                    "analysis_results": analysis_results,
                    "suggestions": self._get_alignment_suggestions(target_level, misaligned),
                    "message": (
                        f"Bloom's Taxonomy alignment: {alignment_score:.0f}% "
                        f"({aligned_count}/{total_count} items align with '{target_level}' level)"
                    ),
                },
            )

        except Exception as e:
            logger.exception("Error checking Bloom alignment")
            return ToolResult(
                success=False,
                data={"message": f"Failed to check alignment: {e}"},
                error=str(e),
            )

    def _detect_bloom_level(self, text: str) -> str:
        """Detect Bloom's taxonomy level from text."""
        text_lower = text.lower()

        # Check patterns in order from highest to lowest
        for level in ["create", "evaluate", "analyze", "apply", "understand", "remember"]:
            level_info = self.BLOOM_LEVELS[level]

            # Check for verbs
            for verb in level_info["verbs"]:
                if verb in text_lower:
                    return level

            # Check for question patterns
            for pattern in level_info["question_patterns"]:
                if pattern in text_lower:
                    return level

        # Default to remember for basic factual questions
        return "remember"

    def _is_level_aligned(self, detected: str, target: str) -> bool:
        """Check if detected level aligns with target."""
        levels_order = ["remember", "understand", "apply", "analyze", "evaluate", "create"]

        detected_idx = levels_order.index(detected)
        target_idx = levels_order.index(target)

        # Allow one level variation
        return abs(detected_idx - target_idx) <= 1

    def _get_alignment_suggestions(self, target_level: str, misaligned: list) -> list[str]:
        """Get suggestions for improving alignment."""
        suggestions = []
        target_info = self.BLOOM_LEVELS[target_level]

        if misaligned:
            suggestions.append(
                f"Use verbs like: {', '.join(target_info['verbs'][:5])} "
                f"to target the '{target_level}' level"
            )
            suggestions.append(
                f"Question patterns for '{target_level}': "
                f"{', '.join(target_info['question_patterns'][:3])}"
            )

            # Check if items are too low or too high
            levels_order = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
            target_idx = levels_order.index(target_level)

            too_low = sum(1 for m in misaligned if levels_order.index(m["detected_level"]) < target_idx)
            too_high = sum(1 for m in misaligned if levels_order.index(m["detected_level"]) > target_idx)

            if too_low > too_high:
                suggestions.append(
                    "Some items test lower cognitive levels. "
                    "Consider requiring more complex thinking."
                )
            elif too_high > too_low:
                suggestions.append(
                    "Some items test higher cognitive levels than intended. "
                    "Consider simplifying if needed."
                )

        return suggestions

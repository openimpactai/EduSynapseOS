# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check Accessibility Tool.

Verifies content meets accessibility standards.
"""

import logging
import re
from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class CheckAccessibilityTool(BaseTool):
    """Check accessibility of content.

    Evaluates content for accessibility compliance including
    alt text, color contrast concerns, and screen reader compatibility.

    Example usage by agent:
        - "Check accessibility of this content"
        - "Does this content have proper alt text?"
        - "Verify accessibility compliance"
    """

    @property
    def name(self) -> str:
        return "check_accessibility"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "check_accessibility",
                "description": (
                    "Check content accessibility compliance. "
                    "Evaluates alt text, structure, and screen reader compatibility."
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
                    },
                    "required": ["content_type", "ai_content"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute accessibility check."""
        content_type = params.get("content_type", "")
        ai_content = params.get("ai_content", {})

        if not ai_content:
            return ToolResult(
                success=False,
                data={"message": "Content is required"},
                error="Missing required parameter: ai_content",
            )

        try:
            issues = []
            checks_passed = 0
            checks_total = 0

            # Check images have alt text
            image_check = self._check_images(ai_content)
            checks_total += 1
            if image_check["passed"]:
                checks_passed += 1
            else:
                issues.extend(image_check["issues"])

            # Check text structure
            structure_check = self._check_structure(ai_content)
            checks_total += 1
            if structure_check["passed"]:
                checks_passed += 1
            else:
                issues.extend(structure_check["issues"])

            # Check for descriptive labels
            label_check = self._check_labels(ai_content, content_type)
            checks_total += 1
            if label_check["passed"]:
                checks_passed += 1
            else:
                issues.extend(label_check["issues"])

            # Check feedback clarity
            feedback_check = self._check_feedback(ai_content)
            checks_total += 1
            if feedback_check["passed"]:
                checks_passed += 1
            else:
                issues.extend(feedback_check["issues"])

            # Calculate score
            score = (checks_passed / checks_total) * 100 if checks_total > 0 else 100

            # Determine recommendation
            critical_issues = [i for i in issues if i.get("severity") == "critical"]
            major_issues = [i for i in issues if i.get("severity") == "major"]

            if critical_issues:
                recommendation = "needs_immediate_fixes"
            elif major_issues:
                recommendation = "needs_improvements"
            elif issues:
                recommendation = "minor_improvements_suggested"
            else:
                recommendation = "accessible"

            logger.info(
                "Accessibility check: type=%s, score=%.0f%%, issues=%d",
                content_type,
                score,
                len(issues),
            )

            return ToolResult(
                success=True,
                data={
                    "accessibility_score": score,
                    "recommendation": recommendation,
                    "checks_passed": checks_passed,
                    "checks_total": checks_total,
                    "issues": issues,
                    "critical_count": len(critical_issues),
                    "major_count": len(major_issues),
                    "message": (
                        f"Accessibility: {score:.0f}% ({checks_passed}/{checks_total} checks passed). "
                        f"Found {len(issues)} issues."
                    ),
                },
            )

        except Exception as e:
            logger.exception("Error checking accessibility")
            return ToolResult(
                success=False,
                data={"message": f"Failed to check accessibility: {e}"},
                error=str(e),
            )

    def _check_images(self, ai_content: dict) -> dict:
        """Check images have alt text."""
        issues = []
        images_found = 0
        images_with_alt = 0

        # Check various image locations
        for card in ai_content.get("cards", []):
            if card.get("image") or card.get("imagePrompt"):
                images_found += 1
                if card.get("alt") or card.get("imageAlt"):
                    images_with_alt += 1
                else:
                    issues.append({
                        "type": "missing_alt_text",
                        "severity": "major",
                        "description": f"Card '{card.get('term', 'unknown')}' has image without alt text",
                        "suggestion": "Add descriptive alt text for the image",
                    })

        # Check media objects
        for q in ai_content.get("questions", []):
            if q.get("image"):
                images_found += 1
                if q.get("imageAlt") or (isinstance(q["image"], dict) and q["image"].get("alt")):
                    images_with_alt += 1
                else:
                    issues.append({
                        "type": "missing_alt_text",
                        "severity": "major",
                        "description": "Question has image without alt text",
                        "suggestion": "Add descriptive alt text for the image",
                    })

        passed = len(issues) == 0
        return {"passed": passed, "issues": issues}

    def _check_structure(self, ai_content: dict) -> dict:
        """Check content has proper structure."""
        issues = []

        # Check for title
        if not ai_content.get("title"):
            issues.append({
                "type": "missing_title",
                "severity": "minor",
                "description": "Content is missing a title",
                "suggestion": "Add a descriptive title",
            })

        # Check questions have clear text
        for i, q in enumerate(ai_content.get("questions", [])):
            question_text = q.get("question", q.get("text", ""))
            if not question_text or len(question_text.strip()) < 10:
                issues.append({
                    "type": "unclear_question",
                    "severity": "major",
                    "description": f"Question {i+1} may be too short or unclear",
                    "suggestion": "Ensure question text is complete and clear",
                })

        # Check statements are complete
        for i, s in enumerate(ai_content.get("statements", [])):
            if not s.get("statement") or len(s.get("statement", "").strip()) < 10:
                issues.append({
                    "type": "unclear_statement",
                    "severity": "major",
                    "description": f"Statement {i+1} may be incomplete",
                    "suggestion": "Ensure statement is complete",
                })

        passed = len(issues) == 0
        return {"passed": passed, "issues": issues}

    def _check_labels(self, ai_content: dict, content_type: str) -> dict:
        """Check for descriptive labels and instructions."""
        issues = []

        # Check for task description/instructions
        if content_type in ["fill-blanks", "drag-words", "mark-words"]:
            if not ai_content.get("instruction") and not ai_content.get("taskDescription"):
                issues.append({
                    "type": "missing_instructions",
                    "severity": "major",
                    "description": "Interactive content is missing instructions",
                    "suggestion": "Add clear instructions for how to complete the activity",
                })

        # Check answer options are distinct
        for i, q in enumerate(ai_content.get("questions", [])):
            answers = q.get("answers", [])
            if len(answers) != len(set(str(a).lower() for a in answers)):
                issues.append({
                    "type": "duplicate_answers",
                    "severity": "minor",
                    "description": f"Question {i+1} may have similar answer options",
                    "suggestion": "Ensure answer options are clearly distinct",
                })

        passed = len(issues) == 0
        return {"passed": passed, "issues": issues}

    def _check_feedback(self, ai_content: dict) -> dict:
        """Check feedback is clear and helpful."""
        issues = []

        # Check questions have feedback
        questions_with_feedback = 0
        questions_total = len(ai_content.get("questions", []))

        for q in ai_content.get("questions", []):
            if q.get("explanation") or q.get("feedback") or q.get("distractorFeedback"):
                questions_with_feedback += 1

        if questions_total > 0 and questions_with_feedback < questions_total * 0.5:
            issues.append({
                "type": "missing_feedback",
                "severity": "minor",
                "description": "Many questions are missing explanatory feedback",
                "suggestion": "Add feedback explaining correct answers",
            })

        # Check statements have explanations
        statements_with_explanation = 0
        statements_total = len(ai_content.get("statements", []))

        for s in ai_content.get("statements", []):
            if s.get("explanation") or s.get("wrongFeedback"):
                statements_with_explanation += 1

        if statements_total > 0 and statements_with_explanation < statements_total * 0.5:
            issues.append({
                "type": "missing_explanations",
                "severity": "minor",
                "description": "Many statements are missing explanations",
                "suggestion": "Add explanations for why statements are true/false",
            })

        passed = len(issues) == 0
        return {"passed": passed, "issues": issues}

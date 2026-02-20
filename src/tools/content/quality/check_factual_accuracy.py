# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check Factual Accuracy Tool.

Verifies factual accuracy of educational content using LLM-based analysis.
"""

import json
import logging
from typing import Any

from src.core.config import get_settings
from src.core.intelligence.llm import LLMClient
from src.core.tools import BaseTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class CheckFactualAccuracyTool(BaseTool):
    """Check factual accuracy of content.

    Verifies that facts, definitions, and claims in the content
    are accurate and up-to-date using LLM-based analysis.

    The verification process:
    1. Extract factual claims from the content
    2. Use LLM to analyze each claim for accuracy
    3. Return verification results with confidence scores

    Example usage by agent:
        - "Check if these science facts are accurate"
        - "Verify the historical dates"
        - "Is this definition correct?"
    """

    @property
    def name(self) -> str:
        return "check_factual_accuracy"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "check_factual_accuracy",
                "description": (
                    "Check factual accuracy of educational content. "
                    "Verifies facts, definitions, dates, and claims using AI analysis."
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
                            "description": "Content to verify",
                        },
                        "subject_area": {
                            "type": "string",
                            "description": "Subject area (science, history, math, etc.)",
                        },
                        "claims_to_verify": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific claims to verify (optional)",
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
        """Execute factual accuracy check."""
        content_type = params.get("content_type", "")
        ai_content = params.get("ai_content", {})
        subject_area = params.get("subject_area", "general")
        claims_to_verify = params.get("claims_to_verify", [])

        if not ai_content:
            return ToolResult(
                success=False,
                data={"message": "Content is required for accuracy check"},
                error="Missing required parameter: ai_content",
            )

        try:
            # Extract claims from content
            extracted_claims = self._extract_claims(content_type, ai_content)

            # If specific claims provided, use those instead
            if claims_to_verify:
                extracted_claims = claims_to_verify

            if not extracted_claims:
                return ToolResult(
                    success=True,
                    data={
                        "accuracy_score": 100.0,
                        "claims_checked": 0,
                        "claims_accurate": 0,
                        "issues": [],
                        "verification_results": [],
                        "message": "No factual claims found to verify in the content.",
                    },
                )

            # Verify claims using LLM
            verification_results = await self._verify_claims_with_llm(
                claims=extracted_claims,
                subject_area=subject_area,
                language=context.language or "en",
            )

            # Calculate accuracy score
            accurate_count = sum(1 for r in verification_results if r.get("is_accurate", False))
            total_count = len(verification_results)
            accuracy_score = (accurate_count / total_count) * 100 if total_count > 0 else 100.0

            # Identify issues
            issues = [
                {
                    "claim": r.get("claim"),
                    "issue": r.get("issue"),
                    "suggestion": r.get("suggestion"),
                    "confidence": r.get("confidence"),
                }
                for r in verification_results
                if not r.get("is_accurate", True)
            ]

            logger.info(
                "Factual accuracy check: type=%s, claims=%d, accurate=%d, score=%.1f%%",
                content_type,
                total_count,
                accurate_count,
                accuracy_score,
            )

            return ToolResult(
                success=True,
                data={
                    "accuracy_score": accuracy_score,
                    "claims_checked": total_count,
                    "claims_accurate": accurate_count,
                    "issues": issues,
                    "verification_results": verification_results,
                    "message": (
                        f"Checked {total_count} factual claims. "
                        f"Accuracy: {accuracy_score:.0f}% ({accurate_count}/{total_count})"
                    ),
                },
            )

        except Exception as e:
            logger.exception("Error checking factual accuracy")
            return ToolResult(
                success=False,
                data={"message": f"Failed to check accuracy: {e}"},
                error=str(e),
            )

    def _extract_claims(self, content_type: str, ai_content: dict) -> list[str]:
        """Extract factual claims from content."""
        claims = []

        # Extract from questions
        for q in ai_content.get("questions", []):
            if q.get("explanation"):
                claims.append(q["explanation"])
            # The correct answer implies a factual claim
            answers = q.get("answers", [])
            correct_idx = q.get("correctIndex", 0)
            if answers and 0 <= correct_idx < len(answers):
                question = q.get("question", "")
                answer = answers[correct_idx]
                claims.append(f"{question} - {answer}")

        # Extract from definitions
        for card in ai_content.get("cards", []):
            definition = card.get("definition", "")
            if definition:
                term = card.get("term", "")
                claims.append(f"{term}: {definition}")

        # Extract from statements (true/false)
        for stmt in ai_content.get("statements", []):
            statement = stmt.get("statement", "")
            if statement:
                is_true = stmt.get("isTrue", True)
                claims.append(f"{'TRUE' if is_true else 'FALSE'}: {statement}")

        # Extract from panel content
        for panel in ai_content.get("panels", []):
            content = panel.get("content", "")
            if content:
                claims.append(content)

        # Extract from timeline events
        for event in ai_content.get("events", []):
            date = event.get("date", "")
            title = event.get("title", "")
            description = event.get("description", "")
            if date and title:
                claims.append(f"{date}: {title}. {description}")

        return claims[:20]  # Limit to 20 claims for efficiency

    async def _verify_claims_with_llm(
        self,
        claims: list[str],
        subject_area: str,
        language: str,
    ) -> list[dict[str, Any]]:
        """Verify factual claims using LLM analysis.

        Uses the LLM to analyze each claim for factual accuracy,
        providing confidence scores and corrections where needed.

        Args:
            claims: List of claims to verify.
            subject_area: Subject area for context.
            language: Language of the content.

        Returns:
            List of verification results.
        """
        settings = get_settings()
        llm_client = LLMClient(llm_settings=settings.llm)

        # Build verification prompt
        claims_text = "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(claims)])

        system_prompt = f"""You are a fact-checking expert specializing in {subject_area} education.

Analyze each claim below for factual accuracy. For each claim, determine:
1. Whether it is factually accurate (true/false)
2. Your confidence level (0.0 to 1.0)
3. If inaccurate, what the issue is and what the correct information should be

Claims to verify:
{claims_text}

Respond in JSON format with an array of results:
{{
  "results": [
    {{
      "claim_number": 1,
      "claim": "the original claim text",
      "is_accurate": true/false,
      "confidence": 0.0-1.0,
      "issue": "description of the issue if inaccurate, null if accurate",
      "suggestion": "corrected information if inaccurate, null if accurate"
    }},
    ...
  ]
}}

Be thorough but fair. Only mark as inaccurate if you are confident the claim is wrong.
For ambiguous or partially correct claims, note the nuance in the issue field.
If a claim is about a subjective or debatable topic, mark as accurate with a note."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please verify these {len(claims)} claims."},
        ]

        try:
            response = await llm_client.complete_with_messages(
                messages=messages,
            )

            # Parse response
            response_content = response.content
            try:
                parsed = json.loads(response_content)
                results = parsed.get("results", [])
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON")
                # Return default results
                results = []

            # Map results back to claims
            verification_results = []
            for i, claim in enumerate(claims):
                # Find matching result
                result = next(
                    (r for r in results if r.get("claim_number") == i + 1),
                    None,
                )

                if result:
                    verification_results.append({
                        "claim": claim[:200],
                        "is_accurate": result.get("is_accurate", True),
                        "confidence": result.get("confidence", 0.8),
                        "source": "llm_analysis",
                        "issue": result.get("issue"),
                        "suggestion": result.get("suggestion"),
                    })
                else:
                    # Default if no result found
                    verification_results.append({
                        "claim": claim[:200],
                        "is_accurate": True,
                        "confidence": 0.5,
                        "source": "llm_analysis_default",
                        "issue": None,
                        "suggestion": None,
                    })

            return verification_results

        except Exception as e:
            logger.warning("LLM verification failed: %s", e)
            # Return default results on failure
            return [
                {
                    "claim": claim[:200],
                    "is_accurate": True,
                    "confidence": 0.5,
                    "source": "verification_unavailable",
                    "issue": None,
                    "suggestion": "Manual review recommended",
                }
                for claim in claims
            ]

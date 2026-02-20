# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Diagnostic analysis capability for learning pattern analysis.

This capability analyzes learning patterns to identify:
- Areas of struggle and their patterns
- Learning style preferences
- Effective and ineffective strategies
- Potential learning difficulties for specialist referral

IMPORTANT: This capability does NOT diagnose learning disabilities.
It only identifies patterns that may warrant professional evaluation.
Any suggestions for evaluation are presented with appropriate disclaimers.

The analysis includes:
- Pattern observations
- Struggle area identification
- Strategy effectiveness
- Recommendations for adaptation
- Flags for specialist referral (with disclaimers)
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.core.agents.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityError,
    CapabilityResult,
)


class PatternType(str, Enum):
    """Types of learning patterns observed."""

    TIME_BASED = "time_based"
    TOPIC_BASED = "topic_based"
    FORMAT_BASED = "format_based"
    DIFFICULTY_BASED = "difficulty_based"
    EMOTIONAL = "emotional"
    BEHAVIORAL = "behavioral"


class SeverityLevel(str, Enum):
    """Severity levels for observations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StrugglePattern(BaseModel):
    """A pattern of struggle observed in learning.

    Attributes:
        area: Topic or skill area.
        pattern: Description of the struggle pattern.
        frequency: How often this occurs.
        severity: How severe the struggle is.
        possible_causes: Potential reasons for struggle.
        observed_in: Where this was observed.
    """

    area: str = Field(description="Topic or skill area")
    pattern: str = Field(description="Description of the pattern")
    frequency: str = Field(description="How often this occurs")
    severity: SeverityLevel = Field(description="Severity level")
    possible_causes: list[str] = Field(
        default_factory=list,
        description="Potential reasons",
    )
    observed_in: list[str] = Field(
        default_factory=list,
        description="Where observed",
    )


class LearningPatternData(BaseModel):
    """Data about learning patterns for analysis.

    Attributes:
        session_history: Summary of recent sessions.
        error_patterns: Common types of errors made.
        time_patterns: Performance at different times.
        topic_performance: Performance by topic.
        response_times: Average response times.
        hint_usage: How hints were used.
        emotional_indicators: Emotional state observations.
        strategy_usage: Strategies used and their effectiveness.
    """

    session_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recent session summaries",
    )
    error_patterns: dict[str, int] = Field(
        default_factory=dict,
        description="Error type -> count",
    )
    time_patterns: dict[str, float] = Field(
        default_factory=dict,
        description="Time of day -> performance",
    )
    topic_performance: dict[str, float] = Field(
        default_factory=dict,
        description="Topic -> mastery level",
    )
    response_times: dict[str, float] = Field(
        default_factory=dict,
        description="Category -> avg response time",
    )
    hint_usage: dict[str, int] = Field(
        default_factory=dict,
        description="Hint level -> usage count",
    )
    emotional_indicators: list[str] = Field(
        default_factory=list,
        description="Observed emotional states",
    )
    strategy_usage: dict[str, float] = Field(
        default_factory=dict,
        description="Strategy -> success rate",
    )


class DiagnosticAnalysisParams(BaseModel):
    """Parameters for diagnostic analysis.

    Attributes:
        pattern_data: Learning pattern data to analyze.
        focus_areas: Specific areas to focus analysis on.
        include_referral_flags: Whether to include specialist referral flags.
        include_strategy_suggestions: Whether to suggest strategies.
        time_period_days: Period of data being analyzed.
        language: Language for the analysis.
    """

    pattern_data: LearningPatternData = Field(
        default_factory=LearningPatternData,
        description="Pattern data to analyze",
    )
    focus_areas: list[str] = Field(
        default_factory=list,
        description="Specific areas to focus on",
    )
    include_referral_flags: bool = Field(
        default=True,
        description="Include specialist referral flags",
    )
    include_strategy_suggestions: bool = Field(
        default=True,
        description="Include strategy suggestions",
    )
    time_period_days: int = Field(
        default=30,
        description="Period of data analyzed",
    )
    language: str = Field(
        default="en",
        description="Language for analysis",
    )


class PatternObservation(BaseModel):
    """An observed learning pattern.

    Attributes:
        pattern_type: Type of pattern.
        observation: What was observed.
        significance: How significant this is.
        evidence: Evidence supporting this observation.
        implication: What this means for learning.
    """

    pattern_type: PatternType = Field(description="Type of pattern")
    observation: str = Field(description="What was observed")
    significance: str = Field(description="Significance level")
    evidence: list[str] = Field(
        default_factory=list,
        description="Supporting evidence",
    )
    implication: str = Field(
        default="",
        description="Implication for learning",
    )


class StrategyRecommendation(BaseModel):
    """A recommended learning strategy.

    Attributes:
        strategy: The recommended strategy.
        rationale: Why this is recommended.
        expected_benefit: Expected outcome.
        implementation: How to implement.
    """

    strategy: str = Field(description="The strategy")
    rationale: str = Field(description="Why recommended")
    expected_benefit: str = Field(description="Expected outcome")
    implementation: str = Field(description="How to implement")


class ReferralFlag(BaseModel):
    """A flag suggesting specialist referral.

    Attributes:
        area: Area of concern.
        observations: What was observed.
        disclaimer: Important disclaimer text.
        suggestion: Suggested action.
    """

    area: str = Field(description="Area of concern")
    observations: list[str] = Field(description="What was observed")
    disclaimer: str = Field(
        default=(
            "This observation is based on learning patterns only. "
            "Only qualified professionals can provide diagnosis. "
            "Please consult appropriate specialists for evaluation."
        ),
        description="Important disclaimer",
    )
    suggestion: str = Field(description="Suggested action")


class DiagnosticResult(CapabilityResult):
    """Result of diagnostic analysis.

    Attributes:
        summary: Overall analysis summary.
        pattern_observations: Observed learning patterns.
        struggle_patterns: Identified struggle patterns.
        strengths_identified: Identified strengths.
        learning_style_insights: Insights about learning style.
        strategy_recommendations: Recommended strategies.
        referral_flags: Flags for specialist consideration.
        adaptation_suggestions: Suggestions for adapting instruction.
        confidence: Confidence in the analysis.
        data_quality: Quality of input data.
        language: Language of the result.
    """

    summary: str = Field(description="Overall analysis summary")
    pattern_observations: list[PatternObservation] = Field(
        default_factory=list,
        description="Observed patterns",
    )
    struggle_patterns: list[StrugglePattern] = Field(
        default_factory=list,
        description="Struggle patterns",
    )
    strengths_identified: list[str] = Field(
        default_factory=list,
        description="Identified strengths",
    )
    learning_style_insights: dict[str, str] = Field(
        default_factory=dict,
        description="Learning style insights",
    )
    strategy_recommendations: list[StrategyRecommendation] = Field(
        default_factory=list,
        description="Strategy recommendations",
    )
    referral_flags: list[ReferralFlag] = Field(
        default_factory=list,
        description="Specialist referral flags",
    )
    adaptation_suggestions: list[str] = Field(
        default_factory=list,
        description="Instruction adaptations",
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Analysis confidence",
    )
    data_quality: str = Field(
        default="sufficient",
        description="Quality of input data",
    )
    language: str = Field(default="en", description="Language")


class DiagnosticAnalysisCapability(Capability):
    """Capability for analyzing learning patterns.

    Analyzes learning data to identify patterns, struggles, and
    provide recommendations. Can flag patterns that may warrant
    specialist evaluation (with appropriate disclaimers).

    IMPORTANT: This does not provide medical/clinical diagnosis.
    It only analyzes learning patterns and suggests when professional
    evaluation might be beneficial.

    Example:
        capability = DiagnosticAnalysisCapability()
        params = DiagnosticAnalysisParams(
            pattern_data=LearningPatternData(
                topic_performance={"fractions": 0.4, "decimals": 0.8},
                error_patterns={"calculation": 15, "concept": 5},
            ),
        )
        prompt = capability.build_prompt(params.model_dump(), context)
        # Agent sends prompt to LLM, gets response
        result = capability.parse_response(llm_response)
    """

    DISCLAIMER_TEXT = (
        "IMPORTANT: This analysis is based on observed learning patterns only. "
        "It does not constitute a diagnosis of any learning disability or condition. "
        "Only qualified professionals (educational psychologists, special education "
        "specialists, etc.) can provide proper assessment and diagnosis. "
        "Any suggestions for evaluation are recommendations to seek professional "
        "guidance, not diagnostic conclusions."
    )

    @property
    def name(self) -> str:
        """Return capability name."""
        return "diagnostic_analysis"

    @property
    def description(self) -> str:
        """Return capability description."""
        return (
            "Analyzes learning patterns to identify struggles and "
            "recommend adaptations (not clinical diagnosis)"
        )

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate diagnostic parameters.

        Args:
            params: Parameters to validate.

        Raises:
            CapabilityError: If parameters are invalid.
        """
        try:
            DiagnosticAnalysisParams(**params)
        except Exception as e:
            raise CapabilityError(
                message=f"Invalid parameters: {e}",
                capability_name=self.name,
                original_error=e,
            ) from e

    def build_prompt(
        self,
        params: dict[str, Any],
        context: CapabilityContext,
    ) -> list[dict[str, str]]:
        """Build prompt for diagnostic analysis.

        Args:
            params: Analysis parameters.
            context: Context from memory, theory, RAG, persona.

        Returns:
            List of messages for LLM.
        """
        self.validate_params(params)
        p = DiagnosticAnalysisParams(**params)

        # Build system message
        system_parts = []

        # Base instruction with disclaimer
        system_parts.append(
            "You are an educational analyst specializing in learning pattern analysis. "
            "Your role is to identify patterns in learning data and suggest adaptations.\n\n"
            f"{self.DISCLAIMER_TEXT}"
        )

        # Add student context if available
        student_summary = context.get_student_summary()
        if student_summary:
            system_parts.append(student_summary)

        system_message = "\n\n".join(system_parts)

        # Build user message
        user_parts = []

        user_parts.append(
            f"Analyze the following learning pattern data from the past "
            f"{p.time_period_days} days:\n"
        )

        data = p.pattern_data

        # Topic performance
        if data.topic_performance:
            user_parts.append("**Topic Performance:**")
            for topic, mastery in data.topic_performance.items():
                status = "struggling" if mastery < 0.5 else "progressing" if mastery < 0.8 else "mastered"
                user_parts.append(f"- {topic}: {mastery:.0%} ({status})")

        # Error patterns
        if data.error_patterns:
            user_parts.append("\n**Error Patterns:**")
            for error_type, count in data.error_patterns.items():
                user_parts.append(f"- {error_type}: {count} occurrences")

        # Time patterns
        if data.time_patterns:
            user_parts.append("\n**Performance by Time of Day:**")
            for time, perf in data.time_patterns.items():
                user_parts.append(f"- {time}: {perf:.0%}")

        # Response times
        if data.response_times:
            user_parts.append("\n**Response Times:**")
            for category, avg_time in data.response_times.items():
                user_parts.append(f"- {category}: {avg_time:.1f}s average")

        # Hint usage
        if data.hint_usage:
            user_parts.append("\n**Hint Usage:**")
            for level, count in data.hint_usage.items():
                user_parts.append(f"- Level {level}: {count} times")

        # Strategy usage
        if data.strategy_usage:
            user_parts.append("\n**Strategy Effectiveness:**")
            for strategy, success in data.strategy_usage.items():
                user_parts.append(f"- {strategy}: {success:.0%} success rate")

        # Emotional indicators
        if data.emotional_indicators:
            user_parts.append(f"\n**Emotional Indicators:** {', '.join(data.emotional_indicators)}")

        # Focus areas
        if p.focus_areas:
            user_parts.append(f"\n**Focus Analysis On:** {', '.join(p.focus_areas)}")

        # Requirements
        user_parts.append(f"\n**Analysis Requirements:**")
        user_parts.append(f"- Language: {p.language}")
        user_parts.append("- Identify clear patterns in the data")
        user_parts.append("- Highlight both strengths and areas of struggle")

        if p.include_strategy_suggestions:
            user_parts.append("- Provide actionable strategy recommendations")

        if p.include_referral_flags:
            user_parts.append(
                "- If patterns suggest need for specialist evaluation, "
                "flag appropriately WITH DISCLAIMER that this is not diagnosis"
            )

        # Output format
        user_parts.append(self._get_output_format_instruction())

        user_message = "\n".join(user_parts)

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _get_output_format_instruction(self) -> str:
        """Get the JSON output format instruction.

        Returns:
            Output format instruction string.
        """
        return """
Respond with a valid JSON object in this exact format:
```json
{
  "summary": "Overall analysis summary",
  "pattern_observations": [
    {
      "pattern_type": "topic_based",
      "observation": "What was observed",
      "significance": "high",
      "evidence": ["Evidence 1", "Evidence 2"],
      "implication": "What this means"
    }
  ],
  "struggle_patterns": [
    {
      "area": "Fractions",
      "pattern": "Consistent errors in denominator operations",
      "frequency": "frequent",
      "severity": "medium",
      "possible_causes": ["Cause 1", "Cause 2"],
      "observed_in": ["Topic 1", "Topic 2"]
    }
  ],
  "strengths_identified": ["Strength 1", "Strength 2"],
  "learning_style_insights": {
    "preferred_format": "visual",
    "optimal_time": "morning"
  },
  "strategy_recommendations": [
    {
      "strategy": "Use visual representations for fractions",
      "rationale": "Based on observed success with visual content",
      "expected_benefit": "Improved understanding of part-whole relationships",
      "implementation": "Include diagrams in explanations"
    }
  ],
  "referral_flags": [
    {
      "area": "Mathematical processing",
      "observations": ["Pattern 1", "Pattern 2"],
      "disclaimer": "This is not a diagnosis. Only professionals can diagnose.",
      "suggestion": "Consider consultation with educational specialist"
    }
  ],
  "adaptation_suggestions": ["Adaptation 1", "Adaptation 2"],
  "confidence": 0.75,
  "data_quality": "sufficient"
}
```
IMPORTANT: For referral_flags, always include the disclaimer. Only include flags when patterns are significant enough to warrant professional review.
"""

    def parse_response(self, response: str) -> DiagnosticResult:
        """Parse LLM response into DiagnosticResult.

        Args:
            response: Raw LLM response text.

        Returns:
            DiagnosticResult.

        Raises:
            CapabilityError: If response cannot be parsed.
        """
        try:
            data = self._extract_json_from_response(response)
        except CapabilityError:
            return self._parse_plain_text_response(response)

        # Parse pattern observations
        observations = []
        if "pattern_observations" in data and data["pattern_observations"]:
            for obs in data["pattern_observations"]:
                try:
                    pattern_type = PatternType(obs.get("pattern_type", "topic_based"))
                except ValueError:
                    pattern_type = PatternType.TOPIC_BASED

                observations.append(
                    PatternObservation(
                        pattern_type=pattern_type,
                        observation=obs.get("observation", ""),
                        significance=obs.get("significance", "medium"),
                        evidence=obs.get("evidence", []),
                        implication=obs.get("implication", ""),
                    )
                )

        # Parse struggle patterns
        struggles = []
        if "struggle_patterns" in data and data["struggle_patterns"]:
            for s in data["struggle_patterns"]:
                try:
                    severity = SeverityLevel(s.get("severity", "medium"))
                except ValueError:
                    severity = SeverityLevel.MEDIUM

                struggles.append(
                    StrugglePattern(
                        area=s.get("area", ""),
                        pattern=s.get("pattern", ""),
                        frequency=s.get("frequency", "occasional"),
                        severity=severity,
                        possible_causes=s.get("possible_causes", []),
                        observed_in=s.get("observed_in", []),
                    )
                )

        # Parse strategy recommendations
        strategies = []
        if "strategy_recommendations" in data and data["strategy_recommendations"]:
            for sr in data["strategy_recommendations"]:
                strategies.append(
                    StrategyRecommendation(
                        strategy=sr.get("strategy", ""),
                        rationale=sr.get("rationale", ""),
                        expected_benefit=sr.get("expected_benefit", ""),
                        implementation=sr.get("implementation", ""),
                    )
                )

        # Parse referral flags (ensure disclaimer)
        flags = []
        if "referral_flags" in data and data["referral_flags"]:
            for rf in data["referral_flags"]:
                flags.append(
                    ReferralFlag(
                        area=rf.get("area", ""),
                        observations=rf.get("observations", []),
                        disclaimer=rf.get("disclaimer", self.DISCLAIMER_TEXT),
                        suggestion=rf.get("suggestion", ""),
                    )
                )

        return DiagnosticResult(
            success=True,
            capability_name=self.name,
            raw_response=response,
            summary=data.get("summary", ""),
            pattern_observations=observations,
            struggle_patterns=struggles,
            strengths_identified=data.get("strengths_identified", []),
            learning_style_insights=data.get("learning_style_insights", {}),
            strategy_recommendations=strategies,
            referral_flags=flags,
            adaptation_suggestions=data.get("adaptation_suggestions", []),
            confidence=float(data.get("confidence", 0.7)),
            data_quality=data.get("data_quality", "sufficient"),
            language=data.get("language", "tr"),
        )

    def _parse_plain_text_response(self, response: str) -> DiagnosticResult:
        """Parse a plain text response when JSON parsing fails.

        Args:
            response: Plain text response.

        Returns:
            DiagnosticResult with basic extraction.
        """
        return DiagnosticResult(
            success=True,
            capability_name=self.name,
            raw_response=response,
            summary=response.strip()[:500],
            pattern_observations=[],
            struggle_patterns=[],
            strengths_identified=[],
            learning_style_insights={},
            strategy_recommendations=[],
            referral_flags=[],
            adaptation_suggestions=[],
            confidence=0.5,
            data_quality="unknown",
            language="tr",
            metadata={"parse_method": "plain_text_fallback"},
        )

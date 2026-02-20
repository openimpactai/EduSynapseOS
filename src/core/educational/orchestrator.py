# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Theory Orchestrator for combining educational theory recommendations.

The TheoryOrchestrator is responsible for:
- Loading and managing all educational theories
- Applying theories to student context
- Combining recommendations into unified guidance
- Resolving conflicts between theory recommendations

The orchestrator uses weighted averaging for numeric values and
priority-based selection for categorical values.
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from src.core.config.yaml_loader import load_yaml

if TYPE_CHECKING:
    from src.core.emotional import EmotionalContext

from src.core.educational.theories.base import (
    BaseTheory,
    BloomLevel,
    ContentFormat,
    DiagnosticIndicatorScores,
    QuestioningStyle,
    ScaffoldLevel,
    StudentContext,
    TheoryConfig,
    TheoryRecommendation,
)
from src.core.educational.theories.bloom import BloomTheory
from src.core.educational.theories.mastery import MasteryTheory
from src.core.educational.theories.scaffolding import ScaffoldingTheory
from src.core.educational.theories.socratic import SocraticTheory
from src.core.educational.theories.spaced_repetition import SpacedRepetitionTheory
from src.core.educational.theories.vark import VARKTheory
from src.core.educational.theories.zpd import ZPDTheory


class CombinedRecommendation(BaseModel):
    """Combined recommendation from all theories.

    Represents the unified pedagogical guidance after all
    theory recommendations have been combined.
    """

    difficulty: float = Field(ge=0.0, le=1.0, description="Target difficulty")
    bloom_level: BloomLevel = Field(description="Cognitive complexity level")
    content_format: ContentFormat = Field(description="Content delivery format")
    scaffold_level: ScaffoldLevel = Field(description="Support level")
    hints_enabled: bool = Field(description="Whether hints are available")
    guide_vs_tell_ratio: float = Field(
        ge=0.0, le=1.0, description="Socratic guide vs tell ratio"
    )
    questioning_style: QuestioningStyle = Field(description="Questioning approach")
    advance_to_next: bool = Field(description="Whether to advance to next topic")
    next_review_at: datetime | None = Field(description="Next scheduled review")
    overall_confidence: float = Field(
        ge=0.0, le=1.0, description="Overall recommendation confidence"
    )
    theory_contributions: dict[str, TheoryRecommendation] = Field(
        description="Individual theory recommendations"
    )
    applied_at: datetime = Field(description="When recommendation was generated")


class TheoryOrchestrator:
    """Orchestrates all educational theories.

    Loads, configures, and applies all theories to produce
    unified pedagogical recommendations.
    """

    THEORY_CLASSES: dict[str, type[BaseTheory]] = {
        "zpd": ZPDTheory,
        "bloom": BloomTheory,
        "vark": VARKTheory,
        "scaffolding": ScaffoldingTheory,
        "mastery": MasteryTheory,
        "socratic": SocraticTheory,
        "spaced_repetition": SpacedRepetitionTheory,
    }

    def __init__(
        self,
        config_dir: str | Path | None = None,
    ) -> None:
        """Initialize orchestrator with optional config.

        Args:
            config_dir: Directory containing theory YAML configs
        """
        self._config_dir = Path(config_dir) if config_dir else None
        self._theories: dict[str, BaseTheory] = {}
        self._load_theories()

    def _load_theories(self) -> None:
        """Load all theories with their configurations."""
        for name, theory_class in self.THEORY_CLASSES.items():
            config = self._load_theory_config(name)
            self._theories[name] = theory_class(config)

    def _load_theory_config(self, theory_name: str) -> TheoryConfig:
        """Load configuration for a specific theory.

        Args:
            theory_name: Name of the theory

        Returns:
            TheoryConfig instance
        """
        if not self._config_dir:
            return TheoryConfig()

        config_path = self._config_dir / f"{theory_name}.yaml"
        if not config_path.exists():
            return TheoryConfig()

        try:
            data = load_yaml(config_path)
            return TheoryConfig(
                enabled=data.get("enabled", True),
                weight=data.get("weight", 1.0),
                parameters=data.get("parameters", {}),
            )
        except Exception:
            return TheoryConfig()

    def apply(self, context: StudentContext) -> CombinedRecommendation:
        """Apply all theories and combine recommendations.

        Args:
            context: Student's current learning context

        Returns:
            Combined recommendation from all theories
        """
        recommendations = self._collect_recommendations(context)
        combined = self._combine_recommendations(recommendations, context)
        return combined

    def _collect_recommendations(
        self, context: StudentContext
    ) -> dict[str, TheoryRecommendation]:
        """Collect recommendations from all enabled theories.

        Args:
            context: Student context

        Returns:
            Dict of theory name to recommendation
        """
        recommendations = {}

        for name, theory in self._theories.items():
            if theory.enabled:
                recommendations[name] = theory.calculate(context)

        return recommendations

    def _combine_recommendations(
        self,
        recommendations: dict[str, TheoryRecommendation],
        context: StudentContext | None = None,
    ) -> CombinedRecommendation:
        """Combine all theory recommendations.

        Uses weighted averaging for numeric values and
        priority-based selection for categorical values.
        Applies emotional adjustment to difficulty if context available.

        Args:
            recommendations: Dict of recommendations
            context: Optional StudentContext for emotional adjustments

        Returns:
            Combined recommendation
        """
        difficulty = self._combine_difficulty(recommendations, context)
        bloom_level = self._combine_bloom_level(recommendations)
        content_format = self._combine_content_format(recommendations)
        scaffold_level = self._combine_scaffold_level(recommendations, context)
        hints_enabled = self._combine_hints_enabled(recommendations, context)
        guide_ratio = self._combine_guide_ratio(recommendations)
        questioning_style = self._combine_questioning_style(recommendations)
        advance = self._combine_advance(recommendations)
        next_review = self._combine_next_review(recommendations)
        confidence = self._calculate_overall_confidence(recommendations)

        return CombinedRecommendation(
            difficulty=difficulty,
            bloom_level=bloom_level,
            content_format=content_format,
            scaffold_level=scaffold_level,
            hints_enabled=hints_enabled,
            guide_vs_tell_ratio=guide_ratio,
            questioning_style=questioning_style,
            advance_to_next=advance,
            next_review_at=next_review,
            overall_confidence=confidence,
            theory_contributions=recommendations,
            applied_at=datetime.now(),
        )

    def _combine_difficulty(
        self,
        recommendations: dict[str, TheoryRecommendation],
        context: StudentContext | None = None,
    ) -> float:
        """Combine difficulty recommendations using weighted average.

        Applies emotional adjustment if context available:
        - Frustrated/anxious/confused: Lower difficulty (-0.1 to -0.2)
        - Confident/excited: Can increase difficulty (+0.05 to +0.1)

        Args:
            recommendations: Theory recommendations
            context: Optional StudentContext for emotional adjustments

        Returns:
            Combined difficulty 0-1
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for name, rec in recommendations.items():
            if rec.difficulty is not None:
                weight = self._theories[name].weight * rec.confidence
                weighted_sum += rec.difficulty * weight
                total_weight += weight

        if total_weight == 0:
            base_difficulty = 0.5
        else:
            base_difficulty = weighted_sum / total_weight

        # Apply emotional adjustment if context available
        if context:
            emotional_adjustment = context.get_emotional_difficulty_adjustment()
            adjusted = base_difficulty + emotional_adjustment
            # Clamp to valid range
            return max(0.0, min(1.0, adjusted))

        return base_difficulty

    def _combine_bloom_level(
        self, recommendations: dict[str, TheoryRecommendation]
    ) -> BloomLevel:
        """Get Bloom level from Bloom theory.

        Args:
            recommendations: Theory recommendations

        Returns:
            Recommended BloomLevel
        """
        if "bloom" in recommendations and recommendations["bloom"].bloom_level:
            return recommendations["bloom"].bloom_level
        return BloomLevel.UNDERSTAND

    def _combine_content_format(
        self, recommendations: dict[str, TheoryRecommendation]
    ) -> ContentFormat:
        """Get content format from VARK theory.

        Args:
            recommendations: Theory recommendations

        Returns:
            Recommended ContentFormat
        """
        if "vark" in recommendations and recommendations["vark"].content_format:
            return recommendations["vark"].content_format
        return ContentFormat.MULTIMODAL

    def _combine_scaffold_level(
        self,
        recommendations: dict[str, TheoryRecommendation],
        context: StudentContext | None = None,
    ) -> ScaffoldLevel:
        """Combine scaffold level from ZPD and Scaffolding theories.

        Takes the higher level if both recommend (conservative approach).
        Increases scaffold level when student is frustrated/anxious/confused.

        Args:
            recommendations: Theory recommendations
            context: Optional StudentContext for emotional adjustments

        Returns:
            Recommended ScaffoldLevel
        """
        levels = []

        if "scaffolding" in recommendations:
            rec = recommendations["scaffolding"]
            if rec.scaffold_level:
                levels.append(rec.scaffold_level)

        if "zpd" in recommendations:
            rec = recommendations["zpd"]
            if rec.scaffold_level:
                levels.append(rec.scaffold_level)

        if not levels:
            base_level = ScaffoldLevel.MODERATE
        else:
            base_level = max(levels, key=lambda x: x.value)

        # Increase scaffold level if student needs emotional support
        if context and context.emotional and context.emotional.needs_support:
            # Move up one level (if not already at maximum)
            if base_level.value < ScaffoldLevel.MAXIMUM.value:
                return ScaffoldLevel(base_level.value + 1)

        return base_level

    def _combine_hints_enabled(
        self,
        recommendations: dict[str, TheoryRecommendation],
        context: StudentContext | None = None,
    ) -> bool:
        """Determine if hints should be enabled.

        Enables hints if any theory recommends it or if student needs
        emotional support (frustrated, confused, anxious).

        Args:
            recommendations: Theory recommendations
            context: Optional StudentContext for emotional adjustments

        Returns:
            Whether hints are enabled
        """
        # Enable hints if student needs support
        if context and context.emotional and context.emotional.needs_support:
            return True

        for rec in recommendations.values():
            if rec.hints_enabled is True:
                return True
        return False

    def _combine_guide_ratio(
        self, recommendations: dict[str, TheoryRecommendation]
    ) -> float:
        """Get guide vs tell ratio from Socratic theory.

        Args:
            recommendations: Theory recommendations

        Returns:
            Guide ratio 0-1
        """
        if "socratic" in recommendations:
            ratio = recommendations["socratic"].guide_vs_tell_ratio
            if ratio is not None:
                return ratio
        return 0.7

    def _combine_questioning_style(
        self, recommendations: dict[str, TheoryRecommendation]
    ) -> QuestioningStyle:
        """Get questioning style from Socratic theory.

        Args:
            recommendations: Theory recommendations

        Returns:
            Recommended QuestioningStyle
        """
        if "socratic" in recommendations:
            style = recommendations["socratic"].questioning_style
            if style:
                return style
        return QuestioningStyle.GUIDED

    def _combine_advance(
        self, recommendations: dict[str, TheoryRecommendation]
    ) -> bool:
        """Determine if student should advance.

        Only advances if mastery theory recommends it.

        Args:
            recommendations: Theory recommendations

        Returns:
            Whether to advance
        """
        if "mastery" in recommendations:
            return recommendations["mastery"].advance_to_next or False
        return False

    def _combine_next_review(
        self, recommendations: dict[str, TheoryRecommendation]
    ) -> datetime | None:
        """Get next review time from spaced repetition.

        Args:
            recommendations: Theory recommendations

        Returns:
            Next review datetime or None
        """
        if "spaced_repetition" in recommendations:
            return recommendations["spaced_repetition"].next_review_at
        return None

    def _calculate_overall_confidence(
        self, recommendations: dict[str, TheoryRecommendation]
    ) -> float:
        """Calculate overall confidence as weighted average.

        Args:
            recommendations: Theory recommendations

        Returns:
            Overall confidence 0-1
        """
        if not recommendations:
            return 0.5

        total_weight = 0.0
        weighted_sum = 0.0

        for name, rec in recommendations.items():
            weight = self._theories[name].weight
            weighted_sum += rec.confidence * weight
            total_weight += weight

        if total_weight == 0:
            return 0.5

        return weighted_sum / total_weight

    def get_theory(self, name: str) -> BaseTheory | None:
        """Get a specific theory instance.

        Args:
            name: Theory name

        Returns:
            Theory instance or None
        """
        return self._theories.get(name)

    def get_enabled_theories(self) -> list[str]:
        """Get list of enabled theory names.

        Returns:
            List of enabled theory names
        """
        return [
            name for name, theory in self._theories.items()
            if theory.enabled
        ]

    def get_theory_summary(self) -> dict[str, Any]:
        """Get summary of all theories and their status.

        Returns:
            Dict with theory summary information
        """
        return {
            name: {
                "enabled": theory.enabled,
                "weight": theory.weight,
            }
            for name, theory in self._theories.items()
        }

    async def get_recommendations(
        self,
        tenant_code: str,
        student_id: str,
        topic: str,
        memory_context: Any | None = None,
        emotional_context: "EmotionalContext | None" = None,
    ) -> CombinedRecommendation | None:
        """Get recommendations from all theories asynchronously.

        This method bridges the workflow's async context to the sync theory
        calculations by building a StudentContext from memory data.

        Args:
            tenant_code: Tenant code (for future async operations).
            student_id: Student identifier.
            topic: Current topic being practiced.
            memory_context: Optional FullMemoryContext from MemoryManager.
            emotional_context: Optional EmotionalContext from EmotionalStateService.

        Returns:
            CombinedRecommendation or None if no context available.
        """
        if not memory_context:
            # Return default recommendations when no context
            return self.apply(
                StudentContext(
                    student_id=self._safe_uuid(student_id),
                    current_topic=None,
                    overall_mastery=0.5,
                    emotional=emotional_context,
                )
            )

        # Build StudentContext from memory_context
        context = self._build_student_context(
            student_id=student_id,
            topic=topic,
            memory_context=memory_context,
            emotional_context=emotional_context,
        )

        # Apply all theories (sync operation)
        return self.apply(context)

    def _calculate_consecutive_streaks(
        self,
        episodes: list[Any],
    ) -> tuple[int, int]:
        """Calculate consecutive correct/incorrect from episodic events.

        Iterates through answer events (expected to be sorted by created_at DESC,
        most recent first) and counts the streak from the most recent event type.

        This enables key theory features:
        - ZPD: Frustration zone (3+ consecutive incorrect) / Confidence zone (5+ consecutive correct)
        - Bloom: Streak boost (3+ consecutive correct)
        - Scaffolding: Struggle detection (2+ consecutive incorrect)
        - Mastery: Streak requirement for advancement
        - Socratic: Style adjustment based on streaks

        Args:
            episodes: List of episodic memory events, most recent first.

        Returns:
            Tuple of (consecutive_correct, consecutive_incorrect).
            Only one will be non-zero based on most recent event type.
        """
        if not episodes:
            return 0, 0

        # Filter to only answer events
        answer_events = [
            e for e in episodes
            if getattr(getattr(e, "event_type", None), "value", "")
            in ("correct_answer", "incorrect_answer")
        ]

        if not answer_events:
            return 0, 0

        # Get the type of most recent answer event
        first_answer_type = getattr(
            getattr(answer_events[0], "event_type", None), "value", ""
        )

        if first_answer_type == "correct_answer":
            streak = 0
            for e in answer_events:
                event_type = getattr(getattr(e, "event_type", None), "value", "")
                if event_type == "correct_answer":
                    streak += 1
                else:
                    break
            return streak, 0

        elif first_answer_type == "incorrect_answer":
            streak = 0
            for e in answer_events:
                event_type = getattr(getattr(e, "event_type", None), "value", "")
                if event_type == "incorrect_answer":
                    streak += 1
                else:
                    break
            return 0, streak

        return 0, 0

    def _calculate_hint_dependency(
        self,
        episodes: list[Any],
    ) -> float:
        """Calculate hint dependency from episodic memory events.

        Hint dependency represents how often the student relies on hints.
        A value of 0.5+ indicates significant hint reliance, which triggers
        scaffolding adjustments in the Scaffolding theory.

        Args:
            episodes: List of episodic memory events.

        Returns:
            Hint dependency ratio 0.0-1.0.
        """
        # Filter to answer events only
        answer_events = [
            e for e in episodes
            if getattr(getattr(e, "event_type", None), "value", "")
            in ("correct_answer", "incorrect_answer")
        ]

        if not answer_events:
            return 0.0

        # Check hints_used in event details
        hints_used_count = 0
        for event in answer_events:
            details = getattr(event, "details", {}) or {}
            if isinstance(details, dict):
                hints_used = details.get("hints_used", 0) or details.get("hints_viewed", 0)
                if hints_used and hints_used > 0:
                    hints_used_count += 1

        return hints_used_count / len(answer_events)

    def _build_student_context(
        self,
        student_id: str,
        topic: str,
        memory_context: Any,
        emotional_context: "EmotionalContext | None" = None,
    ) -> StudentContext:
        """Build StudentContext from FullMemoryContext.

        Converts the FullMemoryContext from MemoryManager into a StudentContext
        that theories can use for calculations. Handles missing or incomplete
        data gracefully by using sensible defaults.

        Args:
            student_id: Student identifier.
            topic: Current topic.
            memory_context: Full memory context from MemoryManager.
            emotional_context: Optional EmotionalContext from EmotionalStateService.

        Returns:
            StudentContext for theory calculations.
            The diagnostic field will be None if no diagnostic data is available,
            allowing theories to use default behavior for new students.
        """
        from src.core.educational.theories.base import (
            RecentPerformance,
            TopicState,
            VARKScores,
        )

        # Extract mastery based on context priority:
        # 1. topic_mastery (specific topic) - has full details including streaks
        # 2. subject_mastery (subject-level practice) - just average mastery
        # 3. overall_mastery (random/mixed) - fallback from MasteryOverview
        semantic = getattr(memory_context, "semantic", None)
        topic_mastery = getattr(memory_context, "topic_mastery", None)
        subject_mastery = getattr(memory_context, "subject_mastery", None)
        overall_mastery = getattr(semantic, "overall_mastery", 0.5) if semantic else 0.5

        # Build TopicState with correct data source
        current_topic = None
        if topic_mastery:
            # Scenario 1: Specific topic - use full topic mastery data
            # Note: entity_name may be None from semantic layer (doesn't join with curriculum)
            # Use 'or' to fall back to topic name when entity_name is None
            current_topic = TopicState(
                topic_full_code=getattr(topic_mastery, "entity_full_code", None) or topic or "",
                topic_name=getattr(topic_mastery, "entity_name", None) or topic or "",
                mastery_level=float(getattr(topic_mastery, "mastery_level", 0.5)),
                attempts_total=getattr(topic_mastery, "attempts_total", 0),
                attempts_correct=getattr(topic_mastery, "attempts_correct", 0),
                current_streak=getattr(topic_mastery, "current_streak", 0),
                best_streak=getattr(topic_mastery, "best_streak", 0),
            )
            # Use topic mastery for overall context as well
            overall_mastery = float(getattr(topic_mastery, "mastery_level", overall_mastery))
        elif subject_mastery is not None:
            # Scenario 2: Subject-level practice - use subject average
            current_topic = TopicState(
                topic_full_code=topic if topic else "",
                topic_name=topic if topic else "",
                mastery_level=float(subject_mastery),
                attempts_total=0,  # No specific topic, no attempt data
                attempts_correct=0,
                current_streak=0,
                best_streak=0,
            )
            overall_mastery = float(subject_mastery)
        elif semantic:
            # Scenario 3: Random/mixed practice - use overall mastery
            current_topic = TopicState(
                topic_full_code=topic if topic else "",
                topic_name=topic if topic else "",
                mastery_level=overall_mastery,
                attempts_total=0,
                attempts_correct=0,
                current_streak=0,
                best_streak=0,
            )

        # Extract VARK scores from procedural memory
        vark_scores = VARKScores()
        procedural = getattr(memory_context, "procedural", None)
        if procedural:
            vark_profile = getattr(procedural, "vark_profile", None)
            if vark_profile:
                vark_scores = VARKScores(
                    visual=getattr(vark_profile, "visual", 0.25),
                    auditory=getattr(vark_profile, "auditory", 0.25),
                    reading=getattr(vark_profile, "reading_writing", 0.25),
                    kinesthetic=getattr(vark_profile, "kinesthetic", 0.25),
                )

        # Build recent performance from episodic memory
        episodes = getattr(memory_context, "episodic", []) or []
        correct = sum(
            1
            for e in episodes
            if getattr(getattr(e, "event_type", None), "value", "") == "correct_answer"
        )
        incorrect = sum(
            1
            for e in episodes
            if getattr(getattr(e, "event_type", None), "value", "") == "incorrect_answer"
        )

        # Calculate consecutive streaks from event sequence
        # Episodes are expected to be sorted by created_at DESC (most recent first)
        # This enables ZPD frustration/confidence zones, Scaffolding struggle detection,
        # Bloom streak boost, and Mastery streak requirement
        consecutive_correct, consecutive_incorrect = self._calculate_consecutive_streaks(
            episodes
        )

        recent = RecentPerformance(
            last_n_correct=correct,
            last_n_incorrect=incorrect,
            consecutive_correct=consecutive_correct,
            consecutive_incorrect=consecutive_incorrect,
        )

        # Calculate hint dependency from episodic events
        # This enables Scaffolding theory to apply hint-based adjustments
        # when hint_dependency > 0.5
        hint_dependency = self._calculate_hint_dependency(episodes)

        # Extract interests from associative memory
        associative = getattr(memory_context, "associative", None)
        interests_list = []
        if associative:
            raw_interests = getattr(associative, "interests", []) or []
            interests_list = [
                getattr(i, "content", str(i))
                for i in raw_interests
                if i
            ]

        # Extract diagnostic indicator scores (may be None for new students)
        diagnostic_scores = self._extract_diagnostic_scores(memory_context)

        return StudentContext(
            student_id=self._safe_uuid(student_id),
            current_topic=current_topic,
            overall_mastery=overall_mastery,
            topics_mastered_count=getattr(semantic, "topics_mastered", 0) if semantic else 0,
            topics_struggling_count=getattr(semantic, "topics_struggling", 0) if semantic else 0,
            vark_scores=vark_scores,
            recent_performance=recent,
            hint_dependency=hint_dependency,
            interests=interests_list,
            diagnostic=diagnostic_scores,
            emotional=emotional_context,
        )

    def _extract_diagnostic_scores(
        self,
        memory_context: Any,
    ) -> DiagnosticIndicatorScores | None:
        """Extract diagnostic indicator scores from memory context.

        Converts DiagnosticContext from FullMemoryContext to
        DiagnosticIndicatorScores for use in theories.

        This method is fault-tolerant and returns None if:
        - memory_context is None
        - diagnostic field is None (no scan performed)
        - diagnostic field is missing
        - Any error occurs during extraction

        Args:
            memory_context: Full memory context from MemoryManager.

        Returns:
            DiagnosticIndicatorScores or None if unavailable.
        """
        if not memory_context:
            return None

        try:
            diagnostic = getattr(memory_context, "diagnostic", None)
            if not diagnostic:
                # No diagnostic data available - normal for new students
                return None

            return DiagnosticIndicatorScores(
                dyslexia_risk=getattr(diagnostic, "dyslexia_risk", 0.0),
                dyscalculia_risk=getattr(diagnostic, "dyscalculia_risk", 0.0),
                attention_risk=getattr(diagnostic, "attention_risk", 0.0),
                auditory_risk=getattr(diagnostic, "auditory_risk", 0.0),
                visual_risk=getattr(diagnostic, "visual_risk", 0.0),
                last_scan_at=getattr(diagnostic, "last_scan_at", None),
            )

        except Exception:
            # On any error, return None to allow theories to use defaults
            return None

    def _safe_uuid(self, value: str) -> "uuid.UUID":
        """Safely convert string to UUID.

        Args:
            value: String value (UUID string or topic name).

        Returns:
            UUID object.
        """
        import uuid

        try:
            return uuid.UUID(value)
        except (ValueError, AttributeError):
            # Generate deterministic UUID from string
            return uuid.uuid5(uuid.NAMESPACE_DNS, value)

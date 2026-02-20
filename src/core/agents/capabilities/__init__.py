# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Agent capabilities for educational AI interactions.

Capabilities define WHAT an agent can do. Each capability:
1. Builds prompts using context (memory, theory, RAG, persona)
2. Parses LLM responses into structured outputs

Capabilities do NOT call the LLM directly - the Agent does that.
This separation allows for:
- Testability: Capabilities can be tested without LLM
- Flexibility: Same capability, different LLM providers
- Composability: Agent can combine multiple capabilities

Available Capabilities:
    - QuestionGenerationCapability: Generate educational questions
    - AnswerEvaluationCapability: Evaluate student answers
    - ConceptExplanationCapability: Explain concepts adaptively
    - FeedbackGenerationCapability: Generate personalized feedback
    - DiagnosticAnalysisCapability: Analyze learning patterns
    - MessageAnalysisCapability: Analyze message intent and sentiment
"""

from src.core.agents.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityError,
    CapabilityResult,
)
from src.core.agents.capabilities.registry import CapabilityRegistry

from src.core.agents.capabilities.question_generation import (
    QuestionGenerationCapability,
    GeneratedQuestion,
    QuestionGenerationParams,
)
from src.core.agents.capabilities.answer_evaluation import (
    AnswerEvaluationCapability,
    AnswerEvaluationResult,
    AnswerEvaluationParams,
)
from src.core.agents.capabilities.concept_explanation import (
    ConceptExplanationCapability,
    ConceptExplanation,
    ConceptExplanationParams,
)
from src.core.agents.capabilities.feedback_generation import (
    FeedbackGenerationCapability,
    GeneratedFeedback,
    FeedbackGenerationParams,
)
from src.core.agents.capabilities.diagnostic_analysis import (
    DiagnosticAnalysisCapability,
    DiagnosticResult,
    DiagnosticAnalysisParams,
)
from src.core.agents.capabilities.message_analysis import (
    MessageAnalysisCapability,
    MessageAnalysisResult,
    MessageAnalysisParams,
)
from src.core.agents.capabilities.comprehension_evaluation import (
    ComprehensionEvaluationCapability,
)
# Companion capabilities
from src.core.agents.capabilities.wellbeing_check import (
    WellbeingCheckCapability,
    WellbeingCheckResult,
    WellbeingCheckParams,
)
from src.core.agents.capabilities.emotional_support import (
    EmotionalSupportCapability,
    EmotionalSupportResult,
    EmotionalSupportParams,
)
from src.core.agents.capabilities.activity_guidance import (
    ActivityGuidanceCapability,
    ActivityGuidanceResult,
    ActivityGuidanceParams,
    ActivityOption,
)
from src.core.agents.capabilities.companion_decision import (
    CompanionDecisionCapability,
    CompanionDecisionResult,
    CompanionDecisionParams,
)
# Gaming capabilities
from src.core.agents.capabilities.game_move_analysis import (
    GameMoveAnalysisCapability,
    MoveAnalysisResult,
    MoveAnalysisParams,
)
from src.core.agents.capabilities.game_coach_response import (
    GameCoachResponseCapability,
    CoachResponseResult,
    CoachResponseParams,
)
from src.core.agents.capabilities.game_hint_generation import (
    GameHintGenerationCapability,
    HintGenerationResult,
    HintGenerationParams,
)

__all__ = [
    # Base
    "Capability",
    "CapabilityContext",
    "CapabilityError",
    "CapabilityResult",
    "CapabilityRegistry",
    # Question Generation
    "QuestionGenerationCapability",
    "GeneratedQuestion",
    "QuestionGenerationParams",
    # Answer Evaluation
    "AnswerEvaluationCapability",
    "AnswerEvaluationResult",
    "AnswerEvaluationParams",
    # Concept Explanation
    "ConceptExplanationCapability",
    "ConceptExplanation",
    "ConceptExplanationParams",
    # Feedback Generation
    "FeedbackGenerationCapability",
    "GeneratedFeedback",
    "FeedbackGenerationParams",
    # Diagnostic Analysis
    "DiagnosticAnalysisCapability",
    "DiagnosticResult",
    "DiagnosticAnalysisParams",
    # Message Analysis
    "MessageAnalysisCapability",
    "MessageAnalysisResult",
    "MessageAnalysisParams",
    # Comprehension Evaluation
    "ComprehensionEvaluationCapability",
    # Wellbeing Check
    "WellbeingCheckCapability",
    "WellbeingCheckResult",
    "WellbeingCheckParams",
    # Emotional Support
    "EmotionalSupportCapability",
    "EmotionalSupportResult",
    "EmotionalSupportParams",
    # Activity Guidance
    "ActivityGuidanceCapability",
    "ActivityGuidanceResult",
    "ActivityGuidanceParams",
    "ActivityOption",
    # Companion Decision
    "CompanionDecisionCapability",
    "CompanionDecisionResult",
    "CompanionDecisionParams",
    # Game Move Analysis
    "GameMoveAnalysisCapability",
    "MoveAnalysisResult",
    "MoveAnalysisParams",
    # Game Coach Response
    "GameCoachResponseCapability",
    "CoachResponseResult",
    "CoachResponseParams",
    # Game Hint Generation
    "GameHintGenerationCapability",
    "HintGenerationResult",
    "HintGenerationParams",
]

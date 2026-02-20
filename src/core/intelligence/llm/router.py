# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model routing and fallback chain management.

This module provides intelligent model selection and fallback chains
for robust LLM operations across multiple providers.

Features:
- Use case based model selection
- Fallback chains for reliability
- Cost and latency optimization
- YAML-based configuration

Example:
    >>> from src.core.intelligence.llm import ModelRouter
    >>> router = ModelRouter()
    >>> model = router.select_model(use_case="tutoring")
    >>> fallbacks = router.get_fallback_chain("tutoring")
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml

from src.core.config.settings import LLMSettings, get_settings

logger = logging.getLogger(__name__)


class UseCase(str, Enum):
    """Predefined use cases for model selection.

    Each use case has different requirements for:
    - Response quality
    - Latency tolerance
    - Cost sensitivity
    - Token limits
    """

    # Educational use cases
    TUTORING = "tutoring"  # Interactive tutoring conversations
    QUESTION_GENERATION = "question_generation"  # Generating practice questions
    EVALUATION = "evaluation"  # Evaluating student answers
    EXPLANATION = "explanation"  # Explaining concepts
    FEEDBACK = "feedback"  # Providing feedback on work

    # System use cases
    SUMMARIZATION = "summarization"  # Summarizing conversations/content
    CLASSIFICATION = "classification"  # Classifying intents/content
    EXTRACTION = "extraction"  # Extracting structured data
    EMBEDDING = "embedding"  # Text embedding (handled by EmbeddingService)

    # General use cases
    LEARNING = "learning"  # Learning/tutoring interactions
    COMPLETION = "completion"  # Text completion
    CODE = "code"  # Code generation/analysis


@dataclass
class ModelConfig:
    """Configuration for a specific model.

    Attributes:
        name: Model name in LiteLLM format.
        provider: Provider name (ollama, openai, anthropic, google).
        max_tokens: Maximum context window.
        default_temperature: Default sampling temperature.
        cost_per_1k_input: Cost per 1000 input tokens (USD).
        cost_per_1k_output: Cost per 1000 output tokens (USD).
        latency_tier: Latency category (fast, medium, slow).
        capabilities: List of supported use cases.
        fallback_to: Model to fallback to on failure.
    """

    name: str
    provider: str = "ollama"
    max_tokens: int = 4096
    default_temperature: float = 0.7
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    latency_tier: str = "medium"
    capabilities: list[str] = field(default_factory=list)
    fallback_to: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Create ModelConfig from dictionary.

        Args:
            data: Dictionary with model configuration.

        Returns:
            ModelConfig instance.
        """
        return cls(
            name=data["name"],
            provider=data.get("provider", "ollama"),
            max_tokens=data.get("max_tokens", 4096),
            default_temperature=data.get("default_temperature", 0.7),
            cost_per_1k_input=data.get("cost_per_1k_input", 0.0),
            cost_per_1k_output=data.get("cost_per_1k_output", 0.0),
            latency_tier=data.get("latency_tier", "medium"),
            capabilities=data.get("capabilities", []),
            fallback_to=data.get("fallback_to"),
        )


@dataclass
class RoutingRule:
    """Rule for routing requests to models.

    Attributes:
        use_case: The use case this rule applies to.
        primary_model: Primary model to use.
        fallback_chain: Ordered list of fallback models.
        max_latency_ms: Maximum acceptable latency.
        max_cost_per_request: Maximum cost per request.
    """

    use_case: str
    primary_model: str
    fallback_chain: list[str] = field(default_factory=list)
    max_latency_ms: Optional[int] = None
    max_cost_per_request: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> "RoutingRule":
        """Create RoutingRule from dictionary.

        Args:
            data: Dictionary with routing rule.

        Returns:
            RoutingRule instance.
        """
        return cls(
            use_case=data["use_case"],
            primary_model=data["primary_model"],
            fallback_chain=data.get("fallback_chain", []),
            max_latency_ms=data.get("max_latency_ms"),
            max_cost_per_request=data.get("max_cost_per_request"),
        )


# Default model configurations
DEFAULT_MODELS: dict[str, ModelConfig] = {
    "ollama/qwen2.5:7b": ModelConfig(
        name="ollama/qwen2.5:7b",
        provider="ollama",
        max_tokens=32768,
        default_temperature=0.7,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        latency_tier="fast",
        capabilities=[
            "tutoring",
            "question_generation",
            "evaluation",
            "explanation",
            "feedback",
            "learning",
            "completion",
            "code",
        ],
    ),
    "ollama/llama3.2:3b": ModelConfig(
        name="ollama/llama3.2:3b",
        provider="ollama",
        max_tokens=8192,
        default_temperature=0.7,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        latency_tier="fast",
        capabilities=[
            "classification",
            "extraction",
            "summarization",
            "learning",
        ],
        fallback_to="ollama/qwen2.5:7b",
    ),
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        provider="openai",
        max_tokens=128000,
        default_temperature=0.7,
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
        latency_tier="medium",
        capabilities=[
            "tutoring",
            "question_generation",
            "evaluation",
            "explanation",
            "feedback",
            "learning",
            "completion",
            "code",
            "classification",
            "extraction",
            "summarization",
        ],
    ),
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        provider="openai",
        max_tokens=128000,
        default_temperature=0.7,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        latency_tier="fast",
        capabilities=[
            "classification",
            "extraction",
            "summarization",
            "learning",
            "completion",
        ],
        fallback_to="gpt-4o",
    ),
    "claude-3-5-sonnet-20241022": ModelConfig(
        name="claude-3-5-sonnet-20241022",
        provider="anthropic",
        max_tokens=200000,
        default_temperature=0.7,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        latency_tier="medium",
        capabilities=[
            "tutoring",
            "question_generation",
            "evaluation",
            "explanation",
            "feedback",
            "learning",
            "completion",
            "code",
            "classification",
            "extraction",
            "summarization",
        ],
    ),
    "gemini/gemini-2.0-flash-exp": ModelConfig(
        name="gemini/gemini-2.0-flash-exp",
        provider="google",
        max_tokens=1000000,
        default_temperature=0.7,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        latency_tier="fast",
        capabilities=[
            "tutoring",
            "question_generation",
            "evaluation",
            "learning",
            "completion",
            "code",
        ],
    ),
}

# Default routing rules
DEFAULT_ROUTING_RULES: dict[str, RoutingRule] = {
    "tutoring": RoutingRule(
        use_case="tutoring",
        primary_model="ollama/qwen2.5:7b",
        fallback_chain=["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022"],
    ),
    "question_generation": RoutingRule(
        use_case="question_generation",
        primary_model="ollama/qwen2.5:7b",
        fallback_chain=["gpt-4o", "claude-3-5-sonnet-20241022"],
    ),
    "evaluation": RoutingRule(
        use_case="evaluation",
        primary_model="ollama/qwen2.5:7b",
        fallback_chain=["gpt-4o", "claude-3-5-sonnet-20241022"],
    ),
    "explanation": RoutingRule(
        use_case="explanation",
        primary_model="ollama/qwen2.5:7b",
        fallback_chain=["gpt-4o", "claude-3-5-sonnet-20241022"],
    ),
    "feedback": RoutingRule(
        use_case="feedback",
        primary_model="ollama/qwen2.5:7b",
        fallback_chain=["gpt-4o-mini", "gpt-4o"],
    ),
    "summarization": RoutingRule(
        use_case="summarization",
        primary_model="ollama/llama3.2:3b",
        fallback_chain=["gpt-4o-mini", "ollama/qwen2.5:7b"],
    ),
    "classification": RoutingRule(
        use_case="classification",
        primary_model="ollama/llama3.2:3b",
        fallback_chain=["gpt-4o-mini", "ollama/qwen2.5:7b"],
    ),
    "extraction": RoutingRule(
        use_case="extraction",
        primary_model="ollama/llama3.2:3b",
        fallback_chain=["gpt-4o-mini", "gpt-4o"],
    ),
    "learning": RoutingRule(
        use_case="learning",
        primary_model="ollama/qwen2.5:7b",
        fallback_chain=["gpt-4o-mini", "gpt-4o"],
    ),
    "completion": RoutingRule(
        use_case="completion",
        primary_model="ollama/qwen2.5:7b",
        fallback_chain=["gpt-4o-mini", "gpt-4o"],
    ),
    "code": RoutingRule(
        use_case="code",
        primary_model="ollama/qwen2.5:7b",
        fallback_chain=["gpt-4o", "claude-3-5-sonnet-20241022"],
    ),
}


class ModelRouter:
    """Intelligent model selection and routing.

    The ModelRouter selects appropriate models based on use case,
    cost constraints, and latency requirements. It also provides
    fallback chains for reliability.

    Configuration can be loaded from YAML files or use defaults.

    Attributes:
        models: Dictionary of available model configurations.
        routing_rules: Dictionary of routing rules per use case.

    Example:
        >>> router = ModelRouter()
        >>> model = router.select_model(use_case="tutoring")
        >>> print(model)
        ollama/qwen2.5:7b
        >>> fallbacks = router.get_fallback_chain("tutoring")
        >>> print(fallbacks)
        ['gpt-4o-mini', 'gpt-4o', 'claude-3-5-sonnet-20241022']
    """

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        llm_settings: Optional[LLMSettings] = None,
    ):
        """Initialize the model router.

        Args:
            config_dir: Directory containing models.yaml and routing.yaml.
                       Falls back to config/llm if not provided.
            llm_settings: LLM settings for default model.
        """
        settings = get_settings()
        self._settings = llm_settings or settings.llm
        self._config_dir = config_dir or Path("config/llm")

        self._models: dict[str, ModelConfig] = {}
        self._routing_rules: dict[str, RoutingRule] = {}

        # Load configuration
        self._load_configuration()

        logger.info(
            "ModelRouter initialized with %d models and %d routing rules",
            len(self._models),
            len(self._routing_rules),
        )

    def _load_configuration(self) -> None:
        """Load model and routing configuration from YAML files or defaults."""
        # Try to load models from YAML
        models_path = self._config_dir / "models.yaml"
        if models_path.exists():
            try:
                with open(models_path) as f:
                    data = yaml.safe_load(f)
                    if data and "models" in data:
                        for model_data in data["models"]:
                            config = ModelConfig.from_dict(model_data)
                            self._models[config.name] = config
                        logger.info("Loaded %d models from %s", len(self._models), models_path)
            except Exception as e:
                logger.warning("Failed to load models.yaml: %s", e)

        # Try to load routing rules from YAML
        routing_path = self._config_dir / "routing.yaml"
        if routing_path.exists():
            try:
                with open(routing_path) as f:
                    data = yaml.safe_load(f)
                    if data and "routing" in data:
                        for rule_data in data["routing"]:
                            rule = RoutingRule.from_dict(rule_data)
                            self._routing_rules[rule.use_case] = rule
                        logger.info(
                            "Loaded %d routing rules from %s",
                            len(self._routing_rules),
                            routing_path,
                        )
            except Exception as e:
                logger.warning("Failed to load routing.yaml: %s", e)

        # Fall back to defaults if nothing loaded
        if not self._models:
            self._models = DEFAULT_MODELS.copy()
            logger.info("Using default model configurations")

        if not self._routing_rules:
            self._routing_rules = DEFAULT_ROUTING_RULES.copy()
            logger.info("Using default routing rules")

    @property
    def models(self) -> dict[str, ModelConfig]:
        """Get all available model configurations.

        Returns:
            Dictionary mapping model names to configurations.
        """
        return self._models

    @property
    def routing_rules(self) -> dict[str, RoutingRule]:
        """Get all routing rules.

        Returns:
            Dictionary mapping use cases to routing rules.
        """
        return self._routing_rules

    def select_model(
        self,
        use_case: Optional[str] = None,
        prefer_local: bool = True,
        max_cost: Optional[float] = None,
        require_streaming: bool = False,
    ) -> str:
        """Select the best model for a given use case.

        Args:
            use_case: The use case (tutoring, evaluation, etc.).
            prefer_local: Prefer local models (Ollama) when possible.
            max_cost: Maximum cost per 1000 tokens.
            require_streaming: Whether streaming is required.

        Returns:
            Model name in LiteLLM format.

        Example:
            >>> model = router.select_model(use_case="tutoring")
            >>> print(model)
            ollama/qwen2.5:7b
        """
        # Get routing rule for use case
        if use_case and use_case in self._routing_rules:
            rule = self._routing_rules[use_case]
            primary = rule.primary_model

            # Check if primary model meets constraints
            if primary in self._models:
                config = self._models[primary]

                # Check cost constraint
                if max_cost is not None:
                    avg_cost = (config.cost_per_1k_input + config.cost_per_1k_output) / 2
                    if avg_cost > max_cost:
                        # Try to find cheaper alternative
                        for fallback in rule.fallback_chain:
                            if fallback in self._models:
                                fb_config = self._models[fallback]
                                fb_avg = (
                                    fb_config.cost_per_1k_input
                                    + fb_config.cost_per_1k_output
                                ) / 2
                                if fb_avg <= max_cost:
                                    return fallback
                        return primary  # Use primary anyway if no cheaper option

                # Check local preference
                if prefer_local and config.provider != "ollama":
                    # Look for local alternative with same capability
                    for model_name, model_config in self._models.items():
                        if (
                            model_config.provider == "ollama"
                            and use_case in model_config.capabilities
                        ):
                            return model_name

                return primary

            # Primary not in models, try fallbacks
            for fallback in rule.fallback_chain:
                if fallback in self._models:
                    return fallback

        # No rule found, return default model
        return self._settings.get_default_model()

    def get_fallback_chain(self, use_case: str) -> list[str]:
        """Get the fallback chain for a use case.

        Args:
            use_case: The use case to get fallbacks for.

        Returns:
            Ordered list of fallback model names.

        Example:
            >>> fallbacks = router.get_fallback_chain("tutoring")
            >>> print(fallbacks)
            ['gpt-4o-mini', 'gpt-4o', 'claude-3-5-sonnet-20241022']
        """
        if use_case in self._routing_rules:
            return self._routing_rules[use_case].fallback_chain.copy()
        return []

    def get_model_config(self, model: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model.

        Args:
            model: Model name to get config for.

        Returns:
            ModelConfig if found, None otherwise.
        """
        return self._models.get(model)

    def get_models_for_use_case(self, use_case: str) -> list[str]:
        """Get all models capable of handling a use case.

        Args:
            use_case: The use case to filter by.

        Returns:
            List of model names that support the use case.
        """
        capable_models: list[str] = []
        for model_name, config in self._models.items():
            if use_case in config.capabilities:
                capable_models.append(model_name)
        return capable_models

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate the cost for a request.

        Args:
            model: Model to use.
            input_tokens: Expected input tokens.
            output_tokens: Expected output tokens.

        Returns:
            Estimated cost in USD.
        """
        if model not in self._models:
            return 0.0

        config = self._models[model]
        input_cost = (input_tokens / 1000) * config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * config.cost_per_1k_output
        return input_cost + output_cost

    def get_available_providers(self) -> list[str]:
        """Get list of available providers.

        Returns:
            List of unique provider names.
        """
        providers = set()
        for config in self._models.values():
            providers.add(config.provider)
        return sorted(providers)

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String showing model and rule counts.
        """
        return (
            f"ModelRouter(models={len(self._models)}, "
            f"routing_rules={len(self._routing_rules)})"
        )

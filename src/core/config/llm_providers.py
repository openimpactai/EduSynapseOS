# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""LLM Provider Configuration from YAML.

This module loads and manages LLM provider configurations from config/llm/providers.yaml.
Each provider has a unique CODE that agents reference in their configurations.

Usage:
    from src.core.config.llm_providers import get_provider_config, get_llm_params

    # Get provider configuration by code
    config = get_provider_config("local")
    print(config.api_base)  # http://localhost:11434
    print(config.default_model)  # qwen2.5:7b

    # Get LiteLLM parameters for a provider
    params = get_llm_params_for_provider("local")
    # Returns: {"model": "ollama/qwen2.5:7b", "api_base": "http://..."}

    # Or get params for a specific model string
    params = get_llm_params("ollama/command-r:35b")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider.

    Attributes:
        code: Provider code (e.g., 'vastai', 'local', 'openai').
        enabled: Whether this provider is enabled.
        type: Provider type for LiteLLM routing (ollama, openai, anthropic, google).
        description: Human-readable description.
        api_base: Base URL for the API endpoint.
        api_key: API key for authentication.
        default_model: Default model to use.
        available_models: List of available models.
        context_length: Context window size (num_ctx for Ollama). Model's max input+output.
        max_output_tokens: Maximum output tokens (num_predict for Ollama).
        timeout_seconds: Request timeout.
        max_retries: Maximum retry attempts.
        cost_per_1k_input: Cost per 1K input tokens.
        cost_per_1k_output: Cost per 1K output tokens.
    """

    code: str
    enabled: bool = True
    type: str = "ollama"
    description: str = ""
    api_base: str | None = None
    api_key: str | None = None
    default_model: str = ""
    available_models: list[str] = field(default_factory=list)
    context_length: int = 4096  # Ollama default, override in providers.yaml
    max_output_tokens: int = 4096  # Reasonable default for most models
    timeout_seconds: int = 60
    max_retries: int = 3
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0

    @classmethod
    def from_dict(cls, code: str, data: dict[str, Any]) -> "ProviderConfig":
        """Create ProviderConfig from dictionary.

        Args:
            code: Provider code.
            data: Configuration dictionary.

        Returns:
            ProviderConfig instance.
        """
        return cls(
            code=code,
            enabled=data.get("enabled", True),
            type=data.get("type", "ollama"),
            description=data.get("description", ""),
            api_base=data.get("api_base"),
            api_key=data.get("api_key") if data.get("api_key") else None,
            default_model=data.get("default_model", ""),
            available_models=data.get("available_models", []),
            context_length=data.get("context_length", 4096),
            max_output_tokens=data.get("max_output_tokens", 4096),
            timeout_seconds=data.get("timeout_seconds", 60),
            max_retries=data.get("max_retries", 3),
            cost_per_1k_input=data.get("cost_per_1k_input", 0.0),
            cost_per_1k_output=data.get("cost_per_1k_output", 0.0),
        )

    def get_litellm_model(self, model: str | None = None) -> str:
        """Get model string in LiteLLM format.

        Args:
            model: Optional model override. Uses default_model if not specified.

        Returns:
            Model string with provider prefix (e.g., 'ollama/command-r:35b').
        """
        use_model = model or self.default_model

        # Add provider prefix for LiteLLM
        # Use ollama_chat/ for chat completion with tool calling support
        if self.type == "ollama" and not use_model.startswith(("ollama/", "ollama_chat/")):
            return f"ollama_chat/{use_model}"
        elif self.type == "google" and not use_model.startswith("gemini/"):
            return f"gemini/{use_model}"

        return use_model

    def get_litellm_params(self, model: str | None = None) -> dict[str, Any]:
        """Get parameters to pass to LiteLLM acompletion().

        Returns model, api_base, api_key, and model-specific settings.
        This is required for LiteLLM 1.80+ to work correctly with Ollama
        and custom endpoints.

        Args:
            model: Optional model override. Uses default_model if not specified.

        Returns:
            Dictionary with model, api_base, api_key, context_length,
            and max_output_tokens if configured.
        """
        params: dict[str, Any] = {
            "model": self.get_litellm_model(model),
            "context_length": self.context_length,
            "max_output_tokens": self.max_output_tokens,
        }

        if self.api_base:
            params["api_base"] = self.api_base

        if self.api_key:
            params["api_key"] = self.api_key

        return params

    def is_available(self) -> bool:
        """Check if provider is available for use.

        A provider is available if:
        - It is enabled
        - It has required credentials (api_key for cloud, api_base for remote)

        Returns:
            True if provider can be used.
        """
        if not self.enabled:
            return False

        # Ollama doesn't require api_key, but needs api_base
        if self.type == "ollama":
            return bool(self.api_base)

        # Cloud providers require api_key
        if self.type in ("openai", "anthropic", "google"):
            return bool(self.api_key)

        return True


@dataclass
class RoutingStrategyConfig:
    """Configuration for a routing strategy.

    Attributes:
        name: Strategy identifier.
        primary: Primary provider code to use.
        fallbacks: List of fallback provider codes.
        description: Human-readable description.
    """

    name: str
    primary: str
    fallbacks: list[str] = field(default_factory=list)
    description: str = ""

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "RoutingStrategyConfig":
        """Create from dictionary."""
        return cls(
            name=name,
            primary=data.get("primary", "vastai"),
            fallbacks=data.get("fallbacks", []),
            description=data.get("description", ""),
        )


@dataclass
class EmbeddingConfig:
    """Configuration for embedding provider.

    Attributes:
        provider: Provider code.
        model: Embedding model name.
        dimension: Vector dimension.
        batch_size: Batch size for embedding generation.
    """

    provider: str = "vastai"
    model: str = "nomic-embed-text"
    dimension: int = 768
    batch_size: int = 32

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingConfig":
        """Create from dictionary."""
        return cls(
            provider=data.get("provider", "vastai"),
            model=data.get("model", "nomic-embed-text"),
            dimension=int(data.get("dimension", 768)),
            batch_size=int(data.get("batch_size", 32)),
        )


class LLMProviderManager:
    """Manages LLM provider configurations.

    Loads providers from YAML and provides access to provider configs by code.

    Attributes:
        default_provider: Code of the default provider.
        providers: Dictionary of provider configurations by code.
        routing_strategies: Dictionary of routing strategy configurations.
        embedding: Embedding configuration.
    """

    DEFAULT_CONFIG_PATH = "config/llm/providers.yaml"

    def __init__(self, config_path: str | Path | None = None):
        """Initialize the provider manager.

        Args:
            config_path: Path to providers.yaml. Uses default if not specified.
        """
        self._config_path = Path(config_path or self.DEFAULT_CONFIG_PATH)
        self._raw_config: dict[str, Any] = {}
        self._providers: dict[str, ProviderConfig] = {}
        self._routing_strategies: dict[str, RoutingStrategyConfig] = {}
        self._embedding: EmbeddingConfig | None = None
        self._default_provider: str = "vastai"

        self._load_config()

    def _load_config(self) -> None:
        """Load and parse the providers YAML configuration."""
        if not self._config_path.exists():
            logger.warning(
                "LLM providers config not found at %s, using defaults",
                self._config_path,
            )
            self._setup_defaults()
            return

        try:
            with open(self._config_path, encoding="utf-8") as f:
                self._raw_config = yaml.safe_load(f)

            # Parse default provider
            self._default_provider = self._raw_config.get("default_provider", "vastai")

            # Parse providers
            providers_data = self._raw_config.get("providers", {})
            for code, config in providers_data.items():
                self._providers[code] = ProviderConfig.from_dict(code, config)

            # Parse routing strategies
            strategies_data = self._raw_config.get("routing_strategies", {})
            for name, config in strategies_data.items():
                self._routing_strategies[name] = RoutingStrategyConfig.from_dict(
                    name, config
                )

            # Parse embedding config
            embedding_data = self._raw_config.get("embedding", {})
            self._embedding = EmbeddingConfig.from_dict(embedding_data)

            logger.info(
                "Loaded LLM provider config: providers=%s, default=%s",
                list(self._providers.keys()),
                self._default_provider,
            )

        except yaml.YAMLError as e:
            logger.error("Failed to parse providers.yaml: %s", e)
            self._setup_defaults()
        except Exception as e:
            logger.error("Failed to load providers config: %s", e)
            self._setup_defaults()

    def _setup_defaults(self) -> None:
        """Setup default provider configuration when YAML is not available."""
        self._default_provider = "local"

        # Default local Ollama provider
        self._providers["local"] = ProviderConfig(
            code="local",
            type="ollama",
            description="Local Ollama Instance",
            api_base="http://localhost:11434",
            default_model="qwen2.5:7b",
            timeout_seconds=60,
        )

        # Default routing strategies
        self._routing_strategies["hybrid"] = RoutingStrategyConfig(
            name="hybrid",
            primary="vastai",
            fallbacks=["openai"],
        )

        self._embedding = EmbeddingConfig()

    @property
    def default_provider(self) -> str:
        """Get the default provider code."""
        return self._default_provider

    def get_provider(self, code: str) -> ProviderConfig | None:
        """Get provider configuration by code.

        Args:
            code: Provider code (e.g., 'vastai', 'local', 'openai').

        Returns:
            ProviderConfig or None if not found.
        """
        return self._providers.get(code)

    def get_default_provider_config(self) -> ProviderConfig:
        """Get the default provider configuration.

        Returns:
            ProviderConfig for the default provider.
        """
        provider = self.get_provider(self._default_provider)
        if not provider:
            # Fallback to first available provider
            for p in self._providers.values():
                if p.is_available():
                    return p
            raise ValueError("No available LLM provider configured")
        return provider

    def get_provider_for_model(self, model: str) -> ProviderConfig | None:
        """Get provider configuration for a model string.

        Determines provider from model string prefix:
        - "ollama/qwen2.5:7b" -> first enabled ollama provider
        - "gpt-4o" -> openai
        - "claude-3-5-sonnet" -> anthropic
        - "gemini/gemini-1.5-pro" -> google

        Args:
            model: Model string in LiteLLM format.

        Returns:
            ProviderConfig for the model's provider.
        """
        provider_type = self._detect_provider_type_from_model(model)

        # Find first enabled provider of this type
        for provider in self._providers.values():
            if provider.type == provider_type and provider.enabled:
                return provider

        return self.get_default_provider_config()

    def _detect_provider_type_from_model(self, model: str) -> str:
        """Detect provider type from model string.

        Args:
            model: Model string.

        Returns:
            Provider type.
        """
        model_lower = model.lower()

        if model_lower.startswith("ollama/") or model_lower.startswith("ollama_chat/"):
            return "ollama"
        elif model_lower.startswith("gpt") or model_lower.startswith("openai/"):
            return "openai"
        elif model_lower.startswith("claude") or model_lower.startswith("anthropic/"):
            return "anthropic"
        elif model_lower.startswith("gemini") or model_lower.startswith("google/"):
            return "google"
        else:
            # Default to ollama for unknown models
            return "ollama"

    def get_litellm_params_for_provider(
        self, provider_code: str, model: str | None = None
    ) -> dict[str, Any]:
        """Get LiteLLM parameters for a provider code.

        This is the primary method agents should use to get LLM parameters.

        Args:
            provider_code: Provider code (e.g., 'vastai', 'openai').
            model: Optional model override. Uses provider's default if not specified.

        Returns:
            Dictionary with model, api_base, and api_key for acompletion().

        Raises:
            ValueError: If provider code is not found.

        Example:
            >>> params = get_llm_params_for_provider("vastai")
            >>> response = await acompletion(**params, messages=[...])
        """
        provider = self.get_provider(provider_code)
        if not provider:
            raise ValueError(f"Unknown provider code: {provider_code}")

        return provider.get_litellm_params(model)

    def get_litellm_params(self, model: str) -> dict[str, Any]:
        """Get LiteLLM parameters for a model string.

        Determines provider from model string and returns appropriate params.

        Args:
            model: Model string in LiteLLM format.

        Returns:
            Dictionary with api_base and/or api_key if configured.
        """
        provider = self.get_provider_for_model(model)
        if provider:
            # Return params without changing the model
            params = provider.get_litellm_params()
            params["model"] = model  # Keep original model string
            return params
        return {"model": model}

    def get_routing_strategy(self, name: str) -> RoutingStrategyConfig | None:
        """Get routing strategy configuration.

        Args:
            name: Strategy name (e.g., 'hybrid', 'privacy_first').

        Returns:
            RoutingStrategyConfig or None.
        """
        return self._routing_strategies.get(name)

    def get_default_model(self, provider_code: str | None = None) -> str:
        """Get default model for a provider.

        Args:
            provider_code: Provider code. Uses default provider if not specified.

        Returns:
            Default model string in LiteLLM format.
        """
        code = provider_code or self._default_provider
        provider = self.get_provider(code)

        if provider:
            return provider.get_litellm_model()

        return "ollama/command-r:35b"

    @property
    def embedding(self) -> EmbeddingConfig:
        """Get embedding configuration."""
        return self._embedding or EmbeddingConfig()

    def list_providers(self) -> list[str]:
        """List all configured provider codes."""
        return list(self._providers.keys())

    def list_available_providers(self) -> list[str]:
        """List provider codes that are available for use."""
        return [
            code for code, config in self._providers.items() if config.is_available()
        ]

    def reload(self) -> None:
        """Reload configuration from file."""
        self._providers.clear()
        self._routing_strategies.clear()
        self._load_config()
        logger.info("LLM provider config reloaded")


# Singleton instance
_provider_manager: LLMProviderManager | None = None


def get_provider_manager() -> LLMProviderManager:
    """Get the singleton LLM provider manager.

    Returns:
        LLMProviderManager instance.
    """
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = LLMProviderManager()
    return _provider_manager


def reset_provider_manager() -> None:
    """Reset the singleton provider manager.

    Useful for testing or configuration reloads.
    """
    global _provider_manager
    _provider_manager = None


def get_provider_config(provider_code: str) -> ProviderConfig | None:
    """Get provider configuration by code.

    Convenience function for accessing provider config.

    Args:
        provider_code: Provider code.

    Returns:
        ProviderConfig or None.
    """
    return get_provider_manager().get_provider(provider_code)


def get_llm_params_for_provider(
    provider_code: str, model: str | None = None
) -> dict[str, Any]:
    """Get LiteLLM parameters for a provider code.

    This is the primary function agents should use.

    Args:
        provider_code: Provider code (e.g., 'vastai', 'openai').
        model: Optional model override.

    Returns:
        Dictionary with model, api_base, api_key for acompletion().

    Example:
        >>> params = get_llm_params_for_provider("vastai")
        >>> response = await acompletion(**params, messages=[...])
    """
    return get_provider_manager().get_litellm_params_for_provider(provider_code, model)


def get_llm_params(model: str) -> dict[str, Any]:
    """Get LiteLLM parameters for a model string.

    Determines provider from model string and returns params.

    Args:
        model: Model string in LiteLLM format.

    Returns:
        Dictionary with api_base and/or api_key.

    Example:
        >>> params = get_llm_params("ollama/command-r:35b")
        >>> response = await acompletion(**params, messages=[...])
    """
    return get_provider_manager().get_litellm_params(model)


def get_default_model() -> str:
    """Get the default model for the default provider.

    Returns:
        Model string in LiteLLM format.
    """
    return get_provider_manager().get_default_model()

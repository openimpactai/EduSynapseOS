# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Configuration package for EduSynapseOS.

This package provides centralized configuration management:
- Settings: Pydantic-based settings loaded from environment variables
- LLM Providers: YAML-based LLM provider configuration
- YAML loader: Utilities for loading YAML configuration files

Example:
    >>> from src.core.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.environment)
    'development'

    >>> from src.core.config import get_llm_params
    >>> params = get_llm_params("ollama/qwen2.5:7b")
    >>> # params contains api_base and api_key for the provider

    >>> from src.core.config import load_yaml
    >>> config = load_yaml(Path("config/persona.yaml"))
"""

from src.core.config.llm_providers import (
    EmbeddingConfig,
    LLMProviderManager,
    ProviderConfig,
    RoutingStrategyConfig,
    get_default_model,
    get_llm_params,
    get_llm_params_for_provider,
    get_provider_config,
    get_provider_manager,
    reset_provider_manager,
)
from src.core.config.settings import (
    APISettings,
    CentralDatabaseSettings,
    CORSSettings,
    EmbeddingSettings,
    JWTSettings,
    LLMSettings,
    OTelSettings,
    QdrantSettings,
    RateLimitSettings,
    RedisSettings,
    Settings,
    TenantDatabaseSettings,
    WorkerSettings,
    clear_settings_cache,
    get_settings,
)
from src.core.config.yaml_loader import (
    YAMLLoadError,
    deep_merge,
    load_yaml,
    load_yaml_directory,
)

__all__ = [
    # Settings
    "Settings",
    "get_settings",
    "clear_settings_cache",
    # Subsettings
    "CentralDatabaseSettings",
    "TenantDatabaseSettings",
    "RedisSettings",
    "QdrantSettings",
    "LLMSettings",
    "EmbeddingSettings",
    "JWTSettings",
    "RateLimitSettings",
    "CORSSettings",
    "OTelSettings",
    "APISettings",
    "WorkerSettings",
    # LLM Provider Configuration
    "LLMProviderManager",
    "ProviderConfig",
    "RoutingStrategyConfig",
    "EmbeddingConfig",
    "get_provider_manager",
    "get_provider_config",
    "get_llm_params",
    "get_llm_params_for_provider",
    "get_default_model",
    "reset_provider_manager",
    # YAML utilities
    "load_yaml",
    "load_yaml_directory",
    "deep_merge",
    "YAMLLoadError",
]

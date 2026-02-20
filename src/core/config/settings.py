# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Application configuration settings using Pydantic Settings.

This module provides centralized configuration management for EduSynapseOS.
Settings are loaded from environment variables with sensible defaults.

The Settings class is the main entry point and aggregates all subsettings.
A singleton instance is provided via get_settings() for dependency injection.

Example:
    >>> from src.core.config.settings import get_settings
    >>> settings = get_settings()
    >>> print(settings.environment)
    'development'
"""

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CentralDatabaseSettings(BaseSettings):
    """Central database configuration for platform management.

    The central database stores:
    - Tenant registry and connection info
    - System administrators
    - Licenses and feature flags

    Attributes:
        user: PostgreSQL username for central database.
        password: PostgreSQL password for central database.
        host: Database host address.
        port: Database port number.
        database: Database name.
        url: Full connection URL (computed from components if not set).
        pool_size: Connection pool size.
        max_overflow: Maximum overflow connections.
    """

    model_config = SettingsConfigDict(
        env_prefix="CENTRAL_DB_",
        extra="ignore",
    )

    user: str = "edusynapse"
    password: SecretStr = SecretStr("edusynapse_central_password")
    host: str = "edusynapse-central-db"
    port: int = 5432
    database: str = "edusynapse_central"
    pool_size: int = 10
    max_overflow: int = 20

    @property
    def url(self) -> str:
        """Build the async database URL from components."""
        pwd = self.password.get_secret_value()
        return f"postgresql+asyncpg://{self.user}:{pwd}@{self.host}:{self.port}/{self.database}"

    @property
    def sync_url(self) -> str:
        """Build the sync database URL for migrations."""
        pwd = self.password.get_secret_value()
        return f"postgresql://{self.user}:{pwd}@{self.host}:{self.port}/{self.database}"


class TenantDatabaseSettings(BaseSettings):
    """Tenant database configuration for dynamic container creation.

    Each tenant gets a separate PostgreSQL container with these credentials.
    Containers are created via Docker SDK using TenantContainerManager.

    Attributes:
        user: PostgreSQL username for tenant databases.
        password: PostgreSQL password for tenant databases.
        port_range_start: Starting port for tenant database containers.
        pool_size: Default connection pool size per tenant.
    """

    model_config = SettingsConfigDict(
        env_prefix="TENANT_DB_",
        extra="ignore",
    )

    user: str = "edusynapse"
    password: SecretStr = SecretStr("edusynapse_tenant_password")
    port_range_start: int = 35000
    pool_size: int = 10


class RedisSettings(BaseSettings):
    """Redis configuration for caching and message brokering.

    Tenant isolation is achieved via key prefix: tenant:{tenant_code}:*

    Attributes:
        host: Redis server host.
        port: Redis server port.
        password: Redis password.
        database: Redis database number.
        url: Full Redis connection URL.
        max_connections: Maximum connection pool size.
    """

    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        extra="ignore",
    )

    host: str = "edusynapse-redis"
    port: int = 6379
    password: SecretStr = SecretStr("edusynapse_redis_password")
    database: int = 0
    max_connections: int = 50

    @property
    def url(self) -> str:
        """Build the Redis connection URL."""
        pwd = self.password.get_secret_value()
        return f"redis://:{pwd}@{self.host}:{self.port}/{self.database}"


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration.

    Tenant isolation is achieved via collection naming: tenant_{code}_{collection}

    Attributes:
        host: Qdrant server host.
        http_port: HTTP API port.
        grpc_port: gRPC API port.
        api_key: Optional API key for authentication.
        prefer_grpc: Whether to prefer gRPC over HTTP.
        timeout: Request timeout in seconds.
    """

    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        extra="ignore",
    )

    host: str = "edusynapse-qdrant"
    http_port: int = 6333
    grpc_port: int = 6334
    api_key: SecretStr | None = None
    prefer_grpc: bool = True
    timeout: float = 30.0

    @property
    def url(self) -> str:
        """Build the Qdrant HTTP URL."""
        return f"http://{self.host}:{self.http_port}"


class LLMSettings(BaseSettings):
    """LLM provider configuration using LiteLLM.

    Supports multiple providers: ollama, openai, anthropic, google.
    LiteLLM handles provider routing based on model prefix.

    Attributes:
        default_provider: Default LLM provider to use.
        ollama_base_url: Base URL for Ollama server (local or remote like Vast.ai).
        ollama_api_key: API key for remote Ollama instances (e.g., Vast.ai).
        ollama_default_model: Default Ollama model.
        openai_api_key: OpenAI API key.
        openai_default_model: Default OpenAI model.
        anthropic_api_key: Anthropic API key.
        anthropic_default_model: Default Anthropic model.
        google_api_key: Google AI API key.
        google_default_model: Default Google model.
        request_timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.
    """

    model_config = SettingsConfigDict(
        extra="ignore",
    )

    default_provider: Literal["ollama", "openai", "anthropic", "google"] = "ollama"

    # Ollama (local or remote like Vast.ai)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    ollama_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OLLAMA_API_KEY",
    )
    ollama_default_model: str = Field(
        default="qwen2.5:7b",
        validation_alias="OLLAMA_DEFAULT_MODEL",
    )

    # OpenAI
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )
    openai_default_model: str = Field(
        default="gpt-4o",
        validation_alias="OPENAI_DEFAULT_MODEL",
    )

    # Anthropic
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
    )
    anthropic_default_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        validation_alias="ANTHROPIC_DEFAULT_MODEL",
    )

    # Google
    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="GOOGLE_API_KEY",
    )
    google_default_model: str = Field(
        default="gemini-2.0-flash-exp",
        validation_alias="GOOGLE_DEFAULT_MODEL",
    )

    request_timeout: float = 60.0
    max_retries: int = 3

    def get_default_model(self) -> str:
        """Get the default model for the configured provider.

        Returns:
            Model identifier string for the default provider.
        """
        models = {
            "ollama": f"ollama/{self.ollama_default_model}",
            "openai": self.openai_default_model,
            "anthropic": self.anthropic_default_model,
            "google": f"gemini/{self.google_default_model}",
        }
        return models[self.default_provider]


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration.

    Uses LiteLLM for API-based embedding generation (Ollama, OpenAI, etc.).

    Attributes:
        model: Model name in LiteLLM format (e.g., 'ollama/nomic-embed-text').
        dimension: Vector dimension (must match model output).
        batch_size: Batch size for embedding generation.
    """

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        extra="ignore",
    )

    model: str = "ollama/nomic-embed-text"
    dimension: int = 768
    batch_size: int = 32


class JWTSettings(BaseSettings):
    """JWT authentication configuration.

    Attributes:
        secret_key: Secret key for signing tokens.
        algorithm: JWT signing algorithm.
        access_token_expire_minutes: Access token expiration time.
        refresh_token_expire_days: Refresh token expiration time.
    """

    model_config = SettingsConfigDict(
        env_prefix="JWT_",
        extra="ignore",
    )

    secret_key: SecretStr = SecretStr("change-this-in-production")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(
        default=30,
        validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES",
    )
    refresh_token_expire_days: int = Field(
        default=7,
        validation_alias="REFRESH_TOKEN_EXPIRE_DAYS",
    )


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration.

    Attributes:
        requests_per_minute: Maximum requests per minute per client.
        burst: Maximum burst size for rate limiting.
    """

    model_config = SettingsConfigDict(
        env_prefix="RATE_LIMIT_",
        extra="ignore",
    )

    requests_per_minute: int = 60
    burst: int = 10


class CORSSettings(BaseSettings):
    """CORS configuration for API.

    Attributes:
        origins: Comma-separated list of allowed origins.
        allow_credentials: Whether to allow credentials.
        allow_methods: Allowed HTTP methods.
        allow_headers: Allowed HTTP headers.
    """

    model_config = SettingsConfigDict(
        env_prefix="CORS_",
        extra="ignore",
    )

    origins: str = "http://localhost:3000,http://localhost:5173"
    allow_credentials: bool = True
    allow_methods: list[str] = ["*"]
    allow_headers: list[str] = ["*"]

    @property
    def origins_list(self) -> list[str]:
        """Parse origins string into a list."""
        return [origin.strip() for origin in self.origins.split(",") if origin.strip()]


class OTelSettings(BaseSettings):
    """OpenTelemetry observability configuration.

    Attributes:
        enabled: Whether OpenTelemetry is enabled.
        service_name: Name of the service for tracing.
        exporter_endpoint: OTLP exporter endpoint.
    """

    model_config = SettingsConfigDict(
        env_prefix="OTEL_",
        extra="ignore",
    )

    enabled: bool = False
    service_name: str = "edusynapseos"
    exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317",
        validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT",
    )


class APISettings(BaseSettings):
    """API server configuration.

    Attributes:
        host: Host to bind to.
        port: Port to listen on.
        workers: Number of worker processes.
        reload: Whether to enable auto-reload.
    """

    model_config = SettingsConfigDict(
        env_prefix="API_",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 34000
    workers: int = 2
    reload: bool = False


class WorkerSettings(BaseSettings):
    """Background worker configuration.

    Attributes:
        processes: Number of worker processes.
        threads: Number of threads per process.
    """

    model_config = SettingsConfigDict(
        env_prefix="WORKER_",
        extra="ignore",
    )

    processes: int = 2
    threads: int = 4


class GameEngineSettings(BaseSettings):
    """Game Engine microservice configuration.

    The Game Engine container provides professional-level AI for board games:
    - Chess (Stockfish)
    - Gomoku (Rapfi)
    - Othello (Edax)
    - Checkers (Raven)
    - Connect4 (Built-in Minimax)

    Attributes:
        host: Game engine service host.
        port: Game engine service port.
        timeout: Request timeout in seconds.
        enabled: Whether to use the external game engine service.
    """

    model_config = SettingsConfigDict(
        env_prefix="GAME_ENGINE_",
        extra="ignore",
    )

    host: str = "edusynapse-game-engine"
    port: int = 34500
    timeout: float = 30.0
    enabled: bool = True

    @property
    def url(self) -> str:
        """Build the Game Engine service URL."""
        return f"http://{self.host}:{self.port}"


class H5PSettings(BaseSettings):
    """H5P server (Creatiq) configuration.

    The H5P server provides H5P content creation, storage, and playback.

    Attributes:
        api_url: Base URL of the H5P API server.
        api_key: API key for authentication.
        timeout: Request timeout in seconds.
    """

    model_config = SettingsConfigDict(
        env_prefix="H5P_",
        extra="ignore",
    )

    api_url: str = Field(
        default="http://localhost:3001",
        validation_alias="H5P_SERVER_URL",
    )
    api_key: SecretStr = SecretStr("")
    timeout: int = 30


class GeminiImageSettings(BaseSettings):
    """Google Gemini/Imagen image generation configuration.

    Attributes:
        api_key: Google AI API key (uses LLM settings if not set).
        model: Image generation model to use.
        timeout: Request timeout in seconds.
    """

    model_config = SettingsConfigDict(
        env_prefix="GEMINI_IMAGE_",
        extra="ignore",
    )

    api_key: SecretStr | None = None
    model: str = "gemini-2.0-flash-exp"
    timeout: int = 60


class GeminiAudioSettings(BaseSettings):
    """Google Gemini TTS audio generation configuration."""

    model_config = SettingsConfigDict(
        env_prefix="GEMINI_AUDIO_",
        extra="ignore",
    )

    api_key: SecretStr | None = None
    model: str = "gemini-2.5-flash-preview-tts"
    timeout: int = 120
    max_duration_seconds: int = 20


class GeminiVideoSettings(BaseSettings):
    """Google Veo video generation configuration."""

    model_config = SettingsConfigDict(
        env_prefix="GEMINI_VIDEO_",
        extra="ignore",
    )

    api_key: SecretStr | None = None
    model: str = "veo-3.1-generate-preview"
    timeout: int = 300
    max_duration_seconds: int = 20
    poll_interval: int = 5


class CentralCurriculumSettings(BaseSettings):
    """Central Curriculum service configuration.

    The Central Curriculum service provides centralized curriculum data
    (frameworks, stages, grades, subjects, units, topics, objectives).
    EduSynapse Backend syncs curriculum data from this service.

    Attributes:
        base_url: Base URL of the Central Curriculum API.
        api_key: API key for authentication.
        api_secret: API secret for authentication.
        timeout: Request timeout in seconds.
        sync_enabled: Whether automatic sync is enabled.
        sync_interval_hours: Hours between sync runs (default: 24).
        trusted_system_secret: Shared secret for SSO token generation.
        frontend_url: Central Curriculum frontend URL for SSO redirects.
    """

    model_config = SettingsConfigDict(
        env_prefix="CENTRAL_CURRICULUM_",
        extra="ignore",
    )

    base_url: str = "http://edusynapse-cc-backend:8000/api/v1"
    api_key: SecretStr = SecretStr("")
    api_secret: SecretStr = SecretStr("")
    timeout: float = 60.0
    sync_enabled: bool = True
    sync_interval_hours: int = 24
    trusted_system_secret: SecretStr = SecretStr("")
    frontend_url: str = "http://localhost:3000"

    @property
    def auth_headers(self) -> dict[str, str]:
        """Build authentication headers for API requests."""
        return {
            "X-API-Key": self.api_key.get_secret_value(),
            "X-API-Secret": self.api_secret.get_secret_value(),
        }

    @property
    def sso_headers(self) -> dict[str, str]:
        """Build headers for SSO token generation."""
        return {
            "X-Trusted-System-Secret": self.trusted_system_secret.get_secret_value(),
            "Content-Type": "application/json",
        }


class Settings(BaseSettings):
    """Main application settings aggregating all subsettings.

    This is the primary configuration class for EduSynapseOS.
    Use get_settings() to obtain a cached singleton instance.

    Attributes:
        environment: Current environment (development, staging, production).
        debug: Enable debug mode.
        log_level: Logging level.
        central_db: Central database settings.
        tenant_db: Tenant database settings.
        redis: Redis settings.
        qdrant: Qdrant settings.
        llm: LLM provider settings.
        embedding: Embedding model settings.
        jwt: JWT authentication settings.
        rate_limit: Rate limiting settings.
        cors: CORS settings.
        otel: OpenTelemetry settings.
        api: API server settings.
        worker: Background worker settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"

    # Subsettings - loaded with their own env prefixes
    central_db: CentralDatabaseSettings = Field(default_factory=CentralDatabaseSettings)
    tenant_db: TenantDatabaseSettings = Field(default_factory=TenantDatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    jwt: JWTSettings = Field(default_factory=JWTSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)
    otel: OTelSettings = Field(default_factory=OTelSettings)
    api: APISettings = Field(default_factory=APISettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    game_engine: GameEngineSettings = Field(default_factory=GameEngineSettings)
    central_curriculum: CentralCurriculumSettings = Field(default_factory=CentralCurriculumSettings)
    h5p: H5PSettings = Field(default_factory=H5PSettings)
    gemini_image: GeminiImageSettings = Field(default_factory=GeminiImageSettings)
    gemini_audio: GeminiAudioSettings = Field(default_factory=GeminiAudioSettings)
    gemini_video: GeminiVideoSettings = Field(default_factory=GeminiVideoSettings)

    @model_validator(mode="after")
    def validate_production_settings(self) -> Self:
        """Validate that production settings are properly configured.

        Raises:
            ValueError: If running in production with insecure defaults.
        """
        if self.environment == "production":
            default_jwt_secret = "change-this-in-production"
            if self.jwt.secret_key.get_secret_value() == default_jwt_secret:
                raise ValueError(
                    "JWT secret key must be changed from default in production. "
                    "Set JWT_SECRET_KEY environment variable."
                )
        return self

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    Call clear_settings_cache() if you need to reload settings.

    Returns:
        Cached Settings instance.
    """
    return Settings()


def clear_settings_cache() -> None:
    """Clear the settings cache.

    Call this if you need to reload settings from environment.
    Useful for testing or dynamic configuration updates.
    """
    get_settings.cache_clear()

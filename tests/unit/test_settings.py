# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for application settings."""

import os
from unittest.mock import patch

import pytest

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


class TestCentralDatabaseSettings:
    """Tests for CentralDatabaseSettings."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = CentralDatabaseSettings()

        assert settings.user == "edusynapse"
        assert settings.host == "localhost"
        assert settings.port == 5432
        assert settings.database == "edusynapse_central"
        assert settings.pool_size == 10
        assert settings.max_overflow == 20

    def test_url_property(self) -> None:
        """Test URL property builds correct connection string."""
        settings = CentralDatabaseSettings(
            user="testuser",
            password="testpass",  # type: ignore[arg-type]
            host="db.example.com",
            port=5433,
            database="testdb",
        )

        url = settings.url

        assert url == "postgresql+asyncpg://testuser:testpass@db.example.com:5433/testdb"

    def test_sync_url_property(self) -> None:
        """Test sync URL property builds correct connection string."""
        settings = CentralDatabaseSettings(
            user="testuser",
            password="testpass",  # type: ignore[arg-type]
            host="db.example.com",
            port=5433,
            database="testdb",
        )

        url = settings.sync_url

        assert url == "postgresql://testuser:testpass@db.example.com:5433/testdb"

    def test_loads_from_environment(self) -> None:
        """Test that settings load from environment variables."""
        env = {
            "CENTRAL_DB_USER": "envuser",
            "CENTRAL_DB_HOST": "envhost",
            "CENTRAL_DB_PORT": "5555",
        }

        with patch.dict(os.environ, env, clear=False):
            settings = CentralDatabaseSettings()

        assert settings.user == "envuser"
        assert settings.host == "envhost"
        assert settings.port == 5555


class TestTenantDatabaseSettings:
    """Tests for TenantDatabaseSettings."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = TenantDatabaseSettings()

        assert settings.user == "edusynapse"
        assert settings.port_range_start == 5500
        assert settings.pool_size == 10

    def test_loads_from_environment(self) -> None:
        """Test that settings load from environment variables."""
        env = {
            "TENANT_DB_USER": "tenant_user",
            "TENANT_DB_PORT_RANGE_START": "6000",
        }

        with patch.dict(os.environ, env, clear=False):
            settings = TenantDatabaseSettings()

        assert settings.user == "tenant_user"
        assert settings.port_range_start == 6000


class TestRedisSettings:
    """Tests for RedisSettings."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = RedisSettings()

        assert settings.host == "localhost"
        assert settings.port == 6379
        assert settings.database == 0
        assert settings.max_connections == 50

    def test_url_property(self) -> None:
        """Test URL property builds correct connection string."""
        settings = RedisSettings(
            host="redis.example.com",
            port=6380,
            password="redispass",  # type: ignore[arg-type]
            database=1,
        )

        url = settings.url

        assert url == "redis://:redispass@redis.example.com:6380/1"


class TestQdrantSettings:
    """Tests for QdrantSettings."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = QdrantSettings()

        assert settings.host == "localhost"
        assert settings.http_port == 6333
        assert settings.grpc_port == 6334
        assert settings.api_key is None
        assert settings.prefer_grpc is True
        assert settings.timeout == 30.0

    def test_url_property(self) -> None:
        """Test URL property builds correct URL."""
        settings = QdrantSettings(host="qdrant.example.com", http_port=6334)

        url = settings.url

        assert url == "http://qdrant.example.com:6334"


class TestLLMSettings:
    """Tests for LLMSettings."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = LLMSettings()

        assert settings.default_provider == "ollama"
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.ollama_default_model == "qwen2.5:7b"
        assert settings.openai_default_model == "gpt-4o"
        assert settings.anthropic_default_model == "claude-3-5-sonnet-20241022"
        assert settings.google_default_model == "gemini-2.0-flash-exp"
        assert settings.request_timeout == 60.0
        assert settings.max_retries == 3

    def test_get_default_model_ollama(self) -> None:
        """Test getting default model for Ollama."""
        settings = LLMSettings(default_provider="ollama")

        model = settings.get_default_model()

        assert model == "ollama/qwen2.5:7b"

    def test_get_default_model_openai(self) -> None:
        """Test getting default model for OpenAI."""
        settings = LLMSettings(default_provider="openai")

        model = settings.get_default_model()

        assert model == "gpt-4o"

    def test_get_default_model_anthropic(self) -> None:
        """Test getting default model for Anthropic."""
        settings = LLMSettings(default_provider="anthropic")

        model = settings.get_default_model()

        assert model == "claude-3-5-sonnet-20241022"

    def test_get_default_model_google(self) -> None:
        """Test getting default model for Google."""
        settings = LLMSettings(default_provider="google")

        model = settings.get_default_model()

        assert model == "gemini/gemini-2.0-flash-exp"


class TestEmbeddingSettings:
    """Tests for EmbeddingSettings."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = EmbeddingSettings()

        assert settings.model == "paraphrase-multilingual-MiniLM-L12-v2"
        assert settings.dimension == 384
        assert settings.batch_size == 32
        assert settings.normalize is True


class TestJWTSettings:
    """Tests for JWTSettings."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = JWTSettings()

        assert settings.algorithm == "HS256"
        assert settings.access_token_expire_minutes == 30
        assert settings.refresh_token_expire_days == 7


class TestRateLimitSettings:
    """Tests for RateLimitSettings."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = RateLimitSettings()

        assert settings.requests_per_minute == 60
        assert settings.burst == 10


class TestCORSSettings:
    """Tests for CORSSettings."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = CORSSettings()

        assert "http://localhost:3000" in settings.origins
        assert settings.allow_credentials is True
        assert settings.allow_methods == ["*"]
        assert settings.allow_headers == ["*"]

    def test_origins_list_property(self) -> None:
        """Test origins_list property parses correctly."""
        settings = CORSSettings(origins="http://a.com, http://b.com , http://c.com")

        origins = settings.origins_list

        assert origins == ["http://a.com", "http://b.com", "http://c.com"]


class TestOTelSettings:
    """Tests for OTelSettings."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = OTelSettings()

        assert settings.enabled is False
        assert settings.service_name == "edusynapseos"
        assert settings.exporter_otlp_endpoint == "http://localhost:4317"


class TestAPISettings:
    """Tests for APISettings."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = APISettings()

        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.workers == 2
        assert settings.reload is False


class TestWorkerSettings:
    """Tests for WorkerSettings."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = WorkerSettings()

        assert settings.processes == 2
        assert settings.threads == 4


class TestSettings:
    """Tests for main Settings class."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = Settings()

        assert settings.environment == "development"
        assert settings.debug is True
        assert settings.log_level == "DEBUG"

    def test_production_with_default_jwt_secret_raises_error(self) -> None:
        """Test that production environment with default JWT secret raises error."""
        with pytest.raises(ValueError) as exc_info:
            Settings(environment="production")

        assert "JWT secret key must be changed" in str(exc_info.value)

    def test_production_with_custom_jwt_secret_succeeds(self) -> None:
        """Test that production environment with custom JWT secret works."""
        env = {"JWT_SECRET_KEY": "my-secure-production-secret-key-12345"}

        with patch.dict(os.environ, env, clear=False):
            settings = Settings(environment="production")

        assert settings.environment == "production"

    def test_subsettings_loaded(self) -> None:
        """Test that all subsettings are loaded."""
        settings = Settings()

        assert isinstance(settings.central_db, CentralDatabaseSettings)
        assert isinstance(settings.tenant_db, TenantDatabaseSettings)
        assert isinstance(settings.redis, RedisSettings)
        assert isinstance(settings.qdrant, QdrantSettings)
        assert isinstance(settings.llm, LLMSettings)
        assert isinstance(settings.embedding, EmbeddingSettings)
        assert isinstance(settings.jwt, JWTSettings)
        assert isinstance(settings.rate_limit, RateLimitSettings)
        assert isinstance(settings.cors, CORSSettings)
        assert isinstance(settings.otel, OTelSettings)
        assert isinstance(settings.api, APISettings)
        assert isinstance(settings.worker, WorkerSettings)

    def test_is_development_property(self) -> None:
        """Test is_development property."""
        dev_settings = Settings(environment="development")
        prod_settings = Settings(environment="production")

        assert dev_settings.is_development is True
        assert prod_settings.is_development is False

    def test_is_production_property(self) -> None:
        """Test is_production property."""
        dev_settings = Settings(environment="development")
        prod_settings = Settings(environment="production")

        assert dev_settings.is_production is False
        assert prod_settings.is_production is True


class TestGetSettings:
    """Tests for get_settings function."""

    def test_returns_settings_instance(self) -> None:
        """Test that get_settings returns a Settings instance."""
        clear_settings_cache()

        settings = get_settings()

        assert isinstance(settings, Settings)

    def test_returns_cached_instance(self) -> None:
        """Test that get_settings returns cached instance."""
        clear_settings_cache()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_clear_cache_allows_reload(self) -> None:
        """Test that clearing cache allows reloading settings."""
        clear_settings_cache()

        settings1 = get_settings()
        clear_settings_cache()
        settings2 = get_settings()

        # After clearing cache, a new instance should be created
        # Note: instances will be equal but not identical
        assert settings1 is not settings2

# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for JWT token utilities.

Tests the JWTManager class and token operations.
"""

import time
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import SecretStr

from src.domains.auth.jwt import (
    JWTManager,
    TokenPayload,
    TokenPair,
    TokenExpiredError,
    InvalidTokenError,
)


@pytest.fixture
def jwt_settings() -> MagicMock:
    """Create mock JWT settings."""
    settings = MagicMock()
    settings.secret_key = SecretStr("test-secret-key-for-jwt-testing")
    settings.algorithm = "HS256"
    settings.access_token_expire_minutes = 30
    settings.refresh_token_expire_days = 7
    return settings


@pytest.fixture
def jwt_manager(jwt_settings: MagicMock) -> JWTManager:
    """Create JWT manager with test settings."""
    return JWTManager(jwt_settings)


class TestJWTManager:
    """Tests for JWTManager class."""

    def test_create_token_pair_returns_valid_tokens(
        self,
        jwt_manager: JWTManager,
    ) -> None:
        """Test that create_token_pair returns valid token pair."""
        user_id = str(uuid4())
        tenant_id = str(uuid4())

        result = jwt_manager.create_token_pair(
            user_id=user_id,
            tenant_id=tenant_id,
            tenant_code="test_tenant",
            user_type="student",
            roles=["student"],
            permissions=["practice.create"],
            school_ids=[str(uuid4())],
        )

        assert isinstance(result, TokenPair)
        assert result.access_token is not None
        assert result.refresh_token is not None
        assert result.token_type == "Bearer"
        assert result.expires_in == 30 * 60  # 30 minutes in seconds
        assert result.refresh_expires_in == 7 * 24 * 60 * 60  # 7 days in seconds

    def test_create_access_token_returns_valid_token(
        self,
        jwt_manager: JWTManager,
    ) -> None:
        """Test that create_access_token returns valid token string."""
        user_id = str(uuid4())

        token = jwt_manager.create_access_token(
            user_id=user_id,
            user_type="teacher",
        )

        assert isinstance(token, str)
        assert len(token) > 0

    def test_decode_access_token_returns_payload(
        self,
        jwt_manager: JWTManager,
    ) -> None:
        """Test that decode_token returns correct payload for access token."""
        user_id = str(uuid4())
        tenant_id = str(uuid4())
        school_id = str(uuid4())

        token = jwt_manager.create_access_token(
            user_id=user_id,
            tenant_id=tenant_id,
            tenant_code="test_tenant",
            user_type="student",
            roles=["student", "reader"],
            permissions=["practice.create", "practice.view"],
            school_ids=[school_id],
        )

        payload = jwt_manager.decode_token(token)

        assert isinstance(payload, TokenPayload)
        assert payload.sub == user_id
        assert payload.type == "access"
        assert payload.tenant_id == tenant_id
        assert payload.tenant_code == "test_tenant"
        assert payload.user_type == "student"
        assert "student" in payload.roles
        assert "reader" in payload.roles
        assert "practice.create" in payload.permissions
        assert school_id in payload.school_ids

    def test_decode_refresh_token_returns_payload(
        self,
        jwt_manager: JWTManager,
    ) -> None:
        """Test that decode_token returns correct payload for refresh token."""
        user_id = str(uuid4())

        tokens = jwt_manager.create_token_pair(user_id=user_id)
        payload = jwt_manager.decode_token(tokens.refresh_token, expected_type="refresh")

        assert payload.sub == user_id
        assert payload.type == "refresh"

    def test_decode_token_with_wrong_type_raises_error(
        self,
        jwt_manager: JWTManager,
    ) -> None:
        """Test that decode_token raises error when type doesn't match."""
        user_id = str(uuid4())
        token = jwt_manager.create_access_token(user_id=user_id)

        with pytest.raises(InvalidTokenError, match="Expected refresh token"):
            jwt_manager.decode_token(token, expected_type="refresh")

    def test_decode_expired_token_raises_error(
        self,
        jwt_settings: MagicMock,
    ) -> None:
        """Test that decode_token raises error for expired token."""
        # Create manager with very short expiration
        jwt_settings.access_token_expire_minutes = 0
        jwt_manager = JWTManager(jwt_settings)

        user_id = str(uuid4())
        token = jwt_manager.create_access_token(user_id=user_id)

        # Wait for expiration (since minutes=0, token expires immediately)
        time.sleep(0.1)

        with pytest.raises(TokenExpiredError, match="Token has expired"):
            jwt_manager.decode_token(token)

    def test_decode_invalid_token_raises_error(
        self,
        jwt_manager: JWTManager,
    ) -> None:
        """Test that decode_token raises error for invalid token."""
        with pytest.raises(InvalidTokenError):
            jwt_manager.decode_token("invalid.token.here")

    def test_decode_token_with_wrong_secret_raises_error(
        self,
        jwt_manager: JWTManager,
        jwt_settings: MagicMock,
    ) -> None:
        """Test that decode fails when secret doesn't match."""
        user_id = str(uuid4())
        token = jwt_manager.create_access_token(user_id=user_id)

        # Create manager with different secret
        jwt_settings.secret_key = SecretStr("different-secret-key")
        other_manager = JWTManager(jwt_settings)

        with pytest.raises(InvalidTokenError):
            other_manager.decode_token(token)

    def test_verify_token_returns_true_for_valid(
        self,
        jwt_manager: JWTManager,
    ) -> None:
        """Test that verify_token returns True for valid token."""
        user_id = str(uuid4())
        token = jwt_manager.create_access_token(user_id=user_id)

        result = jwt_manager.verify_token(token)

        assert result is True

    def test_verify_token_returns_false_for_invalid(
        self,
        jwt_manager: JWTManager,
    ) -> None:
        """Test that verify_token returns False for invalid token."""
        result = jwt_manager.verify_token("invalid.token.here")

        assert result is False

    def test_verify_token_with_type_check(
        self,
        jwt_manager: JWTManager,
    ) -> None:
        """Test that verify_token respects expected_type."""
        user_id = str(uuid4())
        access_token = jwt_manager.create_access_token(user_id=user_id)

        assert jwt_manager.verify_token(access_token, expected_type="access") is True
        assert jwt_manager.verify_token(access_token, expected_type="refresh") is False

    def test_hash_token_returns_consistent_hash(
        self,
        jwt_manager: JWTManager,
    ) -> None:
        """Test that hash_token returns consistent SHA-256 hash."""
        token = "test-token-string"

        hash1 = jwt_manager.hash_token(token)
        hash2 = jwt_manager.hash_token(token)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex characters

    def test_hash_token_different_tokens_different_hash(
        self,
        jwt_manager: JWTManager,
    ) -> None:
        """Test that different tokens produce different hashes."""
        hash1 = jwt_manager.hash_token("token1")
        hash2 = jwt_manager.hash_token("token2")

        assert hash1 != hash2

    def test_token_pair_contains_jti(
        self,
        jwt_manager: JWTManager,
    ) -> None:
        """Test that tokens contain unique JTI claims."""
        user_id = str(uuid4())

        tokens1 = jwt_manager.create_token_pair(user_id=user_id)
        tokens2 = jwt_manager.create_token_pair(user_id=user_id)

        payload1 = jwt_manager.decode_token(tokens1.access_token)
        payload2 = jwt_manager.decode_token(tokens2.access_token)

        assert payload1.jti is not None
        assert payload2.jti is not None
        assert payload1.jti != payload2.jti

    def test_token_payload_timestamps(
        self,
        jwt_manager: JWTManager,
    ) -> None:
        """Test that tokens have correct iat and exp timestamps."""
        user_id = str(uuid4())
        before = int(time.time())

        token = jwt_manager.create_access_token(user_id=user_id)

        after = int(time.time())
        payload = jwt_manager.decode_token(token)

        # iat should be between before and after
        assert before <= payload.iat <= after

        # exp should be about 30 minutes after iat
        expected_exp = payload.iat + 30 * 60
        assert abs(payload.exp - expected_exp) <= 1

    def test_uuid_conversion(
        self,
        jwt_manager: JWTManager,
    ) -> None:
        """Test that UUID objects are properly converted to strings."""
        user_id = uuid4()
        tenant_id = uuid4()
        school_id = uuid4()

        token = jwt_manager.create_access_token(
            user_id=user_id,
            tenant_id=tenant_id,
            school_ids=[school_id],
        )

        payload = jwt_manager.decode_token(token)

        assert payload.sub == str(user_id)
        assert payload.tenant_id == str(tenant_id)
        assert str(school_id) in payload.school_ids

    def test_optional_fields_can_be_none(
        self,
        jwt_manager: JWTManager,
    ) -> None:
        """Test that optional fields work when not provided."""
        user_id = str(uuid4())

        token = jwt_manager.create_access_token(user_id=user_id)
        payload = jwt_manager.decode_token(token)

        assert payload.tenant_id is None
        assert payload.tenant_code is None
        assert payload.user_type is None
        assert payload.roles == []
        assert payload.permissions == []
        assert payload.school_ids == []

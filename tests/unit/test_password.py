# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for password hashing utilities.

Tests the PasswordHasher class and convenience functions.
"""

import pytest

from src.domains.auth.password import (
    PasswordHasher,
    hash_password,
    verify_password,
)


class TestPasswordHasher:
    """Tests for PasswordHasher class."""

    def test_hash_password_returns_bcrypt_hash(self) -> None:
        """Test that hashing returns a bcrypt hash string."""
        hasher = PasswordHasher()
        password = "test_password_123"

        hashed = hasher.hash(password)

        assert hashed is not None
        assert isinstance(hashed, str)
        assert hashed.startswith("$2b$")  # bcrypt prefix
        assert len(hashed) == 60  # bcrypt hash length

    def test_hash_produces_different_hashes_for_same_password(self) -> None:
        """Test that hashing the same password produces different hashes (due to salt)."""
        hasher = PasswordHasher()
        password = "test_password_123"

        hash1 = hasher.hash(password)
        hash2 = hasher.hash(password)

        assert hash1 != hash2

    def test_verify_correct_password(self) -> None:
        """Test that verification succeeds with correct password."""
        hasher = PasswordHasher()
        password = "correct_password"
        hashed = hasher.hash(password)

        result = hasher.verify(password, hashed)

        assert result is True

    def test_verify_incorrect_password(self) -> None:
        """Test that verification fails with incorrect password."""
        hasher = PasswordHasher()
        password = "correct_password"
        wrong_password = "wrong_password"
        hashed = hasher.hash(password)

        result = hasher.verify(wrong_password, hashed)

        assert result is False

    def test_verify_empty_password_returns_false(self) -> None:
        """Test that verification fails with empty password."""
        hasher = PasswordHasher()
        hashed = hasher.hash("valid_password")

        result = hasher.verify("", hashed)

        assert result is False

    def test_verify_empty_hash_returns_false(self) -> None:
        """Test that verification fails with empty hash."""
        hasher = PasswordHasher()

        result = hasher.verify("password", "")

        assert result is False

    def test_verify_invalid_hash_returns_false(self) -> None:
        """Test that verification fails with invalid hash format."""
        hasher = PasswordHasher()

        result = hasher.verify("password", "not_a_valid_bcrypt_hash")

        assert result is False

    def test_hash_empty_password_raises_error(self) -> None:
        """Test that hashing empty password raises ValueError."""
        hasher = PasswordHasher()

        with pytest.raises(ValueError, match="Password cannot be empty"):
            hasher.hash("")

    def test_needs_rehash_returns_false_for_current_hash(self) -> None:
        """Test that needs_rehash returns False for current settings."""
        hasher = PasswordHasher(rounds=12)
        hashed = hasher.hash("password")

        result = hasher.needs_rehash(hashed)

        assert result is False

    def test_needs_rehash_returns_true_for_old_rounds(self) -> None:
        """Test that needs_rehash returns True when rounds changed."""
        old_hasher = PasswordHasher(rounds=4)
        new_hasher = PasswordHasher(rounds=12)

        old_hash = old_hasher.hash("password")
        result = new_hasher.needs_rehash(old_hash)

        assert result is True

    def test_needs_rehash_empty_hash_returns_false(self) -> None:
        """Test that needs_rehash returns False for empty hash."""
        hasher = PasswordHasher()

        result = hasher.needs_rehash("")

        assert result is False

    def test_custom_rounds_parameter(self) -> None:
        """Test that custom rounds parameter is used."""
        hasher_low = PasswordHasher(rounds=4)
        hasher_high = PasswordHasher(rounds=10)

        # Both should produce valid hashes
        hash_low = hasher_low.hash("password")
        hash_high = hasher_high.hash("password")

        assert hasher_low.verify("password", hash_low)
        assert hasher_high.verify("password", hash_high)

    def test_unicode_password(self) -> None:
        """Test that unicode passwords work correctly."""
        hasher = PasswordHasher()
        password = "şifre_parola_密码"

        hashed = hasher.hash(password)
        result = hasher.verify(password, hashed)

        assert result is True

    def test_long_password(self) -> None:
        """Test that long passwords work correctly."""
        hasher = PasswordHasher()
        password = "a" * 1000  # Very long password

        hashed = hasher.hash(password)
        result = hasher.verify(password, hashed)

        assert result is True


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_hash_password_function(self) -> None:
        """Test hash_password convenience function."""
        password = "test_password"

        hashed = hash_password(password)

        assert hashed is not None
        assert hashed.startswith("$2b$")

    def test_verify_password_function_correct(self) -> None:
        """Test verify_password with correct password."""
        password = "test_password"
        hashed = hash_password(password)

        result = verify_password(password, hashed)

        assert result is True

    def test_verify_password_function_incorrect(self) -> None:
        """Test verify_password with incorrect password."""
        password = "test_password"
        hashed = hash_password(password)

        result = verify_password("wrong_password", hashed)

        assert result is False

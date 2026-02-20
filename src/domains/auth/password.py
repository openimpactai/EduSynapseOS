# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Password hashing utilities using bcrypt.

This module provides secure password hashing and verification
using the bcrypt library directly.

Example:
    >>> hasher = PasswordHasher()
    >>> hashed = hasher.hash("my_password")
    >>> hasher.verify("my_password", hashed)
    True
"""

import logging

import bcrypt

logger = logging.getLogger(__name__)


class PasswordHasher:
    """Secure password hashing using bcrypt.

    Uses bcrypt for secure password hashing with automatic salt generation.
    The default rounds value of 12 provides a good balance between security
    and performance.

    Attributes:
        _rounds: Number of bcrypt rounds for hashing.

    Example:
        >>> hasher = PasswordHasher()
        >>> hashed = hasher.hash("secure_password")
        >>> hasher.verify("secure_password", hashed)
        True
        >>> hasher.verify("wrong_password", hashed)
        False
    """

    def __init__(self, rounds: int = 12) -> None:
        """Initialize the password hasher.

        Args:
            rounds: Number of bcrypt rounds. Higher is more secure but slower.
                   Default is 12 which takes ~250ms on modern hardware.
        """
        self._rounds = rounds

    def hash(self, password: str) -> str:
        """Hash a password using bcrypt.

        Args:
            password: Plain text password to hash.

        Returns:
            Bcrypt hash string with salt embedded.

        Raises:
            ValueError: If password is empty.
        """
        if not password:
            raise ValueError("Password cannot be empty")

        salt = bcrypt.gensalt(rounds=self._rounds)
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    def verify(self, password: str, password_hash: str) -> bool:
        """Verify a password against a hash.

        Args:
            password: Plain text password to verify.
            password_hash: Bcrypt hash to verify against.

        Returns:
            True if password matches the hash, False otherwise.
        """
        if not password or not password_hash:
            return False

        try:
            return bcrypt.checkpw(
                password.encode("utf-8"),
                password_hash.encode("utf-8"),
            )
        except Exception as e:
            logger.warning("Password verification failed: %s", str(e))
            return False

    def needs_rehash(self, password_hash: str) -> bool:
        """Check if a hash needs to be updated.

        This is useful when upgrading the hash rounds or algorithm.
        Currently always returns False as we don't track rounds in the hash.

        Args:
            password_hash: Existing password hash to check.

        Returns:
            True if the hash should be updated, False otherwise.
        """
        if not password_hash:
            return False

        return False


# Default instance for convenience
_default_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """Hash a password using the default hasher.

    Convenience function using the default PasswordHasher instance.

    Args:
        password: Plain text password to hash.

    Returns:
        Bcrypt hash string.
    """
    return _default_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password using the default hasher.

    Convenience function using the default PasswordHasher instance.

    Args:
        password: Plain text password to verify.
        password_hash: Bcrypt hash to verify against.

    Returns:
        True if password matches, False otherwise.
    """
    return _default_hasher.verify(password, password_hash)

# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Database migration runner.

This module provides programmatic migration execution for tenant databases.
Used by TenantService when provisioning new tenants.

Example:
    from src.infrastructure.database.migrations.runner import run_tenant_migrations

    # Run all pending migrations for a tenant
    await run_tenant_migrations(
        tenant_code="test_school",
        db_url="postgresql+asyncpg://user:pass@host:port/db"
    )
"""

import asyncio
import importlib
import logging
from pathlib import Path
from typing import Callable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger(__name__)

# Migration files in order (must be maintained manually)
TENANT_MIGRATIONS = [
    "001_initial_schema",
    "002_add_fsrs_fields",
    "003_add_emotional_intelligence_core",
    "004_add_companion_sessions",
    "005_add_student_notes",
    "006_remove_password_auth",
    "007_add_companion_activities",
    "008_add_practice_helper",
    "009_add_learning_sessions",
    "010_add_academic_year_code",
    "011_add_gaming_tables",
    "012_curriculum_composite_keys",
    "013_add_skill_tables",
    "014_add_practice_learning_session_link",
    "015_add_new_session_started_completion_reason",
]


async def run_tenant_migrations(
    db_url: str,
    target_revision: str | None = None,
) -> list[str]:
    """Run pending migrations for a tenant database.

    This function runs migrations programmatically without using alembic CLI.
    It creates a version tracking table if not exists and applies pending migrations.

    Args:
        db_url: Database connection URL (asyncpg format).
        target_revision: Optional specific revision to migrate to.
            If None, runs all pending migrations.

    Returns:
        List of applied migration revision IDs.

    Raises:
        Exception: If any migration fails.
    """
    engine = create_async_engine(db_url, echo=False)

    try:
        # Ensure version table exists
        await _ensure_version_table(engine)

        # Get current version
        current_version = await _get_current_version(engine)
        logger.info("Current migration version: %s", current_version or "None")

        # Determine migrations to apply
        migrations_to_apply = _get_pending_migrations(current_version, target_revision)

        if not migrations_to_apply:
            logger.info("No pending migrations")
            return []

        logger.info(
            "Applying %d migrations: %s",
            len(migrations_to_apply),
            ", ".join(migrations_to_apply),
        )

        applied = []
        for revision in migrations_to_apply:
            await _apply_migration(engine, revision)
            applied.append(revision)
            logger.info("Applied migration: %s", revision)

        return applied

    finally:
        await engine.dispose()


async def _ensure_version_table(engine: AsyncEngine) -> None:
    """Create alembic_version table if not exists."""
    async with engine.begin() as conn:
        await conn.execute(
            text("""
                CREATE TABLE IF NOT EXISTS alembic_version (
                    version_num VARCHAR(128) NOT NULL,
                    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
                )
            """)
        )


async def _get_current_version(engine: AsyncEngine) -> str | None:
    """Get current migration version from database."""
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT version_num FROM alembic_version LIMIT 1")
        )
        row = result.fetchone()
        return row[0] if row else None


def _get_pending_migrations(
    current_version: str | None,
    target_revision: str | None = None,
) -> list[str]:
    """Get list of migrations to apply.

    Args:
        current_version: Current database version.
        target_revision: Target revision to migrate to.

    Returns:
        List of revision IDs to apply in order.
    """
    if current_version is None:
        # No migrations applied yet - start from beginning
        start_idx = 0
    else:
        # Find current version in list
        try:
            current_idx = TENANT_MIGRATIONS.index(current_version)
            start_idx = current_idx + 1
        except ValueError:
            # Current version not in list - might be a custom migration
            logger.warning(
                "Current version %s not in known migrations list", current_version
            )
            return []

    if target_revision:
        try:
            end_idx = TENANT_MIGRATIONS.index(target_revision) + 1
        except ValueError:
            logger.warning("Target revision %s not found", target_revision)
            return []
    else:
        end_idx = len(TENANT_MIGRATIONS)

    return TENANT_MIGRATIONS[start_idx:end_idx]


async def _apply_migration(engine: AsyncEngine, revision: str) -> None:
    """Apply a single migration.

    Args:
        engine: Database engine.
        revision: Migration revision ID.

    Raises:
        Exception: If migration fails.
    """
    # Import the migration module
    module_name = f"src.infrastructure.database.migrations.tenant.{revision}"
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Cannot import migration {revision}: {e}") from e

    # Get upgrade function
    upgrade_fn: Callable = getattr(module, "upgrade", None)
    if upgrade_fn is None:
        raise ValueError(f"Migration {revision} has no upgrade() function")

    # Run migration in a transaction
    async with engine.begin() as conn:
        # Run upgrade in sync context (alembic style)
        await conn.run_sync(_run_upgrade_sync, upgrade_fn)

        # Update version
        await conn.execute(text("DELETE FROM alembic_version"))
        await conn.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:version)"),
            {"version": revision},
        )


def _run_upgrade_sync(connection, upgrade_fn: Callable) -> None:
    """Run upgrade function in sync context with alembic operations.

    This is needed because alembic operations are sync and use
    thread-local context.
    """
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    # Create migration context
    context = MigrationContext.configure(connection)

    # Create operations bound to this context
    with context.begin_transaction():
        with Operations.context(context):
            upgrade_fn()


async def check_migrations_pending(db_url: str) -> bool:
    """Check if there are pending migrations for a tenant database.

    Args:
        db_url: Database connection URL.

    Returns:
        True if there are pending migrations.
    """
    engine = create_async_engine(db_url, echo=False)

    try:
        await _ensure_version_table(engine)
        current_version = await _get_current_version(engine)
        pending = _get_pending_migrations(current_version)
        return len(pending) > 0
    finally:
        await engine.dispose()


async def get_migration_status(db_url: str) -> dict:
    """Get detailed migration status for a tenant database.

    Args:
        db_url: Database connection URL.

    Returns:
        Dict with current version, pending migrations, and all migrations.
    """
    engine = create_async_engine(db_url, echo=False)

    try:
        await _ensure_version_table(engine)
        current_version = await _get_current_version(engine)
        pending = _get_pending_migrations(current_version)

        return {
            "current_version": current_version,
            "latest_version": TENANT_MIGRATIONS[-1] if TENANT_MIGRATIONS else None,
            "pending_count": len(pending),
            "pending_migrations": pending,
            "all_migrations": TENANT_MIGRATIONS,
            "is_up_to_date": len(pending) == 0,
        }
    finally:
        await engine.dispose()

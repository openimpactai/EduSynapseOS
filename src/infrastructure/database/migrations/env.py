# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Alembic migration environment configuration.

This module configures Alembic for running migrations against both
central and tenant databases. The database type is determined by
the MIGRATION_TARGET environment variable.

Usage:
    # Central database migrations
    MIGRATION_TARGET=central alembic upgrade head

    # Tenant database migrations (requires TENANT_DB_URL)
    MIGRATION_TARGET=tenant TENANT_DB_URL=... alembic upgrade head
"""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from src.infrastructure.database.models.base import Base

# Determine migration target
MIGRATION_TARGET = os.environ.get("MIGRATION_TARGET", "central")

if MIGRATION_TARGET == "central":
    from src.infrastructure.database.models.central import (
        APIKeyAuditLog,
        License,
        SystemAuditLog,
        SystemSession,
        SystemUser,
        Tenant,
        TenantAPICredential,
        TenantFeatureFlag,
    )

    # Central database models
    target_metadata = Base.metadata

    # Filter to only include central tables
    CENTRAL_TABLES = {
        "licenses",
        "tenants",
        "system_users",
        "system_sessions",
        "system_audit_logs",
        "tenant_feature_flags",
        "tenant_api_credentials",
        "api_key_audit_logs",
    }

    def include_object(object, name, type_, reflected, compare_to):
        """Filter to include only central database tables."""
        if type_ == "table":
            return name in CENTRAL_TABLES
        return True

elif MIGRATION_TARGET == "tenant":
    from src.infrastructure.database.models.tenant import (
        # User & Auth
        User,
        Role,
        Permission,
        RolePermission,
        UserRole,
        UserSession,
        RefreshToken,
        PasswordResetToken,
        EmailVerification,
        # Organization
        School,
        AcademicYear,
        Class,
        ClassStudent,
        ClassTeacher,
        ParentStudentRelation,
        # Curriculum (synced from Central Curriculum)
        CurriculumFramework,
        CurriculumStage,
        GradeLevel,
        Subject,
        Unit,
        Topic,
        LearningObjective,
        Prerequisite,
        # Practice
        PracticeSession,
        PracticeQuestion,
        StudentAnswer,
        EvaluationResult,
        # Conversation
        Conversation,
        ConversationMessage,
        ConversationSummary,
        # Memory
        EpisodicMemory,
        SemanticMemory,
        ProceduralMemory,
        AssociativeMemory,
        # Spaced Repetition
        ReviewItem,
        ReviewLog,
        # Diagnostics
        DiagnosticScan,
        DiagnosticIndicator,
        DiagnosticRecommendation,
        # Emotional Intelligence
        EmotionalSignal,
        CompanionSession,
        StudentNote,
        # Notifications
        Alert,
        Notification,
        NotificationPreference,
        NotificationTemplate,
        # Analytics
        AnalyticsEvent,
        DailySummary,
        MasterySnapshot,
        EngagementMetric,
        # Settings
        TenantSetting,
        UserPreference,
        FeatureFlag,
        # Localization
        Language,
        Translation,
        # Audit
        AuditLog,
    )

    target_metadata = Base.metadata

    # Central tables to exclude from tenant migrations
    CENTRAL_TABLES = {
        "licenses",
        "tenants",
        "system_users",
        "system_sessions",
        "system_audit_logs",
        "tenant_feature_flags",
        "tenant_api_credentials",
        "api_key_audit_logs",
    }

    def include_object(object, name, type_, reflected, compare_to):
        """Filter to exclude central database tables."""
        if type_ == "table":
            return name not in CENTRAL_TABLES
        return True

else:
    raise ValueError(f"Invalid MIGRATION_TARGET: {MIGRATION_TARGET}. Use 'central' or 'tenant'.")


# Alembic Config object
config = context.config

# Configure logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def get_database_url() -> str:
    """Get database URL from environment.

    Returns:
        Database connection URL.

    Raises:
        ValueError: If required environment variable is not set.
    """
    if MIGRATION_TARGET == "central":
        url = os.environ.get("CENTRAL_DB_URL")
        if not url:
            # Construct from individual components
            host = os.environ.get("CENTRAL_DB_HOST", "localhost")
            port = os.environ.get("CENTRAL_DB_PORT", "5432")
            user = os.environ.get("CENTRAL_DB_USER", "edusynapse")
            password = os.environ.get("CENTRAL_DB_PASSWORD", "")
            database = os.environ.get("CENTRAL_DB_NAME", "edusynapse_central")
            url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
        return url
    else:
        url = os.environ.get("TENANT_DB_URL")
        if not url:
            raise ValueError("TENANT_DB_URL environment variable is required for tenant migrations")
        return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the
    Engine creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with the given connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_database_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

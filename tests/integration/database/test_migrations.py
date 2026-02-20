# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for database migrations.

Tests migration execution against real databases.
Requires PostgreSQL to be running.
"""

import os
from unittest.mock import patch

import pytest

# Skip all tests if database is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL not set",
)


class TestCentralMigrations:
    """Test central database migrations."""

    @pytest.mark.asyncio
    async def test_central_migration_creates_tables(self, central_db_session):
        """Verify central migration creates all required tables."""
        from sqlalchemy import inspect

        inspector = inspect(central_db_session.get_bind())
        tables = inspector.get_table_names()

        expected_tables = [
            "licenses",
            "tenants",
            "system_users",
            "system_sessions",
            "system_audit_logs",
            "tenant_feature_flags",
        ]

        for table in expected_tables:
            assert table in tables, f"Table {table} not found"

    @pytest.mark.asyncio
    async def test_licenses_table_has_correct_columns(self, central_db_session):
        """Verify licenses table has all required columns."""
        from sqlalchemy import inspect

        inspector = inspect(central_db_session.get_bind())
        columns = {col["name"] for col in inspector.get_columns("licenses")}

        expected_columns = {
            "id",
            "license_key",
            "license_type",
            "max_students",
            "max_teachers",
            "features",
            "valid_from",
            "valid_until",
            "is_active",
            "created_at",
            "updated_at",
        }

        for col in expected_columns:
            assert col in columns, f"Column {col} not found in licenses table"

    @pytest.mark.asyncio
    async def test_tenants_table_has_correct_columns(self, central_db_session):
        """Verify tenants table has all required columns."""
        from sqlalchemy import inspect

        inspector = inspect(central_db_session.get_bind())
        columns = {col["name"] for col in inspector.get_columns("tenants")}

        expected_columns = {
            "id",
            "name",
            "slug",
            "domain",
            "license_id",
            "db_host",
            "db_port",
            "db_name",
            "db_user",
            "db_password_encrypted",
            "container_id",
            "container_status",
            "status",
            "settings",
            "metadata_",
            "created_at",
            "updated_at",
            "deleted_at",
        }

        for col in expected_columns:
            assert col in columns, f"Column {col} not found in tenants table"


class TestTenantMigrations:
    """Test tenant database migrations."""

    @pytest.mark.asyncio
    async def test_tenant_migration_creates_user_tables(self, tenant_db_session):
        """Verify tenant migration creates user management tables."""
        from sqlalchemy import inspect

        inspector = inspect(tenant_db_session.get_bind())
        tables = inspector.get_table_names()

        expected_tables = [
            "users",
            "roles",
            "permissions",
            "role_permissions",
            "user_roles",
        ]

        for table in expected_tables:
            assert table in tables, f"Table {table} not found"

    @pytest.mark.asyncio
    async def test_tenant_migration_creates_curriculum_tables(self, tenant_db_session):
        """Verify tenant migration creates curriculum tables."""
        from sqlalchemy import inspect

        inspector = inspect(tenant_db_session.get_bind())
        tables = inspector.get_table_names()

        expected_tables = [
            "curricula",
            "grade_levels",
            "subjects",
            "units",
            "topics",
            "learning_objectives",
            "knowledge_components",
            "prerequisites",
        ]

        for table in expected_tables:
            assert table in tables, f"Table {table} not found"

    @pytest.mark.asyncio
    async def test_tenant_migration_creates_practice_tables(self, tenant_db_session):
        """Verify tenant migration creates practice and assessment tables."""
        from sqlalchemy import inspect

        inspector = inspect(tenant_db_session.get_bind())
        tables = inspector.get_table_names()

        expected_tables = [
            "practice_sessions",
            "practice_questions",
            "student_answers",
            "evaluation_results",
            "assessment_sessions",
            "assessment_results",
        ]

        for table in expected_tables:
            assert table in tables, f"Table {table} not found"

    @pytest.mark.asyncio
    async def test_tenant_migration_creates_memory_tables(self, tenant_db_session):
        """Verify tenant migration creates memory system tables."""
        from sqlalchemy import inspect

        inspector = inspect(tenant_db_session.get_bind())
        tables = inspector.get_table_names()

        expected_tables = [
            "episodic_memories",
            "semantic_memories",
            "procedural_memories",
            "associative_memories",
        ]

        for table in expected_tables:
            assert table in tables, f"Table {table} not found"

    @pytest.mark.asyncio
    async def test_tenant_migration_creates_review_tables(self, tenant_db_session):
        """Verify tenant migration creates spaced repetition tables."""
        from sqlalchemy import inspect

        inspector = inspect(tenant_db_session.get_bind())
        tables = inspector.get_table_names()

        expected_tables = [
            "review_items",
            "review_logs",
        ]

        for table in expected_tables:
            assert table in tables, f"Table {table} not found"

    @pytest.mark.asyncio
    async def test_users_table_has_correct_columns(self, tenant_db_session):
        """Verify users table has all required columns."""
        from sqlalchemy import inspect

        inspector = inspect(tenant_db_session.get_bind())
        columns = {col["name"] for col in inspector.get_columns("users")}

        expected_columns = {
            "id",
            "email",
            "password_hash",
            "first_name",
            "last_name",
            "user_type",
            "avatar_url",
            "phone",
            "birth_date",
            "gender",
            "grade_level",
            "school_id",
            "is_active",
            "is_verified",
            "last_login_at",
            "language",
            "timezone",
            "metadata_",
            "created_at",
            "updated_at",
            "deleted_at",
        }

        for col in expected_columns:
            assert col in columns, f"Column {col} not found in users table"

    @pytest.mark.asyncio
    async def test_review_items_table_has_fsrs_columns(self, tenant_db_session):
        """Verify review_items table has FSRS-5 parameters."""
        from sqlalchemy import inspect

        inspector = inspect(tenant_db_session.get_bind())
        columns = {col["name"] for col in inspector.get_columns("review_items")}

        fsrs_columns = {
            "stability",
            "difficulty",
            "elapsed_days",
            "scheduled_days",
            "reps",
            "lapses",
            "state",
            "last_review",
            "due",
        }

        for col in fsrs_columns:
            assert col in columns, f"FSRS column {col} not found in review_items table"


class TestMigrationRollback:
    """Test migration rollback functionality."""

    @pytest.mark.asyncio
    async def test_central_migration_can_rollback(self, central_db_engine):
        """Verify central migration can be rolled back."""
        from alembic import command
        from alembic.config import Config

        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("script_location", "src/infrastructure/database/migrations")

        with patch.dict(os.environ, {"MIGRATION_TARGET": "central"}):
            command.downgrade(alembic_cfg, "base")
            command.upgrade(alembic_cfg, "head")

    @pytest.mark.asyncio
    async def test_tenant_migration_can_rollback(self, tenant_db_engine):
        """Verify tenant migration can be rolled back."""
        from alembic import command
        from alembic.config import Config

        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("script_location", "src/infrastructure/database/migrations")

        with patch.dict(os.environ, {"MIGRATION_TARGET": "tenant"}):
            command.downgrade(alembic_cfg, "base")
            command.upgrade(alembic_cfg, "head")

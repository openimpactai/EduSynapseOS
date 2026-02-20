# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for database seed scripts.

Tests seed data insertion against real databases.
Requires PostgreSQL to be running.
"""

import os

import pytest

# Skip all tests if database is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL not set",
)


class TestCentralSeeds:
    """Test central database seeds."""

    @pytest.mark.asyncio
    async def test_seed_licenses_creates_default_licenses(self, central_db_session):
        """Verify seed_licenses creates default license types."""
        from src.infrastructure.database.models.central import License
        from src.infrastructure.database.seeds.central import seed_licenses

        licenses = await seed_licenses(central_db_session)
        await central_db_session.commit()

        assert len(licenses) == 4

        license_types = {l.license_type for l in licenses}
        assert license_types == {"trial", "basic", "premium", "enterprise"}

        for license_obj in licenses:
            assert license_obj.id is not None
            assert license_obj.license_key is not None
            assert license_obj.is_active is True

    @pytest.mark.asyncio
    async def test_seed_system_users_creates_admin(self, central_db_session):
        """Verify seed_system_users creates initial admin user."""
        from src.infrastructure.database.seeds.central import seed_system_users

        users = await seed_system_users(
            central_db_session,
            admin_email="test@example.com",
            admin_password="TestPassword123!",
        )
        await central_db_session.commit()

        assert len(users) == 1

        admin = users[0]
        assert admin.email == "test@example.com"
        assert admin.role == "super_admin"
        assert admin.is_active is True
        assert admin.password_hash is not None
        assert admin.password_hash != "TestPassword123!"

    @pytest.mark.asyncio
    async def test_seed_central_database_full(self, central_db_session):
        """Verify full central database seeding."""
        from src.infrastructure.database.seeds.central import seed_central_database

        result = await seed_central_database(
            central_db_session,
            admin_email="full_test@example.com",
            admin_password="FullTest123!",
        )

        assert "licenses" in result
        assert "system_users" in result
        assert len(result["licenses"]) == 4
        assert len(result["system_users"]) == 1


class TestTenantSeeds:
    """Test tenant database seeds."""

    @pytest.mark.asyncio
    async def test_seed_permissions_creates_all_permissions(self, tenant_db_session):
        """Verify seed_permissions creates all default permissions."""
        from src.infrastructure.database.seeds.tenant import seed_permissions

        permissions = await seed_permissions(tenant_db_session)
        await tenant_db_session.commit()

        assert len(permissions) >= 25

        categories = {p.category for p in permissions}
        expected_categories = {"users", "students", "classes", "curriculum", "practice", "assessments", "analytics", "settings", "ai"}

        for cat in expected_categories:
            assert cat in categories, f"Category {cat} not found in permissions"

    @pytest.mark.asyncio
    async def test_seed_roles_creates_system_roles(self, tenant_db_session):
        """Verify seed_roles creates system roles with permissions."""
        from src.infrastructure.database.seeds.tenant import seed_permissions, seed_roles

        permissions = await seed_permissions(tenant_db_session)
        roles = await seed_roles(tenant_db_session, permissions)
        await tenant_db_session.commit()

        assert len(roles) == 5

        role_names = {r.name for r in roles}
        expected_roles = {"admin", "school_admin", "teacher", "student", "parent"}
        assert role_names == expected_roles

        for role in roles:
            assert role.is_system is True

    @pytest.mark.asyncio
    async def test_seed_languages_creates_default_languages(self, tenant_db_session):
        """Verify seed_languages creates supported languages."""
        from src.infrastructure.database.seeds.tenant import seed_languages

        languages = await seed_languages(tenant_db_session)
        await tenant_db_session.commit()

        assert len(languages) >= 2

        lang_codes = {l.code for l in languages}
        assert "tr" in lang_codes
        assert "en" in lang_codes

        turkish = next(l for l in languages if l.code == "tr")
        assert turkish.is_default is True
        assert turkish.is_active is True

    @pytest.mark.asyncio
    async def test_seed_tenant_settings_creates_defaults(self, tenant_db_session):
        """Verify seed_tenant_settings creates default settings."""
        from src.infrastructure.database.seeds.tenant import seed_tenant_settings

        settings = await seed_tenant_settings(tenant_db_session)
        await tenant_db_session.commit()

        assert len(settings) >= 5

        setting_keys = {s.setting_key for s in settings}
        expected_keys = {
            "ai.default_persona",
            "practice.default_question_count",
            "spaced_repetition.algorithm",
        }

        for key in expected_keys:
            assert key in setting_keys, f"Setting {key} not found"

    @pytest.mark.asyncio
    async def test_seed_feature_flags_creates_flags(self, tenant_db_session):
        """Verify seed_feature_flags creates default flags."""
        from src.infrastructure.database.seeds.tenant import seed_feature_flags

        flags = await seed_feature_flags(tenant_db_session)
        await tenant_db_session.commit()

        assert len(flags) >= 4

        flag_keys = {f.feature_key for f in flags}
        expected_flags = {"ai_tutoring_v2", "diagnostic_engine", "proactive_alerts"}

        for key in expected_flags:
            assert key in flag_keys, f"Feature flag {key} not found"

    @pytest.mark.asyncio
    async def test_seed_sample_curriculum_creates_grade5_math(self, tenant_db_session):
        """Verify seed_sample_curriculum creates Grade 5 Math curriculum."""
        from src.infrastructure.database.seeds.tenant import seed_sample_curriculum

        result = await seed_sample_curriculum(tenant_db_session)
        await tenant_db_session.commit()

        assert "curriculum" in result
        assert "grade_levels" in result
        assert "subjects" in result
        assert "units" in result
        assert "topics" in result
        assert "objectives" in result

        curriculum = result["curriculum"]
        assert curriculum.code == "MEB-2024"
        assert curriculum.country == "TR"

        grade = result["grade_levels"][0]
        assert grade.grade_number == 5

        math = result["subjects"][0]
        assert math.code == "MAT"

        assert len(result["units"]) >= 4
        assert len(result["topics"]) >= 5
        assert len(result["objectives"]) >= 3

    @pytest.mark.asyncio
    async def test_seed_sample_users_creates_test_users(self, tenant_db_session):
        """Verify seed_sample_users creates test users with roles."""
        from src.infrastructure.database.seeds.tenant import (
            seed_permissions,
            seed_roles,
            seed_sample_users,
        )

        permissions = await seed_permissions(tenant_db_session)
        roles = await seed_roles(tenant_db_session, permissions)
        users = await seed_sample_users(
            tenant_db_session,
            roles,
            admin_email="tenant_admin@test.com",
            admin_password="TenantAdmin123!",
        )
        await tenant_db_session.commit()

        assert len(users) >= 3

        user_types = {u.user_type for u in users}
        assert "admin" in user_types
        assert "teacher" in user_types
        assert "student" in user_types

        admin = next(u for u in users if u.user_type == "admin")
        assert admin.email == "tenant_admin@test.com"
        assert admin.is_active is True
        assert admin.is_verified is True

    @pytest.mark.asyncio
    async def test_seed_tenant_database_full(self, tenant_db_session):
        """Verify full tenant database seeding."""
        from src.infrastructure.database.seeds.tenant import seed_tenant_database

        result = await seed_tenant_database(
            tenant_db_session,
            include_sample_data=True,
            admin_email="full_tenant_test@example.com",
            admin_password="FullTenantTest123!",
        )

        assert "permissions" in result
        assert "roles" in result
        assert "languages" in result
        assert "settings" in result
        assert "feature_flags" in result
        assert "curriculum" in result
        assert "users" in result

        assert len(result["permissions"]) >= 25
        assert len(result["roles"]) == 5
        assert len(result["languages"]) >= 2
        assert len(result["users"]) >= 3

    @pytest.mark.asyncio
    async def test_seed_tenant_database_without_sample_data(self, tenant_db_session):
        """Verify tenant database seeding without sample data."""
        from src.infrastructure.database.seeds.tenant import seed_tenant_database

        result = await seed_tenant_database(
            tenant_db_session,
            include_sample_data=False,
        )

        assert "permissions" in result
        assert "roles" in result
        assert "languages" in result
        assert "settings" in result
        assert "feature_flags" in result
        assert "curriculum" not in result
        assert "users" not in result


class TestSeedIdempotency:
    """Test that seeds handle re-runs gracefully."""

    @pytest.mark.asyncio
    async def test_running_seeds_twice_fails_on_unique_constraints(self, tenant_db_session):
        """Verify that running seeds twice raises unique constraint errors."""
        from sqlalchemy.exc import IntegrityError

        from src.infrastructure.database.seeds.tenant import seed_permissions

        await seed_permissions(tenant_db_session)
        await tenant_db_session.commit()

        with pytest.raises(IntegrityError):
            await seed_permissions(tenant_db_session)
            await tenant_db_session.commit()

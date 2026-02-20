# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tenant database seed data.

This module provides seed data for tenant databases:
- Roles and permissions
- Default languages
- Default tenant settings
- Turkish K-12 curriculum (optional)
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.models.tenant import (
    CompanionActivity,
    FeatureFlag,
    Language,
    Permission,
    Role,
    RolePermission,
    TenantSetting,
    User,
    UserRole,
)

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def seed_permissions(session: AsyncSession) -> list[Permission]:
    """Seed default permissions.

    Args:
        session: Database session.

    Returns:
        List of created permissions.
    """
    permissions_data = [
        # User management
        {"code": "users.view", "name": "View Users", "category": "users", "description": "View user list and details"},
        {"code": "users.create", "name": "Create Users", "category": "users", "description": "Create new users"},
        {"code": "users.update", "name": "Update Users", "category": "users", "description": "Update user information"},
        {"code": "users.delete", "name": "Delete Users", "category": "users", "description": "Delete users"},
        # Student management
        {"code": "students.view", "name": "View Students", "category": "students", "description": "View student profiles"},
        {"code": "students.progress", "name": "View Student Progress", "category": "students", "description": "View student learning progress"},
        {"code": "students.assign", "name": "Assign Students", "category": "students", "description": "Assign students to classes"},
        # Class management
        {"code": "classes.view", "name": "View Classes", "category": "classes", "description": "View class list"},
        {"code": "classes.create", "name": "Create Classes", "category": "classes", "description": "Create new classes"},
        {"code": "classes.update", "name": "Update Classes", "category": "classes", "description": "Update class details"},
        {"code": "classes.delete", "name": "Delete Classes", "category": "classes", "description": "Delete classes"},
        # Curriculum management
        {"code": "curriculum.view", "name": "View Curriculum", "category": "curriculum", "description": "View curriculum content"},
        {"code": "curriculum.create", "name": "Create Curriculum", "category": "curriculum", "description": "Create curriculum items"},
        {"code": "curriculum.update", "name": "Update Curriculum", "category": "curriculum", "description": "Update curriculum content"},
        {"code": "curriculum.delete", "name": "Delete Curriculum", "category": "curriculum", "description": "Delete curriculum items"},
        # Practice sessions
        {"code": "practice.start", "name": "Start Practice", "category": "practice", "description": "Start practice sessions"},
        {"code": "practice.view_all", "name": "View All Practice", "category": "practice", "description": "View all students' practice sessions"},
        # Assessments
        {"code": "assessments.view", "name": "View Assessments", "category": "assessments", "description": "View assessments"},
        {"code": "assessments.create", "name": "Create Assessments", "category": "assessments", "description": "Create assessments"},
        {"code": "assessments.grade", "name": "Grade Assessments", "category": "assessments", "description": "Grade student assessments"},
        # Analytics
        {"code": "analytics.view_own", "name": "View Own Analytics", "category": "analytics", "description": "View own analytics"},
        {"code": "analytics.view_class", "name": "View Class Analytics", "category": "analytics", "description": "View class-level analytics"},
        {"code": "analytics.view_school", "name": "View School Analytics", "category": "analytics", "description": "View school-level analytics"},
        {"code": "analytics.view_all", "name": "View All Analytics", "category": "analytics", "description": "View tenant-wide analytics"},
        {"code": "analytics.export", "name": "Export Analytics", "category": "analytics", "description": "Export analytics data"},
        # Settings
        {"code": "settings.view", "name": "View Settings", "category": "settings", "description": "View tenant settings"},
        {"code": "settings.update", "name": "Update Settings", "category": "settings", "description": "Update tenant settings"},
        # AI features
        {"code": "ai.learning", "name": "Use AI Learning", "category": "ai", "description": "Use AI tutoring and learning"},
        {"code": "ai.configure", "name": "Configure AI", "category": "ai", "description": "Configure AI settings"},
    ]

    permissions = []
    for data in permissions_data:
        permission = Permission(**data)
        session.add(permission)
        permissions.append(permission)

    await session.flush()
    logger.info(f"Seeded {len(permissions)} permissions")
    return permissions


async def seed_roles(session: AsyncSession, permissions: list[Permission]) -> list[Role]:
    """Seed default roles with permissions.

    Args:
        session: Database session.
        permissions: List of available permissions.

    Returns:
        List of created roles.
    """
    permission_map = {p.code: p for p in permissions}

    roles_config = [
        {
            "name": "admin",
            "display_name": "Administrator",
            "description": "Full access to all features",
            "is_system": True,
            "permissions": list(permission_map.keys()),
        },
        {
            "name": "school_admin",
            "display_name": "School Administrator",
            "description": "Manage school, teachers, and students",
            "is_system": True,
            "permissions": [
                "users.view", "users.create", "users.update",
                "students.view", "students.progress", "students.assign",
                "classes.view", "classes.create", "classes.update", "classes.delete",
                "curriculum.view",
                "practice.view_all",
                "assessments.view", "assessments.create", "assessments.grade",
                "analytics.view_class", "analytics.view_school", "analytics.export",
            ],
        },
        {
            "name": "teacher",
            "display_name": "Teacher",
            "description": "Manage classes and view student progress",
            "is_system": True,
            "permissions": [
                "students.view", "students.progress",
                "classes.view",
                "curriculum.view",
                "practice.view_all",
                "assessments.view", "assessments.create", "assessments.grade",
                "analytics.view_class",
                "ai.learning",
            ],
        },
        {
            "name": "student",
            "display_name": "Student",
            "description": "Access learning features",
            "is_system": True,
            "permissions": [
                "curriculum.view",
                "practice.start",
                "analytics.view_own",
                "ai.learning",
            ],
        },
        {
            "name": "parent",
            "display_name": "Parent",
            "description": "View child's progress",
            "is_system": True,
            "permissions": [
                "students.view", "students.progress",
                "analytics.view_own",
            ],
        },
    ]

    roles = []
    for config in roles_config:
        perm_codes = config.pop("permissions")
        role = Role(**config)
        session.add(role)
        await session.flush()

        for code in perm_codes:
            if code in permission_map:
                role_perm = RolePermission(
                    role_id=role.id,
                    permission_id=permission_map[code].id,
                )
                session.add(role_perm)

        roles.append(role)

    await session.flush()
    logger.info(f"Seeded {len(roles)} roles")
    return roles


async def seed_languages(session: AsyncSession) -> list[Language]:
    """Seed supported languages.

    Args:
        session: Database session.

    Returns:
        List of created languages.
    """
    languages_data = [
        {
            "code": "tr",
            "name": "Turkish",
            "native_name": "Türkçe",
            "is_rtl": False,
            "is_active": True,
            "is_default": True,
            "translation_progress": 100,
        },
        {
            "code": "en",
            "name": "English",
            "native_name": "English",
            "is_rtl": False,
            "is_active": True,
            "is_default": False,
            "translation_progress": 100,
        },
        {
            "code": "ar",
            "name": "Arabic",
            "native_name": "العربية",
            "is_rtl": True,
            "is_active": False,
            "is_default": False,
            "translation_progress": 0,
        },
    ]

    languages = []
    for data in languages_data:
        lang = Language(**data)
        session.add(lang)
        languages.append(lang)

    await session.flush()
    logger.info(f"Seeded {len(languages)} languages")
    return languages


async def seed_tenant_settings(session: AsyncSession) -> list[TenantSetting]:
    """Seed default tenant settings.

    Args:
        session: Database session.

    Returns:
        List of created settings.
    """
    settings_data = [
        {
            "setting_key": "ai.default_persona",
            "setting_value": {"value": "tutor"},
            "description": "Default AI persona for conversations",
            "value_type": "string",
            "allow_user_override": True,
        },
        {
            "setting_key": "ai.max_conversation_length",
            "setting_value": {"value": 100},
            "description": "Maximum messages per conversation before summarization",
            "value_type": "number",
        },
        {
            "setting_key": "practice.default_question_count",
            "setting_value": {"value": 10},
            "description": "Default number of questions per practice session",
            "value_type": "number",
            "allow_user_override": True,
        },
        {
            "setting_key": "practice.hint_penalty",
            "setting_value": {"value": 0.25},
            "description": "Score penalty per hint used (0.0-1.0)",
            "value_type": "number",
        },
        {
            "setting_key": "spaced_repetition.algorithm",
            "setting_value": {"value": "fsrs-5"},
            "description": "Spaced repetition algorithm to use",
            "value_type": "string",
        },
        {
            "setting_key": "spaced_repetition.daily_review_limit",
            "setting_value": {"value": 50},
            "description": "Maximum review items per day",
            "value_type": "number",
            "allow_user_override": True,
        },
        {
            "setting_key": "notifications.email_enabled",
            "setting_value": {"value": True},
            "description": "Enable email notifications",
            "value_type": "boolean",
            "allow_user_override": True,
        },
        {
            "setting_key": "analytics.retention_days",
            "setting_value": {"value": 365},
            "description": "Days to retain detailed analytics data",
            "value_type": "number",
        },
    ]

    settings = []
    for data in settings_data:
        setting = TenantSetting(**data)
        session.add(setting)
        settings.append(setting)

    await session.flush()
    logger.info(f"Seeded {len(settings)} tenant settings")
    return settings


async def seed_feature_flags(session: AsyncSession) -> list[FeatureFlag]:
    """Seed default feature flags.

    Args:
        session: Database session.

    Returns:
        List of created feature flags.
    """
    flags_data = [
        {
            "feature_key": "ai_tutoring_v2",
            "is_enabled": True,
            "rollout_percentage": 100,
            "description": "New AI tutoring system with improved context awareness",
        },
        {
            "feature_key": "diagnostic_engine",
            "is_enabled": True,
            "rollout_percentage": 100,
            "description": "Learning difficulty diagnostic system",
        },
        {
            "feature_key": "proactive_alerts",
            "is_enabled": True,
            "rollout_percentage": 50,
            "description": "Proactive learning struggle alerts",
        },
        {
            "feature_key": "gamification",
            "is_enabled": False,
            "rollout_percentage": 0,
            "description": "Gamification features (badges, leaderboards)",
        },
        {
            "feature_key": "parent_dashboard",
            "is_enabled": True,
            "rollout_percentage": 100,
            "description": "Parent dashboard and notifications",
        },
    ]

    flags = []
    for data in flags_data:
        flag = FeatureFlag(**data)
        session.add(flag)
        flags.append(flag)

    await session.flush()
    logger.info(f"Seeded {len(flags)} feature flags")
    return flags


async def seed_companion_activities(session: AsyncSession) -> list[CompanionActivity]:
    """Seed default companion activities.

    These are the activities that the companion can suggest to students.
    Activities are categorized and filtered by grade level and difficulty.

    Args:
        session: Database session.

    Returns:
        List of created activities.
    """
    activities_data = [
        # Learning activities
        {
            "code": "practice",
            "name": "Practice Session",
            "description": "Practice what you've learned with personalized questions",
            "icon": "📝",
            "category": "learning",
            "route": "/practice",
            "min_grade": 1,
            "max_grade": 12,
            "difficulty": "medium",
            "display_order": 1,
        },
        {
            "code": "review",
            "name": "Review Session",
            "description": "Review topics you've studied before to strengthen your memory",
            "icon": "🔄",
            "category": "learning",
            "route": "/review",
            "min_grade": 1,
            "max_grade": 12,
            "difficulty": "medium",
            "display_order": 2,
        },
        # Fun activities
        {
            "code": "game_math",
            "name": "Math Games",
            "description": "Fun math games to practice your skills",
            "icon": "🎮",
            "category": "fun",
            "route": "/games/math",
            "min_grade": 1,
            "max_grade": 8,
            "difficulty": "easy",
            "display_order": 1,
        },
        {
            "code": "game_word",
            "name": "Word Games",
            "description": "Word puzzles and vocabulary games",
            "icon": "🔤",
            "category": "fun",
            "route": "/games/word",
            "min_grade": 1,
            "max_grade": 12,
            "difficulty": "easy",
            "display_order": 2,
        },
        # Creative activities
        {
            "code": "drawing",
            "name": "Drawing Activity",
            "description": "Express yourself through drawing and art",
            "icon": "🎨",
            "category": "creative",
            "route": "/creative/drawing",
            "min_grade": 1,
            "max_grade": 6,
            "difficulty": "easy",
            "display_order": 1,
        },
        {
            "code": "story",
            "name": "Story Time",
            "description": "Read or create your own stories",
            "icon": "📚",
            "category": "creative",
            "route": "/creative/story",
            "min_grade": 1,
            "max_grade": 8,
            "difficulty": "easy",
            "display_order": 2,
        },
        # Break activities
        {
            "code": "break",
            "name": "Take a Break",
            "description": "Take a short break to rest and recharge",
            "icon": "☕",
            "category": "break",
            "route": "/break",
            "min_grade": 1,
            "max_grade": 12,
            "difficulty": "easy",
            "display_order": 1,
        },
        {
            "code": "breathing",
            "name": "Breathing Exercise",
            "description": "Calm breathing exercises to help you relax",
            "icon": "🧘",
            "category": "break",
            "route": "/break/breathing",
            "min_grade": 1,
            "max_grade": 12,
            "difficulty": "easy",
            "display_order": 2,
        },
    ]

    activities = []
    for data in activities_data:
        activity = CompanionActivity(**data)
        session.add(activity)
        activities.append(activity)

    await session.flush()
    logger.info(f"Seeded {len(activities)} companion activities")
    return activities


async def seed_sample_users(
    session: AsyncSession,
    roles: list[Role],
    admin_email: str = "admin@tenant.local",
    admin_password: str = "TenantAdmin2024",
) -> list[User]:
    """Seed sample users for testing.

    Args:
        session: Database session.
        roles: Available roles.
        admin_email: Admin email.
        admin_password: Admin password.

    Returns:
        List of created users.
    """
    role_map = {r.name: r for r in roles}
    password_hash = pwd_context.hash(admin_password)

    users_data = [
        {
            "email": admin_email,
            "password_hash": password_hash,
            "first_name": "Tenant",
            "last_name": "Admin",
            "user_type": "admin",
            "is_active": True,
            "is_verified": True,
            "role_name": "admin",
        },
        {
            "email": "teacher@tenant.local",
            "password_hash": password_hash,
            "first_name": "Ayşe",
            "last_name": "Öğretmen",
            "user_type": "teacher",
            "is_active": True,
            "is_verified": True,
            "role_name": "teacher",
        },
        {
            "email": "student@tenant.local",
            "password_hash": password_hash,
            "first_name": "Mehmet",
            "last_name": "Öğrenci",
            "user_type": "student",
            "is_active": True,
            "is_verified": True,
            "grade_level": 5,
            "role_name": "student",
        },
    ]

    users = []
    for data in users_data:
        role_name = data.pop("role_name")
        user = User(**data)
        session.add(user)
        await session.flush()

        if role_name in role_map:
            user_role = UserRole(
                user_id=user.id,
                role_id=role_map[role_name].id,
            )
            session.add(user_role)

        users.append(user)

    await session.flush()
    logger.info(f"Seeded {len(users)} sample users")
    return users


async def seed_tenant_database(
    session: AsyncSession,
    include_sample_users: bool = True,
    admin_email: Optional[str] = None,
    admin_password: Optional[str] = None,
) -> dict:
    """Seed a tenant database with initial data.

    Note: Curriculum data is NOT seeded here. Curriculum comes from
    Central Curriculum and should be synced using the curriculum sync
    background task or API endpoint.

    Args:
        session: Database session.
        include_sample_users: Whether to include sample test users.
        admin_email: Optional admin email override.
        admin_password: Optional admin password override.

    Returns:
        Dictionary with seeded entities.
    """
    logger.info("Seeding tenant database...")

    permissions = await seed_permissions(session)
    roles = await seed_roles(session, permissions)
    languages = await seed_languages(session)
    settings = await seed_tenant_settings(session)
    feature_flags = await seed_feature_flags(session)
    companion_activities = await seed_companion_activities(session)

    result = {
        "permissions": permissions,
        "roles": roles,
        "languages": languages,
        "settings": settings,
        "feature_flags": feature_flags,
        "companion_activities": companion_activities,
    }

    if include_sample_users:
        user_kwargs = {}
        if admin_email:
            user_kwargs["admin_email"] = admin_email
        if admin_password:
            user_kwargs["admin_password"] = admin_password

        users = await seed_sample_users(session, roles, **user_kwargs)
        result["users"] = users

    await session.commit()

    logger.info("Tenant database seeding complete")

    return result


if __name__ == "__main__":
    import os

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    async def main():
        database_url = os.environ.get(
            "TENANT_DB_URL",
            "postgresql+asyncpg://edusynapse:edusynapse@localhost:5432/edusynapse_tenant",
        )
        engine = create_async_engine(database_url)
        async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with async_session() as session:
            await seed_tenant_database(session)

    asyncio.run(main())

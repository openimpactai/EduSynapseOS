# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Central database seed data.

This module provides seed data for the central database:
- Licenses: Default license types
- System users: Initial super admin
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.models.central import License, SystemUser

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


async def seed_licenses(session: AsyncSession) -> list[License]:
    """Seed default license types.

    Args:
        session: Database session.

    Returns:
        List of created licenses.
    """
    licenses_data = [
        {
            "license_key": "TRIAL-DEFAULT-001",
            "max_students": 50,
            "max_teachers": 5,
            "max_schools": 1,
            "features": ["ai_tutoring", "practice_sessions", "assessments"],
            "valid_from": datetime.now(timezone.utc),
            "valid_until": datetime.now(timezone.utc) + timedelta(days=30),
            "status": "active",
            "notes": "Trial license - 30 days",
        },
        {
            "license_key": "BASIC-DEFAULT-001",
            "max_students": 200,
            "max_teachers": 20,
            "max_schools": 3,
            "features": ["ai_tutoring", "practice_sessions", "assessments", "analytics"],
            "valid_from": datetime.now(timezone.utc),
            "valid_until": datetime.now(timezone.utc) + timedelta(days=365),
            "status": "active",
            "notes": "Basic license - 1 year",
        },
        {
            "license_key": "PREMIUM-DEFAULT-001",
            "max_students": 1000,
            "max_teachers": 100,
            "max_schools": 10,
            "features": ["ai_tutoring", "practice_sessions", "assessments", "analytics", "custom_curriculum", "priority_support"],
            "valid_from": datetime.now(timezone.utc),
            "valid_until": datetime.now(timezone.utc) + timedelta(days=365),
            "status": "active",
            "notes": "Premium license - 1 year",
        },
        {
            "license_key": "ENTERPRISE-DEFAULT-001",
            "max_students": None,
            "max_teachers": None,
            "max_schools": None,
            "features": ["ai_tutoring", "practice_sessions", "assessments", "analytics", "custom_curriculum", "api_access", "priority_support", "dedicated_support", "custom_integrations"],
            "valid_from": datetime.now(timezone.utc),
            "valid_until": datetime.now(timezone.utc) + timedelta(days=365 * 3),
            "status": "active",
            "notes": "Enterprise license - unlimited, 3 years",
        },
    ]

    licenses = []
    for data in licenses_data:
        license_obj = License(**data)
        session.add(license_obj)
        licenses.append(license_obj)

    await session.flush()
    logger.info(f"Seeded {len(licenses)} licenses")
    return licenses


async def seed_system_users(
    session: AsyncSession,
    admin_email: str = "admin@gdlabs.io",
    admin_password: str = "AdMin*-12345*--*",
) -> list[SystemUser]:
    """Seed initial system users.

    Args:
        session: Database session.
        admin_email: Admin email address.
        admin_password: Admin password.

    Returns:
        List of created system users.
    """
    password_hash = hash_password(admin_password)

    users_data = [
        {
            "email": admin_email,
            "password_hash": password_hash,
            "name": "GDLabs Admin",
            "role": "super_admin",
            "is_active": True,
            "mfa_enabled": False,
        },
    ]

    users = []
    for data in users_data:
        user = SystemUser(**data)
        session.add(user)
        users.append(user)

    await session.flush()
    logger.info(f"Seeded {len(users)} system users")
    return users


async def seed_central_database(
    session: AsyncSession,
    admin_email: Optional[str] = None,
    admin_password: Optional[str] = None,
) -> dict:
    """Seed the central database with initial data.

    Args:
        session: Database session.
        admin_email: Optional admin email override.
        admin_password: Optional admin password override.

    Returns:
        Dictionary with seeded entities.
    """
    logger.info("Seeding central database...")

    licenses = await seed_licenses(session)

    user_kwargs = {}
    if admin_email:
        user_kwargs["admin_email"] = admin_email
    if admin_password:
        user_kwargs["admin_password"] = admin_password

    users = await seed_system_users(session, **user_kwargs)

    await session.commit()

    logger.info("Central database seeding complete")

    return {
        "licenses": licenses,
        "system_users": users,
    }


if __name__ == "__main__":
    import os

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    async def main():
        database_url = os.environ.get(
            "CENTRAL_DB_URL",
            "postgresql+asyncpg://edusynapse:edusynapse@localhost:5432/edusynapse_central",
        )
        engine = create_async_engine(database_url)
        async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with async_session() as session:
            await seed_central_database(session)

    asyncio.run(main())

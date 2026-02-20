# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tenant lifecycle management service.

This module provides complete tenant lifecycle management including:
- Tenant creation with Docker container provisioning
- Database initialization with schema and seed data
- Tenant status management
- Tenant deletion

Example:
    >>> tenant_service = TenantService(db, container_manager)
    >>> tenant = await tenant_service.create_tenant(request)
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from src.infrastructure.database.models.central.tenant import Tenant
from src.infrastructure.database.models.base import Base
from src.infrastructure.docker.tenant_container import (
    TenantContainerManager,
    TenantContainerInfo,
    ContainerStatus,
)

logger = logging.getLogger(__name__)


class TenantProvisioningError(Exception):
    """Raised when tenant provisioning fails."""

    pass


class TenantNotFoundError(Exception):
    """Raised when tenant is not found."""

    pass


class TenantAlreadyExistsError(Exception):
    """Raised when tenant code already exists."""

    pass


class TenantService:
    """Tenant lifecycle management service.

    Handles complete tenant lifecycle from creation to deletion,
    including Docker container management and database provisioning.

    Attributes:
        _db: Central database session.
        _container_manager: Docker container manager.
        _db_password: Tenant database password.

    Example:
        >>> tenant_service = TenantService(db, container_manager)
        >>> tenant = await tenant_service.create_tenant(
        ...     code="acme",
        ...     name="Acme School",
        ...     admin_email="admin@acme.edu",
        ... )
    """

    def __init__(
        self,
        db: AsyncSession,
        container_manager: TenantContainerManager | None = None,
        db_password: str | None = None,
    ) -> None:
        """Initialize the tenant service.

        Args:
            db: Central database async session.
            container_manager: Docker container manager. Creates default if not provided.
            db_password: Tenant database password. Defaults to TENANT_DB_PASSWORD env.
        """
        self._db = db
        self._container_manager = container_manager or TenantContainerManager()
        self._db_password = db_password or os.getenv(
            "TENANT_DB_PASSWORD", "edusynapse_tenant_password"
        )

    async def create_tenant(
        self,
        code: str,
        name: str,
        admin_email: str,
        admin_name: str | None = None,
        tier: str = "standard",
        license_id: str | UUID | None = None,
        settings: dict | None = None,
    ) -> Tenant:
        """Create a new tenant with Docker container provisioning.

        This method:
        1. Validates the tenant code is unique
        2. Creates tenant record in Central DB (status=provisioning)
        3. Creates and starts PostgreSQL container
        4. Waits for container to become healthy
        5. Updates tenant status to active

        Args:
            code: Unique tenant code (alphanumeric with underscores).
            name: Display name for the tenant.
            admin_email: Primary admin contact email.
            admin_name: Primary admin contact name.
            tier: Subscription tier (free, standard, premium, enterprise).
            license_id: Optional license ID to associate.
            settings: Optional tenant settings dict.

        Returns:
            Created Tenant object with active status.

        Raises:
            TenantAlreadyExistsError: If code already exists.
            TenantProvisioningError: If container creation or startup fails.
        """
        code = code.lower().strip()

        if not code or not code.replace("_", "").isalnum():
            raise ValueError(
                f"Invalid tenant code: {code}. Must be alphanumeric with optional underscores."
            )

        existing = await self._get_tenant_by_code(code)
        if existing:
            raise TenantAlreadyExistsError(f"Tenant with code '{code}' already exists")

        container_name = f"edusynapse-tenant-{code}-db"
        database_name = f"edusynapse_tenant_{code}"
        db_user = os.getenv("TENANT_DB_USER", "edusynapse")

        tenant = Tenant(
            code=code,
            name=name,
            status="provisioning",
            hosting_type="managed",
            db_host=container_name,
            db_port=5432,
            db_name=database_name,
            db_username=db_user,
            db_password_encrypted=self._encrypt_password(self._db_password),
            admin_email=admin_email,
            admin_name=admin_name,
            tier=tier,
            license_id=str(license_id) if license_id else None,
            settings=settings or {},
        )

        self._db.add(tenant)
        await self._db.flush()

        logger.info("Created tenant record: %s (%s)", code, tenant.id)

        try:
            container_info = self._container_manager.create_tenant_container(code)
            logger.info(
                "Created container for tenant %s: %s (port %d)",
                code,
                container_info.container_name,
                container_info.host_port,
            )

            is_healthy = self._container_manager.wait_for_healthy(code, timeout=120)

            if not is_healthy:
                self._container_manager.remove_tenant_container(code, remove_volume=True)
                raise TenantProvisioningError(
                    f"Container for tenant '{code}' failed to become healthy"
                )

            # Initialize database schema
            await self._initialize_tenant_schema(code, container_info)
            logger.info("Database schema initialized for tenant %s", code)

            # Create admin user in tenant database
            await self._create_admin_user(
                tenant_code=code,
                container_info=container_info,
                admin_email=admin_email,
                admin_name=admin_name,
            )
            logger.info("Admin user created for tenant %s", code)

            tenant.status = "active"
            tenant.provisioned_at = datetime.now(timezone.utc)

            await self._db.commit()

            logger.info("Tenant %s provisioned successfully", code)

            return tenant

        except Exception as e:
            logger.error("Failed to provision tenant %s: %s", code, str(e))

            try:
                self._container_manager.remove_tenant_container(code, remove_volume=True)
            except Exception:
                pass

            await self._db.rollback()

            if isinstance(e, TenantProvisioningError):
                raise

            raise TenantProvisioningError(f"Failed to provision tenant: {str(e)}") from e

    async def get_tenant(self, tenant_id: str | UUID) -> Tenant | None:
        """Get tenant by ID.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            Tenant if found, None otherwise.
        """
        stmt = select(Tenant).where(
            Tenant.id == str(tenant_id),
            Tenant.deleted_at == None,  # noqa: E711
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_tenant_by_code(self, code: str) -> Tenant | None:
        """Get tenant by code.

        Args:
            code: Tenant code.

        Returns:
            Tenant if found, None otherwise.
        """
        return await self._get_tenant_by_code(code)

    async def list_tenants(
        self,
        status: str | None = None,
        tier: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Tenant], int]:
        """List tenants with optional filters.

        Args:
            status: Filter by status.
            tier: Filter by tier.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            Tuple of (tenants list, total count).
        """
        base_stmt = select(Tenant).where(Tenant.deleted_at == None)  # noqa: E711

        if status:
            base_stmt = base_stmt.where(Tenant.status == status)
        if tier:
            base_stmt = base_stmt.where(Tenant.tier == tier)

        count_stmt = select(func.count()).select_from(base_stmt.subquery())
        count_result = await self._db.execute(count_stmt)
        total = count_result.scalar_one()

        list_stmt = base_stmt.order_by(Tenant.created_at.desc()).limit(limit).offset(offset)
        result = await self._db.execute(list_stmt)
        tenants = list(result.scalars().all())

        return tenants, total

    async def update_tenant(
        self,
        tenant_id: str | UUID,
        name: str | None = None,
        admin_email: str | None = None,
        admin_name: str | None = None,
        tier: str | None = None,
        settings: dict | None = None,
    ) -> Tenant:
        """Update tenant details.

        Args:
            tenant_id: Tenant identifier.
            name: New display name.
            admin_email: New admin email.
            admin_name: New admin name.
            tier: New subscription tier.
            settings: New settings (replaces existing).

        Returns:
            Updated Tenant.

        Raises:
            TenantNotFoundError: If tenant not found.
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(f"Tenant not found: {tenant_id}")

        if name is not None:
            tenant.name = name
        if admin_email is not None:
            tenant.admin_email = admin_email
        if admin_name is not None:
            tenant.admin_name = admin_name
        if tier is not None:
            tenant.tier = tier
        if settings is not None:
            tenant.settings = settings

        await self._db.commit()
        await self._db.refresh(tenant)

        logger.info("Updated tenant: %s", tenant.code)

        return tenant

    async def suspend_tenant(self, tenant_id: str | UUID) -> Tenant:
        """Suspend a tenant.

        Stops the tenant's container but preserves data.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            Updated Tenant with suspended status.

        Raises:
            TenantNotFoundError: If tenant not found.
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(f"Tenant not found: {tenant_id}")

        try:
            self._container_manager.stop_tenant_container(tenant.code)
        except ValueError:
            pass

        tenant.status = "suspended"
        tenant.suspended_at = datetime.now(timezone.utc)

        await self._db.commit()

        logger.info("Suspended tenant: %s", tenant.code)

        return tenant

    async def activate_tenant(self, tenant_id: str | UUID) -> Tenant:
        """Activate a suspended tenant.

        Starts the tenant's container.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            Updated Tenant with active status.

        Raises:
            TenantNotFoundError: If tenant not found.
            TenantProvisioningError: If container start fails.
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(f"Tenant not found: {tenant_id}")

        try:
            self._container_manager.start_tenant_container(tenant.code)

            is_healthy = self._container_manager.wait_for_healthy(tenant.code, timeout=60)
            if not is_healthy:
                raise TenantProvisioningError("Container failed to become healthy")

        except ValueError as e:
            raise TenantProvisioningError(f"Failed to start container: {str(e)}")

        tenant.status = "active"
        tenant.suspended_at = None

        await self._db.commit()

        logger.info("Activated tenant: %s", tenant.code)

        return tenant

    async def delete_tenant(
        self,
        tenant_id: str | UUID,
        hard_delete: bool = False,
    ) -> None:
        """Delete a tenant.

        By default, performs a soft delete (sets deleted_at).
        Hard delete removes the record and container completely.

        Args:
            tenant_id: Tenant identifier.
            hard_delete: If True, permanently deletes tenant and data.

        Raises:
            TenantNotFoundError: If tenant not found.
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(f"Tenant not found: {tenant_id}")

        if hard_delete:
            try:
                self._container_manager.remove_tenant_container(
                    tenant.code, remove_volume=True
                )
            except ValueError:
                pass

            await self._db.delete(tenant)
            logger.info("Hard deleted tenant: %s", tenant.code)
        else:
            try:
                self._container_manager.stop_tenant_container(tenant.code)
            except ValueError:
                pass

            tenant.status = "deleted"
            tenant.deleted_at = datetime.now(timezone.utc)
            logger.info("Soft deleted tenant: %s", tenant.code)

        await self._db.commit()

    async def get_tenant_container_status(self, tenant_id: str | UUID) -> ContainerStatus:
        """Get the Docker container status for a tenant.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            ContainerStatus enum value.

        Raises:
            TenantNotFoundError: If tenant not found.
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(f"Tenant not found: {tenant_id}")

        return self._container_manager.get_tenant_container_status(tenant.code)

    async def get_tenant_container_info(
        self, tenant_id: str | UUID
    ) -> TenantContainerInfo | None:
        """Get Docker container information for a tenant.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            TenantContainerInfo if container exists, None otherwise.

        Raises:
            TenantNotFoundError: If tenant not found.
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(f"Tenant not found: {tenant_id}")

        return self._container_manager.get_tenant_container_info(tenant.code)

    async def _get_tenant_by_code(self, code: str) -> Tenant | None:
        """Get tenant by code (internal helper)."""
        stmt = select(Tenant).where(
            Tenant.code == code,
            Tenant.deleted_at == None,  # noqa: E711
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def _initialize_tenant_schema(
        self,
        tenant_code: str,
        container_info: TenantContainerInfo,
    ) -> None:
        """Initialize database schema for a new tenant.

        Creates all required tables using SQLAlchemy metadata.
        This ensures the schema is always up-to-date with the current models.

        Args:
            tenant_code: Tenant code.
            container_info: Container information with connection details.

        Raises:
            TenantProvisioningError: If schema creation fails.
        """
        # Import all tenant models to ensure they're registered with Base.metadata
        # This import is needed here to avoid circular imports at module level
        from src.infrastructure.database.models.tenant import (
            User,
            Role,
            Permission,
            RolePermission,
            UserRole,
            UserSession,
            RefreshToken,
            EmailVerification,
            School,
            AcademicYear,
            Class,
            ClassStudent,
            ClassTeacher,
            ParentStudentRelation,
            CurriculumFramework,
            CurriculumStage,
            GradeLevel,
            Subject,
            Unit,
            Topic,
            LearningObjective,
            Prerequisite,
            PracticeSession,
            PracticeQuestion,
            StudentAnswer,
            EvaluationResult,
            PracticeHelperSession,
            PracticeHelperMessage,
            LearningSession,
            LearningSessionMessage,
            Conversation,
            ConversationMessage,
            ConversationSummary,
            EpisodicMemory,
            SemanticMemory,
            ProceduralMemory,
            AssociativeMemory,
            ReviewItem,
            ReviewLog,
            DiagnosticScan,
            DiagnosticIndicator,
            DiagnosticRecommendation,
            EmotionalSignal,
            CompanionSession,
            CompanionActivity,
            StudentNote,
            GameSession,
            GameMove,
            GameAnalysis,
            Alert,
            Notification,
            NotificationPreference,
            NotificationTemplate,
            AnalyticsEvent,
            DailySummary,
            MasterySnapshot,
            EngagementMetric,
            TenantSetting,
            UserPreference,
            FeatureFlag,
            Language,
            Translation,
            AuditLog,
        )

        # Build connection URL using Docker network (container name + internal port)
        # This works because both API and tenant DB containers are on edusynapse-network
        db_url = (
            f"postgresql+asyncpg://{container_info.username}:{self._db_password}"
            f"@{container_info.container_name}:5432/{container_info.database_name}"
        )

        engine = create_async_engine(db_url, echo=False)

        try:
            async with engine.begin() as conn:
                # Drop and recreate public schema to ensure clean state
                # This handles all foreign key dependencies
                await conn.execute(text("DROP SCHEMA public CASCADE"))
                await conn.execute(text("CREATE SCHEMA public"))
                await conn.execute(text("GRANT ALL ON SCHEMA public TO PUBLIC"))

                # Enable required PostgreSQL extensions
                await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
                await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "pg_trgm"'))

                # Create only tenant-specific tables (not central DB tables)
                # Filter metadata to exclude central DB tables
                tenant_tables = [
                    "users", "roles", "permissions", "role_permissions", "user_roles",
                    "user_sessions", "refresh_tokens", "email_verifications",
                    "schools", "academic_years", "classes", "class_students", "class_teachers",
                    "parent_student_relations",
                    "curriculum_frameworks", "curriculum_stages", "grade_levels", "subjects",
                    "units", "topics", "learning_objectives", "prerequisites",
                    "practice_sessions", "practice_questions", "student_answers", "evaluation_results",
                    "practice_helper_sessions", "practice_helper_messages",
                    "learning_sessions", "learning_session_messages",
                    "conversations", "conversation_messages", "conversation_summaries",
                    "episodic_memories", "semantic_memories", "procedural_memories", "associative_memories",
                    "review_items", "review_logs",
                    "diagnostic_scans", "diagnostic_indicators", "diagnostic_recommendations",
                    "emotional_signals", "companion_sessions", "companion_activities",
                    "student_notes", "game_sessions", "game_moves", "game_analyses",
                    "alerts", "notifications", "notification_preferences", "notification_templates",
                    "analytics_events", "daily_summaries", "mastery_snapshots", "engagement_metrics",
                    "tenant_settings", "user_preferences", "feature_flags",
                    "languages", "translations", "audit_logs",
                ]

                # Create only tables that are in the tenant_tables list
                from sqlalchemy import MetaData
                tenant_metadata = MetaData()
                for table_name in tenant_tables:
                    if table_name in Base.metadata.tables:
                        table = Base.metadata.tables[table_name]
                        table.to_metadata(tenant_metadata)

                await conn.run_sync(tenant_metadata.create_all)

                # Create alembic_version table and set to latest migration
                # This marks the database as up-to-date with migrations
                await conn.execute(
                    text("""
                        CREATE TABLE IF NOT EXISTS alembic_version (
                            version_num VARCHAR(32) NOT NULL,
                            CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
                        )
                    """)
                )
                await conn.execute(text("DELETE FROM alembic_version"))
                await conn.execute(
                    text(
                        "INSERT INTO alembic_version (version_num) "
                        "VALUES ('012_curriculum_composite_keys')"
                    )
                )

            logger.info("Schema created successfully for tenant %s", tenant_code)

        except Exception as e:
            logger.error("Failed to create schema for tenant %s: %s", tenant_code, e)
            raise TenantProvisioningError(
                f"Failed to initialize database schema: {str(e)}"
            )

        finally:
            await engine.dispose()

    async def _create_admin_user(
        self,
        tenant_code: str,
        container_info: TenantContainerInfo,
        admin_email: str,
        admin_name: str | None = None,
    ) -> None:
        """Create the initial admin user for a new tenant.

        This creates a tenant_admin user that can log in and manage the tenant.
        The user is linked via SSO (LMS provider) using email as external ID.

        Args:
            tenant_code: Tenant code.
            container_info: Container information with connection details.
            admin_email: Admin email address.
            admin_name: Admin full name (optional).

        Raises:
            TenantProvisioningError: If admin user creation fails.
        """
        from src.infrastructure.database.models.tenant.user import User

        # Parse name into first/last name
        first_name = "Admin"
        last_name = "User"
        if admin_name:
            name_parts = admin_name.strip().split(" ", 1)
            first_name = name_parts[0]
            last_name = name_parts[1] if len(name_parts) > 1 else ""

        # Build connection URL using Docker network (container name + internal port)
        db_url = (
            f"postgresql+asyncpg://{container_info.username}:{self._db_password}"
            f"@{container_info.container_name}:5432/{container_info.database_name}"
        )

        engine = create_async_engine(db_url, echo=False)

        try:
            from sqlalchemy.orm import sessionmaker
            from sqlalchemy.ext.asyncio import AsyncSession as TenantAsyncSession

            async_session = sessionmaker(engine, class_=TenantAsyncSession, expire_on_commit=False)

            async with async_session() as session:
                # Create admin user
                admin_user = User(
                    email=admin_email,
                    first_name=first_name,
                    last_name=last_name,
                    user_type="tenant_admin",
                    status="active",
                    email_verified=True,
                    sso_provider="lms",
                    sso_external_id=admin_email,  # Use email as external ID for initial login
                )
                session.add(admin_user)
                await session.commit()

                logger.info(
                    "Created admin user for tenant %s: %s (%s)",
                    tenant_code,
                    admin_user.id,
                    admin_email,
                )

        except Exception as e:
            logger.error("Failed to create admin user for tenant %s: %s", tenant_code, e)
            raise TenantProvisioningError(
                f"Failed to create admin user: {str(e)}"
            )

        finally:
            await engine.dispose()

    def _encrypt_password(self, password: str) -> bytes:
        """Encrypt database password for storage.

        In production, this should use proper encryption.
        For now, using simple encoding.
        """
        return password.encode("utf-8")

    def _decrypt_password(self, encrypted: bytes) -> str:
        """Decrypt database password.

        In production, this should use proper decryption.
        For now, using simple decoding.
        """
        return encrypted.decode("utf-8")

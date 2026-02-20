# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Central Curriculum sync service.

This module provides synchronization of curriculum data from the Central
Curriculum service to the EduSynapse tenant database.

The sync process:
1. Fetch all data from Central Curriculum API
2. Upsert data into tenant database using code-based composite keys
3. Handle deletions (mark as inactive or delete)

Data hierarchy:
- CurriculumFramework (top-level)
- CurriculumStage (within framework)
- GradeLevel (within stage)
- Subject (within framework)
- Unit (within subject + grade)
- Topic (within unit)
- LearningObjective (within topic)
- Prerequisite (topic-to-topic relationships)

Example:
    >>> sync = CurriculumSyncService(db, settings)
    >>> result = await sync.sync_all()
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

import httpx
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config.settings import CentralCurriculumSettings
from src.infrastructure.database.models.tenant.curriculum import (
    CurriculumFramework,
    CurriculumStage,
    GradeLevel,
    Subject,
    Unit,
    Topic,
    LearningObjective,
    Prerequisite,
)
from src.infrastructure.database.models.tenant.skill import (
    SkillTaxonomy,
    SkillCategory,
    SubjectSkillMapping,
)

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of a sync operation.

    Attributes:
        success: Whether sync completed successfully.
        frameworks_synced: Number of frameworks processed.
        stages_synced: Number of stages processed.
        grades_synced: Number of grades processed.
        subjects_synced: Number of subjects processed.
        units_synced: Number of units processed.
        topics_synced: Number of topics processed.
        objectives_synced: Number of objectives processed.
        prerequisites_synced: Number of prerequisites processed.
        taxonomies_synced: Number of skill taxonomies processed.
        categories_synced: Number of skill categories processed.
        mappings_synced: Number of subject-skill mappings processed.
        error: Error message if sync failed.
        started_at: When sync started.
        completed_at: When sync completed.
    """

    success: bool
    frameworks_synced: int = 0
    stages_synced: int = 0
    grades_synced: int = 0
    subjects_synced: int = 0
    units_synced: int = 0
    topics_synced: int = 0
    objectives_synced: int = 0
    prerequisites_synced: int = 0
    taxonomies_synced: int = 0
    categories_synced: int = 0
    mappings_synced: int = 0
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def duration_seconds(self) -> float | None:
        """Get sync duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class CurriculumSyncError(Exception):
    """Exception raised when curriculum sync fails."""

    pass


class CurriculumSyncService:
    """Service for syncing curriculum data from Central Curriculum.

    This service connects to the Central Curriculum API and synchronizes
    all curriculum data to the tenant database using upsert operations.

    Supports two modes:
    1. Global credentials (from environment): Uses settings.auth_headers
    2. Per-tenant credentials: Uses tenant-specific API key/secret to get JWT

    Attributes:
        _db: Async database session.
        _settings: Central Curriculum configuration.
        _client: HTTP client for API requests.
        _jwt_token: JWT token for authenticated requests (per-tenant mode).
        _tenant_profiles: List of framework codes the tenant has access to.

    Example:
        >>> # Global credentials mode
        >>> sync = CurriculumSyncService(db, settings)
        >>> result = await sync.sync_all()

        >>> # Per-tenant credentials mode
        >>> sync = CurriculumSyncService(db, settings, cc_api_key="...", cc_api_secret="...")
        >>> await sync.authenticate()
        >>> result = await sync.sync_tenant_frameworks()
    """

    def __init__(
        self,
        db: AsyncSession,
        settings: CentralCurriculumSettings,
        cc_api_key: str | None = None,
        cc_api_secret: str | None = None,
    ) -> None:
        """Initialize the sync service.

        Args:
            db: Async database session.
            settings: Central Curriculum configuration.
            cc_api_key: Optional per-tenant CC API key.
            cc_api_secret: Optional per-tenant CC API secret.
        """
        self._db = db
        self._settings = settings
        self._cc_api_key = cc_api_key
        self._cc_api_secret = cc_api_secret
        self._jwt_token: str | None = None
        self._tenant_profiles: list[dict] | None = None

        # Initialize client - use global auth headers only if no per-tenant credentials
        # Per-tenant mode will authenticate and use JWT instead
        if cc_api_key and cc_api_secret:
            # Per-tenant mode: start with no auth, will add JWT after authenticate()
            self._client = httpx.AsyncClient(
                base_url=settings.base_url,
                timeout=settings.timeout,
            )
        else:
            # Global mode: use global auth headers (X-API-Key, X-API-Secret)
            self._client = httpx.AsyncClient(
                base_url=settings.base_url,
                headers=settings.auth_headers,
                timeout=settings.timeout,
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def authenticate(self) -> bool:
        """Authenticate with CC using per-tenant credentials.

        Gets a JWT token from CC API using the tenant's API key and secret.
        Must be called before sync operations in per-tenant mode.

        Returns:
            True if authentication was successful.

        Raises:
            CurriculumSyncError: If authentication fails.
        """
        if not self._cc_api_key or not self._cc_api_secret:
            logger.warning("No per-tenant credentials provided, using global auth")
            return True

        logger.info("Authenticating with CC using tenant credentials")

        try:
            response = await self._client.post(
                "/auth/tenant/login",
                json={
                    "api_key": self._cc_api_key,
                    "api_secret": self._cc_api_secret,
                },
            )
            response.raise_for_status()
            data = response.json()

            self._jwt_token = data.get("access_token")
            if not self._jwt_token:
                raise CurriculumSyncError("No access token in response")

            # Update client headers with JWT token
            self._client.headers["Authorization"] = f"Bearer {self._jwt_token}"
            logger.info("Successfully authenticated with CC")
            return True

        except httpx.HTTPStatusError as e:
            logger.error("CC authentication failed: %s", e.response.text)
            raise CurriculumSyncError(f"Authentication failed: {e.response.text}") from e
        except Exception as e:
            logger.error("CC authentication error: %s", str(e))
            raise CurriculumSyncError(f"Authentication error: {str(e)}") from e

    async def fetch_tenant_profiles(self) -> list[dict]:
        """Fetch the tenant's framework profiles from CC.

        Gets the list of frameworks the tenant has access to via their
        curriculum profiles in CC.

        Returns:
            List of profile dictionaries containing framework_code.

        Raises:
            CurriculumSyncError: If fetching profiles fails.
        """
        logger.info("Fetching tenant profiles from CC")

        try:
            response = await self._client.get("/profiles")
            response.raise_for_status()
            data = response.json()

            profiles = data.get("data", [])
            self._tenant_profiles = profiles
            logger.info("Found %d framework profiles for tenant", len(profiles))
            return profiles

        except httpx.HTTPStatusError as e:
            logger.error("Failed to fetch tenant profiles: %s", e.response.text)
            raise CurriculumSyncError(f"Failed to fetch profiles: {e.response.text}") from e
        except Exception as e:
            logger.error("Error fetching tenant profiles: %s", str(e))
            raise CurriculumSyncError(f"Error fetching profiles: {str(e)}") from e

    async def sync_tenant_frameworks(self) -> SyncResult:
        """Sync only the frameworks the tenant has profile access to.

        This method should be used for per-tenant sync. It:
        1. Authenticates with CC using tenant credentials (if not already done)
        2. Fetches the tenant's profiles to get framework list
        3. Syncs only those frameworks

        Returns:
            SyncResult with counts and status.
        """
        started_at = datetime.now(timezone.utc)
        result = SyncResult(success=False, started_at=started_at)

        try:
            # Authenticate if we have per-tenant credentials
            if self._cc_api_key and self._cc_api_secret and not self._jwt_token:
                await self.authenticate()

            # Fetch tenant's profiles to know which frameworks to sync
            profiles = await self.fetch_tenant_profiles()
            if not profiles:
                logger.warning("No framework profiles found for tenant, nothing to sync")
                result.success = True
                result.completed_at = datetime.now(timezone.utc)
                return result

            # Get active framework codes from profiles
            framework_codes = [
                p["framework_code"]
                for p in profiles
                if p.get("is_active", True) and p.get("can_read", True)
            ]
            logger.info("Syncing %d frameworks for tenant: %s", len(framework_codes), framework_codes)

            # Sync each framework
            for framework_code in framework_codes:
                try:
                    fw_result = await self.sync_framework(framework_code)
                    result.frameworks_synced += 1
                    result.stages_synced += fw_result.stages_synced
                    result.grades_synced += fw_result.grades_synced
                    result.subjects_synced += fw_result.subjects_synced
                    result.units_synced += fw_result.units_synced
                    result.topics_synced += fw_result.topics_synced
                    result.objectives_synced += fw_result.objectives_synced
                except Exception as e:
                    logger.error("Failed to sync framework %s: %s", framework_code, str(e))
                    # Continue with other frameworks

            # Sync skill taxonomies and categories
            try:
                skill_result = await self.sync_skills()
                result.taxonomies_synced = skill_result.taxonomies_synced
                result.categories_synced = skill_result.categories_synced
            except Exception as e:
                logger.error("Failed to sync skills: %s", str(e))

            # Sync subject-skill mappings
            try:
                mapping_result = await self.sync_skill_mappings()
                result.mappings_synced = mapping_result.mappings_synced
            except Exception as e:
                logger.error("Failed to sync skill mappings: %s", str(e))

            result.success = True
            result.completed_at = datetime.now(timezone.utc)

            logger.info(
                "Tenant curriculum sync completed: frameworks=%d, stages=%d, grades=%d, "
                "subjects=%d, units=%d, topics=%d, objectives=%d",
                result.frameworks_synced,
                result.stages_synced,
                result.grades_synced,
                result.subjects_synced,
                result.units_synced,
                result.topics_synced,
                result.objectives_synced,
            )

            return result

        except Exception as e:
            await self._db.rollback()
            result.error = str(e)
            result.completed_at = datetime.now(timezone.utc)
            logger.error("Tenant curriculum sync failed: %s", str(e))
            raise CurriculumSyncError(f"Tenant sync failed: {str(e)}") from e

    async def sync_all(self) -> SyncResult:
        """Sync all curriculum data from Central Curriculum.

        Uses the /full endpoint for each framework to fetch all nested data
        in a single request per framework, then aggregates the results.

        Returns:
            SyncResult with counts and status.
        """
        started_at = datetime.now(timezone.utc)
        result = SyncResult(success=False, started_at=started_at)

        try:
            # Get list of frameworks
            frameworks = await self._fetch_frameworks()
            result.frameworks_synced = len(frameworks)

            # Sync each framework using the /full endpoint
            for fw in frameworks:
                fw_result = await self.sync_framework(fw["code"])
                result.stages_synced += fw_result.stages_synced
                result.grades_synced += fw_result.grades_synced
                result.subjects_synced += fw_result.subjects_synced
                result.units_synced += fw_result.units_synced
                result.topics_synced += fw_result.topics_synced
                result.objectives_synced += fw_result.objectives_synced

            # Sync skill taxonomies and categories
            skill_result = await self.sync_skills()
            result.taxonomies_synced = skill_result.taxonomies_synced
            result.categories_synced = skill_result.categories_synced

            # Sync subject-skill mappings (requires subjects to be synced first)
            mapping_result = await self.sync_skill_mappings()
            result.mappings_synced = mapping_result.mappings_synced

            result.success = True
            result.completed_at = datetime.now(timezone.utc)

            logger.info(
                "Curriculum sync completed: frameworks=%d, stages=%d, grades=%d, "
                "subjects=%d, units=%d, topics=%d, objectives=%d, "
                "taxonomies=%d, categories=%d, mappings=%d",
                result.frameworks_synced,
                result.stages_synced,
                result.grades_synced,
                result.subjects_synced,
                result.units_synced,
                result.topics_synced,
                result.objectives_synced,
                result.taxonomies_synced,
                result.categories_synced,
                result.mappings_synced,
            )

            return result

        except Exception as e:
            await self._db.rollback()
            result.error = str(e)
            result.completed_at = datetime.now(timezone.utc)
            logger.error("Curriculum sync failed: %s", str(e))
            raise CurriculumSyncError(f"Sync failed: {str(e)}") from e

    async def sync_framework(self, framework_code: str) -> SyncResult:
        """Sync a single framework and all its children using the /full endpoint.

        Uses the Central Curriculum /frameworks/{code}/full endpoint to fetch
        all nested data in a single request, then processes it hierarchically.

        Args:
            framework_code: Framework code to sync (e.g., "UK-NC-2014").

        Returns:
            SyncResult with counts and status.
        """
        started_at = datetime.now(timezone.utc)
        result = SyncResult(success=False, started_at=started_at)

        try:
            # Fetch full framework data with all nested entities
            logger.info("Fetching full framework data: %s", framework_code)
            fw_data = await self._fetch_framework_full(framework_code)

            # Sync framework
            await self._upsert_framework(fw_data)
            result.frameworks_synced = 1

            # Sync stages and their grades
            for stage in fw_data.get("stages", []):
                await self._upsert_stage(framework_code, stage)
                result.stages_synced += 1

                # Grades are nested within stages
                for grade in stage.get("grades", []):
                    await self._upsert_grade(framework_code, stage["code"], grade)
                    result.grades_synced += 1

            # Sync subjects with nested units and topics
            for subject in fw_data.get("subjects", []):
                await self._upsert_subject(framework_code, subject)
                result.subjects_synced += 1

                # Units are nested within subjects
                for unit in subject.get("units", []):
                    await self._upsert_unit(framework_code, subject["code"], unit)
                    result.units_synced += 1

                    # Topics are nested within units
                    for topic in unit.get("topics", []):
                        await self._upsert_topic(
                            framework_code,
                            subject["code"],
                            unit["grade_code"],
                            unit["code"],
                            topic,
                        )
                        result.topics_synced += 1

                        # Objectives are nested within topics
                        for obj in topic.get("objectives", []):
                            await self._upsert_objective(
                                framework_code,
                                subject["code"],
                                unit["grade_code"],
                                unit["code"],
                                topic["code"],
                                obj,
                            )
                            result.objectives_synced += 1

            await self._db.commit()

            result.success = True
            result.completed_at = datetime.now(timezone.utc)

            logger.info(
                "Framework sync completed: %s - stages=%d, grades=%d, "
                "subjects=%d, units=%d, topics=%d, objectives=%d",
                framework_code,
                result.stages_synced,
                result.grades_synced,
                result.subjects_synced,
                result.units_synced,
                result.topics_synced,
                result.objectives_synced,
            )

            return result

        except Exception as e:
            await self._db.rollback()
            result.error = str(e)
            result.completed_at = datetime.now(timezone.utc)
            logger.error("Framework sync failed: %s - %s", framework_code, str(e))
            raise CurriculumSyncError(f"Sync failed for {framework_code}: {str(e)}") from e

    # =========================================================================
    # API Fetch Methods
    # =========================================================================

    async def _fetch_frameworks(self) -> list[dict]:
        """Fetch all frameworks from Central Curriculum."""
        response = await self._client.get("/frameworks")
        response.raise_for_status()
        return response.json().get("data", [])

    async def _fetch_framework(self, framework_code: str) -> dict:
        """Fetch a single framework from Central Curriculum."""
        response = await self._client.get(f"/frameworks/{framework_code}")
        response.raise_for_status()
        return response.json().get("data", {})

    async def _fetch_framework_full(self, framework_code: str) -> dict:
        """Fetch a framework with all nested data from Central Curriculum.

        Uses the /frameworks/{code}/full endpoint which returns:
        - Framework metadata
        - stages (with grades nested)
        - subjects (with units nested, which have topics nested)

        Args:
            framework_code: Framework code (e.g., "UK-NC-2014").

        Returns:
            Full framework data with all nested entities.
        """
        response = await self._client.get(f"/frameworks/{framework_code}/full")
        response.raise_for_status()
        return response.json().get("data", {})

    async def _fetch_stages(self, framework_code: str) -> list[dict]:
        """Fetch stages for a framework."""
        response = await self._client.get(f"/frameworks/{framework_code}/stages")
        response.raise_for_status()
        return response.json().get("data", [])

    async def _fetch_grades(self, framework_code: str, stage_code: str) -> list[dict]:
        """Fetch grades for a stage."""
        response = await self._client.get(
            f"/frameworks/{framework_code}/stages/{stage_code}/grades"
        )
        response.raise_for_status()
        return response.json().get("data", [])

    async def _fetch_subjects(self, framework_code: str) -> list[dict]:
        """Fetch subjects for a framework."""
        response = await self._client.get(f"/frameworks/{framework_code}/subjects")
        response.raise_for_status()
        return response.json().get("data", [])

    async def _fetch_units(self, framework_code: str, subject_code: str) -> list[dict]:
        """Fetch units for a subject."""
        response = await self._client.get(
            f"/frameworks/{framework_code}/subjects/{subject_code}/units"
        )
        response.raise_for_status()
        return response.json().get("data", [])

    async def _fetch_topics(
        self,
        framework_code: str,
        subject_code: str,
        grade_code: str,
        unit_code: str,
    ) -> list[dict]:
        """Fetch topics for a unit."""
        response = await self._client.get(
            f"/frameworks/{framework_code}/subjects/{subject_code}"
            f"/grades/{grade_code}/units/{unit_code}/topics"
        )
        response.raise_for_status()
        return response.json().get("data", [])

    async def _fetch_objectives(
        self,
        framework_code: str,
        subject_code: str,
        grade_code: str,
        unit_code: str,
        topic_code: str,
    ) -> list[dict]:
        """Fetch objectives for a topic."""
        response = await self._client.get(
            f"/frameworks/{framework_code}/subjects/{subject_code}"
            f"/grades/{grade_code}/units/{unit_code}/topics/{topic_code}/objectives"
        )
        response.raise_for_status()
        return response.json().get("data", [])

    async def _fetch_prerequisites(self) -> list[dict]:
        """Fetch all prerequisites."""
        response = await self._client.get("/prerequisites")
        response.raise_for_status()
        return response.json().get("data", [])

    # =========================================================================
    # Upsert Methods
    # =========================================================================

    async def _upsert_framework(self, data: dict) -> CurriculumFramework:
        """Upsert a curriculum framework."""
        stmt = select(CurriculumFramework).where(
            CurriculumFramework.code == data["code"]
        )
        result = await self._db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.name = data["name"]
            existing.description = data.get("description")
            existing.framework_type = data.get("framework_type", "national")
            existing.country_code = data.get("country_code")
            existing.organization = data.get("organization")
            existing.version = data.get("version")
            existing.language = data.get("language", "en")
            existing.is_active = data.get("is_active", True)
            existing.is_published = data.get("is_published", True)
            return existing

        framework = CurriculumFramework(
            code=data["code"],
            name=data["name"],
            description=data.get("description"),
            framework_type=data.get("framework_type", "national"),
            country_code=data.get("country_code"),
            organization=data.get("organization"),
            version=data.get("version"),
            language=data.get("language", "en"),
            is_active=data.get("is_active", True),
            is_published=data.get("is_published", True),
        )
        self._db.add(framework)
        await self._db.flush()
        return framework

    async def _upsert_stage(self, framework_code: str, data: dict) -> CurriculumStage:
        """Upsert a curriculum stage."""
        stmt = select(CurriculumStage).where(
            CurriculumStage.framework_code == framework_code,
            CurriculumStage.code == data["code"],
        )
        result = await self._db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.name = data["name"]
            existing.description = data.get("description")
            existing.age_start = data.get("age_start")
            existing.age_end = data.get("age_end")
            existing.sequence = data.get("sequence", 0)
            return existing

        stage = CurriculumStage(
            framework_code=framework_code,
            code=data["code"],
            name=data["name"],
            description=data.get("description"),
            age_start=data.get("age_start"),
            age_end=data.get("age_end"),
            sequence=data.get("sequence", 0),
        )
        self._db.add(stage)
        await self._db.flush()
        return stage

    async def _upsert_grade(
        self, framework_code: str, stage_code: str, data: dict
    ) -> GradeLevel:
        """Upsert a grade level."""
        stmt = select(GradeLevel).where(
            GradeLevel.framework_code == framework_code,
            GradeLevel.stage_code == stage_code,
            GradeLevel.code == data["code"],
        )
        result = await self._db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.name = data["name"]
            existing.typical_age = data.get("typical_age")
            existing.sequence = data.get("sequence", 0)
            return existing

        grade = GradeLevel(
            framework_code=framework_code,
            stage_code=stage_code,
            code=data["code"],
            name=data["name"],
            typical_age=data.get("typical_age"),
            sequence=data.get("sequence", 0),
        )
        self._db.add(grade)
        await self._db.flush()
        return grade

    async def _upsert_subject(self, framework_code: str, data: dict) -> Subject:
        """Upsert a subject."""
        stmt = select(Subject).where(
            Subject.framework_code == framework_code,
            Subject.code == data["code"],
        )
        result = await self._db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.name = data["name"]
            existing.description = data.get("description")
            existing.icon = data.get("icon")
            existing.color = data.get("color")
            existing.is_core = data.get("is_core", True)
            existing.sequence = data.get("sequence", 0)
            return existing

        subject = Subject(
            framework_code=framework_code,
            code=data["code"],
            name=data["name"],
            description=data.get("description"),
            icon=data.get("icon"),
            color=data.get("color"),
            is_core=data.get("is_core", True),
            sequence=data.get("sequence", 0),
        )
        self._db.add(subject)
        await self._db.flush()
        return subject

    async def _upsert_unit(
        self, framework_code: str, subject_code: str, data: dict
    ) -> Unit:
        """Upsert a unit."""
        stmt = select(Unit).where(
            Unit.framework_code == framework_code,
            Unit.subject_code == subject_code,
            Unit.grade_code == data["grade_code"],
            Unit.code == data["code"],
        )
        result = await self._db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.name = data["name"]
            existing.description = data.get("description")
            existing.estimated_hours = data.get("estimated_hours")
            existing.sequence = data.get("sequence", 0)
            return existing

        unit = Unit(
            framework_code=framework_code,
            subject_code=subject_code,
            grade_code=data["grade_code"],
            code=data["code"],
            name=data["name"],
            description=data.get("description"),
            estimated_hours=data.get("estimated_hours"),
            sequence=data.get("sequence", 0),
        )
        self._db.add(unit)
        await self._db.flush()
        return unit

    async def _upsert_topic(
        self,
        framework_code: str,
        subject_code: str,
        grade_code: str,
        unit_code: str,
        data: dict,
    ) -> Topic:
        """Upsert a topic."""
        stmt = select(Topic).where(
            Topic.framework_code == framework_code,
            Topic.subject_code == subject_code,
            Topic.grade_code == grade_code,
            Topic.unit_code == unit_code,
            Topic.code == data["code"],
        )
        result = await self._db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.name = data["name"]
            existing.description = data.get("description")
            existing.base_difficulty = Decimal(str(data.get("base_difficulty", 0.5)))
            existing.estimated_minutes = data.get("estimated_minutes")
            existing.sequence = data.get("sequence", 0)
            return existing

        topic = Topic(
            framework_code=framework_code,
            subject_code=subject_code,
            grade_code=grade_code,
            unit_code=unit_code,
            code=data["code"],
            name=data["name"],
            description=data.get("description"),
            base_difficulty=Decimal(str(data.get("base_difficulty", 0.5))),
            estimated_minutes=data.get("estimated_minutes"),
            sequence=data.get("sequence", 0),
        )
        self._db.add(topic)
        await self._db.flush()
        return topic

    async def _upsert_objective(
        self,
        framework_code: str,
        subject_code: str,
        grade_code: str,
        unit_code: str,
        topic_code: str,
        data: dict,
    ) -> LearningObjective:
        """Upsert a learning objective."""
        stmt = select(LearningObjective).where(
            LearningObjective.framework_code == framework_code,
            LearningObjective.subject_code == subject_code,
            LearningObjective.grade_code == grade_code,
            LearningObjective.unit_code == unit_code,
            LearningObjective.topic_code == topic_code,
            LearningObjective.code == data["code"],
        )
        result = await self._db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.objective = data["objective"]
            existing.bloom_level = data.get("bloom_level")
            existing.mastery_threshold = Decimal(str(data.get("mastery_threshold", 0.8)))
            existing.sequence = data.get("sequence", 0)
            return existing

        objective = LearningObjective(
            framework_code=framework_code,
            subject_code=subject_code,
            grade_code=grade_code,
            unit_code=unit_code,
            topic_code=topic_code,
            code=data["code"],
            objective=data["objective"],
            bloom_level=data.get("bloom_level"),
            mastery_threshold=Decimal(str(data.get("mastery_threshold", 0.8))),
            sequence=data.get("sequence", 0),
        )
        self._db.add(objective)
        await self._db.flush()
        return objective

    async def _upsert_prerequisite(self, data: dict) -> Prerequisite:
        """Upsert a prerequisite relationship."""
        # Parse source and target topic full codes
        source = data["source"]
        target = data["target"]

        stmt = select(Prerequisite).where(
            Prerequisite.source_framework_code == source["framework_code"],
            Prerequisite.source_subject_code == source["subject_code"],
            Prerequisite.source_grade_code == source["grade_code"],
            Prerequisite.source_unit_code == source["unit_code"],
            Prerequisite.source_topic_code == source["topic_code"],
            Prerequisite.target_framework_code == target["framework_code"],
            Prerequisite.target_subject_code == target["subject_code"],
            Prerequisite.target_grade_code == target["grade_code"],
            Prerequisite.target_unit_code == target["unit_code"],
            Prerequisite.target_topic_code == target["topic_code"],
        )
        result = await self._db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.strength = Decimal(str(data.get("strength", 1.0)))
            existing.notes = data.get("notes")
            return existing

        prerequisite = Prerequisite(
            source_framework_code=source["framework_code"],
            source_subject_code=source["subject_code"],
            source_grade_code=source["grade_code"],
            source_unit_code=source["unit_code"],
            source_topic_code=source["topic_code"],
            target_framework_code=target["framework_code"],
            target_subject_code=target["subject_code"],
            target_grade_code=target["grade_code"],
            target_unit_code=target["unit_code"],
            target_topic_code=target["topic_code"],
            strength=Decimal(str(data.get("strength", 1.0))),
            notes=data.get("notes"),
        )
        self._db.add(prerequisite)
        await self._db.flush()
        return prerequisite

    # =========================================================================
    # Skill Sync Methods
    # =========================================================================

    async def sync_skills(self) -> SyncResult:
        """Sync all skill taxonomies and categories from Central Curriculum.

        Fetches list of taxonomies, then for each taxonomy fetches full
        data with categories using the /full endpoint.

        Returns:
            SyncResult with taxonomy and category counts.
        """
        started_at = datetime.now(timezone.utc)
        result = SyncResult(success=False, started_at=started_at)

        try:
            # Fetch list of taxonomies (without categories)
            taxonomies = await self._fetch_taxonomies()

            # For each taxonomy, fetch full data with categories
            for tax_basic in taxonomies:
                tax_code = tax_basic["code"]
                logger.debug("Fetching full taxonomy data: %s", tax_code)

                # Get taxonomy with nested categories
                tax_data = await self._fetch_taxonomy_full(tax_code)

                await self._upsert_taxonomy(tax_data)
                result.taxonomies_synced += 1

                # Categories are nested within taxonomy from /full endpoint
                for cat_data in tax_data.get("categories", []):
                    await self._upsert_category(tax_code, cat_data)
                    result.categories_synced += 1

            await self._db.commit()

            result.success = True
            result.completed_at = datetime.now(timezone.utc)

            logger.info(
                "Skill sync completed: taxonomies=%d, categories=%d",
                result.taxonomies_synced,
                result.categories_synced,
            )

            return result

        except Exception as e:
            await self._db.rollback()
            result.error = str(e)
            result.completed_at = datetime.now(timezone.utc)
            logger.error("Skill sync failed: %s", str(e))
            raise CurriculumSyncError(f"Skill sync failed: {str(e)}") from e

    async def sync_skill_mappings(self) -> SyncResult:
        """Sync subject-skill mappings from Central Curriculum.

        Fetches mappings for each framework that has been synced.
        Must be called after sync_skills and sync_framework to ensure
        both subjects and skill categories exist.

        Returns:
            SyncResult with mapping count.
        """
        started_at = datetime.now(timezone.utc)
        result = SyncResult(success=False, started_at=started_at)

        try:
            # Get all synced frameworks from the database
            stmt = select(CurriculumFramework.code).where(
                CurriculumFramework.is_active == True
            )
            frameworks_result = await self._db.execute(stmt)
            framework_codes = [row[0] for row in frameworks_result.fetchall()]

            # Fetch mappings for each framework
            for framework_code in framework_codes:
                logger.debug("Fetching skill mappings for framework: %s", framework_code)
                try:
                    mappings = await self._fetch_skill_mappings_for_framework(framework_code)

                    for mapping_data in mappings:
                        await self._upsert_skill_mapping(mapping_data)
                        result.mappings_synced += 1

                except httpx.HTTPStatusError as e:
                    # Skip 404 errors (framework may not have mappings)
                    if e.response.status_code == 404:
                        logger.debug(
                            "No skill mappings found for framework: %s", framework_code
                        )
                        continue
                    raise

            await self._db.commit()

            result.success = True
            result.completed_at = datetime.now(timezone.utc)

            logger.info(
                "Skill mapping sync completed: mappings=%d",
                result.mappings_synced,
            )

            return result

        except Exception as e:
            await self._db.rollback()
            result.error = str(e)
            result.completed_at = datetime.now(timezone.utc)
            logger.error("Skill mapping sync failed: %s", str(e))
            raise CurriculumSyncError(f"Skill mapping sync failed: {str(e)}") from e

    # =========================================================================
    # Skill API Fetch Methods
    # =========================================================================

    async def _fetch_taxonomies(self) -> list[dict]:
        """Fetch all skill taxonomies from Central Curriculum.

        Returns list of taxonomy codes/metadata without categories.
        """
        response = await self._client.get("/skills/taxonomies")
        response.raise_for_status()
        return response.json().get("data", [])

    async def _fetch_taxonomy_full(self, code: str) -> dict:
        """Fetch a single taxonomy with all categories.

        Args:
            code: Taxonomy code (e.g., "SUBJ-6-ENH").

        Returns:
            Taxonomy data with nested categories list.
        """
        response = await self._client.get(f"/skills/taxonomies/{code}/full")
        response.raise_for_status()
        return response.json().get("data", {})

    async def _fetch_skill_mappings_for_framework(self, framework_code: str) -> list[dict]:
        """Fetch subject-skill mappings for a specific framework.

        Args:
            framework_code: Framework code (e.g., "UK-NC-2014").

        Returns:
            List of mapping data for the framework.
        """
        response = await self._client.get(f"/skills/frameworks/{framework_code}/mappings")
        response.raise_for_status()
        return response.json().get("data", [])

    # =========================================================================
    # Skill Upsert Methods
    # =========================================================================

    async def _upsert_taxonomy(self, data: dict) -> SkillTaxonomy:
        """Upsert a skill taxonomy."""
        stmt = select(SkillTaxonomy).where(SkillTaxonomy.code == data["code"])
        result = await self._db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.name = data["name"]
            existing.slug = data["slug"]
            existing.description = data.get("description")
            existing.taxonomy_type = data.get("taxonomy_type", "hybrid")
            existing.author = data.get("author")
            existing.year_published = data.get("year_published")
            existing.age_range = data.get("age_range")
            existing.skill_count = data.get("skill_count", 0)
            existing.visualization_type = data.get("visualization_type", "hexagon")
            existing.balance_required = data.get("balance_required", False)
            existing.is_default = data.get("is_default", False)
            existing.is_official = data.get("is_official", False)
            existing.is_active = data.get("is_active", True)
            existing.order_index = data.get("order_index", 0)
            existing.extra_data = data.get("extra_data")
            return existing

        taxonomy = SkillTaxonomy(
            code=data["code"],
            name=data["name"],
            slug=data["slug"],
            description=data.get("description"),
            taxonomy_type=data.get("taxonomy_type", "hybrid"),
            author=data.get("author"),
            year_published=data.get("year_published"),
            age_range=data.get("age_range"),
            skill_count=data.get("skill_count", 0),
            visualization_type=data.get("visualization_type", "hexagon"),
            balance_required=data.get("balance_required", False),
            is_default=data.get("is_default", False),
            is_official=data.get("is_official", False),
            is_active=data.get("is_active", True),
            order_index=data.get("order_index", 0),
            extra_data=data.get("extra_data"),
        )
        self._db.add(taxonomy)
        await self._db.flush()
        return taxonomy

    async def _upsert_category(self, taxonomy_code: str, data: dict) -> SkillCategory:
        """Upsert a skill category."""
        stmt = select(SkillCategory).where(
            SkillCategory.taxonomy_code == taxonomy_code,
            SkillCategory.code == data["code"],
        )
        result = await self._db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.name = data["name"]
            existing.short_code = data["short_code"]
            existing.description = data.get("description")
            existing.color = data.get("color")
            existing.icon = data.get("icon")
            existing.age_appropriate_from = data.get("age_appropriate_from", 3)
            existing.age_appropriate_to = data.get("age_appropriate_to", 100)
            existing.order_index = data.get("order_index", 0)
            existing.is_active = data.get("is_active", True)
            existing.visual_themes = data.get("visual_themes")
            existing.extra_data = data.get("extra_data")
            return existing

        category = SkillCategory(
            taxonomy_code=taxonomy_code,
            code=data["code"],
            name=data["name"],
            short_code=data["short_code"],
            description=data.get("description"),
            color=data.get("color"),
            icon=data.get("icon"),
            age_appropriate_from=data.get("age_appropriate_from", 3),
            age_appropriate_to=data.get("age_appropriate_to", 100),
            order_index=data.get("order_index", 0),
            is_active=data.get("is_active", True),
            visual_themes=data.get("visual_themes"),
            extra_data=data.get("extra_data"),
        )
        self._db.add(category)
        await self._db.flush()
        return category

    async def _upsert_skill_mapping(self, data: dict) -> SubjectSkillMapping:
        """Upsert a subject-skill mapping."""
        stmt = select(SubjectSkillMapping).where(
            SubjectSkillMapping.framework_code == data["framework_code"],
            SubjectSkillMapping.subject_code == data["subject_code"],
            SubjectSkillMapping.taxonomy_code == data["taxonomy_code"],
            SubjectSkillMapping.skill_code == data["skill_code"],
        )
        result = await self._db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.impact_weight = Decimal(str(data.get("impact_weight", 1.0)))
            existing.is_active = data.get("is_active", True)
            return existing

        mapping = SubjectSkillMapping(
            framework_code=data["framework_code"],
            subject_code=data["subject_code"],
            taxonomy_code=data["taxonomy_code"],
            skill_code=data["skill_code"],
            impact_weight=Decimal(str(data.get("impact_weight", 1.0))),
            is_active=data.get("is_active", True),
        )
        self._db.add(mapping)
        await self._db.flush()
        return mapping

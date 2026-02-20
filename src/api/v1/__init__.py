# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""API v1 routes package.

This package contains all v1 API endpoint definitions.
Each module provides a FastAPI router for a specific domain.

Modules:
    auth: Authentication endpoints (login, logout, refresh).
    users: User management endpoints (CRUD, activate, suspend).
    schools: School management endpoints (CRUD, admin assignment).
    academic_years: Academic year management endpoints.
    classes: Class/section management endpoints.
    parent_relations: Parent-student relationship endpoints.
    practice: Practice session endpoints.
    practice_helper: Practice helper tutoring endpoints (when student needs help).
    learning: Tutoring conversation endpoints.
    learning_tutor: Proactive learning tutor endpoints (teach new topics).
    companion: Companion agent endpoints (check-in, mood, activities).
    games: Educational games endpoints (chess, connect4 with coaching).
    teacher: Teacher assistant endpoints (class monitoring, student analytics).
    parent: Parent portal endpoints (children, notes, summaries).
    analytics: Learning analytics endpoints.
    memory: Memory and progress endpoints.
    diagnostics: Learning diagnostics endpoints.
    curriculum: Curriculum data access and sync endpoints.
    provisioning: LMS provisioning endpoints (school, student).
    tenant: Tenant admin endpoints (overview, settings).
    content_creation: AI-powered H5P content creation endpoints.
    system: System administration endpoints (admin auth, tenant management).
    system_explainer: Public AI agent for explaining EduSynapseOS.
"""

from fastapi import APIRouter

from src.api.v1 import auth, users, practice, practice_helper, learning, learning_tutor, companion, games, teacher, parent, analytics, memory, diagnostics, curriculum, schools, academic_years, classes, parent_relations, provisioning, tenant, system_explainer, playground, activity, intelligence, content_creation
from src.api.v1.system import router as system_router

# Create the main v1 router
router = APIRouter(prefix="/api/v1")

# Include domain routers
router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
router.include_router(users.router, prefix="/users", tags=["Users"])
router.include_router(schools.router, prefix="/schools", tags=["Schools"])
router.include_router(academic_years.router, prefix="/academic-years", tags=["Academic Years"])
router.include_router(classes.router, prefix="/classes", tags=["Classes"])
router.include_router(parent_relations.router, prefix="/parent-relations", tags=["Parent Relations"])
router.include_router(practice.router, prefix="/practice", tags=["Practice"])
router.include_router(practice_helper.router, prefix="/practice-helper", tags=["Practice Helper"])
router.include_router(learning.router, prefix="/learning", tags=["Learning"])
router.include_router(learning_tutor.router, prefix="/learning-tutor", tags=["Learning Tutor"])
router.include_router(companion.router, prefix="/companion", tags=["Companion"])
router.include_router(games.router, prefix="/games", tags=["Games"])
router.include_router(teacher.router, prefix="/teacher", tags=["Teacher"])
router.include_router(parent.router, prefix="/parent", tags=["Parent"])
router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
router.include_router(memory.router, prefix="/memory", tags=["Memory"])
router.include_router(diagnostics.router, prefix="/diagnostics", tags=["Diagnostics"])
router.include_router(activity.router, prefix="/activity", tags=["Activity Stream"])
router.include_router(intelligence.router, prefix="/intelligence", tags=["AI Intelligence"])
router.include_router(curriculum.router, prefix="/curriculum", tags=["Curriculum"])
router.include_router(provisioning.router, prefix="/lms", tags=["LMS Integration"])
router.include_router(tenant.router, prefix="/tenant", tags=["Tenant Admin"])
router.include_router(content_creation.router, prefix="/content-creation", tags=["Content Creation"])

# Public routes (no authentication required)
router.include_router(system_explainer.router, prefix="/explain", tags=["System Explainer"])
router.include_router(playground.router, prefix="/playground", tags=["Playground"])

# System administration routes (no tenant context required)
router.include_router(system_router)

__all__ = ["router"]

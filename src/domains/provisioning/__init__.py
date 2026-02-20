# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Provisioning domain for LMS integration.

This package provides the ProvisioningService that enables LMS systems
to create/update school, class, and student data in a single atomic
operation using upsert semantics.

NOTE: Curriculum data (frameworks, stages, grades, subjects, units, topics)
is NOT created by this service. Curriculum data is synced from the Central
Curriculum service. This service only references existing curriculum data
when creating classes.
"""

from src.domains.provisioning.service import (
    ProvisioningService,
    ProvisioningError,
    GradeLevelNotFoundError,
    SchoolProvisioningError,
    StudentProvisioningError,
)

__all__ = [
    "ProvisioningService",
    "ProvisioningError",
    "GradeLevelNotFoundError",
    "SchoolProvisioningError",
    "StudentProvisioningError",
]

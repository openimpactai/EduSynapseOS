# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Telemetry infrastructure for EduSynapseOS.

This package provides OpenTelemetry setup and configuration
for distributed tracing across the application.
"""

from src.infrastructure.telemetry.setup import setup_telemetry

__all__ = ["setup_telemetry"]

# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""API Layer for EduSynapseOS.

This module provides the FastAPI application and all HTTP endpoints.
"""

from src.api.app import create_app

__all__ = ["create_app"]

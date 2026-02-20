# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Navigation tools for directing students to activities.

This category contains tools that create navigation actions:
- navigate: Navigate student to activity, game, learning, or review page

These tools are used after gathering all required parameters to
direct the student to the appropriate page or activity.
"""

from src.tools.navigation.navigate import NavigateTool

__all__ = [
    "NavigateTool",
]

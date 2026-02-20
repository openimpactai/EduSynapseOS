# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Activity tools for querying games and learning activities.

This category contains tools that retrieve available activities:
- get_games: Get available games filtered by grade level
- get_activities: Get learning activities filtered by category and difficulty

These tools are used when suggesting activities or games to students.
"""

from src.tools.activities.get_activities import GetActivitiesTool
from src.tools.activities.get_games import GetGamesTool

__all__ = [
    "GetGamesTool",
    "GetActivitiesTool",
]

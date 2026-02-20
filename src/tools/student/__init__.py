# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Student tools for querying student-specific data.

This category contains tools that retrieve personalized student information:
- get_student_context: Get student personalization from memory layers
- get_parent_notes: Get active notes from parents
- get_review_schedule: Get spaced repetition review schedule
- get_my_mastery: Get student's mastery levels across topics
- get_my_weaknesses: Identify topics where student is struggling

These tools help agents personalize interactions based on student history,
preferences, and parent inputs.
"""

from src.tools.student.get_my_mastery import GetMyMasteryTool
from src.tools.student.get_my_weaknesses import GetMyWeaknessesTool
from src.tools.student.get_parent_notes import GetParentNotesTool
from src.tools.student.get_review_schedule import GetReviewScheduleTool
from src.tools.student.get_student_context import GetStudentContextTool

__all__ = [
    "GetStudentContextTool",
    "GetParentNotesTool",
    "GetReviewScheduleTool",
    "GetMyMasteryTool",
    "GetMyWeaknessesTool",
]

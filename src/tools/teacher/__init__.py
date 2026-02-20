# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Teacher tools for the teacher companion agent.

This module provides tools for teachers to:
- View their assigned classes
- Monitor student progress and mastery
- View class analytics
- Identify struggling students
- Manage alerts
- View emotional history

All tools verify that the teacher has access to the requested data
through their class assignments.
"""

from src.tools.teacher.get_alerts import GetAlertsTool
from src.tools.teacher.get_class_analytics import GetClassAnalyticsTool
from src.tools.teacher.get_class_students import GetClassStudentsTool
from src.tools.teacher.get_emotional_history import GetEmotionalHistoryTool
from src.tools.teacher.get_my_classes import GetMyClassesTool
from src.tools.teacher.get_struggling_students import GetStrugglingStudentsTool
from src.tools.teacher.get_student_mastery import GetStudentMasteryTool
from src.tools.teacher.get_student_notes import GetStudentNotesTool
from src.tools.teacher.get_student_progress import GetStudentProgressTool
from src.tools.teacher.get_topic_performance import GetTopicPerformanceTool

__all__ = [
    "GetAlertsTool",
    "GetClassAnalyticsTool",
    "GetClassStudentsTool",
    "GetEmotionalHistoryTool",
    "GetMyClassesTool",
    "GetStrugglingStudentsTool",
    "GetStudentMasteryTool",
    "GetStudentNotesTool",
    "GetStudentProgressTool",
    "GetTopicPerformanceTool",
]

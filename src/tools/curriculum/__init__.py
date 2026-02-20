# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Curriculum tools for querying subjects and topics.

This category contains tools that interact with the curriculum database:
- get_subjects: Get available subjects for a grade level
- get_topics: Get topics for a specific subject

These tools are used by any agent that needs to show curriculum options
to students, such as Companion for practice selection or Tutor for
topic-specific help.
"""

from src.tools.curriculum.get_subjects import GetSubjectsTool
from src.tools.curriculum.get_topics import GetTopicsTool

__all__ = [
    "GetSubjectsTool",
    "GetTopicsTool",
]

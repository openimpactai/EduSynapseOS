# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Learning Tutor domain.

This domain provides proactive teaching when students want to learn
new concepts. It supports multiple entry points:

- Companion handoff: When student asks to learn something
- Practice help: "I need to learn this" button in practice
- Direct access: Learning menu in the app
- LMS deep link: External LMS integration
- Spaced repetition: Review trigger
- Weakness suggestion: Based on mastery gaps

The tutor adapts its teaching approach based on:
- Student's emotional state
- Topic mastery level
- Subject matter
- Educational theory recommendations
"""

from src.domains.learning.service import LearningService

__all__ = [
    "LearningService",
]

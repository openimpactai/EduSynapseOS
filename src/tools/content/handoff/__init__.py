# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Content Creation Handoff Tools.

Tools for delegating content generation to specialized agents.
"""

from src.tools.content.handoff.quiz_generator import HandoffToQuizGeneratorTool
from src.tools.content.handoff.vocabulary_generator import HandoffToVocabularyGeneratorTool

__all__ = [
    "HandoffToQuizGeneratorTool",
    "HandoffToVocabularyGeneratorTool",
]

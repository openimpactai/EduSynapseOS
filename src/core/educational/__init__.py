# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Educational theories and pedagogical framework.

This module implements 7 educational theories that guide the AI tutoring system:
- ZPD (Zone of Proximal Development): Optimal difficulty targeting
- Bloom's Taxonomy: Cognitive complexity levels
- VARK: Learning style preferences
- Scaffolding: Adaptive support levels
- Mastery Learning: Progression thresholds
- Socratic Method: Questioning strategies
- Spaced Repetition: FSRS-based review scheduling

The TheoryOrchestrator combines recommendations from all theories
to provide unified pedagogical guidance for each learning interaction.
"""

from src.core.educational.orchestrator import TheoryOrchestrator
from src.core.educational.theories.base import (
    BaseTheory,
    StudentContext,
    TheoryRecommendation,
)

__all__ = [
    "BaseTheory",
    "StudentContext",
    "TheoryOrchestrator",
    "TheoryRecommendation",
]

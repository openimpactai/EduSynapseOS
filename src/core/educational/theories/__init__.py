# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Educational theories package.

Implements 7 pedagogical theories:
- ZPD: Zone of Proximal Development
- Bloom: Bloom's Taxonomy cognitive levels
- VARK: Visual/Auditory/Read-Write/Kinesthetic learning styles
- Scaffolding: Adaptive support and hints
- Mastery: Mastery-based learning progression
- Socratic: Socratic questioning method
- SpacedRepetition: FSRS-based review scheduling
"""

from src.core.educational.theories.base import (
    BaseTheory,
    StudentContext,
    TheoryRecommendation,
)
from src.core.educational.theories.bloom import BloomTheory
from src.core.educational.theories.mastery import MasteryTheory
from src.core.educational.theories.scaffolding import ScaffoldingTheory
from src.core.educational.theories.socratic import SocraticTheory
from src.core.educational.theories.spaced_repetition import SpacedRepetitionTheory
from src.core.educational.theories.vark import VARKTheory
from src.core.educational.theories.zpd import ZPDTheory

__all__ = [
    "BaseTheory",
    "BloomTheory",
    "MasteryTheory",
    "ScaffoldingTheory",
    "SocraticTheory",
    "SpacedRepetitionTheory",
    "StudentContext",
    "TheoryRecommendation",
    "VARKTheory",
    "ZPDTheory",
]

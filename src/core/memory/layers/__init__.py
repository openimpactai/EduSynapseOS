# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Memory layer implementations.

This package contains the four memory layer services:
- EpisodicMemoryLayer: Session-based learning events with vector search
- SemanticMemoryLayer: Knowledge state and mastery tracking
- ProceduralMemoryLayer: Learning strategies and patterns
- AssociativeMemoryLayer: Interests and concept connections with vector search

Each layer provides CRUD operations and specialized queries for its domain.
Episodic and Associative layers use Qdrant for semantic similarity search.
"""

from src.core.memory.layers.associative import AssociativeMemoryLayer
from src.core.memory.layers.episodic import EpisodicMemoryLayer
from src.core.memory.layers.procedural import ProceduralMemoryLayer
from src.core.memory.layers.semantic import SemanticMemoryLayer

__all__ = [
    "AssociativeMemoryLayer",
    "EpisodicMemoryLayer",
    "ProceduralMemoryLayer",
    "SemanticMemoryLayer",
]

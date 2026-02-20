# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Memory system for personalized learning.

This package implements a 4-layer memory architecture:
- Episodic: Session-based learning events (uses Qdrant vectors)
- Semantic: Knowledge state and mastery levels (DB only)
- Procedural: Learning strategies and patterns (DB only)
- Associative: Interests and concept connections (uses Qdrant vectors)

The memory system provides:
- Unified memory context retrieval for AI tutors
- Semantic search across episodic and associative memories
- Learning pattern analysis
- Mastery tracking with FSRS spaced repetition

Example:
    from src.core.memory import MemoryManager

    # Initialize manager
    manager = MemoryManager(
        tenant_db_manager=tenant_db,
        embedding_service=embedding_service,
        qdrant_client=qdrant,
    )

    # Get full context for AI tutor
    context = await manager.get_full_context(
        tenant_code="acme",
        student_id=student_uuid,
    )

    # Store new episodic memory
    await manager.episodic.store(
        tenant_code="acme",
        student_id=student_uuid,
        event_type="breakthrough",
        summary="Student understood fractions using pizza analogy",
        importance=0.9,
    )
"""

from src.core.memory.layers.associative import AssociativeMemoryLayer
from src.core.memory.layers.episodic import EpisodicMemoryLayer
from src.core.memory.layers.procedural import ProceduralMemoryLayer
from src.core.memory.layers.semantic import SemanticMemoryLayer
from src.core.memory.manager import MemoryManager

__all__ = [
    "AssociativeMemoryLayer",
    "EpisodicMemoryLayer",
    "MemoryManager",
    "ProceduralMemoryLayer",
    "SemanticMemoryLayer",
]

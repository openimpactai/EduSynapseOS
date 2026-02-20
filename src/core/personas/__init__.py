# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Persona system for EduSynapseOS.

This module provides the persona system which defines HOW AI agents communicate
with students. Personas include identity, voice characteristics, response templates,
and behavioral settings for different teaching styles.

Available personas (loaded from config/personas/):
- coach: Motivational performance coach
- mentor: Wise and experienced mentor
- tutor: Patient and supportive tutor (default)
- friend: Friendly study companion
- socratic: Socratic questioning guide

Usage:
    from src.core.personas import PersonaManager, Persona

    # Get the singleton manager
    manager = PersonaManager()

    # Get a specific persona
    persona = manager.get_persona("coach")

    # Get persona for a user based on preferences
    persona = manager.get_persona_for_user(
        user_type="young_learner",
        preferred_persona_id="friend",
    )

    # Generate system prompt segment
    prompt_segment = persona.get_system_prompt_segment()

    # Format a template response
    response = persona.format_response("on_correct")
"""

from src.core.personas.loader import (
    PersonaLoadError,
    load_all_personas,
    load_persona,
    validate_persona_yaml,
)
from src.core.personas.manager import (
    PersonaManager,
    PersonaNotFoundError,
    get_persona_manager,
    reset_persona_manager,
)
from src.core.personas.models import (
    EmojiUsage,
    Formality,
    Persona,
    PersonaBehavior,
    PersonaIdentity,
    PersonaTemplates,
    PersonaVoice,
    Tone,
)

__all__ = [
    # Models
    "Persona",
    "PersonaIdentity",
    "PersonaVoice",
    "PersonaTemplates",
    "PersonaBehavior",
    # Enums
    "Tone",
    "Formality",
    "EmojiUsage",
    # Loader
    "load_persona",
    "load_all_personas",
    "validate_persona_yaml",
    "PersonaLoadError",
    # Manager
    "PersonaManager",
    "PersonaNotFoundError",
    "get_persona_manager",
    "reset_persona_manager",
]

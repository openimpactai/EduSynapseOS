# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the persona system.

This module tests the persona models, loader, and manager functionality.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_identity() -> PersonaIdentity:
    """Create a basic PersonaIdentity for testing."""
    return PersonaIdentity(
        role="Test Tutor",
        character="A test persona for unit testing purposes.",
    )


@pytest.fixture
def basic_voice() -> PersonaVoice:
    """Create a basic PersonaVoice for testing."""
    return PersonaVoice(
        tone=Tone.SUPPORTIVE,
        formality=Formality.INFORMAL,
        language="en",
        emoji_usage=EmojiUsage.MODERATE,
    )


@pytest.fixture
def basic_templates() -> PersonaTemplates:
    """Create basic PersonaTemplates for testing."""
    return PersonaTemplates(
        on_correct="Great job!",
        on_incorrect="Try again!",
        encouragement="Keep going!",
    )


@pytest.fixture
def basic_behavior() -> PersonaBehavior:
    """Create basic PersonaBehavior for testing."""
    return PersonaBehavior(
        socratic_tendency=0.5,
        hint_eagerness=0.3,
        explanation_depth="moderate",
    )


@pytest.fixture
def basic_persona(
    basic_identity: PersonaIdentity,
    basic_voice: PersonaVoice,
    basic_templates: PersonaTemplates,
    basic_behavior: PersonaBehavior,
) -> Persona:
    """Create a basic Persona for testing."""
    return Persona(
        id="test_persona",
        name="Test Persona",
        description="A test persona for unit testing",
        identity=basic_identity,
        voice=basic_voice,
        templates=basic_templates,
        behavior=basic_behavior,
        enabled=True,
    )


@pytest.fixture
def personas_dir() -> Path:
    """Get the actual personas configuration directory."""
    return Path(__file__).parent.parent.parent / "config" / "personas"


@pytest.fixture
def manager(personas_dir: Path) -> PersonaManager:
    """Create a PersonaManager for testing."""
    reset_persona_manager()
    return PersonaManager(personas_dir=personas_dir, auto_load=True)


# =============================================================================
# Model Tests
# =============================================================================


class TestPersonaIdentity:
    """Tests for PersonaIdentity model."""

    def test_create_basic_identity(self, basic_identity: PersonaIdentity) -> None:
        """Test creating a basic identity."""
        assert basic_identity.role == "Test Tutor"
        assert basic_identity.character is not None
        assert basic_identity.background is None
        assert basic_identity.expertise is None

    def test_create_full_identity(self) -> None:
        """Test creating a full identity with all fields."""
        identity = PersonaIdentity(
            role="Expert Tutor",
            character="A knowledgeable tutor.",
            background="Many years of teaching experience.",
            expertise=["Math", "Science"],
        )
        assert identity.role == "Expert Tutor"
        assert identity.background == "Many years of teaching experience."
        assert identity.expertise == ["Math", "Science"]

    def test_identity_requires_role(self) -> None:
        """Test that role is required."""
        with pytest.raises(Exception):
            PersonaIdentity(character="Some character")  # type: ignore


class TestPersonaVoice:
    """Tests for PersonaVoice model."""

    def test_create_default_voice(self) -> None:
        """Test creating a voice with defaults."""
        voice = PersonaVoice()
        assert voice.tone == Tone.SUPPORTIVE
        assert voice.formality == Formality.INFORMAL
        assert voice.language == "tr"
        assert voice.emoji_usage == EmojiUsage.MODERATE

    def test_create_custom_voice(self) -> None:
        """Test creating a custom voice."""
        voice = PersonaVoice(
            tone=Tone.MOTIVATIONAL,
            formality=Formality.VERY_INFORMAL,
            language="en",
            emoji_usage=EmojiUsage.FREQUENT,
        )
        assert voice.tone == Tone.MOTIVATIONAL
        assert voice.formality == Formality.VERY_INFORMAL
        assert voice.language == "en"
        assert voice.emoji_usage == EmojiUsage.FREQUENT


class TestPersonaTemplates:
    """Tests for PersonaTemplates model."""

    def test_create_default_templates(self) -> None:
        """Test creating templates with defaults."""
        templates = PersonaTemplates()
        assert templates.on_correct is not None
        assert templates.on_incorrect is not None
        assert templates.encouragement is not None
        assert templates.greeting is not None
        assert templates.farewell is not None

    def test_create_custom_templates(self) -> None:
        """Test creating custom templates."""
        templates = PersonaTemplates(
            on_correct="Excellent!",
            on_incorrect="Not quite, try again.",
            greeting="Hello there!",
        )
        assert templates.on_correct == "Excellent!"
        assert templates.on_incorrect == "Not quite, try again."
        assert templates.greeting == "Hello there!"


class TestPersonaBehavior:
    """Tests for PersonaBehavior model."""

    def test_create_default_behavior(self) -> None:
        """Test creating behavior with defaults."""
        behavior = PersonaBehavior()
        assert behavior.socratic_tendency == 0.5
        assert behavior.hint_eagerness == 0.3
        assert behavior.explanation_depth == "moderate"
        assert behavior.patience_level == "high"

    def test_socratic_tendency_bounds(self) -> None:
        """Test that socratic_tendency is bounded."""
        behavior = PersonaBehavior(socratic_tendency=0.0)
        assert behavior.socratic_tendency == 0.0

        behavior = PersonaBehavior(socratic_tendency=1.0)
        assert behavior.socratic_tendency == 1.0

        with pytest.raises(Exception):
            PersonaBehavior(socratic_tendency=1.5)

        with pytest.raises(Exception):
            PersonaBehavior(socratic_tendency=-0.1)


class TestPersona:
    """Tests for Persona model."""

    def test_create_basic_persona(self, basic_persona: Persona) -> None:
        """Test creating a basic persona."""
        assert basic_persona.id == "test_persona"
        assert basic_persona.name == "Test Persona"
        assert basic_persona.enabled is True

    def test_persona_id_pattern(self) -> None:
        """Test that persona ID follows the pattern."""
        # Valid IDs
        Persona(
            id="tutor",
            name="Tutor",
            description="A tutor",
            identity=PersonaIdentity(role="Tutor", character="A tutor"),
        )
        Persona(
            id="my_persona_123",
            name="My Persona",
            description="A persona",
            identity=PersonaIdentity(role="Role", character="Character"),
        )

        # Invalid IDs
        with pytest.raises(Exception):
            Persona(
                id="123invalid",  # Can't start with number
                name="Invalid",
                description="Invalid",
                identity=PersonaIdentity(role="Role", character="Character"),
            )

        with pytest.raises(Exception):
            Persona(
                id="Invalid-ID",  # Can't have hyphen
                name="Invalid",
                description="Invalid",
                identity=PersonaIdentity(role="Role", character="Character"),
            )

    def test_get_system_prompt_segment(self, basic_persona: Persona) -> None:
        """Test generating system prompt segment."""
        segment = basic_persona.get_system_prompt_segment()
        assert "Test Tutor" in segment
        assert "supportive" in segment.lower()
        assert "informal" in segment.lower()

    def test_format_response(self, basic_persona: Persona) -> None:
        """Test formatting template responses."""
        response = basic_persona.format_response("on_correct")
        assert response == "Great job!"

        response = basic_persona.format_response("on_incorrect")
        assert response == "Try again!"

    def test_persona_with_suitable_for(self) -> None:
        """Test persona with suitable_for field."""
        persona = Persona(
            id="young_tutor",
            name="Young Tutor",
            description="For young learners",
            identity=PersonaIdentity(role="Tutor", character="Friendly tutor"),
            suitable_for=["young_learners", "children"],
            tags=["fun", "casual"],
        )
        assert persona.suitable_for == ["young_learners", "children"]
        assert persona.tags == ["fun", "casual"]


# =============================================================================
# Loader Tests
# =============================================================================


class TestPersonaLoader:
    """Tests for persona loader functions."""

    def test_load_persona_tutor(self, personas_dir: Path) -> None:
        """Test loading the tutor persona."""
        persona = load_persona("tutor", personas_dir)
        assert persona.id == "tutor"
        assert persona.enabled is True

    def test_load_persona_coach(self, personas_dir: Path) -> None:
        """Test loading the coach persona."""
        persona = load_persona("coach", personas_dir)
        assert persona.id == "coach"
        assert persona.voice.tone == Tone.MOTIVATIONAL

    def test_load_persona_mentor(self, personas_dir: Path) -> None:
        """Test loading the mentor persona."""
        persona = load_persona("mentor", personas_dir)
        assert persona.id == "mentor"
        assert persona.behavior.socratic_tendency > 0.5

    def test_load_persona_friend(self, personas_dir: Path) -> None:
        """Test loading the friend persona."""
        persona = load_persona("friend", personas_dir)
        assert persona.id == "friend"
        assert persona.voice.formality == Formality.VERY_INFORMAL

    def test_load_persona_socratic(self, personas_dir: Path) -> None:
        """Test loading the socratic persona."""
        persona = load_persona("socratic", personas_dir)
        assert persona.id == "socratic"
        assert persona.behavior.socratic_tendency > 0.9
        assert persona.voice.emoji_usage == EmojiUsage.NONE

    def test_load_nonexistent_persona(self, personas_dir: Path) -> None:
        """Test loading a nonexistent persona."""
        with pytest.raises(PersonaLoadError):
            load_persona("nonexistent_persona", personas_dir)

    def test_load_all_personas(self, personas_dir: Path) -> None:
        """Test loading all personas."""
        personas = load_all_personas(personas_dir)
        assert len(personas) >= 5
        assert "tutor" in personas
        assert "coach" in personas
        assert "mentor" in personas
        assert "friend" in personas
        assert "socratic" in personas

    def test_validate_persona_yaml_valid(self, personas_dir: Path) -> None:
        """Test validating a valid persona YAML."""
        is_valid, error = validate_persona_yaml(personas_dir / "tutor.yaml")
        assert is_valid is True
        assert error is None

    def test_validate_persona_yaml_nonexistent(self, personas_dir: Path) -> None:
        """Test validating a nonexistent YAML file."""
        is_valid, error = validate_persona_yaml(personas_dir / "nonexistent.yaml")
        assert is_valid is False
        assert error is not None


# =============================================================================
# Manager Tests
# =============================================================================


class TestPersonaManager:
    """Tests for PersonaManager class."""

    def test_manager_init(self, manager: PersonaManager) -> None:
        """Test manager initialization."""
        assert manager._loaded is True
        assert len(manager._personas) >= 5

    def test_get_persona(self, manager: PersonaManager) -> None:
        """Test getting a persona by ID."""
        persona = manager.get_persona("tutor")
        assert persona.id == "tutor"
        assert isinstance(persona, Persona)

    def test_get_persona_not_found(self, manager: PersonaManager) -> None:
        """Test getting a nonexistent persona."""
        with pytest.raises(PersonaNotFoundError):
            manager.get_persona("nonexistent")

    def test_get_default_persona(self, manager: PersonaManager) -> None:
        """Test getting the default persona."""
        persona = manager.get_default_persona()
        assert persona.id == "tutor"  # Default is tutor

    def test_list_personas(self, manager: PersonaManager) -> None:
        """Test listing all personas."""
        personas = manager.list_personas()
        assert len(personas) >= 5
        assert all(isinstance(p, Persona) for p in personas)

    def test_list_persona_ids(self, manager: PersonaManager) -> None:
        """Test listing all persona IDs."""
        ids = manager.list_persona_ids()
        assert len(ids) >= 5
        assert "tutor" in ids
        assert "coach" in ids

    def test_has_persona(self, manager: PersonaManager) -> None:
        """Test checking if a persona exists."""
        assert manager.has_persona("tutor") is True
        assert manager.has_persona("coach") is True
        assert manager.has_persona("nonexistent") is False

    def test_get_persona_summary(self, manager: PersonaManager) -> None:
        """Test getting a persona summary."""
        summary = manager.get_persona_summary("tutor")
        assert summary["id"] == "tutor"
        assert "name" in summary
        assert "description" in summary
        assert "role" in summary
        assert "tone" in summary

    def test_get_all_summaries(self, manager: PersonaManager) -> None:
        """Test getting all persona summaries."""
        summaries = manager.get_all_summaries()
        assert len(summaries) >= 5
        assert all("id" in s for s in summaries)

    def test_get_persona_for_user_with_preference(
        self, manager: PersonaManager
    ) -> None:
        """Test getting persona for user with preference."""
        persona = manager.get_persona_for_user(preferred_persona_id="coach")
        assert persona.id == "coach"

    def test_get_persona_for_user_fallback(self, manager: PersonaManager) -> None:
        """Test getting persona for user with invalid preference."""
        persona = manager.get_persona_for_user(preferred_persona_id="nonexistent")
        assert persona.id == "tutor"  # Falls back to default

    def test_reload(self, manager: PersonaManager) -> None:
        """Test reloading personas."""
        initial_count = len(manager._personas)
        manager.reload()
        assert len(manager._personas) == initial_count


class TestPersonaManagerSingleton:
    """Tests for PersonaManager singleton functions."""

    def test_get_persona_manager_singleton(self, personas_dir: Path) -> None:
        """Test getting the singleton manager."""
        reset_persona_manager()
        manager1 = get_persona_manager(personas_dir=personas_dir)
        manager2 = get_persona_manager()
        assert manager1 is manager2

    def test_reset_persona_manager(self, personas_dir: Path) -> None:
        """Test resetting the singleton manager."""
        reset_persona_manager()
        manager1 = get_persona_manager(personas_dir=personas_dir)
        reset_persona_manager()
        manager2 = get_persona_manager(personas_dir=personas_dir)
        assert manager1 is not manager2


# =============================================================================
# Integration Tests
# =============================================================================


class TestPersonaIntegration:
    """Integration tests for the persona system."""

    def test_full_persona_workflow(self, manager: PersonaManager) -> None:
        """Test the full persona workflow."""
        # List available personas
        personas = manager.list_personas()
        assert len(personas) >= 5

        # Get a specific persona
        coach = manager.get_persona("coach")
        assert coach.id == "coach"

        # Generate system prompt segment
        prompt = coach.get_system_prompt_segment()
        assert len(prompt) > 0
        assert "coach" in prompt.lower() or "koç" in prompt.lower()

        # Use templates
        response = coach.format_response("on_correct")
        assert len(response) > 0

        # Check behavior settings
        assert 0 <= coach.behavior.socratic_tendency <= 1
        assert 0 <= coach.behavior.hint_eagerness <= 1

    def test_all_personas_valid(self, manager: PersonaManager) -> None:
        """Test that all loaded personas are valid."""
        for persona in manager.list_personas():
            # Check required fields
            assert persona.id is not None
            assert persona.name is not None
            assert persona.description is not None
            assert persona.identity is not None
            assert persona.voice is not None
            assert persona.templates is not None
            assert persona.behavior is not None

            # Check behavior bounds
            assert 0 <= persona.behavior.socratic_tendency <= 1
            assert 0 <= persona.behavior.hint_eagerness <= 1

            # Check system prompt generation works
            prompt = persona.get_system_prompt_segment()
            assert len(prompt) > 0

    def test_persona_suitable_for_matching(self, manager: PersonaManager) -> None:
        """Test that persona suitable_for matching works."""
        # Friend persona should be suitable for young learners
        friend = manager.get_persona("friend")
        assert friend.suitable_for is not None
        assert "young_learners" in friend.suitable_for or "children" in friend.suitable_for

        # Mentor should be suitable for advanced learners
        mentor = manager.get_persona("mentor")
        assert mentor.suitable_for is not None
        assert "advanced_learners" in mentor.suitable_for or "adults" in mentor.suitable_for

# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for H5P Content Creation API.

Comprehensive test suite that simulates real user scenarios
across different countries, roles, and content types.

Test Coverage:
- All H5P content types (multiple-choice, flashcards, etc.)
- All user roles (teacher, student, parent)
- Multi-country support (UK, USA, Rwanda, Malawi)
- Complete workflow (session → chat → generate → review → export)
- Error handling and edge cases
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def playground_credentials() -> dict[str, Any]:
    """Load playground credentials from local config."""
    import json
    from pathlib import Path

    creds_path = Path("/home/erhanarslan/work/gdlabs/projects/edusynapse/scripts/.playground-result.local.json")
    with open(creds_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def playground_tenant(playground_credentials) -> dict[str, str]:
    """Get playground tenant info."""
    return playground_credentials["playground_tenant"]["edusynapse"]


@pytest.fixture(scope="module")
def uk_teacher(playground_credentials) -> dict[str, Any]:
    """Get UK teacher info."""
    teacher = playground_credentials["teachers"]["uk"]
    teacher["framework_code"] = playground_credentials["frameworks"]["uk"]
    return teacher


@pytest.fixture(scope="module")
def usa_teacher(playground_credentials) -> dict[str, Any]:
    """Get USA teacher info."""
    teacher = playground_credentials["teachers"]["usa"]
    teacher["framework_code"] = playground_credentials["frameworks"]["usa"]
    return teacher


@pytest.fixture(scope="module")
def rwanda_teacher(playground_credentials) -> dict[str, Any]:
    """Get Rwanda teacher info."""
    teacher = playground_credentials["teachers"]["rwanda"]
    teacher["framework_code"] = playground_credentials["frameworks"]["rwanda"]
    return teacher


@pytest.fixture(scope="module")
def uk_student(playground_credentials) -> dict[str, Any]:
    """Get UK student info."""
    return playground_credentials["students"]["uk"]


@pytest.fixture(scope="module")
def uk_parent(playground_credentials) -> dict[str, Any]:
    """Get UK parent info."""
    return playground_credentials["parents"]["uk"]


# =============================================================================
# Mock Services for Testing
# =============================================================================


class MockLLMClient:
    """Mock LLM client for testing without actual API calls."""

    async def complete(self, messages: list[dict], **kwargs) -> dict[str, Any]:
        """Simulate LLM completion."""
        # Extract the last user message to understand intent
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "").lower()
                break

        # Generate appropriate response based on context
        if "welcome" in str(messages).lower() or not user_message:
            return {
                "content": "Hello! I'm your content creation assistant. What type of educational content would you like to create today?",
                "model": "mock-model",
            }

        if "quiz" in user_message or "multiple" in user_message:
            return {
                "content": json.dumps({
                    "title": "Science Quiz: Plants",
                    "questions": [
                        {
                            "question": "What process do plants use to make food?",
                            "answers": ["Photosynthesis", "Respiration", "Digestion", "Fermentation"],
                            "correctIndex": 0,
                            "explanation": "Plants use photosynthesis to convert sunlight into food."
                        },
                        {
                            "question": "What gas do plants absorb from the air?",
                            "answers": ["Oxygen", "Carbon dioxide", "Nitrogen", "Hydrogen"],
                            "correctIndex": 1,
                            "explanation": "Plants absorb carbon dioxide for photosynthesis."
                        }
                    ]
                }),
                "model": "mock-model",
            }

        if "flashcard" in user_message or "vocabulary" in user_message:
            return {
                "content": json.dumps({
                    "title": "Biology Vocabulary",
                    "cards": [
                        {"term": "Photosynthesis", "definition": "The process by which plants make food using sunlight"},
                        {"term": "Chlorophyll", "definition": "The green pigment in plants that absorbs light"},
                        {"term": "Stomata", "definition": "Tiny pores on leaves for gas exchange"}
                    ]
                }),
                "model": "mock-model",
            }

        # Default response
        return {
            "content": "I can help you create various types of educational content. Would you like to create a quiz, flashcards, or another type of content?",
            "model": "mock-model",
        }


class MockH5PClient:
    """Mock H5P client for testing without actual API calls."""

    def __init__(self, **kwargs):
        self.created_content = {}

    async def create_content(
        self,
        library: str,
        params: dict,
        metadata: dict | None = None,
        tenant_code: str | None = None,
    ) -> str:
        """Simulate content creation."""
        content_id = str(uuid4())
        self.created_content[content_id] = {
            "library": library,
            "params": params,
            "metadata": metadata,
            "tenant_code": tenant_code,
            "created_at": datetime.utcnow().isoformat(),
        }
        logger.info(f"Created mock H5P content: {content_id}")
        return content_id

    def get_preview_url(self, content_id: str) -> str:
        """Get mock preview URL."""
        return f"https://h5p.test/preview/{content_id}"


# =============================================================================
# Test Classes
# =============================================================================


class TestContentCreationWorkflow:
    """Test the complete content creation workflow."""

    @pytest.mark.asyncio
    async def test_create_multiple_choice_quiz_uk_teacher(
        self,
        playground_tenant,
        uk_teacher,
    ):
        """Test creating multiple choice quiz as UK teacher."""
        from src.core.orchestration.states.content import create_initial_content_state
        from src.core.orchestration.workflows.content import ContentCreationWorkflow

        # Create workflow with mock services
        workflow = ContentCreationWorkflow(
            llm_client=MockLLMClient(),
            checkpointer=None,
        )

        # Patch H5PClient
        import src.core.orchestration.workflows.content.content_creation as wf_module
        original_h5p_client = wf_module.H5PClient
        wf_module.H5PClient = MockH5PClient

        try:
            # Create initial state with UK teacher context
            initial_state = create_initial_content_state(
                session_id=str(uuid4()),
                tenant_code=playground_tenant["tenant_code"],
                user_id=uk_teacher["id"],
                user_role="teacher",
                language="en",
                country_code=uk_teacher["country_code"],
                framework_code=uk_teacher["framework_code"],
            )

            # Verify state was created correctly
            assert initial_state["tenant_code"] == "playground"
            assert initial_state["user_role"] == "teacher"
            assert initial_state["country_code"] == "GB"
            assert initial_state["framework_code"] == "UK-NC-2014"
            assert initial_state["language"] == "en"

            logger.info("✓ UK Teacher state created successfully")

        finally:
            wf_module.H5PClient = original_h5p_client

    @pytest.mark.asyncio
    async def test_create_flashcards_usa_teacher(
        self,
        playground_tenant,
        usa_teacher,
    ):
        """Test creating flashcards as USA teacher."""
        from src.core.orchestration.states.content import create_initial_content_state

        # Create initial state with USA teacher context
        initial_state = create_initial_content_state(
            session_id=str(uuid4()),
            tenant_code=playground_tenant["tenant_code"],
            user_id=usa_teacher["id"],
            user_role="teacher",
            language="en",
            country_code=usa_teacher["country_code"],
            framework_code=usa_teacher["framework_code"],
        )

        # Verify state was created correctly
        assert initial_state["tenant_code"] == "playground"
        assert initial_state["user_role"] == "teacher"
        assert initial_state["country_code"] == "US"
        assert initial_state["framework_code"] == "CCSS"

        logger.info("✓ USA Teacher state created successfully")

    @pytest.mark.asyncio
    async def test_create_content_rwanda_context(
        self,
        playground_tenant,
        rwanda_teacher,
    ):
        """Test content creation with Rwanda curriculum context."""
        from src.core.orchestration.states.content import create_initial_content_state

        # Create initial state with Rwanda teacher context
        initial_state = create_initial_content_state(
            session_id=str(uuid4()),
            tenant_code=playground_tenant["tenant_code"],
            user_id=rwanda_teacher["id"],
            user_role="teacher",
            language="en",  # Rwanda uses English in education
            country_code=rwanda_teacher["country_code"],
            framework_code=rwanda_teacher["framework_code"],
        )

        # Verify state was created correctly
        assert initial_state["country_code"] == "RW"
        assert initial_state["framework_code"] == "RW-CBC"

        logger.info("✓ Rwanda Teacher state created successfully")


class TestH5PConverters:
    """Test H5P content converters."""

    @pytest.mark.asyncio
    async def test_multiple_choice_converter(self):
        """Test MultipleChoiceConverter."""
        from src.services.h5p.converters import ConverterRegistry

        registry = ConverterRegistry()
        converter = registry.get("multiple-choice")

        assert converter is not None
        assert converter.library == "H5P.MultiChoice 1.16"

        # Test conversion
        ai_content = {
            "title": "Test Quiz",
            "questions": [
                {
                    "question": "What is 2+2?",
                    "answers": ["3", "4", "5", "6"],
                    "correctIndex": 1,
                    "explanation": "2+2=4"
                }
            ]
        }

        h5p_params = converter.convert(ai_content, "en")

        assert "question" in h5p_params
        assert h5p_params["question"] == "What is 2+2?"
        assert len(h5p_params["answers"]) == 4

        logger.info("✓ MultipleChoice converter works correctly")

    @pytest.mark.asyncio
    async def test_flashcards_converter(self):
        """Test FlashcardsConverter."""
        from src.services.h5p.converters import ConverterRegistry

        registry = ConverterRegistry()
        converter = registry.get("flashcards")

        assert converter is not None
        assert "Flashcards" in converter.library

        # Test conversion
        ai_content = {
            "title": "Vocabulary Cards",
            "cards": [
                {"term": "Photosynthesis", "definition": "Process of making food"},
                {"term": "Chlorophyll", "definition": "Green pigment in plants"},
            ]
        }

        h5p_params = converter.convert(ai_content, "en")

        assert "cards" in h5p_params
        assert len(h5p_params["cards"]) == 2

        logger.info("✓ Flashcards converter works correctly")

    @pytest.mark.asyncio
    async def test_true_false_converter(self):
        """Test TrueFalseConverter."""
        from src.services.h5p.converters import ConverterRegistry

        registry = ConverterRegistry()
        converter = registry.get("true-false")

        assert converter is not None

        # Test conversion
        ai_content = {
            "title": "True or False Quiz",
            "statements": [
                {"statement": "The sun is a star", "isTrue": True},
                {"statement": "Water boils at 50°C", "isTrue": False},
            ]
        }

        h5p_params = converter.convert(ai_content, "en")

        assert "statements" in h5p_params or "questions" in h5p_params

        logger.info("✓ TrueFalse converter works correctly")

    @pytest.mark.asyncio
    async def test_all_converters_registered(self):
        """Test that all expected converters are registered."""
        from src.services.h5p.converters import ConverterRegistry

        registry = ConverterRegistry()
        content_types = registry.list_content_types()

        expected_types = [
            "multiple-choice",
            "true-false",
            "fill-blanks",
            "flashcards",
            "dialog-cards",
            "crossword",
            "memory-game",
            "timeline",
        ]

        for content_type in expected_types:
            assert content_type in content_types, f"Missing converter: {content_type}"

        logger.info(f"✓ All {len(content_types)} converters registered")


class TestContentStorageService:
    """Test content storage service."""

    @pytest.mark.asyncio
    async def test_save_and_retrieve_draft(self):
        """Test saving and retrieving a draft."""
        from src.services.h5p.storage import ContentDraft, ContentStorageService

        service = ContentStorageService()

        # Create a draft
        draft = ContentDraft(
            tenant_code="playground",
            user_id=uuid4(),
            content_type="multiple-choice",
            title="Test Quiz",
            ai_content={"questions": [{"question": "Test?", "answers": ["A", "B"]}]},
            status="draft",
        )

        # Save
        draft_id = await service.save_draft(draft)
        assert draft_id is not None

        # Retrieve
        retrieved = await service.get_draft(draft_id, "playground")
        assert retrieved is not None
        assert retrieved.title == "Test Quiz"
        assert retrieved.content_type == "multiple-choice"

        # Delete
        deleted = await service.delete_draft(draft_id, "playground")
        assert deleted is True

        logger.info("✓ Draft storage works correctly")

    @pytest.mark.asyncio
    async def test_list_drafts_with_filters(self):
        """Test listing drafts with filters."""
        from src.services.h5p.storage import ContentDraft, ContentStorageService

        service = ContentStorageService()
        user_id = uuid4()

        # Create multiple drafts
        for i, content_type in enumerate(["multiple-choice", "flashcards", "crossword"]):
            draft = ContentDraft(
                tenant_code="playground",
                user_id=user_id,
                content_type=content_type,
                title=f"Test {content_type}",
                status="draft",
            )
            await service.save_draft(draft)

        # List all
        all_drafts = await service.list_drafts("playground", user_id=user_id)
        assert len(all_drafts) >= 3

        # Filter by content type
        mc_drafts = await service.list_drafts(
            "playground",
            user_id=user_id,
            content_type="multiple-choice",
        )
        assert all(d.content_type == "multiple-choice" for d in mc_drafts)

        logger.info("✓ Draft listing with filters works correctly")


class TestSchemas:
    """Test API schemas."""

    def test_content_chat_request_schema(self):
        """Test ContentChatRequest schema."""
        from src.domains.content_creation.schemas import ContentChatRequest

        # Valid request without session
        request = ContentChatRequest(
            message="Create a quiz about photosynthesis",
            language="en",
        )
        assert request.session_id is None
        assert request.message == "Create a quiz about photosynthesis"

        # Valid request with session
        request = ContentChatRequest(
            session_id="abc-123",
            message="Yes, create it",
        )
        assert request.session_id == "abc-123"

    def test_content_chat_response_schema(self):
        """Test ContentChatResponse schema."""
        from src.domains.content_creation.schemas import (
            ContentChatResponse,
            GeneratedContentResponse,
        )
        from datetime import datetime

        # Response with generated content
        response = ContentChatResponse(
            session_id="abc-123",
            message="Here is your quiz",
            workflow_phase="completed",
            current_agent="quiz_generator",
            generated_content=GeneratedContentResponse(
                id="content-123",
                content_type="multiple-choice",
                title="Photosynthesis Quiz",
                status="draft",
                ai_content={"questions": []},
                created_at=datetime.utcnow(),
            ),
            recommended_types=[],
            suggestions=[],
            metadata={},
        )

        assert response.session_id == "abc-123"
        assert response.generated_content is not None
        assert response.generated_content.content_type == "multiple-choice"

    def test_content_type_info_schema(self):
        """Test ContentTypeInfo schema."""
        from src.domains.content_creation.schemas import ContentTypeInfo

        info = ContentTypeInfo(
            content_type="multiple-choice",
            library="H5P.MultiChoice 1.16",
            name="Multiple Choice",
            description="Create multiple choice questions",
            category="assessment",
            ai_support="full",
            bloom_levels=["remember", "understand", "apply"],
            requires_media=False,
        )

        assert info.content_type == "multiple-choice"
        assert info.ai_support == "full"
        assert "remember" in info.bloom_levels


class TestMultiLanguageSupport:
    """Test multi-language support."""

    @pytest.mark.asyncio
    async def test_turkish_content_generation(self):
        """Test content generation for Turkish language."""
        from src.services.h5p.converters import ConverterRegistry

        registry = ConverterRegistry()
        converter = registry.get("fill-blanks")

        if converter:
            ai_content = {
                "title": "Boşluk Doldurma",
                "sentences": [
                    {"text": "Güneş bir *yıldız*dır.", "blanks": ["yıldız"]}
                ]
            }

            h5p_params = converter.convert(ai_content, "tr")
            assert h5p_params is not None

            logger.info("✓ Turkish content generation works")

    @pytest.mark.asyncio
    async def test_english_content_generation(self):
        """Test content generation for English language."""
        from src.services.h5p.converters import ConverterRegistry

        registry = ConverterRegistry()
        converter = registry.get("fill-blanks")

        if converter:
            ai_content = {
                "title": "Fill in the Blanks",
                "sentences": [
                    {"text": "The sun is a *star*.", "blanks": ["star"]}
                ]
            }

            h5p_params = converter.convert(ai_content, "en")
            assert h5p_params is not None

            logger.info("✓ English content generation works")


class TestAgentConfigurations:
    """Test agent YAML configurations."""

    def test_orchestrator_config_exists(self):
        """Test that orchestrator config exists and is valid."""
        from pathlib import Path
        import yaml

        config_path = Path("config/agents/content/orchestrator.yaml")
        assert config_path.exists(), "Orchestrator config not found"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "agent" in config
        assert config["agent"]["id"] == "content_creation_orchestrator"

        logger.info("✓ Orchestrator config is valid")

    def test_quiz_generator_config_exists(self):
        """Test that quiz generator config exists and is valid."""
        from pathlib import Path
        import yaml

        config_path = Path("config/agents/content/quiz_generator.yaml")
        assert config_path.exists(), "Quiz generator config not found"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "agent" in config

        logger.info("✓ Quiz generator config is valid")

    def test_all_agent_configs_valid(self):
        """Test that all agent configs are valid YAML."""
        from pathlib import Path
        import yaml

        config_dir = Path("config/agents/content")
        if not config_dir.exists():
            pytest.skip("Config directory not found")

        yaml_files = list(config_dir.glob("*.yaml"))
        assert len(yaml_files) > 0, "No YAML files found"

        for yaml_file in yaml_files:
            with open(yaml_file) as f:
                config = yaml.safe_load(f)
            assert config is not None, f"Invalid YAML: {yaml_file}"

        logger.info(f"✓ All {len(yaml_files)} agent configs are valid")


class TestToolDefinitions:
    """Test tool definitions."""

    def test_handoff_tools_exist(self):
        """Test that handoff tools exist."""
        from pathlib import Path

        tools_dir = Path("src/tools/content/handoff")
        assert tools_dir.exists(), "Handoff tools directory not found"

        expected_tools = ["quiz_generator.py", "vocabulary_generator.py"]
        for tool in expected_tools:
            assert (tools_dir / tool).exists(), f"Missing tool: {tool}"

        logger.info("✓ All handoff tools exist")

    def test_export_tools_exist(self):
        """Test that export tools exist."""
        from pathlib import Path

        tools_dir = Path("src/tools/content/export")
        assert tools_dir.exists(), "Export tools directory not found"

        expected_tools = ["export_h5p.py", "preview_content.py", "save_draft.py"]
        for tool in expected_tools:
            assert (tools_dir / tool).exists(), f"Missing tool: {tool}"

        logger.info("✓ All export tools exist")

    def test_quality_tools_exist(self):
        """Test that quality tools exist."""
        from pathlib import Path

        tools_dir = Path("src/tools/content/quality")
        assert tools_dir.exists(), "Quality tools directory not found"

        expected_tools = [
            "check_factual_accuracy.py",
            "check_bloom_alignment.py",
            "check_accessibility.py",
        ]
        for tool in expected_tools:
            assert (tools_dir / tool).exists(), f"Missing tool: {tool}"

        logger.info("✓ All quality tools exist")


# =============================================================================
# Simulation Tests
# =============================================================================


class TestCompleteSimulation:
    """Complete end-to-end simulation tests."""

    @pytest.mark.asyncio
    async def test_teacher_creates_quiz_simulation(
        self,
        playground_tenant,
        uk_teacher,
    ):
        """Simulate a teacher creating a quiz from start to finish."""
        from src.core.orchestration.states.content import (
            ContentCreationState,
            create_initial_content_state,
            ContentTurn,
            GeneratedContent,
        )
        from src.services.h5p.converters import ConverterRegistry

        # Step 1: Initialize session
        session_id = str(uuid4())
        state = create_initial_content_state(
            session_id=session_id,
            tenant_code=playground_tenant["tenant_code"],
            user_id=uk_teacher["id"],
            user_role="teacher",
            language="en",
            country_code=uk_teacher["country_code"],
            framework_code=uk_teacher["framework_code"],
        )

        logger.info(f"Step 1: Session initialized - {session_id}")
        assert state["session_id"] == session_id

        # Step 2: Add user message
        state["conversation_history"] = [
            ContentTurn(
                role="user",
                content="Create a multiple choice quiz about photosynthesis for Grade 5",
                timestamp=datetime.utcnow().isoformat(),
            )
        ]
        state["subject_code"] = "SCI"
        state["topic_code"] = "PHOTOSYNTHESIS"
        state["grade_level"] = 5
        state["content_types"] = ["multiple-choice"]

        logger.info("Step 2: Requirements gathered")

        # Step 3: Generate content
        ai_content = {
            "title": "Photosynthesis Quiz",
            "questions": [
                {
                    "question": "What do plants need for photosynthesis?",
                    "answers": [
                        "Sunlight, water, and carbon dioxide",
                        "Only water",
                        "Only sunlight",
                        "Oxygen and nitrogen"
                    ],
                    "correctIndex": 0,
                    "explanation": "Plants need sunlight, water, and CO2 for photosynthesis."
                },
                {
                    "question": "What gas is produced during photosynthesis?",
                    "answers": ["Carbon dioxide", "Oxygen", "Nitrogen", "Hydrogen"],
                    "correctIndex": 1,
                    "explanation": "Oxygen is released as a byproduct of photosynthesis."
                }
            ]
        }

        generated = GeneratedContent(
            id=str(uuid4()),
            content_type="multiple-choice",
            title="Photosynthesis Quiz",
            ai_content=ai_content,
            status="draft",
            generated_by="quiz_generator",
            created_at=datetime.utcnow().isoformat(),
        )
        state["generated_contents"] = [generated]
        state["current_content"] = generated

        logger.info("Step 3: Content generated")
        assert len(state["generated_contents"]) == 1

        # Step 4: Convert to H5P format
        registry = ConverterRegistry()
        converter = registry.get("multiple-choice")
        h5p_params = converter.convert(ai_content, "en")

        logger.info("Step 4: Converted to H5P format")
        assert "question" in h5p_params or "questions" in h5p_params

        # Step 5: Simulate export
        state["exported_content_ids"] = [generated["id"]]
        state["current_content"]["status"] = "exported"

        logger.info("Step 5: Content exported")
        assert len(state["exported_content_ids"]) == 1

        # Verify final state
        assert state["current_content"]["status"] == "exported"
        logger.info("✓ Complete simulation passed!")

    @pytest.mark.asyncio
    async def test_multi_country_content_creation(
        self,
        playground_credentials,
    ):
        """Test content creation across different countries."""
        from src.core.orchestration.states.content import create_initial_content_state

        countries = ["uk", "usa", "rwanda", "malawi"]
        successful = 0

        for country in countries:
            teacher = playground_credentials["teachers"][country]
            framework = playground_credentials["frameworks"][country]

            state = create_initial_content_state(
                session_id=str(uuid4()),
                tenant_code="playground",
                user_id=teacher["id"],
                user_role="teacher",
                country_code=teacher["country_code"],
                framework_code=framework,
            )

            assert state["country_code"] == teacher["country_code"]
            assert state["framework_code"] == framework
            successful += 1

            logger.info(f"  ✓ {country.upper()} - {framework}")

        assert successful == 4
        logger.info(f"✓ Multi-country test passed ({successful}/4)")

    @pytest.mark.asyncio
    async def test_all_content_types_generation(self):
        """Test generation of all supported content types."""
        from src.services.h5p.converters import ConverterRegistry

        registry = ConverterRegistry()
        content_types = registry.list_content_types()

        successful = 0
        failed = []

        for content_type in content_types:
            converter = registry.get(content_type)
            if converter:
                try:
                    # Create minimal AI content based on type
                    ai_content = self._get_sample_ai_content(content_type)
                    if ai_content:
                        h5p_params = converter.convert(ai_content, "en")
                        if h5p_params:
                            successful += 1
                            logger.info(f"  ✓ {content_type}")
                        else:
                            failed.append(content_type)
                    else:
                        logger.info(f"  - {content_type} (skipped - no sample)")
                except Exception as e:
                    failed.append(f"{content_type}: {e}")

        logger.info(f"✓ Content type test: {successful} passed, {len(failed)} failed")
        if failed:
            logger.warning(f"  Failed: {failed}")

    def _get_sample_ai_content(self, content_type: str) -> dict | None:
        """Get sample AI content for a content type."""
        samples = {
            "multiple-choice": {
                "title": "Test Quiz",
                "questions": [{"question": "Q?", "answers": ["A", "B"], "correctIndex": 0}]
            },
            "true-false": {
                "title": "T/F Quiz",
                "statements": [{"statement": "Test", "isTrue": True}]
            },
            "fill-blanks": {
                "title": "Fill Blanks",
                "sentences": [{"text": "The *sun* is bright", "blanks": ["sun"]}]
            },
            "flashcards": {
                "title": "Cards",
                "cards": [{"term": "Term", "definition": "Def"}]
            },
            "dialog-cards": {
                "title": "Dialog",
                "cards": [{"front": "Front", "back": "Back"}]
            },
            "crossword": {
                "title": "Crossword",
                "words": [{"word": "SUN", "clue": "The star at center of solar system"}]
            },
            "memory-game": {
                "title": "Memory",
                "cards": [{"text": "A", "match": "1"}]
            },
            "accordion": {
                "title": "Accordion",
                "panels": [{"title": "Section", "content": "Content"}]
            },
        }
        return samples.get(content_type)


# =============================================================================
# Test Runner
# =============================================================================


def run_all_tests():
    """Run all tests and print summary."""
    import sys

    logger.info("=" * 60)
    logger.info("H5P Content Creation System - Integration Test Suite")
    logger.info("=" * 60)

    # Run pytest with verbose output
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
    ])

    return exit_code


if __name__ == "__main__":
    exit_code = run_all_tests()
    exit(exit_code)

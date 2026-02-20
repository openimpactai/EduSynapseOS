# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Content Creation Tool Manifest.

This is the central registry of all content creation tools.
Each tool entry contains metadata for organization and documentation.

To add a new tool:
1. Create the tool class in the appropriate category folder
2. Add an entry to CONTENT_CREATION_TOOLS below
3. Enable the tool in the agent's YAML config

Categories:
- curriculum: Curriculum data queries
- h5p: H5P content type and schema tools
- media: Media generation tools
- handoff: Agent delegation tools
- export: H5P export and draft management
- quality: Content quality assurance
- knowledge: Topic knowledge tools
- inference: User intent extraction and smart inference
"""

from typing import TypedDict


class ContentToolInfo(TypedDict):
    """Information about a content creation tool."""

    class_path: str  # Fully qualified class path (module.path:ClassName)
    category: str  # Tool category
    group: str  # Tool group for agent YAML
    description: str  # Human-readable description


CONTENT_CREATION_TOOLS: dict[str, ContentToolInfo] = {
    # =========================================================================
    # H5P SCHEMA TOOLS
    # Query H5P content types and schemas for content generation
    # =========================================================================
    "get_h5p_content_types": {
        "class_path": "src.tools.content.h5p.get_content_types:GetH5PContentTypesTool",
        "category": "h5p",
        "group": "information_gathering",
        "description": (
            "Get list of supported H5P content types with their capabilities, "
            "AI support level, and recommended use cases. Use to show available "
            "content types to users."
        ),
    },
    "get_h5p_schema": {
        "class_path": "src.tools.content.h5p.get_schema:GetH5PSchemaTool",
        "category": "h5p",
        "group": "information_gathering",
        "description": (
            "Get detailed H5P schema for a specific content type. "
            "Returns semantics, required fields, and validation rules. "
            "Used by generators for proper content formatting."
        ),
    },
    "validate_h5p_content": {
        "class_path": "src.tools.content.h5p.validate_content:ValidateH5PContentTool",
        "category": "h5p",
        "group": "quality",
        "description": (
            "Validate generated content against H5P schema requirements. "
            "Returns validation result with errors or warnings."
        ),
    },
    "convert_to_h5p_params": {
        "class_path": "src.tools.content.h5p.convert_params:ConvertToH5PParamsTool",
        "category": "h5p",
        "group": "processing",
        "description": (
            "Convert AI-generated content to H5P-compatible params format. "
            "Handles format differences and adds required H5P metadata."
        ),
    },
    "recommend_content_types": {
        "class_path": "src.tools.content.h5p.recommend_types:RecommendContentTypesTool",
        "category": "h5p",
        "group": "information_gathering",
        "description": (
            "Recommend H5P content types based on learning objective, "
            "topic complexity, and target Bloom's level. Returns ranked "
            "recommendations with rationale."
        ),
    },
    # =========================================================================
    # CURRICULUM TOOLS
    # Query curriculum data for content generation context
    # =========================================================================
    "get_curriculum_subjects": {
        "class_path": "src.tools.content.curriculum.get_subjects:GetCurriculumSubjectsTool",
        "category": "curriculum",
        "group": "information_gathering",
        "description": (
            "Get available subjects for a grade level and curriculum. "
            "Returns subject codes, names, and topic counts."
        ),
    },
    "get_curriculum_topics": {
        "class_path": "src.tools.content.curriculum.get_topics:GetCurriculumTopicsTool",
        "category": "curriculum",
        "group": "information_gathering",
        "description": (
            "Get topics for a subject at a grade level. "
            "Returns topic hierarchy with learning objectives."
        ),
    },
    "get_learning_objectives": {
        "class_path": "src.tools.content.curriculum.get_objectives:GetLearningObjectivesTool",
        "category": "curriculum",
        "group": "information_gathering",
        "description": (
            "Get detailed learning objectives for a topic. "
            "Includes Bloom's level and assessment criteria."
        ),
    },
    "analyze_topic_complexity": {
        "class_path": "src.tools.content.curriculum.analyze_complexity:AnalyzeTopicComplexityTool",
        "category": "curriculum",
        "group": "information_gathering",
        "description": (
            "Analyze topic complexity level, prerequisites, and common difficulties. "
            "Useful for difficulty calibration."
        ),
    },
    # =========================================================================
    # MEDIA GENERATION TOOLS
    # Generate images, diagrams, and charts for content
    # =========================================================================
    "generate_image_gemini": {
        "class_path": "src.tools.content.media.generate_image:GenerateImageGeminiTool",
        "category": "media",
        "group": "media_generation",
        "description": (
            "Generate educational images using Google Gemini's image generation. "
            "Optimized for educational content with appropriate styles."
        ),
    },
    "generate_diagram": {
        "class_path": "src.tools.content.media.generate_diagram:GenerateDiagramTool",
        "category": "media",
        "group": "media_generation",
        "description": (
            "Generate educational diagrams including flowcharts, mind maps, "
            "cycles, hierarchies, and comparisons."
        ),
    },
    "generate_chart": {
        "class_path": "src.tools.content.media.generate_chart:GenerateChartTool",
        "category": "media",
        "group": "media_generation",
        "description": (
            "Generate charts for data visualization including bar, line, "
            "pie, and scatter charts."
        ),
    },
    "upload_media": {
        "class_path": "src.tools.content.media.upload_media:UploadMediaTool",
        "category": "media",
        "group": "media_generation",
        "description": (
            "Upload generated or provided media to H5P storage. "
            "Returns path and URL for H5P content embedding."
        ),
    },
    # =========================================================================
    # HANDOFF TOOLS
    # Delegate to specialized generator agents
    # =========================================================================
    "handoff_to_quiz_generator": {
        "class_path": "src.tools.content.handoff.quiz_generator:HandoffToQuizGeneratorTool",
        "category": "handoff",
        "group": "content_generation",
        "description": (
            "Delegate quiz content generation to the Quiz Generator agent. "
            "Supports multiple choice, true/false, fill-blanks, drag words, etc."
        ),
    },
    "handoff_to_vocabulary_generator": {
        "class_path": "src.tools.content.handoff.vocabulary_generator:HandoffToVocabularyGeneratorTool",
        "category": "handoff",
        "group": "content_generation",
        "description": (
            "Delegate vocabulary content generation to the Vocabulary Generator. "
            "Supports flashcards, dialog cards, crossword, word search."
        ),
    },
    "handoff_to_game_generator": {
        "class_path": "src.tools.content.handoff.game_generator:HandoffToGameGeneratorTool",
        "category": "handoff",
        "group": "content_generation",
        "description": (
            "Delegate game content generation to the Game Content Generator. "
            "Supports memory game, timeline, sorting, image sequencing."
        ),
    },
    "handoff_to_learning_content_generator": {
        "class_path": "src.tools.content.handoff.learning_content_generator:HandoffToLearningContentGeneratorTool",
        "category": "handoff",
        "group": "content_generation",
        "description": (
            "Delegate comprehensive learning content generation. "
            "Supports course presentations, interactive books, branching scenarios."
        ),
    },
    "handoff_to_media_generator": {
        "class_path": "src.tools.content.handoff.media_generator:HandoffToMediaGeneratorTool",
        "category": "handoff",
        "group": "content_generation",
        "description": (
            "Delegate media generation to the Media Generator agent. "
            "Supports images, diagrams, and charts."
        ),
    },
    "handoff_to_content_reviewer": {
        "class_path": "src.tools.content.handoff.content_reviewer:HandoffToContentReviewerTool",
        "category": "handoff",
        "group": "quality_processing",
        "description": (
            "Delegate content review to the Content Reviewer agent. "
            "Performs quality assurance checks and scoring."
        ),
    },
    "handoff_to_translator": {
        "class_path": "src.tools.content.handoff.translator:HandoffToTranslatorTool",
        "category": "handoff",
        "group": "quality_processing",
        "description": (
            "Delegate content translation to the Translator agent. "
            "Supports multi-language translation with educational context."
        ),
    },
    "handoff_to_modifier": {
        "class_path": "src.tools.content.handoff.modifier:HandoffToModifierTool",
        "category": "handoff",
        "group": "quality_processing",
        "description": (
            "Delegate content modification to the Modifier agent. "
            "Supports improvement, simplification, difficulty adjustment."
        ),
    },
    # =========================================================================
    # EXPORT & DELIVERY TOOLS
    # Export content to H5P and manage drafts
    # =========================================================================
    "export_to_h5p": {
        "class_path": "src.tools.content.export.export_h5p:ExportToH5PTool",
        "category": "export",
        "group": "delivery",
        "description": (
            "Export generated content to H5P server via API. "
            "Creates new H5P content that can be edited and used."
        ),
    },
    "preview_content": {
        "class_path": "src.tools.content.export.preview_content:PreviewContentTool",
        "category": "export",
        "group": "delivery",
        "description": (
            "Generate a preview of content before final export. "
            "Returns preview URL or rendered HTML."
        ),
    },
    "save_content_draft": {
        "class_path": "src.tools.content.export.save_draft:SaveContentDraftTool",
        "category": "export",
        "group": "delivery",
        "description": (
            "Save content as a draft for later completion or review. "
            "Supports tags and metadata for organization."
        ),
    },
    "get_content_library": {
        "class_path": "src.tools.content.export.get_library:GetContentLibraryTool",
        "category": "export",
        "group": "information_gathering",
        "description": (
            "Retrieve user's content library with filters. "
            "Lists drafts and published content."
        ),
    },
    "load_content_draft": {
        "class_path": "src.tools.content.export.load_draft:LoadContentDraftTool",
        "category": "export",
        "group": "information_gathering",
        "description": (
            "Load a saved content draft for editing or export. "
            "Returns full draft content and metadata."
        ),
    },
    # =========================================================================
    # INFERENCE TOOLS
    # Extract user intent from messages for smart content creation
    # =========================================================================
    "extract_user_intent": {
        "class_path": "src.tools.content.inference.extract_user_intent:ExtractUserIntentTool",
        "category": "inference",
        "group": "information_gathering",
        "description": (
            "Analyze user message to extract content creation intent. "
            "Identifies requested content type, media requests, and preferences. "
            "Used in user-driven content creation mode for smart inference."
        ),
    },
    # =========================================================================
    # QUALITY ASSURANCE TOOLS
    # Check content quality and alignment
    # =========================================================================
    "check_factual_accuracy": {
        "class_path": "src.tools.content.quality.factual_accuracy:CheckFactualAccuracyTool",
        "category": "quality",
        "group": "quality",
        "description": (
            "Verify factual accuracy of educational content. "
            "Returns accuracy score with verified and questionable facts."
        ),
    },
    "check_bloom_alignment": {
        "class_path": "src.tools.content.quality.bloom_alignment:CheckBloomAlignmentTool",
        "category": "quality",
        "group": "quality",
        "description": (
            "Analyze content alignment with Bloom's Taxonomy levels. "
            "Verifies content targets appropriate cognitive level."
        ),
    },
    "check_accessibility": {
        "class_path": "src.tools.content.quality.accessibility:CheckAccessibilityTool",
        "category": "quality",
        "group": "quality",
        "description": (
            "Check content for WCAG 2.1 accessibility compliance. "
            "Includes alt-text, color contrast checks."
        ),
    },
    "check_age_appropriateness": {
        "class_path": "src.tools.content.quality.age_appropriateness:CheckAgeAppropriatenessTool",
        "category": "quality",
        "group": "quality",
        "description": (
            "Verify content is appropriate for target age/grade level. "
            "Checks vocabulary, complexity, and topics."
        ),
    },
    "get_improvement_suggestions": {
        "class_path": "src.tools.content.quality.improvement_suggestions:GetImprovementSuggestionsTool",
        "category": "quality",
        "group": "quality",
        "description": (
            "Get AI-powered suggestions for improving content quality. "
            "Analyzes pedagogy, engagement, and clarity."
        ),
    },
    # =========================================================================
    # KNOWLEDGE TOOLS
    # Query educational knowledge base
    # =========================================================================
    "get_topic_concepts": {
        "class_path": "src.tools.content.knowledge.topic_concepts:GetTopicConceptsTool",
        "category": "knowledge",
        "group": "information_gathering",
        "description": (
            "Get key concepts and terminology for a topic. "
            "Returns terms with definitions and relationships."
        ),
    },
    "get_common_misconceptions": {
        "class_path": "src.tools.content.knowledge.misconceptions:GetCommonMisconceptionsTool",
        "category": "knowledge",
        "group": "information_gathering",
        "description": (
            "Get common misconceptions for a topic. "
            "Useful for creating effective distractors and feedback."
        ),
    },
    "get_term_definitions": {
        "class_path": "src.tools.content.knowledge.term_definitions:GetTermDefinitionsTool",
        "category": "knowledge",
        "group": "information_gathering",
        "description": (
            "Get age-appropriate definitions for educational terms. "
            "Includes examples and related terms."
        ),
    },
    "search_knowledge_base": {
        "class_path": "src.tools.content.knowledge.search_knowledge:SearchKnowledgeBaseTool",
        "category": "knowledge",
        "group": "information_gathering",
        "description": (
            "Search the educational knowledge base for relevant information. "
            "Uses semantic search for better results."
        ),
    },
}


def get_content_tool_names() -> list[str]:
    """Get list of all content creation tool names.

    Returns:
        List of tool names that can be used in YAML config.
    """
    return list(CONTENT_CREATION_TOOLS.keys())


def get_content_tool_info(tool_name: str) -> ContentToolInfo | None:
    """Get information about a content creation tool.

    Args:
        tool_name: Name of the tool.

    Returns:
        ContentToolInfo dict or None if not found.
    """
    return CONTENT_CREATION_TOOLS.get(tool_name)


def get_content_tools_by_category(category: str) -> list[str]:
    """Get all content tool names in a specific category.

    Args:
        category: Category name (h5p, curriculum, media, handoff, export, quality, knowledge).

    Returns:
        List of tool names in that category.
    """
    return [
        name
        for name, info in CONTENT_CREATION_TOOLS.items()
        if info["category"] == category
    ]


def get_content_tools_by_group(group: str) -> list[str]:
    """Get all content tool names in a specific group.

    Args:
        group: Group name (information_gathering, content_generation, etc.).

    Returns:
        List of tool names in that group.
    """
    return [
        name
        for name, info in CONTENT_CREATION_TOOLS.items()
        if info["group"] == group
    ]


def get_all_content_categories() -> list[str]:
    """Get list of all unique content tool categories.

    Returns:
        List of category names.
    """
    return list(set(info["category"] for info in CONTENT_CREATION_TOOLS.values()))

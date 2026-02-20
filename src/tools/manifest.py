# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Central tool manifest for all available tools.

This is the SINGLE source of truth for tool registrations across all agents.
Each tool entry contains metadata that helps with organization and documentation.

To add a new tool:
1. Create the tool class in the appropriate category folder (e.g., src/tools/curriculum/)
2. Add an entry to TOOL_MANIFEST below
3. Enable the tool in the agent's YAML config (config/agents/*.yaml)

The manifest uses a rich format with metadata:
- class_path: Fully qualified path to the tool class
- category: Functional category for organization
- description: Human-readable description (also helps LLM understand the tool)
"""

from typing import TypedDict


class ToolInfo(TypedDict):
    """Information about a registered tool."""

    class_path: str  # Fully qualified class path (module.path:ClassName)
    category: str  # Tool category (curriculum, student, activities, navigation, agents)
    description: str  # Human-readable description


TOOL_MANIFEST: dict[str, ToolInfo] = {
    # =========================================================================
    # CURRICULUM TOOLS
    # Query curriculum data (subjects, topics, units)
    # Used by agents that need to show curriculum options to students
    # =========================================================================
    "get_subjects": {
        "class_path": "src.tools.curriculum.get_subjects:GetSubjectsTool",
        "category": "curriculum",
        "description": (
            "Get available subjects for the student's grade level. "
            "Returns a list of subjects with IDs and a UIElement for frontend selection. "
            "Call this BEFORE asking student which subject they want."
        ),
    },
    "get_topics": {
        "class_path": "src.tools.curriculum.get_topics:GetTopicsTool",
        "category": "curriculum",
        "description": (
            "Get topics for a specific subject at the student's grade level. "
            "Supports lookup by subject_full_code or subject_name. "
            "Returns topics with a searchable UIElement. Includes 'Random Practice' option."
        ),
    },
    # =========================================================================
    # STUDENT TOOLS
    # Query student-specific data (context, notes, reviews, mastery)
    # Used to personalize interactions based on student history
    # =========================================================================
    "get_student_context": {
        "class_path": "src.tools.student.get_student_context:GetStudentContextTool",
        "category": "student",
        "description": (
            "Get student personalization context from memory layers. "
            "Retrieves weak topics, strong topics, interests, recent activities, "
            "and learning patterns. Use to personalize recommendations."
        ),
    },
    "get_parent_notes": {
        "class_path": "src.tools.student.get_parent_notes:GetParentNotesTool",
        "category": "student",
        "description": (
            "Get active notes from parents about the student. "
            "Includes concerns, celebrations, daily mood, and context notes. "
            "Respects validity dates (valid_from, valid_until)."
        ),
    },
    "get_review_schedule": {
        "class_path": "src.tools.student.get_review_schedule:GetReviewScheduleTool",
        "category": "student",
        "description": (
            "Get spaced repetition review schedule using FSRS algorithm. "
            "Returns items categorized as overdue, due_today, and upcoming. "
            "Use to remind students about pending reviews."
        ),
    },
    "get_my_mastery": {
        "class_path": "src.tools.student.get_my_mastery:GetMyMasteryTool",
        "category": "student",
        "description": (
            "Get student's mastery levels across topics. "
            "Returns overall mastery, per-subject breakdown, and top strengths. "
            "Use when student asks 'How am I doing?' or wants to see progress."
        ),
    },
    "get_my_weaknesses": {
        "class_path": "src.tools.student.get_my_weaknesses:GetMyWeaknessesTool",
        "category": "student",
        "description": (
            "Identify topics where student is struggling (mastery < 40%). "
            "Returns weak topics with recommendations for improvement. "
            "Use when student asks 'What should I work on?' or needs practice suggestions."
        ),
    },
    # =========================================================================
    # ACTIVITY TOOLS
    # Query available activities and games
    # Used to show activity options to students
    # =========================================================================
    "get_games": {
        "class_path": "src.tools.activities.get_games:GetGamesTool",
        "category": "activities",
        "description": (
            "Get available games filtered by grade level. "
            "Returns fun and educational games with difficulty info. "
            "Use when student wants to play a game."
        ),
    },
    "get_activities": {
        "class_path": "src.tools.activities.get_activities:GetActivitiesTool",
        "category": "activities",
        "description": (
            "Get learning activities filtered by category and difficulty. "
            "Includes creative activities, breaks, and educational content. "
            "Use for activity suggestions."
        ),
    },
    # =========================================================================
    # AGENT TOOLS
    # Agent-specific actions (emotions, handoffs)
    # Used for emotional tracking and agent-to-agent communication
    # =========================================================================
    "record_emotion": {
        "class_path": "src.tools.agents.record_emotion:RecordEmotionTool",
        "category": "agents",
        "description": (
            "Record detected emotional state for async processing. "
            "Valid emotions: happy, excited, confident, curious, neutral, "
            "bored, confused, frustrated, anxious, tired. "
            "Intensities: low, medium, high."
        ),
    },
    "create_alert": {
        "class_path": "src.tools.agents.create_alert:CreateAlertTool",
        "category": "agents",
        "description": (
            "Create an alert for teachers/parents about this student. "
            "Use when student shows emotional distress, asks for help, "
            "mentions concerning topics, achieves milestones, or seems disengaged. "
            "Alert types: emotional_distress, help_requested, struggle_detected, "
            "milestone_achieved, concerning_topic, engagement_issue. "
            "Severity: info, warning, critical."
        ),
    },
    "search_interests": {
        "class_path": "src.tools.agents.search_interests:SearchInterestsTool",
        "category": "agents",
        "description": (
            "Search for student interests relevant to a topic or query. "
            "Uses semantic search to find connections between what the student likes "
            "and what they are learning. Returns interests with relevance scores."
        ),
    },
    "record_interest": {
        "class_path": "src.tools.agents.record_interest:RecordInterestTool",
        "category": "agents",
        "description": (
            "Record a student interest detected from conversation. "
            "Use when student mentions hobbies, favorites, or things they enjoy. "
            "Categories: gaming, sports, music, art, science, nature, technology, "
            "reading, movies, crafts, animals, food, travel, social, other."
        ),
    },
    "record_learning_event": {
        "class_path": "src.tools.agents.record_learning_event:RecordLearningEventTool",
        "category": "agents",
        "description": (
            "Record a significant learning event for future reference. "
            "Use for notable moments like breakthroughs, struggles, or achievements. "
            "Event types: breakthrough, struggle, confusion, mastery, engagement, "
            "frustration, correct_answer, incorrect_answer, hint_used."
        ),
    },
    "handoff_to_tutor": {
        "class_path": "src.tools.agents.handoff_to_tutor:HandoffToTutorTool",
        "category": "agents",
        "description": (
            "Handoff conversation to tutor agent for academic questions. "
            "CRITICAL: Use IMMEDIATELY when student asks academic questions "
            "(how to solve, explain, teach). Passes emotional context to tutor."
        ),
    },
    "handoff_to_practice": {
        "class_path": "src.tools.agents.handoff_to_practice:HandoffToPracticeTool",
        "category": "agents",
        "description": (
            "Handoff to practice module to start a practice session. "
            "Use AFTER topic selection is confirmed. "
            "Supports quick, deep, review, and random session types."
        ),
    },
    "handoff_to_learning": {
        "class_path": "src.tools.agents.handoff_to_learning:HandoffToLearningTool",
        "category": "agents",
        "description": (
            "Handoff to Learning Tutor for proactive concept teaching. "
            "Use when student says 'I want to learn about...', 'Teach me...', "
            "or wants to understand a new concept. Starts a structured teaching "
            "session with discovery, explanation, examples, and practice modes."
        ),
    },
    "handoff_to_game": {
        "class_path": "src.tools.agents.handoff_to_game:HandoffToGameTool",
        "category": "agents",
        "description": (
            "Handoff to games module for educational gaming (chess, connect4). "
            "Use when student wants to play a game, needs a brain break, "
            "or you want to suggest a strategy game. Includes coaching and hints."
        ),
    },
    "directly_answer": {
        "class_path": "src.tools.agents.directly_answer:DirectlyAnswerTool",
        "category": "agents",
        "description": (
            "Respond directly to the user without calling other tools. "
            "Use for greetings, casual chat, clarifying questions, "
            "or when other tools returned empty results."
        ),
    },
    # =========================================================================
    # TEACHER TOOLS
    # Tools for teachers to monitor students and classes
    # Used by teacher companion agent to provide insights and analytics
    # =========================================================================
    "get_my_classes": {
        "class_path": "src.tools.teacher.get_my_classes:GetMyClassesTool",
        "category": "teacher",
        "description": (
            "Get the list of classes that the teacher is assigned to. "
            "Returns class names, student counts, and subject information. "
            "Use when teacher asks about their classes."
        ),
    },
    "get_class_students": {
        "class_path": "src.tools.teacher.get_class_students:GetClassStudentsTool",
        "category": "teacher",
        "description": (
            "Get the list of students in a specific class. "
            "Requires class_id. Returns student names and enrollment info. "
            "Use when teacher wants to see students in a class."
        ),
    },
    "get_student_progress": {
        "class_path": "src.tools.teacher.get_student_progress:GetStudentProgressTool",
        "category": "teacher",
        "description": (
            "Get a progress summary for a specific student. "
            "Includes practice sessions, mastery levels, and recent activity. "
            "Use when teacher asks about a student's progress."
        ),
    },
    "get_student_mastery": {
        "class_path": "src.tools.teacher.get_student_mastery:GetStudentMasteryTool",
        "category": "teacher",
        "description": (
            "Get detailed mastery levels for a specific student. "
            "Broken down by subject and topic with status labels. "
            "Use when teacher wants detailed mastery information."
        ),
    },
    "get_class_analytics": {
        "class_path": "src.tools.teacher.get_class_analytics:GetClassAnalyticsTool",
        "category": "teacher",
        "description": (
            "Get aggregate analytics for a class. "
            "Includes performance metrics, engagement levels, and trends. "
            "Use when teacher asks about class-level performance."
        ),
    },
    "get_struggling_students": {
        "class_path": "src.tools.teacher.get_struggling_students:GetStrugglingStudentsTool",
        "category": "teacher",
        "description": (
            "Get a list of students who are struggling. "
            "Based on low mastery, declining performance, or lack of engagement. "
            "Use when teacher asks who needs help or attention."
        ),
    },
    "get_topic_performance": {
        "class_path": "src.tools.teacher.get_topic_performance:GetTopicPerformanceTool",
        "category": "teacher",
        "description": (
            "Get performance metrics for topics across students. "
            "Shows how students are doing on different curriculum topics. "
            "Use when teacher wants to see which topics need attention."
        ),
    },
    "get_student_notes": {
        "class_path": "src.tools.teacher.get_student_notes:GetStudentNotesTool",
        "category": "teacher",
        "description": (
            "Get notes about a specific student from various sources. "
            "Includes parent notes, teacher notes, AI observations. "
            "Use when teacher wants context about a student."
        ),
    },
    "get_alerts": {
        "class_path": "src.tools.teacher.get_alerts:GetAlertsTool",
        "category": "teacher",
        "description": (
            "Get active alerts for students in teacher's classes. "
            "Includes academic struggles, engagement issues, emotional concerns. "
            "Use when teacher asks about alerts or students needing attention."
        ),
    },
    "get_emotional_history": {
        "class_path": "src.tools.teacher.get_emotional_history:GetEmotionalHistoryTool",
        "category": "teacher",
        "description": (
            "Get the emotional signal history for a student. "
            "Shows emotional patterns and trends over time. "
            "Use when teacher wants to understand a student's emotional state."
        ),
    },
    # =========================================================================
    # LEARNING TUTOR TOOLS
    # Tools for Learning Tutor workflow to access curriculum data
    # and generate teaching content
    # =========================================================================
    "get_learning_objectives": {
        "class_path": "src.tools.learning.get_learning_objectives:GetLearningObjectivesTool",
        "category": "learning",
        "description": (
            "Get learning objectives and knowledge components for a topic. "
            "Returns structured curriculum data including concepts, skills, and facts. "
            "Use in Learning Tutor to understand what needs to be taught."
        ),
    },
}


def get_available_tool_names() -> list[str]:
    """Get list of all available tool names.

    Returns:
        List of tool names that can be used in YAML config.
    """
    return list(TOOL_MANIFEST.keys())


def get_tool_info(tool_name: str) -> ToolInfo | None:
    """Get information about a tool by name.

    Args:
        tool_name: Name of the tool.

    Returns:
        ToolInfo dict with class_path, category, description, or None if not found.
    """
    return TOOL_MANIFEST.get(tool_name)


def get_tools_by_category(category: str) -> list[str]:
    """Get all tool names in a specific category.

    Args:
        category: Category name (curriculum, student, activities, navigation, agents).

    Returns:
        List of tool names in that category.
    """
    return [
        name
        for name, info in TOOL_MANIFEST.items()
        if info["category"] == category
    ]


def get_all_categories() -> list[str]:
    """Get list of all unique categories.

    Returns:
        List of category names.
    """
    return list(set(info["category"] for info in TOOL_MANIFEST.values()))

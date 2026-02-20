# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Centralized event type definitions for EduSynapseOS.

This module defines all event types used across the system.
Using constants instead of string literals provides:
- Type safety and IDE autocompletion
- Single source of truth for event names
- Easy refactoring and discoverability

Adding a new event:
1. Add constant to appropriate class here
2. If event needs special handling, update EventRegistry
3. Publisher automatically works, pattern subscribers catch new events
"""


class EventTypes:
    """All event types in EduSynapseOS organized by domain."""

    class Student:
        """Student-related events."""

        ANSWER_SUBMITTED = "student.answer.submitted"
        ANSWER_EVALUATED = "student.answer.evaluated"
        CONCEPT_MASTERED = "student.concept.mastered"
        STRUGGLING_DETECTED = "student.struggling.detected"
        MILESTONE_REACHED = "student.milestone.reached"
        ENGAGEMENT_CHANGED = "student.engagement.changed"
        EMOTION_DETECTED = "student.emotion.detected"

    class Practice:
        """Practice domain events."""

        SESSION_STARTED = "practice.session.started"
        SESSION_COMPLETED = "practice.session.completed"
        QUESTION_GENERATED = "practice.question.generated"
        MISCONCEPTION_DETECTED = "practice.misconception.detected"

    class Conversation:
        """Conversation domain events."""

        STARTED = "conversation.started"
        MESSAGE_RECEIVED = "conversation.message.received"
        RESPONSE_GENERATED = "conversation.response.generated"
        ENDED = "conversation.ended"
        INTENT_DETECTED = "conversation.intent.detected"

    class Memory:
        """Memory domain events."""

        UPDATED = "memory.updated"
        CONSOLIDATED = "memory.consolidated"
        EPISODIC_STORED = "memory.episodic.stored"
        MASTERY_UPDATED = "memory.mastery.updated"

    class Review:
        """Spaced repetition review events."""

        ITEM_DUE = "review.item.due"
        COMPLETED = "review.completed"
        SCHEDULED = "review.scheduled"

    class LearningTutor:
        """Learning tutor workflow events."""

        SESSION_STARTED = "learning_tutor.session.started"
        SESSION_COMPLETED = "learning_tutor.session.completed"
        UNDERSTANDING_UPDATED = "learning_tutor.understanding.updated"
        MODE_CHANGED = "learning_tutor.mode.changed"

    class Companion:
        """Companion workflow events."""

        SESSION_STARTED = "companion.session.started"
        SESSION_ENDED = "companion.session.ended"
        EMOTION_DETECTED = "companion.emotion.detected"
        ACTIVITY_SUGGESTED = "companion.activity.suggested"
        HANDOFF_INITIATED = "companion.handoff.initiated"

    class Gaming:
        """Gaming workflow events."""

        SESSION_STARTED = "gaming.session.started"
        SESSION_COMPLETED = "gaming.session.completed"
        MOVE_MADE = "gaming.move.made"
        MISTAKE_DETECTED = "gaming.mistake.detected"

    class PracticeHelper:
        """Practice helper workflow events."""

        SESSION_STARTED = "practice_helper.session.started"
        SESSION_COMPLETED = "practice_helper.session.completed"
        MODE_ESCALATED = "practice_helper.mode.escalated"
        UNDERSTOOD = "practice_helper.understood"


class EventPatterns:
    """Wildcard patterns for subscribing to multiple events.

    Use these patterns to subscribe to groups of related events.
    The EventBus supports pattern matching like 'student.*' which
    matches 'student.answer.submitted', 'student.answer.evaluated', etc.
    """

    # Domain patterns - catch all events from a domain
    ALL_STUDENT = "student.*"
    ALL_PRACTICE = "practice.*"
    ALL_CONVERSATION = "conversation.*"
    ALL_MEMORY = "memory.*"
    ALL_REVIEW = "review.*"
    ALL_LEARNING_TUTOR = "learning_tutor.*"
    ALL_COMPANION = "companion.*"
    ALL_GAMING = "gaming.*"
    ALL_PRACTICE_HELPER = "practice_helper.*"

    # Cross-domain patterns - catch similar events across domains
    ALL_SESSION_STARTED = "*.session.started"
    ALL_SESSION_COMPLETED = "*.session.completed"
    ALL_SESSION_ENDED = "*.session.ended"
    ALL_GAME_EVENTS = "gaming.*"

    # Specific sub-patterns
    ALL_STUDENT_ANSWER = "student.answer.*"

    # Global wildcard
    ALL = "*"


class EventCategory:
    """Event categories for analytics processing."""

    ENGAGEMENT = "engagement"
    INTERACTION = "interaction"
    PERFORMANCE = "performance"
    BEHAVIOR = "behavior"


class EventRegistry:
    """Registry for event metadata and categorization."""

    # Maps event types to analytics categories
    _category_map: dict[str, str] = {
        # Engagement events (session lifecycle)
        EventTypes.Practice.SESSION_STARTED: EventCategory.ENGAGEMENT,
        EventTypes.Practice.SESSION_COMPLETED: EventCategory.ENGAGEMENT,
        EventTypes.Conversation.STARTED: EventCategory.ENGAGEMENT,
        EventTypes.Conversation.ENDED: EventCategory.ENGAGEMENT,
        EventTypes.LearningTutor.SESSION_STARTED: EventCategory.ENGAGEMENT,
        EventTypes.LearningTutor.SESSION_COMPLETED: EventCategory.ENGAGEMENT,
        EventTypes.Companion.SESSION_STARTED: EventCategory.ENGAGEMENT,
        EventTypes.Companion.SESSION_ENDED: EventCategory.ENGAGEMENT,
        EventTypes.Gaming.SESSION_STARTED: EventCategory.ENGAGEMENT,
        EventTypes.Gaming.SESSION_COMPLETED: EventCategory.ENGAGEMENT,
        EventTypes.PracticeHelper.SESSION_STARTED: EventCategory.ENGAGEMENT,
        EventTypes.PracticeHelper.SESSION_COMPLETED: EventCategory.ENGAGEMENT,
        # Interaction events (user interactions)
        EventTypes.Conversation.MESSAGE_RECEIVED: EventCategory.INTERACTION,
        EventTypes.Conversation.RESPONSE_GENERATED: EventCategory.INTERACTION,
        EventTypes.Practice.QUESTION_GENERATED: EventCategory.INTERACTION,
        EventTypes.Gaming.MOVE_MADE: EventCategory.INTERACTION,
        # Performance events (learning outcomes)
        EventTypes.Student.ANSWER_SUBMITTED: EventCategory.PERFORMANCE,
        EventTypes.Student.ANSWER_EVALUATED: EventCategory.PERFORMANCE,
        EventTypes.LearningTutor.UNDERSTANDING_UPDATED: EventCategory.PERFORMANCE,
        # Behavior events (learning patterns)
        EventTypes.Practice.MISCONCEPTION_DETECTED: EventCategory.BEHAVIOR,
        EventTypes.Student.MILESTONE_REACHED: EventCategory.BEHAVIOR,
        EventTypes.Student.CONCEPT_MASTERED: EventCategory.BEHAVIOR,
        EventTypes.Student.STRUGGLING_DETECTED: EventCategory.BEHAVIOR,
        EventTypes.Student.ENGAGEMENT_CHANGED: EventCategory.BEHAVIOR,
        EventTypes.Student.EMOTION_DETECTED: EventCategory.BEHAVIOR,
        EventTypes.LearningTutor.MODE_CHANGED: EventCategory.BEHAVIOR,
        EventTypes.Companion.EMOTION_DETECTED: EventCategory.BEHAVIOR,
        EventTypes.Companion.ACTIVITY_SUGGESTED: EventCategory.BEHAVIOR,
        EventTypes.Companion.HANDOFF_INITIATED: EventCategory.BEHAVIOR,
        EventTypes.Gaming.MISTAKE_DETECTED: EventCategory.BEHAVIOR,
        EventTypes.PracticeHelper.MODE_ESCALATED: EventCategory.BEHAVIOR,
        EventTypes.PracticeHelper.UNDERSTOOD: EventCategory.BEHAVIOR,
    }

    # Maps event types to session types
    _session_type_map: dict[str, str] = {
        # Practice events
        EventTypes.Practice.SESSION_STARTED: "practice",
        EventTypes.Practice.SESSION_COMPLETED: "practice",
        EventTypes.Practice.QUESTION_GENERATED: "practice",
        EventTypes.Practice.MISCONCEPTION_DETECTED: "practice",
        # Conversation events
        EventTypes.Conversation.STARTED: "conversation",
        EventTypes.Conversation.MESSAGE_RECEIVED: "conversation",
        EventTypes.Conversation.RESPONSE_GENERATED: "conversation",
        EventTypes.Conversation.ENDED: "conversation",
        EventTypes.Conversation.INTENT_DETECTED: "conversation",
        # Learning Tutor events
        EventTypes.LearningTutor.SESSION_STARTED: "learning_tutor",
        EventTypes.LearningTutor.SESSION_COMPLETED: "learning_tutor",
        EventTypes.LearningTutor.UNDERSTANDING_UPDATED: "learning_tutor",
        EventTypes.LearningTutor.MODE_CHANGED: "learning_tutor",
        # Companion events
        EventTypes.Companion.SESSION_STARTED: "companion",
        EventTypes.Companion.SESSION_ENDED: "companion",
        EventTypes.Companion.EMOTION_DETECTED: "companion",
        EventTypes.Companion.ACTIVITY_SUGGESTED: "companion",
        EventTypes.Companion.HANDOFF_INITIATED: "companion",
        # Gaming events (session-based for consistency with other workflows)
        EventTypes.Gaming.SESSION_STARTED: "gaming",
        EventTypes.Gaming.SESSION_COMPLETED: "gaming",
        EventTypes.Gaming.MOVE_MADE: "gaming",
        EventTypes.Gaming.MISTAKE_DETECTED: "gaming",
        # Practice Helper events
        EventTypes.PracticeHelper.SESSION_STARTED: "practice_helper",
        EventTypes.PracticeHelper.SESSION_COMPLETED: "practice_helper",
        EventTypes.PracticeHelper.MODE_ESCALATED: "practice_helper",
        EventTypes.PracticeHelper.UNDERSTOOD: "practice_helper",
    }

    @classmethod
    def get_category(cls, event_type: str) -> str | None:
        """Get analytics category for an event type.

        Args:
            event_type: Event type string.

        Returns:
            Category string or None if not categorized.
        """
        return cls._category_map.get(event_type)

    @classmethod
    def get_session_type(cls, event_type: str) -> str:
        """Get session type for an event.

        Args:
            event_type: Event type string.

        Returns:
            Session type string (practice, conversation, or unknown).
        """
        return cls._session_type_map.get(event_type, "unknown")

    @classmethod
    def is_analytics_event(cls, event_type: str) -> bool:
        """Check if event should be forwarded to analytics.

        Args:
            event_type: Event type string.

        Returns:
            True if event has an analytics category.
        """
        return event_type in cls._category_map

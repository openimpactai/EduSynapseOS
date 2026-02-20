# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""System prompt builder for dynamic agents.

This module provides the SystemPromptBuilder class that constructs system prompts
from YAML configuration, persona, and runtime context. It enables config-driven
prompt generation with:
- Variable interpolation ({grade_level}, {language}, etc.)
- Conditional sections (show alerts only if has_alerts)
- Persona integration (voice, behavior, identity)
- Tool instruction generation from registry

Usage:
    from src.core.agents.prompt_builder import SystemPromptBuilder
    from src.core.agents.context import SystemPromptConfig

    config = SystemPromptConfig(
        role="You are {persona_name}, a friendly companion.",
        rules=[...],
        examples=[...],
    )

    builder = SystemPromptBuilder(config, persona=persona, tool_registry=registry)
    prompt = builder.build({
        "persona_name": "Buddy",
        "grade_level": 5,
        "language": "en",
    })
"""

import logging
import re
from typing import Any

from src.core.agents.context import (
    SystemPromptConfig,
    SystemPromptContextSection,
    SystemPromptExample,
    SystemPromptRule,
    ToolsConfig,
)
from src.core.personas.models import Persona
from src.tools.manifest import get_tool_info

logger = logging.getLogger(__name__)


class SystemPromptBuilder:
    """Builds system prompts from YAML config + persona + runtime context.

    The builder assembles a complete system prompt by:
    1. Role definition (core identity) with variable substitution
    2. Persona segment (voice, behavior, identity)
    3. Rules (critical instructions as numbered sections)
    4. Examples (few-shot learning conversations)
    5. Context sections (dynamic data with conditions)
    6. Tool instructions (grouped tool descriptions)
    7. Response format instructions
    8. Personality guidelines

    Attributes:
        config: System prompt configuration from YAML.
        persona: Optional persona for communication style.
        tools_config: Optional tools configuration from agent YAML.

    Example:
        builder = SystemPromptBuilder(
            config=agent_config.system_prompt,
            persona=persona,
            tools_config=agent_config.tools,
        )

        prompt = builder.build({
            "persona_name": persona.name,
            "grade_level": 5,
            "language": "tr",
            "current_emotion": "happy",
            "interests": ["football", "games"],
            "alerts": [],
        })
    """

    def __init__(
        self,
        config: SystemPromptConfig,
        persona: Persona | None = None,
        tools_config: ToolsConfig | None = None,
    ) -> None:
        """Initialize the prompt builder.

        Args:
            config: System prompt configuration from YAML.
            persona: Optional persona for communication style integration.
            tools_config: Optional tools configuration for tool instructions.
        """
        self._config = config
        self._persona = persona
        self._tools_config = tools_config

    def build(self, context: dict[str, Any], intent: str | None = None) -> str:
        """Build complete system prompt from config and context.

        Assembles all prompt components in order:
        1. Role definition
        2. Persona segment
        3. Rules
        4. Examples
        5. Context sections
        6. Intent-specific prompt (for capability-based execution)
        7. Tool instructions (for tool-based execution)
        8. Response format
        9. Personality

        Args:
            context: Runtime context with variables for interpolation.
                Expected keys depend on the templates used:
                - persona_name: str - Name of the persona
                - grade_level: int - Student's grade level
                - language: str - Language code (e.g., "en", "tr")
                - current_emotion: str - Current emotional state
                - interests: list[str] - Student interests
                - alerts: list[dict] - Pending alerts (optional)
            intent: Optional intent/capability name to include intent-specific prompt.
                Used for capability-based execution (e.g., "question_generation").

        Returns:
            Complete system prompt string ready for LLM.

        Example:
            # For tool-based execution (Companion)
            prompt = builder.build({
                "persona_name": "Buddy",
                "grade_level": 5,
                "language": "en",
            })

            # For capability-based execution (Tutor)
            prompt = builder.build({
                "grade_level": 5,
                "language": "en",
            }, intent="question_generation")
        """
        parts: list[str] = []

        # 1. Role definition (with variable substitution)
        role_section = self._interpolate(self._config.role, context)
        parts.append(role_section)

        # 2. Persona segment (if available)
        if self._persona:
            persona_segment = self._persona.get_system_prompt_segment()
            parts.append(persona_segment)

        # 3. Rules section
        if self._config.rules:
            rules_section = self._build_rules_section()
            parts.append(rules_section)

        # 4. Examples section
        if self._config.examples:
            examples_section = self._build_examples_section()
            parts.append(examples_section)

        # 5. Context sections (conditional)
        context_section = self._build_context_sections(context)
        if context_section:
            parts.append(context_section)

        # 6. Intent-specific prompt (for capability-based execution)
        if intent and self._config.intent_prompts:
            intent_section = self._build_intent_section(intent, context)
            if intent_section:
                parts.append(intent_section)

        # 7. Tool instructions (for tool-based execution)
        if self._config.tool_instructions and self._tools_config:
            tool_section = self._build_tool_instructions(context)
            if tool_section:
                parts.append(tool_section)

        # 8. Response format
        if self._config.response_format:
            parts.append(self._config.response_format)

        # 9. Personality (with variable substitution)
        if self._config.personality:
            personality_section = self._interpolate(self._config.personality, context)
            parts.append(personality_section)

        # Join with double newlines for clear separation
        return "\n\n".join(filter(None, parts))

    def _build_intent_section(self, intent: str, context: dict[str, Any]) -> str:
        """Build intent-specific prompt section.

        Retrieves the intent-specific prompt template and applies
        variable interpolation.

        Args:
            intent: Intent/capability name (e.g., "question_generation").
            context: Runtime context for variable substitution.

        Returns:
            Interpolated intent-specific prompt or empty string.
        """
        if intent not in self._config.intent_prompts:
            logger.debug("No intent prompt found for: %s", intent)
            return ""

        intent_template = self._config.intent_prompts[intent]
        return self._interpolate(intent_template, context)

    def _interpolate(self, template: str, context: dict[str, Any]) -> str:
        """Replace {variables} with context values.

        Handles various value types:
        - str: Used directly
        - int/float: Converted to string
        - list: Joined with ", "
        - dict: Converted to formatted string
        - None: Replaced with empty string

        Missing variables are left as-is with a warning logged.

        Args:
            template: Template string with {variable} placeholders.
            context: Dictionary of values to substitute.

        Returns:
            Template with variables replaced by values.

        Example:
            template = "Grade: {grade_level}, Language: {language}"
            context = {"grade_level": 5, "language": "tr"}
            result = self._interpolate(template, context)
            # Returns: "Grade: 5, Language: tr"
        """
        # Find all {variable} patterns
        pattern = r"\{(\w+)\}"

        def replace_var(match: re.Match) -> str:
            var_name = match.group(1)

            if var_name not in context:
                logger.warning("Missing variable in template: %s", var_name)
                return match.group(0)  # Keep original {var} if not found

            value = context[var_name]

            if value is None:
                return ""
            elif isinstance(value, list):
                return ", ".join(str(v) for v in value)
            elif isinstance(value, dict):
                return self._format_dict(value)
            else:
                return str(value)

        return re.sub(pattern, replace_var, template)

    def _format_dict(self, data: dict[str, Any]) -> str:
        """Format a dictionary for prompt inclusion.

        Args:
            data: Dictionary to format.

        Returns:
            Formatted string representation.
        """
        lines = []
        for key, value in data.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _build_rules_section(self) -> str:
        """Build formatted rules section.

        Formats rules as numbered sections with titles and content.

        Returns:
            Formatted rules section string.

        Example output:
            ## CRITICAL RULES

            ### Rule 1: NEVER Teach
            If a student asks an academic question:
            - DO NOT explain or teach
            - Use handoff_to_tutor IMMEDIATELY

            ### Rule 2: ALWAYS Clarify
            Before calling navigate tool...
        """
        if not self._config.rules:
            return ""

        lines = ["## CRITICAL RULES"]

        for idx, rule in enumerate(self._config.rules, 1):
            lines.append("")
            lines.append(f"### Rule {idx}: {rule.title}")
            lines.append(rule.content.strip())

        return "\n".join(lines)

    def _build_examples_section(self) -> str:
        """Build examples section for few-shot learning.

        Formats examples with titles and conversation content.

        Returns:
            Formatted examples section string.

        Example output:
            ## EXAMPLE CONVERSATIONS

            ### Practice Flow
            Student: "I want to practice"
            You: [Call get_subjects] "Great! Which subject?"
            ...
        """
        if not self._config.examples:
            return ""

        lines = ["## EXAMPLE CONVERSATIONS"]

        for example in self._config.examples:
            lines.append("")
            lines.append(f"### {example.title}")
            lines.append(example.conversation.strip())

        return "\n".join(lines)

    def _build_context_sections(self, context: dict[str, Any]) -> str:
        """Build context sections with conditional inclusion.

        Evaluates conditions and includes only sections that pass.
        Supported conditions:
        - has_{key}: Include if context[key] is truthy and non-empty
        - no_{key}: Include if context[key] is falsy or empty

        Args:
            context: Runtime context for condition evaluation and interpolation.

        Returns:
            Formatted context sections string.

        Example output:
            ## STUDENT CONTEXT
            - Grade Level: 5
            - Language: tr
            - Current Emotion: happy

            ## PENDING ALERTS
            [Alert content here...]
        """
        if not self._config.context_sections:
            return ""

        sections: list[str] = []

        for section in self._config.context_sections:
            # Check condition if specified
            if section.condition and not self._evaluate_condition(section.condition, context):
                continue

            # Build section content
            content = self._interpolate(section.template, context)

            # Skip empty content
            if not content.strip():
                continue

            section_text = f"## {section.title}\n{content.strip()}"
            sections.append(section_text)

        return "\n\n".join(sections)

    def _evaluate_condition(self, condition: str, context: dict[str, Any]) -> bool:
        """Evaluate a condition against context.

        Supported condition patterns:
        - has_{key}: True if context[key] exists and is truthy
        - no_{key}: True if context[key] doesn't exist or is falsy
        - {key}: Same as has_{key}

        Args:
            condition: Condition string to evaluate.
            context: Runtime context for evaluation.

        Returns:
            True if condition is satisfied.

        Examples:
            _evaluate_condition("has_alerts", {"alerts": [...]}) -> True
            _evaluate_condition("has_alerts", {"alerts": []}) -> False
            _evaluate_condition("no_alerts", {"alerts": []}) -> True
        """
        # Handle has_X pattern
        if condition.startswith("has_"):
            key = condition[4:]  # Remove "has_" prefix
            value = context.get(key)
            return self._is_truthy(value)

        # Handle no_X pattern
        if condition.startswith("no_"):
            key = condition[3:]  # Remove "no_" prefix
            value = context.get(key)
            return not self._is_truthy(value)

        # Direct key check
        value = context.get(condition)
        return self._is_truthy(value)

    def _is_truthy(self, value: Any) -> bool:
        """Check if a value is truthy for condition evaluation.

        Empty collections are considered falsy.

        Args:
            value: Value to check.

        Returns:
            True if value is truthy.
        """
        if value is None:
            return False
        if isinstance(value, (list, dict, str)):
            return len(value) > 0
        return bool(value)

    def _build_tool_instructions(self, context: dict[str, Any]) -> str:
        """Build tool usage instructions from tools config.

        Groups tools by their group field and generates instructions
        with tool names and descriptions.

        Args:
            context: Runtime context for variable substitution.

        Returns:
            Formatted tool instructions section.

        Example output:
            ## TOOL USAGE ORDER

            **Information Gathering (use FIRST):**
            - get_subjects: Get available subjects for the student's grade level
            - get_topics: Get topics for a specific subject
            ...

            **Navigation (use AFTER gathering info):**
            - navigate: Navigate to activity page
        """
        if not self._tools_config or not self._config.tool_instructions:
            return ""

        # Get tools organized by group
        groups = self._tools_config.get_tools_by_group()

        # Build tool list for each group with descriptions from manifest
        group_lists: dict[str, str] = {}
        for group_name, tools in groups.items():
            tool_lines = []
            for tool in tools:
                # Get description from central manifest
                tool_info = get_tool_info(tool.name)
                if tool_info and tool_info.get("description"):
                    tool_lines.append(f"- {tool.name}: {tool_info['description']}")
                else:
                    tool_lines.append(f"- {tool.name}")
            group_lists[f"tool_list_{group_name}"] = "\n".join(tool_lines)

        # Merge with context for interpolation
        merged_context = {**context, **group_lists}

        # Interpolate tool instructions template
        return self._interpolate(self._config.tool_instructions, merged_context)

    def get_persona_name(self) -> str:
        """Get the persona name for context.

        Returns:
            Persona name or "Assistant" if no persona set.
        """
        if self._persona:
            return self._persona.name
        return "Assistant"

    def get_persona_language(self) -> str:
        """Get the persona's preferred language.

        Returns:
            Language code or "en" if no persona set.
        """
        if self._persona:
            return self._persona.voice.language
        return "en"

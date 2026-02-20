# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""MCP prompt definitions for EduSynapseOS.

This module defines the MCP prompts that provide pre-built templates for
common educational interactions. Prompts help LLM clients generate
consistent, pedagogically-sound responses.

Prompts:
- explain_concept: Template for explaining a concept to a student
- generate_practice: Template for generating practice questions
- analyze_mistake: Template for analyzing a student's mistake

Each prompt returns a system message and user message that can be used
directly with an LLM.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mcp.types import TextContent

if TYPE_CHECKING:
    from src.core.intelligence.mcp.server import EduSynapseMCPServer

logger = logging.getLogger(__name__)


def register_prompts(server: EduSynapseMCPServer) -> None:
    """Register all MCP prompts with the server.

    Args:
        server: The EduSynapseMCPServer instance.
    """
    mcp = server.mcp

    @mcp.prompt()
    def explain_concept(
        topic: str,
        concept: str,
        student_level: str = "intermediate",
        format: str = "detailed",
        language: str = "tr",
        use_analogies: bool = True,
        include_examples: bool = True,
    ) -> str:
        """Generate a prompt for explaining a concept to a student.

        Creates a structured prompt that guides the LLM to explain
        a concept in an educationally effective way, considering
        the student's level and preferred format.

        Args:
            topic: The broader topic area (e.g., "Mathematics").
            concept: The specific concept to explain (e.g., "Fractions").
            student_level: Student's proficiency level
                          (beginner, intermediate, advanced). Default: "intermediate".
            format: Explanation format (brief, detailed, step_by_step).
                   Default: "detailed".
            language: Language for the explanation (default: "tr").
            use_analogies: Include real-world analogies (default: True).
            include_examples: Include worked examples (default: True).

        Returns:
            A prompt string for concept explanation.

        Example:
            >>> prompt = explain_concept(
            ...     topic="Mathematics",
            ...     concept="Fractions",
            ...     student_level="beginner",
            ...     format="step_by_step",
            ... )
        """
        logger.debug(
            "Prompt: explain_concept, topic=%s, concept=%s, level=%s",
            topic,
            concept,
            student_level,
        )

        # Build system instruction
        system_parts = [
            "You are an expert educational tutor specializing in clear, engaging explanations.",
            f"You are explaining concepts in {language.upper()} language.",
            "",
            "Your explanation style:",
            f"- Target audience: {student_level} level student",
            f"- Format: {format}",
        ]

        if use_analogies:
            system_parts.append("- Use real-world analogies to make abstract concepts concrete")

        if include_examples:
            system_parts.append("- Include worked examples that progress from simple to complex")

        system_parts.extend([
            "",
            "Pedagogical principles to follow:",
            "- Start from what the student likely already knows",
            "- Build understanding incrementally",
            "- Anticipate and address common misconceptions",
            "- Check understanding with embedded questions",
            "- Summarize key points at the end",
        ])

        # Build user request
        user_parts = [
            f"Please explain the following concept from {topic}:",
            "",
            f"**Concept:** {concept}",
            "",
        ]

        if format == "brief":
            user_parts.append("Provide a concise explanation (2-3 paragraphs max).")
        elif format == "step_by_step":
            user_parts.append("Break down the explanation into clear, numbered steps.")
        else:  # detailed
            user_parts.append("Provide a thorough explanation covering all important aspects.")

        if use_analogies:
            user_parts.append("Include at least one relatable analogy.")

        if include_examples:
            user_parts.append("Include 2-3 examples of increasing complexity.")

        user_parts.extend([
            "",
            "End with a brief summary and a simple question to check understanding.",
        ])

        # Combine into final prompt
        prompt_text = (
            "=== SYSTEM INSTRUCTION ===\n"
            + "\n".join(system_parts)
            + "\n\n=== USER REQUEST ===\n"
            + "\n".join(user_parts)
        )

        return prompt_text

    @mcp.prompt()
    def generate_practice(
        topic: str,
        question_count: int = 5,
        difficulty: str = "medium",
        question_types: str = "mixed",
        bloom_levels: str = "mixed",
        language: str = "tr",
        include_solutions: bool = True,
    ) -> str:
        """Generate a prompt for creating practice questions.

        Creates a structured prompt that guides the LLM to generate
        a set of practice questions with appropriate difficulty,
        variety, and pedagogical alignment.

        Args:
            topic: The topic for practice questions.
            question_count: Number of questions to generate (default: 5).
            difficulty: Difficulty level (easy, medium, hard, mixed).
                       Default: "medium".
            question_types: Types of questions (multiple_choice, short_answer,
                           true_false, mixed). Default: "mixed".
            bloom_levels: Bloom's taxonomy levels (remember, understand, apply,
                         analyze, mixed). Default: "mixed".
            language: Language for questions (default: "tr").
            include_solutions: Include detailed solutions (default: True).

        Returns:
            A prompt string for practice question generation.

        Example:
            >>> prompt = generate_practice(
            ...     topic="Fractions",
            ...     question_count=5,
            ...     difficulty="medium",
            ...     question_types="mixed",
            ... )
        """
        logger.debug(
            "Prompt: generate_practice, topic=%s, count=%d, difficulty=%s",
            topic,
            question_count,
            difficulty,
        )

        # Map difficulty to numeric range
        difficulty_map = {
            "easy": "0.2-0.4",
            "medium": "0.4-0.6",
            "hard": "0.6-0.8",
            "mixed": "0.2-0.8 (varied)",
        }
        difficulty_range = difficulty_map.get(difficulty, "0.4-0.6")

        # Build system instruction
        system_parts = [
            "You are an expert educational content creator.",
            f"Create practice questions in {language.upper()} language.",
            "",
            "Question quality guidelines:",
            "- Each question should have a clear, unambiguous answer",
            "- Distractors (wrong options) should represent common misconceptions",
            "- Questions should test genuine understanding, not just memorization",
            "- Progress from simpler to more complex within the set",
            "",
            "Bloom's Taxonomy levels:",
            "- Remember: Recall facts and basic concepts",
            "- Understand: Explain ideas or concepts",
            "- Apply: Use information in new situations",
            "- Analyze: Draw connections among ideas",
        ]

        # Build user request
        user_parts = [
            f"Generate {question_count} practice questions about: **{topic}**",
            "",
            "Requirements:",
            f"- Difficulty range: {difficulty_range}",
            f"- Question types: {question_types}",
            f"- Bloom levels: {bloom_levels}",
        ]

        if include_solutions:
            user_parts.extend([
                "",
                "For each question, provide:",
                "1. The question text",
                "2. Answer options (for multiple choice)",
                "3. Correct answer",
                "4. Detailed explanation of why the answer is correct",
                "5. Common mistakes students might make",
            ])
        else:
            user_parts.extend([
                "",
                "For each question, provide:",
                "1. The question text",
                "2. Answer options (for multiple choice)",
                "3. Correct answer marked",
            ])

        user_parts.extend([
            "",
            "Format each question clearly with numbering (1, 2, 3, etc.).",
        ])

        # Combine into final prompt
        prompt_text = (
            "=== SYSTEM INSTRUCTION ===\n"
            + "\n".join(system_parts)
            + "\n\n=== USER REQUEST ===\n"
            + "\n".join(user_parts)
        )

        return prompt_text

    @mcp.prompt()
    def analyze_mistake(
        question: str,
        correct_answer: str,
        student_answer: str,
        topic: str = "",
        language: str = "tr",
        provide_remediation: bool = True,
        identify_root_cause: bool = True,
    ) -> str:
        """Generate a prompt for analyzing a student's mistake.

        Creates a structured prompt that guides the LLM to analyze
        why a student made a mistake, identify the underlying
        misconception, and provide helpful remediation.

        Args:
            question: The original question text.
            correct_answer: The correct answer.
            student_answer: The student's incorrect answer.
            topic: The topic of the question (optional).
            language: Language for the analysis (default: "tr").
            provide_remediation: Include remediation steps (default: True).
            identify_root_cause: Analyze root cause (default: True).

        Returns:
            A prompt string for mistake analysis.

        Example:
            >>> prompt = analyze_mistake(
            ...     question="What is 1/2 + 1/4?",
            ...     correct_answer="3/4",
            ...     student_answer="2/6",
            ...     topic="Fractions",
            ... )
        """
        logger.debug(
            "Prompt: analyze_mistake, topic=%s, student_answer=%s...",
            topic,
            student_answer[:30] if student_answer else "",
        )

        # Build system instruction
        system_parts = [
            "You are an expert educational diagnostician.",
            f"Analyze student mistakes in {language.upper()} language.",
            "",
            "Your analysis approach:",
            "- Be supportive and constructive, never judgmental",
            "- Focus on understanding, not blame",
            "- Identify specific misconceptions, not just 'wrong answer'",
            "- Connect mistakes to underlying conceptual gaps",
            "",
            "Common types of errors to consider:",
            "- Procedural errors (wrong steps, missing steps)",
            "- Conceptual errors (misunderstanding the concept)",
            "- Careless errors (calculation mistakes, misreading)",
            "- Transfer errors (applying wrong concept from similar topic)",
        ]

        # Build user request
        user_parts = [
            "Analyze this student's mistake:",
            "",
        ]

        if topic:
            user_parts.append(f"**Topic:** {topic}")

        user_parts.extend([
            f"**Question:** {question}",
            f"**Correct Answer:** {correct_answer}",
            f"**Student's Answer:** {student_answer}",
            "",
        ])

        if identify_root_cause:
            user_parts.extend([
                "Please analyze:",
                "1. What specific error did the student make?",
                "2. What misconception likely caused this error?",
                "3. What is the root cause (conceptual gap)?",
                "",
            ])

        if provide_remediation:
            user_parts.extend([
                "Provide remediation:",
                "1. A clear explanation of the correct approach",
                "2. Steps to address the underlying misconception",
                "3. A similar practice question to reinforce understanding",
                "",
            ])

        user_parts.extend([
            "Keep your response encouraging and focused on learning.",
        ])

        # Combine into final prompt
        prompt_text = (
            "=== SYSTEM INSTRUCTION ===\n"
            + "\n".join(system_parts)
            + "\n\n=== USER REQUEST ===\n"
            + "\n".join(user_parts)
        )

        return prompt_text

    logger.debug("Registered 3 MCP prompts: explain_concept, generate_practice, analyze_mistake")

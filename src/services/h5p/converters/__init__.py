# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""H5P Content Converters.

This package provides converters that transform AI-generated content
into H5P-compatible params format. Each H5P content type has a
dedicated converter that handles the specific schema requirements.

The converter system follows a registry pattern:
1. BaseH5PConverter: Abstract base class defining the converter interface
2. ConverterRegistry: Central registry for looking up converters by type
3. Type-specific converters: Implement the actual conversion logic

Usage:
    from src.services.h5p.converters import ConverterRegistry

    # Get registry with all converters
    registry = ConverterRegistry()

    # Get converter for a content type
    converter = registry.get("multiple-choice")

    # Convert AI content to H5P params
    h5p_params = converter.convert(ai_content, language="en")

Available Converters:
- Assessment: MultipleChoice, TrueFalse, FillBlanks, DragWords, etc.
- Vocabulary: Flashcards, DialogCards, Crossword, WordSearch
- Learning: Accordion, Summary, SortParagraphs, Essay
- Game: MemoryGame, Timeline, PersonalityQuiz
- Media: Chart, ImageHotspots
- Composite: QuestionSet, CoursePresentation, InteractiveBook
"""

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.converters.registry import ConverterRegistry

__all__ = [
    "BaseH5PConverter",
    "ConverterRegistry",
]

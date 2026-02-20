# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""H5P Converter Registry.

This module provides a central registry for H5P content converters.
The registry allows looking up converters by content type and provides
methods for listing available converters.

Usage:
    from src.services.h5p.converters import ConverterRegistry

    # Create registry (auto-loads all converters)
    registry = ConverterRegistry()

    # Get converter by content type
    converter = registry.get("multiple-choice")

    # List all available content types
    types = registry.list_content_types()

    # Get converter by H5P library
    converter = registry.get_by_library("H5P.MultiChoice 1.16")
"""

import logging
from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter

logger = logging.getLogger(__name__)


class ConverterRegistry:
    """Central registry for H5P content converters.

    Manages a collection of converters and provides lookup methods
    by content type, library name, and category.

    The registry is populated by either:
    1. Passing converters to the constructor
    2. Using load_default_converters() to auto-load all available converters
    3. Manually registering converters with register()

    Attributes:
        _converters: Dictionary mapping content_type to converter instance.
        _library_map: Dictionary mapping H5P library to content_type.
    """

    def __init__(self, converters: list[BaseH5PConverter] | None = None):
        """Initialize the converter registry.

        Args:
            converters: Optional list of converters to register.
                       If None, default converters will be loaded.
        """
        self._converters: dict[str, BaseH5PConverter] = {}
        self._library_map: dict[str, str] = {}

        if converters:
            for converter in converters:
                self.register(converter)
        else:
            self.load_default_converters()

    def register(self, converter: BaseH5PConverter) -> None:
        """Register a converter in the registry.

        Args:
            converter: Converter instance to register.

        Raises:
            ValueError: If a converter with the same content_type exists.
        """
        content_type = converter.content_type

        if content_type in self._converters:
            logger.warning(
                "Replacing existing converter for content_type: %s",
                content_type,
            )

        self._converters[content_type] = converter
        self._library_map[converter.library] = content_type

        logger.debug(
            "Registered converter: %s (%s)",
            content_type,
            converter.library,
        )

    def get(self, content_type: str) -> BaseH5PConverter | None:
        """Get converter by content type.

        Args:
            content_type: Content type identifier (e.g., "multiple-choice").

        Returns:
            Converter instance or None if not found.
        """
        return self._converters.get(content_type)

    def get_by_library(self, library: str) -> BaseH5PConverter | None:
        """Get converter by H5P library identifier.

        Args:
            library: H5P library identifier (e.g., "H5P.MultiChoice 1.16").

        Returns:
            Converter instance or None if not found.
        """
        content_type = self._library_map.get(library)
        if content_type:
            return self._converters.get(content_type)
        return None

    def has(self, content_type: str) -> bool:
        """Check if a converter is registered for the content type.

        Args:
            content_type: Content type identifier.

        Returns:
            True if converter exists.
        """
        return content_type in self._converters

    def list_content_types(self) -> list[str]:
        """List all registered content types.

        Returns:
            List of content type identifiers.
        """
        return list(self._converters.keys())

    def list_libraries(self) -> list[str]:
        """List all registered H5P libraries.

        Returns:
            List of H5P library identifiers.
        """
        return list(self._library_map.keys())

    def list_by_category(self, category: str) -> list[str]:
        """List content types by category.

        Args:
            category: Category name (assessment, vocabulary, learning, game, media).

        Returns:
            List of content type identifiers in the category.
        """
        return [
            ct for ct, conv in self._converters.items()
            if conv.category == category
        ]

    def list_by_ai_support(self, support_level: str) -> list[str]:
        """List content types by AI support level.

        Args:
            support_level: Support level (full, partial, none).

        Returns:
            List of content type identifiers with the support level.
        """
        return [
            ct for ct, conv in self._converters.items()
            if conv.ai_support == support_level
        ]

    def get_all_info(self) -> list[dict[str, Any]]:
        """Get information about all registered converters.

        Returns:
            List of content type information dictionaries.
        """
        return [conv.get_content_info() for conv in self._converters.values()]

    def get_info(self, content_type: str) -> dict[str, Any] | None:
        """Get information about a specific content type.

        Args:
            content_type: Content type identifier.

        Returns:
            Content type information dictionary or None.
        """
        converter = self.get(content_type)
        if converter:
            return converter.get_content_info()
        return None

    def convert(
        self,
        content_type: str,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content using the appropriate converter.

        Args:
            content_type: Content type identifier.
            ai_content: AI-generated content.
            language: Language code.

        Returns:
            H5P params dictionary.

        Raises:
            ValueError: If converter not found.
            H5PConversionError: If conversion fails.
        """
        converter = self.get(content_type)
        if not converter:
            raise ValueError(f"No converter found for content type: {content_type}")

        return converter.safe_convert(ai_content, language)

    def __len__(self) -> int:
        """Get number of registered converters."""
        return len(self._converters)

    def __contains__(self, content_type: str) -> bool:
        """Check if content type is registered."""
        return content_type in self._converters

    def load_default_converters(self) -> None:
        """Load all default converters.

        This method imports and registers all available converters.
        Called automatically if no converters are provided to __init__.
        """
        # Import converters lazily to avoid circular imports
        try:
            from src.services.h5p.converters.multiple_choice import MultipleChoiceConverter
            self.register(MultipleChoiceConverter())
        except ImportError:
            logger.debug("MultipleChoiceConverter not available")

        try:
            from src.services.h5p.converters.true_false import TrueFalseConverter
            self.register(TrueFalseConverter())
        except ImportError:
            logger.debug("TrueFalseConverter not available")

        try:
            from src.services.h5p.converters.fill_blanks import FillBlanksConverter
            self.register(FillBlanksConverter())
        except ImportError:
            logger.debug("FillBlanksConverter not available")

        try:
            from src.services.h5p.converters.drag_words import DragWordsConverter
            self.register(DragWordsConverter())
        except ImportError:
            logger.debug("DragWordsConverter not available")

        try:
            from src.services.h5p.converters.single_choice_set import SingleChoiceSetConverter
            self.register(SingleChoiceSetConverter())
        except ImportError:
            logger.debug("SingleChoiceSetConverter not available")

        try:
            from src.services.h5p.converters.flashcards import FlashcardsConverter
            self.register(FlashcardsConverter())
        except ImportError:
            logger.debug("FlashcardsConverter not available")

        try:
            from src.services.h5p.converters.dialog_cards import DialogCardsConverter
            self.register(DialogCardsConverter())
        except ImportError:
            logger.debug("DialogCardsConverter not available")

        try:
            from src.services.h5p.converters.mark_words import MarkWordsConverter
            self.register(MarkWordsConverter())
        except ImportError:
            logger.debug("MarkWordsConverter not available")

        try:
            from src.services.h5p.converters.summary import SummaryConverter
            self.register(SummaryConverter())
        except ImportError:
            logger.debug("SummaryConverter not available")

        try:
            from src.services.h5p.converters.crossword import CrosswordConverter
            self.register(CrosswordConverter())
        except ImportError:
            logger.debug("CrosswordConverter not available")

        try:
            from src.services.h5p.converters.word_search import WordSearchConverter
            self.register(WordSearchConverter())
        except ImportError:
            logger.debug("WordSearchConverter not available")

        try:
            from src.services.h5p.converters.accordion import AccordionConverter
            self.register(AccordionConverter())
        except ImportError:
            logger.debug("AccordionConverter not available")

        try:
            from src.services.h5p.converters.sort_paragraphs import SortParagraphsConverter
            self.register(SortParagraphsConverter())
        except ImportError:
            logger.debug("SortParagraphsConverter not available")

        try:
            from src.services.h5p.converters.essay import EssayConverter
            self.register(EssayConverter())
        except ImportError:
            logger.debug("EssayConverter not available")

        try:
            from src.services.h5p.converters.memory_game import MemoryGameConverter
            self.register(MemoryGameConverter())
        except ImportError:
            logger.debug("MemoryGameConverter not available")

        try:
            from src.services.h5p.converters.timeline import TimelineConverter
            self.register(TimelineConverter())
        except ImportError:
            logger.debug("TimelineConverter not available")

        try:
            from src.services.h5p.converters.image_pairing import ImagePairingConverter
            self.register(ImagePairingConverter())
        except ImportError:
            logger.debug("ImagePairingConverter not available")

        try:
            from src.services.h5p.converters.image_sequencing import ImageSequencingConverter
            self.register(ImageSequencingConverter())
        except ImportError:
            logger.debug("ImageSequencingConverter not available")

        try:
            from src.services.h5p.converters.chart import ChartConverter
            self.register(ChartConverter())
        except ImportError:
            logger.debug("ChartConverter not available")

        try:
            from src.services.h5p.converters.personality_quiz import PersonalityQuizConverter
            self.register(PersonalityQuizConverter())
        except ImportError:
            logger.debug("PersonalityQuizConverter not available")

        try:
            from src.services.h5p.converters.question_set import QuestionSetConverter
            self.register(QuestionSetConverter())
        except ImportError:
            logger.debug("QuestionSetConverter not available")

        try:
            from src.services.h5p.converters.course_presentation import CoursePresentationConverter
            self.register(CoursePresentationConverter())
        except ImportError:
            logger.debug("CoursePresentationConverter not available")

        try:
            from src.services.h5p.converters.interactive_book import InteractiveBookConverter
            self.register(InteractiveBookConverter())
        except ImportError:
            logger.debug("InteractiveBookConverter not available")

        try:
            from src.services.h5p.converters.branching_scenario import BranchingScenarioConverter
            self.register(BranchingScenarioConverter())
        except ImportError:
            logger.debug("BranchingScenarioConverter not available")

        try:
            from src.services.h5p.converters.arithmetic_quiz import ArithmeticQuizConverter
            self.register(ArithmeticQuizConverter())
        except ImportError:
            logger.debug("ArithmeticQuizConverter not available")

        try:
            from src.services.h5p.converters.image_hotspots import ImageHotspotsConverter
            self.register(ImageHotspotsConverter())
        except ImportError:
            logger.debug("ImageHotspotsConverter not available")

        try:
            from src.services.h5p.converters.documentation_tool import DocumentationToolConverter
            self.register(DocumentationToolConverter())
        except ImportError:
            logger.debug("DocumentationToolConverter not available")

        try:
            from src.services.h5p.converters.agamotto import AgamottoConverter
            self.register(AgamottoConverter())
        except ImportError:
            logger.debug("AgamottoConverter not available")

        try:
            from src.services.h5p.converters.image_juxtaposition import ImageJuxtapositionConverter
            self.register(ImageJuxtapositionConverter())
        except ImportError:
            logger.debug("ImageJuxtapositionConverter not available")

        try:
            from src.services.h5p.converters.collage import CollageConverter
            self.register(CollageConverter())
        except ImportError:
            logger.debug("CollageConverter not available")

        logger.info(
            "Loaded %d converters into registry",
            len(self._converters),
        )

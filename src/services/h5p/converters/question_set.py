# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Question Set H5P Converter.

Converts AI-generated question set content to H5P.QuestionSet format.
Delegates sub-question conversion to standalone converters.
"""

from typing import Any
from uuid import uuid4

from src.services.h5p.converters.base import BaseH5PConverter, _load_locale_file, _LOCALES_DIR
from src.services.h5p.converters.drag_words import DragWordsConverter
from src.services.h5p.converters.fill_blanks import FillBlanksConverter
from src.services.h5p.converters.mark_words import MarkWordsConverter
from src.services.h5p.converters.multiple_choice import MultipleChoiceConverter
from src.services.h5p.converters.true_false import TrueFalseConverter
from src.services.h5p.exceptions import H5PValidationError


class QuestionSetConverter(BaseH5PConverter):
    """Converter for H5P.QuestionSet content type.

    AI Input Format:
        {
            "title": "Comprehensive Quiz",
            "introPage": {
                "showIntroPage": true,
                "title": "Welcome",
                "introduction": "This quiz covers photosynthesis"
            },
            "questions": [
                {
                    "type": "multiple_choice",
                    "params": {
                        "question": "What gas do plants absorb?",
                        "answers": ["CO2", "O2", "N2", "H2"],
                        "correctIndex": 0
                    }
                },
                {
                    "type": "true_false",
                    "params": {
                        "statement": "Plants produce oxygen",
                        "isTrue": true
                    }
                }
            ],
            "passPercentage": 70,
            "randomQuestions": false
        }

    Sub-question conversion is delegated to standalone converters
    (MultipleChoiceConverter, TrueFalseConverter, etc.) to avoid
    code duplication.
    """

    QUESTION_TYPE_MAP = {
        "multiple_choice": "H5P.MultiChoice 1.16",
        "true_false": "H5P.TrueFalse 1.8",
        "fill_blanks": "H5P.Blanks 1.14",
        "drag_words": "H5P.DragText 1.10",
        "mark_words": "H5P.MarkTheWords 1.11",
    }

    CONTENT_TYPE_LABELS = {
        "multiple_choice": "Multiple Choice",
        "true_false": "True/False Question",
        "fill_blanks": "Fill in the Blanks",
        "drag_words": "Drag the Words",
        "mark_words": "Mark the Words",
    }

    def __init__(self) -> None:
        self._mc_converter = MultipleChoiceConverter()
        self._tf_converter = TrueFalseConverter()
        self._fb_converter = FillBlanksConverter()
        self._dw_converter = DragWordsConverter()
        self._mw_converter = MarkWordsConverter()

    @property
    def content_type(self) -> str:
        return "question-set"

    @property
    def library(self) -> str:
        return "H5P.QuestionSet 1.20"

    @property
    def category(self) -> str:
        return "assessment"

    @property
    def bloom_levels(self) -> list[str]:
        return ["remember", "understand", "apply"]

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "questions" not in ai_content:
            raise H5PValidationError(
                message="Missing 'questions' field",
                content_type=self.content_type,
            )

        questions = ai_content.get("questions", [])
        if not questions:
            raise H5PValidationError(
                message="At least one question is required",
                content_type=self.content_type,
            )

        for i, q in enumerate(questions):
            if "type" not in q:
                raise H5PValidationError(
                    message=f"Question {i+1} missing 'type'",
                    content_type=self.content_type,
                )
            if q["type"] not in self.QUESTION_TYPE_MAP:
                raise H5PValidationError(
                    message=f"Question {i+1} has unsupported type: {q['type']}",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P QuestionSet format."""
        questions = ai_content.get("questions", [])
        intro_page = ai_content.get("introPage", {})
        pass_percentage = ai_content.get("passPercentage", 50)
        random_questions = ai_content.get("randomQuestions", False)

        if not questions:
            raise H5PValidationError(
                message="No questions provided",
                content_type=self.content_type,
            )

        h5p_questions = []
        for q in questions:
            q_type = q.get("type", "multiple_choice")
            q_params = q.get("params", {})
            h5p_question = self._convert_sub_question(q_type, q_params, language)
            h5p_questions.append(h5p_question)

        return {
            "introPage": {
                "showIntroPage": intro_page.get("showIntroPage", True),
                "title": intro_page.get("title", ai_content.get("title", "")),
                "introduction": intro_page.get("introduction", ""),
                "startButtonText": _qs_l10n(language, "endGame", "startButtonText", "Start Quiz"),
            },
            "progressType": ai_content.get("progressType", "dots"),
            "passPercentage": pass_percentage,
            "questions": h5p_questions,
            "texts": self.get_l10n(language).get("texts", self._get_texts(language)),
            "disableBackwardsNavigation": False,
            "randomQuestions": random_questions,
            "endGame": self._get_end_game(language),
            "override": {
                "checkButton": True,
                "showSolutionButton": "on",
                "retryButton": "on",
            },
        }

    # ── Sub-question conversion (delegates to standalone converters) ──

    def _convert_sub_question(
        self,
        q_type: str,
        params: dict[str, Any],
        language: str,
    ) -> dict[str, Any]:
        """Convert a single sub-question by delegating to its standalone converter."""
        library = self.QUESTION_TYPE_MAP.get(q_type, "H5P.MultiChoice 1.16")

        # Delegate to standalone converter (build_media handles all media types)
        h5p_params = self._delegate_convert(q_type, params, language)

        # Clean up empty media placeholder if converter left one
        if "media" in h5p_params and not h5p_params["media"].get("type"):
            del h5p_params["media"]

        # Ensure confirmCheck/confirmRetry are present
        if "confirmCheck" not in h5p_params:
            confirm_check, confirm_retry = _get_confirm_dialogs(language)
            h5p_params["confirmCheck"] = confirm_check
            h5p_params["confirmRetry"] = confirm_retry

        # Build title for metadata
        title = self._extract_title(q_type, params)

        return {
            "library": library,
            "params": h5p_params,
            "metadata": _build_metadata(title, self.CONTENT_TYPE_LABELS.get(q_type, "Question")),
            "subContentId": str(uuid4()),
        }

    def _delegate_convert(
        self,
        q_type: str,
        params: dict[str, Any],
        language: str,
    ) -> dict[str, Any]:
        """Call the appropriate standalone converter's convert_single_params."""
        if q_type == "multiple_choice":
            return self._mc_converter.convert_single_params(params, language)
        elif q_type == "true_false":
            return self._tf_converter.convert_single_params(params, language)
        elif q_type == "fill_blanks":
            return self._fb_converter.convert_single_params(params, language)
        elif q_type == "drag_words":
            return self._dw_converter.convert_single_params(params, language)
        elif q_type == "mark_words":
            return self._mw_converter.convert_single_params(params, language)
        else:
            return self._mc_converter.convert_single_params(params, language)

    @staticmethod
    def _extract_title(q_type: str, params: dict[str, Any]) -> str:
        """Extract a display title from sub-question params."""
        if q_type == "multiple_choice":
            return (params.get("question") or "Untitled Question")[:80]
        elif q_type == "true_false":
            return (params.get("statement") or "Untitled Question")[:80]
        elif q_type in ("fill_blanks", "drag_words"):
            return (params.get("text") or "Untitled Question")[:80]
        elif q_type == "mark_words":
            return (params.get("instruction") or params.get("text") or "Untitled Question")[:80]
        return "Untitled Question"

    # ── Top-level QuestionSet fields ──

    def _get_end_game(self, language: str) -> dict[str, Any]:
        l10n = self.get_l10n(language)
        eg = l10n.get("endGame", {})
        return {
            "showResultPage": True,
            "showSolutionButton": True,
            "showRetryButton": True,
            "noResultMessage": eg.get("noResultMessage", "No result"),
            "message": eg.get("message", "Your result:"),
            "overallFeedback": self.get_overall_feedback(language),
            "scoreBarLabel": eg.get("scoreBarLabel", "Score: :num / :total"),
            "solutionButtonText": eg.get("solutionButtonText", "Show solution"),
            "retryButtonText": eg.get("retryButtonText", "Retry"),
            "finishButtonText": eg.get("finishButtonText", "Finish"),
            "submitButtonText": eg.get("submitButtonText", "Submit"),
            "showAnimations": False,
            "skippable": False,
            "skipButtonText": eg.get("skipButtonText", "Skip video"),
        }

    @staticmethod
    def _get_texts(language: str) -> dict[str, str]:
        """Fallback texts if locale file not available."""
        return {
            "prevButton": "Previous question",
            "nextButton": "Next question",
            "finishButton": "Finish",
            "submitButton": "Submit",
            "textualProgress": "Question: @current of @total",
            "jumpToQuestion": "Question %d of %total",
            "questionLabel": "Question",
            "readSpeakerProgress": "Question @current of @total",
            "unansweredText": "Unanswered",
            "answeredText": "Answered",
            "currentQuestionText": "Current question",
            "navigationLabel": "Questions",
            "questionSetInstruction": "",
        }


# ── Module-level helpers ──


def _qs_l10n(language: str, section: str, key: str, default: str) -> str:
    """Get a question-set locale string."""
    en_data = _load_locale_file(str(_LOCALES_DIR / "question-set" / "en.json"))
    if language != "en":
        lang_data = _load_locale_file(str(_LOCALES_DIR / "question-set" / f"{language}.json"))
        if lang_data:
            return lang_data.get(section, {}).get(key, en_data.get(section, {}).get(key, default))
    return en_data.get(section, {}).get(key, default)


def _build_metadata(title: str, content_type: str) -> dict[str, Any]:
    """Build sub-content metadata."""
    return {
        "contentType": content_type,
        "license": "U",
        "title": title,
        "authors": [],
        "changes": [],
        "extraTitle": title,
    }


def _get_confirm_dialogs(language: str) -> tuple[dict[str, str], dict[str, str]]:
    """Get confirmCheck and confirmRetry dialog strings from locale files."""
    common_en = _load_locale_file(str(_LOCALES_DIR / "_common" / "en.json"))
    confirm_check = dict(common_en.get("confirmCheck", {}))
    confirm_retry = dict(common_en.get("confirmRetry", {}))

    if language != "en":
        common_lang = _load_locale_file(str(_LOCALES_DIR / "_common" / f"{language}.json"))
        if common_lang:
            if "confirmCheck" in common_lang:
                confirm_check.update(common_lang["confirmCheck"])
            if "confirmRetry" in common_lang:
                confirm_retry.update(common_lang["confirmRetry"])

    return (confirm_check, confirm_retry)

# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Practice mode configuration definitions.

Each mode defines behavioral parameters for practice sessions.
These configurations are used by PracticeWorkflow to control:
- Session length and timing
- Difficulty adaptation using ZPD theory (enabled by default for all modes)
- Available aids (hints, skips)
- Focus targeting (weakness areas, mixed topics)

The modes are designed based on educational research:
- Quick: Spaced practice for retention (5 min daily)
- Deep: Extended practice for mastery (Bloom's taxonomy)
- Mixed: Interleaved practice for transfer
- Exam Prep: Testing effect without scaffolds
- Weakness Focus: Targeted practice for struggling areas

Note: adaptive_difficulty is True by default for all modes to ensure
personalized learning. It can only be explicitly disabled if needed.
"""

from enum import Enum
from typing import NamedTuple


class PracticeMode(str, Enum):
    """Available practice modes.

    Each mode has a corresponding configuration in PRACTICE_MODES.
    """

    QUICK = "quick"
    DEEP = "deep"
    MIXED = "mixed"
    RANDOM = "random"  # Random topics from subject - rotates through topics
    EXAM_PREP = "exam_prep"
    WEAKNESS_FOCUS = "weakness_focus"


class PracticeModeConfig(NamedTuple):
    """Configuration for a practice mode.

    All fields are required and define the complete behavior
    of a practice session in this mode.

    Attributes:
        mode: Mode identifier.
        display_name: Human-readable name for UI.
        description: Description explaining the mode's purpose.
        question_count: Default number of questions.
        time_limit_minutes: Time limit in minutes (None = unlimited).
        difficulty_range: Tuple of (min, max) difficulty (0.0-1.0).
        hints_enabled: Whether hints can be requested.
        skip_enabled: Whether questions can be skipped.
        adaptive_difficulty: Whether to use ZPD theory for difficulty.
        focus_on_weakness: Whether to prioritize weak areas from memory.
        allow_topic_mix: Whether to include related topics.
    """

    mode: PracticeMode
    display_name: str
    description: str
    question_count: int
    time_limit_minutes: int | None
    difficulty_range: tuple[float, float]
    hints_enabled: bool
    skip_enabled: bool
    adaptive_difficulty: bool
    focus_on_weakness: bool
    allow_topic_mix: bool


PRACTICE_MODES: dict[PracticeMode, PracticeModeConfig] = {
    PracticeMode.QUICK: PracticeModeConfig(
        mode=PracticeMode.QUICK,
        display_name="Quick Practice",
        description="Short daily practice for retention and review",
        question_count=5,
        time_limit_minutes=10,
        difficulty_range=(0.3, 0.7),
        hints_enabled=True,
        skip_enabled=True,
        adaptive_difficulty=True,
        focus_on_weakness=False,
        allow_topic_mix=False,
    ),
    PracticeMode.DEEP: PracticeModeConfig(
        mode=PracticeMode.DEEP,
        display_name="Deep Learning",
        description="Extended practice session for thorough understanding",
        question_count=20,
        time_limit_minutes=30,
        difficulty_range=(0.2, 0.9),
        hints_enabled=True,
        skip_enabled=True,
        adaptive_difficulty=True,
        focus_on_weakness=False,
        allow_topic_mix=False,
    ),
    PracticeMode.MIXED: PracticeModeConfig(
        mode=PracticeMode.MIXED,
        display_name="Mixed Topics",
        description="Interleaved practice across related topics",
        question_count=10,
        time_limit_minutes=20,
        difficulty_range=(0.3, 0.8),
        hints_enabled=True,
        skip_enabled=True,
        adaptive_difficulty=True,
        focus_on_weakness=False,
        allow_topic_mix=True,
    ),
    PracticeMode.RANDOM: PracticeModeConfig(
        mode=PracticeMode.RANDOM,
        display_name="Random Practice",
        description="Practice with randomly rotating topics from a subject",
        question_count=10,
        time_limit_minutes=20,
        difficulty_range=(0.3, 0.8),
        hints_enabled=True,
        skip_enabled=True,
        adaptive_difficulty=True,
        focus_on_weakness=False,
        allow_topic_mix=True,  # Topics are selected randomly from subject
    ),
    PracticeMode.EXAM_PREP: PracticeModeConfig(
        mode=PracticeMode.EXAM_PREP,
        display_name="Exam Preparation",
        description="Simulate exam conditions without aids",
        question_count=20,
        time_limit_minutes=45,
        difficulty_range=(0.4, 0.9),
        hints_enabled=False,
        skip_enabled=False,
        adaptive_difficulty=True,
        focus_on_weakness=False,
        allow_topic_mix=False,
    ),
    PracticeMode.WEAKNESS_FOCUS: PracticeModeConfig(
        mode=PracticeMode.WEAKNESS_FOCUS,
        display_name="Focus on Weaknesses",
        description="Targeted practice on struggling areas",
        question_count=10,
        time_limit_minutes=15,
        difficulty_range=(0.2, 0.6),
        hints_enabled=True,
        skip_enabled=False,
        adaptive_difficulty=True,
        focus_on_weakness=True,
        allow_topic_mix=False,
    ),
}


def get_mode_config(mode: PracticeMode | str) -> PracticeModeConfig:
    """Get configuration for a practice mode.

    Args:
        mode: Practice mode enum or string value.

    Returns:
        PracticeModeConfig for the specified mode.

    Raises:
        ValueError: If mode is not recognized.

    Example:
        >>> config = get_mode_config("exam_prep")
        >>> config.hints_enabled
        False
    """
    if isinstance(mode, str):
        try:
            mode = PracticeMode(mode)
        except ValueError:
            raise ValueError(
                f"Unknown practice mode: {mode}. "
                f"Valid modes: {[m.value for m in PracticeMode]}"
            )

    if mode not in PRACTICE_MODES:
        raise ValueError(f"No configuration for mode: {mode}")

    return PRACTICE_MODES[mode]


def mode_config_to_dict(config: PracticeModeConfig) -> dict:
    """Convert mode config to dictionary for state storage.

    Args:
        config: Practice mode configuration.

    Returns:
        Dictionary representation suitable for PracticeState.
    """
    return {
        "mode": config.mode.value,
        "display_name": config.display_name,
        "description": config.description,
        "question_count": config.question_count,
        "time_limit_minutes": config.time_limit_minutes,
        "difficulty_range": list(config.difficulty_range),
        "hints_enabled": config.hints_enabled,
        "skip_enabled": config.skip_enabled,
        "adaptive_difficulty": config.adaptive_difficulty,
        "focus_on_weakness": config.focus_on_weakness,
        "allow_topic_mix": config.allow_topic_mix,
    }

# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Practice mode configuration system.

This module provides configuration-based practice modes that control
session behavior including question count, time limits, hint availability,
and adaptive difficulty settings.

Example:
    from src.domains.practice.modes import get_mode_config, PracticeMode

    config = get_mode_config(PracticeMode.EXAM_PREP)
    print(config.hints_enabled)  # False
"""

from src.domains.practice.modes.config import (
    PracticeMode,
    PracticeModeConfig,
    get_mode_config,
    mode_config_to_dict,
    PRACTICE_MODES,
)

__all__ = [
    "PracticeMode",
    "PracticeModeConfig",
    "get_mode_config",
    "mode_config_to_dict",
    "PRACTICE_MODES",
]

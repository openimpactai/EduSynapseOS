# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""YAML configuration file loader utilities.

This module provides utilities for loading YAML configuration files,
including support for loading multiple files from a directory and
deep merging configurations.

Example:
    >>> from pathlib import Path
    >>> from src.core.config.yaml_loader import load_yaml, load_yaml_directory
    >>> config = load_yaml(Path("config/base.yaml"))
    >>> all_configs = load_yaml_directory(Path("config/personas"))
"""

from pathlib import Path
from typing import Any

import yaml


class YAMLLoadError(Exception):
    """Raised when YAML file cannot be loaded or parsed."""

    def __init__(self, path: Path, reason: str) -> None:
        """Initialize YAMLLoadError.

        Args:
            path: Path to the YAML file that failed to load.
            reason: Description of why the file failed to load.
        """
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to load YAML file '{path}': {reason}")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a single YAML file and return its contents as a dictionary.

    Args:
        path: Path to the YAML file to load.

    Returns:
        Dictionary containing the parsed YAML contents.
        Empty dict if file is empty.

    Raises:
        YAMLLoadError: If the file doesn't exist, cannot be read,
            or contains invalid YAML.
    """
    if not path.exists():
        raise YAMLLoadError(path, "File does not exist")

    if not path.is_file():
        raise YAMLLoadError(path, "Path is not a file")

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        raise YAMLLoadError(path, f"Cannot read file: {e}") from e

    try:
        parsed = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise YAMLLoadError(path, f"Invalid YAML syntax: {e}") from e

    # Handle empty files
    if parsed is None:
        return {}

    if not isinstance(parsed, dict):
        raise YAMLLoadError(
            path, f"YAML root must be a mapping, got {type(parsed).__name__}"
        )

    return parsed


def load_yaml_directory(path: Path) -> dict[str, dict[str, Any]]:
    """Load all YAML files from a directory.

    Files are loaded with their stem (filename without extension) as the key.
    Only files with .yaml or .yml extensions are loaded.

    Args:
        path: Path to the directory containing YAML files.

    Returns:
        Dictionary mapping file stems to their parsed contents.
        For example, if directory contains "base.yaml" and "custom.yaml",
        returns {"base": {...}, "custom": {...}}.

    Raises:
        YAMLLoadError: If the path is not a directory or if any
            YAML file fails to load.
    """
    if not path.exists():
        raise YAMLLoadError(path, "Directory does not exist")

    if not path.is_dir():
        raise YAMLLoadError(path, "Path is not a directory")

    result: dict[str, dict[str, Any]] = {}

    yaml_files = sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml"))

    for yaml_file in yaml_files:
        if yaml_file.is_file():
            result[yaml_file.stem] = load_yaml(yaml_file)

    return result


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.

    Nested dictionaries are merged recursively. For non-dict values,
    the override value replaces the base value.

    Args:
        base: The base dictionary to merge into.
        override: The dictionary whose values take precedence.

    Returns:
        A new dictionary containing the merged result.
        Neither input dictionary is modified.

    Example:
        >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> override = {"b": {"c": 10}, "e": 5}
        >>> deep_merge(base, override)
        {"a": 1, "b": {"c": 10, "d": 3}, "e": 5}
    """
    result: dict[str, Any] = base.copy()

    for key, override_value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(override_value, dict)
        ):
            result[key] = deep_merge(result[key], override_value)
        else:
            result[key] = override_value

    return result

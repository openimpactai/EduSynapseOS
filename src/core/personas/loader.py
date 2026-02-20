# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Persona YAML loader for EduSynapseOS.

This module provides functions to load persona definitions from YAML files.
Personas are loaded from the config/personas directory and validated against
the Persona Pydantic model.
"""

from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from src.core.config.yaml_loader import load_yaml, load_yaml_directory
from src.core.personas.models import Persona
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PersonaLoadError(Exception):
    """Raised when a persona fails to load or validate."""

    pass


def get_personas_directory() -> Path:
    """Get the default personas configuration directory.

    Returns:
        Path to the config/personas directory
    """
    return Path(__file__).parent.parent.parent.parent / "config" / "personas"


def load_persona(persona_id: str, personas_dir: Optional[Path] = None) -> Persona:
    """Load a single persona from its YAML file.

    Args:
        persona_id: The persona identifier (matches filename without .yaml)
        personas_dir: Optional path to personas directory (defaults to config/personas)

    Returns:
        Validated Persona object

    Raises:
        PersonaLoadError: If the persona file doesn't exist or validation fails
    """
    if personas_dir is None:
        personas_dir = get_personas_directory()

    persona_file = personas_dir / f"{persona_id}.yaml"

    if not persona_file.exists():
        raise PersonaLoadError(f"Persona file not found: {persona_file}")

    try:
        data = load_yaml(persona_file)
    except Exception as e:
        raise PersonaLoadError(f"Failed to load YAML for persona '{persona_id}': {e}")

    # Get the persona data from the 'persona' key if present
    if "persona" in data:
        persona_data = data["persona"]
    else:
        persona_data = data

    # Ensure the id is set
    if "id" not in persona_data:
        persona_data["id"] = persona_id

    try:
        persona = Persona.model_validate(persona_data)
        logger.debug(
            "loaded_persona",
            persona_id=persona.id,
            name=persona.name,
        )
        return persona
    except ValidationError as e:
        raise PersonaLoadError(
            f"Validation failed for persona '{persona_id}': {e}"
        )


def load_all_personas(personas_dir: Optional[Path] = None) -> dict[str, Persona]:
    """Load all personas from the personas directory.

    Args:
        personas_dir: Optional path to personas directory (defaults to config/personas)

    Returns:
        Dictionary mapping persona IDs to Persona objects

    Raises:
        PersonaLoadError: If the directory doesn't exist or any persona fails to load
    """
    if personas_dir is None:
        personas_dir = get_personas_directory()

    if not personas_dir.exists():
        logger.warning(
            "personas_directory_not_found",
            path=str(personas_dir),
        )
        return {}

    personas: dict[str, Persona] = {}
    errors: list[str] = []

    try:
        all_data = load_yaml_directory(personas_dir)
    except Exception as e:
        raise PersonaLoadError(f"Failed to load personas directory: {e}")

    for filename, data in all_data.items():
        # Extract persona_id from filename (remove .yaml extension)
        persona_id = filename.replace(".yaml", "").replace(".yml", "")

        # Get the persona data from the 'persona' key if present
        if "persona" in data:
            persona_data = data["persona"]
        else:
            persona_data = data

        # Ensure the id is set
        if "id" not in persona_data:
            persona_data["id"] = persona_id

        try:
            persona = Persona.model_validate(persona_data)
            if persona.enabled:
                personas[persona.id] = persona
                logger.debug(
                    "loaded_persona",
                    persona_id=persona.id,
                    name=persona.name,
                )
            else:
                logger.debug(
                    "skipped_disabled_persona",
                    persona_id=persona.id,
                )
        except ValidationError as e:
            error_msg = f"Validation failed for persona '{persona_id}': {e}"
            errors.append(error_msg)
            logger.warning(
                "persona_validation_failed",
                persona_id=persona_id,
                error=str(e),
            )

    if errors:
        logger.warning(
            "some_personas_failed_to_load",
            error_count=len(errors),
            loaded_count=len(personas),
        )

    logger.info(
        "personas_loaded",
        count=len(personas),
        persona_ids=list(personas.keys()),
    )

    return personas


def validate_persona_yaml(yaml_path: Path) -> tuple[bool, Optional[str]]:
    """Validate a persona YAML file without loading it into the system.

    Args:
        yaml_path: Path to the YAML file to validate

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, error_message) if invalid
    """
    try:
        data = load_yaml(yaml_path)

        # Get the persona data
        if "persona" in data:
            persona_data = data["persona"]
        else:
            persona_data = data

        # Extract persona_id from filename
        persona_id = yaml_path.stem
        if "id" not in persona_data:
            persona_data["id"] = persona_id

        Persona.model_validate(persona_data)
        return True, None

    except ValidationError as e:
        return False, f"Validation error: {e}"
    except Exception as e:
        return False, f"Load error: {e}"

# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Configuration management endpoints.

This module provides API endpoints for managing YAML configuration files:
- GET /configs - List all configuration categories
- GET /configs/{category} - List files in a category
- GET /configs/{category}/{filename} - Get specific configuration
- PUT /configs/{category}/{filename} - Update configuration
- POST /configs/{category}/{filename}/backup - Create backup
- GET /configs/{category}/{filename}/history - Get version history
- POST /configs/{category}/{filename}/restore - Restore from backup
- POST /configs/llm/test-connection - Test LLM provider connection
- PUT /configs/llm/providers/{provider_code}/api-key - Update API key
"""

import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_central_db, get_jwt_manager, get_password_hasher
from src.core.config import load_yaml, YAMLLoadError
from src.domains.auth.jwt import JWTManager
from src.domains.auth.password import PasswordHasher
from src.domains.system.auth_service import SystemAuthService

logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration directory paths
CONFIG_BASE_PATH = Path(__file__).parents[4] / "config"
BACKUP_DIR_NAME = "backups"

# Category metadata
CATEGORY_METADATA = {
    "agents": {
        "display_name": "AI Agents",
        "description": "Agent behavior, prompts, and tool configurations",
    },
    "personas": {
        "display_name": "Personas",
        "description": "Character personalities and response templates",
    },
    "theories": {
        "display_name": "Learning Theories",
        "description": "Pedagogical algorithms and parameters",
    },
    "llm": {
        "display_name": "LLM Providers",
        "description": "LLM provider configuration and routing strategies",
    },
    "diagnostics": {
        "display_name": "Diagnostics",
        "description": "Learning difficulty detection thresholds and recommendations",
    },
}

VALID_CATEGORIES = set(CATEGORY_METADATA.keys())


# Response Models
class ConfigCategory(BaseModel):
    """Configuration category information."""

    name: str = Field(..., description="Category identifier")
    display_name: str = Field(..., description="Display name for UI")
    description: str = Field(..., description="Category description")
    file_count: int = Field(..., description="Number of configuration files")
    path: str = Field(..., description="Relative path to category directory")


class CategoriesResponse(BaseModel):
    """Response for list categories endpoint."""

    categories: list[ConfigCategory] = Field(..., description="List of categories")


class ConfigFileSummary(BaseModel):
    """Summary information for LLM providers file."""

    providers_enabled: int = Field(..., description="Number of enabled providers")
    providers_total: int = Field(..., description="Total number of providers")
    default_provider: str = Field(..., description="Default provider code")


class ConfigFile(BaseModel):
    """Configuration file information."""

    filename: str = Field(..., description="File name")
    display_name: str = Field(..., description="Display name from config")
    description: str = Field(..., description="Description from config")
    enabled: bool | None = Field(None, description="Enabled status if applicable")
    last_modified: datetime = Field(..., description="Last modification time")
    size_bytes: int = Field(..., description="File size in bytes")
    summary: ConfigFileSummary | None = Field(
        None, description="Summary for LLM providers"
    )


class CategoryFilesResponse(BaseModel):
    """Response for list files in category endpoint."""

    category: str = Field(..., description="Category name")
    files: list[ConfigFile] = Field(..., description="List of configuration files")
    total_files: int = Field(..., description="Total number of files")


class ConfigContentResponse(BaseModel):
    """Response for get configuration endpoint."""

    filename: str = Field(..., description="File name")
    category: str = Field(..., description="Category name")
    last_modified: datetime = Field(..., description="Last modification time")
    content: dict[str, Any] = Field(..., description="Configuration content")
    schema_version: str = Field(default="1.0", description="Schema version")


class ValidationIssue(BaseModel):
    """Validation error or warning."""

    path: str = Field(..., description="Path to the field with issue")
    message: str = Field(..., description="Issue description")
    value: Any = Field(None, description="Current value causing the issue")


class ValidationResult(BaseModel):
    """Configuration validation result."""

    valid: bool = Field(..., description="Whether configuration is valid")
    errors: list[ValidationIssue] = Field(
        default_factory=list, description="Validation errors"
    )
    warnings: list[ValidationIssue] = Field(
        default_factory=list, description="Validation warnings"
    )


class UpdateConfigRequest(BaseModel):
    """Request body for updating configuration."""

    content: dict[str, Any] = Field(..., description="New configuration content")
    create_backup: bool = Field(default=True, description="Create backup before update")


class UpdateConfigResponse(BaseModel):
    """Response for update configuration endpoint."""

    success: bool = Field(..., description="Whether update was successful")
    filename: str = Field(..., description="File name")
    category: str = Field(..., description="Category name")
    backup_created: bool = Field(..., description="Whether backup was created")
    backup_filename: str | None = Field(None, description="Backup file name if created")
    last_modified: datetime = Field(..., description="New modification time")
    validation: ValidationResult = Field(..., description="Validation result")


class BackupResponse(BaseModel):
    """Response for create backup endpoint."""

    success: bool = Field(..., description="Whether backup was successful")
    backup_filename: str = Field(..., description="Backup file name")
    backup_path: str = Field(..., description="Path to backup directory")
    created_at: datetime = Field(..., description="Backup creation time")
    size_bytes: int = Field(..., description="Backup file size")


class HistoryEntry(BaseModel):
    """Version history entry."""

    backup_filename: str = Field(..., description="Backup file name")
    created_at: datetime = Field(..., description="Backup creation time")
    size_bytes: int = Field(..., description="Backup file size")


class HistoryResponse(BaseModel):
    """Response for version history endpoint."""

    filename: str = Field(..., description="Original file name")
    category: str = Field(..., description="Category name")
    current_version: datetime = Field(
        ..., description="Current file modification time"
    )
    history: list[HistoryEntry] = Field(..., description="Version history")
    total_backups: int = Field(..., description="Total backup count")


class RestoreRequest(BaseModel):
    """Request body for restore endpoint."""

    backup_filename: str = Field(..., description="Backup file to restore from")
    create_backup_of_current: bool = Field(
        default=True, description="Backup current before restore"
    )


class RestoreResponse(BaseModel):
    """Response for restore endpoint."""

    success: bool = Field(..., description="Whether restore was successful")
    restored_from: str = Field(..., description="Backup file restored from")
    current_backed_up_as: str | None = Field(
        None, description="Current file backed up as"
    )
    restored_at: datetime = Field(..., description="Restore time")


class TestConnectionRequest(BaseModel):
    """Request body for test connection endpoint."""

    provider_code: str = Field(..., description="Provider code to test")


class TestConnectionResponse(BaseModel):
    """Response for test connection endpoint."""

    success: bool = Field(..., description="Whether connection succeeded")
    provider_code: str = Field(..., description="Provider code tested")
    status: str = Field(..., description="Provider status")
    latency_ms: int | None = Field(None, description="Response latency in ms")
    models_available: list[str] | None = Field(None, description="Available models")
    error: str | None = Field(None, description="Error message if failed")
    tested_at: datetime = Field(..., description="Test timestamp")


class UpdateApiKeyRequest(BaseModel):
    """Request body for update API key endpoint."""

    api_key: str = Field(..., description="New API key")


class UpdateApiKeyResponse(BaseModel):
    """Response for update API key endpoint."""

    success: bool = Field(..., description="Whether update was successful")
    provider_code: str = Field(..., description="Provider code")
    api_key_masked: str = Field(..., description="Masked API key")
    api_key_set: bool = Field(..., description="Whether API key is now set")
    last_changed: datetime = Field(..., description="Change timestamp")


class ErrorResponse(BaseModel):
    """Error response model."""

    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(None, description="Additional details")


# Helper Functions
async def _verify_system_admin(
    request: Request,
    db: AsyncSession,
    jwt_manager: JWTManager,
    password_hasher: PasswordHasher,
) -> None:
    """Verify the request is from a system admin."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header.split(" ")[1]
    auth_service = SystemAuthService(db, jwt_manager, password_hasher)

    user = await auth_service.get_current_user(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _validate_category(category: str) -> None:
    """Validate category name."""
    if category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error="not_found",
                message=f"Category '{category}' not found",
                details={"valid_categories": list(VALID_CATEGORIES)},
            ).model_dump(),
        )


def _validate_filename(filename: str) -> None:
    """Validate filename format."""
    if not re.match(r"^[\w\-]+\.ya?ml$", filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(
                error="invalid_filename",
                message="Invalid filename format",
                details={"expected_format": "alphanumeric_with_dashes.yaml"},
            ).model_dump(),
        )


def _get_category_path(category: str) -> Path:
    """Get path to category directory."""
    return CONFIG_BASE_PATH / category


def _get_file_path(category: str, filename: str) -> Path:
    """Get path to configuration file."""
    return CONFIG_BASE_PATH / category / filename


def _get_backup_dir(category: str) -> Path:
    """Get path to backup directory for a category."""
    return CONFIG_BASE_PATH / category / BACKUP_DIR_NAME


def _mask_api_key(key: str | None) -> str:
    """Mask API key for display."""
    if not key:
        return ""
    if len(key) <= 4:
        return "●" * len(key)
    return "●" * (len(key) - 4) + key[-4:]


def _generate_backup_filename(filename: str) -> str:
    """Generate backup filename with timestamp."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d%H%M%S")
    return f"{filename}.{timestamp}.bak"


def _extract_display_info(content: dict[str, Any], category: str) -> tuple[str, str, bool | None]:
    """Extract display name, description, and enabled status from config content."""
    display_name = ""
    description = ""
    enabled = None

    if category == "agents":
        agent = content.get("agent", {})
        display_name = agent.get("name", "")
        description = agent.get("description", "")[:150] if agent.get("description") else ""
    elif category == "personas":
        persona = content.get("persona", {})
        display_name = persona.get("name", "")
        description = persona.get("description", "")[:150] if persona.get("description") else ""
        enabled = persona.get("enabled", True)
    elif category == "theories":
        display_name = content.get("name", {}).get("en", "") or Path().stem
        description = f"Weight: {content.get('weight', 1.0)}"
        enabled = content.get("enabled", True)
    elif category == "llm":
        display_name = "LLM Providers"
        description = "Provider and routing configuration"
    elif category == "diagnostics":
        display_name = "Diagnostics"
        description = "Threshold and recommendation configuration"

    return display_name, description, enabled


def _mask_llm_config(content: dict[str, Any]) -> dict[str, Any]:
    """Mask sensitive fields in LLM provider configuration."""
    if "providers" not in content:
        return content

    masked = content.copy()
    masked_providers = {}

    for code, provider in content.get("providers", {}).items():
        masked_provider = provider.copy()
        api_key = masked_provider.pop("api_key", "")
        masked_provider["api_key_masked"] = _mask_api_key(api_key)
        masked_provider["api_key_set"] = bool(api_key)
        masked_providers[code] = masked_provider

    masked["providers"] = masked_providers
    return masked


def _validate_agent_config(content: dict[str, Any]) -> ValidationResult:
    """Validate agent configuration."""
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    agent = content.get("agent", {})

    # Required fields
    if not agent.get("id"):
        errors.append(ValidationIssue(path="agent.id", message="Required field"))

    if not agent.get("name"):
        errors.append(ValidationIssue(path="agent.name", message="Required field"))

    # LLM configuration
    llm = agent.get("llm", {})
    temperature = llm.get("temperature")
    if temperature is not None and not (0.0 <= temperature <= 1.0):
        errors.append(
            ValidationIssue(
                path="agent.llm.temperature",
                message="Must be between 0.0 and 1.0",
                value=temperature,
            )
        )

    max_tokens = llm.get("max_tokens")
    if max_tokens is not None and not (256 <= max_tokens <= 8192):
        warnings.append(
            ValidationIssue(
                path="agent.llm.max_tokens",
                message="Recommended range is 256-8192",
                value=max_tokens,
            )
        )

    timeout = llm.get("timeout_seconds")
    if timeout is not None and not (30 <= timeout <= 300):
        warnings.append(
            ValidationIssue(
                path="agent.llm.timeout_seconds",
                message="Recommended range is 30-300",
                value=timeout,
            )
        )

    # Persona reference validation
    persona = agent.get("default_persona")
    if persona:
        persona_path = _get_file_path("personas", f"{persona}.yaml")
        if not persona_path.exists():
            errors.append(
                ValidationIssue(
                    path="agent.default_persona",
                    message=f"Persona '{persona}' does not exist",
                    value=persona,
                )
            )

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def _validate_persona_config(content: dict[str, Any]) -> ValidationResult:
    """Validate persona configuration."""
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    persona = content.get("persona", {})

    # Required fields
    if not persona.get("id"):
        errors.append(ValidationIssue(path="persona.id", message="Required field"))

    if not persona.get("name"):
        errors.append(ValidationIssue(path="persona.name", message="Required field"))

    # Behavior validation
    behavior = persona.get("behavior", {})
    socratic = behavior.get("socratic_tendency")
    if socratic is not None and not (0.0 <= socratic <= 1.0):
        errors.append(
            ValidationIssue(
                path="persona.behavior.socratic_tendency",
                message="Must be between 0.0 and 1.0",
                value=socratic,
            )
        )

    hint_eagerness = behavior.get("hint_eagerness")
    if hint_eagerness is not None and not (0.0 <= hint_eagerness <= 1.0):
        errors.append(
            ValidationIssue(
                path="persona.behavior.hint_eagerness",
                message="Must be between 0.0 and 1.0",
                value=hint_eagerness,
            )
        )

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def _validate_theory_config(content: dict[str, Any]) -> ValidationResult:
    """Validate theory configuration."""
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    # Weight validation
    weight = content.get("weight")
    if weight is not None and not (0.0 <= weight <= 2.0):
        warnings.append(
            ValidationIssue(
                path="weight",
                message="Recommended range is 0.0-2.0",
                value=weight,
            )
        )

    # Parameters validation
    params = content.get("parameters", {})

    # Check for indicator weights sum (diagnostics thresholds)
    if "indicators" in params:
        indicators = params["indicators"]
        if isinstance(indicators, dict):
            total_weight = sum(
                ind.get("weight", 0) for ind in indicators.values() if isinstance(ind, dict)
            )
            if abs(total_weight - 1.0) > 0.01:
                errors.append(
                    ValidationIssue(
                        path="parameters.indicators",
                        message=f"Weights must sum to 1.0, got {total_weight:.2f}",
                        value=total_weight,
                    )
                )

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def _validate_llm_config(content: dict[str, Any]) -> ValidationResult:
    """Validate LLM provider configuration."""
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    providers = content.get("providers", {})
    default_provider = content.get("default_provider")

    # Check default provider exists and is enabled
    if default_provider:
        if default_provider not in providers:
            errors.append(
                ValidationIssue(
                    path="default_provider",
                    message=f"Default provider '{default_provider}' not found in providers",
                    value=default_provider,
                )
            )
        elif not providers.get(default_provider, {}).get("enabled", False):
            warnings.append(
                ValidationIssue(
                    path="default_provider",
                    message=f"Default provider '{default_provider}' is not enabled",
                    value=default_provider,
                )
            )

    # Validate individual providers
    for code, provider in providers.items():
        provider_type = provider.get("type")

        # Type-specific validation
        if provider_type == "ollama":
            if not provider.get("api_base"):
                errors.append(
                    ValidationIssue(
                        path=f"providers.{code}.api_base",
                        message="API base URL required for Ollama providers",
                    )
                )
        elif provider_type in ("openai", "anthropic", "google"):
            if provider.get("enabled") and not provider.get("api_key"):
                warnings.append(
                    ValidationIssue(
                        path=f"providers.{code}.api_key",
                        message=f"API key required for enabled {provider_type} provider",
                    )
                )

        # Timeout validation
        timeout = provider.get("timeout_seconds")
        if timeout is not None and not (30 <= timeout <= 300):
            warnings.append(
                ValidationIssue(
                    path=f"providers.{code}.timeout_seconds",
                    message="Recommended range is 30-300",
                    value=timeout,
                )
            )

    # Validate routing strategies
    strategies = content.get("routing_strategies", {})
    for name, strategy in strategies.items():
        primary = strategy.get("primary")
        if primary and primary not in providers:
            errors.append(
                ValidationIssue(
                    path=f"routing_strategies.{name}.primary",
                    message=f"Primary provider '{primary}' not found",
                    value=primary,
                )
            )

        for fallback in strategy.get("fallbacks", []):
            if fallback not in providers:
                errors.append(
                    ValidationIssue(
                        path=f"routing_strategies.{name}.fallbacks",
                        message=f"Fallback provider '{fallback}' not found",
                        value=fallback,
                    )
                )

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def _validate_diagnostics_config(content: dict[str, Any]) -> ValidationResult:
    """Validate diagnostics configuration."""
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    # Thresholds file validation
    if "diagnostics" in content:
        diagnostics = content["diagnostics"]
        thresholds = diagnostics.get("thresholds", {})

        for detector_name, detector in thresholds.items():
            if not isinstance(detector, dict):
                continue

            concern = detector.get("concern_threshold")
            alert = detector.get("alert_threshold")

            if concern is not None and alert is not None:
                if concern >= alert:
                    errors.append(
                        ValidationIssue(
                            path=f"diagnostics.thresholds.{detector_name}",
                            message="concern_threshold must be less than alert_threshold",
                            value={"concern": concern, "alert": alert},
                        )
                    )

            # Validate indicator weights
            indicators = detector.get("indicators", {})
            if indicators:
                total_weight = sum(
                    ind.get("weight", 0)
                    for ind in indicators.values()
                    if isinstance(ind, dict)
                )
                if abs(total_weight - 1.0) > 0.01:
                    errors.append(
                        ValidationIssue(
                            path=f"diagnostics.thresholds.{detector_name}.indicators",
                            message=f"Weights must sum to 1.0, got {total_weight:.2f}",
                            value=total_weight,
                        )
                    )

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def _validate_config(content: dict[str, Any], category: str) -> ValidationResult:
    """Validate configuration based on category."""
    validators = {
        "agents": _validate_agent_config,
        "personas": _validate_persona_config,
        "theories": _validate_theory_config,
        "llm": _validate_llm_config,
        "diagnostics": _validate_diagnostics_config,
    }

    validator = validators.get(category)
    if validator:
        return validator(content)

    return ValidationResult(valid=True, errors=[], warnings=[])


# Endpoints
@router.get(
    "/configs",
    response_model=CategoriesResponse,
    summary="List configuration categories",
    description="List all available configuration categories.",
)
async def list_categories(
    request: Request,
    db: AsyncSession = Depends(get_central_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
    password_hasher: PasswordHasher = Depends(get_password_hasher),
) -> CategoriesResponse:
    """List all configuration categories."""
    await _verify_system_admin(request, db, jwt_manager, password_hasher)

    categories = []
    for name, metadata in CATEGORY_METADATA.items():
        category_path = _get_category_path(name)
        file_count = 0

        if category_path.exists() and category_path.is_dir():
            yaml_files = list(category_path.glob("*.yaml")) + list(
                category_path.glob("*.yml")
            )
            file_count = len([f for f in yaml_files if f.is_file()])

        categories.append(
            ConfigCategory(
                name=name,
                display_name=metadata["display_name"],
                description=metadata["description"],
                file_count=file_count,
                path=f"/config/{name}",
            )
        )

    return CategoriesResponse(categories=categories)


@router.get(
    "/configs/{category}",
    response_model=CategoryFilesResponse,
    summary="List files in category",
    description="List all configuration files in a category.",
)
async def list_category_files(
    request: Request,
    category: str,
    db: AsyncSession = Depends(get_central_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
    password_hasher: PasswordHasher = Depends(get_password_hasher),
) -> CategoryFilesResponse:
    """List all configuration files in a category."""
    await _verify_system_admin(request, db, jwt_manager, password_hasher)
    _validate_category(category)

    category_path = _get_category_path(category)
    files = []

    if category_path.exists() and category_path.is_dir():
        yaml_files = sorted(category_path.glob("*.yaml")) + sorted(
            category_path.glob("*.yml")
        )

        for file_path in yaml_files:
            if not file_path.is_file() or file_path.name.endswith(".bak"):
                continue

            stat = file_path.stat()
            last_modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

            # Load content for display info
            display_name = file_path.stem
            description = ""
            enabled = None
            summary = None

            try:
                content = load_yaml(file_path)
                display_name, description, enabled = _extract_display_info(
                    content, category
                )

                # Generate summary for LLM providers
                if category == "llm" and "providers" in content:
                    providers = content.get("providers", {})
                    enabled_count = sum(
                        1 for p in providers.values() if p.get("enabled", False)
                    )
                    summary = ConfigFileSummary(
                        providers_enabled=enabled_count,
                        providers_total=len(providers),
                        default_provider=content.get("default_provider", ""),
                    )

                if not display_name:
                    display_name = file_path.stem

            except YAMLLoadError:
                logger.warning(f"Failed to load {file_path} for display info")

            files.append(
                ConfigFile(
                    filename=file_path.name,
                    display_name=display_name,
                    description=description,
                    enabled=enabled,
                    last_modified=last_modified,
                    size_bytes=stat.st_size,
                    summary=summary,
                )
            )

    return CategoryFilesResponse(
        category=category,
        files=files,
        total_files=len(files),
    )


@router.get(
    "/configs/{category}/{filename}",
    response_model=ConfigContentResponse,
    summary="Get configuration",
    description="Get the content of a specific configuration file.",
)
async def get_config(
    request: Request,
    category: str,
    filename: str,
    db: AsyncSession = Depends(get_central_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
    password_hasher: PasswordHasher = Depends(get_password_hasher),
) -> ConfigContentResponse:
    """Get the content of a specific configuration file."""
    await _verify_system_admin(request, db, jwt_manager, password_hasher)
    _validate_category(category)
    _validate_filename(filename)

    file_path = _get_file_path(category, filename)

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error="not_found",
                message=f"Configuration file '{filename}' not found in '{category}'",
                details={"category": category, "filename": filename},
            ).model_dump(),
        )

    try:
        content = load_yaml(file_path)
    except YAMLLoadError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="read_failed",
                message=f"Failed to read configuration: {e.reason}",
            ).model_dump(),
        ) from e

    stat = file_path.stat()
    last_modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

    # Mask sensitive data for LLM providers
    if category == "llm":
        content = _mask_llm_config(content)

    return ConfigContentResponse(
        filename=filename,
        category=category,
        last_modified=last_modified,
        content=content,
        schema_version="1.0",
    )


@router.put(
    "/configs/{category}/{filename}",
    response_model=UpdateConfigResponse,
    summary="Update configuration",
    description="Update the content of a specific configuration file.",
)
async def update_config(
    request: Request,
    category: str,
    filename: str,
    body: UpdateConfigRequest,
    db: AsyncSession = Depends(get_central_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
    password_hasher: PasswordHasher = Depends(get_password_hasher),
) -> UpdateConfigResponse:
    """Update the content of a specific configuration file."""
    await _verify_system_admin(request, db, jwt_manager, password_hasher)
    _validate_category(category)
    _validate_filename(filename)

    file_path = _get_file_path(category, filename)

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error="not_found",
                message=f"Configuration file '{filename}' not found in '{category}'",
            ).model_dump(),
        )

    # Validate configuration
    validation = _validate_config(body.content, category)
    if not validation.valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(
                error="validation_failed",
                message="Configuration validation failed",
                details={"validation": validation.model_dump()},
            ).model_dump(),
        )

    # Create backup if requested
    backup_filename = None
    if body.create_backup:
        backup_dir = _get_backup_dir(category)
        backup_dir.mkdir(parents=True, exist_ok=True)

        backup_filename = _generate_backup_filename(filename)
        backup_path = backup_dir / backup_filename

        try:
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
        except OSError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ErrorResponse(
                    error="backup_failed",
                    message=f"Failed to create backup: {e}",
                ).model_dump(),
            ) from e

    # For LLM config, preserve existing API keys if not provided
    content_to_write = body.content
    if category == "llm":
        try:
            existing_content = load_yaml(file_path)
            content_to_write = _preserve_api_keys(existing_content, body.content)
        except YAMLLoadError:
            pass

    # Write updated configuration
    try:
        yaml_content = yaml.dump(
            content_to_write,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        file_path.write_text(yaml_content, encoding="utf-8")
        logger.info(f"Updated configuration: {file_path}")
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="write_failed",
                message=f"Failed to write configuration: {e}",
            ).model_dump(),
        ) from e

    stat = file_path.stat()
    last_modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

    return UpdateConfigResponse(
        success=True,
        filename=filename,
        category=category,
        backup_created=backup_filename is not None,
        backup_filename=backup_filename,
        last_modified=last_modified,
        validation=validation,
    )


def _preserve_api_keys(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Preserve API keys from existing config when updating LLM providers."""
    result = new.copy()

    if "providers" in result and "providers" in existing:
        for code, provider in result.get("providers", {}).items():
            # If api_key_masked is present, restore original key
            if "api_key_masked" in provider:
                del provider["api_key_masked"]
            if "api_key_set" in provider:
                del provider["api_key_set"]

            # If api_key is empty or missing, use existing
            if not provider.get("api_key") and code in existing.get("providers", {}):
                existing_key = existing["providers"][code].get("api_key", "")
                provider["api_key"] = existing_key

    return result


@router.post(
    "/configs/{category}/{filename}/backup",
    response_model=BackupResponse,
    summary="Create backup",
    description="Create a backup of a configuration file.",
)
async def create_backup(
    request: Request,
    category: str,
    filename: str,
    db: AsyncSession = Depends(get_central_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
    password_hasher: PasswordHasher = Depends(get_password_hasher),
) -> BackupResponse:
    """Create a backup of a configuration file."""
    await _verify_system_admin(request, db, jwt_manager, password_hasher)
    _validate_category(category)
    _validate_filename(filename)

    file_path = _get_file_path(category, filename)

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error="not_found",
                message=f"Configuration file '{filename}' not found",
            ).model_dump(),
        )

    backup_dir = _get_backup_dir(category)
    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_filename = _generate_backup_filename(filename)
    backup_path = backup_dir / backup_filename

    try:
        shutil.copy2(file_path, backup_path)
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="backup_failed",
                message=f"Failed to create backup: {e}",
            ).model_dump(),
        ) from e

    stat = backup_path.stat()

    return BackupResponse(
        success=True,
        backup_filename=backup_filename,
        backup_path=str(backup_dir.relative_to(CONFIG_BASE_PATH)),
        created_at=datetime.now(timezone.utc),
        size_bytes=stat.st_size,
    )


@router.get(
    "/configs/{category}/{filename}/history",
    response_model=HistoryResponse,
    summary="Get version history",
    description="Get the backup history of a configuration file.",
)
async def get_history(
    request: Request,
    category: str,
    filename: str,
    limit: int = Query(default=10, ge=1, le=100),
    db: AsyncSession = Depends(get_central_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
    password_hasher: PasswordHasher = Depends(get_password_hasher),
) -> HistoryResponse:
    """Get the backup history of a configuration file."""
    await _verify_system_admin(request, db, jwt_manager, password_hasher)
    _validate_category(category)
    _validate_filename(filename)

    file_path = _get_file_path(category, filename)

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error="not_found",
                message=f"Configuration file '{filename}' not found",
            ).model_dump(),
        )

    stat = file_path.stat()
    current_version = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

    backup_dir = _get_backup_dir(category)
    history = []
    total_backups = 0

    if backup_dir.exists():
        pattern = f"{filename}.*.bak"
        backup_files = sorted(
            backup_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        total_backups = len(backup_files)

        for backup_path in backup_files[:limit]:
            backup_stat = backup_path.stat()
            created_at = datetime.fromtimestamp(backup_stat.st_mtime, tz=timezone.utc)

            history.append(
                HistoryEntry(
                    backup_filename=backup_path.name,
                    created_at=created_at,
                    size_bytes=backup_stat.st_size,
                )
            )

    return HistoryResponse(
        filename=filename,
        category=category,
        current_version=current_version,
        history=history,
        total_backups=total_backups,
    )


@router.post(
    "/configs/{category}/{filename}/restore",
    response_model=RestoreResponse,
    summary="Restore from backup",
    description="Restore a configuration file from a backup.",
)
async def restore_backup(
    request: Request,
    category: str,
    filename: str,
    body: RestoreRequest,
    db: AsyncSession = Depends(get_central_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
    password_hasher: PasswordHasher = Depends(get_password_hasher),
) -> RestoreResponse:
    """Restore a configuration file from a backup."""
    await _verify_system_admin(request, db, jwt_manager, password_hasher)
    _validate_category(category)
    _validate_filename(filename)

    file_path = _get_file_path(category, filename)
    backup_dir = _get_backup_dir(category)
    backup_path = backup_dir / body.backup_filename

    if not backup_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error="not_found",
                message=f"Backup file '{body.backup_filename}' not found",
            ).model_dump(),
        )

    # Backup current file if requested
    current_backup_filename = None
    if body.create_backup_of_current and file_path.exists():
        current_backup_filename = _generate_backup_filename(filename)
        current_backup_path = backup_dir / current_backup_filename

        try:
            shutil.copy2(file_path, current_backup_path)
        except OSError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ErrorResponse(
                    error="backup_failed",
                    message=f"Failed to backup current file: {e}",
                ).model_dump(),
            ) from e

    # Restore from backup
    try:
        shutil.copy2(backup_path, file_path)
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="restore_failed",
                message=f"Failed to restore backup: {e}",
            ).model_dump(),
        ) from e

    return RestoreResponse(
        success=True,
        restored_from=body.backup_filename,
        current_backed_up_as=current_backup_filename,
        restored_at=datetime.now(timezone.utc),
    )


@router.post(
    "/configs/llm/test-connection",
    response_model=TestConnectionResponse,
    summary="Test LLM provider connection",
    description="Test connection to an LLM provider.",
)
async def test_llm_connection(
    request: Request,
    body: TestConnectionRequest,
    db: AsyncSession = Depends(get_central_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
    password_hasher: PasswordHasher = Depends(get_password_hasher),
) -> TestConnectionResponse:
    """Test connection to an LLM provider."""
    await _verify_system_admin(request, db, jwt_manager, password_hasher)

    llm_config_path = _get_file_path("llm", "providers.yaml")

    if not llm_config_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error="not_found",
                message="LLM providers configuration not found",
            ).model_dump(),
        )

    try:
        config = load_yaml(llm_config_path)
    except YAMLLoadError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="read_failed",
                message=f"Failed to read LLM configuration: {e.reason}",
            ).model_dump(),
        ) from e

    providers = config.get("providers", {})
    provider = providers.get(body.provider_code)

    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error="not_found",
                message=f"Provider '{body.provider_code}' not found",
                details={"available_providers": list(providers.keys())},
            ).model_dump(),
        )

    now = datetime.now(timezone.utc)
    provider_type = provider.get("type", "")
    api_base = provider.get("api_base", "")

    # Test connection based on provider type
    if provider_type == "ollama":
        try:
            import time
            start_time = time.time()

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{api_base}/api/tags")
                latency_ms = int((time.time() - start_time) * 1000)

                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("name", "") for m in data.get("models", [])]

                    return TestConnectionResponse(
                        success=True,
                        provider_code=body.provider_code,
                        status="healthy",
                        latency_ms=latency_ms,
                        models_available=models,
                        tested_at=now,
                    )
                else:
                    return TestConnectionResponse(
                        success=False,
                        provider_code=body.provider_code,
                        status="unhealthy",
                        error=f"HTTP {response.status_code}",
                        tested_at=now,
                    )
        except httpx.TimeoutException:
            return TestConnectionResponse(
                success=False,
                provider_code=body.provider_code,
                status="unhealthy",
                error="Connection timeout",
                tested_at=now,
            )
        except Exception as e:
            return TestConnectionResponse(
                success=False,
                provider_code=body.provider_code,
                status="unhealthy",
                error=str(e),
                tested_at=now,
            )
    else:
        # For cloud providers, just check if API key is set
        api_key = provider.get("api_key", "")
        if api_key:
            return TestConnectionResponse(
                success=True,
                provider_code=body.provider_code,
                status="configured",
                tested_at=now,
            )
        else:
            return TestConnectionResponse(
                success=False,
                provider_code=body.provider_code,
                status="not_configured",
                error="API key not set",
                tested_at=now,
            )


@router.put(
    "/configs/llm/providers/{provider_code}/api-key",
    response_model=UpdateApiKeyResponse,
    summary="Update LLM provider API key",
    description="Update the API key for an LLM provider.",
)
async def update_api_key(
    request: Request,
    provider_code: str,
    body: UpdateApiKeyRequest,
    db: AsyncSession = Depends(get_central_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
    password_hasher: PasswordHasher = Depends(get_password_hasher),
) -> UpdateApiKeyResponse:
    """Update the API key for an LLM provider."""
    await _verify_system_admin(request, db, jwt_manager, password_hasher)

    llm_config_path = _get_file_path("llm", "providers.yaml")

    if not llm_config_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error="not_found",
                message="LLM providers configuration not found",
            ).model_dump(),
        )

    try:
        config = load_yaml(llm_config_path)
    except YAMLLoadError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="read_failed",
                message=f"Failed to read LLM configuration: {e.reason}",
            ).model_dump(),
        ) from e

    providers = config.get("providers", {})

    if provider_code not in providers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error="not_found",
                message=f"Provider '{provider_code}' not found",
            ).model_dump(),
        )

    # Create backup before modifying
    backup_dir = _get_backup_dir("llm")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_filename = _generate_backup_filename("providers.yaml")
    backup_path = backup_dir / backup_filename

    try:
        shutil.copy2(llm_config_path, backup_path)
    except OSError as e:
        logger.warning(f"Failed to create backup before API key update: {e}")

    # Update API key
    config["providers"][provider_code]["api_key"] = body.api_key

    # Write updated configuration
    try:
        yaml_content = yaml.dump(
            config,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        llm_config_path.write_text(yaml_content, encoding="utf-8")
        logger.info(f"Updated API key for provider: {provider_code}")
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="write_failed",
                message=f"Failed to write configuration: {e}",
            ).model_dump(),
        ) from e

    return UpdateApiKeyResponse(
        success=True,
        provider_code=provider_code,
        api_key_masked=_mask_api_key(body.api_key),
        api_key_set=bool(body.api_key),
        last_changed=datetime.now(timezone.utc),
    )

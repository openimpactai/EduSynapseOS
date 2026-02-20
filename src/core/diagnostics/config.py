# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Diagnostic configuration management.

Loads and provides access to diagnostic thresholds, recommendations,
and other configuration from YAML files.

Usage:
    from src.core.diagnostics.config import get_diagnostic_config

    config = get_diagnostic_config()

    # Get detector threshold
    dyslexia_config = config.get_detector_config("dyslexia")
    print(f"Alert threshold: {dyslexia_config.alert_threshold}")

    # Get recommendations
    recs = config.get_recommendations("dyslexia", "concern", "for_teacher", "en")

    # Get disclaimer
    disclaimer = config.get_disclaimer("en")
"""

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.core.config.yaml_loader import YAMLLoadError, load_yaml

logger = logging.getLogger(__name__)

# Default config directory - can be overridden by DIAGNOSTIC_CONFIG_DIR env var
CONFIG_DIR = Path(
    os.environ.get(
        "DIAGNOSTIC_CONFIG_DIR",
        Path(__file__).parents[3] / "config" / "diagnostics",
    )
)


@dataclass
class IndicatorConfig:
    """Configuration for a single indicator.

    Attributes:
        weight: How much this indicator contributes to overall score (0.0-1.0).
        threshold: Threshold value for this specific indicator.
    """

    weight: float
    threshold: float


@dataclass
class DetectorConfig:
    """Configuration for a detector type.

    Attributes:
        name: Localized names {lang: name}.
        alert_threshold: Risk score threshold for alert level.
        concern_threshold: Risk score threshold for concern level.
        min_data_points: Minimum data points required for analysis.
        scan_frequency_days: Recommended days between scans.
        indicators: Configuration for individual indicators.
    """

    name: dict[str, str]
    alert_threshold: float
    concern_threshold: float
    min_data_points: int
    scan_frequency_days: int
    indicators: dict[str, IndicatorConfig] = field(default_factory=dict)

    def get_name(self, lang: str = "en") -> str:
        """Get localized name.

        Args:
            lang: Language code (en, tr).

        Returns:
            Localized name or English fallback.
        """
        return self.name.get(lang, self.name.get("en", "Unknown"))


@dataclass
class GlobalSettings:
    """Global diagnostic settings.

    Attributes:
        min_data_points: Default minimum data points.
        confidence_threshold: Minimum confidence for results.
        scan_frequency_days: Default scan frequency.
        require_parental_consent: Whether parental consent is required.
    """

    min_data_points: int = 30
    confidence_threshold: float = 0.6
    scan_frequency_days: int = 7
    require_parental_consent: bool = True


@dataclass
class NotificationConfig:
    """Notification settings for a recipient type.

    Attributes:
        threshold: Risk score threshold for notification.
        message_type: Type of message (concern_gentle, professional, formal).
        include_recommendations: Include recommendations in notification.
        include_evidence: Include evidence details.
        include_full_report: Include full report.
    """

    threshold: float
    message_type: str
    include_recommendations: bool = False
    include_evidence: bool = False
    include_full_report: bool = False


@dataclass
class DiagnosticConfig:
    """Complete diagnostic configuration.

    Provides access to thresholds, recommendations, disclaimers,
    and notification settings loaded from YAML files.
    """

    disclaimer: dict[str, str]
    global_settings: GlobalSettings
    thresholds: dict[str, DetectorConfig]
    notifications: dict[str, NotificationConfig]
    recommendations: dict[str, Any]

    def get_disclaimer(self, lang: str = "en") -> str:
        """Get localized disclaimer.

        Args:
            lang: Language code.

        Returns:
            Disclaimer text.
        """
        return self.disclaimer.get(lang, self.disclaimer.get("en", ""))

    def get_detector_config(self, detector_type: str) -> DetectorConfig | None:
        """Get configuration for a specific detector type.

        Args:
            detector_type: Type of detector (dyslexia, dyscalculia, etc.).

        Returns:
            DetectorConfig or None if not configured.
        """
        return self.thresholds.get(detector_type)

    def get_recommendations(
        self,
        detector_type: str,
        level: str = "concern",
        audience: str = "for_teacher",
        lang: str = "en",
    ) -> list[str]:
        """Get localized recommendations.

        Args:
            detector_type: Type of detector (e.g., "dyslexia").
            level: Concern level ("concern" or "alert").
            audience: Target audience ("for_teacher" or "for_parent").
            lang: Language code.

        Returns:
            List of recommendation strings.
        """
        detector_recs = self.recommendations.get(detector_type, {})
        level_recs = detector_recs.get(level, {})
        audience_recs = level_recs.get(audience, [])

        return [
            rec.get(lang, rec.get("en", str(rec)))
            for rec in audience_recs
            if isinstance(rec, dict)
        ]

    def get_general_recommendations(self, lang: str = "en") -> list[str]:
        """Get general recommendations applicable to all indicators.

        Args:
            lang: Language code.

        Returns:
            List of general recommendation strings.
        """
        general = self.recommendations.get("general", [])
        return [
            rec.get(lang, rec.get("en", str(rec)))
            for rec in general
            if isinstance(rec, dict)
        ]

    def get_notification_config(self, recipient: str) -> NotificationConfig | None:
        """Get notification configuration for a recipient type.

        Args:
            recipient: Recipient type (parent, teacher, professional_referral).

        Returns:
            NotificationConfig or None.
        """
        return self.notifications.get(recipient)


def _parse_indicator_config(data: dict) -> IndicatorConfig:
    """Parse indicator configuration from YAML data.

    Args:
        data: Raw YAML data for indicator.

    Returns:
        IndicatorConfig instance.
    """
    return IndicatorConfig(
        weight=float(data.get("weight", 0.0)),
        threshold=float(data.get("threshold", 0.0)),
    )


def _parse_detector_config(data: dict) -> DetectorConfig:
    """Parse detector configuration from YAML data.

    Args:
        data: Raw YAML data for detector.

    Returns:
        DetectorConfig instance.
    """
    name = data.get("name", {})
    if isinstance(name, str):
        name = {"en": name}

    indicators = {}
    for ind_name, ind_data in data.get("indicators", {}).items():
        indicators[ind_name] = _parse_indicator_config(ind_data)

    return DetectorConfig(
        name=name,
        alert_threshold=float(data.get("alert_threshold", 0.6)),
        concern_threshold=float(data.get("concern_threshold", 0.4)),
        min_data_points=int(data.get("min_data_points", 30)),
        scan_frequency_days=int(data.get("scan_frequency_days", 7)),
        indicators=indicators,
    )


def _parse_notification_config(data: dict) -> NotificationConfig:
    """Parse notification configuration from YAML data.

    Args:
        data: Raw YAML data for notification.

    Returns:
        NotificationConfig instance.
    """
    return NotificationConfig(
        threshold=float(data.get("threshold", 0.5)),
        message_type=str(data.get("message_type", "standard")),
        include_recommendations=bool(data.get("include_recommendations", False)),
        include_evidence=bool(data.get("include_evidence", False)),
        include_full_report=bool(data.get("include_full_report", False)),
    )


def _parse_global_settings(data: dict) -> GlobalSettings:
    """Parse global settings from YAML data.

    Args:
        data: Raw YAML data for global settings.

    Returns:
        GlobalSettings instance.
    """
    return GlobalSettings(
        min_data_points=int(data.get("min_data_points", 30)),
        confidence_threshold=float(data.get("confidence_threshold", 0.6)),
        scan_frequency_days=int(data.get("scan_frequency_days", 7)),
        require_parental_consent=bool(data.get("require_parental_consent", True)),
    )


@lru_cache(maxsize=1)
def load_diagnostic_config(config_dir: str | None = None) -> DiagnosticConfig:
    """Load diagnostic configuration from YAML files.

    Uses LRU cache to avoid reloading on every access.
    Call `load_diagnostic_config.cache_clear()` to reload.

    Args:
        config_dir: Optional config directory override (as string for caching).

    Returns:
        DiagnosticConfig instance.
    """
    if config_dir is None:
        dir_path = CONFIG_DIR
    else:
        dir_path = Path(config_dir)

    logger.debug("Loading diagnostic config from: %s", dir_path)

    # Load thresholds
    try:
        thresholds_path = dir_path / "thresholds.yaml"
        thresholds_data = load_yaml(thresholds_path)
    except YAMLLoadError as e:
        logger.warning("Failed to load thresholds config: %s", e)
        thresholds_data = {}

    # Load recommendations
    try:
        recommendations_path = dir_path / "recommendations.yaml"
        recommendations_data = load_yaml(recommendations_path)
    except YAMLLoadError as e:
        logger.warning("Failed to load recommendations config: %s", e)
        recommendations_data = {}

    # Parse diagnostics section
    diagnostics = thresholds_data.get("diagnostics", {})

    # Parse disclaimer
    disclaimer = diagnostics.get("disclaimer", {})
    if isinstance(disclaimer, str):
        disclaimer = {"en": disclaimer}

    # Parse global settings
    global_settings = _parse_global_settings(
        diagnostics.get("global_settings", {})
    )

    # Parse detector thresholds
    thresholds = {}
    for detector_type, detector_data in diagnostics.get("thresholds", {}).items():
        thresholds[detector_type] = _parse_detector_config(detector_data)

    # Parse notifications
    notifications = {}
    for notif_type, notif_data in diagnostics.get("notifications", {}).items():
        notifications[notif_type] = _parse_notification_config(notif_data)

    # Get recommendations
    recommendations = recommendations_data.get("recommendations", {})

    config = DiagnosticConfig(
        disclaimer=disclaimer,
        global_settings=global_settings,
        thresholds=thresholds,
        notifications=notifications,
        recommendations=recommendations,
    )

    logger.info(
        "Loaded diagnostic config: %d detectors, %d notification types",
        len(thresholds),
        len(notifications),
    )

    return config


def get_diagnostic_config() -> DiagnosticConfig:
    """Get the cached diagnostic configuration.

    Returns:
        DiagnosticConfig instance.
    """
    return load_diagnostic_config()


def reload_diagnostic_config() -> DiagnosticConfig:
    """Force reload of diagnostic configuration.

    Clears the cache and loads fresh configuration.

    Returns:
        Fresh DiagnosticConfig instance.
    """
    load_diagnostic_config.cache_clear()
    return load_diagnostic_config()

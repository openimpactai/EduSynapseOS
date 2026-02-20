# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Timeline H5P Converter.

Converts AI-generated timeline content to H5P.Timeline format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class TimelineConverter(BaseH5PConverter):
    """Converter for H5P.Timeline content type.

    AI Input Format:
        {
            "title": "History of Computing",
            "headline": "Major Events in Computing History",
            "events": [
                {
                    "startDate": "1946",
                    "endDate": "1946",
                    "headline": "ENIAC",
                    "text": "First general-purpose electronic computer",
                    "media": {
                        "url": "eniac.jpg",
                        "caption": "ENIAC computer",
                        "credit": "Public domain"
                    }
                }
            ],
            "eras": [
                {
                    "startDate": "1940",
                    "endDate": "1960",
                    "headline": "First Generation"
                }
            ]
        }

    Events with dates, descriptions, and optional media.
    """

    @property
    def content_type(self) -> str:
        return "timeline"

    @property
    def library(self) -> str:
        return "H5P.Timeline 1.1"

    @property
    def category(self) -> str:
        return "game"

    @property
    def bloom_levels(self) -> list[str]:
        return ["understand"]

    @property
    def ai_support(self) -> str:
        return "partial"

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "events" not in ai_content:
            raise H5PValidationError(
                message="Missing 'events' field",
                content_type=self.content_type,
            )

        events = ai_content.get("events", [])
        if not events:
            raise H5PValidationError(
                message="At least one event is required",
                content_type=self.content_type,
            )

        for i, event in enumerate(events):
            if "startDate" not in event:
                raise H5PValidationError(
                    message=f"Event {i+1} missing 'startDate'",
                    content_type=self.content_type,
                )
            if "headline" not in event:
                raise H5PValidationError(
                    message=f"Event {i+1} missing 'headline'",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P Timeline format."""
        events = ai_content.get("events", [])
        eras = ai_content.get("eras", [])
        title = ai_content.get("title", "Timeline")
        headline = ai_content.get("headline", title)
        text = ai_content.get("text", "")

        if not events:
            raise H5PValidationError(
                message="No events provided",
                content_type=self.content_type,
            )

        # Convert events to Timeline JSON format
        timeline_events = []
        for event in events:
            tl_event = {
                "start_date": self._parse_date(event.get("startDate", "")),
                "text": {
                    "headline": event.get("headline", ""),
                    "text": f"<p>{event.get('text', '')}</p>",
                },
            }

            # Add end date if provided
            if event.get("endDate"):
                tl_event["end_date"] = self._parse_date(event.get("endDate"))

            # Add media if provided
            if event.get("media"):
                media = event["media"]
                tl_event["media"] = {
                    "url": media.get("url", ""),
                    "caption": media.get("caption", ""),
                    "credit": media.get("credit", ""),
                }

            timeline_events.append(tl_event)

        # Build Timeline JSON
        timeline_data = {
            "title": {
                "text": {
                    "headline": headline,
                    "text": f"<p>{text}</p>" if text else "",
                },
            },
            "events": timeline_events,
        }

        # Add eras if provided
        if eras:
            timeline_data["eras"] = [
                {
                    "start_date": self._parse_date(era.get("startDate", "")),
                    "end_date": self._parse_date(era.get("endDate", "")),
                    "text": {
                        "headline": era.get("headline", ""),
                    },
                }
                for era in eras
            ]

        h5p_params = {
            "timeline": timeline_data,
            "height": 600,
            "language": self._get_timeline_language(language),
        }

        return h5p_params

    def _parse_date(self, date_str: str) -> dict[str, Any]:
        """Parse date string into Timeline date object."""
        if not date_str:
            return {"year": 0}

        # Handle various date formats
        date_str = str(date_str).strip()

        # Just a year
        if date_str.isdigit():
            return {"year": int(date_str)}

        # Year with BC/BCE
        if date_str.upper().endswith(("BC", "BCE")):
            year_str = date_str[:-2].strip() if date_str.upper().endswith("BC") else date_str[:-3].strip()
            if year_str.isdigit():
                return {"year": -int(year_str)}

        # Try YYYY-MM-DD format
        parts = date_str.split("-")
        if len(parts) >= 1:
            result = {}
            try:
                result["year"] = int(parts[0])
                if len(parts) >= 2:
                    result["month"] = int(parts[1])
                if len(parts) >= 3:
                    result["day"] = int(parts[2])
                return result
            except ValueError:
                pass

        # Default to just returning the string as year if parsing fails
        return {"year": 0, "display_date": date_str}

    def _get_timeline_language(self, language: str) -> str:
        """Get Timeline.js language code."""
        language_map = {
            "tr": "tr",
            "en": "en",
        }
        return language_map.get(language, "en")

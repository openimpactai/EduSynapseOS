# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Chart H5P Converter.

Converts AI-generated chart data to H5P.Chart format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class ChartConverter(BaseH5PConverter):
    """Converter for H5P.Chart content type.

    AI Input Format:
        {
            "title": "Population by Country",
            "chartType": "bar",
            "data": [
                {"label": "China", "value": 1400},
                {"label": "India", "value": 1380},
                {"label": "USA", "value": 330}
            ],
            "xAxisLabel": "Country",
            "yAxisLabel": "Population (millions)",
            "colors": ["#4285F4", "#EA4335", "#34A853"]
        }

    Supports bar and pie chart types.
    """

    DEFAULT_COLORS = [
        "#4285F4",  # Blue
        "#EA4335",  # Red
        "#FBBC04",  # Yellow
        "#34A853",  # Green
        "#FF6D01",  # Orange
        "#46BDC6",  # Teal
        "#7BAAF7",  # Light Blue
        "#F07B72",  # Light Red
    ]

    @property
    def content_type(self) -> str:
        return "chart"

    @property
    def library(self) -> str:
        return "H5P.Chart 1.2"

    @property
    def category(self) -> str:
        return "media"

    @property
    def bloom_levels(self) -> list[str]:
        return ["understand"]

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "data" not in ai_content:
            raise H5PValidationError(
                message="Missing 'data' field",
                content_type=self.content_type,
            )

        data = ai_content.get("data", [])
        if not data:
            raise H5PValidationError(
                message="At least one data point is required",
                content_type=self.content_type,
            )

        for i, point in enumerate(data):
            if "label" not in point:
                raise H5PValidationError(
                    message=f"Data point {i+1} missing 'label'",
                    content_type=self.content_type,
                )
            if "value" not in point:
                raise H5PValidationError(
                    message=f"Data point {i+1} missing 'value'",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P Chart format."""
        data = ai_content.get("data", [])
        title = ai_content.get("title", "")
        chart_type = ai_content.get("chartType", "bar")
        x_axis_label = ai_content.get("xAxisLabel", "")
        y_axis_label = ai_content.get("yAxisLabel", "")
        custom_colors = ai_content.get("colors", [])

        if not data:
            raise H5PValidationError(
                message="No data provided",
                content_type=self.content_type,
            )

        # Assign colors
        colors = custom_colors if custom_colors else self.DEFAULT_COLORS

        # Convert data to H5P format
        chart_data = []
        for i, point in enumerate(data):
            color = colors[i % len(colors)]
            if isinstance(point.get("color"), str):
                color = point["color"]

            chart_data.append({
                "text": str(point.get("label", "")),
                "value": float(point.get("value", 0)),
                "color": color,
            })

        # Determine graph mode
        graph_mode = "pie" if chart_type == "pie" else "bar"

        h5p_params = {
            "graphMode": graph_mode,
            "listOfTypes": chart_data,
        }

        # Add title if provided
        if title:
            h5p_params["title"] = title

        # Add axis labels for bar charts
        if graph_mode == "bar":
            if x_axis_label:
                h5p_params["xAxisTitle"] = x_axis_label
            if y_axis_label:
                h5p_params["yAxisTitle"] = y_axis_label

        return h5p_params

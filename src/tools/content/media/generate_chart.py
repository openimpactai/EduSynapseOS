# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate Chart Tool.

Creates bar charts, pie charts, and line graphs for H5P content.
"""

import logging
from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class GenerateChartTool(BaseTool):
    """Generate charts for H5P.Chart content type.

    Creates data visualizations including bar charts, pie charts,
    and line graphs. Output is compatible with H5P.Chart format.

    Example usage by agent:
        - "Create a bar chart showing population data"
        - "Generate a pie chart of energy sources"
        - "Make a line graph showing temperature changes"
    """

    CHART_TYPES = ["bar", "pie", "line"]
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
    def name(self) -> str:
        return "generate_chart"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "generate_chart",
                "description": (
                    "Generate a chart (bar, pie, or line) for H5P content. "
                    "Provide data points with labels and values."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chart_type": {
                            "type": "string",
                            "enum": ["bar", "pie", "line"],
                            "description": "Type of chart to generate",
                        },
                        "title": {
                            "type": "string",
                            "description": "Chart title",
                        },
                        "data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "value": {"type": "number"},
                                    "color": {"type": "string"},
                                },
                                "required": ["label", "value"],
                            },
                            "description": "Data points with labels and values",
                        },
                        "x_axis_label": {
                            "type": "string",
                            "description": "Label for X axis (bar/line charts)",
                        },
                        "y_axis_label": {
                            "type": "string",
                            "description": "Label for Y axis (bar/line charts)",
                        },
                        "show_legend": {
                            "type": "boolean",
                            "description": "Whether to show legend (default true)",
                        },
                        "show_values": {
                            "type": "boolean",
                            "description": "Whether to show values on chart (default true)",
                        },
                    },
                    "required": ["chart_type", "title", "data"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the chart generation."""
        chart_type = params.get("chart_type", "bar")
        title = params.get("title", "Untitled Chart")
        data = params.get("data", [])
        x_axis_label = params.get("x_axis_label", "")
        y_axis_label = params.get("y_axis_label", "")
        show_legend = params.get("show_legend", True)
        show_values = params.get("show_values", True)

        if not data:
            return ToolResult(
                success=False,
                data={"message": "At least one data point is required"},
                error="Missing required parameter: data",
            )

        # Validate chart type
        if chart_type not in self.CHART_TYPES:
            chart_type = "bar"

        # Validate data points
        for i, point in enumerate(data):
            if "label" not in point:
                return ToolResult(
                    success=False,
                    data={"message": f"Data point {i+1} missing 'label'"},
                    error="Invalid data point",
                )
            if "value" not in point:
                return ToolResult(
                    success=False,
                    data={"message": f"Data point {i+1} missing 'value'"},
                    error="Invalid data point",
                )

        try:
            # Generate H5P Chart params
            h5p_params = self._generate_h5p_chart(
                chart_type=chart_type,
                title=title,
                data=data,
                x_axis_label=x_axis_label,
                y_axis_label=y_axis_label,
                show_legend=show_legend,
                show_values=show_values,
            )

            logger.info(
                "Generated chart: type=%s, title=%s, data_points=%d",
                chart_type,
                title,
                len(data),
            )

            return ToolResult(
                success=True,
                data={
                    "chart_type": chart_type,
                    "title": title,
                    "data_points": len(data),
                    "h5p_params": h5p_params,
                    "message": f"Generated {chart_type} chart with {len(data)} data points.",
                },
                state_update={
                    "last_generated_chart": {
                        "type": chart_type,
                        "title": title,
                        "h5p_params": h5p_params,
                    },
                },
            )

        except Exception as e:
            logger.exception("Error generating chart")
            return ToolResult(
                success=False,
                data={"message": f"Failed to generate chart: {e}"},
                error=str(e),
            )

    def _generate_h5p_chart(
        self,
        chart_type: str,
        title: str,
        data: list[dict],
        x_axis_label: str,
        y_axis_label: str,
        show_legend: bool,
        show_values: bool,
    ) -> dict[str, Any]:
        """Generate H5P.Chart compatible params."""
        # Assign colors if not provided
        for i, point in enumerate(data):
            if "color" not in point:
                point["color"] = self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]

        if chart_type == "pie":
            return self._generate_pie_chart(title, data, show_legend, show_values)
        elif chart_type == "line":
            return self._generate_line_chart(
                title, data, x_axis_label, y_axis_label, show_legend, show_values
            )
        else:  # bar
            return self._generate_bar_chart(
                title, data, x_axis_label, y_axis_label, show_legend, show_values
            )

    def _generate_bar_chart(
        self,
        title: str,
        data: list[dict],
        x_axis_label: str,
        y_axis_label: str,
        show_legend: bool,
        show_values: bool,
    ) -> dict[str, Any]:
        """Generate H5P bar chart params."""
        chart_data = []
        for point in data:
            chart_data.append({
                "text": point.get("label", ""),
                "value": point.get("value", 0),
                "color": point.get("color", self.DEFAULT_COLORS[0]),
            })

        return {
            "graphMode": "bar",
            "listOfTypes": chart_data,
            "title": title,
            "xAxisTitle": x_axis_label,
            "yAxisTitle": y_axis_label,
            "showLegend": show_legend,
            "showValues": show_values,
        }

    def _generate_pie_chart(
        self,
        title: str,
        data: list[dict],
        show_legend: bool,
        show_values: bool,
    ) -> dict[str, Any]:
        """Generate H5P pie chart params."""
        chart_data = []
        for point in data:
            chart_data.append({
                "text": point.get("label", ""),
                "value": point.get("value", 0),
                "color": point.get("color", self.DEFAULT_COLORS[0]),
            })

        return {
            "graphMode": "pie",
            "listOfTypes": chart_data,
            "title": title,
            "showLegend": show_legend,
            "showValues": show_values,
        }

    def _generate_line_chart(
        self,
        title: str,
        data: list[dict],
        x_axis_label: str,
        y_axis_label: str,
        show_legend: bool,
        show_values: bool,
    ) -> dict[str, Any]:
        """Generate H5P line chart params (using bar chart with line mode)."""
        chart_data = []
        for point in data:
            chart_data.append({
                "text": point.get("label", ""),
                "value": point.get("value", 0),
                "color": point.get("color", self.DEFAULT_COLORS[0]),
            })

        # H5P.Chart doesn't have native line mode, use bar with styling
        # In production, consider using a different visualization library
        return {
            "graphMode": "bar",  # H5P.Chart limitation
            "listOfTypes": chart_data,
            "title": f"{title} (Line Chart)",
            "xAxisTitle": x_axis_label,
            "yAxisTitle": y_axis_label,
            "showLegend": show_legend,
            "showValues": show_values,
        }

# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate Diagram Tool.

Creates diagrams, flowcharts, and mind maps for educational content.
"""

import logging
from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class GenerateDiagramTool(BaseTool):
    """Generate diagrams and flowcharts.

    Creates visual diagrams for educational content including
    flowcharts, mind maps, process diagrams, and org charts.

    Example usage by agent:
        - "Create a flowchart showing the scientific method"
        - "Generate a mind map about the solar system"
        - "Make a process diagram for photosynthesis"
    """

    DIAGRAM_TYPES = ["flowchart", "mindmap", "process", "org", "sequence", "cycle"]
    NODE_TYPES = ["start", "end", "process", "decision", "input", "output", "data"]

    @property
    def name(self) -> str:
        return "generate_diagram"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "generate_diagram",
                "description": (
                    "Generate an educational diagram such as a flowchart, mind map, "
                    "or process diagram. Provide nodes and connections to define the structure."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "diagram_type": {
                            "type": "string",
                            "enum": ["flowchart", "mindmap", "process", "org", "sequence", "cycle"],
                            "description": "Type of diagram to generate",
                        },
                        "title": {
                            "type": "string",
                            "description": "Title for the diagram",
                        },
                        "nodes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "label": {"type": "string"},
                                    "type": {
                                        "type": "string",
                                        "enum": ["start", "end", "process", "decision", "input", "output", "data"],
                                    },
                                    "description": {"type": "string"},
                                },
                                "required": ["id", "label"],
                            },
                            "description": "Nodes/boxes in the diagram",
                        },
                        "connections": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string"},
                                    "to": {"type": "string"},
                                    "label": {"type": "string"},
                                },
                                "required": ["from", "to"],
                            },
                            "description": "Connections/arrows between nodes",
                        },
                        "style": {
                            "type": "string",
                            "enum": ["default", "colorful", "minimal", "professional"],
                            "description": "Visual style of the diagram",
                        },
                    },
                    "required": ["diagram_type", "title", "nodes"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the diagram generation."""
        diagram_type = params.get("diagram_type", "flowchart")
        title = params.get("title", "Untitled Diagram")
        nodes = params.get("nodes", [])
        connections = params.get("connections", [])
        style = params.get("style", "default")

        if not nodes:
            return ToolResult(
                success=False,
                data={"message": "At least one node is required"},
                error="Missing required parameter: nodes",
            )

        # Validate diagram type
        if diagram_type not in self.DIAGRAM_TYPES:
            diagram_type = "flowchart"

        # Validate node IDs are unique
        node_ids = [n.get("id") for n in nodes]
        if len(node_ids) != len(set(node_ids)):
            return ToolResult(
                success=False,
                data={"message": "Node IDs must be unique"},
                error="Duplicate node IDs",
            )

        # Validate connections reference valid nodes
        for conn in connections:
            if conn.get("from") not in node_ids or conn.get("to") not in node_ids:
                return ToolResult(
                    success=False,
                    data={"message": "Connection references invalid node ID"},
                    error="Invalid connection",
                )

        try:
            # Generate diagram
            diagram_data = self._generate_diagram(
                diagram_type=diagram_type,
                title=title,
                nodes=nodes,
                connections=connections,
                style=style,
            )

            # Generate Mermaid syntax for rendering
            mermaid_code = self._to_mermaid(diagram_type, nodes, connections)

            logger.info(
                "Generated diagram: type=%s, title=%s, nodes=%d, connections=%d",
                diagram_type,
                title,
                len(nodes),
                len(connections),
            )

            return ToolResult(
                success=True,
                data={
                    "diagram_type": diagram_type,
                    "title": title,
                    "nodes": nodes,
                    "connections": connections,
                    "mermaid_code": mermaid_code,
                    "style": style,
                    "message": f"Generated {diagram_type} diagram with {len(nodes)} nodes.",
                },
                state_update={
                    "last_generated_diagram": {
                        "type": diagram_type,
                        "title": title,
                        "mermaid_code": mermaid_code,
                    },
                },
            )

        except Exception as e:
            logger.exception("Error generating diagram")
            return ToolResult(
                success=False,
                data={"message": f"Failed to generate diagram: {e}"},
                error=str(e),
            )

    def _generate_diagram(
        self,
        diagram_type: str,
        title: str,
        nodes: list[dict],
        connections: list[dict],
        style: str,
    ) -> dict[str, Any]:
        """Generate diagram data structure."""
        return {
            "type": diagram_type,
            "title": title,
            "nodes": nodes,
            "connections": connections,
            "style": style,
            "metadata": {
                "node_count": len(nodes),
                "connection_count": len(connections),
            },
        }

    def _to_mermaid(
        self,
        diagram_type: str,
        nodes: list[dict],
        connections: list[dict],
    ) -> str:
        """Convert diagram to Mermaid syntax for rendering."""
        if diagram_type == "flowchart":
            return self._flowchart_to_mermaid(nodes, connections)
        elif diagram_type == "mindmap":
            return self._mindmap_to_mermaid(nodes, connections)
        elif diagram_type == "sequence":
            return self._sequence_to_mermaid(nodes, connections)
        elif diagram_type == "cycle":
            return self._cycle_to_mermaid(nodes, connections)
        else:
            return self._flowchart_to_mermaid(nodes, connections)

    def _flowchart_to_mermaid(
        self,
        nodes: list[dict],
        connections: list[dict],
    ) -> str:
        """Convert to Mermaid flowchart syntax."""
        lines = ["flowchart TD"]

        # Add nodes
        for node in nodes:
            node_id = node.get("id", "")
            label = node.get("label", "")
            node_type = node.get("type", "process")

            # Format based on type
            if node_type == "start" or node_type == "end":
                lines.append(f"    {node_id}(({label}))")
            elif node_type == "decision":
                lines.append(f"    {node_id}{{{label}}}")
            elif node_type == "input" or node_type == "output":
                lines.append(f"    {node_id}[/{label}/]")
            else:
                lines.append(f"    {node_id}[{label}]")

        # Add connections
        for conn in connections:
            from_id = conn.get("from", "")
            to_id = conn.get("to", "")
            label = conn.get("label", "")

            if label:
                lines.append(f"    {from_id} -->|{label}| {to_id}")
            else:
                lines.append(f"    {from_id} --> {to_id}")

        return "\n".join(lines)

    def _mindmap_to_mermaid(
        self,
        nodes: list[dict],
        connections: list[dict],
    ) -> str:
        """Convert to Mermaid mindmap syntax."""
        lines = ["mindmap"]

        # Find root node (node with no incoming connections)
        incoming = {conn.get("to") for conn in connections}
        root_nodes = [n for n in nodes if n.get("id") not in incoming]

        if root_nodes:
            root = root_nodes[0]
            lines.append(f"  root(({root.get('label', 'Root')}))")

            # Build hierarchy
            self._add_mindmap_children(lines, root.get("id"), nodes, connections, 2)

        return "\n".join(lines)

    def _add_mindmap_children(
        self,
        lines: list[str],
        parent_id: str,
        nodes: list[dict],
        connections: list[dict],
        depth: int,
    ) -> None:
        """Recursively add mindmap children."""
        indent = "  " * depth
        children_ids = [c.get("to") for c in connections if c.get("from") == parent_id]

        for child_id in children_ids:
            child = next((n for n in nodes if n.get("id") == child_id), None)
            if child:
                lines.append(f"{indent}{child.get('label', '')}")
                self._add_mindmap_children(lines, child_id, nodes, connections, depth + 1)

    def _sequence_to_mermaid(
        self,
        nodes: list[dict],
        connections: list[dict],
    ) -> str:
        """Convert to Mermaid sequence diagram syntax."""
        lines = ["sequenceDiagram"]

        # Add participants
        for node in nodes:
            lines.append(f"    participant {node.get('id')} as {node.get('label', node.get('id'))}")

        # Add messages
        for conn in connections:
            label = conn.get("label", "")
            lines.append(f"    {conn.get('from')}->>+{conn.get('to')}: {label}")

        return "\n".join(lines)

    def _cycle_to_mermaid(
        self,
        nodes: list[dict],
        connections: list[dict],
    ) -> str:
        """Convert to Mermaid flowchart with cycle styling."""
        lines = ["flowchart LR"]

        # Add nodes in circular layout style
        for node in nodes:
            node_id = node.get("id", "")
            label = node.get("label", "")
            lines.append(f"    {node_id}(({label}))")

        # Add connections
        for conn in connections:
            from_id = conn.get("from", "")
            to_id = conn.get("to", "")
            label = conn.get("label", "")

            if label:
                lines.append(f"    {from_id} -->|{label}| {to_id}")
            else:
                lines.append(f"    {from_id} --> {to_id}")

        return "\n".join(lines)

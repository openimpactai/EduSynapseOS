# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Documentation Tool H5P Converter.

Converts AI-generated documentation templates to H5P.DocumentationTool format.
"""

from typing import Any
from uuid import uuid4

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class DocumentationToolConverter(BaseH5PConverter):
    """Converter for H5P.DocumentationTool content type.

    AI Input Format:
        {
            "title": "Science Lab Report Template",
            "pages": [
                {
                    "title": "Hypothesis",
                    "introduction": "State your prediction.",
                    "fields": [
                        {
                            "type": "textarea",
                            "label": "Your Hypothesis",
                            "description": "Write a testable hypothesis.",
                            "required": true
                        }
                    ]
                }
            ]
        }

    Field types: text (display), textarea (input), goal, assessment.
    """

    VALID_FIELD_TYPES = {"text", "textarea", "goal", "assessment"}

    @property
    def content_type(self) -> str:
        return "documentation-tool"

    @property
    def library(self) -> str:
        return "H5P.DocumentationTool 1.8"

    @property
    def category(self) -> str:
        return "learning"

    @property
    def bloom_levels(self) -> list[str]:
        return ["apply", "create"]

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "pages" not in ai_content:
            raise H5PValidationError(
                message="Missing 'pages' field",
                content_type=self.content_type,
            )

        pages = ai_content.get("pages", [])
        if not pages:
            raise H5PValidationError(
                message="At least one page is required",
                content_type=self.content_type,
            )

        for i, page in enumerate(pages):
            if "title" not in page:
                raise H5PValidationError(
                    message=f"Page {i+1} missing 'title'",
                    content_type=self.content_type,
                )
            if "fields" not in page or not page["fields"]:
                raise H5PValidationError(
                    message=f"Page {i+1} must have at least one field",
                    content_type=self.content_type,
                )

            for j, field in enumerate(page.get("fields", [])):
                field_type = field.get("type", "textarea")
                if field_type not in self.VALID_FIELD_TYPES:
                    raise H5PValidationError(
                        message=f"Page {i+1}, Field {j+1}: invalid type '{field_type}'",
                        content_type=self.content_type,
                    )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P DocumentationTool format."""
        title = ai_content.get("title", "Documentation")
        pages = ai_content.get("pages", [])

        # Build pages in H5P format
        pages_list = []
        for page in pages:
            page_title = page.get("title", "")
            introduction = page.get("introduction", "")
            fields = page.get("fields", [])

            # Build element list for this page
            element_list = []

            # Add title and introduction as text
            intro_text = f"<h2>{page_title}</h2>"
            if introduction:
                intro_text += f"<p>{introduction}</p>"

            element_list.append({
                "library": "H5P.Text 1.1",
                "params": {
                    "text": intro_text,
                },
            })

            # Add input fields
            for field in fields:
                field_type = field.get("type", "textarea")
                label = field.get("label", "")
                description = field.get("description", "")

                if field_type == "text":
                    # Display-only text
                    element_list.append({
                        "library": "H5P.Text 1.1",
                        "params": {
                            "text": f"<p><strong>{label}</strong>: {description}</p>",
                        },
                    })
                elif field_type == "textarea":
                    # Input field
                    element_list.append({
                        "library": "H5P.TextInputField 1.2",
                        "params": {
                            "taskDescription": label,
                            "placeholderText": description,
                            "inputFieldSize": 10,
                        },
                    })
                elif field_type == "goal":
                    # Goal setting field
                    element_list.append({
                        "library": "H5P.GoalsPage 1.5",
                        "params": {
                            "title": label,
                            "description": description,
                        },
                    })
                elif field_type == "assessment":
                    # Self-assessment field
                    element_list.append({
                        "library": "H5P.GoalsAssessmentPage 1.5",
                        "params": {
                            "title": label,
                            "description": description,
                        },
                    })

            page_params = {
                "library": "H5P.StandardPage 1.5",
                "params": {
                    "elementList": element_list,
                    "helpTextLabel": self.get_l10n(language).get("helpLabel", "Help"),
                    "helpText": "",
                },
            }
            pages_list.append(page_params)

        h5p_params = {
            "taskDescription": title,
            "pagesList": pages_list,
            "enableDocumentExport": True,
            "l10n": self.get_l10n(language),
        }

        return h5p_params


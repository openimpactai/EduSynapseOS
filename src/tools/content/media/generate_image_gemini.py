# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate Image with Gemini Tool.

Uses Google's Gemini/Imagen API to generate educational images.
"""

import base64
import logging
from typing import Any

import aiohttp

from src.core.config import get_settings
from src.core.tools import BaseTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class GenerateImageGeminiTool(BaseTool):
    """Generate images using Google Gemini/Imagen.

    Creates educational images based on text prompts.
    Supports various styles and aspect ratios.

    Uses the Google Generative AI API for image generation.
    Generated images are uploaded to the tenant's media storage.

    Example usage by agent:
        - "Generate an image of a butterfly"
        - "Create a picture showing photosynthesis"
        - "Make an illustration for the water cycle"
    """

    SUPPORTED_STYLES = ["cartoon", "realistic", "diagram", "illustration", "sketch"]
    SUPPORTED_RATIOS = ["1:1", "16:9", "4:3", "3:4", "9:16"]

    # Gemini API endpoint for image generation
    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    @property
    def name(self) -> str:
        return "generate_image_gemini"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "generate_image_gemini",
                "description": (
                    "Generate an educational image using Google Gemini/Imagen AI. "
                    "Provide a detailed prompt describing what the image should show. "
                    "Best for illustrations, diagrams, and educational visuals."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": (
                                "Detailed description of the image to generate. "
                                "Include subject, style, colors, mood, and educational context."
                            ),
                        },
                        "style": {
                            "type": "string",
                            "enum": ["cartoon", "realistic", "diagram", "illustration", "sketch"],
                            "description": "Visual style of the image",
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "enum": ["1:1", "16:9", "4:3", "3:4", "9:16"],
                            "description": "Aspect ratio of the image",
                        },
                        "purpose": {
                            "type": "string",
                            "description": "How the image will be used (e.g., 'flashcard front', 'quiz illustration')",
                        },
                        "alt_text": {
                            "type": "string",
                            "description": "Accessibility description for the image",
                        },
                        "grade_level": {
                            "type": "integer",
                            "description": "Target grade level (1-12) for age-appropriate content",
                        },
                    },
                    "required": ["prompt"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the image generation."""
        prompt = params.get("prompt", "")
        style = params.get("style", "illustration")
        aspect_ratio = params.get("aspect_ratio", "1:1")
        purpose = params.get("purpose", "")
        alt_text = params.get("alt_text", "")
        grade_level = params.get("grade_level", context.grade_level or 5)

        if not prompt:
            return ToolResult(
                success=False,
                data={"message": "Image prompt is required"},
                error="Missing required parameter: prompt",
            )

        # Validate style
        if style not in self.SUPPORTED_STYLES:
            style = "illustration"

        # Validate aspect ratio
        if aspect_ratio not in self.SUPPORTED_RATIOS:
            aspect_ratio = "1:1"

        # Enhance prompt with educational context
        enhanced_prompt = self._enhance_prompt(prompt, style, grade_level)

        try:
            # Generate image via Gemini API
            image_data = await self._generate_image_gemini(
                prompt=enhanced_prompt,
                aspect_ratio=aspect_ratio,
            )

            if not image_data:
                return ToolResult(
                    success=False,
                    data={"message": "Failed to generate image - no data returned"},
                    error="Image generation returned no data",
                )

            # Upload to H5P server media storage
            image_url = await self._upload_to_storage(
                image_data=image_data,
                tenant_code=context.tenant_code,
                filename_prefix=f"generated_{style}",
            )

            logger.info(
                "Generated image: prompt=%s..., style=%s, ratio=%s",
                prompt[:50],
                style,
                aspect_ratio,
            )

            return ToolResult(
                success=True,
                data={
                    "image_url": image_url,
                    "prompt": prompt,
                    "enhanced_prompt": enhanced_prompt,
                    "style": style,
                    "aspect_ratio": aspect_ratio,
                    "alt_text": alt_text or prompt[:100],
                    "purpose": purpose,
                    "message": f"Image generated successfully. Style: {style}, Ratio: {aspect_ratio}",
                },
                state_update={
                    "last_generated_image": {
                        "url": image_url,
                        "prompt": prompt,
                        "alt_text": alt_text or prompt[:100],
                    },
                },
            )

        except Exception as e:
            logger.exception("Error generating image")
            return ToolResult(
                success=False,
                data={"message": f"Failed to generate image: {e}"},
                error=str(e),
            )

    def _enhance_prompt(self, prompt: str, style: str, grade_level: int) -> str:
        """Enhance the prompt with style and safety guidelines."""
        style_modifiers = {
            "cartoon": "friendly cartoon style, colorful, simple shapes, suitable for children",
            "realistic": "photorealistic, high quality, detailed, educational photo style",
            "diagram": "clean diagram style, labeled parts, educational illustration, white background",
            "illustration": "educational illustration, clear and informative, professional quality",
            "sketch": "hand-drawn sketch style, pencil illustration, educational drawing",
        }

        grade_modifiers = {
            range(1, 4): "suitable for young children, simple and colorful",
            range(4, 7): "appropriate for elementary students, clear and engaging",
            range(7, 10): "suitable for middle school, more detailed and informative",
            range(10, 13): "appropriate for high school, professional and accurate",
        }

        # Get style modifier
        style_mod = style_modifiers.get(style, "educational illustration")

        # Get grade modifier
        grade_mod = "educational content"
        for grade_range, modifier in grade_modifiers.items():
            if grade_level in grade_range:
                grade_mod = modifier
                break

        # Build enhanced prompt
        enhanced = f"{prompt}, {style_mod}, {grade_mod}, safe for school, no text in image"

        return enhanced

    async def _generate_image_gemini(
        self,
        prompt: str,
        aspect_ratio: str,
    ) -> bytes | None:
        """Generate image using Google Gemini API.

        Uses the Gemini model with image generation capability.

        Args:
            prompt: Enhanced prompt for image generation.
            aspect_ratio: Aspect ratio for the image.

        Returns:
            Image data as bytes, or None if generation failed.
        """
        settings = get_settings()

        # Get API key - prefer dedicated image API key, fall back to LLM settings, then provider config
        api_key = None
        if settings.gemini_image.api_key:
            api_key = settings.gemini_image.api_key.get_secret_value()
        elif settings.llm.google_api_key:
            api_key = settings.llm.google_api_key.get_secret_value()

        if not api_key:
            # Fall back to provider config from providers.yaml
            try:
                from src.core.config.llm_providers import get_provider_config
                google_config = get_provider_config("google")
                if google_config and google_config.api_key:
                    api_key = google_config.api_key
            except Exception:
                pass

        if not api_key:
            raise ValueError("Google API key not configured for image generation")

        # Use the image-generation-capable Gemini model
        # gemini-2.5-flash-image supports native image output via generateContent
        model = "gemini-2.5-flash-image"
        url = f"{self.GEMINI_API_URL}/{model}:generateContent"

        # Build request payload — responseModalities must include TEXT and IMAGE
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"Generate an educational image: {prompt}. Aspect ratio: {aspect_ratio}."
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
            },
        }

        timeout = aiohttp.ClientTimeout(total=settings.gemini_image.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        "Gemini API error: status=%d, response=%s",
                        response.status,
                        error_text[:500],
                    )
                    raise RuntimeError(f"Gemini API error: {response.status}")

                response_data = await response.json()

                # Extract image data from response
                candidates = response_data.get("candidates", [])
                if not candidates:
                    logger.warning("No candidates in Gemini response")
                    return None

                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    if "inlineData" in part:
                        inline_data = part["inlineData"]
                        mime_type = inline_data.get("mimeType", "")
                        if mime_type.startswith("image/"):
                            # Decode base64 image data
                            image_b64 = inline_data.get("data", "")
                            return base64.b64decode(image_b64)

                logger.warning("No image data found in Gemini response")
                return None

    async def _upload_to_storage(
        self,
        image_data: bytes,
        tenant_code: str,
        filename_prefix: str,
    ) -> str:
        """Upload generated image to media storage.

        Args:
            image_data: Image bytes to upload.
            tenant_code: Tenant code for storage path.
            filename_prefix: Prefix for the filename.

        Returns:
            URL of the uploaded image.
        """
        import hashlib
        from datetime import datetime

        # Generate unique filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(image_data).hexdigest()[:8]
        filename = f"{filename_prefix}_{timestamp}_{content_hash}.png"

        # Get H5P client for media upload
        settings = get_settings()
        h5p_api_url = settings.h5p.api_url
        h5p_api_key = settings.h5p.api_key.get_secret_value()

        if not h5p_api_key:
            # Fall back to local storage path format
            return f"/api/v1/media/{tenant_code}/generated/{filename}"

        # Upload to H5P server
        from src.services.h5p import H5PClient

        client = H5PClient(
            api_url=h5p_api_url,
            api_key=h5p_api_key,
            timeout=settings.h5p.timeout,
        )

        await client.upload_media(
            file_data=image_data,
            filename=filename,
            content_type="image/png",
            tenant_code=tenant_code,
        )

        # Return H5P content-relative path (images will be copied to content dir on export)
        return f"images/{filename}"

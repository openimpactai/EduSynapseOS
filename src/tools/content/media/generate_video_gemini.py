# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate Video with Google Veo Tool.

Uses Google's Veo API to generate short educational video clips.
"""

import asyncio
import base64
import hashlib
import logging
from datetime import datetime
from typing import Any

import aiohttp

from src.core.config import get_settings
from src.core.tools import BaseTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class GenerateVideoGeminiTool(BaseTool):
    """Generate short video clips using Google Veo.

    Creates educational video clips from text prompts.
    Uses the Veo model via Vertex AI / Generative AI API.
    """

    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    @property
    def name(self) -> str:
        return "generate_video_gemini"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "generate_video_gemini",
                "description": (
                    "Generate a short educational video clip using Google Veo AI. "
                    "Provide a detailed prompt describing the video scene. "
                    "Videos are max 8 seconds per clip."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Detailed description of the video to generate.",
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "enum": ["16:9", "9:16", "1:1"],
                            "description": "Aspect ratio of the video.",
                        },
                        "alt_text": {
                            "type": "string",
                            "description": "Accessibility description for the video.",
                        },
                        "grade_level": {
                            "type": "integer",
                            "description": "Target grade level (1-12).",
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
        """Execute video generation."""
        prompt = params.get("prompt", "")
        aspect_ratio = params.get("aspect_ratio", "16:9")
        alt_text = params.get("alt_text", "")
        grade_level = params.get("grade_level", context.grade_level or 5)

        if not prompt:
            return ToolResult(
                success=False,
                data={"message": "Video prompt is required"},
                error="Missing required parameter: prompt",
            )

        enhanced_prompt = self._enhance_prompt(prompt, grade_level)

        try:
            video_data = await self._generate_video(
                prompt=enhanced_prompt,
                aspect_ratio=aspect_ratio,
            )

            if not video_data:
                return ToolResult(
                    success=False,
                    data={"message": "Failed to generate video - no data returned"},
                    error="Video generation returned no data",
                )

            video_url = await self._upload_to_storage(
                video_data=video_data,
                tenant_code=context.tenant_code,
                filename_prefix="generated_video",
            )

            logger.info(
                "Generated video: prompt=%s..., ratio=%s",
                prompt[:50],
                aspect_ratio,
            )

            return ToolResult(
                success=True,
                data={
                    "video_url": video_url,
                    "prompt": prompt,
                    "enhanced_prompt": enhanced_prompt,
                    "aspect_ratio": aspect_ratio,
                    "alt_text": alt_text or prompt[:100],
                    "message": f"Video generated successfully. Ratio: {aspect_ratio}",
                },
                state_update={
                    "last_generated_video": {
                        "url": video_url,
                        "prompt": prompt,
                        "alt_text": alt_text or prompt[:100],
                    },
                },
            )

        except Exception as e:
            logger.exception("Error generating video")
            return ToolResult(
                success=False,
                data={"message": f"Failed to generate video: {e}"},
                error=str(e),
            )

    def _enhance_prompt(self, prompt: str, grade_level: int) -> str:
        """Enhance the prompt with educational context and safety."""
        grade_mod = "educational content"
        if grade_level <= 3:
            grade_mod = "suitable for young children, simple and colorful"
        elif grade_level <= 6:
            grade_mod = "appropriate for elementary students, clear and engaging"
        elif grade_level <= 9:
            grade_mod = "suitable for middle school, informative"
        else:
            grade_mod = "appropriate for high school, professional and accurate"

        return (
            f"{prompt}, educational video, {grade_mod}, "
            f"safe for school, no text overlays, smooth motion"
        )

    async def _generate_video(
        self,
        prompt: str,
        aspect_ratio: str,
    ) -> bytes | None:
        """Generate video using Google Veo API (async long-running operation).

        Returns MP4 video bytes.
        """
        settings = get_settings()

        api_key = self._get_api_key(settings)
        if not api_key:
            raise ValueError("Google API key not configured for video generation")

        model = settings.gemini_video.model
        url = f"{self.GEMINI_API_URL}/{model}:predictLongRunning"

        payload = {
            "instances": [
                {
                    "prompt": prompt,
                }
            ],
            "parameters": {
                "aspectRatio": aspect_ratio,
                "sampleCount": 1,
                "durationSeconds": 8,
            },
        }

        timeout = aiohttp.ClientTimeout(total=settings.gemini_video.timeout)
        poll_interval = settings.gemini_video.poll_interval

        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Start the long-running operation
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
                        "Veo API error: status=%d, response=%s",
                        response.status,
                        error_text[:500],
                    )
                    raise RuntimeError(f"Veo API error: {response.status} - {error_text[:200]}")

                op_data = await response.json()

            # Poll for completion
            op_name = op_data.get("name", "")
            if not op_name:
                logger.warning("No operation name in Veo response: %s", op_data)
                return None

            poll_url = f"https://generativelanguage.googleapis.com/v1beta/{op_name}"
            max_polls = settings.gemini_video.timeout // poll_interval

            for i in range(max_polls):
                await asyncio.sleep(poll_interval)

                async with session.get(
                    poll_url,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": api_key,
                    },
                ) as poll_response:
                    if poll_response.status != 200:
                        continue

                    poll_data = await poll_response.json()

                    if poll_data.get("done"):
                        return await self._extract_video(poll_data, session, api_key)

                    logger.debug(
                        "Video generation poll %d/%d: not done yet",
                        i + 1,
                        max_polls,
                    )

            logger.warning("Video generation timed out after %d polls", max_polls)
            return None

    async def _extract_video(
        self,
        op_data: dict,
        session: aiohttp.ClientSession,
        api_key: str,
    ) -> bytes | None:
        """Extract video bytes from completed operation response."""
        response_obj = op_data.get("response", {})
        videos = response_obj.get("generateVideoResponse", {}).get("generatedSamples", [])

        if not videos:
            # Try alternate response format
            videos = response_obj.get("predictions", [])

        if not videos:
            logger.warning("No videos in Veo response: %s", list(response_obj.keys()))
            return None

        video = videos[0]

        # Check for inline video data (base64)
        if "video" in video and "uri" in video["video"]:
            video_uri = video["video"]["uri"]
            # Download the video from the URI
            async with session.get(
                video_uri,
                headers={"x-goog-api-key": api_key},
            ) as dl_response:
                if dl_response.status == 200:
                    return await dl_response.read()
                logger.error("Failed to download video: %d", dl_response.status)
                return None

        # Check for bytesBase64Encoded
        b64_data = video.get("bytesBase64Encoded") or video.get("video", {}).get("bytesBase64Encoded")
        if b64_data:
            return base64.b64decode(b64_data)

        logger.warning("Could not extract video data from response")
        return None

    def _get_api_key(self, settings) -> str | None:
        """Get Google API key from settings with fallback chain."""
        if settings.gemini_video.api_key:
            return settings.gemini_video.api_key.get_secret_value()
        if settings.gemini_image.api_key:
            return settings.gemini_image.api_key.get_secret_value()
        if settings.llm.google_api_key:
            return settings.llm.google_api_key.get_secret_value()
        try:
            from src.core.config.llm_providers import get_provider_config
            google_config = get_provider_config("google")
            if google_config and google_config.api_key:
                return google_config.api_key
        except Exception:
            pass
        return None

    async def _upload_to_storage(
        self,
        video_data: bytes,
        tenant_code: str,
        filename_prefix: str,
    ) -> str:
        """Upload generated video to H5P media storage."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(video_data).hexdigest()[:8]
        filename = f"{filename_prefix}_{timestamp}_{content_hash}.mp4"

        settings = get_settings()
        h5p_api_url = settings.h5p.api_url
        h5p_api_key = settings.h5p.api_key.get_secret_value()

        if not h5p_api_key:
            return f"/api/v1/media/{tenant_code}/generated/{filename}"

        from src.services.h5p import H5PClient

        client = H5PClient(
            api_url=h5p_api_url,
            api_key=h5p_api_key,
            timeout=settings.h5p.timeout,
        )

        await client.upload_media(
            file_data=video_data,
            filename=filename,
            content_type="video/mp4",
            tenant_code=tenant_code,
        )

        return f"videos/{filename}"

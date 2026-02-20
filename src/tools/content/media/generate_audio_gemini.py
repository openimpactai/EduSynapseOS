# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate Audio with Gemini TTS Tool.

Uses Google's Gemini TTS API to generate educational audio narration.
"""

import base64
import hashlib
import io
import logging
import struct
import wave
from datetime import datetime
from typing import Any

import aiohttp

from src.core.config import get_settings
from src.core.tools import BaseTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class GenerateAudioGeminiTool(BaseTool):
    """Generate audio narration using Google Gemini TTS.

    Creates educational audio clips from text prompts.
    Uses the Gemini TTS model for natural speech synthesis.
    """

    SUPPORTED_VOICES = [
        "Kore", "Charon", "Fenrir", "Aoede", "Puck",
        "Leda", "Orus", "Zephyr",
    ]

    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    @property
    def name(self) -> str:
        return "generate_audio_gemini"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "generate_audio_gemini",
                "description": (
                    "Generate educational audio narration using Google Gemini TTS. "
                    "Provide text to be spoken aloud as narration for H5P content."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to convert to speech.",
                        },
                        "voice": {
                            "type": "string",
                            "enum": SUPPORTED_VOICES,
                            "description": "Voice to use for narration.",
                        },
                        "language": {
                            "type": "string",
                            "description": "Language code (e.g., 'en', 'tr').",
                        },
                        "alt_text": {
                            "type": "string",
                            "description": "Accessibility description for the audio.",
                        },
                    },
                    "required": ["text"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute audio generation."""
        text = params.get("text", "")
        voice = params.get("voice", "Kore")
        alt_text = params.get("alt_text", "")

        if not text:
            return ToolResult(
                success=False,
                data={"message": "Text is required for audio generation"},
                error="Missing required parameter: text",
            )

        if voice not in self.SUPPORTED_VOICES:
            voice = "Kore"

        try:
            audio_data = await self._generate_audio(text=text, voice=voice)

            if not audio_data:
                return ToolResult(
                    success=False,
                    data={"message": "Failed to generate audio - no data returned"},
                    error="Audio generation returned no data",
                )

            audio_url = await self._upload_to_storage(
                audio_data=audio_data,
                tenant_code=context.tenant_code,
                filename_prefix=f"narration_{voice.lower()}",
            )

            logger.info(
                "Generated audio: text=%s..., voice=%s",
                text[:50],
                voice,
            )

            return ToolResult(
                success=True,
                data={
                    "audio_url": audio_url,
                    "text": text,
                    "voice": voice,
                    "alt_text": alt_text or text[:100],
                    "message": f"Audio generated successfully. Voice: {voice}",
                },
                state_update={
                    "last_generated_audio": {
                        "url": audio_url,
                        "text": text,
                        "alt_text": alt_text or text[:100],
                    },
                },
            )

        except Exception as e:
            logger.exception("Error generating audio")
            return ToolResult(
                success=False,
                data={"message": f"Failed to generate audio: {e}"},
                error=str(e),
            )

    async def _generate_audio(self, text: str, voice: str) -> bytes | None:
        """Generate audio using Gemini TTS API.

        Returns WAV file bytes.
        """
        settings = get_settings()

        api_key = self._get_api_key(settings)
        if not api_key:
            raise ValueError("Google API key not configured for audio generation")

        model = settings.gemini_audio.model
        url = f"{self.GEMINI_API_URL}/{model}:generateContent"

        payload = {
            "contents": [
                {
                    "parts": [{"text": text}]
                }
            ],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voice,
                        }
                    }
                },
            },
        }

        timeout = aiohttp.ClientTimeout(total=settings.gemini_audio.timeout)

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
                        "Gemini TTS API error: status=%d, response=%s",
                        response.status,
                        error_text[:500],
                    )
                    raise RuntimeError(f"Gemini TTS API error: {response.status}")

                response_data = await response.json()

                candidates = response_data.get("candidates", [])
                if not candidates:
                    logger.warning("No candidates in Gemini TTS response")
                    return None

                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    if "inlineData" in part:
                        inline_data = part["inlineData"]
                        mime_type = inline_data.get("mimeType", "")
                        if "audio" in mime_type:
                            audio_b64 = inline_data.get("data", "")
                            pcm_data = base64.b64decode(audio_b64)
                            # Convert raw PCM to WAV
                            return self._pcm_to_wav(pcm_data)

                logger.warning("No audio data found in Gemini TTS response")
                return None

    @staticmethod
    def _pcm_to_wav(
        pcm_data: bytes,
        sample_rate: int = 24000,
        channels: int = 1,
        sample_width: int = 2,
    ) -> bytes:
        """Convert raw PCM bytes to WAV format."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()

    def _get_api_key(self, settings) -> str | None:
        """Get Google API key from settings with fallback chain."""
        if settings.gemini_audio.api_key:
            return settings.gemini_audio.api_key.get_secret_value()
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
        audio_data: bytes,
        tenant_code: str,
        filename_prefix: str,
    ) -> str:
        """Upload generated audio to H5P media storage."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(audio_data).hexdigest()[:8]
        filename = f"{filename_prefix}_{timestamp}_{content_hash}.wav"

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
            file_data=audio_data,
            filename=filename,
            content_type="audio/wav",
            tenant_code=tenant_code,
        )

        return f"audios/{filename}"

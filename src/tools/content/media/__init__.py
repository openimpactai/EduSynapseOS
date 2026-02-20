# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Media Generation Tools.

Tools for generating images, diagrams, charts, audio, and video for H5P content.
"""

from src.tools.content.media.generate_audio_gemini import GenerateAudioGeminiTool
from src.tools.content.media.generate_chart import GenerateChartTool
from src.tools.content.media.generate_diagram import GenerateDiagramTool
from src.tools.content.media.generate_image_gemini import GenerateImageGeminiTool
from src.tools.content.media.generate_video_gemini import GenerateVideoGeminiTool

__all__ = [
    "GenerateAudioGeminiTool",
    "GenerateChartTool",
    "GenerateDiagramTool",
    "GenerateImageGeminiTool",
    "GenerateVideoGeminiTool",
]

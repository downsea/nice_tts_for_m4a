"""
Nice-TTS: AI-Powered Audio Transcription Tool with Chinese Language Optimization

A powerful, batch-capable command-line tool that leverages OpenAI's Whisper models
to transcribe audio recordings, with default optimization for Chinese language processing.
"""

__version__ = "0.1.0"
__author__ = "Nice-TTS Developer"
__email__ = "developer@nice-tts.com"
__description__ = "AI-Powered Audio Transcription Tool with Chinese Language Optimization"

from . import audio_processor, gpu_utils, transcriber, cli, m4a_handler, text_formatter

__all__ = [
    "audio_processor",
    "gpu_utils", 
    "transcriber",
    "cli",
    "m4a_handler",
    "text_formatter",
]
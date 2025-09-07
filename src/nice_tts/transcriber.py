"""
Whisper transcription engine for Nice-TTS.
Handles AI-powered audio transcription with multi-language support.
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import time

# Import text formatter
from .text_formatter import TextFormatter, format_transcription_readable

logger = logging.getLogger(__name__)

# Suppress whisper warnings
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")


class Transcriber:
    """Handles AI-powered audio transcription using OpenAI Whisper."""
    
    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"]
    DEFAULT_MODEL = "base"
    DEFAULT_LANGUAGE = "zh"  # Chinese as default
    
    def __init__(self, model_name: str = None, device: str = None, 
                 language: str = None, compute_type: str = None,
                 enable_formatting: bool = True, phone_call_mode: bool = False):
        """
        Initialize the transcriber.
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large, etc.)
            device: Processing device ('cuda' or 'cpu')
            language: Default language code ('zh', 'en', etc.)
            compute_type: Compute precision ('float16', 'float32', 'int8')
            enable_formatting: Enable text formatting for better readability
            phone_call_mode: Enable phone call transcription optimizations
        """
        self.model_name = model_name or os.getenv("WHISPER_MODEL", self.DEFAULT_MODEL)
        self.language = language or os.getenv("DEFAULT_LANGUAGE", self.DEFAULT_LANGUAGE)
        self.device = device or "cpu"  # Will be set by GPU manager
        self.compute_type = compute_type or "float16" if device == "cuda" else "float32"
        self.enable_formatting = enable_formatting
        self.phone_call_mode = phone_call_mode
        
        # Phone call specific settings
        if phone_call_mode:
            self.default_initial_prompt = "以下为电话录音的内容。"
            self.phone_call_fp16 = True  # Always use FP16 for phone calls when available
            logger.info("Phone call transcription mode enabled")
        else:
            self.default_initial_prompt = None
            self.phone_call_fp16 = False
        
        self._model = None
        self._model_loaded = False
        self._load_time = None
        self._text_formatter = TextFormatter(language=self.language) if enable_formatting else None
        
        logger.info(f"Transcriber initialized with model: {self.model_name}, "
                   f"device: {self.device}, language: {self.language}, "
                   f"formatting: {enable_formatting}, phone_call_mode: {phone_call_mode}")
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model_loaded
    
    @property
    def model(self):
        """Get the loaded Whisper model."""
        if not self._model_loaded:
            self._load_model()
        return self._model
    
    def _load_model(self) -> bool:
        """
        Load the Whisper model.
        
        Returns:
            True if model loaded successfully
        """
        if self._model_loaded:
            return True
            
        try:
            import whisper
            
            logger.info(f"Loading Whisper model: {self.model_name}")
            start_time = time.time()
            
            # Load model with device specification
            self._model = whisper.load_model(
                self.model_name, 
                device=self.device,
                download_root=None  # Use default cache location
            )
            
            self._load_time = time.time() - start_time
            self._model_loaded = True
            
            logger.info(f"Model loaded successfully in {self._load_time:.2f}s")
            return True
            
        except ImportError:
            logger.error("OpenAI Whisper not installed. Run: pip install openai-whisper")
            return False
        except Exception as e:
            logger.error(f"Failed to load Whisper model {self.model_name}: {e}")
            return False
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model_loaded:
            self._model = None
            self._model_loaded = False
            logger.info("Whisper model unloaded")
    
    def transcribe_file(self, audio_file: Union[str, Path], 
                       language: Optional[str] = None,
                       temperature: float = 0.0,
                       beam_size: Optional[int] = None,
                       best_of: Optional[int] = None,
                       patience: Optional[float] = None,
                       length_penalty: Optional[float] = None,
                       suppress_tokens: Optional[List[int]] = None,
                       initial_prompt: Optional[str] = None,
                       condition_on_previous_text: bool = True,
                       fp16: Optional[bool] = None,
                       compression_ratio_threshold: float = 2.4,
                       logprob_threshold: float = -1.0,
                       no_speech_threshold: float = 0.6,
                       enable_formatting: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """
        Transcribe a single audio file.
        
        Args:
            audio_file: Path to the audio file
            language: Language code (overrides default)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = random)
            beam_size: Beam size for decoding
            best_of: Number of candidates when sampling
            patience: Beam search patience factor
            length_penalty: Length penalty factor
            suppress_tokens: List of token IDs to suppress
            initial_prompt: Initial prompt to condition the model
            condition_on_previous_text: Whether to condition on previous text
            fp16: Whether to use FP16 precision
            compression_ratio_threshold: Threshold for compression ratio
            logprob_threshold: Threshold for log probability
            no_speech_threshold: Threshold for no speech detection
            enable_formatting: Override default formatting setting
            
        Returns:
            Transcription result dictionary or None if error
        """
        audio_file = Path(audio_file)
        
        if not audio_file.exists():
            logger.error(f"Audio file does not exist: {audio_file}")
            return None
            
        # Use specified language or default
        transcribe_language = language or self.language
        
        # Apply phone call specific settings
        if self.phone_call_mode:
            # Use phone call specific initial prompt if not provided
            if initial_prompt is None and self.default_initial_prompt:
                initial_prompt = self.default_initial_prompt
                logger.info(f"Using phone call initial prompt: {initial_prompt}")
            
            # Use FP16 for phone calls when CUDA is available, unless explicitly overridden
            if fp16 is None:
                fp16 = self.phone_call_fp16 and self.device == "cuda"
                if fp16:
                    logger.info("Using FP16 precision for phone call transcription")
        else:
            # Default fp16 behavior for non-phone call mode
            if fp16 is None:
                fp16 = self.device == "cuda"
        
        # Determine if formatting should be applied
        apply_formatting = enable_formatting if enable_formatting is not None else self.enable_formatting
        
        try:
            logger.info(f"Transcribing {audio_file} in {transcribe_language}")
            start_time = time.time()
            
            # Load audio file
            import whisper
            audio = whisper.load_audio(str(audio_file))
            
            # Transcribe with specified parameters
            result = self.model.transcribe(
                audio,
                language=transcribe_language,
                temperature=temperature,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                length_penalty=length_penalty,
                suppress_tokens=suppress_tokens,
                initial_prompt=initial_prompt,
                condition_on_previous_text=condition_on_previous_text,
                fp16=fp16,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
            )
            
            processing_time = time.time() - start_time
            audio_duration = len(audio) / whisper.audio.SAMPLE_RATE
            
            # Add metadata to result
            result["processing_time"] = processing_time
            result["audio_duration"] = audio_duration
            result["real_time_factor"] = processing_time / audio_duration if audio_duration > 0 else 0
            result["model"] = self.model_name
            result["language"] = transcribe_language
            result["device"] = self.device
            
            # Apply text formatting if enabled
            if apply_formatting and self._text_formatter:
                original_text = result.get("text", "")
                if original_text:
                    formatted_text = self._text_formatter.format_text(original_text)
                    result["text"] = formatted_text
                    result["text_formatted"] = True
                    logger.info("Applied text formatting for improved readability")
                else:
                    result["text_formatted"] = False
            else:
                result["text_formatted"] = False
            
            logger.info(f"Transcription completed in {processing_time:.2f}s "
                       f"(RTF: {result['real_time_factor']:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_file}: {e}")
            return None
    
    def transcribe_batch(self, audio_files: List[Union[str, Path]], 
                        language: Optional[str] = None,
                        **transcribe_kwargs) -> List[Optional[Dict[str, Any]]]:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            language: Language code for all files
            **transcribe_kwargs: Additional arguments for transcribe_file
            
        Returns:
            List of transcription results
        """
        results = []
        total_files = len(audio_files)
        
        logger.info(f"Starting batch transcription of {total_files} files")
        
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"Processing file {i}/{total_files}: {audio_file}")
            
            result = self.transcribe_file(audio_file, language=language, **transcribe_kwargs)
            results.append(result)
            
            if result:
                logger.info(f"✅ Completed {audio_file}")
            else:
                logger.error(f"❌ Failed {audio_file}")
                
        logger.info(f"Batch transcription completed: {len([r for r in results if r])}/{total_files} successful")
        return results
    
    def detect_language(self, audio_file: Union[str, Path]) -> Optional[str]:
        """
        Detect the language of an audio file.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Detected language code or None if error
        """
        audio_file = Path(audio_file)
        
        if not audio_file.exists():
            logger.error(f"Audio file does not exist: {audio_file}")
            return None
            
        try:
            logger.info(f"Detecting language for {audio_file}")
            
            import whisper
            audio = whisper.load_audio(str(audio_file))
            
            # Transcribe without language specification to detect language
            result = self.model.transcribe(audio, language=None)
            
            detected_language = result.get("language")
            if detected_language:
                logger.info(f"Detected language: {detected_language}")
                return detected_language
            else:
                logger.warning(f"Could not detect language for {audio_file}")
                return None
                
        except Exception as e:
            logger.error(f"Language detection failed for {audio_file}: {e}")
            return None
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        # Whisper supports many languages, these are common ones
        return [
            "zh",  # Chinese
            "en",  # English
            "es",  # Spanish
            "fr",  # French
            "de",  # German
            "it",  # Italian
            "pt",  # Portuguese
            "ru",  # Russian
            "ja",  # Japanese
            "ko",  # Korean
            "ar",  # Arabic
            "hi",  # Hindi
        ]
    
    def format_transcription_result(self, result: Dict[str, Any], 
                                  include_segments: bool = False,
                                  enable_formatting: bool = True) -> str:
        """
        Format transcription result as a readable string.
        
        Args:
            result: Transcription result dictionary
            include_segments: Whether to include segment-level details
            enable_formatting: Whether to apply text formatting
            
        Returns:
            Formatted transcription text
        """
        if not result:
            return "Transcription failed"
            
        output_lines = []
        
        # Main transcription text
        text = result.get("text", "").strip()
        if text:
            output_lines.append("Transcription:")
            output_lines.append("=" * 50)
            
            # Apply formatting if available and enabled
            if enable_formatting and self._text_formatter:
                formatted_text = self._text_formatter.format_text(text)
                output_lines.append(formatted_text)
            else:
                # Basic formatting - break long lines
                if len(text) > 80:
                    lines = []
                    while len(text) > 80:
                        # Find a good break point (space or punctuation)
                        break_point = 80
                        for i in range(75, 85):
                            if i < len(text) and (text[i] in '。！？，.!? '):
                                break_point = i + 1
                                break
                        lines.append(text[:break_point].strip())
                        text = text[break_point:].strip()
                    if text:
                        lines.append(text)
                    output_lines.extend(lines)
                else:
                    output_lines.append(text)
            
            output_lines.append("")
        else:
            output_lines.append("No speech detected")
            
        # Metadata
        output_lines.append("Metadata:")
        output_lines.append("-" * 20)
        
        if "language" in result:
            output_lines.append(f"Language: {result['language']}")
        if "model" in result:
            output_lines.append(f"Model: {result['model']}")
        if "device" in result:
            output_lines.append(f"Device: {result['device']}")
        if "processing_time" in result:
            output_lines.append(f"Processing time: {result['processing_time']:.2f}s")
        if "audio_duration" in result:
            output_lines.append(f"Audio duration: {result['audio_duration']:.2f}s")
        if "real_time_factor" in result:
            output_lines.append(f"Real-time factor: {result['real_time_factor']:.2f}")
            
        # Segments if requested
        if include_segments and "segments" in result:
            output_lines.append("")
            output_lines.append("Segments:")
            output_lines.append("-" * 30)
            
            for i, segment in enumerate(result["segments"], 1):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                text = segment.get("text", "").strip()
                
                # Format segment text if formatting is enabled
                if enable_formatting and self._text_formatter and text:
                    formatted_segment = self._text_formatter.format_text(
                        text, max_line_length=60, segment_sentences=False
                    )
                    output_lines.append(f"[{start:.1f}s - {end:.1f}s] {formatted_segment}")
                else:
                    output_lines.append(f"[{start:.1f}s - {end:.1f}s] {text}")
                
        return "\n".join(output_lines)


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a Whisper model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model information dictionary
    """
    model_info = {
        "tiny": {"parameters": "39M", "memory": "~79MB", "speed": "~32x", "accuracy": "Lowest"},
        "base": {"parameters": "74M", "memory": "~148MB", "speed": "~16x", "accuracy": "Low"},
        "small": {"parameters": "244M", "memory": "~466MB", "speed": "~6x", "accuracy": "Medium"},
        "medium": {"parameters": "769M", "memory": "~1.5GB", "speed": "~2x", "accuracy": "High"},
        "large": {"parameters": "1550M", "memory": "~3.1GB", "speed": "1x", "accuracy": "Highest"},
        "large-v1": {"parameters": "1550M", "memory": "~3.1GB", "speed": "1x", "accuracy": "Highest"},
        "large-v2": {"parameters": "1550M", "memory": "~3.1GB", "speed": "1x", "accuracy": "Highest"},
        "large-v3": {"parameters": "1550M", "memory": "~3.1GB", "speed": "1x", "accuracy": "Highest"},
    }
    
    return model_info.get(model_name, {"error": "Unknown model"})


def list_available_models() -> List[str]:
    """Get list of available Whisper model names."""
    return Transcriber.AVAILABLE_MODELS.copy()


# Global transcriber instance
transcriber = Transcriber()
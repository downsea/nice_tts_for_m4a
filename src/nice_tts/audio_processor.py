"""
Audio processing utilities for Nice-TTS.
Handles audio file validation, format conversion, and preprocessing.
"""

import os
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple, Union
import wave
import contextlib

logger = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_FORMATS = {
    '.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.aiff', '.au'
}

# Audio processing parameters
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1  # Mono
DEFAULT_FORMAT = "wav"


class AudioProcessor:
    """Handles audio file processing and validation."""
    
    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, 
                 channels: int = DEFAULT_CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels
        self.temp_files = []
        
    def __del__(self):
        """Clean up temporary files."""
        self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """Remove temporary files created during processing."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
        self.temp_files.clear()
    
    def validate_audio_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate if the file is a supported audio format.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            True if file is valid and supported
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            logger.error(f"Audio file does not exist: {file_path}")
            return False
            
        # Check if it's a file
        if not file_path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            return False
            
        # Check file extension
        file_extension = file_path.suffix.lower()
        if file_extension not in SUPPORTED_FORMATS:
            logger.error(f"Unsupported audio format: {file_extension}")
            logger.info(f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}")
            return False
            
        # Check file size (reasonable limits)
        file_size = file_path.stat().st_size
        if file_size == 0:
            logger.error(f"Audio file is empty: {file_path}")
            return False
            
        if file_size > 2 * 1024**3:  # 2GB limit
            logger.warning(f"Audio file is very large ({file_size / (1024**3):.1f}GB): {file_path}")
            
        # Try to get audio info using ffprobe
        try:
            info = self.get_audio_info(file_path)
            if info is None:
                logger.error(f"Could not read audio file: {file_path}")
                return False
                
            # Check duration (reasonable limits)
            duration = info.get('duration', 0)
            if duration > 7200:  # 2 hours
                logger.warning(f"Audio file is very long ({duration/60:.0f} minutes): {file_path}")
            elif duration < 1:  # Less than 1 second
                logger.warning(f"Audio file is very short ({duration:.1f}s): {file_path}")
                
        except Exception as e:
            logger.error(f"Error validating audio file {file_path}: {e}")
            return False
            
        logger.info(f"Audio file validated: {file_path}")
        return True
    
    def get_audio_info(self, file_path: Union[str, Path]) -> Optional[dict]:
        """
        Get audio file information using ffprobe.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with audio information or None if error
        """
        file_path = Path(file_path)
        
        try:
            # Use ffprobe to get audio information
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(file_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"ffprobe failed for {file_path}: {result.stderr}")
                return None
                
            import json
            probe_data = json.loads(result.stdout)
            
            # Extract relevant information
            format_info = probe_data.get('format', {})
            streams = probe_data.get('streams', [])
            
            # Find audio stream
            audio_stream = None
            for stream in streams:
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
                    
            if not audio_stream:
                logger.error(f"No audio stream found in {file_path}")
                return None
                
            info = {
                'duration': float(format_info.get('duration', 0)),
                'size': int(format_info.get('size', 0)),
                'bitrate': int(format_info.get('bit_rate', 0)),
                'format': format_info.get('format_name', 'unknown'),
                'sample_rate': int(audio_stream.get('sample_rate', 0)),
                'channels': int(audio_stream.get('channels', 0)),
                'codec': audio_stream.get('codec_name', 'unknown'),
            }
            
            return info
            
        except subprocess.TimeoutExpired:
            logger.error(f"ffprobe timeout for {file_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ffprobe output for {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting audio info for {file_path}: {e}")
            return None
    
    def convert_to_wav(self, input_file: Union[str, Path], 
                      output_file: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """
        Convert audio file to WAV format with standard parameters.
        
        Args:
            input_file: Input audio file path
            output_file: Output WAV file path (optional)
            
        Returns:
            Path to converted WAV file or None if error
        """
        input_file = Path(input_file)
        
        if not self.validate_audio_file(input_file):
            return None
            
        # Generate output filename if not provided
        if output_file is None:
            output_file = input_file.with_suffix('.wav')
        else:
            output_file = Path(output_file)
            
        try:
            logger.info(f"Converting {input_file} to WAV format")
            
            # Use ffmpeg to convert to standard WAV format
            cmd = [
                'ffmpeg', '-y', '-i', str(input_file),
                '-ar', str(self.sample_rate),  # Sample rate
                '-ac', str(self.channels),     # Channels
                '-sample_fmt', 's16',          # 16-bit samples
                str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"ffmpeg conversion failed: {result.stderr}")
                return None
                
            if not output_file.exists():
                logger.error(f"Output file was not created: {output_file}")
                return None
                
            logger.info(f"Successfully converted to {output_file}")
            return output_file
            
        except subprocess.TimeoutExpired:
            logger.error(f"ffmpeg conversion timeout for {input_file}")
            return None
        except Exception as e:
            logger.error(f"Error converting {input_file} to WAV: {e}")
            return None
    
    def preprocess_for_whisper(self, input_file: Union[str, Path]) -> Optional[Path]:
        """
        Preprocess audio file for optimal Whisper transcription.
        
        Args:
            input_file: Input audio file path
            
        Returns:
            Path to preprocessed WAV file or None if error
        """
        input_file = Path(input_file)
        
        if not self.validate_audio_file(input_file):
            return None
            
        # Check if file is already in optimal format
        info = self.get_audio_info(input_file)
        if info and info['format'] == 'wav' and info['sample_rate'] == self.sample_rate:
            logger.info(f"Audio file {input_file} is already in optimal format")
            return input_file
            
        # Create temporary file for conversion
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_file = Path(f.name)
                
            self.temp_files.append(str(temp_file))
            
            # Convert to optimal format
            result = self.convert_to_wav(input_file, temp_file)
            if result:
                logger.info(f"Preprocessed audio for Whisper: {result}")
                return result
            else:
                logger.error(f"Failed to preprocess {input_file}")
                return None
                
        except Exception as e:
            logger.error(f"Error preprocessing {input_file}: {e}")
            # Cleanup temp file on error
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass
            return None
    
    def get_duration(self, file_path: Union[str, Path]) -> float:
        """
        Get audio file duration in seconds.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Duration in seconds, 0 if error
        """
        info = self.get_audio_info(file_path)
        return info.get('duration', 0.0) if info else 0.0
    
    def is_valid_for_processing(self, file_path: Union[str, Path], 
                              max_duration: Optional[float] = None) -> Tuple[bool, str]:
        """
        Comprehensive validation for audio processing.
        
        Args:
            file_path: Path to the audio file
            max_duration: Maximum allowed duration in seconds
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        file_path = Path(file_path)
        
        # Basic validation
        if not self.validate_audio_file(file_path):
            return False, "Invalid audio file"
            
        # Duration check
        if max_duration is not None:
            duration = self.get_duration(file_path)
            if duration > max_duration:
                return False, f"Audio too long ({duration:.1f}s > {max_duration}s)"
                
        return True, ""


def check_ffmpeg_installation() -> bool:
    """
    Check if ffmpeg is installed and accessible.
    
    Returns:
        True if ffmpeg is available
    """
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("ffmpeg is available")
            return True
        else:
            logger.error("ffmpeg found but returned error")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.error("ffmpeg not found. Please install ffmpeg.")
        return False


def get_supported_formats() -> List[str]:
    """Get list of supported audio formats."""
    return sorted(list(SUPPORTED_FORMATS))


# Global audio processor instance
audio_processor = AudioProcessor()
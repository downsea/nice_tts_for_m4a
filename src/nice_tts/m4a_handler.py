"""
M4A file handling utilities for Nice-TTS.
Provides robust handling of M4A files with validation, repair, and conversion.
"""

import os
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple

logger = logging.getLogger(__name__)


class M4AHandler:
    """Handles M4A file validation, repair, and conversion."""
    
    def __init__(self):
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
    
    def validate_m4a(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Validate M4A file structure and compatibility.
        
        Args:
            file_path: Path to the M4A file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return False, "File does not exist"
        
        if not file_path.suffix.lower() == '.m4a':
            return False, "Not an M4A file"
        
        try:
            # Use ffprobe to validate M4A structure
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(file_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return False, f"ffprobe validation failed: {result.stderr}"
            
            import json
            probe_data = json.loads(result.stdout)
            
            # Check for audio stream
            streams = probe_data.get('streams', [])
            audio_streams = [s for s in streams if s.get('codec_type') == 'audio']
            
            if not audio_streams:
                return False, "No audio stream found"
            
            # Check format
            format_info = probe_data.get('format', {})
            format_name = format_info.get('format_name', '')
            
            if 'mp4' not in format_name.lower():
                return False, f"Invalid format: {format_name}"
            
            # Check for potential issues
            audio_stream = audio_streams[0]
            
            # Check codec
            codec = audio_stream.get('codec_name', '')
            if codec not in ['aac', 'alac', 'mp4a']:
                logger.warning(f"Unusual M4A codec: {codec}")
            
            # Check duration
            duration = float(format_info.get('duration', 0))
            if duration == 0:
                return False, "Zero duration detected"
            
            logger.info(f"M4A file validated: {file_path}")
            return True, ""
            
        except subprocess.TimeoutExpired:
            return False, "Validation timeout"
        except json.JSONDecodeError:
            return False, "Invalid file structure"
        except Exception as e:
            logger.error(f"M4A validation error: {e}")
            return False, f"Validation error: {str(e)}"
    
    def repair_m4a(self, input_file: Union[str, Path], 
                   output_file: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """
        Attempt to repair corrupted M4A file.
        
        Args:
            input_file: Path to the corrupted M4A file
            output_file: Path for repaired file (optional)
            
        Returns:
            Path to repaired file or None if repair failed
        """
        input_file = Path(input_file)
        
        # Validate first
        is_valid, error_msg = self.validate_m4a(input_file)
        if is_valid:
            logger.info(f"M4A file is already valid: {input_file}")
            return input_file
        
        logger.info(f"Attempting to repair M4A file: {input_file}")
        
        # Generate output filename if not provided
        if output_file is None:
            output_file = input_file.with_suffix('.repaired.m4a')
        else:
            output_file = Path(output_file)
        
        try:
            # Try different repair strategies
            
            # Strategy 1: Re-encode with AAC
            cmd1 = [
                'ffmpeg', '-y', '-i', str(input_file),
                '-c:a', 'aac', '-b:a', '192k',
                '-movflags', 'faststart',
                str(output_file)
            ]
            
            result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=300)
            
            if result1.returncode == 0:
                # Validate repaired file
                is_valid, _ = self.validate_m4a(output_file)
                if is_valid:
                    logger.info(f"M4A file repaired successfully: {output_file}")
                    return output_file
            
            # Strategy 2: Extract audio stream only
            temp_file = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.aac', delete=False) as f:
                    temp_file = Path(f.name)
                
                self.temp_files.append(str(temp_file))
                
                # Extract audio stream
                cmd2a = [
                    'ffmpeg', '-y', '-i', str(input_file),
                    '-c:a', 'copy', '-vn',
                    str(temp_file)
                ]
                
                result2a = subprocess.run(cmd2a, capture_output=True, text=True, timeout=300)
                
                if result2a.returncode == 0 and temp_file.exists():
                    # Re-containerize as M4A
                    cmd2b = [
                        'ffmpeg', '-y', '-i', str(temp_file),
                        '-c:a', 'copy', '-movflags', 'faststart',
                        str(output_file)
                    ]
                    
                    result2b = subprocess.run(cmd2b, capture_output=True, text=True, timeout=300)
                    
                    if result2b.returncode == 0:
                        # Validate final file
                        is_valid, _ = self.validate_m4a(output_file)
                        if is_valid:
                            logger.info(f"M4A file repaired using stream extraction: {output_file}")
                            return output_file
            
            except Exception as e:
                logger.error(f"Stream extraction repair failed: {e}")
            finally:
                if temp_file and temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass
            
            # Strategy 3: Convert to intermediate format and back
            temp_wav = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_wav = Path(f.name)
                
                self.temp_files.append(str(temp_wav))
                
                # Convert to WAV first
                cmd3a = [
                    'ffmpeg', '-y', '-i', str(input_file),
                    '-ar', '44100', '-ac', '2', '-sample_fmt', 's16',
                    str(temp_wav)
                ]
                
                result3a = subprocess.run(cmd3a, capture_output=True, text=True, timeout=300)
                
                if result3a.returncode == 0 and temp_wav.exists():
                    # Convert back to M4A
                    cmd3b = [
                        'ffmpeg', '-y', '-i', str(temp_wav),
                        '-c:a', 'aac', '-b:a', '192k',
                        '-movflags', 'faststart',
                        str(output_file)
                    ]
                    
                    result3b = subprocess.run(cmd3b, capture_output=True, text=True, timeout=300)
                    
                    if result3b.returncode == 0:
                        # Validate final file
                        is_valid, _ = self.validate_m4a(output_file)
                        if is_valid:
                            logger.info(f"M4A file repaired via intermediate conversion: {output_file}")
                            return output_file
            
            except Exception as e:
                logger.error(f"Intermediate conversion repair failed: {e}")
            finally:
                if temp_wav and temp_wav.exists():
                    try:
                        temp_wav.unlink()
                    except:
                        pass
            
            logger.error(f"All repair strategies failed for {input_file}")
            return None
            
        except subprocess.TimeoutExpired:
            logger.error(f"M4A repair timeout for {input_file}")
            return None
        except Exception as e:
            logger.error(f"M4A repair error for {input_file}: {e}")
            return None
    
    def convert_m4a_to_wav(self, input_file: Union[str, Path], 
                          output_file: Optional[Union[str, Path]] = None,
                          sample_rate: int = 16000, channels: int = 1,
                          output_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """
        Convert M4A file to WAV format optimized for transcription.
        
        Args:
            input_file: Path to the M4A file
            output_file: Path for output WAV file (optional)
            sample_rate: Output sample rate (default: 16000)
            channels: Output channels (default: 1 for mono)
            output_dir: Output directory for converted files (optional)
            
        Returns:
            Path to converted WAV file or None if error/skip
        """
        input_file = Path(input_file)
        
        # Validate first
        is_valid, error_msg = self.validate_m4a(input_file)
        if not is_valid:
            logger.error(f"Invalid M4A file: {error_msg}")
            return None
        
        # Generate output filename if not provided
        if output_file is None:
            if output_dir:
                # Place converted file in output directory
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{input_file.stem}.wav"
            else:
                # Default behavior: same directory as input
                output_file = input_file.with_suffix('.wav')
        else:
            output_file = Path(output_file)
        
        # Check if output file already exists
        if output_file.exists():
            logger.info(f"[SKIP] Converted WAV file already exists, skipping conversion: {output_file}")
            return output_file
        
        try:
            logger.info(f"Converting M4A to WAV: {input_file} -> {output_file}")
            
            # Convert to WAV with specified parameters
            cmd = [
                'ffmpeg', '-y', '-i', str(input_file),
                '-ar', str(sample_rate),      # Sample rate
                '-ac', str(channels),         # Channels
                '-sample_fmt', 's16',         # 16-bit samples
                str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"M4A to WAV conversion failed: {result.stderr}")
                return None
            
            if not output_file.exists():
                logger.error(f"Output file was not created: {output_file}")
                return None
            
            logger.info(f"M4A conversion successful: {output_file}")
            return output_file
            
        except subprocess.TimeoutExpired:
            logger.error(f"M4A conversion timeout for {input_file}")
            return None
        except Exception as e:
            logger.error(f"M4A conversion error for {input_file}: {e}")
            return None
    
    def get_m4a_info(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an M4A file.
        
        Args:
            file_path: Path to the M4A file
            
        Returns:
            Dictionary with M4A information or None if error
        """
        file_path = Path(file_path)
        
        try:
            # Use ffprobe to get detailed information
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
            
            format_info = probe_data.get('format', {})
            streams = probe_data.get('streams', [])
            
            # Find audio stream
            audio_stream = None
            for stream in streams:
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            if not audio_stream:
                return None
            
            info = {
                'duration': float(format_info.get('duration', 0)),
                'size': int(format_info.get('size', 0)),
                'bitrate': int(format_info.get('bit_rate', 0)),
                'format': format_info.get('format_name', 'unknown'),
                'sample_rate': int(audio_stream.get('sample_rate', 0)),
                'channels': int(audio_stream.get('channels', 0)),
                'codec': audio_stream.get('codec_name', 'unknown'),
                'profile': audio_stream.get('profile', 'unknown'),
                'is_valid': True,
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting M4A info for {file_path}: {e}")
            return None
    
    def is_m4a_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if a file is an M4A file.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file is M4A
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() == '.m4a'


# Global M4A handler instance
m4a_handler = M4AHandler()
"""
Simple CLI for Nice-TTS without Unicode characters.
Provides basic functionality for testing.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List

from . import gpu_utils, audio_processor, transcriber, m4a_handler
from .gpu_utils import GPUManager
from .audio_processor import AudioProcessor
from .transcriber import Transcriber
from .m4a_handler import M4AHandler
from .ignore_handler import IgnoreHandler


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('nice-tts.log', mode='a', encoding='utf-8')
        ]
    )


def check_gpu():
    """Check GPU availability and configuration."""
    print("[GPU Check] Checking GPU availability...")
    print("-" * 50)
    
    gpu_manager = GPUManager()
    
    # Check PyTorch
    if gpu_manager.is_torch_available:
        print("[OK] PyTorch is installed")
    else:
        print("[!] PyTorch not found")
        print("Install with: pip install torch torchvision torchaudio")
        return
    
    # Check CUDA
    if gpu_manager.is_cuda_available:
        print("[OK] CUDA is available")
        
        info = gpu_manager.gpu_info
        if info:
            print(f"[INFO] CUDA version: {info['cuda_version']}")
            print(f"[INFO] GPU devices: {info['device_count']}")
            if info['devices']:
                device = info['devices'][0]
                print(f"[INFO] Primary GPU: {device['name']}")
                memory_gb = device['total_memory'] / (1024**3)
                print(f"[INFO] GPU Memory: {memory_gb:.1f} GB")
    else:
        print("[INFO] CUDA not available")
        print("For GPU acceleration:")
        print("1. Install CUDA: https://developer.nvidia.com/cuda-downloads")
        print("2. Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
    
    # Check ffmpeg
    if audio_processor.check_ffmpeg_installation():
        print("[OK] ffmpeg is available")
    else:
        print("[!] ffmpeg not found")
        print("Install from: https://ffmpeg.org/download.html")


def show_info():
    """Show system information and supported formats."""
    print("[INFO] Nice-TTS System Information")
    print("-" * 50)
    
    from . import __version__
    print(f"Version: {__version__}")
    print(f"Python: {sys.version}")
    
    # GPU info
    gpu_manager = GPUManager()
    if gpu_manager.is_cuda_available:
        print("GPU: Available")
    else:
        print("Processing: CPU only")
    
    # Supported formats
    formats = audio_processor.get_supported_formats()
    print(f"Supported formats: {', '.join(formats)}")
    
    # Whisper models
    models = transcriber.list_available_models()
    print(f"Available models: {', '.join(models)}")
    
    # Languages
    transcriber_instance = Transcriber()
    languages = transcriber_instance.get_supported_languages()
    print(f"Supported languages: {', '.join(languages[:10])}...")
    print(f"   (and {len(languages)-10} more)")


def process_audio(input_path: str, output_dir: Optional[str] = None, 
                 language: str = 'zh', model: str = 'base', device: Optional[str] = None,
                 force_cpu: bool = False, verbose: bool = False, 
                 enable_formatting: bool = True, max_line_length: int = 80,
                 phone_call_mode: bool = False):
    """Process audio files for transcription."""
    setup_logging(verbose)
    
    print("[AUDIO] Nice-TTS Audio Transcription")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Mode: {'Phone Call' if phone_call_mode else 'Standard'}")
    print(f"Language: {language}")
    print(f"Model: {model}")
    print("=" * 70)
    
    input_path = Path(input_path)
    
    # Initialize components
    gpu_manager = GPUManager()
    audio_proc = AudioProcessor()
    m4a_proc = M4AHandler()
    
    # Check GPU availability
    if force_cpu:
        device = "cpu"
        print("[DEVICE] CPU Mode (Forced)")
    elif device is None:
        device = gpu_manager.get_device()
        if device == "cuda":
            print("[DEVICE] GPU Acceleration Enabled")
            # Show GPU info if available
            gpu_info = gpu_manager.gpu_info
            if gpu_info and gpu_info.get('devices'):
                primary_gpu = gpu_info['devices'][0]
                memory_gb = primary_gpu['total_memory'] / (1024**3)
                print(f"[DEVICE] GPU: {primary_gpu['name']} ({memory_gb:.1f}GB)")
                print(f"[DEVICE] CUDA: {gpu_info.get('cuda_version', 'Unknown')}")
        else:
            print("[DEVICE] CPU Processing Mode")
    else:
        print(f"[DEVICE] {device.upper()} Mode (Specified)")
    
    # Check ffmpeg
    if not audio_processor.check_ffmpeg_installation():
        print("[!] ffmpeg is not installed")
        print("Please install ffmpeg: https://ffmpeg.org/download.html")
        return False
    
    # Determine files to process
    if input_path.is_file():
        audio_files = [input_path]
        print(f"[INPUT] Single File: {input_path.name}")
    else:
        # Get all supported audio files from directory
        audio_files = []
        print(f"[INPUT] Directory: {input_path}")
        print("[SCAN] Scanning for audio files...")
        
        supported_formats = audio_processor.get_supported_formats()
        for ext in supported_formats:
            found_files = list(input_path.glob(f"*{ext}"))
            if found_files:
                print(f"[SCAN] Found {len(found_files)} {ext.upper()} files")
            audio_files.extend(found_files)
        
        if not audio_files:
            print(f"[!] No supported audio files found in {input_path}")
            print(f"[INFO] Supported formats: {', '.join(supported_formats)}")
            return False
            
        print(f"[SCAN] Total: {len(audio_files)} files found")
    
    # Apply ignore file filtering
    ignore_handler = IgnoreHandler(input_path / ".ignorefile" if input_path.is_dir() else ".ignorefile")
    if ignore_handler.load_patterns():
        print(f"[FILTER] Applying ignore patterns from .ignorefile")
        original_count = len(audio_files)
        audio_files = ignore_handler.filter_files(audio_files, input_path if input_path.is_dir() else None)
        filtered_count = original_count - len(audio_files)
        if filtered_count > 0:
            print(f"[FILTER] Excluded {filtered_count} files based on ignore patterns")
    
    if not audio_files:
        print("[!] No audio files remaining after filtering")
        return False
    
    # Set output directory
    if output_dir is None:
        output_dir = "transcriptions"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")
    
    # Filter valid audio files
    valid_files = []
    for audio_file in audio_files:
        # Special handling for M4A files
        if m4a_proc.is_m4a_file(audio_file):
            print(f"[M4A] Validating M4A file: {audio_file.name}")
            is_valid, error_msg = m4a_proc.validate_m4a(audio_file)
            if is_valid:
                valid_files.append(audio_file)
            else:
                print(f"[!] M4A validation failed: {audio_file.name} - {error_msg}")
                # Try to repair
                print("[M4A] Attempting to repair M4A file...")
                repaired_file = m4a_proc.repair_m4a(audio_file)
                if repaired_file:
                    print(f"[OK] M4A file repaired: {repaired_file.name}")
                    valid_files.append(repaired_file)
                else:
                    print(f"[!] M4A repair failed: {audio_file.name}")
        elif audio_proc.validate_audio_file(audio_file):
            valid_files.append(audio_file)
        else:
            print(f"[!] Skipping invalid file: {audio_file}")
    
    if not valid_files:
        print("[!] No valid audio files to process")
        return False
    
    print(f"[FILTER] {len(valid_files)} valid files to process")
    
    # Initialize transcriber
    print(f"\n[SETUP] Initializing transcription system")
    print(f"[MODEL] Loading Whisper model: {model}")
    if phone_call_mode:
        print("[MODE] Phone call transcription mode enabled")
        print("[MODE] Using optimized settings: fp16=True, initial_prompt='以下为电话录音的内容。'")
    transcriber = Transcriber(model_name=model, device=device, language=language, phone_call_mode=phone_call_mode)
    
    # Explicitly load the model
    if not transcriber._load_model():
        print("[!] Failed to load Whisper model")
        return False
    
    print("[OK] Model loaded successfully")
    
    # Process files
    print("\n[WORK] Starting transcription...")
    print("=" * 70)
    
    successful = 0
    failed = 0
    
    for i, audio_file in enumerate(valid_files, 1):
        # Calculate progress percentage
        progress_pct = (i / len(valid_files)) * 100
        progress_bar = f"[{int(progress_pct):3d}%]"
        
        print(f"\n[PROGRESS] {progress_bar} File {i}/{len(valid_files)}")
        print(f"[FILE] {audio_file.name}")
        
        # Check if transcription already exists
        expected_transcription = output_path / f"{audio_file.stem}_transcription.txt"
        if expected_transcription.exists():
            print(f"[SKIP] Transcription already exists: {expected_transcription.name}")
            print(f"[SKIP] Size: {expected_transcription.stat().st_size} bytes")
            successful += 1  # Count as successful since we have a result
            continue
        
        try:
            # Special handling for M4A files - convert to WAV
            if m4a_proc.is_m4a_file(audio_file):
                # Check if converted WAV already exists in output directory
                expected_wav_file = output_path / f"{audio_file.stem}.wav"
                if expected_wav_file.exists():
                    print(f"[M4A] Using existing converted WAV file: {expected_wav_file.name}")
                    processed_file = expected_wav_file
                else:
                    print(f"[M4A] Converting M4A to WAV: {audio_file.name}")
                    processed_file = m4a_proc.convert_m4a_to_wav(audio_file, output_dir=output_path)
                    if not processed_file:
                        print(f"[!] Failed to convert M4A file")
                        failed += 1
                        continue
            else:
                # Preprocess other audio formats
                processed_file = audio_proc.preprocess_for_whisper(audio_file)
                if not processed_file:
                    print(f"[!] Failed to preprocess {audio_file}")
                    failed += 1
                    continue
            
            # Transcribe
            result = transcriber.transcribe_file(
                processed_file,
                language=language,
                enable_formatting=enable_formatting
            )
            
            if result:
                # Save transcription
                output_file = output_path / f"{audio_file.stem}_transcription.txt"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result['text'])
                
                print(f"[SUCCESS] Transcription completed")
                print(f"[RESULT] File saved: {output_file.name}")
                print(f"[STATS] Duration: {result['audio_duration']:.1f}s | "
                      f"Processing: {result['processing_time']:.1f}s | "
                      f"RTF: {result['real_time_factor']:.2f}")
                if verbose and result.get('device') == 'cuda':
                    print(f"[PERF] GPU acceleration active")
                
                successful += 1
                
            else:
                failed += 1
                print(f"[!] Failed to transcribe {audio_file.name}")
                
        except Exception as e:
            failed += 1
            print(f"[!] Error processing {audio_file.name}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("[SUMMARY] Transcription Summary")
    print("=" * 70)
    
    total_files = len(valid_files)
    success_rate = (successful / total_files * 100) if total_files > 0 else 0
    
    print(f"[RESULTS] Total Files: {total_files}")
    print(f"[RESULTS] Successful: {successful} ({success_rate:.1f}%)")
    print(f"[RESULTS] Failed: {failed}")
    print(f"[RESULTS] Skipped: {total_files - successful - failed}")
    
    if successful > 0:
        print(f"\n[OUTPUT] Results saved to: {output_path}")
        
        # Show file count by type
        transcription_files = list(output_path.glob("*_transcription.txt"))
        wav_files = list(output_path.glob("*.wav"))
        
        if transcription_files:
            print(f"[OUTPUT] Transcription files: {len(transcription_files)}")
        if wav_files:
            print(f"[OUTPUT] Converted audio files: {len(wav_files)}")
        
        print(f"\n[SUCCESS] Transcription completed successfully!")
        return True
    else:
        print(f"\n[FAILED] All transcriptions failed")
        return False


def detect_language_simple(audio_file: str, model: str = 'base', 
                          device: Optional[str] = None, force_cpu: bool = False,
                          phone_call_mode: bool = False):
    """Simple language detection."""
    print("[LANG] Language Detection")
    print("-" * 50)
    print(f"File: {audio_file}")
    
    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"[!] Audio file does not exist: {audio_file}")
        return None
    
    # Initialize transcriber
    if force_cpu:
        device = "cpu"
    elif device is None:
        device = GPUManager().get_device()
    
    transcriber = Transcriber(model_name=model, device=device, phone_call_mode=phone_call_mode)
    
    if not transcriber.is_model_loaded:
        print("[!] Failed to load Whisper model")
        return None
    
    # Detect language
    detected_lang = transcriber.detect_language(audio_file)
    
    if detected_lang:
        print(f"[OK] Detected language: {detected_lang}")
        
        # Show language name if possible
        lang_names = {
            "zh": "Chinese",
            "en": "English", 
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
        }
        
        if detected_lang in lang_names:
            print(f"Language name: {lang_names[detected_lang]}")
        
        return detected_lang
    else:
        print("[!] Could not detect language")
        return None


def main():
    """Main entry point for simple CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Nice-TTS: AI-Powered Audio Transcription')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Check GPU command
    parser_check = subparsers.add_parser('check-gpu', help='Check GPU availability')
    
    # Info command
    parser_info = subparsers.add_parser('info', help='Show system information')
    
    # Process command
    parser_process = subparsers.add_parser('process', help='Process audio files')
    parser_process.add_argument('input_path', help='Input audio file or directory')
    parser_process.add_argument('--output-dir', '-o', help='Output directory')
    parser_process.add_argument('--language', '-l', default='zh', help='Language code')
    parser_process.add_argument('--model', '-m', default='base', help='Whisper model')
    parser_process.add_argument('--device', '-d', help='Processing device')
    parser_process.add_argument('--force-cpu', action='store_true', help='Force CPU usage')
    parser_process.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser_process.add_argument('--no-format', action='store_true', help='Disable text formatting')
    parser_process.add_argument('--max-line-length', type=int, default=80, help='Maximum line length for formatting')
    parser_process.add_argument('--phone-call', action='store_true', help='Enable phone call transcription mode with optimized settings')
    
    # Detect language command
    parser_detect = subparsers.add_parser('detect-language', help='Detect audio language')
    parser_detect.add_argument('audio_file', help='Audio file to analyze')
    parser_detect.add_argument('--model', '-m', default='base', help='Whisper model')
    parser_detect.add_argument('--device', '-d', help='Processing device')
    parser_detect.add_argument('--force-cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    if args.command == 'check-gpu':
        check_gpu()
    elif args.command == 'info':
        show_info()
    elif args.command == 'process':
        success = process_audio(
            args.input_path,
            output_dir=args.output_dir,
            language=args.language,
            model=args.model,
            device=args.device,
            force_cpu=args.force_cpu,
            verbose=args.verbose,
            enable_formatting=not args.no_format,
            max_line_length=args.max_line_length,
            phone_call_mode=args.phone_call
        )
        sys.exit(0 if success else 1)
    elif args.command == 'detect-language':
        result = detect_language_simple(
            args.audio_file,
            model=args.model,
            device=args.device,
            force_cpu=args.force_cpu
        )
        sys.exit(0 if result else 1)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
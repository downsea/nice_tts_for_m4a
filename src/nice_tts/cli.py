"""
Command-line interface for Nice-TTS.
Provides user-friendly CLI commands for audio transcription.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List
import click
from dotenv import load_dotenv
from tqdm import tqdm
import colorama
from colorama import Fore, Style

# Import our modules
from . import gpu_utils, audio_processor, transcriber, m4a_handler
from .gpu_utils import GPUManager
from .audio_processor import AudioProcessor
from .transcriber import Transcriber
from .m4a_handler import M4AHandler

# Initialize colorama for cross-platform colored output
colorama.init()

# Load environment variables
load_dotenv()

# Configure logging
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

# Custom Click group for better help formatting
class NiceTTSGroup(click.Group):
    def format_help(self, ctx, formatter):
        """Custom help formatting."""
        print(f"{Fore.CYAN}🎤 Nice-TTS: AI-Powered Audio Transcription{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Optimized for Chinese Language Processing{Style.RESET_ALL}")
        print()
        super().format_help(ctx, formatter)


@click.group(cls=NiceTTSGroup, invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version information')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def main(ctx, version, verbose):
    """Nice-TTS: AI-powered audio transcription tool with Chinese optimization."""
    
    if version:
        from . import __version__
        print(f"Nice-TTS version {__version__}")
        return
        
    if ctx.invoked_subcommand is None:
        print(f"{Fore.CYAN}🎤 Nice-TTS{Style.RESET_ALL}")
        print(f"Use {Fore.YELLOW}--help{Style.RESET_ALL} to see available commands")
        print(f"Use {Fore.YELLOW}process{Style.RESET_ALL} command to transcribe audio files")


@main.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output-dir', '-o', type=click.Path(path_type=Path), 
              help='Output directory for transcriptions (default: ./transcriptions)')
@click.option('--language', '-l', default='zh', 
              help='Language code (default: zh for Chinese)')
@click.option('--model', '-m', default='large-v3-turbo',
              help='Whisper model (tiny, base, small, medium, large, large-v3-turbo, etc.)')
@click.option('--device', '-d', 
              help='Processing device (cuda/cpu, auto-detected by default)')
@click.option('--force-cpu', is_flag=True, 
              help='Force CPU usage even if GPU is available')
@click.option('--temperature', '-t', type=float, default=0.0,
              help='Sampling temperature (0.0-1.0, default: 0.0)')
@click.option('--beam-size', type=int, default=5,
              help='Beam size for decoding (default: 5)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--dry-run', is_flag=True, 
              help='Show what would be processed without actually transcribing')
def process(input_path: Path, output_dir: Optional[Path], language: str, 
           model: str, device: Optional[str], force_cpu: bool, temperature: float,
           beam_size: int, verbose: bool, dry_run: bool):
    """
    Process audio files for transcription.
    
    INPUT_PATH can be a single audio file or a directory containing audio files.
    Supported formats: wav, mp3, m4a, flac, ogg, wma, aac, aiff, au
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    print(f"{Fore.CYAN}🎤 Nice-TTS Audio Transcription{Style.RESET_ALL}")
    print(f"{Fore.GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.RESET_ALL}")
    
    # Initialize components
    gpu_manager = GPUManager()
    audio_proc = AudioProcessor()
    m4a_proc = M4AHandler()
    
    # Check GPU availability
    if force_cpu:
        device = "cpu"
        print(f"{Fore.YELLOW}⚠️  Forcing CPU usage{Style.RESET_ALL}")
    elif device is None:
        device = gpu_manager.get_device()
        if device == "cuda":
            print(f"{Fore.GREEN}✅ Using GPU acceleration{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}ℹ️  Using CPU processing{Style.RESET_ALL}")
    
    # Check ffmpeg
    if not audio_processor.check_ffmpeg_installation():
        print(f"{Fore.RED}❌ ffmpeg is not installed{Style.RESET_ALL}")
        print(f"Please install ffmpeg: https://ffmpeg.org/download.html")
        sys.exit(1)
    
    # Determine files to process
    if input_path.is_file():
        audio_files = [input_path]
        print(f"📁 Processing single file: {input_path}")
    else:
        # Get all supported audio files from directory
        audio_files = []
        for ext in audio_processor.get_supported_formats():
            audio_files.extend(input_path.glob(f"*{ext}"))
        
        if not audio_files:
            print(f"{Fore.RED}❌ No supported audio files found in {input_path}{Style.RESET_ALL}")
            sys.exit(1)
            
        print(f"📁 Found {len(audio_files)} audio files in {input_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = Path("transcriptions")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Output directory: {output_dir}")
    
    # Filter valid audio files
    valid_files = []
    for audio_file in audio_files:
        # Special handling for M4A files
        if m4a_proc.is_m4a_file(audio_file):
            print(f"{Fore.BLUE}🔍 Validating M4A file: {audio_file.name}{Style.RESET_ALL}")
            is_valid, error_msg = m4a_proc.validate_m4a(audio_file)
            if is_valid:
                valid_files.append(audio_file)
            else:
                print(f"{Fore.YELLOW}⚠️  M4A validation failed: {audio_file.name} - {error_msg}{Style.RESET_ALL}")
                # Try to repair
                print(f"{Fore.CYAN}🔧 Attempting to repair M4A file...{Style.RESET_ALL}")
                repaired_file = m4a_proc.repair_m4a(audio_file)
                if repaired_file:
                    print(f"{Fore.GREEN}✅ M4A file repaired: {repaired_file.name}{Style.RESET_ALL}")
                    valid_files.append(repaired_file)
                else:
                    print(f"{Fore.RED}❌ M4A repair failed: {audio_file.name}{Style.RESET_ALL}")
        elif audio_proc.validate_audio_file(audio_file):
            valid_files.append(audio_file)
        else:
            print(f"{Fore.YELLOW}⚠️  Skipping invalid file: {audio_file}{Style.RESET_ALL}")
    
    if not valid_files:
        print(f"{Fore.RED}❌ No valid audio files to process{Style.RESET_ALL}")
        sys.exit(1)
    
    print(f"✅ {len(valid_files)} valid files to process")
    
    if dry_run:
        print(f"\n{Fore.CYAN}🔍 Dry run - would process:{Style.RESET_ALL}")
        for file in valid_files:
            print(f"  • {file}")
        print(f"\n{Fore.GREEN}Dry run completed{Style.RESET_ALL}")
        return
    
    # Initialize transcriber
    print(f"\n{Fore.CYAN}🤖 Loading Whisper model: {model}{Style.RESET_ALL}")
    transcriber = Transcriber(model_name=model, device=device, language=language)
    
    if not transcriber.is_model_loaded:
        print(f"{Fore.RED}❌ Failed to load Whisper model{Style.RESET_ALL}")
        sys.exit(1)
    
    print(f"{Fore.GREEN}✅ Model loaded successfully{Style.RESET_ALL}")
    
    # Process files
    print(f"\n{Fore.CYAN}🔄 Starting transcription...{Style.RESET_ALL}")
    print(f"{Fore.GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.RESET_ALL}")
    
    successful = 0
    failed = 0
    
    for audio_file in tqdm(valid_files, desc="Transcribing", unit="file"):
        try:
            # Special handling for M4A files - convert to WAV
            if m4a_proc.is_m4a_file(audio_file):
                print(f"{Fore.BLUE}🔄 Converting M4A to WAV: {audio_file.name}{Style.RESET_ALL}")
                processed_file = m4a_proc.convert_m4a_to_wav(audio_file)
                if not processed_file:
                    print(f"{Fore.RED}❌ Failed to convert M4A file{Style.RESET_ALL}")
                    failed += 1
                    continue
            else:
                # Preprocess other audio formats
                processed_file = audio_proc.preprocess_for_whisper(audio_file)
                if not processed_file:
                    print(f"{Fore.RED}❌ Failed to preprocess {audio_file}{Style.RESET_ALL}")
                    failed += 1
                    continue
            
            # Transcribe
            result = transcriber.transcribe_file(
                processed_file,
                language=language,
                temperature=temperature,
                beam_size=beam_size
            )
            
            if result:
                # Save transcription
                output_file = output_dir / f"{audio_file.stem}_transcription.txt"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result['text'])
                
                # Save detailed result if verbose
                if verbose:
                    detailed_file = output_dir / f"{audio_file.stem}_detailed.txt"
                    formatted = transcriber.format_transcription_result(result, include_segments=True)
                    with open(detailed_file, 'w', encoding='utf-8') as f:
                        f.write(formatted)
                
                successful += 1
                
                if verbose:
                    print(f"\n{Fore.GREEN}✅ {audio_file.name}{Style.RESET_ALL}")
                    print(f"   Duration: {result['audio_duration']:.1f}s")
                    print(f"   Processing time: {result['processing_time']:.1f}s")
                    print(f"   RTF: {result['real_time_factor']:.2f}")
                    
            else:
                failed += 1
                print(f"{Fore.RED}❌ Failed to transcribe {audio_file.name}{Style.RESET_ALL}")
                
        except Exception as e:
            failed += 1
            print(f"{Fore.RED}❌ Error processing {audio_file.name}: {e}{Style.RESET_ALL}")
            logger.exception(f"Error processing {audio_file}")
    
    # Summary
    print(f"\n{Fore.CYAN}📊 Transcription Summary{Style.RESET_ALL}")
    print(f"{Fore.GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.RESET_ALL}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📂 Results saved to: {output_dir}")
    
    if successful > 0:
        print(f"\n{Fore.GREEN}🎉 Transcription completed successfully!{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}❌ All transcriptions failed{Style.RESET_ALL}")
        sys.exit(1)


@main.command()
@click.option('--verbose', '-v', is_flag=True, help='Show detailed GPU information')
def check_gpu(verbose: bool):
    """Check GPU availability and configuration."""
    setup_logging(verbose)
    
    print(f"{Fore.CYAN}🎮 GPU Support Check{Style.RESET_ALL}")
    print(f"{Fore.GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.RESET_ALL}")
    
    gpu_manager = GPUManager()
    
    # Check PyTorch
    if gpu_manager.is_torch_available:
        print(f"{Fore.GREEN}✅ PyTorch is installed{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}❌ PyTorch not found{Style.RESET_ALL}")
        print(f"Install with: pip install torch torchvision torchaudio")
        return
    
    # Check CUDA
    if gpu_manager.is_cuda_available:
        print(f"{Fore.GREEN}✅ CUDA is available{Style.RESET_ALL}")
        
        if verbose:
            gpu_manager.print_gpu_info()
        else:
            info = gpu_manager.gpu_info
            if info:
                print(f"📊 CUDA version: {info['cuda_version']}")
                print(f"🔢 GPU devices: {info['device_count']}")
                if info['devices']:
                    device = info['devices'][0]
                    print(f"🎯 Primary GPU: {device['name']}")
                    memory_gb = device['total_memory'] / (1024**3)
                    print(f"💾 GPU Memory: {memory_gb:.1f} GB")
    else:
        print(f"{Fore.YELLOW}ℹ️  CUDA not available{Style.RESET_ALL}")
        print(f"For GPU acceleration:")
        print(f"1. Install CUDA: https://developer.nvidia.com/cuda-downloads")
        print(f"2. Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
    
    # Check ffmpeg
    if audio_processor.check_ffmpeg_installation():
        print(f"{Fore.GREEN}✅ ffmpeg is available{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}❌ ffmpeg not found{Style.RESET_ALL}")
        print(f"Install from: https://ffmpeg.org/download.html")


@main.command()
def info():
    """Show system information and supported formats."""
    print(f"{Fore.CYAN}ℹ️  Nice-TTS System Information{Style.RESET_ALL}")
    print(f"{Fore.GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.RESET_ALL}")
    
    from . import __version__
    print(f"📦 Version: {__version__}")
    print(f"🐍 Python: {sys.version}")
    
    # GPU info
    gpu_manager = GPUManager()
    if gpu_manager.is_cuda_available:
        print(f"🎮 GPU: Available")
    else:
        print(f"💻 Processing: CPU only")
    
    # Supported formats
    formats = audio_processor.get_supported_formats()
    print(f"🎵 Supported formats: {', '.join(formats)}")
    
    # Whisper models
    models = transcriber.list_available_models()
    print(f"🤖 Available models: {', '.join(models)}")
    
    # Languages
    languages = transcriber.get_supported_languages()
    print(f"🌍 Supported languages: {', '.join(languages[:10])}...")
    print(f"   (and {len(languages)-10} more)")


@main.command()
@click.argument('audio_file', type=click.Path(exists=True, path_type=Path))
@click.option('--language', '-l', help='Expected language (for faster processing)')
@click.option('--model', '-m', default='base', help='Whisper model to use')
@click.option('--device', '-d', help='Processing device')
@click.option('--force-cpu', is_flag=True, help='Force CPU usage')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed information')
def detect_language(audio_file: Path, language: Optional[str], model: str, 
                   device: Optional[str], force_cpu: bool, verbose: bool):
    """Detect the language of an audio file."""
    setup_logging(verbose)
    
    print(f"{Fore.CYAN}🔍 Language Detection{Style.RESET_ALL}")
    print(f"{Fore.GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.RESET_ALL}")
    print(f"📁 File: {audio_file}")
    
    # Initialize transcriber
    if force_cpu:
        device = "cpu"
    elif device is None:
        device = GPUManager().get_device()
    
    transcriber = Transcriber(model_name=model, device=device)
    
    if not transcriber.is_model_loaded:
        print(f"{Fore.RED}❌ Failed to load Whisper model{Style.RESET_ALL}")
        return
    
    # Detect language
    detected_lang = transcriber.detect_language(audio_file)
    
    if detected_lang:
        print(f"{Fore.GREEN}✅ Detected language: {detected_lang}{Style.RESET_ALL}")
        
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
            print(f"🌍 Language name: {lang_names[detected_lang]}")
    else:
        print(f"{Fore.YELLOW}⚠️  Could not detect language{Style.RESET_ALL}")


if __name__ == '__main__':
    main()
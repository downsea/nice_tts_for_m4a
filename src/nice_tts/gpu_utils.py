"""
GPU utilities for Nice-TTS.
Handles CUDA detection and GPU acceleration configuration.
"""

import os
import sys
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GPUManager:
    """Manages GPU detection and configuration for audio transcription."""
    
    def __init__(self):
        self._cuda_available = None
        self._gpu_info = None
        self._torch_available = None
        
    @property
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available on the system."""
        if self._cuda_available is None:
            self._cuda_available = self._check_cuda_availability()
        return self._cuda_available
    
    @property
    def is_torch_available(self) -> bool:
        """Check if PyTorch is available."""
        if self._torch_available is None:
            self._torch_available = self._check_torch_availability()
        return self._torch_available
    
    @property
    def gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed GPU information."""
        if self._gpu_info is None:
            self._gpu_info = self._get_gpu_info()
        return self._gpu_info
    
    def _check_torch_availability(self) -> bool:
        """Check if PyTorch is available and properly installed."""
        try:
            import torch
            return True
        except ImportError:
            logger.warning("PyTorch not found. GPU acceleration will not be available.")
            return False
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available on the system."""
        if not self.is_torch_available:
            return False
            
        try:
            import torch
            
            # Check if CUDA is available through PyTorch
            if not torch.cuda.is_available():
                logger.info("CUDA not available through PyTorch")
                return False
            
            # Check CUDA version
            cuda_version = torch.version.cuda
            if cuda_version is None:
                logger.warning("CUDA version not detected")
                return False
                
            logger.info(f"CUDA {cuda_version} detected")
            
            # Check GPU count
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                logger.warning("No CUDA devices found")
                return False
                
            logger.info(f"Found {gpu_count} CUDA device(s)")
            return True
            
        except Exception as e:
            logger.error(f"Error checking CUDA availability: {e}")
            return False
    
    def _get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed information about available GPUs."""
        if not self.is_cuda_available:
            return None
            
        try:
            import torch
            
            gpu_info = {
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "devices": []
            }
            
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    "id": i,
                    "name": device_props.name,
                    "total_memory": device_props.total_memory,
                    "major": device_props.major,
                    "minor": device_props.minor,
                    "multi_processor_count": device_props.multi_processor_count,
                }
                gpu_info["devices"].append(device_info)
                
            return gpu_info
            
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            return None
    
    def get_device(self, force_cpu: bool = False) -> str:
        """
        Get the best available device for processing.
        
        Args:
            force_cpu: Force CPU usage even if GPU is available
            
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if force_cpu:
            logger.info("Forcing CPU usage as requested")
            return "cpu"
            
        if self.is_cuda_available:
            logger.info("Using CUDA for acceleration")
            return "cuda"
        else:
            logger.info("Using CPU for processing")
            return "cpu"
    
    def get_optimal_batch_size(self, device: str = None) -> int:
        """
        Get optimal batch size for the given device.
        
        Args:
            device: Device type ('cuda' or 'cpu'). If None, uses best available.
            
        Returns:
            Recommended batch size
        """
        if device is None:
            device = self.get_device()
            
        if device == "cuda" and self.gpu_info:
            # Calculate based on GPU memory
            # Conservative estimate: 1GB per batch for medium models
            total_memory = self.gpu_info["devices"][0]["total_memory"]
            # Reserve some memory for system and other processes
            available_memory = total_memory - 2 * 1024**3  # Reserve 2GB
            batch_size = max(1, available_memory // (1024**3))  # 1GB per batch
            return min(batch_size, 8)  # Cap at 8 for safety
        else:
            # Conservative batch size for CPU
            return 1
    
    def print_gpu_info(self) -> None:
        """Print detailed GPU information."""
        if not self.is_cuda_available:
            print("❌ CUDA is not available")
            print("💡 GPU acceleration will not be used")
            return
            
        info = self.gpu_info
        if info:
            print(f"✅ CUDA {info['cuda_version']} is available")
            print(f"📊 Found {info['device_count']} GPU device(s):")
            
            for device in info["devices"]:
                memory_gb = device["total_memory"] / (1024**3)
                print(f"   🔸 Device {device['id']}: {device['name']}")
                print(f"      💾 Memory: {memory_gb:.1f} GB")
                print(f"      🔧 Compute Capability: {device['major']}.{device['minor']}")
                print(f"      ⚡ Multi-processors: {device['multi_processor_count']}")
        else:
            print("❌ Could not retrieve GPU information")


def check_gpu_requirements() -> bool:
    """
    Check if the system meets GPU requirements for Nice-TTS.
    
    Returns:
        True if GPU acceleration is available and recommended
    """
    gpu_manager = GPUManager()
    
    if not gpu_manager.is_torch_available:
        print("[!] PyTorch not found. Install with: pip install torch torchvision torchaudio")
        return False
        
    if gpu_manager.is_cuda_available:
        print("[OK] GPU acceleration is available and will be used")
        return True
    else:
        print("[INFO] GPU acceleration not available, using CPU")
        print("[HINT] For GPU support, install CUDA and PyTorch with CUDA support")
        return False


def get_device_preference() -> str:
    """
    Get user device preference from environment or system detection.
    
    Returns:
        Device preference ('cuda' or 'cpu')
    """
    # Check environment variable
    force_cpu = os.getenv("FORCE_CPU", "false").lower() == "true"
    
    gpu_manager = GPUManager()
    return gpu_manager.get_device(force_cpu=force_cpu)


# Global GPU manager instance
gpu_manager = GPUManager()
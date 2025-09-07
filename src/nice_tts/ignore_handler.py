"""
Ignore file handler for Nice-TTS.
Handles .ignorefile parsing and file exclusion patterns.
"""

import logging
from pathlib import Path
from typing import List, Set
import fnmatch
import os

logger = logging.getLogger(__name__)


class IgnoreHandler:
    """Handles .ignorefile parsing and file exclusion patterns."""
    
    def __init__(self, ignore_file_path: str = ".ignorefile"):
        """
        Initialize the ignore handler.
        
        Args:
            ignore_file_path: Path to the .ignorefile
        """
        self.ignore_file_path = Path(ignore_file_path)
        self.patterns = []
        self.loaded = False
        
    def load_patterns(self) -> bool:
        """
        Load ignore patterns from the .ignorefile.
        
        Returns:
            True if patterns were loaded successfully
        """
        if not self.ignore_file_path.exists():
            logger.debug(f"No ignore file found at: {self.ignore_file_path}")
            return False
        
        try:
            with open(self.ignore_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            self.patterns = []
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Skip invalid patterns
                try:
                    # Test the pattern to make sure it's valid
                    fnmatch.fnmatch("test", line)
                    self.patterns.append(line)
                    logger.debug(f"Loaded ignore pattern: {line}")
                except Exception as e:
                    logger.warning(f"Invalid pattern at line {line_num}: {line} - {e}")
            
            self.loaded = True
            logger.info(f"Loaded {len(self.patterns)} ignore patterns from {self.ignore_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ignore file {self.ignore_file_path}: {e}")
            return False
    
    def should_ignore(self, file_path: Path, base_path: Path = None) -> bool:
        """
        Check if a file should be ignored based on loaded patterns.
        
        Args:
            file_path: Path to the file to check
            base_path: Base path for relative pattern matching (optional)
            
        Returns:
            True if the file should be ignored
        """
        if not self.patterns:
            return False
        
        try:
            # Convert to relative path for pattern matching
            if base_path and file_path.is_absolute():
                try:
                    rel_path = file_path.relative_to(base_path)
                except ValueError:
                    # File is not relative to base_path, use absolute path
                    rel_path = file_path
            else:
                rel_path = file_path
            
            # Convert to string for pattern matching
            path_str = str(rel_path).replace('\\', '/')  # Normalize path separators
            name_str = file_path.name
            
            # Check each pattern
            for pattern in self.patterns:
                # Check against full path
                if fnmatch.fnmatch(path_str, pattern):
                    logger.debug(f"Ignoring file (path match): {file_path} (pattern: {pattern})")
                    return True
                
                # Check against filename only
                if fnmatch.fnmatch(name_str, pattern):
                    logger.debug(f"Ignoring file (name match): {file_path} (pattern: {pattern})")
                    return True
                
                # Check against parent directory patterns (for directory patterns)
                if pattern.endswith('/'):
                    # Check if any parent directory matches
                    for parent in rel_path.parents:
                        if fnmatch.fnmatch(str(parent).replace('\\', '/'), pattern.rstrip('/')):
                            logger.debug(f"Ignoring file (parent dir match): {file_path} (pattern: {pattern})")
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking ignore pattern for {file_path}: {e}")
            return False
    
    def filter_files(self, file_paths: List[Path], base_path: Path = None) -> List[Path]:
        """
        Filter a list of files, removing ignored ones.
        
        Args:
            file_paths: List of file paths to filter
            base_path: Base path for relative pattern matching
            
        Returns:
            List of files that should not be ignored
        """
        if not self.patterns:
            return file_paths
        
        filtered_files = []
        ignored_count = 0
        
        for file_path in file_paths:
            if self.should_ignore(file_path, base_path):
                ignored_count += 1
            else:
                filtered_files.append(file_path)
        
        if ignored_count > 0:
            logger.info(f"Filtered out {ignored_count} files based on ignore patterns")
        
        return filtered_files
    
    def get_ignore_stats(self) -> dict:
        """Get statistics about loaded ignore patterns."""
        return {
            'loaded': self.loaded,
            'patterns_count': len(self.patterns),
            'patterns': self.patterns.copy(),
            'ignore_file_path': str(self.ignore_file_path)
        }
    
    def __str__(self) -> str:
        """String representation of the ignore handler."""
        status = "loaded" if self.loaded else "not loaded"
        return f"IgnoreHandler({status}, {len(self.patterns)} patterns)"
    
    def __repr__(self) -> str:
        """Detailed representation of the ignore handler."""
        return (f"IgnoreHandler(loaded={self.loaded}, patterns={len(self.patterns)}, "
                f"ignore_file={self.ignore_file_path})")
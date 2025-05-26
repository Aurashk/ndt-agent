"""
debug.py

Debug logging system for ultrasonic NDT C-scan generation project.
Provides comprehensive logging and data capture when running in debug mode.
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt


class DebugLogger:
    """
    Global debug logger that activates only when Python is run in debug mode.
    
    Creates timestamped folders and captures debug information, images,
    and intermediate data structures.
    """
    
    def __init__(self):
        self._debug_active = self._check_debug_mode()
        self._session_folder = None
        self._image_counter = 0
        self._logger = None
        
        if self._debug_active:
            self._initialize_debug_session()
    
    def _check_debug_mode(self) -> bool:
        """Check if Python is running in debug mode."""
        # Check various debug indicators
        debug_indicators = [
            __debug__,  # Python's built-in debug flag
            os.environ.get('PYTHON_DEBUG', '').lower() in ('1', 'true', 'yes'),
            hasattr(sys, 'gettrace') and sys.gettrace() is not None,
            '-d' in sys.argv or '--debug' in sys.argv
        ]
        
        return any(debug_indicators)
    
    def _initialize_debug_session(self):
        """Initialize debug session with timestamped folder."""
        if not self._debug_active:
            return
            
        # Create debug root directory
        project_root = Path.cwd()
        debug_root = project_root / "debug"
        debug_root.mkdir(exist_ok=True)
        
        # Create timestamped session folder
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._session_folder = debug_root / timestamp
        self._session_folder.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self._session_folder / "images").mkdir(exist_ok=True)
        (self._session_folder / "data").mkdir(exist_ok=True)
        
        # Initialize logger
        self._setup_logger()
        
        # Save session configuration
        self._save_session_config()
        
        self.info("Debug session initialized", extra={
            'session_folder': str(self._session_folder),
            'python_version': sys.version,
            'debug_mode': self._debug_active
        })
    
    def _setup_logger(self):
        """Set up the debug logger with file and console handlers."""
        if not self._debug_active:
            return
            
        # Create logger
        self._logger = logging.getLogger('ultrasonic_debug')
        self._logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self._logger.handlers.clear()
        
        # File handler
        log_file = self._session_folder / "debug.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)
    
    def _save_session_config(self):
        """Save session configuration to JSON file."""
        if not self._debug_active:
            return
            
        config = {
            'session_start': datetime.now().isoformat(),
            'python_version': sys.version,
            'platform': sys.platform,
            'debug_mode': self._debug_active,
            'command_line': sys.argv,
            'working_directory': str(Path.cwd()),
            'environment_variables': {
                k: v for k, v in os.environ.items() 
                if 'PYTHON' in k or 'DEBUG' in k
            }
        }
        
        config_file = self._session_folder / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _format_message(self, message: str, extra: Optional[Dict] = None) -> str:
        """Format log message with optional context."""
        if extra:
            extra_str = " | " + " ".join(f"{k}={v}" for k, v in extra.items())
            return message + extra_str
        return message
    
    def info(self, message: str, extra: Optional[Dict] = None) -> None:
        """Log info message with optional context."""
        if not self._debug_active or not self._logger:
            return
        self._logger.info(self._format_message(message, extra))
    
    def warning(self, message: str, extra: Optional[Dict] = None) -> None:
        """Log warning message."""
        if not self._debug_active or not self._logger:
            return
        self._logger.warning(self._format_message(message, extra))
    
    def error(self, message: str, extra: Optional[Dict] = None) -> None:
        """Log error message."""
        if not self._debug_active or not self._logger:
            return
        self._logger.error(self._format_message(message, extra))
    
    def debug(self, message: str, extra: Optional[Dict] = None) -> None:
        """Log debug message (only in debug mode)."""
        if not self._debug_active or not self._logger:
            return
        self._logger.debug(self._format_message(message, extra))
    
    def save_image(self, figure: plt.Figure, name: str, description: str = "") -> Optional[str]:
        """
        Save matplotlib figure with auto-incrementing filename.
        
        Args:
            figure: Matplotlib figure to save
            name: Base name for the image file
            description: Optional description for logging
            
        Returns:
            Path to saved image file, or None if debug mode inactive
        """
        if not self._debug_active:
            return None
            
        self._image_counter += 1
        filename = f"{self._image_counter:03d}_{name}.png"
        filepath = self._session_folder / "images" / filename
        
        try:
            figure.savefig(filepath, dpi=150, bbox_inches='tight')
            self.debug(f"Saved debug image: {filename}", extra={
                'description': description,
                'image_path': str(filepath)
            })
            return str(filepath)
        except Exception as e:
            self.error(f"Failed to save image {filename}: {e}")
            return None
    
    def save_data(self, data: Any, name: str, description: str = "") -> Optional[str]:
        """
        Save Python object using pickle.
        
        Args:
            data: Python object to save
            name: Base name for the data file
            description: Optional description for logging
            
        Returns:
            Path to saved data file, or None if debug mode inactive
        """
        if not self._debug_active:
            return None
            
        filename = f"{name}.pkl"
        filepath = self._session_folder / "data" / filename
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            self.debug(f"Saved debug data: {filename}", extra={
                'description': description,
                'data_path': str(filepath),
                'data_type': type(data).__name__
            })
            return str(filepath)
        except Exception as e:
            self.error(f"Failed to save data {filename}: {e}")
            return None
    
    def save_array(self, array: np.ndarray, name: str, description: str = "") -> Optional[str]:
        """
        Save numpy array efficiently.
        
        Args:
            array: Numpy array to save
            name: Base name for the array file
            description: Optional description for logging
            
        Returns:
            Path to saved array file, or None if debug mode inactive
        """
        if not self._debug_active:
            return None
            
        filename = f"{name}.npy"
        filepath = self._session_folder / "data" / filename
        
        try:
            np.save(filepath, array)
            self.debug(f"Saved debug array: {filename}", extra={
                'description': description,
                'array_path': str(filepath),
                'array_shape': array.shape,
                'array_dtype': str(array.dtype)
            })
            return str(filepath)
        except Exception as e:
            self.error(f"Failed to save array {filename}: {e}")
            return None
    
    def save_config(self, config: Dict, name: str, description: str = "") -> Optional[str]:
        """
        Save configuration as JSON.
        
        Args:
            config: Configuration dictionary to save
            name: Base name for the config file
            description: Optional description for logging
            
        Returns:
            Path to saved config file, or None if debug mode inactive
        """
        if not self._debug_active:
            return None
            
        filename = f"{name}.json"
        filepath = self._session_folder / "data" / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            self.debug(f"Saved debug config: {filename}", extra={
                'description': description,
                'config_path': str(filepath),
                'config_keys': list(config.keys())
            })
            return str(filepath)
        except Exception as e:
            self.error(f"Failed to save config {filename}: {e}")
            return None
    
    def get_debug_folder(self) -> Optional[Path]:
        """Get current debug session folder path."""
        return self._session_folder if self._debug_active else None
    
    def is_active(self) -> bool:
        """Check if debug mode is active."""
        return self._debug_active
    
    def start_session(self, session_name: str) -> None:
        """Start a new debug session with custom name."""
        if not self._debug_active:
            return
            
        # Close current session
        self.end_session()
        
        # Create new session folder
        project_root = Path.cwd()
        debug_root = project_root / "debug"
        debug_root.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._session_folder = debug_root / f"{timestamp}_{session_name}"
        self._session_folder.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self._session_folder / "images").mkdir(exist_ok=True)
        (self._session_folder / "data").mkdir(exist_ok=True)
        
        # Reset counters
        self._image_counter = 0
        
        # Reinitialize logger
        self._setup_logger()
        self._save_session_config()
        
        self.info(f"Started new debug session: {session_name}")
    
    def end_session(self) -> None:
        """End current debug session."""
        if not self._debug_active or not self._logger:
            return
            
        self.info("Ending debug session")
        
        # Close logger handlers
        for handler in self._logger.handlers[:]:
            handler.close()
            self._logger.removeHandler(handler)
    
    def cleanup_old_sessions(self, days: int = 7) -> None:
        """Remove debug sessions older than specified days."""
        if not self._debug_active:
            return
            
        project_root = Path.cwd()
        debug_root = project_root / "debug"
        
        if not debug_root.exists():
            return
            
        import time
        cutoff_time = time.time() - (days * 24 * 3600)
        
        for session_dir in debug_root.iterdir():
            if session_dir.is_dir() and session_dir.stat().st_mtime < cutoff_time:
                import shutil
                shutil.rmtree(session_dir)
                self.info(f"Cleaned up old debug session: {session_dir.name}")
    
    def session(self, session_name: str):
        """Context manager for debug sessions."""
        return DebugSession(self, session_name)


class DebugSession:
    """Context manager for debug sessions."""
    
    def __init__(self, debug_logger: DebugLogger, session_name: str):
        self.debug_logger = debug_logger
        self.session_name = session_name
        self.original_session = None
    
    def __enter__(self):
        self.original_session = self.debug_logger.get_debug_folder()
        self.debug_logger.start_session(self.session_name)
        return self.debug_logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.debug_logger.end_session()
        if self.original_session:
            # Restore original session if it existed
            self.debug_logger._session_folder = self.original_session
            self.debug_logger._setup_logger()


# Global debug logger instance
debug_logger = DebugLogger()


# Convenience functions for backward compatibility
def debug_info(message: str, extra: Optional[Dict] = None) -> None:
    """Log info message."""
    debug_logger.info(message, extra)


def debug_warning(message: str, extra: Optional[Dict] = None) -> None:
    """Log warning message."""
    debug_logger.warning(message, extra)


def debug_error(message: str, extra: Optional[Dict] = None) -> None:
    """Log error message."""
    debug_logger.error(message, extra)


def debug_debug(message: str, extra: Optional[Dict] = None) -> None:
    """Log debug message."""
    debug_logger.debug(message, extra)


def save_debug_image(figure: plt.Figure, name: str, description: str = "") -> Optional[str]:
    """Save debug image."""
    return debug_logger.save_image(figure, name, description)


def save_debug_data(data: Any, name: str, description: str = "") -> Optional[str]:
    """Save debug data."""
    return debug_logger.save_data(data, name, description)


def is_debug_active() -> bool:
    """Check if debug mode is active."""
    return debug_logger.is_active()
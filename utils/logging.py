"""
Logging Utilities
Handles logging configuration and formatting.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Set up logging with consistent formatting.
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    
    # Create default log file in logs directory
    else:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        default_log_file = log_dir / f"agent_{timestamp}.log"
        
        file_handler = logging.FileHandler(default_log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging to {default_log_file}")
    
    # Set library logging levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    logging.debug("Logging initialized")

class LoggerMixin:
    """
    Mixin class that adds a logger to any class.
    The logger name will be derived from the class name.
    """
    
    @property
    def logger(self):
        """Get a logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger


def log_execution_time(logger, description: str = "Operation"):
    """
    Decorator to log the execution time of a function.
    
    Args:
        logger: Logger to use
        description: Description of the operation
    
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"Starting {description}...")
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                logger.debug(f"Completed {description} in {duration:.3f} seconds")
                return result
            
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.error(f"Failed {description} after {duration:.3f} seconds: {e}")
                raise
        
        return wrapper
    
    return decorator

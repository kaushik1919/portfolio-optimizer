"""
Standardized logging configuration for Portfolio GNN.

This module provides a centralized logging setup that:
- Uses a consistent format across all modules
- Supports both console and file output
- Is configured via experiment configuration
- Has no side effects at import time

Usage:
    from portfolio_gnn.core.logging import setup_logging
    
    setup_logging(config)  # Configure based on experiment config
    
    # Then in any module:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Message")
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from portfolio_gnn.core.paths import ProjectPaths


# Default format: timestamp - level - module - message
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = "INFO"


class LoggingConfig:
    """
    Container for logging configuration parameters.
    
    Extracts logging-relevant settings from experiment configuration.
    """
    
    def __init__(
        self,
        level: str = DEFAULT_LEVEL,
        format_string: str = DEFAULT_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT,
        console_enabled: bool = True,
        file_enabled: bool = False,
        file_path: Optional[str] = None,
        propagate: bool = True,
    ) -> None:
        self.level = level.upper()
        self.format_string = format_string
        self.date_format = date_format
        self.console_enabled = console_enabled
        self.file_enabled = file_enabled
        self.file_path = file_path
        self.propagate = propagate
    
    @classmethod
    def from_config(cls, config: Any) -> LoggingConfig:
        """
        Create LoggingConfig from experiment configuration.
        
        Expects configuration structure:
            runtime:
                logging:
                    level: INFO
                    console: true
                    file: true
                    file_path: experiment.log  # optional
        
        Args:
            config: ExperimentConfig or dict-like object
            
        Returns:
            LoggingConfig instance
        """
        # Navigate to logging section with defaults
        runtime = getattr(config, "runtime", config.get("runtime", {})) if hasattr(config, "get") else config.runtime
        
        if hasattr(runtime, "get"):
            logging_cfg = runtime.get("logging", {})
        elif hasattr(runtime, "logging"):
            logging_cfg = runtime.logging
        else:
            logging_cfg = {}
        
        # Handle FrozenDict or regular dict
        def get_val(obj: Any, key: str, default: Any) -> Any:
            if hasattr(obj, "get"):
                return obj.get(key, default)
            return getattr(obj, key, default)
        
        return cls(
            level=get_val(logging_cfg, "level", DEFAULT_LEVEL),
            format_string=get_val(logging_cfg, "format", DEFAULT_FORMAT),
            date_format=get_val(logging_cfg, "date_format", DEFAULT_DATE_FORMAT),
            console_enabled=get_val(logging_cfg, "console", True),
            file_enabled=get_val(logging_cfg, "file", False),
            file_path=get_val(logging_cfg, "file_path", None),
            propagate=get_val(logging_cfg, "propagate", True),
        )


def setup_logging(
    config: Optional[Any] = None,
    level: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> logging.Logger:
    """
    Configure the root logger for the Portfolio GNN project.
    
    This function should be called once at application startup.
    Subsequent calls will be ignored unless force=True.
    
    Args:
        config: ExperimentConfig containing runtime.logging settings.
            If None, uses sensible defaults.
        level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Takes precedence over config.
        log_file: Override log file path. Takes precedence over config.
            If True, uses auto-generated filename.
        force: If True, reconfigure logging even if already configured.
        
    Returns:
        The configured root logger for the portfolio_gnn package.
    """
    # Get the package root logger
    root_logger = logging.getLogger("portfolio_gnn")
    
    # Check if already configured
    if root_logger.handlers and not force:
        return root_logger
    
    # Clear existing handlers if forcing reconfiguration
    if force:
        root_logger.handlers.clear()
    
    # Build configuration
    if config is not None:
        log_config = LoggingConfig.from_config(config)
    else:
        log_config = LoggingConfig()
    
    # Apply overrides
    if level is not None:
        log_config.level = level.upper()
    
    if log_file is not None:
        log_config.file_enabled = True
        if isinstance(log_file, (str, Path)):
            log_config.file_path = str(log_file)
    
    # Set log level
    numeric_level = getattr(logging, log_config.level, logging.INFO)
    root_logger.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=log_config.format_string,
        datefmt=log_config.date_format,
    )
    
    # Console handler
    if log_config.console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_config.file_enabled:
        file_path = _resolve_log_file_path(log_config.file_path)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Prevent propagation to root logger if we have handlers
    root_logger.propagate = log_config.propagate and not root_logger.handlers
    
    return root_logger


def _resolve_log_file_path(file_path: Optional[str]) -> Path:
    """
    Resolve the log file path, using auto-generated name if not specified.
    
    Args:
        file_path: Specified file path or None
        
    Returns:
        Resolved absolute path to log file
    """
    paths = ProjectPaths()
    
    if file_path is None:
        # Auto-generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"experiment_{timestamp}.log"
    
    # If relative path, place in logs directory
    path = Path(file_path)
    if not path.is_absolute():
        path = paths.logs / path
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    return path


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    This is a convenience function that ensures the logger is under
    the portfolio_gnn namespace.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Logger instance
    """
    # If name already starts with portfolio_gnn, use as-is
    if name.startswith("portfolio_gnn"):
        return logging.getLogger(name)
    
    # Otherwise, prefix with portfolio_gnn
    return logging.getLogger(f"portfolio_gnn.{name}")


class LogContext:
    """
    Context manager for temporarily adjusting log level.
    
    Useful for debugging specific sections of code.
    
    Usage:
        with LogContext(level="DEBUG"):
            # Verbose logging here
            pass
        # Back to normal level
    """
    
    def __init__(self, level: str = "DEBUG", logger_name: str = "portfolio_gnn") -> None:
        self.level = level.upper()
        self.logger_name = logger_name
        self._original_level: Optional[int] = None
    
    def __enter__(self) -> LogContext:
        logger = logging.getLogger(self.logger_name)
        self._original_level = logger.level
        logger.setLevel(getattr(logging, self.level))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._original_level is not None:
            logger = logging.getLogger(self.logger_name)
            logger.setLevel(self._original_level)


def log_experiment_header(
    config: Any,
    seed: int,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Log a standardized experiment header with key metadata.
    
    Args:
        config: ExperimentConfig with config_hash property
        seed: Random seed being used
        logger: Logger to use (defaults to portfolio_gnn logger)
    """
    if logger is None:
        logger = logging.getLogger("portfolio_gnn")
    
    timestamp = datetime.now().isoformat()
    config_hash = getattr(config, "config_hash", "unknown")
    source = getattr(config, "source_path", None)
    source_str = source.name if source else "inline"
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT RUN")
    logger.info("=" * 60)
    logger.info(f"Timestamp:   {timestamp}")
    logger.info(f"Config:      {source_str}")
    logger.info(f"Config Hash: {config_hash}")
    logger.info(f"Seed:        {seed}")
    logger.info("=" * 60)

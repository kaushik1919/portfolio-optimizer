"""
Configuration loading and validation for Portfolio GNN experiments.

This module provides strict configuration management with:
- YAML file loading
- Environment variable overrides
- Required section validation
- Immutable, attribute-style access

Usage:
    from portfolio_gnn.core.config import load_config
    
    config = load_config("experiments/base.yaml")
    seed = config.runtime.seed
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


class FrozenDict:
    """
    An immutable, attribute-accessible dictionary wrapper.
    
    Provides dot-notation access to configuration values while
    preventing modification after initialization.
    """
    
    __slots__ = ("_data", "_hash")
    
    def __init__(self, data: Dict[str, Any]) -> None:
        """
        Initialize with a dictionary, recursively converting nested dicts.
        
        Args:
            data: Dictionary to wrap
        """
        object.__setattr__(self, "_data", self._freeze_recursive(data))
        object.__setattr__(self, "_hash", None)
    
    @staticmethod
    def _freeze_recursive(obj: Any) -> Any:
        """Recursively convert dicts to FrozenDict and lists to tuples."""
        if isinstance(obj, dict):
            # Always use base FrozenDict for nested dicts to avoid validation issues
            frozen = FrozenDict.__new__(FrozenDict)
            object.__setattr__(frozen, "_data", {k: FrozenDict._freeze_recursive(v) for k, v in obj.items()})
            object.__setattr__(frozen, "_hash", None)
            return frozen
        elif isinstance(obj, list):
            return tuple(FrozenDict._freeze_recursive(item) for item in obj)
        return obj
    
    def __getattr__(self, name: str) -> Any:
        """Access configuration values via attribute notation."""
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(
                f"Configuration has no attribute '{name}'. "
                f"Available: {list(self._data.keys())}"
            )
    
    def __getitem__(self, key: str) -> Any:
        """Access configuration values via dictionary notation."""
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(
                f"Configuration has no key '{key}'. "
                f"Available: {list(self._data.keys())}"
            )
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent modification of configuration values."""
        raise AttributeError(
            "Configuration is immutable. "
            "Create a new configuration instead of modifying existing one."
        )
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the configuration."""
        return key in self._data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with an optional default."""
        try:
            return self._data[key]
        except KeyError:
            return default
    
    def keys(self):
        """Return configuration keys."""
        return self._data.keys()
    
    def values(self):
        """Return configuration values."""
        return self._data.values()
    
    def items(self):
        """Return configuration items."""
        return self._data.items()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to a regular dictionary (deep copy)."""
        def unfreeze(obj: Any) -> Any:
            if isinstance(obj, FrozenDict):
                return {k: unfreeze(v) for k, v in obj._data.items()}
            elif isinstance(obj, tuple):
                return [unfreeze(item) for item in obj]
            return obj
        return unfreeze(self)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"FrozenDict({self._data})"
    
    def __eq__(self, other: Any) -> bool:
        """Equality comparison."""
        if isinstance(other, FrozenDict):
            return self._data == other._data
        return False


class ExperimentConfig(FrozenDict):
    """
    Experiment configuration with validation and metadata.
    
    Extends FrozenDict with:
    - Required section validation
    - Configuration hash for reproducibility tracking
    - Source file tracking
    """
    
    __slots__ = ("_source_path",)
    
    REQUIRED_SECTIONS: tuple = ("data", "model", "training", "optimization", "runtime")
    
    def __init__(self, data: Dict[str, Any], source_path: Optional[Path] = None) -> None:
        """
        Initialize and validate experiment configuration.
        
        Args:
            data: Configuration dictionary
            source_path: Path to the source YAML file (for tracking)
            
        Raises:
            ConfigurationError: If required sections are missing
        """
        self._validate_required_sections(data)
        # Directly initialize FrozenDict without triggering recursive validation
        object.__setattr__(self, "_data", self._freeze_recursive(data))
        object.__setattr__(self, "_hash", None)
        object.__setattr__(self, "_source_path", source_path)
    
    @classmethod
    def _validate_required_sections(cls, data: Dict[str, Any]) -> None:
        """
        Validate that all required sections are present.
        
        Args:
            data: Configuration dictionary to validate
            
        Raises:
            ConfigurationError: If any required section is missing
        """
        missing = [section for section in cls.REQUIRED_SECTIONS if section not in data]
        if missing:
            raise ConfigurationError(
                f"Configuration is missing required sections: {missing}. "
                f"Required sections are: {list(cls.REQUIRED_SECTIONS)}. "
                f"Please add these sections to your configuration file."
            )
        
        # Validate sections are dictionaries (not None or scalar)
        for section in cls.REQUIRED_SECTIONS:
            if data[section] is None:
                raise ConfigurationError(
                    f"Configuration section '{section}' is empty (null). "
                    f"Please provide valid configuration values."
                )
            if not isinstance(data[section], dict):
                raise ConfigurationError(
                    f"Configuration section '{section}' must be a dictionary, "
                    f"got {type(data[section]).__name__}."
                )
    
    @property
    def source_path(self) -> Optional[Path]:
        """Path to the source configuration file."""
        return self._source_path
    
    @property
    def config_hash(self) -> str:
        """
        Compute a deterministic hash of the configuration.
        
        Returns:
            SHA-256 hash (first 12 characters) of the configuration
        """
        # Use YAML dump for deterministic serialization
        config_str = yaml.dump(self.to_dict(), sort_keys=True, default_flow_style=False)
        full_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()
        return full_hash[:12]
    
    def __repr__(self) -> str:
        """String representation with metadata."""
        source = self._source_path.name if self._source_path else "inline"
        return f"ExperimentConfig(source={source}, hash={self.config_hash})"


def _expand_env_vars(value: Any) -> Any:
    """
    Recursively expand environment variables in configuration values.
    
    Supports ${VAR} and ${VAR:-default} syntax.
    
    Args:
        value: Configuration value (may be nested)
        
    Returns:
        Value with environment variables expanded
    """
    if isinstance(value, str):
        # Pattern: ${VAR} or ${VAR:-default}
        pattern = r"\$\{([^}:]+)(?::-([^}]*))?\}"
        
        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            # Return original if no env var and no default
            return match.group(0)
        
        return re.sub(pattern, replacer, value)
    
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    
    return value


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to configuration.
    
    Environment variables with prefix PGNN_ override config values.
    Format: PGNN_SECTION_KEY=value (e.g., PGNN_RUNTIME_SEED=42)
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with environment overrides applied
    """
    prefix = "PGNN_"
    
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        
        # Parse key: PGNN_SECTION_SUBSECTION_KEY -> section.subsection.key
        parts = key[len(prefix):].lower().split("_")
        if len(parts) < 2:
            continue
        
        # Navigate to the right location and set value
        section = parts[0]
        if section not in config:
            continue
        
        # Convert value to appropriate type
        typed_value = _parse_env_value(value)
        
        # Handle nested keys
        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            if not isinstance(current[part], dict):
                break
            current = current[part]
        else:
            current[parts[-1]] = typed_value
    
    return config


def _parse_env_value(value: str) -> Union[int, float, bool, str]:
    """
    Parse an environment variable value to the appropriate Python type.
    
    Args:
        value: String value from environment
        
    Returns:
        Parsed value (int, float, bool, or str)
    """
    # Boolean
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False
    
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
    
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    
    # String
    return value


def load_config(
    config_path: Union[str, Path],
    expand_env: bool = True,
    apply_overrides: bool = True
) -> ExperimentConfig:
    """
    Load and validate an experiment configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        expand_env: Whether to expand ${VAR} environment variables in values
        apply_overrides: Whether to apply PGNN_* environment variable overrides
        
    Returns:
        Validated, immutable ExperimentConfig
        
    Raises:
        ConfigurationError: If the file cannot be read or validation fails
        FileNotFoundError: If the configuration file does not exist
    """
    path = Path(config_path).resolve()
    
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}. "
            f"Please ensure the file exists and the path is correct."
        )
    
    if not path.is_file():
        raise ConfigurationError(
            f"Configuration path is not a file: {path}. "
            f"Please provide a path to a YAML file."
        )
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Failed to parse YAML configuration: {e}. "
            f"Please check your YAML syntax."
        ) from e
    
    if raw_config is None:
        raise ConfigurationError(
            f"Configuration file is empty: {path}. "
            f"Please add configuration content."
        )
    
    if not isinstance(raw_config, dict):
        raise ConfigurationError(
            f"Configuration root must be a dictionary, got {type(raw_config).__name__}. "
            f"Please structure your YAML with top-level keys."
        )
    
    # Expand environment variables in values
    if expand_env:
        raw_config = _expand_env_vars(raw_config)
    
    # Apply environment variable overrides
    if apply_overrides:
        raw_config = _apply_env_overrides(raw_config)
    
    return ExperimentConfig(raw_config, source_path=path)


def validate_config_structure(config: ExperimentConfig) -> List[str]:
    """
    Perform extended validation on configuration structure.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of warning messages (empty if fully valid)
    """
    warnings = []
    
    # Check for seed in runtime
    if "seed" not in config.runtime:
        warnings.append("No 'seed' specified in runtime section. Reproducibility may be affected.")
    
    return warnings

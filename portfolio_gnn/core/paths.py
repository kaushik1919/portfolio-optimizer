"""
Centralized path resolution for the Portfolio GNN project.

This module provides robust, cross-platform path resolution without
making assumptions about the current working directory.

All paths are resolved relative to the project root, which is determined
by locating the portfolio_gnn package directory.

Usage:
    from portfolio_gnn.core.paths import ProjectPaths
    
    paths = ProjectPaths()
    data_dir = paths.data
    artifacts_dir = paths.artifacts
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class ProjectPaths:
    """
    Centralized path management for the Portfolio GNN project.
    
    Provides canonical paths for standard project directories and ensures
    directories are created lazily when accessed.
    
    Attributes:
        root: Project root directory
        data: Data storage directory
        artifacts: Model artifacts and outputs directory
        logs: Log files directory
        experiments: Experiment configuration directory
    """
    
    _instance: Optional[ProjectPaths] = None
    
    def __new__(cls) -> ProjectPaths:
        """Singleton pattern to ensure consistent path resolution."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize project paths relative to the project root."""
        if self._initialized:
            return
            
        self._root = self._resolve_project_root()
        self._initialized = True
    
    @staticmethod
    def _resolve_project_root() -> Path:
        """
        Resolve the project root directory robustly.
        
        Strategy:
        1. Look for PROJECT_ROOT environment variable
        2. Traverse up from this file's location to find the project root
           (identified by presence of portfolio_gnn package directory)
        
        Returns:
            Path to the project root directory
            
        Raises:
            RuntimeError: If project root cannot be determined
        """
        # Check for explicit environment variable override
        env_root = os.environ.get("PROJECT_ROOT")
        if env_root:
            root = Path(env_root).resolve()
            if root.is_dir():
                return root
        
        # Traverse up from this file's location
        # This file is at: <project_root>/portfolio_gnn/core/paths.py
        current = Path(__file__).resolve()
        
        # Go up to find project root (parent of portfolio_gnn/)
        for parent in [current] + list(current.parents):
            portfolio_gnn_dir = parent / "portfolio_gnn"
            if portfolio_gnn_dir.is_dir() and (portfolio_gnn_dir / "__init__.py").exists():
                return parent
        
        raise RuntimeError(
            "Could not determine project root. "
            "Ensure the portfolio_gnn package is properly installed, "
            "or set the PROJECT_ROOT environment variable."
        )
    
    @property
    def root(self) -> Path:
        """Project root directory."""
        return self._root
    
    @property
    def data(self) -> Path:
        """Data storage directory. Created lazily if missing."""
        return self._ensure_dir(self._root / "data")
    
    @property
    def artifacts(self) -> Path:
        """Model artifacts and outputs directory. Created lazily if missing."""
        return self._ensure_dir(self._root / "artifacts")
    
    @property
    def logs(self) -> Path:
        """Log files directory. Created lazily if missing."""
        return self._ensure_dir(self._root / "logs")
    
    @property
    def experiments(self) -> Path:
        """Experiment configuration directory. Created lazily if missing."""
        return self._ensure_dir(self._root / "experiments")
    
    @staticmethod
    def _ensure_dir(path: Path) -> Path:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            path: Directory path to ensure exists
            
        Returns:
            The same path, guaranteed to exist as a directory
        """
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_artifact_path(self, *parts: str) -> Path:
        """
        Get a path within the artifacts directory.
        
        Args:
            *parts: Path components relative to artifacts directory
            
        Returns:
            Full path within artifacts directory
        """
        return self._ensure_dir(self.artifacts.joinpath(*parts[:-1])) / parts[-1] if parts else self.artifacts
    
    def get_data_path(self, *parts: str) -> Path:
        """
        Get a path within the data directory.
        
        Args:
            *parts: Path components relative to data directory
            
        Returns:
            Full path within data directory
        """
        return self._ensure_dir(self.data.joinpath(*parts[:-1])) / parts[-1] if parts else self.data
    
    def get_log_path(self, filename: str) -> Path:
        """
        Get a path for a log file.
        
        Args:
            filename: Name of the log file
            
        Returns:
            Full path to the log file
        """
        return self.logs / filename
    
    def __repr__(self) -> str:
        """String representation showing key paths."""
        return (
            f"ProjectPaths(\n"
            f"  root={self._root},\n"
            f"  data={self._root / 'data'},\n"
            f"  artifacts={self._root / 'artifacts'},\n"
            f"  logs={self._root / 'logs'}\n"
            f")"
        )


def get_project_root() -> Path:
    """
    Convenience function to get the project root path.
    
    Returns:
        Path to the project root directory
    """
    return ProjectPaths().root

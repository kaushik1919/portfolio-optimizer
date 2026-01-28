#!/usr/bin/env python
"""
Single experiment entry point for Portfolio GNN.

This script serves as the canonical way to run experiments. It:
1. Loads and validates configuration from a YAML file
2. Initializes logging based on configuration
3. Sets the global random seed for reproducibility
4. Emits a run header with metadata
5. Exits cleanly

This is Phase 1 infrastructure only - no ML, data, graph, or optimization logic.

Usage:
    python scripts/run.py experiments/base.yaml
    python scripts/run.py path/to/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_gnn.core.config import load_config, ConfigurationError, validate_config_structure
from portfolio_gnn.core.logging import setup_logging, log_experiment_header
from portfolio_gnn.core.seeding import set_global_seed
from portfolio_gnn.core.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run a Portfolio GNN experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run.py experiments/base.yaml
    python scripts/run.py path/to/custom_config.yaml

Environment Variables:
    PGNN_RUNTIME_SEED=123     Override the random seed
    PGNN_RUNTIME_LEVEL=DEBUG  Override log level
        """,
    )
    
    parser.add_argument(
        "config",
        type=str,
        help="Path to the experiment configuration YAML file",
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration, do not run",
    )
    
    return parser.parse_args()


def resolve_config_path(config_arg: str) -> Path:
    """
    Resolve the configuration file path.
    
    Handles both absolute and relative paths, checking:
    1. Path as given
    2. Path relative to current directory
    3. Path relative to project root
    4. Path relative to experiments directory
    
    Args:
        config_arg: Config path from command line
        
    Returns:
        Resolved absolute path
        
    Raises:
        FileNotFoundError: If config file cannot be found
    """
    paths = ProjectPaths()
    
    # Try paths in order of specificity
    candidates = [
        Path(config_arg),
        Path.cwd() / config_arg,
        paths.root / config_arg,
        paths.experiments / config_arg,
    ]
    
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved
    
    # Build helpful error message
    tried = "\n  - ".join(str(c.resolve()) for c in candidates)
    raise FileNotFoundError(
        f"Configuration file not found: {config_arg}\n"
        f"Tried:\n  - {tried}\n"
        f"Please provide a valid path to a YAML configuration file."
    )


def main() -> int:
    """
    Main entry point for experiment execution.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    args = parse_args()
    
    # Resolve configuration path
    try:
        config_path = resolve_config_path(args.config)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    
    # Load and validate configuration
    try:
        config = load_config(config_path)
    except ConfigurationError as e:
        print(f"CONFIGURATION ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"UNEXPECTED ERROR loading config: {e}", file=sys.stderr)
        return 1
    
    # Extended validation
    warnings = validate_config_structure(config)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    
    # If validate-only, exit here
    if args.validate_only:
        print(f"Configuration valid: {config_path}")
        print(f"Config hash: {config.config_hash}")
        return 0
    
    # Initialize logging
    try:
        logger = setup_logging(config)
    except Exception as e:
        print(f"ERROR initializing logging: {e}", file=sys.stderr)
        return 1
    
    # Get seed from configuration
    seed = config.runtime.get("seed", 42) if hasattr(config.runtime, "get") else getattr(config.runtime, "seed", 42)
    
    # Set global random seed
    try:
        set_global_seed(seed)
    except ValueError as e:
        logger.error(f"Invalid seed configuration: {e}")
        return 1
    
    # Emit experiment header
    log_experiment_header(config, seed, logger)
    
    # Phase 1: Infrastructure only
    # Future phases will add:
    # - Data loading
    # - Graph construction
    # - Model training
    # - Portfolio optimization
    
    logger.info("Phase 1 scaffold complete. No ML operations in this phase.")
    logger.info("Exiting cleanly.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

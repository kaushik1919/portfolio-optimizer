"""
Deterministic execution guarantees for Portfolio GNN.

This module ensures reproducible results across runs by properly
seeding all random number generators used in the project.

Supports:
- Python random module
- NumPy random
- PyTorch (CPU and CUDA)
- Deterministic PyTorch algorithms (when available)

Usage:
    from portfolio_gnn.core.seeding import set_global_seed
    
    set_global_seed(42)  # All subsequent random operations are reproducible
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np

# PyTorch import with graceful fallback for CPU-only environments
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


logger = logging.getLogger(__name__)


def set_global_seed(seed: int, deterministic_algorithms: bool = True) -> None:
    """
    Set seeds for all random number generators to ensure reproducibility.
    
    This function sets seeds for:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch's random number generators (CPU and CUDA if available)
    
    Additionally, it enables deterministic behavior in PyTorch when possible.
    
    Args:
        seed: Integer seed value. Must be non-negative.
        deterministic_algorithms: Whether to enable PyTorch's deterministic
            algorithms mode. May impact performance. Defaults to True.
            
    Raises:
        ValueError: If seed is negative or not an integer
        
    Note:
        This function logs the seed value for audit purposes.
        Call this function once at the start of your experiment.
    """
    # Validate seed
    if not isinstance(seed, int):
        raise ValueError(
            f"Seed must be an integer, got {type(seed).__name__}. "
            f"Please provide an integer seed value."
        )
    
    if seed < 0:
        raise ValueError(
            f"Seed must be non-negative, got {seed}. "
            f"Please provide a seed >= 0."
        )
    
    # Log the seed being set
    logger.info(f"Setting global random seed: {seed}")
    
    # Python random
    random.seed(seed)
    logger.debug("Python random seeded")
    
    # NumPy
    np.random.seed(seed)
    logger.debug("NumPy random seeded")
    
    # Set PYTHONHASHSEED for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.debug("PYTHONHASHSEED set")
    
    # PyTorch
    if TORCH_AVAILABLE:
        _set_torch_seed(seed, deterministic_algorithms)
    else:
        logger.debug("PyTorch not available, skipping torch seeding")


def _set_torch_seed(seed: int, deterministic_algorithms: bool) -> None:
    """
    Set PyTorch-specific seeds and deterministic behavior.
    
    Args:
        seed: Integer seed value
        deterministic_algorithms: Whether to enable deterministic algorithms
    """
    # CPU seed
    torch.manual_seed(seed)
    logger.debug("PyTorch CPU seeded")
    
    # CUDA seeds (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        logger.debug(f"PyTorch CUDA seeded (devices: {torch.cuda.device_count()})")
    else:
        logger.debug("CUDA not available, skipping CUDA seeding")
    
    # Deterministic algorithms
    if deterministic_algorithms:
        _enable_deterministic_algorithms()


def _enable_deterministic_algorithms() -> None:
    """
    Enable deterministic behavior in PyTorch.
    
    This may impact performance but ensures reproducibility.
    Some operations may raise errors if no deterministic implementation exists.
    """
    # Set cuDNN to deterministic mode
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug("cuDNN set to deterministic mode")
    
    # Use deterministic algorithms globally (PyTorch 1.8+)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            # warn_only=True allows operations without deterministic implementations
            # to proceed with a warning rather than raising an error
            torch.use_deterministic_algorithms(True, warn_only=True)
            logger.debug("PyTorch deterministic algorithms enabled")
        except Exception as e:
            logger.warning(
                f"Could not enable deterministic algorithms: {e}. "
                f"Some operations may not be reproducible."
            )
    elif hasattr(torch, "set_deterministic"):
        # Fallback for older PyTorch versions
        try:
            torch.set_deterministic(True)
            logger.debug("PyTorch deterministic mode enabled (legacy API)")
        except Exception as e:
            logger.warning(f"Could not enable deterministic mode: {e}")


def get_random_state() -> dict:
    """
    Capture the current state of all random number generators.
    
    Useful for checkpointing and debugging reproducibility issues.
    
    Returns:
        Dictionary containing states for all RNGs
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    
    if TORCH_AVAILABLE:
        state["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: dict) -> None:
    """
    Restore the state of all random number generators.
    
    Args:
        state: Dictionary containing RNG states (from get_random_state)
    """
    if "python" in state:
        random.setstate(state["python"])
    
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    
    if TORCH_AVAILABLE:
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])


def verify_reproducibility(seed: int, n_samples: int = 10) -> bool:
    """
    Verify that seeding produces reproducible results.
    
    This function sets the seed twice and verifies that the same
    random numbers are generated each time.
    
    Args:
        seed: Seed value to test
        n_samples: Number of random samples to generate
        
    Returns:
        True if reproducibility is verified, False otherwise
    """
    # First run
    set_global_seed(seed)
    python_vals_1 = [random.random() for _ in range(n_samples)]
    numpy_vals_1 = np.random.rand(n_samples).tolist()
    
    torch_vals_1: Optional[list] = None
    if TORCH_AVAILABLE:
        torch_vals_1 = torch.rand(n_samples).tolist()
    
    # Second run with same seed
    set_global_seed(seed)
    python_vals_2 = [random.random() for _ in range(n_samples)]
    numpy_vals_2 = np.random.rand(n_samples).tolist()
    
    torch_vals_2: Optional[list] = None
    if TORCH_AVAILABLE:
        torch_vals_2 = torch.rand(n_samples).tolist()
    
    # Verify
    python_match = python_vals_1 == python_vals_2
    numpy_match = numpy_vals_1 == numpy_vals_2
    torch_match = (torch_vals_1 == torch_vals_2) if TORCH_AVAILABLE else True
    
    all_match = python_match and numpy_match and torch_match
    
    if not all_match:
        logger.error(
            f"Reproducibility verification failed: "
            f"python={python_match}, numpy={numpy_match}, torch={torch_match}"
        )
    else:
        logger.debug("Reproducibility verification passed")
    
    return all_match

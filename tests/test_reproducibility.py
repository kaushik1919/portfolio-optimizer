"""
Reproducibility tests for Portfolio GNN.

These tests verify that:
1. Same seed produces identical random outputs
2. Different seeds produce different random outputs
3. Reproducibility works on CPU-only machines
4. No external data dependencies

Run with: pytest tests/test_reproducibility.py -v
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_gnn.core.seeding import (
    set_global_seed,
    get_random_state,
    set_random_state,
    verify_reproducibility,
    TORCH_AVAILABLE,
)

if TORCH_AVAILABLE:
    import torch


class TestDeterministicSeeding:
    """Tests for deterministic random number generation."""
    
    def test_python_random_same_seed(self):
        """Same seed produces identical Python random outputs."""
        seed = 12345
        n_samples = 100
        
        # First run
        set_global_seed(seed)
        values_1 = [random.random() for _ in range(n_samples)]
        
        # Second run with same seed
        set_global_seed(seed)
        values_2 = [random.random() for _ in range(n_samples)]
        
        assert values_1 == values_2, "Python random outputs differ with same seed"
    
    def test_numpy_random_same_seed(self):
        """Same seed produces identical NumPy random outputs."""
        seed = 12345
        n_samples = 100
        
        # First run
        set_global_seed(seed)
        values_1 = np.random.rand(n_samples)
        
        # Second run with same seed
        set_global_seed(seed)
        values_2 = np.random.rand(n_samples)
        
        np.testing.assert_array_equal(
            values_1, values_2,
            err_msg="NumPy random outputs differ with same seed"
        )
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_torch_random_same_seed(self):
        """Same seed produces identical PyTorch random outputs."""
        seed = 12345
        n_samples = 100
        
        # First run
        set_global_seed(seed)
        values_1 = torch.rand(n_samples)
        
        # Second run with same seed
        set_global_seed(seed)
        values_2 = torch.rand(n_samples)
        
        assert torch.equal(values_1, values_2), "PyTorch random outputs differ with same seed"
    
    def test_all_generators_same_seed(self):
        """Same seed produces identical outputs across all generators."""
        seed = 42
        
        # First run
        set_global_seed(seed)
        python_1 = [random.random() for _ in range(10)]
        numpy_1 = np.random.rand(10).tolist()
        torch_1 = torch.rand(10).tolist() if TORCH_AVAILABLE else []
        
        # Second run with same seed
        set_global_seed(seed)
        python_2 = [random.random() for _ in range(10)]
        numpy_2 = np.random.rand(10).tolist()
        torch_2 = torch.rand(10).tolist() if TORCH_AVAILABLE else []
        
        assert python_1 == python_2, "Python random not reproducible"
        assert numpy_1 == numpy_2, "NumPy random not reproducible"
        if TORCH_AVAILABLE:
            assert torch_1 == torch_2, "PyTorch random not reproducible"


class TestDifferentSeeds:
    """Tests that different seeds produce different outputs."""
    
    def test_python_random_different_seeds(self):
        """Different seeds produce different Python random outputs."""
        n_samples = 100
        
        set_global_seed(111)
        values_1 = [random.random() for _ in range(n_samples)]
        
        set_global_seed(222)
        values_2 = [random.random() for _ in range(n_samples)]
        
        assert values_1 != values_2, "Different seeds produced identical Python outputs"
    
    def test_numpy_random_different_seeds(self):
        """Different seeds produce different NumPy random outputs."""
        n_samples = 100
        
        set_global_seed(111)
        values_1 = np.random.rand(n_samples)
        
        set_global_seed(222)
        values_2 = np.random.rand(n_samples)
        
        assert not np.array_equal(values_1, values_2), "Different seeds produced identical NumPy outputs"
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_torch_random_different_seeds(self):
        """Different seeds produce different PyTorch random outputs."""
        n_samples = 100
        
        set_global_seed(111)
        values_1 = torch.rand(n_samples)
        
        set_global_seed(222)
        values_2 = torch.rand(n_samples)
        
        assert not torch.equal(values_1, values_2), "Different seeds produced identical PyTorch outputs"


class TestSeedValidation:
    """Tests for seed value validation."""
    
    def test_negative_seed_raises(self):
        """Negative seed values should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            set_global_seed(-1)
    
    def test_non_integer_seed_raises(self):
        """Non-integer seed values should raise ValueError."""
        with pytest.raises(ValueError, match="integer"):
            set_global_seed(3.14)  # type: ignore
        
        with pytest.raises(ValueError, match="integer"):
            set_global_seed("42")  # type: ignore
    
    def test_zero_seed_valid(self):
        """Zero is a valid seed."""
        # Should not raise
        set_global_seed(0)
        value = random.random()
        assert 0.0 <= value <= 1.0
    
    def test_large_seed_valid(self):
        """Large seed values should work."""
        # Should not raise
        set_global_seed(2**31 - 1)
        value = random.random()
        assert 0.0 <= value <= 1.0


class TestRandomStateManagement:
    """Tests for random state capture and restoration."""
    
    def test_capture_and_restore_state(self):
        """Can capture and restore random state."""
        set_global_seed(42)
        
        # Generate some values
        _ = [random.random() for _ in range(50)]
        _ = np.random.rand(50)
        
        # Capture state mid-stream
        state = get_random_state()
        
        # Generate more values
        python_after = [random.random() for _ in range(10)]
        numpy_after = np.random.rand(10).tolist()
        
        # Restore state
        set_random_state(state)
        
        # Should get same values
        python_restored = [random.random() for _ in range(10)]
        numpy_restored = np.random.rand(10).tolist()
        
        assert python_after == python_restored, "Python state restoration failed"
        assert numpy_after == numpy_restored, "NumPy state restoration failed"
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_capture_and_restore_torch_state(self):
        """Can capture and restore PyTorch random state."""
        set_global_seed(42)
        
        # Generate some values
        _ = torch.rand(50)
        
        # Capture state
        state = get_random_state()
        
        # Generate more values
        torch_after = torch.rand(10).tolist()
        
        # Restore state
        set_random_state(state)
        
        # Should get same values
        torch_restored = torch.rand(10).tolist()
        
        assert torch_after == torch_restored, "PyTorch state restoration failed"


class TestVerifyReproducibility:
    """Tests for the verify_reproducibility function."""
    
    def test_verify_reproducibility_passes(self):
        """verify_reproducibility returns True for valid seeds."""
        assert verify_reproducibility(42) is True
        assert verify_reproducibility(0) is True
        assert verify_reproducibility(999999) is True
    
    def test_verify_reproducibility_multiple_seeds(self):
        """verify_reproducibility works for multiple different seeds."""
        seeds = [1, 42, 123, 9999, 2**16]
        for seed in seeds:
            assert verify_reproducibility(seed) is True, f"Reproducibility failed for seed {seed}"


class TestCPUOnlyCompatibility:
    """Tests to ensure everything works on CPU-only machines."""
    
    def test_seeding_works_without_cuda(self):
        """Seeding works even without CUDA."""
        # This should not raise, regardless of CUDA availability
        set_global_seed(42)
        
        # Should be able to generate random numbers
        _ = random.random()
        _ = np.random.rand(10)
        if TORCH_AVAILABLE:
            _ = torch.rand(10)
    
    def test_no_cuda_dependency_in_reproducibility(self):
        """Reproducibility verification works without CUDA."""
        # Should work on CPU-only machines
        result = verify_reproducibility(42)
        assert result is True


class TestSequentialOperations:
    """Tests for sequential random operations."""
    
    def test_sequential_operations_reproducible(self):
        """Sequential operations are reproducible with same seed."""
        seed = 42
        
        # First run: mixed operations
        set_global_seed(seed)
        seq_1 = []
        for i in range(10):
            seq_1.append(random.random())
            seq_1.extend(np.random.rand(3).tolist())
            if TORCH_AVAILABLE:
                seq_1.extend(torch.rand(2).tolist())
        
        # Second run: same operations
        set_global_seed(seed)
        seq_2 = []
        for i in range(10):
            seq_2.append(random.random())
            seq_2.extend(np.random.rand(3).tolist())
            if TORCH_AVAILABLE:
                seq_2.extend(torch.rand(2).tolist())
        
        assert seq_1 == seq_2, "Sequential operations not reproducible"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_sample_reproducible(self):
        """Single random sample is reproducible."""
        set_global_seed(42)
        val_1 = random.random()
        
        set_global_seed(42)
        val_2 = random.random()
        
        assert val_1 == val_2
    
    def test_large_array_reproducible(self):
        """Large arrays are reproducible."""
        set_global_seed(42)
        arr_1 = np.random.rand(10000)
        
        set_global_seed(42)
        arr_2 = np.random.rand(10000)
        
        np.testing.assert_array_equal(arr_1, arr_2)
    
    def test_multiple_reseeding(self):
        """Multiple reseed operations work correctly."""
        values = []
        for seed in [1, 2, 3, 2, 1]:
            set_global_seed(seed)
            values.append(random.random())
        
        # Same seeds should produce same values
        assert values[0] == values[4], "Same seed (1) produced different values"
        assert values[1] == values[3], "Same seed (2) produced different values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

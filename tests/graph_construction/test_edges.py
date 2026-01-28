"""
Tests for Edge Builders.

Tests ensure:
- No look-ahead bias (explicit time checks)
- Correct correlation computation on synthetic data
- Proper edge filtering by threshold
- Deterministic outputs
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_gnn.graph_construction.edges.base import EdgeBuilder, EdgeResult
from portfolio_gnn.graph_construction.edges.correlation import RollingCorrelationEdgeBuilder


def create_synthetic_prices(
    tickers: list,
    num_days: int = 100,
    start_date: str = "2023-01-01",
    seed: int = 42,
    correlation_structure: str = "random",
) -> pd.DataFrame:
    """
    Create synthetic price data for testing.
    
    Args:
        tickers: List of ticker symbols
        num_days: Number of trading days
        start_date: Start date string
        seed: Random seed for reproducibility
        correlation_structure: "random", "high_corr", or "uncorrelated"
        
    Returns:
        DataFrame with prices (columns=tickers, index=dates)
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=num_days, freq="B")
    
    n_assets = len(tickers)
    
    if correlation_structure == "high_corr":
        # Create highly correlated returns
        common_factor = np.random.randn(num_days) * 0.02
        returns = np.column_stack([
            common_factor + np.random.randn(num_days) * 0.005
            for _ in range(n_assets)
        ])
    elif correlation_structure == "uncorrelated":
        # Create uncorrelated returns
        returns = np.random.randn(num_days, n_assets) * 0.02
    else:
        # Random correlation structure
        returns = np.random.randn(num_days, n_assets) * 0.02
    
    # Convert returns to prices
    prices = 100 * np.cumprod(1 + returns, axis=0)
    
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    df.index.name = "Date"
    
    return df


class TestEdgeResult:
    """Tests for EdgeResult dataclass."""
    
    def test_create_valid_result(self):
        """Create valid edge result."""
        result = EdgeResult(
            source_nodes=np.array([0, 1]),
            target_nodes=np.array([1, 0]),
            edge_weights=np.array([0.8, 0.8]),
            edge_type="correlation",
            timestamp=pd.Timestamp("2023-06-01"),
            lookback_window=60,
        )
        
        assert result.num_edges == 2
        assert result.edge_type == "correlation"
    
    def test_edge_index_shape(self):
        """Edge index has correct COO format."""
        result = EdgeResult(
            source_nodes=np.array([0, 1, 2]),
            target_nodes=np.array([1, 2, 0]),
            edge_weights=np.array([0.5, 0.6, 0.7]),
            edge_type="test",
            timestamp=pd.Timestamp("2023-01-01"),
            lookback_window=30,
        )
        
        edge_index = result.edge_index
        
        assert edge_index.shape == (2, 3)
        assert np.array_equal(edge_index[0], [0, 1, 2])
        assert np.array_equal(edge_index[1], [1, 2, 0])
    
    def test_mismatched_arrays_raises(self):
        """Mismatched array lengths raise ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            EdgeResult(
                source_nodes=np.array([0, 1, 2]),
                target_nodes=np.array([1, 2]),  # Wrong length
                edge_weights=np.array([0.5, 0.6, 0.7]),
                edge_type="test",
                timestamp=pd.Timestamp("2023-01-01"),
                lookback_window=30,
            )
    
    def test_self_loop_detection(self):
        """Detect self-loops in edge result."""
        result = EdgeResult(
            source_nodes=np.array([0, 1, 1]),
            target_nodes=np.array([1, 1, 0]),  # (1,1) is self-loop
            edge_weights=np.array([0.5, 0.6, 0.7]),
            edge_type="test",
            timestamp=pd.Timestamp("2023-01-01"),
            lookback_window=30,
        )
        
        assert result.has_self_loops()
        assert result.get_self_loop_count() == 1


class TestRollingCorrelationEdgeBuilder:
    """Tests for RollingCorrelationEdgeBuilder."""
    
    def test_create_builder(self):
        """Create edge builder with valid parameters."""
        builder = RollingCorrelationEdgeBuilder(
            lookback_window=60,
            threshold=0.5,
            directed=False,
        )
        
        assert builder.lookback_window == 60
        assert builder.threshold == 0.5
        assert not builder.directed
        assert builder.edge_type == "correlation"
    
    def test_invalid_lookback_raises(self):
        """Lookback window < 2 raises ValueError."""
        with pytest.raises(ValueError, match="lookback_window must be >= 2"):
            RollingCorrelationEdgeBuilder(lookback_window=1)
    
    def test_invalid_threshold_raises(self):
        """Threshold outside [0,1] raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be in"):
            RollingCorrelationEdgeBuilder(threshold=1.5)
    
    def test_compute_edges_basic(self):
        """Compute edges for a single timestamp."""
        prices = create_synthetic_prices(
            tickers=["A", "B", "C"],
            num_days=100,
            seed=42,
        )
        
        builder = RollingCorrelationEdgeBuilder(
            lookback_window=30,
            threshold=0.3,
        )
        
        timestamp = prices.index[50]  # Mid-point timestamp
        result = builder.compute_edges(
            prices=prices,
            timestamp=timestamp,
            node_order=["A", "B", "C"],
        )
        
        assert isinstance(result, EdgeResult)
        assert result.timestamp == timestamp
        assert result.lookback_window == 30
        assert result.num_edges >= 0
    
    def test_no_self_loops(self):
        """Computed edges have no self-loops."""
        prices = create_synthetic_prices(
            tickers=["A", "B", "C"],
            num_days=100,
            correlation_structure="high_corr",
        )
        
        builder = RollingCorrelationEdgeBuilder(
            lookback_window=30,
            threshold=0.1,  # Low threshold to get edges
        )
        
        result = builder.compute_edges(
            prices=prices,
            timestamp=prices.index[-1],
            node_order=["A", "B", "C"],
        )
        
        assert not result.has_self_loops()
    
    def test_high_threshold_fewer_edges(self):
        """Higher threshold produces fewer edges."""
        prices = create_synthetic_prices(
            tickers=["A", "B", "C", "D"],
            num_days=100,
            seed=42,
        )
        
        timestamp = prices.index[-1]
        node_order = sorted(prices.columns.tolist())
        
        builder_low = RollingCorrelationEdgeBuilder(lookback_window=30, threshold=0.1)
        builder_high = RollingCorrelationEdgeBuilder(lookback_window=30, threshold=0.8)
        
        result_low = builder_low.compute_edges(prices, timestamp, node_order)
        result_high = builder_high.compute_edges(prices, timestamp, node_order)
        
        assert result_low.num_edges >= result_high.num_edges
    
    def test_undirected_edges_symmetric(self):
        """Undirected edges include both directions."""
        prices = create_synthetic_prices(
            tickers=["A", "B"],
            num_days=100,
            correlation_structure="high_corr",  # Ensure we get edges
        )
        
        builder = RollingCorrelationEdgeBuilder(
            lookback_window=30,
            threshold=0.1,
            directed=False,
        )
        
        result = builder.compute_edges(
            prices=prices,
            timestamp=prices.index[-1],
            node_order=["A", "B"],
        )
        
        # For 2 assets with undirected edges, should have 0 or 2 edges
        # (either no edge or both directions)
        if result.num_edges > 0:
            assert result.num_edges == 2  # Both (0,1) and (1,0)
    
    def test_directed_edges_one_direction(self):
        """Directed edges iterate through all ordered pairs (i,j) where i != j."""
        prices = create_synthetic_prices(
            tickers=["A", "B"],
            num_days=100,
            correlation_structure="high_corr",
        )
        
        builder = RollingCorrelationEdgeBuilder(
            lookback_window=30,
            threshold=0.1,
            directed=True,
        )
        
        result = builder.compute_edges(
            prices=prices,
            timestamp=prices.index[-1],
            node_order=["A", "B"],
        )
        
        # For 2 assets with directed=True, iterates all (i,j) where i!=j
        # So should have 2 edges: (0,1) and (1,0) 
        if result.num_edges > 0:
            assert result.num_edges == 2
    
    def test_deterministic_output(self):
        """Same inputs produce identical outputs."""
        prices = create_synthetic_prices(
            tickers=["A", "B", "C"],
            num_days=100,
            seed=42,
        )
        
        builder = RollingCorrelationEdgeBuilder(lookback_window=30, threshold=0.3)
        timestamp = prices.index[-1]
        node_order = ["A", "B", "C"]
        
        result1 = builder.compute_edges(prices, timestamp, node_order)
        result2 = builder.compute_edges(prices, timestamp, node_order)
        
        np.testing.assert_array_equal(result1.source_nodes, result2.source_nodes)
        np.testing.assert_array_equal(result1.target_nodes, result2.target_nodes)
        np.testing.assert_array_equal(result1.edge_weights, result2.edge_weights)
    
    def test_edge_weights_are_correlations(self):
        """Edge weights are actual correlation values."""
        prices = create_synthetic_prices(
            tickers=["A", "B"],
            num_days=100,
            correlation_structure="high_corr",
        )
        
        builder = RollingCorrelationEdgeBuilder(
            lookback_window=30,
            threshold=0.0,  # Include all edges
        )
        
        result = builder.compute_edges(
            prices=prices,
            timestamp=prices.index[-1],
            node_order=["A", "B"],
        )
        
        # All weights should be valid correlations (-1 to 1)
        assert np.all(result.edge_weights >= -1.0)
        assert np.all(result.edge_weights <= 1.0)


class TestNoLookAheadBias:
    """Tests to ensure no look-ahead bias in edge computation."""
    
    def test_only_past_data_used(self):
        """Edge computation uses only data up to timestamp."""
        # Create prices where future data is dramatically different
        prices = create_synthetic_prices(
            tickers=["A", "B"],
            num_days=100,
            seed=42,
        )
        
        # Compute edges at mid-point
        mid_timestamp = prices.index[50]
        
        builder = RollingCorrelationEdgeBuilder(lookback_window=30, threshold=0.0)
        
        # Compute with full data
        result_full = builder.compute_edges(
            prices=prices,
            timestamp=mid_timestamp,
            node_order=["A", "B"],
        )
        
        # Compute with truncated data (only up to mid_timestamp)
        prices_truncated = prices.loc[:mid_timestamp]
        result_truncated = builder.compute_edges(
            prices=prices_truncated,
            timestamp=mid_timestamp,
            node_order=["A", "B"],
        )
        
        # Results should be IDENTICAL
        np.testing.assert_array_equal(
            result_full.source_nodes,
            result_truncated.source_nodes,
        )
        np.testing.assert_array_equal(
            result_full.edge_weights,
            result_truncated.edge_weights,
        )
    
    def test_future_data_modification_no_effect(self):
        """Modifying future data doesn't change current edges."""
        prices = create_synthetic_prices(
            tickers=["A", "B", "C"],
            num_days=100,
            seed=42,
        )
        
        timestamp = prices.index[60]
        builder = RollingCorrelationEdgeBuilder(lookback_window=30, threshold=0.3)
        node_order = ["A", "B", "C"]
        
        # Original computation
        result_original = builder.compute_edges(prices, timestamp, node_order)
        
        # Modify future data (after timestamp)
        prices_modified = prices.copy()
        future_mask = prices_modified.index > timestamp
        prices_modified.loc[future_mask] = prices_modified.loc[future_mask] * 2
        
        # Recompute
        result_modified = builder.compute_edges(prices_modified, timestamp, node_order)
        
        # Results MUST be identical
        np.testing.assert_array_equal(
            result_original.source_nodes,
            result_modified.source_nodes,
        )
        np.testing.assert_array_equal(
            result_original.edge_weights,
            result_modified.edge_weights,
        )
    
    def test_lookback_window_enforced(self):
        """Only lookback_window days are used for correlation."""
        prices = create_synthetic_prices(
            tickers=["A", "B"],
            num_days=200,
            seed=42,
        )
        
        # Make old data (before lookback window) have opposite correlation
        prices_modified = prices.copy()
        prices_modified.iloc[:100, 1] = -prices_modified.iloc[:100, 1]
        
        # Compute at end with 30-day lookback (should ignore old data)
        timestamp = prices.index[-1]
        builder = RollingCorrelationEdgeBuilder(lookback_window=30, threshold=0.0)
        
        result_original = builder.compute_edges(prices, timestamp, ["A", "B"])
        result_modified = builder.compute_edges(prices_modified, timestamp, ["A", "B"])
        
        # Results should be identical (old data outside window)
        np.testing.assert_array_almost_equal(
            result_original.edge_weights,
            result_modified.edge_weights,
            decimal=5,
        )
    
    def test_insufficient_history_raises(self):
        """Insufficient history for lookback raises ValueError."""
        prices = create_synthetic_prices(
            tickers=["A", "B"],
            num_days=20,  # Only 20 days
        )
        
        builder = RollingCorrelationEdgeBuilder(
            lookback_window=60,  # Need 60 days
        )
        
        with pytest.raises(ValueError, match="Insufficient history"):
            builder.compute_edges(
                prices=prices,
                timestamp=prices.index[-1],
                node_order=["A", "B"],
            )


class TestEdgeBuilderAbstraction:
    """Tests for EdgeBuilder base class."""
    
    def test_validate_no_lookahead(self):
        """validate_no_lookahead filters future data."""
        prices = create_synthetic_prices(
            tickers=["A", "B"],
            num_days=100,
        )
        
        builder = RollingCorrelationEdgeBuilder(lookback_window=30)
        
        cutoff = prices.index[50]
        filtered = builder.validate_no_lookahead(prices, cutoff)
        
        # Filtered should have only rows <= cutoff
        assert len(filtered) == 51  # indices 0-50 inclusive
        assert filtered.index.max() <= cutoff
    
    def test_get_lookback_slice(self):
        """get_lookback_slice returns correct window."""
        prices = create_synthetic_prices(
            tickers=["A", "B"],
            num_days=100,
        )
        
        builder = RollingCorrelationEdgeBuilder(lookback_window=30)
        
        timestamp = prices.index[-1]
        lookback = builder.get_lookback_slice(prices, timestamp)
        
        assert len(lookback) == 30
        assert lookback.index[-1] == timestamp

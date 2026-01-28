"""
Tests for Graph Builder.

Tests ensure:
- Deterministic graph construction
- Correct snapshot sequence generation
- Node index stability across time
- Proper handling of lookback requirements
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_gnn.graph_construction.graph_builder import (
    GraphBuilder,
    GraphSnapshot,
    build_graph_sequence,
)
from portfolio_gnn.graph_construction.nodes import NodeRegistry
from portfolio_gnn.graph_construction.edges.correlation import RollingCorrelationEdgeBuilder


def create_synthetic_prices(
    tickers: list,
    num_days: int = 100,
    start_date: str = "2023-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic price data for testing."""
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=num_days, freq="B")
    
    n_assets = len(tickers)
    returns = np.random.randn(num_days, n_assets) * 0.02
    prices = 100 * np.cumprod(1 + returns, axis=0)
    
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    df.index.name = "Date"
    
    return df


class TestGraphSnapshot:
    """Tests for GraphSnapshot class."""
    
    def test_create_valid_snapshot(self):
        """Create a valid graph snapshot."""
        registry = NodeRegistry.from_tickers(["A", "B", "C"])
        
        snapshot = GraphSnapshot(
            timestamp=pd.Timestamp("2023-06-01"),
            num_nodes=3,
            edge_index=np.array([[0, 1], [1, 2]]),
            edge_attr=np.array([0.5, 0.6]),
            edge_types=["correlation"],
            node_registry=registry,
            lookback_window=30,
        )
        
        assert snapshot.num_nodes == 3
        assert snapshot.num_edges == 2
        assert snapshot.tickers == ["A", "B", "C"]
    
    def test_invalid_edge_index_shape_raises(self):
        """Invalid edge_index shape raises ValueError."""
        registry = NodeRegistry.from_tickers(["A", "B"])
        
        with pytest.raises(ValueError, match="edge_index must have shape"):
            GraphSnapshot(
                timestamp=pd.Timestamp("2023-06-01"),
                num_nodes=2,
                edge_index=np.array([0, 1, 2]),  # Wrong shape
                edge_attr=np.array([0.5]),
                edge_types=["test"],
                node_registry=registry,
                lookback_window=30,
            )
    
    def test_mismatched_edge_counts_raises(self):
        """Mismatched edge_attr length raises ValueError."""
        registry = NodeRegistry.from_tickers(["A", "B"])
        
        with pytest.raises(ValueError, match="edge_attr length"):
            GraphSnapshot(
                timestamp=pd.Timestamp("2023-06-01"),
                num_nodes=2,
                edge_index=np.array([[0], [1]]),
                edge_attr=np.array([0.5, 0.6]),  # Wrong length
                edge_types=["test"],
                node_registry=registry,
                lookback_window=30,
            )
    
    def test_to_dict(self):
        """Convert snapshot to dictionary."""
        registry = NodeRegistry.from_tickers(["A", "B"])
        
        snapshot = GraphSnapshot(
            timestamp=pd.Timestamp("2023-06-01"),
            num_nodes=2,
            edge_index=np.array([[0], [1]]),
            edge_attr=np.array([0.5]),
            edge_types=["correlation"],
            node_registry=registry,
            lookback_window=30,
        )
        
        d = snapshot.to_dict()
        
        assert d["num_nodes"] == 2
        assert d["num_edges"] == 1
        assert d["lookback_window"] == 30


class TestGraphBuilder:
    """Tests for GraphBuilder class."""
    
    def test_create_builder(self):
        """Create graph builder with edge builders."""
        edge_builder = RollingCorrelationEdgeBuilder(lookback_window=30)
        builder = GraphBuilder(edge_builders=[edge_builder])
        
        assert builder.max_lookback == 30
        assert len(builder.edge_builders) == 1
    
    def test_empty_edge_builders_raises(self):
        """Empty edge builders list raises ValueError."""
        with pytest.raises(ValueError, match="At least one edge builder"):
            GraphBuilder(edge_builders=[])
    
    def test_build_sequence(self):
        """Build sequence of graph snapshots."""
        prices = create_synthetic_prices(
            tickers=["A", "B", "C"],
            num_days=100,
        )
        
        edge_builder = RollingCorrelationEdgeBuilder(
            lookback_window=20,
            threshold=0.0,  # Include all edges
        )
        builder = GraphBuilder(edge_builders=[edge_builder])
        
        snapshots = builder.build_sequence(prices)
        
        # Should have (100 - 20) = 80 valid timestamps
        assert len(snapshots) == 80
        assert all(isinstance(s, GraphSnapshot) for s in snapshots)
    
    def test_snapshots_in_chronological_order(self):
        """Snapshots are returned in chronological order."""
        prices = create_synthetic_prices(
            tickers=["A", "B"],
            num_days=50,
        )
        
        edge_builder = RollingCorrelationEdgeBuilder(lookback_window=10)
        builder = GraphBuilder(edge_builders=[edge_builder])
        
        snapshots = builder.build_sequence(prices)
        
        timestamps = [s.timestamp for s in snapshots]
        assert timestamps == sorted(timestamps)
    
    def test_node_count_consistent(self):
        """All snapshots have same number of nodes."""
        prices = create_synthetic_prices(
            tickers=["A", "B", "C", "D"],
            num_days=50,
        )
        
        edge_builder = RollingCorrelationEdgeBuilder(lookback_window=10)
        builder = GraphBuilder(edge_builders=[edge_builder])
        
        snapshots = builder.build_sequence(prices)
        
        num_nodes = [s.num_nodes for s in snapshots]
        assert len(set(num_nodes)) == 1
        assert num_nodes[0] == 4
    
    def test_node_order_stable(self):
        """Node order is stable across all snapshots."""
        prices = create_synthetic_prices(
            tickers=["D", "A", "C", "B"],  # Unsorted
            num_days=50,
        )
        
        edge_builder = RollingCorrelationEdgeBuilder(lookback_window=10)
        builder = GraphBuilder(edge_builders=[edge_builder])
        
        snapshots = builder.build_sequence(prices)
        
        # All snapshots should have same (sorted) ticker order
        expected_order = ["A", "B", "C", "D"]
        for snapshot in snapshots:
            assert snapshot.tickers == expected_order
    
    def test_insufficient_history_raises(self):
        """Insufficient data for lookback raises ValueError."""
        prices = create_synthetic_prices(
            tickers=["A", "B"],
            num_days=20,
        )
        
        edge_builder = RollingCorrelationEdgeBuilder(lookback_window=50)
        builder = GraphBuilder(edge_builders=[edge_builder])
        
        with pytest.raises(ValueError, match="Insufficient history"):
            builder.build_sequence(prices)
    
    def test_empty_prices_raises(self):
        """Empty price DataFrame raises ValueError."""
        prices = pd.DataFrame()
        
        edge_builder = RollingCorrelationEdgeBuilder(lookback_window=10)
        builder = GraphBuilder(edge_builders=[edge_builder])
        
        with pytest.raises(ValueError, match="cannot be empty"):
            builder.build_sequence(prices)
    
    def test_custom_node_registry(self):
        """Use custom node registry."""
        tickers = ["X", "Y", "Z"]
        prices = create_synthetic_prices(tickers=tickers, num_days=50)
        
        # Create custom registry
        registry = NodeRegistry.from_tickers(
            tickers,
            sectors={"X": "Tech", "Y": "Finance", "Z": "Healthcare"},
        )
        
        edge_builder = RollingCorrelationEdgeBuilder(lookback_window=10)
        builder = GraphBuilder(edge_builders=[edge_builder])
        
        snapshots = builder.build_sequence(prices, node_registry=registry)
        
        # Check sector metadata is preserved
        for snapshot in snapshots:
            assert snapshot.node_registry.get_node("X").sector == "Tech"


class TestDeterministicConstruction:
    """Tests for deterministic graph construction."""
    
    def test_same_input_same_output(self):
        """Same inputs produce identical graph sequences."""
        prices = create_synthetic_prices(
            tickers=["A", "B", "C"],
            num_days=60,
            seed=42,
        )
        
        edge_builder1 = RollingCorrelationEdgeBuilder(lookback_window=20, threshold=0.3)
        edge_builder2 = RollingCorrelationEdgeBuilder(lookback_window=20, threshold=0.3)
        
        builder1 = GraphBuilder(edge_builders=[edge_builder1])
        builder2 = GraphBuilder(edge_builders=[edge_builder2])
        
        snapshots1 = builder1.build_sequence(prices.copy())
        snapshots2 = builder2.build_sequence(prices.copy())
        
        assert len(snapshots1) == len(snapshots2)
        
        for s1, s2 in zip(snapshots1, snapshots2):
            assert s1.timestamp == s2.timestamp
            assert s1.num_nodes == s2.num_nodes
            assert s1.num_edges == s2.num_edges
            np.testing.assert_array_equal(s1.edge_index, s2.edge_index)
            np.testing.assert_array_equal(s1.edge_attr, s2.edge_attr)
    
    def test_column_order_invariant(self):
        """Results are invariant to input column order."""
        tickers = ["C", "A", "B"]
        
        prices1 = create_synthetic_prices(tickers=tickers, num_days=50, seed=42)
        prices2 = prices1[["A", "B", "C"]]  # Reorder columns
        prices3 = prices1[["B", "C", "A"]]  # Different order
        
        edge_builder = RollingCorrelationEdgeBuilder(lookback_window=10, threshold=0.0)
        builder = GraphBuilder(edge_builders=[edge_builder])
        
        snapshots1 = builder.build_sequence(prices1)
        snapshots2 = builder.build_sequence(prices2)
        snapshots3 = builder.build_sequence(prices3)
        
        # All should produce same sorted ticker order
        for s1, s2, s3 in zip(snapshots1, snapshots2, snapshots3):
            assert s1.tickers == s2.tickers == s3.tickers == ["A", "B", "C"]


class TestTimeAwareConstruction:
    """Tests for time-aware graph construction (no lookahead)."""
    
    def test_each_snapshot_uses_only_past_data(self):
        """Each snapshot is computed using only past data."""
        prices = create_synthetic_prices(
            tickers=["A", "B"],
            num_days=100,
            seed=42,
        )
        
        lookback = 20
        edge_builder = RollingCorrelationEdgeBuilder(
            lookback_window=lookback,
            threshold=0.0,
        )
        builder = GraphBuilder(edge_builders=[edge_builder])
        
        snapshots = builder.build_sequence(prices)
        
        # Each snapshot's timestamp should be the last date in its lookback
        for snapshot in snapshots:
            # The timestamp should exist in the original prices
            assert snapshot.timestamp in prices.index
            
            # The lookback window should end at the snapshot timestamp
            assert snapshot.lookback_window == lookback
    
    def test_future_modification_no_effect(self):
        """Modifying future data doesn't affect past snapshots."""
        prices = create_synthetic_prices(
            tickers=["A", "B"],
            num_days=100,
            seed=42,
        )
        
        edge_builder = RollingCorrelationEdgeBuilder(lookback_window=20)
        builder = GraphBuilder(edge_builders=[edge_builder])
        
        # Build with original data
        snapshots_original = builder.build_sequence(prices)
        
        # Modify future data
        prices_modified = prices.copy()
        prices_modified.iloc[60:] = prices_modified.iloc[60:] * 10
        
        snapshots_modified = builder.build_sequence(prices_modified)
        
        # Snapshots before timestamp 60 should be identical
        cutoff_timestamp = prices.index[59]
        
        for s_orig, s_mod in zip(snapshots_original, snapshots_modified):
            if s_orig.timestamp <= cutoff_timestamp:
                np.testing.assert_array_equal(
                    s_orig.edge_index, s_mod.edge_index
                )
                np.testing.assert_array_almost_equal(
                    s_orig.edge_attr, s_mod.edge_attr, decimal=5
                )
    
    def test_skip_insufficient_lookback_timestamps(self):
        """Timestamps without sufficient lookback are skipped."""
        prices = create_synthetic_prices(
            tickers=["A", "B"],
            num_days=50,
        )
        
        lookback = 20
        edge_builder = RollingCorrelationEdgeBuilder(lookback_window=lookback)
        builder = GraphBuilder(edge_builders=[edge_builder])
        
        snapshots = builder.build_sequence(prices)
        
        # First snapshot should be at index 20 (0-indexed)
        first_valid_timestamp = prices.index[lookback]
        assert snapshots[0].timestamp == first_valid_timestamp


class TestBuildGraphSequenceFunction:
    """Tests for build_graph_sequence convenience function."""
    
    def test_with_mock_config(self):
        """Build sequence using config object."""
        # Create mock processed data
        class MockProcessedData:
            def __init__(self):
                self.prices = create_synthetic_prices(
                    tickers=["A", "B", "C"],
                    num_days=100,
                )
                self.tickers = list(self.prices.columns)
        
        # Create mock config
        class MockGraphConfig:
            lookback_window = 30
            correlation_threshold = 0.3
            directed = False
            min_edges_per_node = 0
        
        class MockConfig:
            graph = MockGraphConfig()
        
        processed_data = MockProcessedData()
        config = MockConfig()
        
        snapshots = build_graph_sequence(processed_data, config)
        
        assert len(snapshots) > 0
        assert all(isinstance(s, GraphSnapshot) for s in snapshots)

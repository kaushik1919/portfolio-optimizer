"""
Tests for Graph Validation.

Tests ensure:
- Hard failures on validation errors (no warnings)
- Correct detection of node inconsistencies
- Correct detection of edge integrity issues
- Actionable error messages
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

from portfolio_gnn.graph_construction.validation import (
    validate_graph_sequence,
    compute_graph_statistics,
    GraphValidationError,
    NodeConsistencyError,
    EdgeIntegrityError,
    EmptyGraphError,
)
from portfolio_gnn.graph_construction.graph_builder import GraphSnapshot
from portfolio_gnn.graph_construction.nodes import NodeRegistry


def create_valid_snapshot(
    timestamp: pd.Timestamp,
    num_nodes: int = 3,
    num_edges: int = 4,
) -> GraphSnapshot:
    """Create a valid graph snapshot for testing."""
    registry = NodeRegistry.from_tickers([f"T{i}" for i in range(num_nodes)])
    
    # Create valid edges (no self-loops)
    sources = []
    targets = []
    weights = []
    
    for i in range(min(num_edges, num_nodes * (num_nodes - 1))):
        src = i % num_nodes
        tgt = (i + 1) % num_nodes
        if src != tgt:
            sources.append(src)
            targets.append(tgt)
            weights.append(0.5 + i * 0.1)
    
    edge_index = np.array([sources, targets], dtype=np.int64)
    edge_attr = np.array(weights, dtype=np.float32)
    
    return GraphSnapshot(
        timestamp=timestamp,
        num_nodes=num_nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_types=["correlation"],
        node_registry=registry,
        lookback_window=30,
    )


class TestValidateGraphSequence:
    """Tests for validate_graph_sequence function."""
    
    def test_valid_sequence_passes(self):
        """Valid sequence passes validation."""
        snapshots = [
            create_valid_snapshot(pd.Timestamp("2023-01-01")),
            create_valid_snapshot(pd.Timestamp("2023-01-02")),
            create_valid_snapshot(pd.Timestamp("2023-01-03")),
        ]
        
        # Should not raise
        validate_graph_sequence(snapshots)
    
    def test_empty_sequence_raises(self):
        """Empty sequence raises GraphValidationError."""
        with pytest.raises(GraphValidationError, match="cannot be empty"):
            validate_graph_sequence([])
    
    def test_single_snapshot_valid(self):
        """Single snapshot sequence is valid."""
        snapshots = [create_valid_snapshot(pd.Timestamp("2023-01-01"))]
        
        # Should not raise
        validate_graph_sequence(snapshots)


class TestNodeConsistency:
    """Tests for node consistency validation."""
    
    def test_inconsistent_node_count_raises(self):
        """Inconsistent node count across snapshots raises error."""
        registry_3 = NodeRegistry.from_tickers(["A", "B", "C"])
        registry_4 = NodeRegistry.from_tickers(["A", "B", "C", "D"])
        
        snapshot1 = GraphSnapshot(
            timestamp=pd.Timestamp("2023-01-01"),
            num_nodes=3,
            edge_index=np.array([[0], [1]]),
            edge_attr=np.array([0.5]),
            edge_types=["test"],
            node_registry=registry_3,
            lookback_window=30,
        )
        
        snapshot2 = GraphSnapshot(
            timestamp=pd.Timestamp("2023-01-02"),
            num_nodes=4,  # Different!
            edge_index=np.array([[0], [1]]),
            edge_attr=np.array([0.5]),
            edge_types=["test"],
            node_registry=registry_4,
            lookback_window=30,
        )
        
        with pytest.raises(NodeConsistencyError, match="Node count mismatch"):
            validate_graph_sequence([snapshot1, snapshot2])
    
    def test_inconsistent_ticker_order_raises(self):
        """Inconsistent ticker order across snapshots raises error."""
        registry1 = NodeRegistry.from_tickers(["A", "B", "C"])
        registry2 = NodeRegistry.from_tickers(["A", "B", "D"])  # D instead of C
        
        snapshot1 = GraphSnapshot(
            timestamp=pd.Timestamp("2023-01-01"),
            num_nodes=3,
            edge_index=np.array([[0], [1]]),
            edge_attr=np.array([0.5]),
            edge_types=["test"],
            node_registry=registry1,
            lookback_window=30,
        )
        
        snapshot2 = GraphSnapshot(
            timestamp=pd.Timestamp("2023-01-02"),
            num_nodes=3,
            edge_index=np.array([[0], [1]]),
            edge_attr=np.array([0.5]),
            edge_types=["test"],
            node_registry=registry2,  # Different tickers!
            lookback_window=30,
        )
        
        with pytest.raises(NodeConsistencyError, match="Ticker order mismatch"):
            validate_graph_sequence([snapshot1, snapshot2])


class TestEdgeIntegrity:
    """Tests for edge integrity validation."""
    
    def test_invalid_source_node_raises(self):
        """Edge with invalid source node ID raises error."""
        registry = NodeRegistry.from_tickers(["A", "B"])
        
        snapshot = GraphSnapshot(
            timestamp=pd.Timestamp("2023-01-01"),
            num_nodes=2,
            edge_index=np.array([[5], [1]]),  # Source 5 is invalid
            edge_attr=np.array([0.5]),
            edge_types=["test"],
            node_registry=registry,
            lookback_window=30,
        )
        
        with pytest.raises(EdgeIntegrityError, match="Invalid source node"):
            validate_graph_sequence([snapshot])
    
    def test_invalid_target_node_raises(self):
        """Edge with invalid target node ID raises error."""
        registry = NodeRegistry.from_tickers(["A", "B"])
        
        snapshot = GraphSnapshot(
            timestamp=pd.Timestamp("2023-01-01"),
            num_nodes=2,
            edge_index=np.array([[0], [10]]),  # Target 10 is invalid
            edge_attr=np.array([0.5]),
            edge_types=["test"],
            node_registry=registry,
            lookback_window=30,
        )
        
        with pytest.raises(EdgeIntegrityError, match="Invalid target node"):
            validate_graph_sequence([snapshot])
    
    def test_negative_node_id_raises(self):
        """Edge with negative node ID raises error."""
        registry = NodeRegistry.from_tickers(["A", "B"])
        
        snapshot = GraphSnapshot(
            timestamp=pd.Timestamp("2023-01-01"),
            num_nodes=2,
            edge_index=np.array([[-1], [1]]),  # Negative source
            edge_attr=np.array([0.5]),
            edge_types=["test"],
            node_registry=registry,
            lookback_window=30,
        )
        
        with pytest.raises(EdgeIntegrityError, match="Invalid source node"):
            validate_graph_sequence([snapshot])
    
    def test_self_loop_raises(self):
        """Self-loop edges raise error."""
        registry = NodeRegistry.from_tickers(["A", "B"])
        
        snapshot = GraphSnapshot(
            timestamp=pd.Timestamp("2023-01-01"),
            num_nodes=2,
            edge_index=np.array([[0, 1], [0, 0]]),  # (0,0) is self-loop
            edge_attr=np.array([0.5, 0.6]),
            edge_types=["test"],
            node_registry=registry,
            lookback_window=30,
        )
        
        with pytest.raises(EdgeIntegrityError, match="Self-loops detected"):
            validate_graph_sequence([snapshot])
    
    def test_nan_edge_weight_raises(self):
        """NaN in edge attributes raises error."""
        registry = NodeRegistry.from_tickers(["A", "B"])
        
        snapshot = GraphSnapshot(
            timestamp=pd.Timestamp("2023-01-01"),
            num_nodes=2,
            edge_index=np.array([[0], [1]]),
            edge_attr=np.array([np.nan]),  # NaN weight
            edge_types=["test"],
            node_registry=registry,
            lookback_window=30,
        )
        
        with pytest.raises(EdgeIntegrityError, match="NaN values"):
            validate_graph_sequence([snapshot])
    
    def test_inf_edge_weight_raises(self):
        """Infinity in edge attributes raises error."""
        registry = NodeRegistry.from_tickers(["A", "B"])
        
        snapshot = GraphSnapshot(
            timestamp=pd.Timestamp("2023-01-01"),
            num_nodes=2,
            edge_index=np.array([[0], [1]]),
            edge_attr=np.array([np.inf]),  # Infinity weight
            edge_types=["test"],
            node_registry=registry,
            lookback_window=30,
        )
        
        with pytest.raises(EdgeIntegrityError, match="Infinity values"):
            validate_graph_sequence([snapshot])


class TestEmptyGraphDetection:
    """Tests for empty graph detection."""
    
    def test_empty_graph_raises(self):
        """Graph with no edges raises error."""
        registry = NodeRegistry.from_tickers(["A", "B"])
        
        snapshot = GraphSnapshot(
            timestamp=pd.Timestamp("2023-01-01"),
            num_nodes=2,
            edge_index=np.array([[], []]).reshape(2, 0),  # No edges
            edge_attr=np.array([]),
            edge_types=["test"],
            node_registry=registry,
            lookback_window=30,
        )
        
        with pytest.raises(EmptyGraphError, match="Too few edges"):
            validate_graph_sequence([snapshot], min_edges_per_graph=1)
    
    def test_min_edges_configurable(self):
        """Minimum edges threshold is configurable."""
        registry = NodeRegistry.from_tickers(["A", "B", "C"])
        
        snapshot = GraphSnapshot(
            timestamp=pd.Timestamp("2023-01-01"),
            num_nodes=3,
            edge_index=np.array([[0, 1], [1, 2]]),  # 2 edges
            edge_attr=np.array([0.5, 0.6]),
            edge_types=["test"],
            node_registry=registry,
            lookback_window=30,
        )
        
        # Should pass with min_edges_per_graph=2
        validate_graph_sequence([snapshot], min_edges_per_graph=2)
        
        # Should fail with min_edges_per_graph=5
        with pytest.raises(EmptyGraphError):
            validate_graph_sequence([snapshot], min_edges_per_graph=5)


class TestTemporalConsistency:
    """Tests for temporal consistency validation."""
    
    def test_non_monotonic_timestamps_raises(self):
        """Non-monotonically increasing timestamps raise error."""
        snapshots = [
            create_valid_snapshot(pd.Timestamp("2023-01-02")),
            create_valid_snapshot(pd.Timestamp("2023-01-01")),  # Out of order!
        ]
        
        with pytest.raises(GraphValidationError, match="not monotonically increasing"):
            validate_graph_sequence(snapshots)
    
    def test_duplicate_timestamps_raises(self):
        """Duplicate timestamps raise error."""
        snapshots = [
            create_valid_snapshot(pd.Timestamp("2023-01-01")),
            create_valid_snapshot(pd.Timestamp("2023-01-01")),  # Duplicate!
        ]
        
        with pytest.raises(GraphValidationError, match="not monotonically increasing"):
            validate_graph_sequence(snapshots)


class TestActionableErrors:
    """Tests for actionable error messages."""
    
    def test_node_count_error_shows_expected_and_actual(self):
        """Node count error shows expected and actual values."""
        registry_3 = NodeRegistry.from_tickers(["A", "B", "C"])
        registry_4 = NodeRegistry.from_tickers(["A", "B", "C", "D"])
        
        snapshot1 = GraphSnapshot(
            timestamp=pd.Timestamp("2023-01-01"),
            num_nodes=3,
            edge_index=np.array([[0], [1]]),
            edge_attr=np.array([0.5]),
            edge_types=["test"],
            node_registry=registry_3,
            lookback_window=30,
        )
        
        snapshot2 = GraphSnapshot(
            timestamp=pd.Timestamp("2023-01-02"),
            num_nodes=4,
            edge_index=np.array([[0], [1]]),
            edge_attr=np.array([0.5]),
            edge_types=["test"],
            node_registry=registry_4,
            lookback_window=30,
        )
        
        with pytest.raises(NodeConsistencyError) as exc_info:
            validate_graph_sequence([snapshot1, snapshot2])
        
        error_msg = str(exc_info.value)
        assert "Expected 3" in error_msg
        assert "got 4" in error_msg
    
    def test_edge_error_shows_timestamp(self):
        """Edge error shows the offending timestamp."""
        registry = NodeRegistry.from_tickers(["A", "B"])
        
        snapshot = GraphSnapshot(
            timestamp=pd.Timestamp("2023-06-15"),
            num_nodes=2,
            edge_index=np.array([[99], [1]]),  # Invalid source
            edge_attr=np.array([0.5]),
            edge_types=["test"],
            node_registry=registry,
            lookback_window=30,
        )
        
        with pytest.raises(EdgeIntegrityError) as exc_info:
            validate_graph_sequence([snapshot])
        
        error_msg = str(exc_info.value)
        assert "2023-06-15" in error_msg


class TestComputeGraphStatistics:
    """Tests for compute_graph_statistics function."""
    
    def test_compute_stats(self):
        """Compute statistics for graph sequence."""
        snapshots = [
            create_valid_snapshot(pd.Timestamp("2023-01-01"), num_edges=4),
            create_valid_snapshot(pd.Timestamp("2023-01-02"), num_edges=6),
            create_valid_snapshot(pd.Timestamp("2023-01-03"), num_edges=2),
        ]
        
        stats = compute_graph_statistics(snapshots)
        
        assert stats["num_snapshots"] == 3
        assert stats["num_nodes"] == 3
        # Check that min/max/mean are computed (actual values depend on create_valid_snapshot)
        assert "min_edges" in stats
        assert "max_edges" in stats
        assert "mean_edges" in stats
        assert "first_timestamp" in stats
        assert "last_timestamp" in stats
    
    def test_empty_sequence_stats(self):
        """Empty sequence returns empty dict."""
        stats = compute_graph_statistics([])
        
        assert stats == {}


class TestNoWarningsOnlyFailures:
    """Tests to ensure no warnings, only hard failures."""
    
    def test_invalid_data_raises_exception(self):
        """Invalid data raises exception, not warning."""
        registry = NodeRegistry.from_tickers(["A", "B"])
        
        # Create snapshot with self-loop
        snapshot = GraphSnapshot(
            timestamp=pd.Timestamp("2023-01-01"),
            num_nodes=2,
            edge_index=np.array([[0], [0]]),  # Self-loop
            edge_attr=np.array([0.5]),
            edge_types=["test"],
            node_registry=registry,
            lookback_window=30,
        )
        
        # Must raise, not warn
        with pytest.raises(EdgeIntegrityError):
            validate_graph_sequence([snapshot])

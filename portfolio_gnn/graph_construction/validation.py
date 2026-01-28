"""
Graph Validation for Portfolio GNN.

Implements hard validation checks for graph sequences.

HARD FAILURES (no warnings, no silent fixes):
- Node count inconsistent across snapshots
- Node index inconsistency across time
- Edges referencing invalid node IDs
- NaNs or infinities in edge attributes
- Empty graphs after edge filtering

All failures raise exceptions with actionable error messages.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from portfolio_gnn.graph_construction.graph_builder import GraphSnapshot

logger = logging.getLogger(__name__)


class GraphValidationError(Exception):
    """Base exception for graph validation failures."""
    pass


class NodeConsistencyError(GraphValidationError):
    """Raised when node structure is inconsistent across snapshots."""
    pass


class EdgeIntegrityError(GraphValidationError):
    """Raised when edges reference invalid nodes or have invalid attributes."""
    pass


class EmptyGraphError(GraphValidationError):
    """Raised when a graph has no edges after filtering."""
    pass


def validate_graph_sequence(
    snapshots: List[GraphSnapshot],
    config: Any = None,
    min_edges_per_graph: int = 1,
) -> None:
    """
    Validate a sequence of graph snapshots.
    
    This function performs comprehensive validation checks and raises
    hard exceptions on any failure. NO warnings, NO silent fixes.
    
    Checks performed:
    1. Node count consistent across all snapshots
    2. Node registry consistent across time
    3. No edges referencing invalid node IDs
    4. No NaNs or infinities in edge attributes
    5. No empty graphs after edge filtering
    6. Timestamps are monotonically increasing
    
    Args:
        snapshots: List of GraphSnapshot objects
        config: Optional configuration for additional validation
        min_edges_per_graph: Minimum edges required per graph
        
    Raises:
        GraphValidationError: If any validation check fails
    """
    logger.info("=" * 60)
    logger.info("GRAPH SEQUENCE VALIDATION")
    logger.info("=" * 60)
    
    if not snapshots:
        raise GraphValidationError("Graph sequence cannot be empty")
    
    logger.info(f"Validating {len(snapshots)} graph snapshots...")
    
    # Extract reference values from first snapshot
    reference_num_nodes = snapshots[0].num_nodes
    reference_tickers = snapshots[0].tickers
    
    logger.info(f"Reference node count: {reference_num_nodes}")
    logger.info(f"Reference tickers: {reference_tickers[:5]}{'...' if len(reference_tickers) > 5 else ''}")
    
    # Validate each snapshot
    for i, snapshot in enumerate(snapshots):
        _validate_single_snapshot(
            snapshot=snapshot,
            index=i,
            reference_num_nodes=reference_num_nodes,
            reference_tickers=reference_tickers,
            min_edges_per_graph=min_edges_per_graph,
        )
    
    # Validate temporal consistency
    _validate_temporal_consistency(snapshots)
    
    logger.info("-" * 60)
    logger.info("✓ Node count consistent across all snapshots")
    logger.info("✓ Node registry consistent across time")
    logger.info("✓ No edges referencing invalid node IDs")
    logger.info("✓ No NaNs or infinities in edge attributes")
    logger.info("✓ No empty graphs after edge filtering")
    logger.info("✓ Timestamps monotonically increasing")
    logger.info("=" * 60)
    logger.info("VALIDATION PASSED")
    logger.info("=" * 60)


def _validate_single_snapshot(
    snapshot: GraphSnapshot,
    index: int,
    reference_num_nodes: int,
    reference_tickers: List[str],
    min_edges_per_graph: int,
) -> None:
    """
    Validate a single graph snapshot.
    
    Args:
        snapshot: GraphSnapshot to validate
        index: Index in the sequence (for error messages)
        reference_num_nodes: Expected number of nodes
        reference_tickers: Expected ticker order
        min_edges_per_graph: Minimum edges required
        
    Raises:
        NodeConsistencyError: If node structure is inconsistent
        EdgeIntegrityError: If edges are invalid
        EmptyGraphError: If graph has too few edges
    """
    timestamp = snapshot.timestamp
    
    # Check 1: Node count consistency
    if snapshot.num_nodes != reference_num_nodes:
        raise NodeConsistencyError(
            f"Node count mismatch at snapshot {index} (timestamp: {timestamp}). "
            f"Expected {reference_num_nodes}, got {snapshot.num_nodes}. "
            f"All snapshots must have the same number of nodes."
        )
    
    # Check 2: Node registry consistency
    if snapshot.tickers != reference_tickers:
        raise NodeConsistencyError(
            f"Ticker order mismatch at snapshot {index} (timestamp: {timestamp}). "
            f"Expected {reference_tickers}, got {snapshot.tickers}. "
            f"Node ordering must be consistent across all snapshots."
        )
    
    # Check 3: Edge index validity
    edge_index = snapshot.edge_index
    if edge_index.shape[1] > 0:  # Only check if there are edges
        # Check source nodes
        source_nodes = edge_index[0]
        invalid_sources = np.where(
            (source_nodes < 0) | (source_nodes >= reference_num_nodes)
        )[0]
        
        if len(invalid_sources) > 0:
            raise EdgeIntegrityError(
                f"Invalid source node IDs at snapshot {index} (timestamp: {timestamp}). "
                f"Found {len(invalid_sources)} edges with invalid sources. "
                f"First invalid: index={invalid_sources[0]}, "
                f"node_id={source_nodes[invalid_sources[0]]}. "
                f"Valid range: [0, {reference_num_nodes - 1}]."
            )
        
        # Check target nodes
        target_nodes = edge_index[1]
        invalid_targets = np.where(
            (target_nodes < 0) | (target_nodes >= reference_num_nodes)
        )[0]
        
        if len(invalid_targets) > 0:
            raise EdgeIntegrityError(
                f"Invalid target node IDs at snapshot {index} (timestamp: {timestamp}). "
                f"Found {len(invalid_targets)} edges with invalid targets. "
                f"First invalid: index={invalid_targets[0]}, "
                f"node_id={target_nodes[invalid_targets[0]]}. "
                f"Valid range: [0, {reference_num_nodes - 1}]."
            )
        
        # Check for self-loops (should not exist)
        self_loops = np.where(source_nodes == target_nodes)[0]
        if len(self_loops) > 0:
            raise EdgeIntegrityError(
                f"Self-loops detected at snapshot {index} (timestamp: {timestamp}). "
                f"Found {len(self_loops)} self-loop edges. "
                f"Self-loops are FORBIDDEN in this graph construction."
            )
    
    # Check 4: Edge attribute validity (NaN/Inf)
    edge_attr = snapshot.edge_attr
    if len(edge_attr) > 0:
        nan_mask = np.isnan(edge_attr)
        if np.any(nan_mask):
            nan_count = np.sum(nan_mask)
            nan_indices = np.where(nan_mask)[0][:5]
            raise EdgeIntegrityError(
                f"NaN values in edge attributes at snapshot {index} (timestamp: {timestamp}). "
                f"Found {nan_count} NaN values. "
                f"First affected edge indices: {nan_indices.tolist()}. "
                f"Edge attributes must not contain NaN."
            )
        
        inf_mask = np.isinf(edge_attr)
        if np.any(inf_mask):
            inf_count = np.sum(inf_mask)
            inf_indices = np.where(inf_mask)[0][:5]
            raise EdgeIntegrityError(
                f"Infinity values in edge attributes at snapshot {index} (timestamp: {timestamp}). "
                f"Found {inf_count} Inf values. "
                f"First affected edge indices: {inf_indices.tolist()}. "
                f"Edge attributes must be finite."
            )
    
    # Check 5: Minimum edges
    if snapshot.num_edges < min_edges_per_graph:
        raise EmptyGraphError(
            f"Too few edges at snapshot {index} (timestamp: {timestamp}). "
            f"Expected at least {min_edges_per_graph}, got {snapshot.num_edges}. "
            f"Consider lowering the correlation threshold or increasing lookback window."
        )


def _validate_temporal_consistency(snapshots: List[GraphSnapshot]) -> None:
    """
    Validate temporal consistency of the snapshot sequence.
    
    Args:
        snapshots: List of GraphSnapshot objects
        
    Raises:
        GraphValidationError: If timestamps are not monotonically increasing
    """
    timestamps = [s.timestamp for s in snapshots]
    
    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i - 1]:
            raise GraphValidationError(
                f"Timestamps not monotonically increasing. "
                f"Snapshot {i - 1}: {timestamps[i - 1]}, "
                f"Snapshot {i}: {timestamps[i]}. "
                f"Graph sequence must be ordered chronologically."
            )


def compute_graph_statistics(snapshots: List[GraphSnapshot]) -> Dict[str, Any]:
    """
    Compute statistics for a graph sequence.
    
    Args:
        snapshots: List of GraphSnapshot objects
        
    Returns:
        Dictionary with statistics
    """
    if not snapshots:
        return {}
    
    edge_counts = [s.num_edges for s in snapshots]
    
    return {
        "num_snapshots": len(snapshots),
        "num_nodes": snapshots[0].num_nodes,
        "min_edges": int(np.min(edge_counts)),
        "max_edges": int(np.max(edge_counts)),
        "mean_edges": float(np.mean(edge_counts)),
        "std_edges": float(np.std(edge_counts)),
        "first_timestamp": str(snapshots[0].timestamp),
        "last_timestamp": str(snapshots[-1].timestamp),
        "lookback_window": snapshots[0].lookback_window,
        "edge_types": snapshots[0].edge_types,
    }

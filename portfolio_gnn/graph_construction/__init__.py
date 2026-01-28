"""
Graph Construction Module for Portfolio GNN.

This module converts preprocessed market time series into time-indexed
financial graphs suitable for GNN training.

CRITICAL GUARANTEES:
- All graph snapshots are causal (no look-ahead bias)
- Edge computations use ONLY data ≤ current timestamp
- Node ordering is deterministic and stable across time
- Compatible with PyTorch Geometric Data format

This module depends ONLY on:
- Phase 1 core infrastructure
- Phase 2 processed market data

NO dependency on models, training, or optimization.
"""

from __future__ import annotations

from portfolio_gnn.graph_construction.nodes import (
    NodeRegistry,
    NodeMetadata,
)

from portfolio_gnn.graph_construction.edges import (
    EdgeBuilder,
    RollingCorrelationEdgeBuilder,
)

from portfolio_gnn.graph_construction.graph_builder import (
    GraphBuilder,
    GraphSnapshot,
    build_graph_sequence,
)

from portfolio_gnn.graph_construction.validation import (
    validate_graph_sequence,
    GraphValidationError,
    NodeConsistencyError,
    EdgeIntegrityError,
    EmptyGraphError,
)

__all__ = [
    # Nodes
    "NodeRegistry",
    "NodeMetadata",
    # Edges
    "EdgeBuilder",
    "RollingCorrelationEdgeBuilder",
    # Graph Builder
    "GraphBuilder",
    "GraphSnapshot",
    "build_graph_sequence",
    # Validation
    "validate_graph_sequence",
    "GraphValidationError",
    "NodeConsistencyError",
    "EdgeIntegrityError",
    "EmptyGraphError",
]

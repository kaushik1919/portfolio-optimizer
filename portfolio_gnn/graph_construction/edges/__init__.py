"""
Edge Construction Module for Portfolio GNN.

This module provides edge builders that compute cross-asset relationships
from market time series data.

CRITICAL GUARANTEES:
- All edge computations are causal (use data ≤ t only)
- No full-sample statistics
- Deterministic given same inputs
- No implicit global state
"""

from __future__ import annotations

from portfolio_gnn.graph_construction.edges.base import EdgeBuilder, EdgeResult
from portfolio_gnn.graph_construction.edges.correlation import RollingCorrelationEdgeBuilder

__all__ = [
    "EdgeBuilder",
    "EdgeResult",
    "RollingCorrelationEdgeBuilder",
]

"""
Graph Builder for Portfolio GNN.

Constructs time-indexed graph snapshots from preprocessed market data.

CRITICAL GUARANTEES:
- Graphs are built per timestamp in chronological order
- Each graph uses ONLY data ≤ that timestamp
- Timestamps without sufficient lookback history are skipped
- Output is compatible with PyTorch Geometric Data format
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    import torch
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    Data = None

from portfolio_gnn.graph_construction.nodes import NodeRegistry
from portfolio_gnn.graph_construction.edges.base import EdgeBuilder, EdgeResult

logger = logging.getLogger(__name__)


@dataclass
class GraphSnapshot:
    """
    A single graph snapshot at a specific timestamp.
    
    This is an intermediate representation that can be converted
    to PyTorch Geometric Data format.
    
    Attributes:
        timestamp: The timestamp this graph represents
        num_nodes: Number of nodes (assets) in the graph
        edge_index: Edge connectivity in COO format (2, num_edges)
        edge_attr: Edge attributes/weights (num_edges, num_features)
        edge_types: List of edge type names
        node_registry: Reference to the node registry
        lookback_window: Lookback window used for edge computation
    """
    timestamp: pd.Timestamp
    num_nodes: int
    edge_index: np.ndarray
    edge_attr: np.ndarray
    edge_types: List[str]
    node_registry: NodeRegistry
    lookback_window: int
    
    def __post_init__(self) -> None:
        """Validate graph snapshot."""
        if self.edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape (2, num_edges), "
                f"got {self.edge_index.shape}"
            )
        
        num_edges = self.edge_index.shape[1]
        if len(self.edge_attr) != num_edges:
            raise ValueError(
                f"edge_attr length ({len(self.edge_attr)}) must match "
                f"number of edges ({num_edges})"
            )
    
    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return self.edge_index.shape[1]
    
    @property
    def tickers(self) -> List[str]:
        """List of tickers in node order."""
        return self.node_registry.tickers
    
    def to_pyg_data(self) -> "Data":
        """
        Convert to PyTorch Geometric Data object.
        
        Returns:
            Data object with:
                - x: None (node features come in later phase)
                - edge_index: LongTensor of shape (2, num_edges)
                - edge_attr: FloatTensor of shape (num_edges, 1)
                - timestamp: Stored in data object
                
        Raises:
            ImportError: If PyTorch Geometric is not installed
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric is required for to_pyg_data(). "
                "Install with: pip install torch-geometric"
            )
        
        # Convert to PyTorch tensors
        edge_index = torch.from_numpy(self.edge_index).long()
        edge_attr = torch.from_numpy(self.edge_attr).float()
        
        # Reshape edge_attr to (num_edges, 1) if 1D
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)
        
        data = Data(
            x=None,  # Node features come in Phase 4
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=self.num_nodes,
        )
        
        # Store metadata
        data.timestamp = self.timestamp
        data.lookback_window = self.lookback_window
        data.edge_types = self.edge_types
        
        return data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": str(self.timestamp),
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "edge_types": self.edge_types,
            "lookback_window": self.lookback_window,
            "tickers": self.tickers,
        }


class GraphBuilder:
    """
    Builds time-indexed graph snapshots from market data.
    
    The builder:
    1. Iterates over valid timestamps t
    2. Skips timestamps without sufficient lookback history
    3. Constructs a graph snapshot per t using configured edge builders
    4. Returns an ordered sequence of graph snapshots
    
    Example:
        >>> from portfolio_gnn.data_ingestion.preprocessing import PreprocessedData
        >>> from portfolio_gnn.graph_construction.edges import RollingCorrelationEdgeBuilder
        >>> 
        >>> edge_builder = RollingCorrelationEdgeBuilder(lookback_window=60)
        >>> graph_builder = GraphBuilder(
        ...     edge_builders=[edge_builder],
        ...     min_edges_per_node=1,
        ... )
        >>> snapshots = graph_builder.build_sequence(processed_data)
    """
    
    def __init__(
        self,
        edge_builders: List[EdgeBuilder],
        min_edges_per_node: int = 0,
    ) -> None:
        """
        Initialize graph builder.
        
        Args:
            edge_builders: List of edge builders to use
            min_edges_per_node: Minimum edges per node (advisory, not enforced)
        """
        if not edge_builders:
            raise ValueError("At least one edge builder is required")
        
        self._edge_builders = edge_builders
        self._min_edges_per_node = min_edges_per_node
        
        # Compute maximum lookback needed
        self._max_lookback = max(eb.lookback_window for eb in edge_builders)
        
        logger.info(
            f"GraphBuilder initialized with {len(edge_builders)} edge builder(s)"
        )
        logger.info(f"Maximum lookback window: {self._max_lookback}")
    
    @property
    def max_lookback(self) -> int:
        """Maximum lookback window across all edge builders."""
        return self._max_lookback
    
    @property
    def edge_builders(self) -> List[EdgeBuilder]:
        """List of edge builders."""
        return list(self._edge_builders)
    
    def build_sequence(
        self,
        prices: pd.DataFrame,
        node_registry: Optional[NodeRegistry] = None,
    ) -> List[GraphSnapshot]:
        """
        Build a sequence of graph snapshots from price data.
        
        Args:
            prices: DataFrame with columns = tickers, index = dates (sorted)
            node_registry: Optional pre-built registry. If None, created from prices.
            
        Returns:
            List of GraphSnapshot objects in chronological order
            
        Raises:
            ValueError: If insufficient data or invalid configuration
        """
        logger.info("=" * 60)
        logger.info("GRAPH SEQUENCE CONSTRUCTION")
        logger.info("=" * 60)
        
        # Validate input
        if prices.empty:
            raise ValueError("prices DataFrame cannot be empty")
        
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("prices index must be DatetimeIndex")
        
        # Ensure sorted
        if not prices.index.is_monotonic_increasing:
            raise ValueError("prices index must be monotonically increasing")
        
        # Create node registry if not provided
        if node_registry is None:
            node_registry = NodeRegistry.from_tickers(list(prices.columns))
        
        # Validate node registry matches prices
        if set(node_registry.tickers) != set(prices.columns):
            raise ValueError(
                f"Node registry tickers do not match price columns. "
                f"Registry: {sorted(node_registry.tickers)}, "
                f"Prices: {sorted(prices.columns)}"
            )
        
        # Get node-ordered column list
        node_order = node_registry.tickers
        
        # Determine valid timestamps (those with sufficient lookback)
        all_timestamps = prices.index
        valid_start_idx = self._max_lookback
        
        if valid_start_idx >= len(all_timestamps):
            raise ValueError(
                f"Insufficient history for lookback window. "
                f"Need at least {self._max_lookback + 1} days, "
                f"have {len(all_timestamps)}"
            )
        
        valid_timestamps = all_timestamps[valid_start_idx:]
        
        logger.info(f"Total timestamps: {len(all_timestamps)}")
        logger.info(f"Valid timestamps (after lookback): {len(valid_timestamps)}")
        logger.info(f"Lookback window: {self._max_lookback}")
        logger.info(f"First valid timestamp: {valid_timestamps[0]}")
        logger.info(f"Last valid timestamp: {valid_timestamps[-1]}")
        
        # Build snapshots
        snapshots: List[GraphSnapshot] = []
        
        for i, timestamp in enumerate(valid_timestamps):
            snapshot = self._build_single_snapshot(
                prices=prices,
                timestamp=timestamp,
                node_registry=node_registry,
                node_order=node_order,
            )
            snapshots.append(snapshot)
            
            if (i + 1) % 100 == 0:
                logger.debug(f"Built {i + 1}/{len(valid_timestamps)} snapshots")
        
        logger.info("-" * 60)
        logger.info(f"Built {len(snapshots)} graph snapshots")
        
        # Compute edge statistics
        edge_counts = [s.num_edges for s in snapshots]
        logger.info(f"Edges per graph: min={min(edge_counts)}, max={max(edge_counts)}, "
                   f"mean={np.mean(edge_counts):.1f}")
        logger.info("=" * 60)
        
        return snapshots
    
    def _build_single_snapshot(
        self,
        prices: pd.DataFrame,
        timestamp: pd.Timestamp,
        node_registry: NodeRegistry,
        node_order: List[str],
    ) -> GraphSnapshot:
        """
        Build a single graph snapshot for a timestamp.
        
        Args:
            prices: Full price DataFrame
            timestamp: Current timestamp
            node_registry: Node registry
            node_order: Ordered ticker list
            
        Returns:
            GraphSnapshot for the timestamp
        """
        all_edges: List[EdgeResult] = []
        
        for builder in self._edge_builders:
            edge_result = builder.compute_edges(
                prices=prices,
                timestamp=timestamp,
                node_order=node_order,
            )
            all_edges.append(edge_result)
        
        # Combine edges from all builders
        if len(all_edges) == 1:
            combined_edge_index = all_edges[0].edge_index
            combined_edge_attr = all_edges[0].edge_weights
            edge_types = [all_edges[0].edge_type]
        else:
            # Stack edges from multiple builders
            edge_indices = [e.edge_index for e in all_edges]
            edge_attrs = [e.edge_weights for e in all_edges]
            edge_types = [e.edge_type for e in all_edges]
            
            combined_edge_index = np.concatenate(edge_indices, axis=1)
            combined_edge_attr = np.concatenate(edge_attrs, axis=0)
        
        return GraphSnapshot(
            timestamp=timestamp,
            num_nodes=node_registry.num_nodes,
            edge_index=combined_edge_index,
            edge_attr=combined_edge_attr,
            edge_types=edge_types,
            node_registry=node_registry,
            lookback_window=self._max_lookback,
        )


def build_graph_sequence(
    processed_data: Any,
    config: Any,
) -> List[GraphSnapshot]:
    """
    Build graph sequence from preprocessed data using configuration.
    
    This is the main entry point for Phase 3 graph construction.
    
    Args:
        processed_data: PreprocessedData from Phase 2
        config: Configuration object with graph section
        
    Returns:
        List of GraphSnapshot objects
        
    Raises:
        ValueError: If configuration is invalid
    """
    logger.info("")
    logger.info("Building graph sequence from configuration...")
    
    # Extract graph configuration
    graph_config = _extract_graph_config(config)
    
    logger.info(f"Graph config: {graph_config}")
    
    # Create edge builder(s)
    edge_builder = RollingCorrelationEdgeBuilder(
        lookback_window=graph_config["lookback_window"],
        threshold=graph_config["correlation_threshold"],
        directed=graph_config["directed"],
    )
    
    # Create graph builder
    graph_builder = GraphBuilder(
        edge_builders=[edge_builder],
        min_edges_per_node=graph_config["min_edges_per_node"],
    )
    
    # Create node registry from processed data
    node_registry = NodeRegistry.from_tickers(processed_data.tickers)
    
    # Build graph sequence
    snapshots = graph_builder.build_sequence(
        prices=processed_data.prices,
        node_registry=node_registry,
    )
    
    return snapshots


def _extract_graph_config(config: Any) -> Dict[str, Any]:
    """
    Extract graph configuration from config object.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary with graph parameters
        
    Raises:
        ValueError: If required parameters are missing
    """
    # Try different access patterns
    if hasattr(config, "graph"):
        graph_section = config.graph
    elif hasattr(config, "__getitem__"):
        try:
            graph_section = config["graph"]
        except (KeyError, TypeError):
            graph_section = None
    else:
        graph_section = None
    
    if graph_section is None:
        raise ValueError(
            "Configuration must have a 'graph' section with parameters: "
            "lookback_window, correlation_threshold, directed, min_edges_per_node"
        )
    
    # Extract parameters with defaults
    def get_param(section, key, default=None):
        if hasattr(section, key):
            return getattr(section, key)
        elif hasattr(section, "get"):
            return section.get(key, default)
        elif hasattr(section, "__getitem__"):
            try:
                return section[key]
            except (KeyError, TypeError):
                return default
        return default
    
    lookback_window = get_param(graph_section, "lookback_window", 60)
    correlation_threshold = get_param(graph_section, "correlation_threshold", 0.5)
    directed = get_param(graph_section, "directed", False)
    min_edges_per_node = get_param(graph_section, "min_edges_per_node", 0)
    
    return {
        "lookback_window": int(lookback_window),
        "correlation_threshold": float(correlation_threshold),
        "directed": bool(directed),
        "min_edges_per_node": int(min_edges_per_node),
    }


# Import for convenience
from portfolio_gnn.graph_construction.edges.correlation import RollingCorrelationEdgeBuilder

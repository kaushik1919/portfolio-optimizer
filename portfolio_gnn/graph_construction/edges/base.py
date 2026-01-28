"""
Edge Builder Base Class for Portfolio GNN.

Defines the strict interface for all edge computation methods.

CRITICAL GUARANTEES:
- All edge computations MUST use data ≤ t (no lookahead)
- No full-sample statistics
- No implicit global state
- Deterministic given same inputs

This abstraction supports multiple edge types in the future.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EdgeResult:
    """
    Result of edge computation for a single timestamp.
    
    All edge indices reference the NodeRegistry indices.
    
    Attributes:
        source_nodes: Array of source node indices
        target_nodes: Array of target node indices
        edge_weights: Array of edge weights/attributes
        edge_type: String identifier for edge type
        timestamp: The timestamp this edge set was computed for
        lookback_window: Number of days used in computation
    """
    source_nodes: np.ndarray
    target_nodes: np.ndarray
    edge_weights: np.ndarray
    edge_type: str
    timestamp: pd.Timestamp
    lookback_window: int
    
    def __post_init__(self) -> None:
        """Validate edge result."""
        # Ensure arrays have same length
        n_edges = len(self.source_nodes)
        if len(self.target_nodes) != n_edges:
            raise ValueError(
                f"source_nodes ({n_edges}) and target_nodes ({len(self.target_nodes)}) "
                f"must have same length"
            )
        if len(self.edge_weights) != n_edges:
            raise ValueError(
                f"source_nodes ({n_edges}) and edge_weights ({len(self.edge_weights)}) "
                f"must have same length"
            )
        
        # Ensure numpy arrays
        if not isinstance(self.source_nodes, np.ndarray):
            object.__setattr__(self, 'source_nodes', np.array(self.source_nodes))
        if not isinstance(self.target_nodes, np.ndarray):
            object.__setattr__(self, 'target_nodes', np.array(self.target_nodes))
        if not isinstance(self.edge_weights, np.ndarray):
            object.__setattr__(self, 'edge_weights', np.array(self.edge_weights))
    
    @property
    def num_edges(self) -> int:
        """Number of edges."""
        return len(self.source_nodes)
    
    @property
    def edge_index(self) -> np.ndarray:
        """
        Edge index in COO format for PyTorch Geometric.
        
        Returns:
            2D array of shape (2, num_edges) with [source; target]
        """
        return np.stack([self.source_nodes, self.target_nodes], axis=0)
    
    def has_self_loops(self) -> bool:
        """Check if any self-loops exist."""
        return np.any(self.source_nodes == self.target_nodes)
    
    def get_self_loop_count(self) -> int:
        """Count number of self-loops."""
        return int(np.sum(self.source_nodes == self.target_nodes))


class EdgeBuilder(ABC):
    """
    Abstract base class for edge builders.
    
    All edge builders must implement the compute_edges method
    with strict guarantees:
    
    - Input: Aligned price series, explicit time index t, lookback L
    - Output: EdgeResult with edges computed using data ≤ t only
    - No lookahead bias
    - Deterministic given same inputs
    
    Subclasses:
        - RollingCorrelationEdgeBuilder: Pearson correlation edges
        - (Future) SectorEdgeBuilder: Same-sector edges
        - (Future) MomentumEdgeBuilder: Momentum-based edges
    """
    
    def __init__(
        self,
        lookback_window: int,
        threshold: float,
        directed: bool = False,
    ) -> None:
        """
        Initialize edge builder.
        
        Args:
            lookback_window: Number of days to use for edge computation
            threshold: Minimum absolute value to include edge
            directed: Whether to produce directed edges
        """
        if lookback_window < 2:
            raise ValueError(f"lookback_window must be >= 2, got {lookback_window}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        
        self._lookback_window = lookback_window
        self._threshold = threshold
        self._directed = directed
        
        logger.debug(
            f"{self.__class__.__name__} initialized: "
            f"lookback={lookback_window}, threshold={threshold}, directed={directed}"
        )
    
    @property
    def lookback_window(self) -> int:
        """Lookback window size."""
        return self._lookback_window
    
    @property
    def threshold(self) -> float:
        """Edge inclusion threshold."""
        return self._threshold
    
    @property
    def directed(self) -> bool:
        """Whether edges are directed."""
        return self._directed
    
    @property
    @abstractmethod
    def edge_type(self) -> str:
        """Unique identifier for this edge type."""
        pass
    
    @abstractmethod
    def compute_edges(
        self,
        prices: pd.DataFrame,
        timestamp: pd.Timestamp,
        node_order: List[str],
    ) -> EdgeResult:
        """
        Compute edges for a single timestamp.
        
        CRITICAL: This method MUST use only data with index ≤ timestamp.
        No lookahead is permitted.
        
        Args:
            prices: DataFrame with columns = tickers, index = dates
                    Contains the FULL price history up to and including timestamp
            timestamp: The current timestamp to compute edges for
            node_order: Ordered list of tickers matching node indices
            
        Returns:
            EdgeResult with edges computed from data ≤ timestamp
            
        Raises:
            ValueError: If insufficient data for computation
        """
        pass
    
    def validate_no_lookahead(
        self,
        prices: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Filter prices to enforce no lookahead.
        
        This method ensures only data ≤ timestamp is used.
        
        Args:
            prices: Full price DataFrame
            timestamp: Current timestamp
            
        Returns:
            Filtered DataFrame with only past/current data
        """
        # CRITICAL: Only use data up to and including timestamp
        valid_mask = prices.index <= timestamp
        filtered = prices.loc[valid_mask]
        
        logger.debug(
            f"Lookahead filter: {len(prices)} -> {len(filtered)} rows "
            f"(cutoff: {timestamp})"
        )
        
        return filtered
    
    def get_lookback_slice(
        self,
        prices: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Get the lookback window ending at timestamp (inclusive).
        
        Args:
            prices: DataFrame filtered to ≤ timestamp
            timestamp: End of lookback window
            
        Returns:
            Last lookback_window rows of prices
            
        Raises:
            ValueError: If insufficient history
        """
        # First apply no-lookahead filter
        filtered = self.validate_no_lookahead(prices, timestamp)
        
        if len(filtered) < self._lookback_window:
            raise ValueError(
                f"Insufficient history for lookback window. "
                f"Need {self._lookback_window} days, have {len(filtered)} "
                f"(timestamp: {timestamp})"
            )
        
        # Take last lookback_window rows
        lookback_data = filtered.iloc[-self._lookback_window:]
        
        return lookback_data

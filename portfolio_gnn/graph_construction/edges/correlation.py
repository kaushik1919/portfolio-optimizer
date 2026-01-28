"""
Rolling Correlation Edge Builder for Portfolio GNN.

Implements lag-respecting rolling correlation edges between assets.

CRITICAL GUARANTEES:
- Returns computed ONLY within the rolling window [t-L, t]
- Correlation computed per snapshot, NOT globally
- No future data
- No expanding windows
- No normalization beyond correlation math
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from portfolio_gnn.graph_construction.edges.base import EdgeBuilder, EdgeResult

logger = logging.getLogger(__name__)


class RollingCorrelationEdgeBuilder(EdgeBuilder):
    """
    Builds edges based on rolling Pearson correlation of returns.
    
    For each timestamp t:
    1. Extract returns from window [t-L, t] (L = lookback_window)
    2. Compute pairwise Pearson correlations
    3. Create edges where |corr| >= threshold
    4. Store correlation as edge weight
    
    Edge rules:
    - Self-edges are FORBIDDEN
    - Graph can be undirected (default) or directed
    - Correlation values stored as edge attributes
    
    Example:
        >>> builder = RollingCorrelationEdgeBuilder(
        ...     lookback_window=60,
        ...     threshold=0.5,
        ...     directed=False,
        ... )
        >>> edges = builder.compute_edges(prices, timestamp, node_order)
    """
    
    def __init__(
        self,
        lookback_window: int = 60,
        threshold: float = 0.5,
        directed: bool = False,
        min_periods: int = None,
    ) -> None:
        """
        Initialize rolling correlation edge builder.
        
        Args:
            lookback_window: Number of days for correlation window
            threshold: Minimum |correlation| to include edge
            directed: If True, include both (i,j) and (j,i) edges
            min_periods: Minimum valid observations required.
                         Default: lookback_window (require full window)
        """
        super().__init__(
            lookback_window=lookback_window,
            threshold=threshold,
            directed=directed,
        )
        
        # Require full window by default (no partial correlations)
        self._min_periods = min_periods if min_periods is not None else lookback_window
        
        if self._min_periods > lookback_window:
            raise ValueError(
                f"min_periods ({self._min_periods}) cannot exceed "
                f"lookback_window ({lookback_window})"
            )
        
        logger.info(
            f"RollingCorrelationEdgeBuilder initialized: "
            f"lookback={lookback_window}, threshold={threshold}, "
            f"directed={directed}, min_periods={self._min_periods}"
        )
    
    @property
    def edge_type(self) -> str:
        """Edge type identifier."""
        return "correlation"
    
    @property
    def min_periods(self) -> int:
        """Minimum periods required for correlation."""
        return self._min_periods
    
    def compute_edges(
        self,
        prices: pd.DataFrame,
        timestamp: pd.Timestamp,
        node_order: List[str],
    ) -> EdgeResult:
        """
        Compute correlation-based edges for a single timestamp.
        
        Args:
            prices: DataFrame with columns = tickers, index = dates
            timestamp: Current timestamp to compute edges for
            node_order: Ordered list of tickers matching node indices
            
        Returns:
            EdgeResult with correlation edges
            
        Raises:
            ValueError: If insufficient data for computation
        """
        # Step 1: Get lookback slice (enforces no-lookahead)
        lookback_prices = self.get_lookback_slice(prices, timestamp)
        
        # Step 2: Compute returns within the window ONLY
        # Use simple returns: (P_t - P_{t-1}) / P_{t-1}
        returns = lookback_prices.pct_change().dropna()
        
        if len(returns) < self._min_periods - 1:
            raise ValueError(
                f"Insufficient returns for correlation. "
                f"Need {self._min_periods - 1}, have {len(returns)} "
                f"(timestamp: {timestamp})"
            )
        
        # Step 3: Ensure columns match node_order
        # This ensures edge indices match node registry
        missing_cols = set(node_order) - set(returns.columns)
        if missing_cols:
            raise ValueError(
                f"Missing tickers in price data: {missing_cols}"
            )
        
        # Reorder columns to match node_order
        returns = returns[node_order]
        
        # Step 4: Compute correlation matrix
        corr_matrix = returns.corr(method="pearson")
        
        # Step 5: Extract edges above threshold
        source_nodes, target_nodes, edge_weights = self._extract_edges(
            corr_matrix,
            self._threshold,
            self._directed,
        )
        
        logger.debug(
            f"Computed {len(source_nodes)} edges for {timestamp} "
            f"(threshold={self._threshold})"
        )
        
        return EdgeResult(
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            edge_weights=edge_weights,
            edge_type=self.edge_type,
            timestamp=timestamp,
            lookback_window=self._lookback_window,
        )
    
    def _extract_edges(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float,
        directed: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract edges from correlation matrix.
        
        Args:
            corr_matrix: NxN correlation matrix
            threshold: Minimum |correlation| for edge
            directed: Whether to include both directions
            
        Returns:
            Tuple of (source_indices, target_indices, weights)
        """
        n = len(corr_matrix)
        corr_values = corr_matrix.values
        
        source_list = []
        target_list = []
        weight_list = []
        
        for i in range(n):
            # Start from 0 if directed, else from i+1 to avoid duplicates
            start_j = 0 if directed else i + 1
            
            for j in range(start_j, n):
                # Skip self-loops (FORBIDDEN)
                if i == j:
                    continue
                
                corr_val = corr_values[i, j]
                
                # Skip NaN correlations
                if np.isnan(corr_val):
                    continue
                
                # Check threshold on absolute value
                if abs(corr_val) >= threshold:
                    source_list.append(i)
                    target_list.append(j)
                    weight_list.append(corr_val)
                    
                    # For undirected, add reverse edge
                    if not directed:
                        source_list.append(j)
                        target_list.append(i)
                        weight_list.append(corr_val)
        
        return (
            np.array(source_list, dtype=np.int64),
            np.array(target_list, dtype=np.int64),
            np.array(weight_list, dtype=np.float32),
        )
    
    def compute_returns(
        self,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute simple returns from prices.
        
        This is exposed for testing purposes.
        
        Args:
            prices: Price DataFrame
            
        Returns:
            Returns DataFrame
        """
        return prices.pct_change().dropna()

"""
Node Registry for Portfolio GNN.

Defines node semantics for the financial graph.

Each node represents exactly one tradable asset.
Node ordering is deterministic and stable across time.
Node index is consistent across all graph snapshots.

FORBIDDEN in this phase:
- No node features computed here
- No time-varying node attributes
- No learned embeddings

Nodes are identity holders only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeMetadata:
    """
    Immutable metadata for a single node (asset).
    
    This class holds static identity information only.
    NO time-varying attributes. NO computed features.
    
    Attributes:
        ticker: Asset ticker symbol (e.g., "AAPL")
        node_index: Fixed integer index for this node (0-indexed)
        asset_type: Type of asset (e.g., "equity", "etf")
        sector: Optional sector classification
    """
    ticker: str
    node_index: int
    asset_type: str = "equity"
    sector: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate node metadata."""
        if not self.ticker:
            raise ValueError("ticker cannot be empty")
        if self.node_index < 0:
            raise ValueError(f"node_index must be non-negative, got {self.node_index}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "node_index": self.node_index,
            "asset_type": self.asset_type,
            "sector": self.sector,
        }


class NodeRegistry:
    """
    Registry that maps assets to stable node indices.
    
    This class ensures:
    - Deterministic node ordering (sorted by ticker)
    - Stable indices across all graph snapshots
    - Consistent mapping from ticker to index
    
    The registry is immutable after construction.
    
    Example:
        >>> registry = NodeRegistry.from_tickers(["MSFT", "AAPL", "GOOGL"])
        >>> registry.num_nodes
        3
        >>> registry.get_index("AAPL")
        0  # Sorted alphabetically
        >>> registry.get_ticker(0)
        "AAPL"
    """
    
    def __init__(self, nodes: List[NodeMetadata]) -> None:
        """
        Initialize registry with pre-built node metadata.
        
        Use NodeRegistry.from_tickers() for standard construction.
        
        Args:
            nodes: List of NodeMetadata objects, already sorted by ticker
        """
        if not nodes:
            raise ValueError("nodes list cannot be empty")
        
        self._nodes: List[NodeMetadata] = nodes
        self._ticker_to_index: Dict[str, int] = {
            node.ticker: node.node_index for node in nodes
        }
        self._index_to_ticker: Dict[int, str] = {
            node.node_index: node.ticker for node in nodes
        }
        
        # Validate index consistency
        expected_indices = set(range(len(nodes)))
        actual_indices = set(self._index_to_ticker.keys())
        if expected_indices != actual_indices:
            raise ValueError(
                f"Node indices must be contiguous 0..{len(nodes)-1}. "
                f"Got: {sorted(actual_indices)}"
            )
        
        # Freeze after construction
        self._frozen = True
        
        logger.debug(f"NodeRegistry created with {len(nodes)} nodes")
    
    @classmethod
    def from_tickers(
        cls,
        tickers: List[str],
        asset_types: Optional[Dict[str, str]] = None,
        sectors: Optional[Dict[str, str]] = None,
    ) -> "NodeRegistry":
        """
        Create a NodeRegistry from a list of ticker symbols.
        
        Tickers are sorted alphabetically to ensure deterministic ordering.
        
        Args:
            tickers: List of ticker symbols
            asset_types: Optional mapping of ticker -> asset_type
            sectors: Optional mapping of ticker -> sector
            
        Returns:
            NodeRegistry with stable, sorted node indices
            
        Raises:
            ValueError: If tickers list is empty or contains duplicates
        """
        if not tickers:
            raise ValueError("tickers list cannot be empty")
        
        # Check for duplicates
        unique_tickers = set(tickers)
        if len(unique_tickers) != len(tickers):
            duplicates = [t for t in tickers if tickers.count(t) > 1]
            raise ValueError(f"Duplicate tickers found: {set(duplicates)}")
        
        # Sort tickers for deterministic ordering
        sorted_tickers = sorted(tickers)
        
        # Build node metadata
        asset_types = asset_types or {}
        sectors = sectors or {}
        
        nodes = []
        for idx, ticker in enumerate(sorted_tickers):
            node = NodeMetadata(
                ticker=ticker,
                node_index=idx,
                asset_type=asset_types.get(ticker, "equity"),
                sector=sectors.get(ticker),
            )
            nodes.append(node)
        
        logger.info(f"Created NodeRegistry with {len(nodes)} nodes (sorted)")
        logger.debug(f"Node order: {sorted_tickers}")
        
        return cls(nodes)
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the registry."""
        return len(self._nodes)
    
    @property
    def tickers(self) -> List[str]:
        """List of tickers in node index order."""
        return [self._index_to_ticker[i] for i in range(self.num_nodes)]
    
    @property
    def nodes(self) -> List[NodeMetadata]:
        """List of node metadata in index order."""
        return list(self._nodes)
    
    def get_index(self, ticker: str) -> int:
        """
        Get node index for a ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Node index (0-indexed)
            
        Raises:
            KeyError: If ticker not in registry
        """
        if ticker not in self._ticker_to_index:
            raise KeyError(f"Ticker not in registry: {ticker}")
        return self._ticker_to_index[ticker]
    
    def get_ticker(self, index: int) -> str:
        """
        Get ticker for a node index.
        
        Args:
            index: Node index (0-indexed)
            
        Returns:
            Ticker symbol
            
        Raises:
            KeyError: If index not in registry
        """
        if index not in self._index_to_ticker:
            raise KeyError(f"Index not in registry: {index}")
        return self._index_to_ticker[index]
    
    def get_node(self, ticker: str) -> NodeMetadata:
        """
        Get full node metadata for a ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            NodeMetadata for the ticker
            
        Raises:
            KeyError: If ticker not in registry
        """
        idx = self.get_index(ticker)
        return self._nodes[idx]
    
    def contains(self, ticker: str) -> bool:
        """Check if ticker is in registry."""
        return ticker in self._ticker_to_index
    
    def validate_index(self, index: int) -> bool:
        """Check if index is valid."""
        return 0 <= index < self.num_nodes
    
    def __len__(self) -> int:
        """Return number of nodes."""
        return self.num_nodes
    
    def __contains__(self, ticker: str) -> bool:
        """Check if ticker is in registry."""
        return self.contains(ticker)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"NodeRegistry(num_nodes={self.num_nodes}, tickers={self.tickers[:5]}{'...' if self.num_nodes > 5 else ''})"

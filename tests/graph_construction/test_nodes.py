"""
Tests for Node Registry.

Tests ensure:
- Deterministic node ordering (sorted by ticker)
- Stable indices across all operations
- Consistent mapping from ticker to index and vice versa
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_gnn.graph_construction.nodes import NodeRegistry, NodeMetadata


class TestNodeMetadata:
    """Tests for NodeMetadata dataclass."""
    
    def test_create_valid_metadata(self):
        """Create valid node metadata."""
        node = NodeMetadata(
            ticker="AAPL",
            node_index=0,
            asset_type="equity",
            sector="Technology",
        )
        
        assert node.ticker == "AAPL"
        assert node.node_index == 0
        assert node.asset_type == "equity"
        assert node.sector == "Technology"
    
    def test_empty_ticker_raises(self):
        """Empty ticker raises ValueError."""
        with pytest.raises(ValueError, match="ticker cannot be empty"):
            NodeMetadata(ticker="", node_index=0)
    
    def test_negative_index_raises(self):
        """Negative node_index raises ValueError."""
        with pytest.raises(ValueError, match="node_index must be non-negative"):
            NodeMetadata(ticker="AAPL", node_index=-1)
    
    def test_to_dict(self):
        """Convert to dictionary."""
        node = NodeMetadata(
            ticker="MSFT",
            node_index=1,
            sector="Technology",
        )
        
        d = node.to_dict()
        
        assert d["ticker"] == "MSFT"
        assert d["node_index"] == 1
        assert d["asset_type"] == "equity"
        assert d["sector"] == "Technology"
    
    def test_immutable(self):
        """NodeMetadata is immutable (frozen dataclass)."""
        node = NodeMetadata(ticker="AAPL", node_index=0)
        
        with pytest.raises(AttributeError):
            node.ticker = "MSFT"


class TestNodeRegistry:
    """Tests for NodeRegistry."""
    
    def test_create_from_tickers(self):
        """Create registry from ticker list."""
        tickers = ["MSFT", "AAPL", "GOOGL"]
        registry = NodeRegistry.from_tickers(tickers)
        
        assert registry.num_nodes == 3
    
    def test_tickers_are_sorted(self):
        """Tickers are sorted alphabetically for determinism."""
        tickers = ["MSFT", "AAPL", "GOOGL"]
        registry = NodeRegistry.from_tickers(tickers)
        
        # Should be sorted: AAPL, GOOGL, MSFT
        assert registry.tickers == ["AAPL", "GOOGL", "MSFT"]
    
    def test_deterministic_ordering(self):
        """Same input produces same ordering every time."""
        tickers = ["NVDA", "AMZN", "META", "TSLA"]
        
        registry1 = NodeRegistry.from_tickers(tickers)
        registry2 = NodeRegistry.from_tickers(tickers)
        
        assert registry1.tickers == registry2.tickers
        assert registry1.tickers == ["AMZN", "META", "NVDA", "TSLA"]
    
    def test_index_assignment(self):
        """Indices are assigned correctly after sorting."""
        tickers = ["MSFT", "AAPL", "GOOGL"]
        registry = NodeRegistry.from_tickers(tickers)
        
        # Sorted order: AAPL=0, GOOGL=1, MSFT=2
        assert registry.get_index("AAPL") == 0
        assert registry.get_index("GOOGL") == 1
        assert registry.get_index("MSFT") == 2
    
    def test_get_ticker_by_index(self):
        """Get ticker by index."""
        tickers = ["MSFT", "AAPL"]
        registry = NodeRegistry.from_tickers(tickers)
        
        assert registry.get_ticker(0) == "AAPL"
        assert registry.get_ticker(1) == "MSFT"
    
    def test_get_index_invalid_ticker_raises(self):
        """Get index for unknown ticker raises KeyError."""
        registry = NodeRegistry.from_tickers(["AAPL"])
        
        with pytest.raises(KeyError, match="Ticker not in registry"):
            registry.get_index("INVALID")
    
    def test_get_ticker_invalid_index_raises(self):
        """Get ticker for invalid index raises KeyError."""
        registry = NodeRegistry.from_tickers(["AAPL"])
        
        with pytest.raises(KeyError, match="Index not in registry"):
            registry.get_ticker(999)
    
    def test_contains_ticker(self):
        """Check if ticker is in registry."""
        registry = NodeRegistry.from_tickers(["AAPL", "MSFT"])
        
        assert registry.contains("AAPL")
        assert "MSFT" in registry
        assert not registry.contains("GOOGL")
        assert "GOOGL" not in registry
    
    def test_validate_index(self):
        """Validate index is in range."""
        registry = NodeRegistry.from_tickers(["A", "B", "C"])
        
        assert registry.validate_index(0)
        assert registry.validate_index(1)
        assert registry.validate_index(2)
        assert not registry.validate_index(-1)
        assert not registry.validate_index(3)
    
    def test_empty_tickers_raises(self):
        """Empty ticker list raises ValueError."""
        with pytest.raises(ValueError, match="tickers list cannot be empty"):
            NodeRegistry.from_tickers([])
    
    def test_duplicate_tickers_raises(self):
        """Duplicate tickers raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate tickers found"):
            NodeRegistry.from_tickers(["AAPL", "MSFT", "AAPL"])
    
    def test_get_node_metadata(self):
        """Get full node metadata."""
        registry = NodeRegistry.from_tickers(
            ["AAPL", "MSFT"],
            sectors={"AAPL": "Technology", "MSFT": "Technology"},
        )
        
        node = registry.get_node("AAPL")
        
        assert node.ticker == "AAPL"
        assert node.node_index == 0
        assert node.sector == "Technology"
    
    def test_len(self):
        """Test __len__ method."""
        registry = NodeRegistry.from_tickers(["A", "B", "C", "D"])
        
        assert len(registry) == 4
    
    def test_asset_types(self):
        """Custom asset types are applied."""
        tickers = ["SPY", "AAPL"]
        asset_types = {"SPY": "etf", "AAPL": "equity"}
        
        registry = NodeRegistry.from_tickers(tickers, asset_types=asset_types)
        
        assert registry.get_node("SPY").asset_type == "etf"
        assert registry.get_node("AAPL").asset_type == "equity"


class TestNodeRegistryConsistency:
    """Tests for node registry consistency across operations."""
    
    def test_nodes_property_returns_sorted_list(self):
        """Nodes property returns list in index order."""
        registry = NodeRegistry.from_tickers(["Z", "A", "M"])
        
        nodes = registry.nodes
        
        assert len(nodes) == 3
        assert nodes[0].ticker == "A"
        assert nodes[1].ticker == "M"
        assert nodes[2].ticker == "Z"
    
    def test_index_to_ticker_roundtrip(self):
        """Index -> ticker -> index roundtrip is consistent."""
        registry = NodeRegistry.from_tickers(["AAPL", "MSFT", "GOOGL"])
        
        for i in range(registry.num_nodes):
            ticker = registry.get_ticker(i)
            index = registry.get_index(ticker)
            assert index == i
    
    def test_ticker_to_index_roundtrip(self):
        """Ticker -> index -> ticker roundtrip is consistent."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        registry = NodeRegistry.from_tickers(tickers)
        
        for ticker in tickers:
            index = registry.get_index(ticker)
            recovered_ticker = registry.get_ticker(index)
            assert recovered_ticker == ticker

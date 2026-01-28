"""
Data Ingestion Module for Portfolio GNN.

This module provides market data ingestion with strict guarantees:
- Time alignment across all series
- No look-ahead bias
- Deterministic preprocessing
- Hard validation failures

Components:
- sources: Raw data loaders (Yahoo Finance, FRED)
- loader: Unified data loading dispatcher
- preprocessing: Time-series alignment and cleaning
- validation: Data integrity checks
"""

from portfolio_gnn.data_ingestion.loader import load_market_data
from portfolio_gnn.data_ingestion.preprocessing import preprocess_market_data
from portfolio_gnn.data_ingestion.validation import validate_timeseries

__all__ = [
    "load_market_data",
    "preprocess_market_data",
    "validate_timeseries",
]

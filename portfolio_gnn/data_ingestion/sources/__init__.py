"""
Data source loaders for Portfolio GNN.

Each loader provides raw data access to a specific data source:
- yahoo: Yahoo Finance for equity/ETF OHLCV data
- fred: Federal Reserve Economic Data for macro indicators

All loaders return raw pandas DataFrames with DatetimeIndex.
No preprocessing, caching, or alignment is performed here.
"""

from portfolio_gnn.data_ingestion.sources.yahoo import load_yahoo_prices
from portfolio_gnn.data_ingestion.sources.fred import load_fred_series

__all__ = [
    "load_yahoo_prices",
    "load_fred_series",
]

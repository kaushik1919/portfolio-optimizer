"""
Unified data loader for Portfolio GNN.

This module dispatches to appropriate data source loaders based on
configuration and returns raw, unprocessed data.

Key guarantees:
- All series loaded with the SAME date window
- No alignment or modification of time series
- No preprocessing or feature engineering
- No leakage fixes

Output is a dictionary of raw DataFrames ready for preprocessing.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from portfolio_gnn.data_ingestion.sources.yahoo import (
    load_yahoo_prices,
    YahooDataError,
)
from portfolio_gnn.data_ingestion.sources.fred import (
    load_fred_series,
    FREDDataError,
)

logger = logging.getLogger(__name__)


class DataLoadingError(Exception):
    """Raised when data loading fails."""
    pass


class MarketDataResult:
    """
    Container for raw market data loading results.
    
    Separates asset price data from macro series data for clarity.
    Does NOT perform any alignment or preprocessing.
    
    Attributes:
        asset_prices: Dict mapping ticker -> DataFrame with OHLCV data
        macro_series: Dict mapping series_id -> DataFrame with values
        start_date: Requested start date
        end_date: Requested end date
        config: Original configuration used for loading
    """
    
    def __init__(
        self,
        asset_prices: Dict[str, pd.DataFrame],
        macro_series: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        config: Any,
    ) -> None:
        self.asset_prices = asset_prices
        self.macro_series = macro_series
        self.start_date = start_date
        self.end_date = end_date
        self.config = config
    
    @property
    def num_assets(self) -> int:
        """Number of successfully loaded assets."""
        return len(self.asset_prices)
    
    @property
    def num_macro_series(self) -> int:
        """Number of successfully loaded macro series."""
        return len(self.macro_series)
    
    @property
    def asset_tickers(self) -> List[str]:
        """List of loaded asset tickers."""
        return list(self.asset_prices.keys())
    
    @property
    def macro_ids(self) -> List[str]:
        """List of loaded macro series IDs."""
        return list(self.macro_series.keys())
    
    def __repr__(self) -> str:
        return (
            f"MarketDataResult("
            f"assets={self.num_assets}, "
            f"macro={self.num_macro_series}, "
            f"range={self.start_date} to {self.end_date})"
        )


def load_market_data(config: Any) -> MarketDataResult:
    """
    Load all market data specified in configuration.
    
    Dispatches to appropriate data source loaders based on config.data
    section. All series are loaded using the SAME date window.
    
    Args:
        config: ExperimentConfig with data section containing:
            - asset_universe: List of ticker symbols OR dict with 'tickers' key
            - macro_series: List of FRED series IDs (optional)
            - start_date: Start date string (YYYY-MM-DD)
            - end_date: End date string (YYYY-MM-DD)
    
    Returns:
        MarketDataResult containing raw, unprocessed data
        
    Raises:
        DataLoadingError: If critical data loading fails
        ValueError: If configuration is invalid
        
    Note:
        - No preprocessing is performed
        - No alignment is performed
        - Time series may have different lengths/frequencies
    """
    # Extract data configuration
    data_config = _extract_data_config(config)
    
    # Get date range
    start_date = data_config.get("start_date")
    end_date = data_config.get("end_date")
    
    if not start_date or not end_date:
        raise ValueError(
            "Configuration must specify data.start_date and data.end_date"
        )
    
    # Validate dates
    _validate_date_range(start_date, end_date)
    
    # Get asset universe
    asset_universe = _extract_asset_universe(data_config)
    
    # Get macro series (optional)
    macro_series_ids = _extract_macro_series(data_config)
    
    logger.info("=" * 60)
    logger.info("MARKET DATA LOADING")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Asset universe: {len(asset_universe)} tickers")
    logger.info(f"Macro series: {len(macro_series_ids)} series")
    
    # Load asset prices
    asset_prices: Dict[str, pd.DataFrame] = {}
    if asset_universe:
        try:
            asset_prices = load_yahoo_prices(
                tickers=asset_universe,
                start_date=start_date,
                end_date=end_date,
            )
        except YahooDataError as e:
            raise DataLoadingError(f"Failed to load asset prices: {e}") from e
    
    # Load macro series
    macro_data: Dict[str, pd.DataFrame] = {}
    if macro_series_ids:
        try:
            macro_data = load_fred_series(
                series_ids=macro_series_ids,
                start_date=start_date,
                end_date=end_date,
            )
        except FREDDataError as e:
            # Macro data is optional, log warning but don't fail
            logger.warning(f"Failed to load macro series: {e}")
    
    # Create result container
    result = MarketDataResult(
        asset_prices=asset_prices,
        macro_series=macro_data,
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
    
    logger.info("-" * 60)
    logger.info(f"Loaded {result.num_assets} assets")
    logger.info(f"Loaded {result.num_macro_series} macro series")
    logger.info("=" * 60)
    
    return result


def _extract_data_config(config: Any) -> Dict[str, Any]:
    """
    Extract data configuration section from config object.
    
    Handles both FrozenDict and regular dict access patterns.
    """
    if hasattr(config, "data"):
        data = config.data
    elif hasattr(config, "get"):
        data = config.get("data", {})
    else:
        raise ValueError("Configuration must have a 'data' section")
    
    # Convert FrozenDict to regular dict for easier manipulation
    if hasattr(data, "to_dict"):
        return data.to_dict()
    elif hasattr(data, "_data"):
        # Handle FrozenDict internal access
        return _frozen_to_dict(data)
    else:
        return dict(data) if data else {}


def _frozen_to_dict(obj: Any) -> Any:
    """Recursively convert FrozenDict to regular dict."""
    if hasattr(obj, "_data"):
        return {k: _frozen_to_dict(v) for k, v in obj._data.items()}
    elif isinstance(obj, tuple):
        return [_frozen_to_dict(item) for item in obj]
    return obj


def _extract_asset_universe(data_config: Dict[str, Any]) -> List[str]:
    """
    Extract list of asset tickers from data configuration.
    
    Supports:
        - data.asset_universe: ["AAPL", "MSFT", ...]
        - data.asset_universe.tickers: ["AAPL", "MSFT", ...]
        - data.universe.tickers: ["AAPL", "MSFT", ...]
    """
    # Try direct asset_universe list
    universe = data_config.get("asset_universe")
    if universe is not None:
        if isinstance(universe, (list, tuple)):
            return list(universe)
        elif isinstance(universe, dict) or hasattr(universe, "get"):
            tickers = universe.get("tickers") if hasattr(universe, "get") else universe.get("tickers")
            if tickers:
                return list(tickers)
    
    # Try universe.tickers
    universe = data_config.get("universe")
    if universe is not None:
        if isinstance(universe, dict) or hasattr(universe, "get"):
            tickers = universe.get("tickers") if hasattr(universe, "get") else None
            if tickers:
                return list(tickers)
    
    logger.warning("No asset universe specified in configuration")
    return []


def _extract_macro_series(data_config: Dict[str, Any]) -> List[str]:
    """
    Extract list of macro series IDs from data configuration.
    
    Supports:
        - data.macro_series: ["DGS10", "VIXCLS", ...]
        - data.macro.series_ids: ["DGS10", "VIXCLS", ...]
    """
    # Try direct macro_series list
    macro = data_config.get("macro_series")
    if macro is not None:
        if isinstance(macro, (list, tuple)):
            return list(macro)
    
    # Try macro.series_ids
    macro = data_config.get("macro")
    if macro is not None:
        if isinstance(macro, dict) or hasattr(macro, "get"):
            series_ids = macro.get("series_ids") if hasattr(macro, "get") else None
            if series_ids:
                return list(series_ids)
    
    logger.debug("No macro series specified in configuration")
    return []


def _validate_date_range(start_date: str, end_date: str) -> None:
    """
    Validate that date range is sensible.
    
    Raises:
        ValueError: If dates are invalid or range is backwards
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(
            f"Invalid date format. Expected YYYY-MM-DD. Error: {e}"
        )
    
    if start_dt >= end_dt:
        raise ValueError(
            f"start_date ({start_date}) must be before end_date ({end_date})"
        )
    
    # Check for future dates
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if end_dt > today:
        logger.warning(
            f"end_date ({end_date}) is in the future. "
            f"Data may be incomplete."
        )

"""
Yahoo Finance data loader for Portfolio GNN.

Loads OHLCV-adjusted price data for equities and ETFs using yfinance.

This module provides RAW data only:
- No caching
- No resampling
- No forward-filling
- No silent column renaming

All data is returned as-is from the source with explicit date filtering.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import List, Optional, Union

import pandas as pd

# Import yfinance at module level for testability
try:
    import yfinance as yf
except ImportError:
    yf = None  # Will be checked at runtime

logger = logging.getLogger(__name__)


class YahooDataError(Exception):
    """Raised when Yahoo Finance data loading fails."""
    pass


def load_yahoo_prices(
    tickers: List[str],
    start_date: Union[str, date, datetime],
    end_date: Union[str, date, datetime],
    price_column: str = "Adj Close",
) -> dict[str, pd.DataFrame]:
    """
    Load OHLCV-adjusted price data from Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols (e.g., ["AAPL", "MSFT", "SPY"])
        start_date: Start date for data retrieval (inclusive)
        end_date: End date for data retrieval (inclusive)
        price_column: Which price column to extract. Default "Adj Close".
            Options: "Open", "High", "Low", "Close", "Adj Close", "Volume"
    
    Returns:
        Dictionary mapping ticker -> DataFrame with columns:
            - Date index (DatetimeIndex)
            - Open, High, Low, Close, Adj Close, Volume
        
    Raises:
        YahooDataError: If data loading fails or ticker is invalid
        ValueError: If parameters are invalid
        
    Note:
        - Daily frequency ONLY
        - No data cleaning or filling is performed
        - Missing data is preserved as NaN
    """
    # Check if yfinance is available
    if yf is None:
        raise YahooDataError(
            "yfinance is required for Yahoo data loading. "
            "Install with: pip install yfinance"
        )
    
    # Validate inputs
    if not tickers:
        raise ValueError("tickers list cannot be empty")
    
    if not isinstance(tickers, (list, tuple)):
        raise ValueError(f"tickers must be a list, got {type(tickers).__name__}")
    
    # Normalize dates to strings for yfinance
    start_str = _normalize_date(start_date)
    end_str = _normalize_date(end_date)
    
    # Validate date order
    if start_str >= end_str:
        raise ValueError(
            f"start_date ({start_str}) must be before end_date ({end_str})"
        )
    
    logger.info(f"Loading Yahoo Finance data for {len(tickers)} tickers")
    logger.info(f"Date range: {start_str} to {end_str}")
    
    results: dict[str, pd.DataFrame] = {}
    failed_tickers: List[str] = []
    
    for ticker in tickers:
        ticker = ticker.strip().upper()
        
        try:
            df = _load_single_ticker(ticker, start_str, end_str)
            
            if df is None or df.empty:
                logger.warning(f"No data returned for ticker: {ticker}")
                failed_tickers.append(ticker)
                continue
            
            # Validate we got the expected columns
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            # Handle both "Adj Close" and "Adj_Close" formats
            has_adj_close = "Adj Close" in df.columns or "Adj_Close" in df.columns
            
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols or not has_adj_close:
                logger.warning(
                    f"Ticker {ticker} missing expected columns. "
                    f"Got: {list(df.columns)}"
                )
                failed_tickers.append(ticker)
                continue
            
            # Standardize column name if needed
            if "Adj_Close" in df.columns and "Adj Close" not in df.columns:
                df = df.rename(columns={"Adj_Close": "Adj Close"})
            
            results[ticker] = df
            logger.debug(f"Loaded {ticker}: {len(df)} rows")
            
        except Exception as e:
            logger.warning(f"Failed to load ticker {ticker}: {e}")
            failed_tickers.append(ticker)
    
    if failed_tickers:
        logger.warning(f"Failed tickers: {failed_tickers}")
    
    if not results:
        raise YahooDataError(
            f"No data loaded for any ticker. Failed: {failed_tickers}. "
            f"Check ticker symbols and date range."
        )
    
    logger.info(f"Successfully loaded {len(results)}/{len(tickers)} tickers")
    
    return results


def _load_single_ticker(
    ticker: str,
    start_str: str,
    end_str: str,
) -> Optional[pd.DataFrame]:
    """
    Load data for a single ticker.
    
    Args:
        ticker: Ticker symbol
        start_str: Start date string (YYYY-MM-DD)
        end_str: End date string (YYYY-MM-DD)
        
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    # Create ticker object
    yf_ticker = yf.Ticker(ticker)
    
    # Download historical data
    # auto_adjust=False to get raw Adj Close
    # actions=False to skip dividends/splits columns
    df = yf_ticker.history(
        start=start_str,
        end=end_str,
        interval="1d",
        auto_adjust=False,
        actions=False,
    )
    
    if df is None or df.empty:
        return None
    
    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Normalize index to date only (remove time component)
    df.index = df.index.normalize()
    df.index.name = "Date"
    
    # Remove timezone if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    return df


def _normalize_date(d: Union[str, date, datetime]) -> str:
    """
    Normalize a date input to YYYY-MM-DD string format.
    
    Args:
        d: Date as string, date, or datetime
        
    Returns:
        Date string in YYYY-MM-DD format
        
    Raises:
        ValueError: If date format is invalid
    """
    if isinstance(d, str):
        # Validate string format
        try:
            parsed = datetime.strptime(d, "%Y-%m-%d")
            return d
        except ValueError:
            raise ValueError(
                f"Invalid date string format: {d}. Expected YYYY-MM-DD."
            )
    elif isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    elif isinstance(d, date):
        return d.strftime("%Y-%m-%d")
    else:
        raise ValueError(
            f"Invalid date type: {type(d).__name__}. "
            f"Expected str, date, or datetime."
        )


def get_available_tickers() -> List[str]:
    """
    Return a list of commonly available ticker symbols for testing.
    
    These are major US stocks/ETFs that should have reliable data.
    
    Returns:
        List of ticker symbols
    """
    return [
        # Major indices ETFs
        "SPY",  # S&P 500
        "QQQ",  # Nasdaq 100
        "IWM",  # Russell 2000
        "DIA",  # Dow Jones
        # Major tech stocks
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        # Major financials
        "JPM",
        "BAC",
        "GS",
        # Major healthcare
        "JNJ",
        "UNH",
        "PFE",
    ]

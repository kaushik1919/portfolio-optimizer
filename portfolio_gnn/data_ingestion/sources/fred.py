"""
FRED (Federal Reserve Economic Data) loader for Portfolio GNN.

Loads macroeconomic time series from the FRED API.

This module provides RAW data only:
- No caching
- No resampling
- No forward-filling
- No frequency conversion

All data is returned as-is from the source with native FRED frequency.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class FREDDataError(Exception):
    """Raised when FRED data loading fails."""
    pass


# Common FRED series IDs for macro factors
COMMON_FRED_SERIES = {
    # Interest rates
    "DFF": "Federal Funds Effective Rate (Daily)",
    "DGS10": "10-Year Treasury Constant Maturity Rate (Daily)",
    "DGS2": "2-Year Treasury Constant Maturity Rate (Daily)",
    "DGS3MO": "3-Month Treasury Constant Maturity Rate (Daily)",
    # Spreads
    "T10Y2Y": "10-Year Treasury Minus 2-Year Treasury (Daily)",
    "T10Y3M": "10-Year Treasury Minus 3-Month Treasury (Daily)",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Index OAS (Daily)",
    # Volatility
    "VIXCLS": "CBOE Volatility Index VIX (Daily)",
    # Economic indicators
    "UNRATE": "Unemployment Rate (Monthly)",
    "CPIAUCSL": "Consumer Price Index (Monthly)",
    "INDPRO": "Industrial Production Index (Monthly)",
    "UMCSENT": "University of Michigan Consumer Sentiment (Monthly)",
}


def load_fred_series(
    series_ids: List[str],
    start_date: Union[str, date, datetime],
    end_date: Union[str, date, datetime],
    api_key: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load macroeconomic time series from FRED.
    
    Args:
        series_ids: List of FRED series IDs (e.g., ["DGS10", "VIXCLS"])
        start_date: Start date for data retrieval (inclusive)
        end_date: End date for data retrieval (inclusive)
        api_key: Optional FRED API key. If not provided, uses
            FRED_API_KEY environment variable or falls back to
            pandas_datareader's default handling.
    
    Returns:
        Dictionary mapping series_id -> DataFrame with columns:
            - Date index (DatetimeIndex)
            - Value column named after series_id
        
    Raises:
        FREDDataError: If data loading fails or series is invalid
        ValueError: If parameters are invalid
        
    Note:
        - Native FRED frequency is preserved (daily, monthly, etc.)
        - No resampling or frequency conversion
        - Missing data is preserved as NaN
    """
    # Validate inputs
    if not series_ids:
        raise ValueError("series_ids list cannot be empty")
    
    if not isinstance(series_ids, (list, tuple)):
        raise ValueError(f"series_ids must be a list, got {type(series_ids).__name__}")
    
    # Normalize dates
    start_str = _normalize_date(start_date)
    end_str = _normalize_date(end_date)
    
    # Validate date order
    if start_str >= end_str:
        raise ValueError(
            f"start_date ({start_str}) must be before end_date ({end_str})"
        )
    
    logger.info(f"Loading FRED data for {len(series_ids)} series")
    logger.info(f"Date range: {start_str} to {end_str}")
    
    # Check for API key
    if api_key is None:
        import os
        api_key = os.environ.get("FRED_API_KEY")
    
    results: Dict[str, pd.DataFrame] = {}
    failed_series: List[str] = []
    
    for series_id in series_ids:
        series_id = series_id.strip().upper()
        
        try:
            df = _load_single_series(
                series_id, 
                start_str, 
                end_str, 
                api_key
            )
            
            if df is None or df.empty:
                logger.warning(f"No data returned for series: {series_id}")
                failed_series.append(series_id)
                continue
            
            results[series_id] = df
            logger.debug(f"Loaded {series_id}: {len(df)} rows")
            
        except Exception as e:
            logger.warning(f"Failed to load series {series_id}: {e}")
            failed_series.append(series_id)
    
    if failed_series:
        logger.warning(f"Failed series: {failed_series}")
    
    if not results:
        raise FREDDataError(
            f"No data loaded for any series. Failed: {failed_series}. "
            f"Check series IDs and date range. "
            f"You may need to set FRED_API_KEY environment variable."
        )
    
    logger.info(f"Successfully loaded {len(results)}/{len(series_ids)} series")
    
    return results


def _load_single_series(
    series_id: str,
    start_str: str,
    end_str: str,
    api_key: Optional[str],
) -> Optional[pd.DataFrame]:
    """
    Load data for a single FRED series.
    
    Uses pandas_datareader if available, falls back to fredapi.
    
    Args:
        series_id: FRED series ID
        start_str: Start date string (YYYY-MM-DD)
        end_str: End date string (YYYY-MM-DD)
        api_key: Optional FRED API key
        
    Returns:
        DataFrame with value data or None if failed
    """
    df = None
    
    # Try pandas_datareader first (simpler, no API key required for basic use)
    try:
        df = _load_via_pandas_datareader(series_id, start_str, end_str)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        logger.debug(f"pandas_datareader failed for {series_id}: {e}")
    
    # Fall back to fredapi if available
    try:
        df = _load_via_fredapi(series_id, start_str, end_str, api_key)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        logger.debug(f"fredapi failed for {series_id}: {e}")
    
    return df


def _load_via_pandas_datareader(
    series_id: str,
    start_str: str,
    end_str: str,
) -> Optional[pd.DataFrame]:
    """Load FRED data using pandas_datareader."""
    try:
        from pandas_datareader import data as pdr
    except ImportError:
        raise ImportError(
            "pandas_datareader is required for FRED data loading. "
            "Install with: pip install pandas-datareader"
        )
    
    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_str, "%Y-%m-%d")
    
    # pandas_datareader returns a Series or DataFrame
    data = pdr.DataReader(series_id, "fred", start_dt, end_dt)
    
    if isinstance(data, pd.Series):
        df = data.to_frame(name=series_id)
    else:
        df = data
        # Rename column to series_id if it's different
        if len(df.columns) == 1 and df.columns[0] != series_id:
            df = df.rename(columns={df.columns[0]: series_id})
    
    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Normalize index to date only
    df.index = df.index.normalize()
    df.index.name = "Date"
    
    # Remove timezone if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    return df


def _load_via_fredapi(
    series_id: str,
    start_str: str,
    end_str: str,
    api_key: Optional[str],
) -> Optional[pd.DataFrame]:
    """Load FRED data using fredapi."""
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError(
            "fredapi is required for FRED data loading. "
            "Install with: pip install fredapi"
        )
    
    if api_key is None:
        raise FREDDataError(
            "FRED API key required for fredapi. "
            "Set FRED_API_KEY environment variable or pass api_key parameter."
        )
    
    fred = Fred(api_key=api_key)
    
    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_str, "%Y-%m-%d")
    
    series = fred.get_series(
        series_id,
        observation_start=start_dt,
        observation_end=end_dt,
    )
    
    if series is None or series.empty:
        return None
    
    df = series.to_frame(name=series_id)
    
    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Normalize index to date only
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


def get_common_macro_series() -> Dict[str, str]:
    """
    Return a dictionary of commonly used FRED series IDs and descriptions.
    
    Returns:
        Dictionary mapping series_id -> description
    """
    return COMMON_FRED_SERIES.copy()

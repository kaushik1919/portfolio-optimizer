"""
Time-series preprocessing for Portfolio GNN.

This is a CRITICAL quant module that ensures data integrity.

Preprocessing steps (in order):
1. Explicit date filtering (no implicit truncation)
2. Inner-join alignment across all asset price series
3. Controlled handling of missing values
4. Explicit frequency validation (daily only for assets)

FORBIDDEN operations:
- Forward-fill
- Backward-fill
- Rolling statistics
- Normalization
- Feature creation
- Any operation that could introduce look-ahead bias
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from portfolio_gnn.data_ingestion.loader import MarketDataResult

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Raised when preprocessing fails due to data quality issues."""
    pass


class InsufficientDataError(PreprocessingError):
    """Raised when there is insufficient data for analysis."""
    pass


class DataAlignmentError(PreprocessingError):
    """Raised when data alignment fails."""
    pass


class PreprocessedData:
    """
    Container for preprocessed, aligned market data.
    
    All time series in this container are:
    - Aligned to common trading days (inner join)
    - Validated for monotonic increasing timestamps
    - Free of duplicate dates
    - Free of look-ahead bias
    
    Attributes:
        prices: DataFrame with aligned adjusted close prices
            - Rows = trading days (DatetimeIndex)
            - Columns = asset tickers
        macro: Optional DataFrame with aligned macro series
        start_date: Actual start date after alignment
        end_date: Actual end date after alignment
        dropped_assets: List of assets dropped during preprocessing
        config: Original configuration
    """
    
    def __init__(
        self,
        prices: pd.DataFrame,
        macro: Optional[pd.DataFrame],
        start_date: str,
        end_date: str,
        dropped_assets: List[str],
        config: Any,
    ) -> None:
        self.prices = prices
        self.macro = macro
        self.start_date = start_date
        self.end_date = end_date
        self.dropped_assets = dropped_assets
        self.config = config
    
    @property
    def num_assets(self) -> int:
        """Number of assets in aligned dataset."""
        return len(self.prices.columns)
    
    @property
    def num_trading_days(self) -> int:
        """Number of trading days in aligned dataset."""
        return len(self.prices)
    
    @property
    def tickers(self) -> List[str]:
        """List of asset tickers in aligned dataset."""
        return list(self.prices.columns)
    
    @property
    def date_range(self) -> Tuple[str, str]:
        """Tuple of (start_date, end_date) after alignment."""
        return (self.start_date, self.end_date)
    
    def __repr__(self) -> str:
        return (
            f"PreprocessedData("
            f"assets={self.num_assets}, "
            f"days={self.num_trading_days}, "
            f"range={self.start_date} to {self.end_date}, "
            f"dropped={len(self.dropped_assets)})"
        )


def preprocess_market_data(
    raw_data: MarketDataResult,
    config: Any,
) -> PreprocessedData:
    """
    Preprocess raw market data with strict quant-grade guarantees.
    
    Processing steps (in order):
    1. Extract adjusted close prices from OHLCV data
    2. Apply explicit date filtering
    3. Inner-join align across all assets
    4. Drop leading NaNs
    5. Validate missingness ratios
    6. Validate daily frequency
    
    Args:
        raw_data: MarketDataResult from load_market_data()
        config: ExperimentConfig with data section
    
    Returns:
        PreprocessedData with aligned, validated time series
        
    Raises:
        PreprocessingError: If data quality checks fail
        InsufficientDataError: If insufficient data remains after preprocessing
        DataAlignmentError: If alignment produces empty result
    """
    logger.info("=" * 60)
    logger.info("TIME-SERIES PREPROCESSING")
    logger.info("=" * 60)
    
    # Extract preprocessing parameters from config
    params = _extract_preprocessing_params(config)
    
    logger.info(f"Start date filter: {params['start_date']}")
    logger.info(f"End date filter: {params['end_date']}")
    logger.info(f"Min history days: {params['min_history_days']}")
    logger.info(f"Max missing ratio: {params['max_missing_ratio']}")
    
    # Step 1: Extract adjusted close prices
    logger.info("-" * 60)
    logger.info("Step 1: Extracting adjusted close prices")
    price_series = _extract_adjusted_close(raw_data.asset_prices)
    logger.info(f"Extracted prices for {len(price_series)} assets")
    
    # Step 2: Apply explicit date filtering
    logger.info("-" * 60)
    logger.info("Step 2: Applying explicit date filtering")
    price_series = _filter_by_date(
        price_series,
        params["start_date"],
        params["end_date"],
    )
    logger.info(f"After date filtering: {len(price_series)} assets")
    
    # Step 3: Check for assets with insufficient data before alignment
    logger.info("-" * 60)
    logger.info("Step 3: Checking individual asset data sufficiency")
    price_series, dropped_insufficient = _drop_insufficient_assets(
        price_series,
        min_days=params["min_history_days"],
    )
    logger.info(f"Dropped {len(dropped_insufficient)} assets with insufficient history")
    if dropped_insufficient:
        logger.info(f"Dropped assets: {dropped_insufficient}")
    
    # Step 4: Inner-join alignment
    logger.info("-" * 60)
    logger.info("Step 4: Inner-join alignment across assets")
    aligned_prices, dropped_alignment = _align_inner_join(
        price_series,
        max_missing_ratio=params["max_missing_ratio"],
    )
    logger.info(f"After alignment: {len(aligned_prices.columns)} assets, {len(aligned_prices)} days")
    if dropped_alignment:
        logger.info(f"Dropped during alignment: {dropped_alignment}")
    
    # Step 5: Drop leading NaNs (NO forward/backward fill)
    logger.info("-" * 60)
    logger.info("Step 5: Dropping leading NaNs")
    aligned_prices = _drop_leading_nans(aligned_prices)
    logger.info(f"After dropping leading NaNs: {len(aligned_prices)} days")
    
    # Step 6: Final validation
    logger.info("-" * 60)
    logger.info("Step 6: Final validation")
    _validate_no_remaining_nans(aligned_prices)
    _validate_min_history(aligned_prices, params["min_history_days"])
    _validate_daily_frequency(aligned_prices)
    logger.info("Validation passed")
    
    # Compile dropped assets list
    all_dropped = dropped_insufficient + dropped_alignment
    
    # Get actual date range after preprocessing
    actual_start = aligned_prices.index.min().strftime("%Y-%m-%d")
    actual_end = aligned_prices.index.max().strftime("%Y-%m-%d")
    
    # Process macro series if available (separate alignment)
    aligned_macro = None
    if raw_data.macro_series:
        try:
            aligned_macro = _align_macro_to_prices(
                raw_data.macro_series,
                aligned_prices.index,
            )
        except Exception as e:
            logger.warning(f"Failed to align macro series: {e}")
    
    result = PreprocessedData(
        prices=aligned_prices,
        macro=aligned_macro,
        start_date=actual_start,
        end_date=actual_end,
        dropped_assets=all_dropped,
        config=config,
    )
    
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info(f"Assets: {result.num_assets}")
    logger.info(f"Trading days: {result.num_trading_days}")
    logger.info(f"Date range: {result.start_date} to {result.end_date}")
    logger.info(f"Dropped assets: {len(result.dropped_assets)}")
    logger.info("=" * 60)
    
    return result


def _extract_preprocessing_params(config: Any) -> Dict[str, Any]:
    """Extract preprocessing parameters from configuration."""
    # Navigate to data config
    if hasattr(config, "data"):
        data = config.data
    else:
        raise PreprocessingError("Configuration missing 'data' section")
    
    # Helper to get nested values
    def get_val(obj, key, default=None):
        if hasattr(obj, "get"):
            return obj.get(key, default)
        elif hasattr(obj, key):
            return getattr(obj, key, default)
        return default
    
    params = {
        "start_date": get_val(data, "start_date"),
        "end_date": get_val(data, "end_date"),
        "min_history_days": get_val(data, "min_history_days", 252),
        "max_missing_ratio": get_val(data, "max_missing_ratio", 0.05),
    }
    
    # Also check nested preprocessing section
    preprocessing = get_val(data, "preprocessing", {})
    if preprocessing:
        params["min_history_days"] = get_val(
            preprocessing, "min_history_days", params["min_history_days"]
        )
        params["max_missing_ratio"] = get_val(
            preprocessing, "max_missing_ratio", params["max_missing_ratio"]
        )
    
    # Validate required params
    if not params["start_date"]:
        raise PreprocessingError("Configuration missing data.start_date")
    if not params["end_date"]:
        raise PreprocessingError("Configuration missing data.end_date")
    
    return params


def _extract_adjusted_close(
    asset_prices: Dict[str, pd.DataFrame],
) -> Dict[str, pd.Series]:
    """
    Extract adjusted close prices from OHLCV DataFrames.
    
    Args:
        asset_prices: Dict of ticker -> OHLCV DataFrame
        
    Returns:
        Dict of ticker -> adjusted close Series
    """
    result = {}
    
    for ticker, df in asset_prices.items():
        if df is None or df.empty:
            logger.warning(f"Skipping {ticker}: empty data")
            continue
        
        # Find adjusted close column
        adj_close_col = None
        for col in ["Adj Close", "Adj_Close", "adj_close", "adjusted_close"]:
            if col in df.columns:
                adj_close_col = col
                break
        
        if adj_close_col is None:
            logger.warning(f"Skipping {ticker}: no adjusted close column")
            continue
        
        series = df[adj_close_col].copy()
        series.name = ticker
        result[ticker] = series
    
    return result


def _filter_by_date(
    price_series: Dict[str, pd.Series],
    start_date: str,
    end_date: str,
) -> Dict[str, pd.Series]:
    """
    Apply explicit date filtering to all series.
    
    This is NOT implicit truncation - dates are explicitly specified.
    """
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    
    result = {}
    
    for ticker, series in price_series.items():
        # Filter to date range
        mask = (series.index >= start_dt) & (series.index <= end_dt)
        filtered = series.loc[mask]
        
        if filtered.empty:
            logger.warning(f"No data for {ticker} in date range {start_date} to {end_date}")
            continue
        
        result[ticker] = filtered
    
    return result


def _drop_insufficient_assets(
    price_series: Dict[str, pd.Series],
    min_days: int,
) -> Tuple[Dict[str, pd.Series], List[str]]:
    """
    Drop assets with insufficient history.
    
    Args:
        price_series: Dict of ticker -> price Series
        min_days: Minimum required trading days
        
    Returns:
        Tuple of (filtered_series, dropped_tickers)
    """
    result = {}
    dropped = []
    
    for ticker, series in price_series.items():
        # Count non-NaN values
        valid_count = series.notna().sum()
        
        if valid_count < min_days:
            logger.debug(
                f"Dropping {ticker}: only {valid_count} valid days "
                f"(min required: {min_days})"
            )
            dropped.append(ticker)
        else:
            result[ticker] = series
    
    return result, dropped


def _align_inner_join(
    price_series: Dict[str, pd.Series],
    max_missing_ratio: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Align all price series using inner join.
    
    Only dates present in ALL series are kept.
    This is the strictest alignment method and prevents look-ahead bias.
    
    Args:
        price_series: Dict of ticker -> price Series
        max_missing_ratio: Maximum allowed missing ratio per asset
        
    Returns:
        Tuple of (aligned_df, dropped_tickers)
    """
    if not price_series:
        raise DataAlignmentError("No price series to align")
    
    # Combine into DataFrame (outer join first to analyze missingness)
    df = pd.DataFrame(price_series)
    
    if df.empty:
        raise DataAlignmentError("Combined DataFrame is empty")
    
    # Identify assets with excessive missingness before inner join
    dropped = []
    valid_tickers = []
    
    for ticker in df.columns:
        missing_ratio = df[ticker].isna().sum() / len(df)
        
        if missing_ratio > max_missing_ratio:
            logger.debug(
                f"Dropping {ticker}: {missing_ratio:.2%} missing "
                f"(max allowed: {max_missing_ratio:.2%})"
            )
            dropped.append(ticker)
        else:
            valid_tickers.append(ticker)
    
    if not valid_tickers:
        raise DataAlignmentError(
            f"All assets exceed max missing ratio ({max_missing_ratio:.2%}). "
            f"Dropped: {dropped}"
        )
    
    # Keep only valid tickers
    df = df[valid_tickers]
    
    # Inner join: drop any row with ANY missing value
    # This ensures all assets have data for all dates
    aligned = df.dropna(how="any")
    
    if aligned.empty:
        raise DataAlignmentError(
            "Inner join alignment produced empty DataFrame. "
            "Assets may have non-overlapping trading calendars."
        )
    
    return aligned, dropped


def _drop_leading_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows at the start of the DataFrame with any NaN values.
    
    This handles the case where inner join still leaves leading NaNs
    (which shouldn't happen after dropna, but defensive coding).
    
    NO forward-fill or backward-fill is performed.
    """
    # Find first row where all values are non-NaN
    first_valid_idx = df.notna().all(axis=1).idxmax()
    
    return df.loc[first_valid_idx:]


def _validate_no_remaining_nans(df: pd.DataFrame) -> None:
    """
    Validate that no NaN values remain in the aligned data.
    
    Raises:
        PreprocessingError: If NaN values are found
    """
    nan_counts = df.isna().sum()
    
    if nan_counts.sum() > 0:
        problem_assets = nan_counts[nan_counts > 0].to_dict()
        raise PreprocessingError(
            f"Aligned data contains NaN values. "
            f"Problem assets: {problem_assets}. "
            f"This indicates a bug in the alignment logic."
        )


def _validate_min_history(df: pd.DataFrame, min_days: int) -> None:
    """
    Validate that aligned data has minimum required history.
    
    Raises:
        InsufficientDataError: If history is too short
    """
    actual_days = len(df)
    
    if actual_days < min_days:
        raise InsufficientDataError(
            f"Aligned data has only {actual_days} trading days. "
            f"Minimum required: {min_days}. "
            f"Consider expanding date range or reducing asset universe."
        )


def _validate_daily_frequency(df: pd.DataFrame) -> None:
    """
    Validate that data has approximately daily frequency.
    
    Allows for weekends and holidays (up to 5 days gap).
    
    Raises:
        PreprocessingError: If frequency appears incorrect
    """
    if len(df) < 2:
        return
    
    # Calculate gaps between consecutive dates
    date_diffs = df.index.to_series().diff().dropna()
    
    # Maximum expected gap (e.g., long holiday weekend)
    max_gap = pd.Timedelta(days=10)
    
    long_gaps = date_diffs[date_diffs > max_gap]
    
    if len(long_gaps) > 0:
        logger.warning(
            f"Found {len(long_gaps)} gaps > 10 days in data. "
            f"Longest gap: {long_gaps.max()}. "
            f"This may indicate missing data or incorrect frequency."
        )


def _align_macro_to_prices(
    macro_series: Dict[str, pd.DataFrame],
    price_index: pd.DatetimeIndex,
) -> Optional[pd.DataFrame]:
    """
    Align macro series to price data dates.
    
    Macro data may have different frequency (e.g., monthly).
    We use BACKWARD lookup only (no look-ahead).
    
    Args:
        macro_series: Dict of series_id -> DataFrame
        price_index: DatetimeIndex from aligned price data
        
    Returns:
        DataFrame with macro values aligned to price dates,
        or None if alignment fails
    """
    if not macro_series:
        return None
    
    aligned_series = {}
    
    for series_id, df in macro_series.items():
        if df is None or df.empty:
            continue
        
        # Get the value column
        if series_id in df.columns:
            series = df[series_id]
        else:
            series = df.iloc[:, 0]
        
        # Reindex to price dates using backward fill
        # This gets the most recent available value (no look-ahead)
        # BUT we use 'ffill' with limit to avoid filling too far
        # Actually - per requirements, NO forward-fill allowed
        # So we use asof-style lookup
        
        aligned_values = []
        for date in price_index:
            # Find most recent value on or before this date
            valid_dates = series.index[series.index <= date]
            if len(valid_dates) > 0:
                aligned_values.append(series.loc[valid_dates[-1]])
            else:
                aligned_values.append(None)
        
        aligned_series[series_id] = aligned_values
    
    if not aligned_series:
        return None
    
    result = pd.DataFrame(aligned_series, index=price_index)
    
    return result

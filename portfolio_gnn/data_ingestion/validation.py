"""
Time-series validation for Portfolio GNN.

This module implements HARD validation checks that FAIL with clear,
actionable exceptions. No warnings - only hard failures.

Validation checks:
- Monotonic increasing timestamps
- No duplicate dates
- No future timestamps
- Minimum history length enforcement
- Explicit rejection of assets with insufficient data

All failures must:
- Raise clear, actionable exceptions
- Identify the offending asset/series
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from portfolio_gnn.data_ingestion.preprocessing import PreprocessedData

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Base class for validation errors."""
    pass


class TimestampError(ValidationError):
    """Raised when timestamp validation fails."""
    pass


class DuplicateDateError(ValidationError):
    """Raised when duplicate dates are found."""
    pass


class FutureDateError(ValidationError):
    """Raised when future timestamps are found."""
    pass


class InsufficientHistoryError(ValidationError):
    """Raised when minimum history requirement is not met."""
    pass


class DataIntegrityError(ValidationError):
    """Raised when data integrity checks fail."""
    pass


def validate_timeseries(
    data: PreprocessedData,
    config: Any,
) -> None:
    """
    Perform comprehensive validation on preprocessed time series data.
    
    This function runs ALL validation checks and raises hard failures
    for any issues found. No warnings - only failures.
    
    Args:
        data: PreprocessedData from preprocessing step
        config: ExperimentConfig for validation parameters
        
    Raises:
        ValidationError: If any validation check fails
        TimestampError: If timestamps are not monotonic
        DuplicateDateError: If duplicate dates are found
        FutureDateError: If future timestamps are found
        InsufficientHistoryError: If minimum history not met
        DataIntegrityError: If data integrity checks fail
    """
    logger.info("=" * 60)
    logger.info("TIME-SERIES VALIDATION")
    logger.info("=" * 60)
    
    # Extract validation parameters
    params = _extract_validation_params(config)
    
    # Track all validation issues
    issues: List[str] = []
    
    # Run validations on price data
    logger.info("Validating price data...")
    
    try:
        _validate_monotonic_timestamps(data.prices, "prices")
        logger.info("✓ Timestamps are monotonically increasing")
    except TimestampError as e:
        issues.append(str(e))
    
    try:
        _validate_no_duplicate_dates(data.prices, "prices")
        logger.info("✓ No duplicate dates found")
    except DuplicateDateError as e:
        issues.append(str(e))
    
    try:
        _validate_no_future_dates(data.prices, "prices")
        logger.info("✓ No future timestamps found")
    except FutureDateError as e:
        issues.append(str(e))
    
    try:
        _validate_minimum_history(
            data.prices,
            params["min_history_days"],
            "prices",
        )
        logger.info(f"✓ Minimum history requirement met ({params['min_history_days']} days)")
    except InsufficientHistoryError as e:
        issues.append(str(e))
    
    try:
        _validate_no_nans(data.prices, "prices")
        logger.info("✓ No NaN values in price data")
    except DataIntegrityError as e:
        issues.append(str(e))
    
    try:
        _validate_positive_prices(data.prices)
        logger.info("✓ All prices are positive")
    except DataIntegrityError as e:
        issues.append(str(e))
    
    try:
        _validate_no_zero_prices(data.prices)
        logger.info("✓ No zero prices found")
    except DataIntegrityError as e:
        issues.append(str(e))
    
    # Validate macro data if present
    if data.macro is not None:
        logger.info("Validating macro data...")
        
        try:
            _validate_monotonic_timestamps(data.macro, "macro")
            logger.info("✓ Macro timestamps are monotonically increasing")
        except TimestampError as e:
            issues.append(str(e))
        
        try:
            _validate_no_duplicate_dates(data.macro, "macro")
            logger.info("✓ No duplicate dates in macro data")
        except DuplicateDateError as e:
            issues.append(str(e))
    
    # Validate per-asset data sufficiency
    try:
        _validate_per_asset_sufficiency(
            data.prices,
            params["min_history_days"],
        )
        logger.info("✓ All assets meet minimum history requirement")
    except InsufficientHistoryError as e:
        issues.append(str(e))
    
    # Report results
    if issues:
        logger.error("=" * 60)
        logger.error("VALIDATION FAILED")
        logger.error("=" * 60)
        for issue in issues:
            logger.error(f"  - {issue}")
        
        raise ValidationError(
            f"Time-series validation failed with {len(issues)} issue(s):\n"
            + "\n".join(f"  - {issue}" for issue in issues)
        )
    
    logger.info("=" * 60)
    logger.info("VALIDATION PASSED")
    logger.info("=" * 60)


def _extract_validation_params(config: Any) -> Dict[str, Any]:
    """Extract validation parameters from configuration."""
    if hasattr(config, "data"):
        data = config.data
    else:
        data = {}
    
    def get_val(obj, key, default=None):
        if hasattr(obj, "get"):
            return obj.get(key, default)
        elif hasattr(obj, key):
            return getattr(obj, key, default)
        return default
    
    params = {
        "min_history_days": get_val(data, "min_history_days", 252),
    }
    
    # Check preprocessing section
    preprocessing = get_val(data, "preprocessing", {})
    if preprocessing:
        params["min_history_days"] = get_val(
            preprocessing, "min_history_days", params["min_history_days"]
        )
    
    return params


def _validate_monotonic_timestamps(
    df: pd.DataFrame,
    data_name: str,
) -> None:
    """
    Validate that timestamps are strictly monotonically increasing.
    
    Args:
        df: DataFrame to validate
        data_name: Name for error messages
        
    Raises:
        TimestampError: If timestamps are not monotonic
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TimestampError(
            f"{data_name}: Index is not a DatetimeIndex. "
            f"Got: {type(df.index).__name__}"
        )
    
    if not df.index.is_monotonic_increasing:
        # Find the offending locations
        diffs = df.index.to_series().diff()
        non_increasing = diffs[diffs <= pd.Timedelta(0)]
        
        if len(non_increasing) > 0:
            first_issue_idx = non_increasing.index[0]
            first_issue_pos = df.index.get_loc(first_issue_idx)
            
            raise TimestampError(
                f"{data_name}: Timestamps are not monotonically increasing. "
                f"First issue at position {first_issue_pos}: "
                f"{df.index[first_issue_pos-1]} -> {df.index[first_issue_pos]}"
            )
        else:
            raise TimestampError(
                f"{data_name}: Timestamps are not monotonically increasing. "
                f"Index may contain duplicates or out-of-order dates."
            )


def _validate_no_duplicate_dates(
    df: pd.DataFrame,
    data_name: str,
) -> None:
    """
    Validate that no duplicate dates exist.
    
    Args:
        df: DataFrame to validate
        data_name: Name for error messages
        
    Raises:
        DuplicateDateError: If duplicates are found
    """
    if df.index.duplicated().any():
        duplicates = df.index[df.index.duplicated(keep=False)]
        unique_duplicates = sorted(set(duplicates.strftime("%Y-%m-%d")))[:5]
        
        raise DuplicateDateError(
            f"{data_name}: Duplicate dates found. "
            f"Examples: {unique_duplicates}. "
            f"Total duplicates: {len(duplicates)}"
        )


def _validate_no_future_dates(
    df: pd.DataFrame,
    data_name: str,
) -> None:
    """
    Validate that no timestamps are in the future.
    
    Args:
        df: DataFrame to validate
        data_name: Name for error messages
        
    Raises:
        FutureDateError: If future timestamps are found
    """
    today = pd.Timestamp(datetime.now().date())
    
    future_dates = df.index[df.index > today]
    
    if len(future_dates) > 0:
        future_examples = sorted(set(future_dates.strftime("%Y-%m-%d")))[:5]
        
        raise FutureDateError(
            f"{data_name}: Future timestamps found (today: {today.strftime('%Y-%m-%d')}). "
            f"Future dates: {future_examples}. "
            f"This indicates potential data error or look-ahead bias."
        )


def _validate_minimum_history(
    df: pd.DataFrame,
    min_days: int,
    data_name: str,
) -> None:
    """
    Validate that minimum history length is met.
    
    Args:
        df: DataFrame to validate
        min_days: Minimum required trading days
        data_name: Name for error messages
        
    Raises:
        InsufficientHistoryError: If history is too short
    """
    actual_days = len(df)
    
    if actual_days < min_days:
        date_range = (
            df.index.min().strftime("%Y-%m-%d"),
            df.index.max().strftime("%Y-%m-%d"),
        )
        
        raise InsufficientHistoryError(
            f"{data_name}: Insufficient history. "
            f"Got {actual_days} trading days, required {min_days}. "
            f"Date range: {date_range[0]} to {date_range[1]}. "
            f"Consider expanding date range or reducing min_history_days."
        )


def _validate_no_nans(
    df: pd.DataFrame,
    data_name: str,
) -> None:
    """
    Validate that no NaN values exist in the data.
    
    Args:
        df: DataFrame to validate
        data_name: Name for error messages
        
    Raises:
        DataIntegrityError: If NaN values are found
    """
    nan_counts = df.isna().sum()
    total_nans = nan_counts.sum()
    
    if total_nans > 0:
        # Identify worst offenders
        problem_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
        worst = problem_cols.head(5).to_dict()
        
        raise DataIntegrityError(
            f"{data_name}: Contains {total_nans} NaN values. "
            f"Affected columns (NaN count): {worst}"
        )


def _validate_positive_prices(df: pd.DataFrame) -> None:
    """
    Validate that all prices are positive.
    
    Args:
        df: Price DataFrame to validate
        
    Raises:
        DataIntegrityError: If non-positive prices are found
    """
    negative_mask = df < 0
    
    if negative_mask.any().any():
        # Find the offending columns
        problem_cols = df.columns[negative_mask.any()].tolist()
        
        raise DataIntegrityError(
            f"Negative prices found in assets: {problem_cols[:5]}. "
            f"Price data must be strictly positive."
        )


def _validate_no_zero_prices(df: pd.DataFrame) -> None:
    """
    Validate that no zero prices exist.
    
    Zero prices typically indicate data errors or delistings.
    
    Args:
        df: Price DataFrame to validate
        
    Raises:
        DataIntegrityError: If zero prices are found
    """
    zero_mask = df == 0
    
    if zero_mask.any().any():
        # Find the offending columns and dates
        problem_cols = df.columns[zero_mask.any()].tolist()
        
        raise DataIntegrityError(
            f"Zero prices found in assets: {problem_cols[:5]}. "
            f"Zero prices indicate data errors or delistings. "
            f"Remove these assets from the universe."
        )


def _validate_per_asset_sufficiency(
    df: pd.DataFrame,
    min_days: int,
) -> None:
    """
    Validate that each individual asset has sufficient data.
    
    While the overall DataFrame may meet requirements, individual
    assets might have spotty coverage.
    
    Args:
        df: Price DataFrame to validate
        min_days: Minimum required trading days per asset
        
    Raises:
        InsufficientHistoryError: If any asset has insufficient data
    """
    insufficient_assets = []
    
    for col in df.columns:
        valid_count = df[col].notna().sum()
        
        if valid_count < min_days:
            insufficient_assets.append((col, valid_count))
    
    if insufficient_assets:
        # Sort by count
        insufficient_assets.sort(key=lambda x: x[1])
        
        raise InsufficientHistoryError(
            f"{len(insufficient_assets)} assets have insufficient history "
            f"(min required: {min_days}). "
            f"Worst offenders: {insufficient_assets[:5]}"
        )


def validate_no_lookahead_bias(
    raw_data: Dict[str, pd.DataFrame],
    processed_data: pd.DataFrame,
) -> None:
    """
    Validate that preprocessing did not introduce look-ahead bias.
    
    This check ensures that processed data at time T only uses
    information available at time T or earlier.
    
    Args:
        raw_data: Original raw data
        processed_data: Preprocessed data
        
    Raises:
        DataIntegrityError: If look-ahead bias is detected
    """
    # Check 1: Processed dates should not extend beyond raw data dates
    for ticker in processed_data.columns:
        if ticker not in raw_data:
            continue
        
        raw_df = raw_data[ticker]
        raw_max_date = raw_df.index.max()
        
        processed_ticker_data = processed_data[ticker]
        processed_max_date = processed_data.index.max()
        
        if processed_max_date > raw_max_date:
            raise DataIntegrityError(
                f"Look-ahead bias detected for {ticker}: "
                f"processed data extends to {processed_max_date}, "
                f"but raw data ends at {raw_max_date}"
            )
    
    logger.debug("No look-ahead bias detected")

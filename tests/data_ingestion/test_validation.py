"""
Tests for time-series validation module.

Tests ensure:
- Hard failures on data quality issues
- Clear, actionable error messages
- No warnings - only hard failures
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_gnn.data_ingestion.validation import (
    validate_timeseries,
    ValidationError,
    TimestampError,
    DuplicateDateError,
    FutureDateError,
    InsufficientHistoryError,
    DataIntegrityError,
    _validate_monotonic_timestamps,
    _validate_no_duplicate_dates,
    _validate_no_future_dates,
    _validate_minimum_history,
    _validate_no_nans,
    _validate_positive_prices,
    _validate_no_zero_prices,
)
from portfolio_gnn.data_ingestion.preprocessing import PreprocessedData


def create_mock_config(min_history_days: int = 100):
    """Create a mock configuration object."""
    class MockConfig:
        class Data:
            def __init__(self):
                self.min_history_days = min_history_days
                self.preprocessing = {
                    "min_history_days": min_history_days,
                }
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        def __init__(self):
            self.data = MockConfig.Data()
    
    return MockConfig()


def create_valid_price_data(num_days: int = 252) -> pd.DataFrame:
    """Create valid price data for testing."""
    dates = pd.date_range("2023-01-01", periods=num_days, freq="B")
    np.random.seed(42)
    
    prices = pd.DataFrame({
        "AAPL": 100 * np.cumprod(1 + np.random.randn(num_days) * 0.02),
        "MSFT": 200 * np.cumprod(1 + np.random.randn(num_days) * 0.02),
        "GOOGL": 150 * np.cumprod(1 + np.random.randn(num_days) * 0.02),
    }, index=dates)
    
    return prices


def create_preprocessed_data(
    prices: pd.DataFrame,
    macro: pd.DataFrame = None,
) -> PreprocessedData:
    """Create PreprocessedData for testing."""
    return PreprocessedData(
        prices=prices,
        macro=macro,
        start_date=prices.index.min().strftime("%Y-%m-%d"),
        end_date=prices.index.max().strftime("%Y-%m-%d"),
        dropped_assets=[],
        config=None,
    )


class TestMonotonicTimestamps:
    """Tests for monotonic timestamp validation."""
    
    def test_valid_monotonic_passes(self):
        """Valid monotonic timestamps pass validation."""
        prices = create_valid_price_data()
        # Should not raise
        _validate_monotonic_timestamps(prices, "test")
    
    def test_non_monotonic_fails(self):
        """Non-monotonic timestamps fail with TimestampError."""
        dates = pd.DatetimeIndex([
            "2023-01-01", "2023-01-03", "2023-01-02", "2023-01-04"
        ])
        prices = pd.DataFrame({"AAPL": [100, 101, 102, 103]}, index=dates)
        
        with pytest.raises(TimestampError, match="not monotonically increasing"):
            _validate_monotonic_timestamps(prices, "test")
    
    def test_non_datetime_index_fails(self):
        """Non-DatetimeIndex fails with TimestampError."""
        prices = pd.DataFrame({"AAPL": [100, 101, 102]}, index=[0, 1, 2])
        
        with pytest.raises(TimestampError, match="not a DatetimeIndex"):
            _validate_monotonic_timestamps(prices, "test")


class TestDuplicateDates:
    """Tests for duplicate date validation."""
    
    def test_no_duplicates_passes(self):
        """Data without duplicates passes validation."""
        prices = create_valid_price_data()
        # Should not raise
        _validate_no_duplicate_dates(prices, "test")
    
    def test_duplicates_fail(self):
        """Duplicate dates fail with DuplicateDateError."""
        dates = pd.DatetimeIndex([
            "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03"
        ])
        prices = pd.DataFrame({"AAPL": [100, 101, 102, 103]}, index=dates)
        
        with pytest.raises(DuplicateDateError, match="Duplicate dates found"):
            _validate_no_duplicate_dates(prices, "test")


class TestFutureDates:
    """Tests for future date validation."""
    
    def test_past_dates_pass(self):
        """Historical dates pass validation."""
        prices = create_valid_price_data()
        # Should not raise
        _validate_no_future_dates(prices, "test")
    
    def test_future_dates_fail(self):
        """Future dates fail with FutureDateError."""
        future_date = datetime.now() + timedelta(days=30)
        dates = pd.DatetimeIndex([
            "2023-01-01", 
            "2023-01-02",
            future_date.strftime("%Y-%m-%d"),
        ])
        prices = pd.DataFrame({"AAPL": [100, 101, 102]}, index=dates)
        
        with pytest.raises(FutureDateError, match="Future timestamps found"):
            _validate_no_future_dates(prices, "test")


class TestMinimumHistory:
    """Tests for minimum history validation."""
    
    def test_sufficient_history_passes(self):
        """Sufficient history passes validation."""
        prices = create_valid_price_data(num_days=252)
        # Should not raise
        _validate_minimum_history(prices, min_days=100, data_name="test")
    
    def test_insufficient_history_fails(self):
        """Insufficient history fails with InsufficientHistoryError."""
        prices = create_valid_price_data(num_days=50)
        
        with pytest.raises(InsufficientHistoryError, match="Insufficient history"):
            _validate_minimum_history(prices, min_days=100, data_name="test")


class TestNaNValidation:
    """Tests for NaN value validation."""
    
    def test_no_nans_passes(self):
        """Data without NaN passes validation."""
        prices = create_valid_price_data()
        # Should not raise
        _validate_no_nans(prices, "test")
    
    def test_nans_fail(self):
        """Data with NaN fails with DataIntegrityError."""
        prices = create_valid_price_data()
        prices.iloc[10, 0] = np.nan
        
        with pytest.raises(DataIntegrityError, match="Contains .* NaN values"):
            _validate_no_nans(prices, "test")


class TestPositivePrices:
    """Tests for positive price validation."""
    
    def test_positive_prices_pass(self):
        """Positive prices pass validation."""
        prices = create_valid_price_data()
        # Should not raise
        _validate_positive_prices(prices)
    
    def test_negative_prices_fail(self):
        """Negative prices fail with DataIntegrityError."""
        prices = create_valid_price_data()
        prices.iloc[10, 0] = -100
        
        with pytest.raises(DataIntegrityError, match="Negative prices found"):
            _validate_positive_prices(prices)


class TestZeroPrices:
    """Tests for zero price validation."""
    
    def test_non_zero_prices_pass(self):
        """Non-zero prices pass validation."""
        prices = create_valid_price_data()
        # Should not raise
        _validate_no_zero_prices(prices)
    
    def test_zero_prices_fail(self):
        """Zero prices fail with DataIntegrityError."""
        prices = create_valid_price_data()
        prices.iloc[10, 0] = 0
        
        with pytest.raises(DataIntegrityError, match="Zero prices found"):
            _validate_no_zero_prices(prices)


class TestFullValidation:
    """Integration tests for full validation pipeline."""
    
    def test_valid_data_passes(self):
        """Valid data passes all validation checks."""
        prices = create_valid_price_data(num_days=252)
        data = create_preprocessed_data(prices)
        config = create_mock_config(min_history_days=100)
        
        # Should not raise
        validate_timeseries(data, config)
    
    def test_multiple_issues_reported(self):
        """Multiple validation issues are all reported."""
        # Create data with multiple issues
        prices = create_valid_price_data(num_days=50)  # Insufficient history
        prices.iloc[10, 0] = np.nan  # NaN value
        
        data = create_preprocessed_data(prices)
        config = create_mock_config(min_history_days=100)
        
        with pytest.raises(ValidationError, match="validation failed"):
            validate_timeseries(data, config)


class TestActionableErrors:
    """Tests that errors are actionable (identify offending data)."""
    
    def test_duplicate_error_shows_dates(self):
        """Duplicate date error shows which dates are duplicated."""
        dates = pd.DatetimeIndex([
            "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03"
        ])
        prices = pd.DataFrame({"AAPL": [100, 101, 102, 103]}, index=dates)
        
        with pytest.raises(DuplicateDateError) as exc_info:
            _validate_no_duplicate_dates(prices, "test")
        
        assert "2023-01-02" in str(exc_info.value)
    
    def test_nan_error_shows_columns(self):
        """NaN error shows which columns have NaN values."""
        prices = create_valid_price_data()
        prices.iloc[10, 0] = np.nan  # AAPL
        
        with pytest.raises(DataIntegrityError) as exc_info:
            _validate_no_nans(prices, "test")
        
        assert "AAPL" in str(exc_info.value)
    
    def test_insufficient_history_shows_counts(self):
        """Insufficient history error shows actual vs required counts."""
        prices = create_valid_price_data(num_days=50)
        
        with pytest.raises(InsufficientHistoryError) as exc_info:
            _validate_minimum_history(prices, min_days=100, data_name="test")
        
        assert "50" in str(exc_info.value)
        assert "100" in str(exc_info.value)


class TestNoWarningsOnlyFailures:
    """Ensure validation produces hard failures, not warnings."""
    
    def test_invalid_data_raises_exception(self):
        """Invalid data always raises exceptions, never just warnings."""
        prices = create_valid_price_data()
        prices.iloc[10, 0] = np.nan
        
        data = create_preprocessed_data(prices)
        config = create_mock_config()
        
        # Must raise, not just warn
        with pytest.raises(ValidationError):
            validate_timeseries(data, config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

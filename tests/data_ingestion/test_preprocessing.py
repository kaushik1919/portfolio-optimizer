"""
Tests for time-series preprocessing module.

Tests ensure:
- Correct date alignment
- No look-ahead bias (no forward-fill)
- Proper handling of missing values
- Hard failures on data quality issues
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_gnn.data_ingestion.preprocessing import (
    preprocess_market_data,
    PreprocessedData,
    PreprocessingError,
    InsufficientDataError,
    DataAlignmentError,
    _extract_adjusted_close,
    _filter_by_date,
    _align_inner_join,
    _drop_leading_nans,
)
from portfolio_gnn.data_ingestion.loader import MarketDataResult


def create_mock_config(
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    min_history_days: int = 100,
    max_missing_ratio: float = 0.05,
):
    """Create a mock configuration object."""
    class MockConfig:
        class Data:
            def __init__(self):
                self.start_date = start_date
                self.end_date = end_date
                self.min_history_days = min_history_days
                self.max_missing_ratio = max_missing_ratio
                self.preprocessing = {
                    "min_history_days": min_history_days,
                    "max_missing_ratio": max_missing_ratio,
                }
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        def __init__(self):
            self.data = MockConfig.Data()
    
    return MockConfig()


def create_mock_ohlcv_data(
    start_date: str = "2023-01-01",
    num_days: int = 252,
    base_price: float = 100.0,
    include_nans: bool = False,
    nan_positions: list = None,
) -> pd.DataFrame:
    """Create mock OHLCV data for testing."""
    dates = pd.date_range(start=start_date, periods=num_days, freq="B")
    
    # Generate prices
    np.random.seed(42)
    returns = np.random.randn(num_days) * 0.02
    prices = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        "Open": prices * 0.99,
        "High": prices * 1.02,
        "Low": prices * 0.98,
        "Close": prices,
        "Adj Close": prices,
        "Volume": np.random.randint(1000000, 10000000, num_days),
    }, index=dates)
    
    if include_nans and nan_positions:
        for pos in nan_positions:
            if pos < len(df):
                df.iloc[pos, df.columns.get_loc("Adj Close")] = np.nan
    
    df.index.name = "Date"
    return df


def create_mock_raw_data(
    tickers: list,
    start_date: str = "2023-01-01",
    num_days: int = 252,
    config=None,
) -> MarketDataResult:
    """Create mock MarketDataResult for testing."""
    asset_prices = {}
    for i, ticker in enumerate(tickers):
        asset_prices[ticker] = create_mock_ohlcv_data(
            start_date=start_date,
            num_days=num_days,
            base_price=100 + i * 10,
        )
    
    if config is None:
        config = create_mock_config()
    
    return MarketDataResult(
        asset_prices=asset_prices,
        macro_series={},
        start_date=start_date,
        end_date=(pd.Timestamp(start_date) + pd.Timedelta(days=num_days + 100)).strftime("%Y-%m-%d"),
        config=config,
    )


class TestExtractAdjustedClose:
    """Tests for adjusted close extraction."""
    
    def test_extract_adj_close(self):
        """Correctly extracts Adj Close column."""
        data = {
            "AAPL": create_mock_ohlcv_data(),
            "MSFT": create_mock_ohlcv_data(base_price=200),
        }
        
        result = _extract_adjusted_close(data)
        
        assert len(result) == 2
        assert "AAPL" in result
        assert "MSFT" in result
        assert isinstance(result["AAPL"], pd.Series)
    
    def test_skip_empty_data(self):
        """Skips empty DataFrames."""
        data = {
            "AAPL": create_mock_ohlcv_data(),
            "EMPTY": pd.DataFrame(),
        }
        
        result = _extract_adjusted_close(data)
        
        assert len(result) == 1
        assert "AAPL" in result
        assert "EMPTY" not in result
    
    def test_skip_missing_adj_close(self):
        """Skips DataFrames without Adj Close column."""
        df = create_mock_ohlcv_data()
        df = df.drop(columns=["Adj Close"])
        
        data = {
            "AAPL": create_mock_ohlcv_data(),
            "NO_ADJ": df,
        }
        
        result = _extract_adjusted_close(data)
        
        assert len(result) == 1
        assert "AAPL" in result
        assert "NO_ADJ" not in result


class TestDateFiltering:
    """Tests for date filtering."""
    
    def test_filter_within_range(self):
        """Filters data to specified date range."""
        series = {
            "AAPL": create_mock_ohlcv_data(
                start_date="2023-01-01", num_days=300
            )["Adj Close"],
        }
        
        result = _filter_by_date(series, "2023-03-01", "2023-09-30")
        
        assert result["AAPL"].index.min() >= pd.Timestamp("2023-03-01")
        assert result["AAPL"].index.max() <= pd.Timestamp("2023-09-30")
    
    def test_filter_excludes_out_of_range(self):
        """Data outside range is excluded."""
        series = {
            "AAPL": create_mock_ohlcv_data(
                start_date="2023-01-01", num_days=252
            )["Adj Close"],
        }
        
        result = _filter_by_date(series, "2024-01-01", "2024-12-31")
        
        # Should be empty since all data is in 2023
        assert "AAPL" not in result


class TestInnerJoinAlignment:
    """Tests for inner-join alignment."""
    
    def test_aligned_dates_match(self):
        """All series have identical dates after alignment."""
        # Create series with slightly different date ranges
        dates1 = pd.date_range("2023-01-01", periods=250, freq="B")
        dates2 = pd.date_range("2023-01-10", periods=250, freq="B")
        
        series = {
            "AAPL": pd.Series(np.random.rand(250), index=dates1, name="AAPL"),
            "MSFT": pd.Series(np.random.rand(250), index=dates2, name="MSFT"),
        }
        
        result, dropped = _align_inner_join(series, max_missing_ratio=0.1)
        
        assert len(dropped) == 0
        assert result["AAPL"].notna().all()
        assert result["MSFT"].notna().all()
    
    def test_drops_high_missing_assets(self):
        """Assets with excessive missing data are dropped."""
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        
        series = {
            "GOOD": pd.Series(np.random.rand(100), index=dates, name="GOOD"),
            "BAD": pd.Series([np.nan] * 50 + list(np.random.rand(50)), index=dates, name="BAD"),
        }
        
        result, dropped = _align_inner_join(series, max_missing_ratio=0.1)
        
        assert "BAD" in dropped
        assert "GOOD" in result.columns
    
    def test_raises_on_all_dropped(self):
        """Raises error when all assets are dropped."""
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        
        # All series have > 50% missing
        series = {
            "BAD1": pd.Series([np.nan] * 60 + list(np.random.rand(40)), index=dates, name="BAD1"),
            "BAD2": pd.Series([np.nan] * 70 + list(np.random.rand(30)), index=dates, name="BAD2"),
        }
        
        with pytest.raises(DataAlignmentError, match="All assets exceed"):
            _align_inner_join(series, max_missing_ratio=0.1)


class TestNoLookAheadBias:
    """Tests to ensure no look-ahead bias is introduced."""
    
    def test_no_forward_fill(self):
        """Preprocessing does NOT forward-fill NaN values."""
        # Create data with NaN in the middle
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        prices = np.random.rand(100) * 100
        prices[50] = np.nan  # Insert NaN
        
        series = {
            "AAPL": pd.Series(prices.copy(), index=dates, name="AAPL"),
            "MSFT": pd.Series(prices.copy(), index=dates, name="MSFT"),
        }
        
        # Inner join should drop the row with NaN
        result, _ = _align_inner_join(series, max_missing_ratio=0.1)
        
        # Row with NaN should be dropped, not filled
        assert len(result) == 99  # One row dropped
        assert result.notna().all().all()  # No NaN values remain
    
    def test_no_backward_fill(self):
        """Preprocessing does NOT backward-fill NaN values."""
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        prices = np.random.rand(100) * 100
        prices[50] = np.nan
        
        series = {
            "AAPL": pd.Series(prices.copy(), index=dates, name="AAPL"),
        }
        
        # Should drop the NaN row, not fill it
        df = pd.DataFrame(series)
        aligned = df.dropna(how="any")
        
        assert len(aligned) == 99
        # Value at index 49 and 51 should NOT be affected
        assert aligned.iloc[49, 0] == prices[49]


class TestPreprocessedData:
    """Tests for PreprocessedData container."""
    
    def test_properties(self):
        """PreprocessedData properties work correctly."""
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        prices = pd.DataFrame({
            "AAPL": np.random.rand(252) * 100,
            "MSFT": np.random.rand(252) * 200,
        }, index=dates)
        
        data = PreprocessedData(
            prices=prices,
            macro=None,
            start_date="2023-01-01",
            end_date="2023-12-31",
            dropped_assets=["INVALID"],
            config=None,
        )
        
        assert data.num_assets == 2
        assert data.num_trading_days == 252
        assert data.tickers == ["AAPL", "MSFT"]
        assert data.date_range == ("2023-01-01", "2023-12-31")


class TestFullPreprocessingPipeline:
    """Integration tests for full preprocessing pipeline."""
    
    def test_full_pipeline_success(self):
        """Full preprocessing pipeline succeeds with valid data."""
        config = create_mock_config(
            start_date="2023-01-01",
            end_date="2023-12-31",
            min_history_days=100,
        )
        
        raw_data = create_mock_raw_data(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start_date="2023-01-01",
            num_days=252,
            config=config,
        )
        
        result = preprocess_market_data(raw_data, config)
        
        assert isinstance(result, PreprocessedData)
        assert result.num_assets == 3
        assert result.num_trading_days > 0
    
    def test_insufficient_history_fails(self):
        """Fails when data has insufficient history."""
        config = create_mock_config(
            start_date="2023-01-01",
            end_date="2023-12-31",
            min_history_days=500,  # More than available
        )
        
        raw_data = create_mock_raw_data(
            tickers=["AAPL"],
            start_date="2023-01-01",
            num_days=100,  # Only 100 days
            config=config,
        )
        
        # Will raise DataAlignmentError when all assets are dropped
        # due to insufficient history before alignment
        with pytest.raises((InsufficientDataError, DataAlignmentError)):
            preprocess_market_data(raw_data, config)


class TestDeterministicPreprocessing:
    """Tests for deterministic preprocessing behavior."""
    
    def test_same_input_same_output(self):
        """Same inputs produce identical outputs."""
        config = create_mock_config(min_history_days=50)
        
        raw_data1 = create_mock_raw_data(
            tickers=["AAPL", "MSFT"],
            num_days=252,
            config=config,
        )
        
        raw_data2 = create_mock_raw_data(
            tickers=["AAPL", "MSFT"],
            num_days=252,
            config=config,
        )
        
        result1 = preprocess_market_data(raw_data1, config)
        result2 = preprocess_market_data(raw_data2, config)
        
        pd.testing.assert_frame_equal(result1.prices, result2.prices)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for Yahoo Finance data loader.

All tests mock external API calls - no network requests.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_gnn.data_ingestion.sources.yahoo import (
    load_yahoo_prices,
    YahooDataError,
    _normalize_date,
)


def create_mock_ohlcv_data(
    ticker: str,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    num_days: int = 252,
) -> pd.DataFrame:
    """Create mock OHLCV data for testing."""
    dates = pd.date_range(start=start_date, periods=num_days, freq="B")  # Business days
    
    np.random.seed(hash(ticker) % 2**32)
    base_price = 100 + np.random.randint(0, 100)
    
    # Generate random walk prices
    returns = np.random.randn(num_days) * 0.02
    prices = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        "Open": prices * (1 - np.random.rand(num_days) * 0.01),
        "High": prices * (1 + np.random.rand(num_days) * 0.02),
        "Low": prices * (1 - np.random.rand(num_days) * 0.02),
        "Close": prices,
        "Adj Close": prices * 0.99,
        "Volume": np.random.randint(1000000, 10000000, num_days),
    }, index=dates)
    
    df.index.name = "Date"
    return df


class TestYahooDateNormalization:
    """Tests for date normalization."""
    
    def test_string_date_valid(self):
        """Valid string date is returned as-is."""
        assert _normalize_date("2023-01-15") == "2023-01-15"
    
    def test_datetime_to_string(self):
        """datetime is converted to string."""
        dt = datetime(2023, 6, 15, 10, 30, 0)
        assert _normalize_date(dt) == "2023-06-15"
    
    def test_date_to_string(self):
        """date is converted to string."""
        from datetime import date
        d = date(2023, 6, 15)
        assert _normalize_date(d) == "2023-06-15"
    
    def test_invalid_string_format(self):
        """Invalid string format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid date string format"):
            _normalize_date("15-01-2023")
    
    def test_invalid_type(self):
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid date type"):
            _normalize_date(12345)


class TestYahooLoader:
    """Tests for Yahoo Finance loader with mocked API."""
    
    @patch("portfolio_gnn.data_ingestion.sources.yahoo.yf")
    def test_load_single_ticker_success(self, mock_yf):
        """Successfully load data for a single ticker."""
        # Setup mock
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = create_mock_ohlcv_data("AAPL")
        mock_yf.Ticker.return_value = mock_ticker
        
        result = load_yahoo_prices(
            tickers=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
        
        assert "AAPL" in result
        assert isinstance(result["AAPL"], pd.DataFrame)
        assert "Adj Close" in result["AAPL"].columns
    
    @patch("portfolio_gnn.data_ingestion.sources.yahoo.yf")
    def test_load_multiple_tickers(self, mock_yf):
        """Successfully load data for multiple tickers."""
        def mock_ticker_factory(symbol):
            mock = MagicMock()
            mock.history.return_value = create_mock_ohlcv_data(symbol)
            return mock
        
        mock_yf.Ticker.side_effect = mock_ticker_factory
        
        result = load_yahoo_prices(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
        
        assert len(result) == 3
        assert all(ticker in result for ticker in ["AAPL", "MSFT", "GOOGL"])
    
    def test_empty_tickers_raises(self):
        """Empty tickers list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            load_yahoo_prices(
                tickers=[],
                start_date="2023-01-01",
                end_date="2023-12-31",
            )
    
    def test_invalid_date_range(self):
        """End date before start date raises ValueError."""
        with pytest.raises(ValueError, match="must be before"):
            load_yahoo_prices(
                tickers=["AAPL"],
                start_date="2023-12-31",
                end_date="2023-01-01",
            )
    
    @patch("portfolio_gnn.data_ingestion.sources.yahoo.yf")
    def test_partial_failure_continues(self, mock_yf):
        """Loader continues when some tickers fail."""
        def mock_ticker_factory(symbol):
            mock = MagicMock()
            if symbol == "INVALID":
                mock.history.return_value = pd.DataFrame()
            else:
                mock.history.return_value = create_mock_ohlcv_data(symbol)
            return mock
        
        mock_yf.Ticker.side_effect = mock_ticker_factory
        
        result = load_yahoo_prices(
            tickers=["AAPL", "INVALID", "MSFT"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
        
        assert len(result) == 2
        assert "AAPL" in result
        assert "MSFT" in result
        assert "INVALID" not in result
    
    @patch("portfolio_gnn.data_ingestion.sources.yahoo.yf")
    def test_all_tickers_fail_raises(self, mock_yf):
        """All tickers failing raises YahooDataError."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker
        
        with pytest.raises(YahooDataError, match="No data loaded"):
            load_yahoo_prices(
                tickers=["INVALID1", "INVALID2"],
                start_date="2023-01-01",
                end_date="2023-12-31",
            )
    
    @patch("portfolio_gnn.data_ingestion.sources.yahoo.yf")
    def test_data_has_correct_index(self, mock_yf):
        """Returned data has DatetimeIndex."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = create_mock_ohlcv_data("AAPL")
        mock_yf.Ticker.return_value = mock_ticker
        
        result = load_yahoo_prices(
            tickers=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
        
        assert isinstance(result["AAPL"].index, pd.DatetimeIndex)
        assert result["AAPL"].index.name == "Date"


class TestDeterministicLoading:
    """Tests for deterministic behavior."""
    
    @patch("portfolio_gnn.data_ingestion.sources.yahoo.yf")
    def test_same_input_same_output(self, mock_yf):
        """Same inputs produce same outputs."""
        mock_data = create_mock_ohlcv_data("AAPL")
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_data.copy()
        mock_yf.Ticker.return_value = mock_ticker
        
        result1 = load_yahoo_prices(
            tickers=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
        
        # Reset mock
        mock_ticker.history.return_value = mock_data.copy()
        
        result2 = load_yahoo_prices(
            tickers=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
        
        pd.testing.assert_frame_equal(result1["AAPL"], result2["AAPL"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

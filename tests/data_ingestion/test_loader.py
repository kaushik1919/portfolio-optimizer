"""
Tests for unified data loader module.

All tests mock external API calls - no network requests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_gnn.data_ingestion.loader import (
    load_market_data,
    MarketDataResult,
    DataLoadingError,
    _extract_data_config,
    _extract_asset_universe,
    _extract_macro_series,
    _validate_date_range,
)


def create_mock_ohlcv_data(
    ticker: str,
    start_date: str = "2023-01-01",
    num_days: int = 252,
) -> pd.DataFrame:
    """Create mock OHLCV data for testing."""
    dates = pd.date_range(start=start_date, periods=num_days, freq="B")
    
    np.random.seed(hash(ticker) % 2**32)
    base_price = 100 + np.random.randint(0, 100)
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
    
    df.index.name = "Date"
    return df


def create_mock_config(
    asset_universe: list = None,
    macro_series: list = None,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
):
    """Create a mock configuration object."""
    if asset_universe is None:
        asset_universe = ["AAPL", "MSFT", "GOOGL"]
    if macro_series is None:
        macro_series = []
    
    class MockConfig:
        class Data:
            def __init__(self):
                self.asset_universe = asset_universe
                self.macro_series = macro_series
                self.start_date = start_date
                self.end_date = end_date
                self.min_history_days = 100
                self.max_missing_ratio = 0.05
            
            def get(self, key, default=None):
                return getattr(self, key, default)
            
            def to_dict(self):
                return {
                    "asset_universe": self.asset_universe,
                    "macro_series": self.macro_series,
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "min_history_days": self.min_history_days,
                    "max_missing_ratio": self.max_missing_ratio,
                }
        
        def __init__(self):
            self.data = MockConfig.Data()
    
    return MockConfig()


class TestExtractDataConfig:
    """Tests for data configuration extraction."""
    
    def test_extract_from_object(self):
        """Extracts data config from object with .data attribute."""
        config = create_mock_config()
        result = _extract_data_config(config)
        
        assert "asset_universe" in result or "start_date" in result
    
    def test_missing_data_section_raises(self):
        """Missing data section raises ValueError."""
        class EmptyConfig:
            pass
        
        with pytest.raises(ValueError, match="'data' section"):
            _extract_data_config(EmptyConfig())


class TestExtractAssetUniverse:
    """Tests for asset universe extraction."""
    
    def test_extract_direct_list(self):
        """Extracts asset_universe as direct list."""
        data_config = {
            "asset_universe": ["AAPL", "MSFT", "GOOGL"],
        }
        
        result = _extract_asset_universe(data_config)
        
        assert result == ["AAPL", "MSFT", "GOOGL"]
    
    def test_extract_from_nested_dict(self):
        """Extracts tickers from nested dict."""
        data_config = {
            "asset_universe": {
                "tickers": ["AAPL", "MSFT"],
            },
        }
        
        result = _extract_asset_universe(data_config)
        
        assert result == ["AAPL", "MSFT"]
    
    def test_empty_universe(self):
        """Returns empty list when no universe specified."""
        data_config = {}
        
        result = _extract_asset_universe(data_config)
        
        assert result == []


class TestExtractMacroSeries:
    """Tests for macro series extraction."""
    
    def test_extract_direct_list(self):
        """Extracts macro_series as direct list."""
        data_config = {
            "macro_series": ["DGS10", "VIXCLS"],
        }
        
        result = _extract_macro_series(data_config)
        
        assert result == ["DGS10", "VIXCLS"]
    
    def test_empty_macro(self):
        """Returns empty list when no macro series specified."""
        data_config = {}
        
        result = _extract_macro_series(data_config)
        
        assert result == []


class TestDateRangeValidation:
    """Tests for date range validation."""
    
    def test_valid_range_passes(self):
        """Valid date range passes validation."""
        # Should not raise
        _validate_date_range("2023-01-01", "2023-12-31")
    
    def test_invalid_range_fails(self):
        """End before start fails validation."""
        with pytest.raises(ValueError, match="must be before"):
            _validate_date_range("2023-12-31", "2023-01-01")
    
    def test_invalid_format_fails(self):
        """Invalid date format fails validation."""
        with pytest.raises(ValueError, match="Invalid date format"):
            _validate_date_range("01-01-2023", "2023-12-31")


class TestMarketDataResult:
    """Tests for MarketDataResult container."""
    
    def test_properties(self):
        """MarketDataResult properties work correctly."""
        asset_prices = {
            "AAPL": create_mock_ohlcv_data("AAPL"),
            "MSFT": create_mock_ohlcv_data("MSFT"),
        }
        
        result = MarketDataResult(
            asset_prices=asset_prices,
            macro_series={},
            start_date="2023-01-01",
            end_date="2023-12-31",
            config=None,
        )
        
        assert result.num_assets == 2
        assert result.num_macro_series == 0
        assert result.asset_tickers == ["AAPL", "MSFT"]
        assert result.macro_ids == []


class TestLoadMarketData:
    """Integration tests for load_market_data with mocked APIs."""
    
    @patch("portfolio_gnn.data_ingestion.loader.load_yahoo_prices")
    def test_load_assets_success(self, mock_yahoo):
        """Successfully loads asset data."""
        mock_yahoo.return_value = {
            "AAPL": create_mock_ohlcv_data("AAPL"),
            "MSFT": create_mock_ohlcv_data("MSFT"),
        }
        
        config = create_mock_config(
            asset_universe=["AAPL", "MSFT"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
        
        result = load_market_data(config)
        
        assert isinstance(result, MarketDataResult)
        assert result.num_assets == 2
        mock_yahoo.assert_called_once()
    
    @patch("portfolio_gnn.data_ingestion.loader.load_yahoo_prices")
    def test_load_failure_raises(self, mock_yahoo):
        """Data loading failure raises DataLoadingError."""
        from portfolio_gnn.data_ingestion.sources.yahoo import YahooDataError
        mock_yahoo.side_effect = YahooDataError("API failure")
        
        config = create_mock_config(
            asset_universe=["AAPL"],
        )
        
        with pytest.raises(DataLoadingError, match="Failed to load"):
            load_market_data(config)
    
    def test_missing_dates_raises(self):
        """Missing date configuration raises ValueError."""
        class BadConfig:
            class Data:
                def get(self, key, default=None):
                    return None
                def to_dict(self):
                    return {}
            data = Data()
        
        with pytest.raises(ValueError, match="start_date and data.end_date"):
            load_market_data(BadConfig())


class TestDeterministicLoading:
    """Tests for deterministic data loading behavior."""
    
    @patch("portfolio_gnn.data_ingestion.loader.load_yahoo_prices")
    def test_same_config_same_result(self, mock_yahoo):
        """Same configuration produces same result structure."""
        mock_data = {
            "AAPL": create_mock_ohlcv_data("AAPL"),
        }
        mock_yahoo.return_value = mock_data.copy()
        
        config = create_mock_config(asset_universe=["AAPL"])
        
        result1 = load_market_data(config)
        
        mock_yahoo.return_value = mock_data.copy()
        result2 = load_market_data(config)
        
        assert result1.asset_tickers == result2.asset_tickers
        assert result1.start_date == result2.start_date
        assert result1.end_date == result2.end_date


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

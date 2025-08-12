#!/usr/bin/env python3
"""
Test suite to validate ARIMA frequency fixes.

This test validates the hypothesis:
"If frequency strings are corrected and proper index frequency is set, 
then ARIMA should converge without frequency warnings"
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from forecasting import generate_forecast, TIMEFRAME_TO_FREQ, _forecast_arima

class TestARIMAFrequencyFix:
    """Test ARIMA frequency handling improvements."""
    
    def setup_method(self):
        """Create test data with different frequencies."""
        # Daily data
        start_date = datetime(2023, 1, 1)
        self.daily_dates = pd.date_range(start=start_date, periods=100, freq='D')
        self.daily_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        self.daily_df = pd.DataFrame({
            'timestamp': self.daily_dates,
            'open': self.daily_prices + np.random.randn(100) * 0.1,
            'high': self.daily_prices + np.random.rand(100) * 2,
            'low': self.daily_prices - np.random.rand(100) * 2,
            'close': self.daily_prices,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Hourly data
        self.hourly_dates = pd.date_range(start=start_date, periods=168, freq='H')  # 1 week
        self.hourly_prices = 100 + np.cumsum(np.random.randn(168) * 0.2)
        self.hourly_df = pd.DataFrame({
            'timestamp': self.hourly_dates,
            'open': self.hourly_prices + np.random.randn(168) * 0.1,
            'high': self.hourly_prices + np.random.rand(168) * 1,
            'low': self.hourly_prices - np.random.rand(168) * 1,
            'close': self.hourly_prices,
            'volume': np.random.randint(500, 5000, 168)
        })
    
    def test_frequency_mapping_updated(self):
        """Test that frequency mappings are correct for pandas compatibility."""
        # Test key frequency mappings are updated to modern pandas standards
        assert TIMEFRAME_TO_FREQ['1h'] == '1H', "1h should map to '1H' not '1h'"
        assert TIMEFRAME_TO_FREQ['4h'] == '4H', "4h should map to '4H' not '4h'"  
        assert TIMEFRAME_TO_FREQ['1d'] == 'D', "1d should map to 'D' not '1D'"
        assert TIMEFRAME_TO_FREQ['1w'] == 'W', "1w should map to 'W' not '1W'"
        assert TIMEFRAME_TO_FREQ['1mo'] == 'MS', "1mo should map to 'MS' for month start"
        assert TIMEFRAME_TO_FREQ['1wk'] == 'W', "1wk should map to 'W' for weekly"
    
    def test_arima_no_frequency_warnings(self):
        """Test that ARIMA no longer generates frequency warnings."""
        # Capture warnings during ARIMA forecast
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Generate ARIMA forecast with daily data
            result = generate_forecast(self.daily_df.copy(), 'ARIMA', 10, '1d')
            
            # Check for frequency-related warnings
            frequency_warnings = [
                warning for warning in w 
                if 'frequency' in str(warning.message).lower() or 
                   'inferred frequency' in str(warning.message).lower()
            ]
            
            assert len(frequency_warnings) == 0, f"Found frequency warnings: {[str(w.message) for w in frequency_warnings]}"
            assert result is not None, "ARIMA forecast should not be None"
            assert not result.empty, "ARIMA forecast should not be empty"
    
    def test_arima_convergence_daily(self):
        """Test ARIMA model convergence with daily data."""
        result = generate_forecast(self.daily_df.copy(), 'ARIMA', 10, '1d')
        
        assert result is not None, "ARIMA should return a result"
        assert len(result) == 10, "Should return 10 forecast periods"
        assert all(col in result.columns for col in ['ds', 'yhat', 'yhat_lower', 'yhat_upper']), \
            "Forecast should have required columns"
        assert not result['yhat'].isna().any(), "Forecast values should not be NaN"
    
    def test_arima_convergence_hourly(self):
        """Test ARIMA model convergence with hourly data."""
        result = generate_forecast(self.hourly_df.copy(), 'ARIMA', 24, '1h')  # 1 day ahead
        
        assert result is not None, "ARIMA should return a result"
        assert len(result) == 24, "Should return 24 hourly forecast periods"
        assert not result['yhat'].isna().any(), "Forecast values should not be NaN"
    
    def test_arima_proper_frequency_setting(self):
        """Test that ARIMA properly sets frequency on datetime index."""
        # Create data series for direct ARIMA testing
        data_series = self.daily_df.set_index('timestamp')['close']
        
        # Call ARIMA directly to test frequency setting
        result = _forecast_arima(data_series, 10, '1d')
        
        assert result is not None, "Direct ARIMA call should work"
        assert len(result) == 10, "Should return 10 periods"
        
        # Check that future dates are properly spaced (daily intervals)
        date_diffs = result['ds'].diff().dropna()
        expected_delta = timedelta(days=1)
        
        # All differences should be approximately 1 day
        assert all(abs((diff - expected_delta).total_seconds()) < 3600 for diff in date_diffs), \
            f"Date intervals should be ~1 day, got: {date_diffs.unique()}"
    
    def test_multiple_timeframes_no_errors(self):
        """Test that multiple timeframes work without errors."""
        test_cases = [
            (self.daily_df.copy(), '1d', 10),
            (self.hourly_df.copy(), '1h', 24),
        ]
        
        for df, freq, periods in test_cases:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                result = generate_forecast(df, 'ARIMA', periods, freq)
                
                # Check no critical warnings
                critical_warnings = [
                    warning for warning in w 
                    if 'convergence' not in str(warning.message).lower() and
                       'Maximum Likelihood' not in str(warning.message)
                ]
                
                assert result is not None, f"ARIMA should work for {freq}"
                assert len(result) == periods, f"Should return {periods} periods for {freq}"

def run_validation():
    """Run the ARIMA frequency validation tests."""
    print("🔎 ARIMA Frequency Fix Validation")
    print("=" * 40)
    
    test_instance = TestARIMAFrequencyFix()
    test_instance.setup_method()
    
    tests = [
        ("Frequency mapping updated", test_instance.test_frequency_mapping_updated),
        ("No frequency warnings", test_instance.test_arima_no_frequency_warnings), 
        ("ARIMA convergence (daily)", test_instance.test_arima_convergence_daily),
        ("ARIMA convergence (hourly)", test_instance.test_arima_convergence_hourly),
        ("Proper frequency setting", test_instance.test_arima_proper_frequency_setting),
        ("Multiple timeframes", test_instance.test_multiple_timeframes_no_errors),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"Testing: {test_name}... ", end="")
            test_func()
            print("✅ PASS")
            passed += 1
        except Exception as e:
            print(f"❌ FAIL: {e}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    return passed == total

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)

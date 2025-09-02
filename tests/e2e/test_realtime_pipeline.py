"""End-to-End tests for multiframe analysis components.

This module provides comprehensive E2E tests for multiframe analysis
including TimeframeConverter, MultiTimeframeAnalyzer, RCI calculation,
and performance benchmarks.
"""

from datetime import datetime, timedelta
import sys
import time
import tracemalloc
from typing import List, Dict, Any

import numpy as np
import polars as pl
import pytest


# Add project root to path
sys.path.insert(0, "src")

from data_processing.analyzer import MultiTimeframeAnalyzer
from data_processing.timeframe_converter import TimeframeConverter
from data_processing.rci import RCICalculatorEngine


class TestMultiframeAnalysisE2E:
    """End-to-end tests for complete multiframe analysis workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = None
        self.converter = None
        self.rci_engine = None
        self.base_time = datetime.now()
    
    def _create_realistic_market_data(self, n_bars: int = 1000) -> pl.DataFrame:
        """Generate realistic market data with trends and volatility.
        
        Args:
            n_bars: Number of 1-minute bars to generate
            
        Returns:
            DataFrame with OHLC data
        """
        timestamps = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        # Market parameters
        base_price = 150.0
        trend_strength = 0.0001
        volatility = 0.002
        volume_base = 1000
        
        # Add market cycles
        trend_cycle = 240  # 4 hour trend cycle
        volatility_cycle = 60  # 1 hour volatility cycle
        
        for i in range(n_bars):
            timestamp = self.base_time + timedelta(minutes=i)
            timestamps.append(timestamp)
            
            # Calculate trend component (sine wave)
            trend = trend_strength * np.sin(2 * np.pi * i / trend_cycle)
            
            # Calculate volatility (varies over time)
            current_volatility = volatility * (1 + 0.5 * np.sin(2 * np.pi * i / volatility_cycle))
            
            # Generate OHLC
            open_price = base_price
            
            # Intrabar movements
            movements = np.random.normal(trend, current_volatility, size=4)
            prices = base_price + np.cumsum(movements)
            
            high_price = max(prices.max(), open_price)
            low_price = min(prices.min(), open_price)
            close_price = prices[-1]
            
            # Volume varies with volatility
            volume = int(volume_base * (1 + current_volatility * 100))
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
            
            base_price = close_price
        
        return pl.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        }).sort("timestamp")
    
    def test_complete_analysis_workflow(self):
        """Test complete workflow from raw data to RCI calculation."""
        # Generate test data
        raw_data = self._create_realistic_market_data(500)
        
        # Initialize components
        self.converter = TimeframeConverter(
            source_timeframe="1T",
            target_timeframe="5T"
        )
        
        self.analyzer = MultiTimeframeAnalyzer(
            short_term_periods=[9, 13, 24],
            long_term_periods=[24, 33, 48],
            use_parallel=False
        )
        
        # Perform analysis
        result = self.analyzer.analyze(raw_data)
        
        # Verify result structure
        assert result is not None
        assert len(result) == len(raw_data)
        assert "timestamp" in result.columns
        
        # Debug: Print column names to understand the structure
        print(f"Result columns: {result.columns}")
        
        # Check short-term RCI columns (1-minute)
        short_cols = [col for col in result.columns if "short" in col and "rci" in col]
        if not short_cols:
            # Try alternative naming patterns
            short_cols = [col for col in result.columns if col.startswith("short_rci_")]
        assert len(short_cols) == 3, f"Expected 3 short RCI columns, got {short_cols}"
        
        # Check long-term RCI columns (5-minute)
        long_cols = [col for col in result.columns if "long" in col and "rci" in col]
        if not long_cols:
            # Try alternative naming patterns
            long_cols = [col for col in result.columns if col.startswith("long_rci_")]
        assert len(long_cols) == 3, f"Expected 3 long RCI columns, got {long_cols}"
        
        # Verify RCI values are valid
        for col in short_cols + long_cols:
            values = result[col].drop_nulls()
            assert values.min() >= -100.1  # Small tolerance
            assert values.max() <= 100.1
            
            # Check that we have sufficient non-null values
            non_null_ratio = len(values) / len(result)
            assert non_null_ratio > 0.5  # At least 50% should have values
    
    def test_streaming_analysis_simulation(self):
        """Test streaming analysis with simulated real-time data."""
        # Initialize analyzer
        self.analyzer = MultiTimeframeAnalyzer(
            short_term_periods=[9, 13],
            long_term_periods=[24, 33],
            use_parallel=False
        )
        
        # Generate initial history
        history_size = 200
        history_data = self._create_realistic_market_data(history_size)
        
        # Simulate streaming for 100 minutes
        streaming_duration = 100
        results = []
        
        current_data = history_data
        
        for minute in range(streaming_duration):
            # Generate new minute bar
            new_bar = self._create_realistic_market_data(1)
            
            # Adjust timestamp to continue from history
            last_timestamp = current_data["timestamp"][-1]
            new_timestamp = last_timestamp + timedelta(minutes=1)
            new_bar = new_bar.with_columns(
                pl.lit(new_timestamp).alias("timestamp")
            )
            
            # Append new bar
            current_data = pl.concat([current_data, new_bar])
            
            # Keep only recent history (limit to 300 bars)
            max_history = 300
            if len(current_data) > max_history:
                current_data = current_data[-max_history:]
            
            # Analyze with streaming mode
            stream_result = self.analyzer.analyze_streaming(
                current_data,
                last_complete_timestamp=current_data["timestamp"][-2]
            )
            
            if stream_result is not None:
                results.append(stream_result)
        
        # Verify streaming results
        assert len(results) > 0
        
        # Check result structure
        for result in results:
            assert "timestamp" in result
            assert "short_term_rci" in result
            assert "long_term_rci" in result
    
    def test_data_consistency_across_timeframes(self):
        """Test that data remains consistent across different timeframes."""
        # Generate data with known pattern
        n_bars = 300
        timestamps = []
        closes = []
        
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        
        # Create a clear trend: upward for first half, downward for second half
        for i in range(n_bars):
            timestamp = base_time + timedelta(minutes=i)
            timestamps.append(timestamp)
            
            if i < n_bars // 2:
                # Uptrend
                price = 150.0 + (i * 0.01)
            else:
                # Downtrend
                price = 150.0 + (n_bars // 2 * 0.01) - ((i - n_bars // 2) * 0.01)
            
            # Add small noise
            price += np.random.normal(0, 0.001)
            closes.append(price)
        
        test_data = pl.DataFrame({
            "timestamp": timestamps,
            "open": closes,
            "high": [c * 1.0001 for c in closes],
            "low": [c * 0.9999 for c in closes],
            "close": closes,
            "volume": [100] * n_bars
        })
        
        # Analyze with multiframe
        self.analyzer = MultiTimeframeAnalyzer(
            short_term_periods=[9],
            long_term_periods=[24],
            use_parallel=False
        )
        
        result = self.analyzer.analyze(test_data)
        
        # Check trend detection
        mid_point = len(result) // 2
        
        # Short-term RCI should reflect immediate trend
        short_rci = result["short_rci_9"].drop_nulls()
        
        # Sample from uptrend period (avoiding initial nulls)
        uptrend_sample = result[50:100]["short_rci_9"].drop_nulls()
        assert uptrend_sample.mean() > 30  # Should be positive during uptrend
        
        # Sample from downtrend period
        downtrend_sample = result[200:250]["short_rci_9"].drop_nulls()
        assert downtrend_sample.mean() < -30  # Should be negative during downtrend
        
        # Long-term RCI should be smoother
        long_rci = result["long_rci_24"].drop_nulls()
        assert long_rci.std() < short_rci.std()  # Long-term should be less volatile


class TestTimeframeConversionE2E:
    """End-to-end tests for timeframe conversion."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.converter = None
        self.base_time = datetime.now()
    
    def test_multiple_timeframe_conversions(self):
        """Test conversion to multiple timeframes."""
        # Generate 1-minute data for 1 day
        n_bars = 1440  # 24 hours
        timestamps = []
        closes = []
        volumes = []
        
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        
        for i in range(n_bars):
            timestamp = base_time + timedelta(minutes=i)
            timestamps.append(timestamp)
            closes.append(150.0 + np.random.normal(0, 0.1))
            volumes.append(np.random.randint(50, 150))
        
        source_data = pl.DataFrame({
            "timestamp": timestamps,
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": volumes
        })
        
        # Test different timeframe conversions
        timeframes = ["5T", "15T", "30T", "1H", "4H", "1D"]
        expected_bars = {
            "5T": 288,   # 1440 / 5
            "15T": 96,   # 1440 / 15
            "30T": 48,   # 1440 / 30
            "1H": 24,    # 1440 / 60
            "4H": 6,     # 1440 / 240
            "1D": 1      # 1440 / 1440
        }
        
        for tf in timeframes:
            converter = TimeframeConverter(
                source_timeframe="1T",
                target_timeframe=tf
            )
            
            converted = converter.convert(source_data)
            
            # Check number of bars (allowing for incomplete last bar)
            assert len(converted) <= expected_bars[tf]
            assert len(converted) >= expected_bars[tf] - 1
            
            # Verify aggregation rules
            assert all(converted["volume"].sum() <= source_data["volume"].sum() * 1.01)
            
            # Check time alignment
            first_bar = converted[0]
            assert first_bar["timestamp"][0] >= source_data["timestamp"][0]
    
    def test_streaming_conversion_accuracy(self):
        """Test accuracy of streaming timeframe conversion."""
        self.converter = TimeframeConverter(
            source_timeframe="1T",
            target_timeframe="5T"
        )
        
        # Generate streaming data
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        
        all_completed = []
        last_incomplete = None
        
        for minute in range(30):  # 30 minutes = 6 complete 5-min bars
            # Create single minute bar
            timestamp = base_time + timedelta(minutes=minute)
            minute_bar = pl.DataFrame({
                "timestamp": [timestamp],
                "open": [150.0],
                "high": [150.1],
                "low": [149.9],
                "close": [150.05],
                "volume": [100]
            })
            
            # Convert streaming
            if last_incomplete is not None:
                # Combine with incomplete bar from previous iteration
                data_to_convert = pl.concat([last_incomplete, minute_bar])
            else:
                data_to_convert = minute_bar
            
            completed, incomplete = self.converter.convert_streaming(
                data_to_convert,
                last_complete_timestamp=None
            )
            
            if completed is not None and len(completed) > 0:
                all_completed.append(completed)
            
            last_incomplete = incomplete
        
        # Verify results
        if all_completed:
            final_result = pl.concat(all_completed)
            assert len(final_result) == 6  # Should have 6 complete 5-minute bars
            
            # Verify each bar represents 5 minutes of data
            for i in range(len(final_result) - 1):
                time_diff = (final_result["timestamp"][i+1] - final_result["timestamp"][i]).total_seconds()
                assert time_diff == 300  # 5 minutes in seconds


class TestRCICalculationE2E:
    """End-to-end tests for RCI calculation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.rci_engine = None
        self.base_time = datetime.now()
    
    def test_rci_calculation_accuracy(self):
        """Test accuracy of RCI calculation with known patterns."""
        # Create perfectly trending data
        n_bars = 100
        timestamps = []
        prices = []
        
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        
        # Perfect uptrend
        for i in range(n_bars):
            timestamps.append(base_time + timedelta(minutes=i))
            prices.append(150.0 + i * 0.1)  # Linear increase
        
        uptrend_data = pl.DataFrame({
            "timestamp": timestamps,
            "close": prices
        })
        
        # Calculate RCI
        self.rci_engine = RCICalculatorEngine(periods=[9, 24])
        rci_result = self.rci_engine.calculate(uptrend_data, column_name="close")
        
        # For perfect uptrend, RCI should be close to +100
        rci_9 = rci_result["rci_9"].drop_nulls()
        assert rci_9.mean() > 90  # Should be very high for perfect uptrend
        
        # Test perfect downtrend
        downtrend_prices = [150.0 - i * 0.1 for i in range(n_bars)]
        downtrend_data = pl.DataFrame({
            "timestamp": timestamps,
            "close": downtrend_prices
        })
        
        rci_result_down = self.rci_engine.calculate(downtrend_data, column_name="close")
        rci_9_down = rci_result_down["rci_9"].drop_nulls()
        assert rci_9_down.mean() < -90  # Should be very low for perfect downtrend
    
    def test_rci_with_various_periods(self):
        """Test RCI calculation with different period settings."""
        # Generate random walk data
        n_bars = 500
        timestamps = []
        prices = []
        
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        base_price = 150.0
        
        for i in range(n_bars):
            timestamps.append(base_time + timedelta(minutes=i))
            base_price *= (1 + np.random.normal(0, 0.001))
            prices.append(base_price)
        
        test_data = pl.DataFrame({
            "timestamp": timestamps,
            "close": prices
        })
        
        # Test with multiple periods
        periods = [9, 13, 24, 33, 48, 66, 108]
        self.rci_engine = RCICalculatorEngine(periods=periods)
        
        result = self.rci_engine.calculate(test_data, column_name="close")
        
        # Verify all period columns exist
        for period in periods:
            assert f"rci_{period}" in result.columns
            
            # Check values are in valid range
            values = result[f"rci_{period}"].drop_nulls()
            assert values.min() >= -100.1
            assert values.max() <= 100.1
            
            # Longer periods should have fewer null values at the start
            null_count = result[f"rci_{period}"].null_count()
            assert null_count >= period - 1  # At least period-1 nulls at start


class TestPerformanceE2E:
    """Performance benchmark tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = None
        self.converter = None
        self.base_time = datetime.now()
    
    def test_large_dataset_processing(self):
        """Test performance with large datasets."""
        # Generate large dataset
        n_bars = 10000  # 10,000 minutes ≈ 1 week of data
        
        print(f"\nGenerating {n_bars} bars of test data...")
        start_time = time.perf_counter()
        
        timestamps = []
        prices = []
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        base_price = 150.0
        
        for i in range(n_bars):
            timestamps.append(base_time + timedelta(minutes=i))
            base_price *= (1 + np.random.normal(0, 0.0001))
            prices.append(base_price)
        
        large_data = pl.DataFrame({
            "timestamp": timestamps,
            "open": prices,
            "high": [p * 1.0001 for p in prices],
            "low": [p * 0.9999 for p in prices],
            "close": prices,
            "volume": [100] * n_bars
        })
        
        data_gen_time = time.perf_counter() - start_time
        print(f"Data generation took {data_gen_time:.2f} seconds")
        
        # Test multiframe analysis performance
        self.analyzer = MultiTimeframeAnalyzer(
            short_term_periods=[9, 13, 24],
            long_term_periods=[24, 33, 48],
            use_parallel=True
        )
        
        print("Starting multiframe analysis...")
        start_time = time.perf_counter()
        
        result = self.analyzer.analyze(large_data)
        
        analysis_time = time.perf_counter() - start_time
        print(f"Analysis took {analysis_time:.2f} seconds")
        
        # Performance assertions
        assert analysis_time < 10, f"Analysis took {analysis_time:.2f}s (exceeds 10s limit)"
        assert result is not None
        assert len(result) == n_bars
        
        # Calculate throughput
        throughput = n_bars / analysis_time
        print(f"Throughput: {throughput:.1f} bars/second")
        assert throughput > 1000, f"Throughput {throughput:.1f} bars/s below 1000 bars/s"
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency."""
        # Start memory tracking
        tracemalloc.start()
        
        # Create analyzer with history limit
        self.analyzer = MultiTimeframeAnalyzer(
            short_term_periods=[9, 13, 24, 33, 48],
            long_term_periods=[24, 33, 48, 66, 108],
            use_parallel=False
        )
        
        # Generate and process data in chunks
        chunk_size = 100
        n_chunks = 50
        
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        
        for chunk_idx in range(n_chunks):
            # Generate chunk
            timestamps = []
            prices = []
            
            for i in range(chunk_size):
                minute_offset = chunk_idx * chunk_size + i
                timestamps.append(base_time + timedelta(minutes=minute_offset))
                prices.append(150.0 + np.random.normal(0, 0.1))
            
            chunk_data = pl.DataFrame({
                "timestamp": timestamps,
                "open": prices,
                "high": [p * 1.001 for p in prices],
                "low": [p * 0.999 for p in prices],
                "close": prices,
                "volume": [100] * chunk_size
            })
            
            # Process chunk
            _ = self.analyzer.analyze(chunk_data)
        
        # Check memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 1024 / 1024
        print(f"Peak memory usage: {peak_mb:.1f} MB")
        
        # Memory should stay bounded despite processing many chunks
        assert peak_mb < 500, f"Peak memory {peak_mb:.1f}MB exceeds 500MB limit"
    
    def test_parallel_vs_sequential_performance(self):
        """Compare performance of parallel vs sequential processing."""
        # Generate test data
        n_bars = 2000
        timestamps = []
        prices = []
        
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        base_price = 150.0
        
        for i in range(n_bars):
            timestamps.append(base_time + timedelta(minutes=i))
            base_price *= (1 + np.random.normal(0, 0.0001))
            prices.append(base_price)
        
        test_data = pl.DataFrame({
            "timestamp": timestamps,
            "open": prices,
            "high": [p * 1.0001 for p in prices],
            "low": [p * 0.9999 for p in prices],
            "close": prices,
            "volume": [100] * n_bars
        })
        
        # Test sequential processing
        analyzer_seq = MultiTimeframeAnalyzer(
            short_term_periods=[9, 13, 24, 33, 48],
            long_term_periods=[24, 33, 48, 66, 108],
            use_parallel=False
        )
        
        start_time = time.perf_counter()
        result_seq = analyzer_seq.analyze(test_data)
        seq_time = time.perf_counter() - start_time
        
        # Test parallel processing
        analyzer_par = MultiTimeframeAnalyzer(
            short_term_periods=[9, 13, 24, 33, 48],
            long_term_periods=[24, 33, 48, 66, 108],
            use_parallel=True
        )
        
        start_time = time.perf_counter()
        result_par = analyzer_par.analyze(test_data)
        par_time = time.perf_counter() - start_time
        
        print(f"\nSequential processing: {seq_time:.3f} seconds")
        print(f"Parallel processing: {par_time:.3f} seconds")
        print(f"Speedup: {seq_time / par_time:.2f}x")
        
        # Parallel should be faster (at least not slower)
        assert par_time <= seq_time * 1.1  # Allow 10% tolerance
        
        # Results should be identical
        for col in result_seq.columns:
            if col.startswith("rci_"):
                seq_values = result_seq[col].drop_nulls()
                par_values = result_par[col].drop_nulls()
                
                if len(seq_values) > 0 and len(par_values) > 0:
                    # Values should be very close (floating point tolerance)
                    diff = abs(seq_values - par_values).max()
                    assert diff < 0.01, f"Column {col} differs by {diff}"


class TestErrorHandlingE2E:
    """End-to-end tests for error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = None
        self.converter = None
    
    def test_handling_missing_data(self):
        """Test handling of missing data points."""
        # Create data with gaps
        timestamps = []
        prices = []
        
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        
        # Create data with 10-minute gap
        for i in range(50):
            timestamps.append(base_time + timedelta(minutes=i))
            prices.append(150.0 + np.random.normal(0, 0.1))
        
        # Skip 10 minutes
        for i in range(60, 100):
            timestamps.append(base_time + timedelta(minutes=i))
            prices.append(151.0 + np.random.normal(0, 0.1))
        
        gapped_data = pl.DataFrame({
            "timestamp": timestamps,
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": [100] * len(prices)
        })
        
        # Analyzer should handle gaps gracefully
        self.analyzer = MultiTimeframeAnalyzer(
            short_term_periods=[9],
            long_term_periods=[24],
            use_parallel=False
        )
        
        result = self.analyzer.analyze(gapped_data)
        assert result is not None
        assert len(result) == len(gapped_data)
    
    def test_handling_extreme_values(self):
        """Test handling of extreme price values."""
        # Create data with extreme values
        n_bars = 100
        timestamps = []
        prices = []
        
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        
        for i in range(n_bars):
            timestamps.append(base_time + timedelta(minutes=i))
            
            if i == 50:
                # Inject extreme spike
                prices.append(1500.0)  # 10x normal price
            elif i == 51:
                # Return to normal
                prices.append(150.0)
            else:
                prices.append(150.0 + np.random.normal(0, 0.1))
        
        extreme_data = pl.DataFrame({
            "timestamp": timestamps,
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": [100] * n_bars
        })
        
        # Components should handle extreme values
        self.converter = TimeframeConverter(
            source_timeframe="1T",
            target_timeframe="5T"
        )
        
        converted = self.converter.convert(extreme_data)
        assert converted is not None
        
        # Check that extreme value is preserved in conversion
        max_high = converted["high"].max()
        assert max_high >= 1500.0  # Extreme value should be captured
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        empty_data = pl.DataFrame({
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": []
        })
        
        # Converter should handle empty data
        self.converter = TimeframeConverter(
            source_timeframe="1T",
            target_timeframe="5T"
        )
        
        result = self.converter.convert(empty_data)
        assert len(result) == 0
        
        # Analyzer should handle empty data
        self.analyzer = MultiTimeframeAnalyzer(
            short_term_periods=[9],
            long_term_periods=[24],
            use_parallel=False
        )
        
        result = self.analyzer.analyze(empty_data)
        assert len(result) == 0
    
    def test_single_bar_handling(self):
        """Test handling of single bar data."""
        single_bar = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 9, 0, 0)],
            "open": [150.0],
            "high": [150.1],
            "low": [149.9],
            "close": [150.05],
            "volume": [100]
        })
        
        # Converter should handle single bar
        self.converter = TimeframeConverter(
            source_timeframe="1T",
            target_timeframe="5T"
        )
        
        result = self.converter.convert(single_bar)
        assert len(result) <= 1
        
        # RCI engine should handle single bar (but return NaN)
        rci_engine = RCICalculatorEngine(periods=[9])
        result = rci_engine.calculate(single_bar, column_name="close")
        assert len(result) == 1
        assert result["rci_9"][0] is None or pl.Series([result["rci_9"][0]]).is_null()[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
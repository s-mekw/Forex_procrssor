"""Test for pipelines.py bug fix - get_metrics buffer size issue."""

import asyncio
from unittest.mock import Mock, MagicMock
import pytest

from src.data_processing.pipelines import RealtimePipeline


class TestPipelinesBugFix:
    """Test suite to verify the buffer size bug fix in pipelines.py."""

    def test_get_metrics_with_multiframe_analyzer(self):
        """
        Test that get_metrics correctly calls get_buffer_size() on the analyzer
        instead of trying to access the removed _data_buffer attribute.
        """
        # Create a mock analyzer that implements the protocol
        mock_analyzer = Mock()
        mock_analyzer.add_new_bar = Mock()
        mock_analyzer.is_ready = Mock(return_value=True)
        mock_analyzer.analyze_streaming = Mock(return_value={"test": "result"})
        mock_analyzer.get_buffer_size = Mock(return_value=150)  # Mock buffer size
        
        # Create pipeline with the mock analyzer
        pipeline = RealtimePipeline(
            enable_multiframe=True,
            analyzer=mock_analyzer
        )
        
        # Get metrics - this should not raise AttributeError
        metrics = pipeline.get_metrics()
        
        # Verify the analyzer's get_buffer_size was called
        mock_analyzer.get_buffer_size.assert_called_once()
        
        # Verify the buffer size is in the metrics
        assert "data_buffer_size" in metrics
        assert metrics["data_buffer_size"] == 150
        
    def test_get_metrics_without_multiframe(self):
        """
        Test that get_metrics works correctly when multiframe is disabled.
        """
        # Create pipeline without multiframe
        pipeline = RealtimePipeline(
            enable_multiframe=False
        )
        
        # Get metrics - should not have data_buffer_size
        metrics = pipeline.get_metrics()
        
        # Verify data_buffer_size is not in metrics when multiframe is disabled
        assert "data_buffer_size" not in metrics
        
    @pytest.mark.asyncio
    async def test_pipeline_processing_with_mock_analyzer(self):
        """
        Test that the pipeline processes data correctly with a mock analyzer.
        """
        # Create a mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.add_new_bar = Mock()
        mock_analyzer.is_ready = Mock(return_value=True)
        mock_analyzer.analyze_streaming = Mock(return_value={
            "short_rci": -50.0,
            "long_rci": 25.0,
            "is_new_short_bar": False,
            "is_new_long_bar": False
        })
        mock_analyzer.get_buffer_size = Mock(return_value=100)
        
        # Create pipeline with mock analyzer
        pipeline = RealtimePipeline(
            enable_multiframe=True,
            analyzer=mock_analyzer
        )
        
        # Start the pipeline
        asyncio.create_task(pipeline.start())
        
        # Give the pipeline time to start
        await asyncio.sleep(0.1)
        
        # Submit some test data
        test_data = {
            "timestamp": "2024-01-01 12:00:00",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000
        }
        
        await pipeline.submit(test_data)
        
        # Process the data
        result = await pipeline.get_result()
        
        assert result is not None
        assert result["status"] == "success"
        assert "multiframe_rci" in result
        
        # Verify analyzer methods were called
        mock_analyzer.add_new_bar.assert_called()
        mock_analyzer.is_ready.assert_called()
        mock_analyzer.analyze_streaming.assert_called()
        
        # Stop the pipeline
        await pipeline.stop()
        
    def test_ruff_compliance(self):
        """
        Test that the file passes ruff checks.
        This is a placeholder to verify that our fixes comply with linting rules.
        """
        import subprocess
        result = subprocess.run(
            ["uv", "run", "ruff", "check", "src/data_processing/pipelines.py"],
            capture_output=True,
            text=True
        )
        # Check if ruff passes (exit code 0)
        assert result.returncode == 0, f"Ruff check failed: {result.stderr}"
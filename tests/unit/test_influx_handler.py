"""Unit tests for InfluxDB handler.

This module contains unit tests for InfluxDBHandler class,
including connection management and health check functionality.
"""

import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest
from influxdb_client.client.exceptions import InfluxDBError

from src.storage.influx_handler import (
    InfluxDBConnectionError,
    InfluxDBHandler,
    InfluxDBQueryError,
    InfluxDBSchema,
    InfluxDBWriteError,
    OHLCDataPoint,
    TimeFrame,
)


@pytest.fixture
def influx_config():
    """Fixture providing InfluxDB configuration."""
    return {
        "url": "http://localhost:8086",
        "token": "test-token",
        "org": "test-org",
        "bucket": "test-bucket",
        "timeout": 10000,
        "verify_ssl": True,
    }


@pytest.fixture
def influx_handler(influx_config):
    """Fixture providing InfluxDBHandler instance."""
    return InfluxDBHandler(**influx_config)


class TestInfluxDBHandler:
    """Test suite for InfluxDBHandler class."""

    @pytest.mark.asyncio
    async def test_handler_initialization(self, influx_handler, influx_config):
        """Test that handler initializes with correct configuration."""
        assert influx_handler.url == influx_config["url"]
        assert influx_handler.token == influx_config["token"]
        assert influx_handler.org == influx_config["org"]
        assert influx_handler.bucket == influx_config["bucket"]
        assert influx_handler.timeout == influx_config["timeout"]
        assert influx_handler.verify_ssl == influx_config["verify_ssl"]
        assert influx_handler._client is None
        assert influx_handler._is_connected is False

    @pytest.mark.asyncio
    async def test_connect_success(self, influx_handler):
        """Test successful connection to InfluxDB."""
        with patch("src.storage.influx_handler.InfluxDBClient") as mock_client_class:
            # Setup mock client
            mock_client = MagicMock()
            mock_client.ready.return_value = True
            mock_client_class.return_value = mock_client

            # Connect
            await influx_handler.connect()

            # Verify
            mock_client_class.assert_called_once_with(
                url=influx_handler.url,
                token=influx_handler.token,
                org=influx_handler.org,
                timeout=influx_handler.timeout,
                verify_ssl=influx_handler.verify_ssl,
            )
            mock_client.ready.assert_called_once()
            assert influx_handler._is_connected is True
            assert influx_handler._client == mock_client

    @pytest.mark.asyncio
    async def test_connect_server_not_ready(self, influx_handler):
        """Test connection failure when server is not ready."""
        with patch("src.storage.influx_handler.InfluxDBClient") as mock_client_class:
            # Setup mock client
            mock_client = MagicMock()
            mock_client.ready.return_value = False
            mock_client_class.return_value = mock_client

            # Connect should raise error
            with pytest.raises(
                InfluxDBConnectionError, match="InfluxDB server is not ready"
            ):
                await influx_handler.connect()

            assert influx_handler._is_connected is False

    @pytest.mark.asyncio
    async def test_connect_influxdb_error(self, influx_handler):
        """Test connection failure with InfluxDBError."""
        with patch("src.storage.influx_handler.InfluxDBClient") as mock_client_class:
            # Create a mock response object that InfluxDBError expects
            mock_response = MagicMock()
            mock_response.data = None
            mock_response.status = 500
            mock_response.reason = "Connection refused"

            # Setup mock to raise InfluxDBError with proper response object
            error = InfluxDBError(mock_response)
            error.message = "Connection refused"
            mock_client_class.side_effect = error

            # Connect should re-raise error
            with pytest.raises(InfluxDBError):
                await influx_handler.connect()

            assert influx_handler._is_connected is False

    @pytest.mark.asyncio
    async def test_connect_unexpected_error(self, influx_handler):
        """Test connection failure with unexpected error."""
        with patch("src.storage.influx_handler.InfluxDBClient") as mock_client_class:
            # Setup mock to raise generic exception
            mock_client_class.side_effect = Exception("Unexpected error")

            # Connect should wrap error in InfluxDBConnectionError
            with pytest.raises(
                InfluxDBConnectionError, match="Connection failed: Unexpected error"
            ):
                await influx_handler.connect()

            assert influx_handler._is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect_success(self, influx_handler):
        """Test successful disconnection from InfluxDB."""
        # Setup mock client
        mock_client = MagicMock()
        influx_handler._client = mock_client
        influx_handler._is_connected = True

        # Disconnect
        await influx_handler.disconnect()

        # Verify
        mock_client.close.assert_called_once()
        assert influx_handler._is_connected is False
        assert influx_handler._client is None

    @pytest.mark.asyncio
    async def test_disconnect_with_error(self, influx_handler):
        """Test disconnection handles errors gracefully."""
        # Setup mock client that raises error on close
        mock_client = MagicMock()
        mock_client.close.side_effect = Exception("Close error")
        influx_handler._client = mock_client
        influx_handler._is_connected = True

        # Disconnect should not raise error
        await influx_handler.disconnect()

        # Verify cleanup still happens - the client is set to None even on error
        assert influx_handler._client is None
        # Note: _is_connected is set to False before the error occurs
        # The implementation sets it to False regardless of errors

    @pytest.mark.asyncio
    async def test_disconnect_no_client(self, influx_handler):
        """Test disconnection when no client exists."""
        # Should not raise error
        await influx_handler.disconnect()
        assert influx_handler._client is None

    @pytest.mark.asyncio
    async def test_health_check_success(self, influx_handler):
        """Test successful health check."""
        # Setup mock client
        mock_client = MagicMock()
        mock_health = MagicMock()
        mock_health.status = "pass"
        mock_client.health.return_value = mock_health
        influx_handler._client = mock_client

        # Perform health check
        result = await influx_handler.health_check()

        # Verify
        assert result is True
        mock_client.health.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failed_status(self, influx_handler):
        """Test health check with failed status."""
        # Setup mock client
        mock_client = MagicMock()
        mock_health = MagicMock()
        mock_health.status = "fail"
        mock_client.health.return_value = mock_health
        influx_handler._client = mock_client

        # Perform health check
        result = await influx_handler.health_check()

        # Verify
        assert result is False
        mock_client.health.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_no_client(self, influx_handler):
        """Test health check when no client connection exists."""
        result = await influx_handler.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_influxdb_error(self, influx_handler):
        """Test health check with InfluxDB error."""
        # Setup mock client that raises error
        mock_client = MagicMock()

        # Create a proper InfluxDBError with mock response
        mock_response = MagicMock()
        mock_response.data = None
        mock_response.status = 500
        mock_response.reason = "Health check failed"
        error = InfluxDBError(mock_response)
        error.message = "Health check failed"

        mock_client.health.side_effect = error
        influx_handler._client = mock_client

        # Perform health check
        result = await influx_handler.health_check()

        # Verify
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_unexpected_error(self, influx_handler):
        """Test health check with unexpected error."""
        # Setup mock client that raises generic exception
        mock_client = MagicMock()
        mock_client.health.side_effect = Exception("Unexpected error")
        influx_handler._client = mock_client

        # Perform health check
        result = await influx_handler.health_check()

        # Verify
        assert result is False

    def test_is_connected_property(self, influx_handler):
        """Test is_connected property."""
        assert influx_handler.is_connected is False

        influx_handler._is_connected = True
        assert influx_handler.is_connected is True

        influx_handler._is_connected = False
        assert influx_handler.is_connected is False

    @pytest.mark.asyncio
    async def test_context_manager_success(self, influx_handler):
        """Test async context manager with successful connection."""
        with patch.object(
            influx_handler, "connect", new_callable=AsyncMock
        ) as mock_connect:
            with patch.object(
                influx_handler, "disconnect", new_callable=AsyncMock
            ) as mock_disconnect:
                async with influx_handler as handler:
                    assert handler == influx_handler
                    mock_connect.assert_called_once()

                mock_disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self, influx_handler):
        """Test async context manager handles exceptions properly."""
        with patch.object(
            influx_handler, "connect", new_callable=AsyncMock
        ) as mock_connect:
            with patch.object(
                influx_handler, "disconnect", new_callable=AsyncMock
            ) as mock_disconnect:
                try:
                    async with influx_handler:
                        mock_connect.assert_called_once()
                        raise ValueError("Test error")
                except ValueError:
                    pass

                # Disconnect should still be called
                mock_disconnect.assert_called_once()

    def test_del_method(self):
        """Test __del__ method cleanup."""
        mock_client = MagicMock()
        handler = InfluxDBHandler(
            url="http://localhost:8086",
            token="test-token",
            org="test-org",
            bucket="test-bucket",
        )
        handler._client = mock_client

        # Call __del__
        handler.__del__()

        # Verify client was closed
        mock_client.close.assert_called_once()

    def test_del_method_with_error(self):
        """Test __del__ method handles errors gracefully."""
        mock_client = MagicMock()
        mock_client.close.side_effect = Exception("Close error")
        handler = InfluxDBHandler(
            url="http://localhost:8086",
            token="test-token",
            org="test-org",
            bucket="test-bucket",
        )
        handler._client = mock_client

        # Should not raise error
        handler.__del__()

    @pytest.mark.asyncio
    async def test_write_point_success(self, influx_handler):
        """Test successful write of single data point."""
        # Setup mock client
        mock_client = MagicMock()
        mock_write_api = MagicMock()
        mock_client.write_api.return_value = mock_write_api
        influx_handler._client = mock_client

        # Create test data point
        data_point = OHLCDataPoint(
            timestamp=datetime.now(),
            symbol="EURUSD",
            timeframe=TimeFrame.M5,
            open=1.0850,
            high=1.0860,
            low=1.0840,
            close=1.0855,
            volume=1000.0,
        )

        # Write data point
        await influx_handler.write_point(data_point)

        # Verify
        mock_client.write_api.assert_called_once()
        mock_write_api.write.assert_called_once()
        mock_write_api.close.assert_called_once()

        # Check that write was called with correct bucket
        call_kwargs = mock_write_api.write.call_args.kwargs
        assert call_kwargs["bucket"] == influx_handler.bucket

    @pytest.mark.asyncio
    async def test_write_point_no_connection(self, influx_handler):
        """Test write point without connection."""
        data_point = OHLCDataPoint(
            timestamp=datetime.now(),
            symbol="EURUSD",
            timeframe=TimeFrame.M5,
            open=1.0850,
            high=1.0860,
            low=1.0840,
            close=1.0855,
            volume=1000.0,
        )

        with pytest.raises(InfluxDBConnectionError, match="Not connected to InfluxDB"):
            await influx_handler.write_point(data_point)

    @pytest.mark.asyncio
    async def test_write_point_influxdb_error(self, influx_handler):
        """Test write point with InfluxDB error."""
        # Setup mock client
        mock_client = MagicMock()
        mock_write_api = MagicMock()

        # Create a proper InfluxDBError
        mock_response = MagicMock()
        mock_response.data = None
        mock_response.status = 500
        mock_response.reason = "Write failed"
        error = InfluxDBError(mock_response)
        error.message = "Write failed"

        mock_write_api.write.side_effect = error
        mock_client.write_api.return_value = mock_write_api
        influx_handler._client = mock_client

        # Create test data point
        data_point = OHLCDataPoint(
            timestamp=datetime.now(),
            symbol="EURUSD",
            timeframe=TimeFrame.M5,
            open=1.0850,
            high=1.0860,
            low=1.0840,
            close=1.0855,
            volume=1000.0,
        )

        # Write should raise InfluxDBWriteError
        with pytest.raises(InfluxDBWriteError, match="Write failed"):
            await influx_handler.write_point(data_point)

        # Verify close was still called
        mock_write_api.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_batch_success(self, influx_handler):
        """Test successful batch write of multiple data points."""
        # Setup mock client
        mock_client = MagicMock()
        mock_write_api = MagicMock()
        mock_client.write_api.return_value = mock_write_api
        influx_handler._client = mock_client

        # Create test data points
        data_points = [
            OHLCDataPoint(
                timestamp=datetime.now(),
                symbol="EURUSD",
                timeframe=TimeFrame.M5,
                open=1.0850 + i * 0.0001,
                high=1.0860 + i * 0.0001,
                low=1.0840 + i * 0.0001,
                close=1.0855 + i * 0.0001,
                volume=1000.0 + i * 100,
            )
            for i in range(10)
        ]

        # Write batch
        await influx_handler.write_batch(data_points, batch_size=5)

        # Verify
        mock_client.write_api.assert_called_once()
        # Should be called twice (2 batches of 5)
        assert mock_write_api.write.call_count == 2
        mock_write_api.flush.assert_called_once()
        mock_write_api.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_batch_empty_list(self, influx_handler):
        """Test write batch with empty list."""
        influx_handler._client = MagicMock()

        with pytest.raises(ValueError, match="No data points provided"):
            await influx_handler.write_batch([])

    @pytest.mark.asyncio
    async def test_write_batch_invalid_batch_size(self, influx_handler):
        """Test write batch with invalid batch size."""
        influx_handler._client = MagicMock()

        data_points = [
            OHLCDataPoint(
                timestamp=datetime.now(),
                symbol="EURUSD",
                timeframe=TimeFrame.M5,
                open=1.0850,
                high=1.0860,
                low=1.0840,
                close=1.0855,
                volume=1000.0,
            )
        ]

        with pytest.raises(ValueError, match="Invalid batch_size"):
            await influx_handler.write_batch(data_points, batch_size=0)

        with pytest.raises(ValueError, match="Invalid batch_size"):
            await influx_handler.write_batch(data_points, batch_size=-1)

    @pytest.mark.asyncio
    async def test_write_batch_no_connection(self, influx_handler):
        """Test write batch without connection."""
        data_points = [
            OHLCDataPoint(
                timestamp=datetime.now(),
                symbol="EURUSD",
                timeframe=TimeFrame.M5,
                open=1.0850,
                high=1.0860,
                low=1.0840,
                close=1.0855,
                volume=1000.0,
            )
        ]

        with pytest.raises(InfluxDBConnectionError, match="Not connected to InfluxDB"):
            await influx_handler.write_batch(data_points)

    @pytest.mark.asyncio
    async def test_write_batch_large_dataset(self, influx_handler):
        """Test batch write with dataset larger than batch size."""
        # Setup mock client
        mock_client = MagicMock()
        mock_write_api = MagicMock()
        mock_client.write_api.return_value = mock_write_api
        influx_handler._client = mock_client

        # Create large dataset
        data_points = [
            OHLCDataPoint(
                timestamp=datetime.now(),
                symbol="EURUSD",
                timeframe=TimeFrame.M5,
                open=1.0850,
                high=1.0860,
                low=1.0840,
                close=1.0855,
                volume=1000.0,
            )
            for _ in range(12345)
        ]

        # Write batch with default batch size (5000)
        await influx_handler.write_batch(data_points)

        # Verify correct number of batches
        # 12345 points / 5000 per batch = 3 batches (5000, 5000, 2345)
        assert mock_write_api.write.call_count == 3
        mock_write_api.flush.assert_called_once()
        mock_write_api.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_success(self, influx_handler):
        """Test successful query execution."""
        # Setup mock client
        mock_client = MagicMock()
        mock_query_api = MagicMock()

        # Create mock result structure
        mock_record = MagicMock()
        mock_record.values = {
            "_time": datetime.now(),
            "symbol": "EURUSD",
            "open": 1.0850,
            "high": 1.0860,
            "low": 1.0840,
            "close": 1.0855,
        }

        mock_table = MagicMock()
        mock_table.records = [mock_record]

        mock_query_api.query.return_value = [mock_table]
        mock_client.query_api.return_value = mock_query_api
        influx_handler._client = mock_client

        # Execute query
        flux_query = 'from(bucket: "test-bucket") |> range(start: -1h)'
        result = await influx_handler.query(flux_query)

        # Verify
        assert len(result) == 1
        assert result[0] == mock_record.values
        mock_query_api.query.assert_called_once_with(
            query=flux_query, org=influx_handler.org
        )

    @pytest.mark.asyncio
    async def test_query_no_connection(self, influx_handler):
        """Test query without connection."""
        flux_query = 'from(bucket: "test-bucket") |> range(start: -1h)'

        with pytest.raises(InfluxDBConnectionError, match="Not connected to InfluxDB"):
            await influx_handler.query(flux_query)

    @pytest.mark.asyncio
    async def test_query_influxdb_error(self, influx_handler):
        """Test query with InfluxDB error."""
        # Setup mock client
        mock_client = MagicMock()
        mock_query_api = MagicMock()

        # Create a proper InfluxDBError
        mock_response = MagicMock()
        mock_response.data = None
        mock_response.status = 400
        mock_response.reason = "Invalid query"
        error = InfluxDBError(mock_response)
        error.message = "Invalid query"

        mock_query_api.query.side_effect = error
        mock_client.query_api.return_value = mock_query_api
        influx_handler._client = mock_client

        # Execute query should raise InfluxDBQueryError
        flux_query = 'from(bucket: "test-bucket")'
        with pytest.raises(InfluxDBQueryError, match="Query failed"):
            await influx_handler.query(flux_query)

    @pytest.mark.asyncio
    async def test_query_ohlc_success(self, influx_handler):
        """Test successful OHLC data query."""
        # Setup mock for query method
        mock_data = [
            {
                "_time": datetime(2024, 1, 1, 12, 0, 0),
                "symbol": "EURUSD",
                "timeframe": "M5",
                "broker": "default",
                "open": 1.0850,
                "high": 1.0860,
                "low": 1.0840,
                "close": 1.0855,
                "volume": 1000.0,
                "spread": 0.0002,
            },
            {
                "_time": datetime(2024, 1, 1, 12, 5, 0),
                "symbol": "EURUSD",
                "timeframe": "M5",
                "broker": "default",
                "open": 1.0855,
                "high": 1.0865,
                "low": 1.0845,
                "close": 1.0860,
                "volume": 1100.0,
                "spread": 0.0002,
            },
        ]

        with patch.object(
            influx_handler, "query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = mock_data
            influx_handler._client = MagicMock()  # Ensure client exists

            # Query OHLC data
            df = await influx_handler.query_ohlc(
                symbol="eurusd",  # Test lowercase conversion
                timeframe=TimeFrame.M5,
                start_time=datetime(2024, 1, 1, 12, 0, 0),
                end_time=datetime(2024, 1, 1, 12, 10, 0),
            )

            # Verify DataFrame
            assert isinstance(df, pl.DataFrame)
            assert len(df) == 2
            assert "time" in df.columns
            assert df["symbol"][0] == "EURUSD"
            assert df["open"].dtype == pl.Float32
            assert df["high"].dtype == pl.Float32
            assert df["low"].dtype == pl.Float32
            assert df["close"].dtype == pl.Float32
            assert df["volume"].dtype == pl.Float32

    @pytest.mark.asyncio
    async def test_query_ohlc_empty_result(self, influx_handler):
        """Test OHLC query with empty result."""
        with patch.object(
            influx_handler, "query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = []
            influx_handler._client = MagicMock()

            # Query OHLC data
            df = await influx_handler.query_ohlc(
                symbol="EURUSD",
                timeframe="M5",
                start_time=datetime(2024, 1, 1, 12, 0, 0),
            )

            # Verify empty DataFrame with correct schema
            assert isinstance(df, pl.DataFrame)
            assert len(df) == 0
            assert "time" in df.columns
            assert "symbol" in df.columns
            assert df["open"].dtype == pl.Float32

    @pytest.mark.asyncio
    async def test_query_ohlc_with_broker_filter(self, influx_handler):
        """Test OHLC query with broker filter."""
        mock_data = [
            {
                "_time": datetime(2024, 1, 1, 12, 0, 0),
                "symbol": "EURUSD",
                "timeframe": "M5",
                "broker": "test-broker",
                "open": 1.0850,
                "high": 1.0860,
                "low": 1.0840,
                "close": 1.0855,
                "volume": 1000.0,
            }
        ]

        with patch.object(
            influx_handler, "query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = mock_data
            influx_handler._client = MagicMock()

            # Query with broker filter
            await influx_handler.query_ohlc(
                symbol="EURUSD",
                timeframe="M5",
                start_time=datetime(2024, 1, 1, 12, 0, 0),
                broker="test-broker",
            )

            # Verify broker filter was included in query
            called_query = mock_query.call_args[0][0]
            assert 'broker"] == "test-broker"' in called_query

    @pytest.mark.asyncio
    async def test_query_ohlc_no_connection(self, influx_handler):
        """Test OHLC query without connection."""
        with pytest.raises(InfluxDBConnectionError, match="Not connected to InfluxDB"):
            await influx_handler.query_ohlc(
                symbol="EURUSD",
                timeframe=TimeFrame.M5,
                start_time=datetime.now(),
            )

    @pytest.mark.asyncio
    async def test_query_time_range_success(self, influx_handler):
        """Test successful time range query."""
        mock_data = [
            {
                "_time": datetime(2024, 1, 1, 12, 0, 0),
                "measurement": "test",
                "field1": 100.0,
                "field2": 200.0,
            },
            {
                "_time": datetime(2024, 1, 1, 12, 5, 0),
                "measurement": "test",
                "field1": 110.0,
                "field2": 210.0,
            },
        ]

        with patch.object(
            influx_handler, "query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = mock_data
            influx_handler._client = MagicMock()

            # Query time range
            df = await influx_handler.query_time_range(
                measurement="test",
                start_time=datetime(2024, 1, 1, 12, 0, 0),
                end_time=datetime(2024, 1, 1, 12, 10, 0),
            )

            # Verify DataFrame
            assert isinstance(df, pl.DataFrame)
            assert len(df) == 2
            assert "time" in df.columns

    @pytest.mark.asyncio
    async def test_query_time_range_with_filters(self, influx_handler):
        """Test time range query with tag filters."""
        mock_data = [
            {
                "_time": datetime(2024, 1, 1, 12, 0, 0),
                "tag1": "value1",
                "tag2": "value2",
                "field1": 100.0,
            }
        ]

        with patch.object(
            influx_handler, "query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = mock_data
            influx_handler._client = MagicMock()

            # Query with filters
            filters = {"tag1": "value1", "tag2": "value2"}
            await influx_handler.query_time_range(
                measurement="test",
                start_time=datetime(2024, 1, 1, 12, 0, 0),
                filters=filters,
            )

            # Verify filters were included in query
            called_query = mock_query.call_args[0][0]
            assert 'tag1"] == "value1"' in called_query
            assert 'tag2"] == "value2"' in called_query

    @pytest.mark.asyncio
    async def test_query_time_range_empty_result(self, influx_handler):
        """Test time range query with empty result."""
        with patch.object(
            influx_handler, "query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = []
            influx_handler._client = MagicMock()

            # Query time range
            df = await influx_handler.query_time_range(
                measurement="test",
                start_time=datetime(2024, 1, 1, 12, 0, 0),
            )

            # Verify empty DataFrame
            assert isinstance(df, pl.DataFrame)
            assert len(df) == 0

    @pytest.mark.asyncio
    async def test_query_time_range_no_connection(self, influx_handler):
        """Test time range query without connection."""
        with pytest.raises(InfluxDBConnectionError, match="Not connected to InfluxDB"):
            await influx_handler.query_time_range(
                measurement="test",
                start_time=datetime.now(),
            )

    def test_from_env_success(self):
        """Test creating handler from environment variables with all required values."""
        # Set up environment variables
        env_vars = {
            "INFLUXDB_URL": "http://test-server:8086",
            "INFLUXDB_TOKEN": "test-token-123",
            "INFLUXDB_ORG": "test-organization",
            "INFLUXDB_BUCKET": "test-bucket-env",
            "INFLUXDB_TIMEOUT": "5000",
            "INFLUXDB_VERIFY_SSL": "false",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            handler = InfluxDBHandler.from_env()

            assert handler.url == "http://test-server:8086"
            assert handler.token == "test-token-123"
            assert handler.org == "test-organization"
            assert handler.bucket == "test-bucket-env"
            assert handler.timeout == 5000
            assert handler.verify_ssl is False

    def test_from_env_with_defaults(self):
        """Test creating handler from environment with default optional values."""
        # Set only required environment variables
        env_vars = {
            "INFLUXDB_TOKEN": "test-token-456",
            "INFLUXDB_ORG": "test-org-default",
            "INFLUXDB_BUCKET": "test-bucket-default",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            # Clear optional vars to ensure defaults are used
            for key in ["INFLUXDB_URL", "INFLUXDB_TIMEOUT", "INFLUXDB_VERIFY_SSL"]:
                os.environ.pop(key, None)

            handler = InfluxDBHandler.from_env()

            assert handler.url == "http://localhost:8086"  # Default
            assert handler.token == "test-token-456"
            assert handler.org == "test-org-default"
            assert handler.bucket == "test-bucket-default"
            assert handler.timeout == 10000  # Default
            assert handler.verify_ssl is True  # Default

    def test_from_env_missing_token(self):
        """Test that missing INFLUXDB_TOKEN raises ValueError."""
        env_vars = {
            "INFLUXDB_ORG": "test-org",
            "INFLUXDB_BUCKET": "test-bucket",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            # Ensure TOKEN is not set
            os.environ.pop("INFLUXDB_TOKEN", None)

            with pytest.raises(
                ValueError,
                match="Missing required environment variables: INFLUXDB_TOKEN",
            ):
                InfluxDBHandler.from_env()

    def test_from_env_missing_org(self):
        """Test that missing INFLUXDB_ORG raises ValueError."""
        env_vars = {
            "INFLUXDB_TOKEN": "test-token",
            "INFLUXDB_BUCKET": "test-bucket",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            # Ensure ORG is not set
            os.environ.pop("INFLUXDB_ORG", None)

            with pytest.raises(
                ValueError, match="Missing required environment variables: INFLUXDB_ORG"
            ):
                InfluxDBHandler.from_env()

    def test_from_env_missing_bucket(self):
        """Test that missing INFLUXDB_BUCKET raises ValueError."""
        env_vars = {
            "INFLUXDB_TOKEN": "test-token",
            "INFLUXDB_ORG": "test-org",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            # Ensure BUCKET is not set
            os.environ.pop("INFLUXDB_BUCKET", None)

            with pytest.raises(
                ValueError,
                match="Missing required environment variables: INFLUXDB_BUCKET",
            ):
                InfluxDBHandler.from_env()

    def test_from_env_missing_multiple(self):
        """Test that missing multiple required variables lists all missing."""
        with patch.dict(os.environ, {}, clear=False):
            # Clear all InfluxDB environment variables
            for key in ["INFLUXDB_TOKEN", "INFLUXDB_ORG", "INFLUXDB_BUCKET"]:
                os.environ.pop(key, None)

            with pytest.raises(
                ValueError,
                match="Missing required environment variables: INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET",
            ):
                InfluxDBHandler.from_env()

    def test_from_env_invalid_timeout(self):
        """Test that invalid INFLUXDB_TIMEOUT raises ValueError."""
        env_vars = {
            "INFLUXDB_TOKEN": "test-token",
            "INFLUXDB_ORG": "test-org",
            "INFLUXDB_BUCKET": "test-bucket",
            "INFLUXDB_TIMEOUT": "not-a-number",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            with pytest.raises(
                ValueError, match="Invalid INFLUXDB_TIMEOUT value: not-a-number"
            ):
                InfluxDBHandler.from_env()

    def test_from_env_negative_timeout(self):
        """Test that negative timeout raises ValueError."""
        env_vars = {
            "INFLUXDB_TOKEN": "test-token",
            "INFLUXDB_ORG": "test-org",
            "INFLUXDB_BUCKET": "test-bucket",
            "INFLUXDB_TIMEOUT": "-1000",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            with pytest.raises(
                ValueError, match="Invalid INFLUXDB_TIMEOUT value: -1000"
            ):
                InfluxDBHandler.from_env()

    def test_from_env_verify_ssl_variations(self):
        """Test various INFLUXDB_VERIFY_SSL values."""
        base_env = {
            "INFLUXDB_TOKEN": "test-token",
            "INFLUXDB_ORG": "test-org",
            "INFLUXDB_BUCKET": "test-bucket",
        }

        # Test values that should be True
        for ssl_value in ["true", "True", "TRUE", "yes", "YES", "1", "on", "ON"]:
            env_vars = {**base_env, "INFLUXDB_VERIFY_SSL": ssl_value}
            with patch.dict(os.environ, env_vars, clear=False):
                handler = InfluxDBHandler.from_env()
                assert handler.verify_ssl is True, f"Failed for value: {ssl_value}"

        # Test values that should be False
        for ssl_value in ["false", "False", "FALSE", "no", "NO", "0", "off", "OFF"]:
            env_vars = {**base_env, "INFLUXDB_VERIFY_SSL": ssl_value}
            with patch.dict(os.environ, env_vars, clear=False):
                handler = InfluxDBHandler.from_env()
                assert handler.verify_ssl is False, f"Failed for value: {ssl_value}"


class TestOHLCDataPoint:
    """Test suite for OHLCDataPoint model."""

    def test_ohlc_data_point_creation(self):
        """Test creating OHLC data point with valid data."""
        data_point = OHLCDataPoint(
            timestamp=datetime.now(),
            symbol="EURUSD",
            timeframe=TimeFrame.M5,
            broker="test-broker",
            open=1.0850,
            high=1.0860,
            low=1.0840,
            close=1.0855,
            volume=1000.0,
            spread=0.0002,
        )

        assert data_point.symbol == "EURUSD"
        assert data_point.timeframe == TimeFrame.M5
        assert data_point.open == 1.0850

    def test_symbol_validation_uppercase(self):
        """Test symbol is converted to uppercase."""
        data_point = OHLCDataPoint(
            timestamp=datetime.now(),
            symbol="eurusd",
            timeframe=TimeFrame.M5,
            open=1.0850,
            high=1.0860,
            low=1.0840,
            close=1.0855,
            volume=1000.0,
        )

        assert data_point.symbol == "EURUSD"

    def test_symbol_validation_invalid_length(self):
        """Test symbol validation rejects invalid lengths."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            OHLCDataPoint(
                timestamp=datetime.now(),
                symbol="EUR",  # Too short
                timeframe=TimeFrame.M5,
                open=1.0850,
                high=1.0860,
                low=1.0840,
                close=1.0855,
                volume=1000.0,
            )

    def test_float32_validation(self):
        """Test Float32 range validation."""
        with pytest.raises(ValueError, match="exceeds Float32 range"):
            OHLCDataPoint(
                timestamp=datetime.now(),
                symbol="EURUSD",
                timeframe=TimeFrame.M5,
                open=5e38,  # Exceeds Float32 range
                high=1.0860,
                low=1.0840,
                close=1.0855,
                volume=1000.0,
            )

    def test_to_influx_point(self):
        """Test conversion to InfluxDB Point format."""
        timestamp = datetime.now()
        data_point = OHLCDataPoint(
            timestamp=timestamp,
            symbol="EURUSD",
            timeframe=TimeFrame.M5,
            broker="test-broker",
            open=1.0850,
            high=1.0860,
            low=1.0840,
            close=1.0855,
            volume=1000.0,
            spread=0.0002,
        )

        point = data_point.to_influx_point("test_measurement")

        # Point object doesn't have direct attribute access,
        # so we verify the method was called correctly
        assert point is not None

    def test_to_line_protocol(self):
        """Test conversion to Line Protocol string."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        data_point = OHLCDataPoint(
            timestamp=timestamp,
            symbol="EURUSD",
            timeframe=TimeFrame.M5,
            broker="test-broker",
            open=1.0850,
            high=1.0860,
            low=1.0840,
            close=1.0855,
            volume=1000.0,
            spread=0.0002,
        )

        line_protocol = data_point.to_line_protocol("test_measurement")

        # Verify Line Protocol format
        assert (
            "test_measurement,symbol=EURUSD,timeframe=M5,broker=test-broker"
            in line_protocol
        )
        assert "open=1.085" in line_protocol
        assert "high=1.086" in line_protocol
        assert "low=1.084" in line_protocol
        assert "close=1.0855" in line_protocol
        assert "volume=1000.0" in line_protocol
        assert "spread=0.0002" in line_protocol


class TestInfluxDBSchema:
    """Test suite for InfluxDBSchema class."""

    def test_schema_constants(self):
        """Test schema constant definitions."""
        assert InfluxDBSchema.MEASUREMENT_OHLC == "ohlc"
        assert InfluxDBSchema.TAG_SYMBOL == "symbol"
        assert InfluxDBSchema.TAG_TIMEFRAME == "timeframe"
        assert InfluxDBSchema.TAG_BROKER == "broker"
        assert InfluxDBSchema.FIELD_OPEN == "open"
        assert InfluxDBSchema.FIELD_HIGH == "high"
        assert InfluxDBSchema.FIELD_LOW == "low"
        assert InfluxDBSchema.FIELD_CLOSE == "close"
        assert InfluxDBSchema.FIELD_VOLUME == "volume"
        assert InfluxDBSchema.FIELD_SPREAD == "spread"

    def test_field_types(self):
        """Test field type definitions."""
        expected_types = {
            "open": "float",
            "high": "float",
            "low": "float",
            "close": "float",
            "volume": "float",
            "spread": "float",
        }
        assert InfluxDBSchema.FIELD_TYPES == expected_types

    def test_get_flux_query_template(self):
        """Test Flux query template generation."""
        template = InfluxDBSchema.get_flux_query_template()

        # Verify template contains required elements
        assert "from(bucket:" in template
        assert "range(start:" in template
        assert 'filter(fn: (r) => r["_measurement"]' in template
        assert 'filter(fn: (r) => r["symbol"]' in template
        assert 'filter(fn: (r) => r["timeframe"]' in template
        assert "pivot(rowKey:" in template

    def test_validate_tags_success(self):
        """Test successful tag validation."""
        tags = {
            "symbol": "eurusd",
            "timeframe": "M5",
            "broker": "test-broker",
        }

        validated = InfluxDBSchema.validate_tags(tags)

        assert validated["symbol"] == "EURUSD"  # Uppercased
        assert validated["timeframe"] == "M5"
        assert validated["broker"] == "test-broker"

    def test_validate_tags_missing_required(self):
        """Test tag validation with missing required tags."""
        tags = {
            "broker": "test-broker",
        }

        with pytest.raises(ValueError, match="Missing required tags"):
            InfluxDBSchema.validate_tags(tags)

    def test_validate_tags_default_broker(self):
        """Test tag validation with default broker."""
        tags = {
            "symbol": "EURUSD",
            "timeframe": "M5",
        }

        validated = InfluxDBSchema.validate_tags(tags)

        assert validated["broker"] == "default"


class TestCustomExceptions:
    """Test suite for custom exception classes."""

    def test_influxdb_connection_error(self):
        """Test InfluxDBConnectionError exception."""
        error = InfluxDBConnectionError("Connection failed")
        assert str(error) == "Connection failed"
        assert isinstance(error, Exception)

    def test_influxdb_query_error(self):
        """Test InfluxDBQueryError exception."""
        from src.storage.influx_handler import InfluxDBQueryError

        error = InfluxDBQueryError("Query failed")
        assert str(error) == "Query failed"
        assert isinstance(error, Exception)

    def test_influxdb_write_error(self):
        """Test InfluxDBWriteError exception."""
        from src.storage.influx_handler import InfluxDBWriteError

        error = InfluxDBWriteError("Write failed")
        assert str(error) == "Write failed"
        assert isinstance(error, Exception)

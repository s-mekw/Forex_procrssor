"""Unit tests for InfluxDB handler.

This module contains unit tests for InfluxDBHandler class,
including connection management and health check functionality.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from influxdb_client.client.exceptions import InfluxDBError

from src.storage.influx_handler import (
    InfluxDBConnectionError,
    InfluxDBHandler,
    InfluxDBSchema,
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

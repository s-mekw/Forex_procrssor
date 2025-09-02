"""InfluxDB handler for time series data storage.

This module provides an asynchronous interface for interacting with InfluxDB,
including connection management, health checks, and basic CRUD operations.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.exceptions import InfluxDBError
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class TimeFrame(str, Enum):
    """Supported timeframe enumerations for OHLC data."""

    M1 = "M1"  # 1 minute
    M5 = "M5"  # 5 minutes
    M15 = "M15"  # 15 minutes
    M30 = "M30"  # 30 minutes
    H1 = "H1"  # 1 hour
    H4 = "H4"  # 4 hours
    D1 = "D1"  # 1 day
    W1 = "W1"  # 1 week
    MN = "MN"  # 1 month


class OHLCDataPoint(BaseModel):
    """Data model for OHLC (Open-High-Low-Close) data point.

    This model represents a single OHLC data point with tags and fields
    formatted for InfluxDB Line Protocol.

    Attributes:
        timestamp: The timestamp of the data point.
        symbol: Currency pair (e.g., 'EURUSD', 'GBPJPY').
        timeframe: Time interval for the candle.
        broker: Broker identifier (optional).
        open: Opening price (Float32).
        high: Highest price (Float32).
        low: Lowest price (Float32).
        close: Closing price (Float32).
        volume: Trading volume (Float32).
        spread: Bid-ask spread (Float32, optional).
    """

    # Timestamp
    timestamp: datetime

    # Tags (indexed fields for filtering)
    symbol: str = Field(..., description="Currency pair (e.g., EURUSD)")
    timeframe: TimeFrame = Field(..., description="Time interval")
    broker: str = Field(default="default", description="Broker identifier")

    # Fields (actual data values - all Float32 for memory efficiency)
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Highest price")
    low: float = Field(..., description="Lowest price")
    close: float = Field(..., description="Closing price")
    volume: float = Field(..., description="Trading volume")
    spread: float | None = Field(default=None, description="Bid-ask spread")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate currency pair symbol format.

        Args:
            v: The symbol string to validate.

        Returns:
            The uppercase symbol string.

        Raises:
            ValueError: If symbol format is invalid.
        """
        v = v.upper()
        # Basic validation for forex pair format (6 characters)
        if len(v) < 6 or len(v) > 10:
            raise ValueError(f"Invalid symbol format: {v}")
        return v

    @field_validator("open", "high", "low", "close", "volume", "spread")
    @classmethod
    def validate_float32(cls, v: float | None) -> float | None:
        """Ensure values are valid for Float32 storage.

        Args:
            v: The value to validate.

        Returns:
            The validated float value or None.

        Raises:
            ValueError: If value is out of Float32 range.
        """
        if v is None:
            return None

        # Float32 range: approximately ±3.4e38
        max_float32 = 3.4e38
        if abs(v) > max_float32:
            raise ValueError(f"Value {v} exceeds Float32 range")

        return float(v)

    def to_influx_point(self, measurement: str = "ohlc") -> Point:
        """Convert data point to InfluxDB Point format.

        Args:
            measurement: The measurement name for InfluxDB.

        Returns:
            InfluxDB Point object ready for writing.
        """
        point = (
            Point(measurement)
            .tag("symbol", self.symbol)
            .tag("timeframe", self.timeframe.value)
            .tag("broker", self.broker)
            .field("open", float(self.open))
            .field("high", float(self.high))
            .field("low", float(self.low))
            .field("close", float(self.close))
            .field("volume", float(self.volume))
            .time(self.timestamp)
        )

        if self.spread is not None:
            point = point.field("spread", float(self.spread))

        return point

    def to_line_protocol(self, measurement: str = "ohlc") -> str:
        """Convert data point to InfluxDB Line Protocol string.

        Args:
            measurement: The measurement name for InfluxDB.

        Returns:
            Line Protocol formatted string.
        """
        # Tags
        tags = f"symbol={self.symbol},timeframe={self.timeframe.value},broker={self.broker}"

        # Fields
        fields = [
            f"open={self.open}",
            f"high={self.high}",
            f"low={self.low}",
            f"close={self.close}",
            f"volume={self.volume}",
        ]

        if self.spread is not None:
            fields.append(f"spread={self.spread}")

        fields_str = ",".join(fields)

        # Timestamp in nanoseconds
        timestamp_ns = int(self.timestamp.timestamp() * 1e9)

        return f"{measurement},{tags} {fields_str} {timestamp_ns}"


class InfluxDBSchema:
    """Schema definitions and constants for InfluxDB operations.

    This class defines the schema structure for OHLC data storage
    in InfluxDB, including measurement names, tag keys, and field keys.
    """

    # Measurement name
    MEASUREMENT_OHLC = "ohlc"

    # Tag keys (indexed for efficient queries)
    TAG_SYMBOL = "symbol"
    TAG_TIMEFRAME = "timeframe"
    TAG_BROKER = "broker"

    # Field keys (actual data values)
    FIELD_OPEN = "open"
    FIELD_HIGH = "high"
    FIELD_LOW = "low"
    FIELD_CLOSE = "close"
    FIELD_VOLUME = "volume"
    FIELD_SPREAD = "spread"

    # Data types for fields (all Float32 for memory optimization)
    FIELD_TYPES = {
        FIELD_OPEN: "float",
        FIELD_HIGH: "float",
        FIELD_LOW: "float",
        FIELD_CLOSE: "float",
        FIELD_VOLUME: "float",
        FIELD_SPREAD: "float",
    }

    @classmethod
    def get_flux_query_template(cls) -> str:
        """Get a Flux query template for OHLC data retrieval.

        Returns:
            Flux query template string.
        """
        return """
        from(bucket: "{bucket}")
            |> range(start: {start}, stop: {stop})
            |> filter(fn: (r) => r["_measurement"] == "{measurement}")
            |> filter(fn: (r) => r["symbol"] == "{symbol}")
            |> filter(fn: (r) => r["timeframe"] == "{timeframe}")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> keep(columns: ["_time", "symbol", "timeframe", "broker", "open", "high", "low", "close", "volume", "spread"])
        """

    @classmethod
    def validate_tags(cls, tags: dict[str, Any]) -> dict[str, str]:
        """Validate and normalize tag values.

        Args:
            tags: Dictionary of tag key-value pairs.

        Returns:
            Validated and normalized tags dictionary.

        Raises:
            ValueError: If required tags are missing or invalid.
        """
        required_tags = {cls.TAG_SYMBOL, cls.TAG_TIMEFRAME}
        provided_tags = set(tags.keys())

        if not required_tags.issubset(provided_tags):
            missing = required_tags - provided_tags
            raise ValueError(f"Missing required tags: {missing}")

        # Normalize values
        normalized = {}
        normalized[cls.TAG_SYMBOL] = str(tags[cls.TAG_SYMBOL]).upper()
        normalized[cls.TAG_TIMEFRAME] = str(tags[cls.TAG_TIMEFRAME])
        normalized[cls.TAG_BROKER] = str(tags.get(cls.TAG_BROKER, "default"))

        return normalized


class InfluxDBHandler:
    """Handler for InfluxDB operations.

    This class manages the connection to InfluxDB and provides methods for
    health checks, data writing, and querying operations.

    Attributes:
        url: The InfluxDB server URL.
        token: The authentication token.
        org: The organization name.
        bucket: The bucket name for data storage.
        timeout: Connection timeout in milliseconds.
        verify_ssl: Whether to verify SSL certificates.
    """

    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        bucket: str,
        timeout: int = 10000,
        verify_ssl: bool = True,
    ) -> None:
        """Initialize InfluxDB handler.

        Args:
            url: The InfluxDB server URL.
            token: The authentication token.
            org: The organization name.
            bucket: The bucket name for data storage.
            timeout: Connection timeout in milliseconds. Defaults to 10000.
            verify_ssl: Whether to verify SSL certificates. Defaults to True.
        """
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._client: InfluxDBClient | None = None
        self._is_connected = False

    async def connect(self) -> None:
        """Establish connection to InfluxDB.

        Creates a new InfluxDB client instance and verifies the connection.

        Raises:
            InfluxDBError: If connection fails.
        """
        try:
            # Create InfluxDB client
            self._client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org,
                timeout=self.timeout,
                verify_ssl=self.verify_ssl,
            )

            # Verify connection by checking if ready
            if not self._client.ready():
                raise InfluxDBError("InfluxDB server is not ready")

            self._is_connected = True
            logger.info(f"Successfully connected to InfluxDB at {self.url}")

        except InfluxDBError as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            self._is_connected = False
            raise
        except Exception as e:
            logger.error(f"Unexpected error during InfluxDB connection: {e}")
            self._is_connected = False
            raise InfluxDBError(f"Connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Close connection to InfluxDB.

        Safely closes the InfluxDB client connection.
        """
        if self._client:
            try:
                self._client.close()
                self._is_connected = False
                logger.info("Successfully disconnected from InfluxDB")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
            finally:
                self._client = None

    async def health_check(self) -> bool:
        """Check the health status of InfluxDB connection.

        Performs a health check by pinging the InfluxDB server.

        Returns:
            True if the connection is healthy, False otherwise.
        """
        if not self._client:
            logger.warning("Health check failed: No client connection")
            return False

        try:
            # Perform health check
            health = self._client.health()
            if health.status == "pass":
                logger.debug("Health check passed")
                return True
            else:
                logger.warning(f"Health check failed with status: {health.status}")
                return False

        except InfluxDBError as e:
            logger.error(f"Health check failed with error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during health check: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        """Check if handler is connected to InfluxDB.

        Returns:
            True if connected, False otherwise.
        """
        return self._is_connected

    def __del__(self) -> None:
        """Cleanup method to ensure connection is closed."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass  # Ignore errors during cleanup

    async def __aenter__(self) -> "InfluxDBHandler":
        """Async context manager entry.

        Returns:
            Self after establishing connection.
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception value if raised.
            exc_tb: Exception traceback if raised.
        """
        await self.disconnect()


class InfluxDBConnectionError(Exception):
    """Custom exception for InfluxDB connection errors."""

    pass


class InfluxDBQueryError(Exception):
    """Custom exception for InfluxDB query errors."""

    pass


class InfluxDBWriteError(Exception):
    """Custom exception for InfluxDB write errors."""

    pass

"""InfluxDB handler for time series data storage.

This module provides an asynchronous interface for interacting with InfluxDB,
including connection management, health checks, and basic CRUD operations.
"""

import logging
import os
from datetime import datetime
from enum import Enum
from typing import Any

import polars as pl
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.client.query_api import QueryApi
from influxdb_client.client.write_api import ASYNCHRONOUS, SYNCHRONOUS
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

    @classmethod
    def from_env(cls) -> "InfluxDBHandler":
        """Create InfluxDBHandler instance from environment variables.

        Environment variables:
            INFLUXDB_URL: The InfluxDB server URL (default: http://localhost:8086)
            INFLUXDB_TOKEN: The authentication token (required)
            INFLUXDB_ORG: The organization name (required)
            INFLUXDB_BUCKET: The bucket name (required)
            INFLUXDB_TIMEOUT: Connection timeout in ms (default: 10000)
            INFLUXDB_VERIFY_SSL: Whether to verify SSL (default: true)

        Returns:
            InfluxDBHandler instance configured from environment variables.

        Raises:
            ValueError: If required environment variables are missing.

        Example:
            >>> # Set environment variables first
            >>> import os
            >>> os.environ['INFLUXDB_TOKEN'] = 'your-token'
            >>> os.environ['INFLUXDB_ORG'] = 'your-org'
            >>> os.environ['INFLUXDB_BUCKET'] = 'your-bucket'
            >>> # Create handler from environment
            >>> handler = InfluxDBHandler.from_env()
        """
        # Get required environment variables
        token = os.getenv("INFLUXDB_TOKEN")
        org = os.getenv("INFLUXDB_ORG")
        bucket = os.getenv("INFLUXDB_BUCKET")

        # Validate required variables
        missing_vars = []
        if not token:
            missing_vars.append("INFLUXDB_TOKEN")
        if not org:
            missing_vars.append("INFLUXDB_ORG")
        if not bucket:
            missing_vars.append("INFLUXDB_BUCKET")

        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        # Get optional environment variables with defaults
        url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
        timeout_str = os.getenv("INFLUXDB_TIMEOUT", "10000")
        verify_ssl_str = os.getenv("INFLUXDB_VERIFY_SSL", "true")

        # Parse timeout
        try:
            timeout = int(timeout_str)
            if timeout <= 0:
                raise ValueError(f"Invalid timeout value: {timeout}")
        except ValueError as e:
            raise ValueError(f"Invalid INFLUXDB_TIMEOUT value: {timeout_str}") from e

        # Parse verify_ssl
        verify_ssl = verify_ssl_str.lower() in ("true", "yes", "1", "on")

        # Log configuration (without sensitive token)
        logger.info(
            f"Creating InfluxDBHandler from environment: "
            f"url={url}, org={org}, bucket={bucket}, "
            f"timeout={timeout}, verify_ssl={verify_ssl}"
        )

        return cls(
            url=url,
            token=token,
            org=org,
            bucket=bucket,
            timeout=timeout,
            verify_ssl=verify_ssl,
        )

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
                raise InfluxDBConnectionError("InfluxDB server is not ready")

            self._is_connected = True
            logger.info(f"Successfully connected to InfluxDB at {self.url}")

        except InfluxDBError as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            self._is_connected = False
            raise
        except Exception as e:
            logger.error(f"Unexpected error during InfluxDB connection: {e}")
            self._is_connected = False
            raise InfluxDBConnectionError(f"Connection failed: {e}") from e

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

    async def write_point(
        self, data_point: OHLCDataPoint, measurement: str = "ohlc"
    ) -> None:
        """Write a single data point to InfluxDB.

        Args:
            data_point: The OHLC data point to write.
            measurement: The measurement name for InfluxDB. Defaults to "ohlc".

        Raises:
            InfluxDBWriteError: If write operation fails.
            InfluxDBConnectionError: If not connected to InfluxDB.
        """
        if not self._client:
            raise InfluxDBConnectionError("Not connected to InfluxDB")

        try:
            # Convert data point to InfluxDB Point format
            point = data_point.to_influx_point(measurement)

            # Get write API with synchronous mode for single point
            write_api = self._client.write_api(write_options=SYNCHRONOUS)

            # Write the point
            write_api.write(bucket=self.bucket, record=point)

            logger.debug(
                f"Successfully wrote data point: {data_point.symbol} "
                f"at {data_point.timestamp}"
            )

        except InfluxDBError as e:
            logger.error(f"Failed to write data point: {e}")
            raise InfluxDBWriteError(f"Write failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during write: {e}")
            raise InfluxDBWriteError(f"Unexpected write error: {e}") from e
        finally:
            # Close write API to free resources
            if "write_api" in locals():
                write_api.close()

    async def write_batch(
        self,
        data_points: list[OHLCDataPoint],
        measurement: str = "ohlc",
        batch_size: int = 5000,
    ) -> None:
        """Write multiple data points to InfluxDB in batches.

        Args:
            data_points: List of OHLC data points to write.
            measurement: The measurement name for InfluxDB. Defaults to "ohlc".
            batch_size: Maximum number of points per batch. Defaults to 5000.

        Raises:
            InfluxDBWriteError: If write operation fails.
            InfluxDBConnectionError: If not connected to InfluxDB.
            ValueError: If batch_size is invalid or data_points is empty.
        """
        if not self._client:
            raise InfluxDBConnectionError("Not connected to InfluxDB")

        if not data_points:
            raise ValueError("No data points provided for batch write")

        if batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {batch_size}")

        try:
            # Convert all data points to InfluxDB Point format
            points = [dp.to_influx_point(measurement) for dp in data_points]

            # Get write API with asynchronous mode for batch write
            write_api = self._client.write_api(write_options=ASYNCHRONOUS)

            # Write points in batches
            total_points = len(points)
            written_count = 0

            for i in range(0, total_points, batch_size):
                batch = points[i : i + batch_size]
                write_api.write(bucket=self.bucket, records=batch)
                written_count += len(batch)

                logger.debug(
                    f"Wrote batch of {len(batch)} points "
                    f"({written_count}/{total_points} total)"
                )

            # Flush any remaining data
            write_api.flush()

            logger.info(
                f"Successfully wrote {total_points} data points in "
                f"{(total_points + batch_size - 1) // batch_size} batches"
            )

        except InfluxDBError as e:
            logger.error(f"Failed to write batch: {e}")
            raise InfluxDBWriteError(f"Batch write failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during batch write: {e}")
            raise InfluxDBWriteError(f"Unexpected batch write error: {e}") from e
        finally:
            # Close write API to free resources
            if "write_api" in locals():
                write_api.close()

    async def query(self, flux_query: str) -> list[dict[str, Any]]:
        """Execute a Flux query and return results.

        Args:
            flux_query: The Flux query string to execute.

        Returns:
            List of dictionaries containing query results.

        Raises:
            InfluxDBQueryError: If query execution fails.
            InfluxDBConnectionError: If not connected to InfluxDB.
        """
        if not self._client:
            raise InfluxDBConnectionError("Not connected to InfluxDB")

        try:
            # Get query API
            query_api: QueryApi = self._client.query_api()

            # Execute query
            result = query_api.query(query=flux_query, org=self.org)

            # Convert result to list of dictionaries
            data = []
            for table in result:
                for record in table.records:
                    data.append(record.values)

            logger.debug(f"Query returned {len(data)} records")
            return data

        except InfluxDBError as e:
            logger.error(f"Query failed: {e}")
            raise InfluxDBQueryError(f"Query failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during query: {e}")
            raise InfluxDBQueryError(f"Unexpected query error: {e}") from e

    async def query_ohlc(
        self,
        symbol: str,
        timeframe: TimeFrame | str,
        start_time: datetime,
        end_time: datetime | None = None,
        broker: str | None = None,
    ) -> pl.DataFrame:
        """Query OHLC data for a specific symbol and timeframe.

        Args:
            symbol: Currency pair (e.g., 'EURUSD').
            timeframe: Time interval for the candles.
            start_time: Start time for the query range.
            end_time: End time for the query range. If None, uses current time.
            broker: Optional broker filter.

        Returns:
            Polars DataFrame containing OHLC data.

        Raises:
            InfluxDBQueryError: If query execution fails.
            InfluxDBConnectionError: If not connected to InfluxDB.
        """
        if not self._client:
            raise InfluxDBConnectionError("Not connected to InfluxDB")

        # Normalize inputs
        symbol = symbol.upper()
        if isinstance(timeframe, str):
            timeframe = TimeFrame(timeframe)
        if end_time is None:
            end_time = datetime.now()

        # Build Flux query
        flux_query = f"""
        from(bucket: "{self.bucket}")
            |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
            |> filter(fn: (r) => r["_measurement"] == "ohlc")
            |> filter(fn: (r) => r["symbol"] == "{symbol}")
            |> filter(fn: (r) => r["timeframe"] == "{timeframe.value}")
        """

        # Add broker filter if specified
        if broker:
            flux_query += f'    |> filter(fn: (r) => r["broker"] == "{broker}")\n'

        flux_query += """    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> keep(columns: ["_time", "symbol", "timeframe", "broker", "open", "high", "low", "close", "volume", "spread"])
        """

        try:
            # Execute query
            result = await self.query(flux_query)

            # Convert to Polars DataFrame
            if not result:
                # Return empty DataFrame with correct schema
                return pl.DataFrame(
                    schema={
                        "time": pl.Datetime,
                        "symbol": pl.Utf8,
                        "timeframe": pl.Utf8,
                        "broker": pl.Utf8,
                        "open": pl.Float32,
                        "high": pl.Float32,
                        "low": pl.Float32,
                        "close": pl.Float32,
                        "volume": pl.Float32,
                        "spread": pl.Float32,
                    }
                )

            # Create DataFrame from results
            df = pl.DataFrame(result)

            # Rename _time column to time
            if "_time" in df.columns:
                df = df.rename({"_time": "time"})

            # Cast numeric columns to Float32
            numeric_columns = ["open", "high", "low", "close", "volume", "spread"]
            for col in numeric_columns:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(pl.Float32))

            # Sort by time
            df = df.sort("time")

            logger.info(
                f"Retrieved {len(df)} OHLC records for {symbol}/{timeframe.value}"
            )
            return df

        except InfluxDBQueryError:
            raise
        except Exception as e:
            logger.error(f"Failed to query OHLC data: {e}")
            raise InfluxDBQueryError(f"Failed to query OHLC data: {e}") from e

    async def query_time_range(
        self,
        measurement: str,
        start_time: datetime,
        end_time: datetime | None = None,
        filters: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Query data within a specific time range.

        Args:
            measurement: The measurement name to query.
            start_time: Start time for the query range.
            end_time: End time for the query range. If None, uses current time.
            filters: Optional dictionary of tag filters.

        Returns:
            Polars DataFrame containing query results.

        Raises:
            InfluxDBQueryError: If query execution fails.
            InfluxDBConnectionError: If not connected to InfluxDB.
        """
        if not self._client:
            raise InfluxDBConnectionError("Not connected to InfluxDB")

        if end_time is None:
            end_time = datetime.now()

        # Build Flux query
        flux_query = f"""
        from(bucket: "{self.bucket}")
            |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
            |> filter(fn: (r) => r["_measurement"] == "{measurement}")
        """

        # Add filters if specified
        if filters:
            for tag, value in filters.items():
                flux_query += f'    |> filter(fn: (r) => r["{tag}"] == "{value}")\n'

        flux_query += """    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        """

        try:
            # Execute query
            result = await self.query(flux_query)

            # Convert to Polars DataFrame
            if not result:
                # Return empty DataFrame
                return pl.DataFrame()

            # Create DataFrame from results
            df = pl.DataFrame(result)

            # Rename _time column to time if it exists
            if "_time" in df.columns:
                df = df.rename({"_time": "time"})

            # Sort by time if column exists
            if "time" in df.columns:
                df = df.sort("time")

            logger.info(
                f"Retrieved {len(df)} records from {measurement} "
                f"between {start_time} and {end_time}"
            )
            return df

        except InfluxDBQueryError:
            raise
        except Exception as e:
            logger.error(f"Failed to query time range: {e}")
            raise InfluxDBQueryError(f"Failed to query time range: {e}") from e


class InfluxDBConnectionError(Exception):
    """Custom exception for InfluxDB connection errors."""

    pass


class InfluxDBQueryError(Exception):
    """Custom exception for InfluxDB query errors."""

    pass


class InfluxDBWriteError(Exception):
    """Custom exception for InfluxDB write errors."""

    pass

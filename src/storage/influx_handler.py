"""InfluxDB handler for time series data storage.

This module provides an asynchronous interface for interacting with InfluxDB,
including connection management, health checks, and basic CRUD operations.
"""

import logging

from influxdb_client import InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError

logger = logging.getLogger(__name__)


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

"""
Polars-based data processing engine for Forex data.

This module provides high-performance data processing capabilities
using Polars library with Float32 optimization for memory efficiency.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PolarsProcessingEngine:
    """
    High-performance data processing engine using Polars.

    This class provides memory-optimized data processing capabilities
    with Float32 unification and lazy evaluation support.
    """

    def __init__(self, chunk_size: int = 100_000) -> None:
        """
        Initialize the Polars processing engine.

        Args:
            chunk_size: Number of rows to process per chunk (default: 100,000)
        """
        self.chunk_size = chunk_size
        self._float32_schema = {
            "open": pl.Float32,
            "high": pl.Float32,
            "low": pl.Float32,
            "close": pl.Float32,
            "volume": pl.Float32,
        }
        logger.info(f"PolarsProcessingEngine initialized with chunk_size={chunk_size}")

    def optimize_dtypes(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Optimize DataFrame data types by converting to Float32.

        This method converts numeric columns to Float32 for memory efficiency
        while preserving data precision up to 2 decimal places.

        Args:
            df: Input DataFrame to optimize

        Returns:
            DataFrame with optimized data types
        """
        # Convert Float64 columns to Float32
        float_cols = [col for col in df.columns if df[col].dtype == pl.Float64]

        if float_cols:
            df = df.with_columns([pl.col(col).cast(pl.Float32) for col in float_cols])
            logger.debug(f"Converted {len(float_cols)} columns from Float64 to Float32")

        return df

    def create_lazyframe(self, df: pl.DataFrame) -> pl.LazyFrame:
        """
        Create a LazyFrame for deferred execution.

        LazyFrames enable query optimization and lazy evaluation,
        improving performance for complex data transformations.

        Args:
            df: Input DataFrame to convert

        Returns:
            LazyFrame for deferred execution
        """
        # First optimize data types
        df = self.optimize_dtypes(df)

        # Convert to LazyFrame
        lazy_df = df.lazy()
        logger.debug("Created LazyFrame for deferred execution")

        return lazy_df

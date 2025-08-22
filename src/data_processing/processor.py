"""
Polars-based data processing engine for Forex data.

This module provides high-performance data processing capabilities
using Polars library with Float32 optimization for memory efficiency.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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

    def optimize_dtypes(
        self, df: pl.DataFrame, categorical_threshold: float = 0.5
    ) -> pl.DataFrame:
        """
        Optimize DataFrame data types for memory efficiency.

        This method performs comprehensive data type optimization:
        - Converts Float64 to Float32
        - Downcasts integers to smallest appropriate type
        - Converts string columns to categorical when appropriate

        Args:
            df: Input DataFrame to optimize
            categorical_threshold: Threshold for unique value ratio to convert to categorical (default: 0.5)

        Returns:
            DataFrame with optimized data types
        """
        original_memory = df.estimated_size()
        optimizations = []

        # 1. Convert Float64 columns to Float32
        float_cols = [col for col in df.columns if df[col].dtype == pl.Float64]
        if float_cols:
            df = df.with_columns([pl.col(col).cast(pl.Float32) for col in float_cols])
            optimizations.append(f"Float64→Float32: {len(float_cols)} columns")

        # 2. Optimize integer columns
        int_cols = [
            col
            for col in df.columns
            if df[col].dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8]
        ]

        for col in int_cols:
            if df[col].dtype == pl.Int64:
                # Get min and max values
                col_min = df[col].min()
                col_max = df[col].max()

                if col_min is not None and col_max is not None:
                    # Determine optimal integer type
                    if -128 <= col_min and col_max <= 127:
                        df = df.with_columns(pl.col(col).cast(pl.Int8))
                        optimizations.append(f"{col}: Int64→Int8")
                    elif -32768 <= col_min and col_max <= 32767:
                        df = df.with_columns(pl.col(col).cast(pl.Int16))
                        optimizations.append(f"{col}: Int64→Int16")
                    elif -2147483648 <= col_min and col_max <= 2147483647:
                        df = df.with_columns(pl.col(col).cast(pl.Int32))
                        optimizations.append(f"{col}: Int64→Int32")

        # 3. Convert string columns to categorical if appropriate
        str_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]

        for col in str_cols:
            unique_ratio = df[col].n_unique() / len(df)
            if unique_ratio <= categorical_threshold:
                df = df.with_columns(pl.col(col).cast(pl.Categorical))
                optimizations.append(
                    f"{col}: String→Categorical (unique ratio: {unique_ratio:.2%})"
                )

        # Report memory savings
        optimized_memory = df.estimated_size()
        memory_reduction = 1 - (optimized_memory / original_memory)

        if optimizations:
            logger.info(
                f"Memory optimization completed: {memory_reduction:.1%} reduction "
                f"({original_memory:,} → {optimized_memory:,} bytes)"
            )
            for optimization in optimizations[:5]:  # Log first 5 optimizations
                logger.debug(f"  - {optimization}")
            if len(optimizations) > 5:
                logger.debug(f"  ... and {len(optimizations) - 5} more optimizations")

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

    def get_memory_report(self, df: pl.DataFrame) -> dict[str, Any]:
        """
        Generate a detailed memory usage report for a DataFrame.

        This method analyzes memory usage by column and data type,
        providing insights for optimization opportunities.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary containing memory usage statistics
        """
        total_memory = df.estimated_size()
        n_rows = len(df)
        n_cols = len(df.columns)

        # Analyze memory by column
        column_memory = {}
        dtype_summary = {}

        for col in df.columns:
            col_df = df.select(col)
            col_memory = col_df.estimated_size()
            col_dtype = str(df[col].dtype)

            column_memory[col] = {
                "memory_bytes": col_memory,
                "memory_mb": col_memory / (1024 * 1024),
                "dtype": col_dtype,
                "percentage": (col_memory / total_memory) * 100
                if total_memory > 0
                else 0,
            }

            # Aggregate by data type
            if col_dtype not in dtype_summary:
                dtype_summary[col_dtype] = {
                    "count": 0,
                    "total_memory_bytes": 0,
                    "columns": [],
                }

            dtype_summary[col_dtype]["count"] += 1
            dtype_summary[col_dtype]["total_memory_bytes"] += col_memory
            dtype_summary[col_dtype]["columns"].append(col)

        # Calculate potential savings
        optimization_potential = self._calculate_optimization_potential(df)

        report = {
            "total_memory_bytes": total_memory,
            "total_memory_mb": total_memory / (1024 * 1024),
            "n_rows": n_rows,
            "n_columns": n_cols,
            "bytes_per_row": total_memory / n_rows if n_rows > 0 else 0,
            "column_memory": column_memory,
            "dtype_summary": dtype_summary,
            "optimization_potential": optimization_potential,
        }

        logger.info(
            f"Memory Report: {report['total_memory_mb']:.2f} MB total, "
            f"{report['bytes_per_row']:.0f} bytes/row, "
            f"potential savings: {optimization_potential['potential_savings_mb']:.2f} MB"
        )

        return report

    def _calculate_optimization_potential(self, df: pl.DataFrame) -> dict[str, Any]:
        """
        Calculate potential memory savings from optimization.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with optimization potential metrics
        """
        potential_savings = 0
        suggestions = []

        # Check for Float64 columns that could be Float32
        float64_cols = [col for col in df.columns if df[col].dtype == pl.Float64]
        if float64_cols:
            for col in float64_cols:
                col_memory = df.select(col).estimated_size()
                # Float32 uses half the memory of Float64
                potential_savings += col_memory * 0.5
                suggestions.append(f"{col}: Float64 → Float32")

        # Check for oversized integer columns
        for col in df.columns:
            if df[col].dtype == pl.Int64:
                col_min = df[col].min()
                col_max = df[col].max()
                if col_min is not None and col_max is not None:
                    col_memory = df.select(col).estimated_size()
                    if -128 <= col_min and col_max <= 127:
                        # Could use Int8 (1 byte) instead of Int64 (8 bytes)
                        potential_savings += col_memory * 0.875
                        suggestions.append(f"{col}: Int64 → Int8")
                    elif -32768 <= col_min and col_max <= 32767:
                        # Could use Int16 (2 bytes) instead of Int64 (8 bytes)
                        potential_savings += col_memory * 0.75
                        suggestions.append(f"{col}: Int64 → Int16")
                    elif -2147483648 <= col_min and col_max <= 2147483647:
                        # Could use Int32 (4 bytes) instead of Int64 (8 bytes)
                        potential_savings += col_memory * 0.5
                        suggestions.append(f"{col}: Int64 → Int32")

        # Check for string columns that could be categorical
        for col in df.columns:
            if df[col].dtype == pl.Utf8:
                unique_ratio = df[col].n_unique() / len(df)
                if unique_ratio <= 0.5:
                    col_memory = df.select(col).estimated_size()
                    # Categorical typically saves 50-90% for low cardinality
                    estimated_savings = col_memory * 0.7
                    potential_savings += estimated_savings
                    suggestions.append(
                        f"{col}: String → Categorical (unique ratio: {unique_ratio:.2%})"
                    )

        return {
            "potential_savings_bytes": int(potential_savings),
            "potential_savings_mb": potential_savings / (1024 * 1024),
            "potential_savings_percentage": (
                potential_savings / df.estimated_size() * 100
            )
            if df.estimated_size() > 0
            else 0,
            "suggestions": suggestions[:10],  # Limit to top 10 suggestions
        }

    def apply_filters(
        self, lazy_df: pl.LazyFrame, filters: list[tuple[str, str, Any]]
    ) -> pl.LazyFrame:
        """
        Apply filtering operations to a LazyFrame.

        This method applies multiple filter conditions to a LazyFrame
        using lazy evaluation for optimal performance.

        Args:
            lazy_df: Input LazyFrame to filter
            filters: List of filter conditions as tuples (column, operator, value)
                    Supported operators: '>', '<', '>=', '<=', '==', '!='

        Returns:
            LazyFrame with filters applied (not yet executed)

        Example:
            filters = [
                ('volume', '>', 5000),
                ('close', '>=', 1.0850)
            ]
        """
        result = lazy_df

        for column, operator, value in filters:
            if operator == ">":
                result = result.filter(pl.col(column) > value)
            elif operator == "<":
                result = result.filter(pl.col(column) < value)
            elif operator == ">=":
                result = result.filter(pl.col(column) >= value)
            elif operator == "<=":
                result = result.filter(pl.col(column) <= value)
            elif operator == "==":
                result = result.filter(pl.col(column) == value)
            elif operator == "!=":
                result = result.filter(pl.col(column) != value)
            else:
                raise ValueError(f"Unsupported operator: {operator}")

            logger.debug(f"Applied filter: {column} {operator} {value}")

        return result

    def apply_aggregations(
        self,
        lazy_df: pl.LazyFrame,
        group_by: list[str] | None = None,
        aggregations: dict[str, list[str]] | None = None,
    ) -> pl.LazyFrame:
        """
        Apply aggregation operations to a LazyFrame.

        This method performs various aggregation operations with optional grouping,
        maintaining lazy evaluation for performance optimization.

        Args:
            lazy_df: Input LazyFrame to aggregate
            group_by: Optional list of columns to group by
            aggregations: Dictionary mapping column names to aggregation functions
                         Supported: 'mean', 'sum', 'min', 'max', 'std', 'count', 'first', 'last'

        Returns:
            LazyFrame with aggregations applied (not yet executed)

        Example:
            aggregations = {
                'open': ['mean', 'std'],
                'volume': ['sum', 'mean']
            }
        """
        if aggregations is None:
            aggregations = {}

        # Build aggregation expressions
        agg_exprs = []
        for column, funcs in aggregations.items():
            for func in funcs:
                if func == "mean":
                    agg_exprs.append(pl.col(column).mean().alias(f"{column}_{func}"))
                elif func == "sum":
                    agg_exprs.append(pl.col(column).sum().alias(f"{column}_{func}"))
                elif func == "min":
                    agg_exprs.append(pl.col(column).min().alias(f"{column}_{func}"))
                elif func == "max":
                    agg_exprs.append(pl.col(column).max().alias(f"{column}_{func}"))
                elif func == "std":
                    agg_exprs.append(pl.col(column).std().alias(f"{column}_{func}"))
                elif func == "count":
                    agg_exprs.append(pl.col(column).count().alias(f"{column}_{func}"))
                elif func == "first":
                    agg_exprs.append(pl.col(column).first().alias(f"{column}_{func}"))
                elif func == "last":
                    agg_exprs.append(pl.col(column).last().alias(f"{column}_{func}"))
                else:
                    raise ValueError(f"Unsupported aggregation function: {func}")

        if not agg_exprs:
            logger.warning("No aggregation expressions provided")
            return lazy_df

        # Apply aggregations with or without grouping
        if group_by:
            result = lazy_df.group_by(group_by).agg(agg_exprs)
            logger.debug(f"Applied grouped aggregations by: {group_by}")
        else:
            # For non-grouped aggregations, use select
            result = lazy_df.select(agg_exprs)
            logger.debug("Applied non-grouped aggregations")

        return result

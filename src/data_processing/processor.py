"""
Polars-based data processing engine for Forex data.

This module provides high-performance data processing capabilities
using Polars library with Float32 optimization for memory efficiency.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import psutil

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Custom exception classes
class ProcessingError(Exception):
    """Base class for data processing errors."""
    pass


class DataTypeError(ProcessingError):
    """Raised when data type validation fails."""
    pass


class MemoryLimitError(ProcessingError):
    """Raised when memory limits are exceeded."""
    pass


class FileValidationError(ProcessingError):
    """Raised when file validation fails."""
    pass


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
        
        Raises:
            ValueError: If chunk_size is not positive
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self._float32_schema = {
            "open": pl.Float32,
            "high": pl.Float32,
            "low": pl.Float32,
            "close": pl.Float32,
            "volume": pl.Float32,
        }
        logger.info(f"PolarsProcessingEngine initialized with chunk_size={chunk_size}")

    def calculate_rci(
        self,
        df: pl.DataFrame,
        periods: Optional[List[int]] = None,
        column_name: str = 'close',
        add_reliability: bool = True
    ) -> pl.DataFrame:
        """
        Calculate RCI (Rank Correlation Index) indicators.
        
        This method integrates RCI calculation into the processing pipeline,
        providing efficient computation of RCI values for multiple periods.
        
        Args:
            df: Input DataFrame with price data
            periods: List of RCI periods to calculate (default: [9, 13, 24, 33, 48, 66, 108])
            column_name: Column name for price data (default: 'close')
            add_reliability: Whether to add reliability flags (default: True)
            
        Returns:
            DataFrame with RCI columns added
            
        Raises:
            ValueError: If DataFrame is empty or column_name doesn't exist
            RCICalculationError: If RCI calculation fails
            
        Example:
            df = engine.calculate_rci(df, periods=[9, 13, 24])
            # Adds columns: rci_9, rci_13, rci_24, rci_9_reliable, etc.
        """
        from .rci import RCIProcessor, RCICalculationError
        
        # Validate input
        if df.is_empty():
            raise ValueError("Cannot calculate RCI on empty DataFrame")
        
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")
        
        logger.info(f"Calculating RCI for periods: {periods}")
        
        try:
            # Initialize RCI processor
            rci_processor = RCIProcessor(use_float32=True)
            
            # Calculate RCI
            result = rci_processor.apply_to_dataframe(
                df,
                periods=periods,
                column_name=column_name,
                add_reliability=add_reliability
            )
            
            # Log statistics
            if periods:
                valid_counts = {
                    f"rci_{p}": result[f"rci_{p}"].is_not_null().sum()
                    for p in periods or [9, 13, 24, 33, 48, 66, 108]
                    if f"rci_{p}" in result.columns
                }
                logger.info(f"RCI calculation complete. Valid values: {valid_counts}")
            
            return result
            
        except Exception as e:
            logger.error(f"RCI calculation failed: {e}")
            raise RCICalculationError(f"Failed to calculate RCI: {e}")
    
    def process_with_rci(
        self,
        df: pl.DataFrame,
        rci_periods: Optional[List[int]] = None,
        other_indicators: Optional[List[str]] = None,
        price_column: str = 'close'
    ) -> pl.DataFrame:
        """
        Process DataFrame with RCI and optionally other technical indicators.
        
        This method provides a unified interface for calculating multiple
        indicators including RCI in a single pipeline.
        
        Args:
            df: Input DataFrame
            rci_periods: RCI periods to calculate
            other_indicators: List of other indicators to calculate
                            (e.g., ['rsi', 'macd', 'bollinger'])
            price_column: Column name for price data
            
        Returns:
            DataFrame with all requested indicators
            
        Example:
            df = engine.process_with_rci(
                df,
                rci_periods=[9, 13, 24],
                other_indicators=['rsi', 'macd']
            )
        """
        result = df
        
        # Calculate RCI if periods specified
        if rci_periods:
            logger.info(f"Adding RCI indicators for periods: {rci_periods}")
            result = self.calculate_rci(
                result,
                periods=rci_periods,
                column_name=price_column
            )
        
        # Calculate other indicators if specified
        if other_indicators:
            from .indicators import TechnicalIndicatorEngine
            
            logger.info(f"Adding technical indicators: {other_indicators}")
            indicator_engine = TechnicalIndicatorEngine()
            
            # Map indicator names to methods
            indicator_map = {
                'rsi': lambda df: indicator_engine.calculate_rsi(df, price_column=price_column),
                'macd': lambda df: indicator_engine.calculate_macd(df, price_column=price_column),
                'bollinger': lambda df: indicator_engine.calculate_bollinger_bands(df, price_column=price_column),
                'ema': lambda df: indicator_engine.calculate_ema(df, price_column=price_column)
            }
            
            for indicator in other_indicators:
                if indicator.lower() in indicator_map:
                    result = indicator_map[indicator.lower()](result)
                else:
                    logger.warning(f"Unknown indicator: {indicator}")
        
        return result
    
    def calculate_rci_lazy(
        self,
        lf: pl.LazyFrame,
        periods: Optional[List[int]] = None,
        column_name: str = 'close',
        add_reliability: bool = True
    ) -> pl.LazyFrame:
        """
        Calculate RCI on LazyFrame for memory-efficient processing.
        
        Args:
            lf: Input LazyFrame
            periods: List of RCI periods
            column_name: Column name for price data
            add_reliability: Whether to add reliability flags
            
        Returns:
            LazyFrame with RCI calculations (not yet computed)
        """
        from .rci import RCIProcessor
        
        logger.info("Setting up lazy RCI calculation")
        
        # Initialize RCI processor
        rci_processor = RCIProcessor(use_float32=True)
        
        # Apply to LazyFrame
        result = rci_processor.apply_to_lazyframe(
            lf,
            periods=periods,
            column_name=column_name,
            add_reliability=add_reliability
        )
        
        return result

    def validate_datatypes(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Validate and fix data types in a DataFrame.
        
        This method attempts to convert columns to appropriate numeric types
        and handles mixed data types gracefully.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            DataFrame with validated and corrected data types
            
        Raises:
            DataTypeError: If data types cannot be corrected
        """
        try:
            for col in df.columns:
                dtype = df[col].dtype
                
                # Check for string columns that should be numeric
                if dtype == pl.Utf8:
                    # Try to convert to numeric if possible
                    try:
                        # First try to convert to Float32
                        df = df.with_columns(
                            pl.col(col).str.replace(",", "").cast(pl.Float32, strict=False)
                        )
                        logger.info(f"Converted string column {col} to Float32")
                    except Exception as e:
                        # If conversion fails, keep as string but log warning
                        logger.warning(f"Column {col} contains non-numeric strings, keeping as text: {e}")
                
                # Convert Float64 to Float32 for consistency
                elif dtype == pl.Float64:
                    df = df.with_columns(pl.col(col).cast(pl.Float32))
                    logger.debug(f"Converted {col} from Float64 to Float32")
                
                # Ensure integer types are handled appropriately
                elif dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
                    # Convert to Float32 for consistency with float-based processing
                    df = df.with_columns(pl.col(col).cast(pl.Float32))
                    logger.debug(f"Converted {col} from {dtype} to Float32")
                
                # Handle datetime columns - keep as is
                elif dtype == pl.Datetime:
                    logger.debug(f"Column {col} is datetime, keeping as is")
                
                # Handle null values
                elif dtype == pl.Null:
                    logger.warning(f"Column {col} contains only null values")
                
            return df
            
        except Exception as e:
            logger.error(f"Data type validation failed: {e}")
            raise DataTypeError(f"Failed to validate data types: {e}")
    
    def handle_empty_dataframe(self, df: pl.DataFrame | pl.LazyFrame) -> bool:
        """
        Check if a DataFrame is empty and handle appropriately.
        
        Args:
            df: DataFrame or LazyFrame to check
            
        Returns:
            True if the DataFrame is empty, False otherwise
        """
        try:
            if isinstance(df, pl.DataFrame):
                is_empty = len(df) == 0
                if is_empty:
                    logger.warning("Empty DataFrame detected")
                return is_empty
            elif isinstance(df, pl.LazyFrame):
                # For LazyFrame, check schema
                try:
                    schema = df.collect_schema()
                    if not schema:
                        logger.warning("Empty LazyFrame schema detected")
                        return True
                    # Try to get count without collecting all data
                    count = df.select(pl.count()).collect()[0, 0]
                    if count == 0:
                        logger.warning("LazyFrame has no rows")
                        return True
                    return False
                except Exception:
                    # If we can't determine, assume not empty
                    return False
            return False
        except Exception as e:
            logger.error(f"Error checking for empty DataFrame: {e}")
            return False
    
    def handle_memory_limit(self, required_memory_mb: float) -> bool:
        """
        Check if required memory exceeds available memory.
        
        Args:
            required_memory_mb: Required memory in megabytes
            
        Returns:
            True if memory is available, False if limit exceeded
        """
        try:
            memory_info = psutil.virtual_memory()
            available_mb = memory_info.available / (1024 * 1024)
            
            if required_memory_mb > available_mb:
                logger.warning(
                    f"Memory limit exceeded: required {required_memory_mb:.1f}MB, "
                    f"available {available_mb:.1f}MB"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking memory limit: {e}")
            return True  # Assume memory is available if we can't check
    
    def validate_file(self, file_path: str | Path) -> bool:
        """
        Validate that a file exists and is readable.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            if not file_path.is_file():
                logger.error(f"Path is not a file: {file_path}")
                return False
            
            # Check file extension
            valid_extensions = ['.csv', '.parquet', '.json', '.txt']
            if file_path.suffix.lower() not in valid_extensions:
                logger.warning(f"Unusual file extension: {file_path.suffix}")
            
            # Check if file is readable
            if not file_path.stat().st_size > 0:
                logger.warning(f"File is empty: {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False
    
    def handle_memory_pressure(self) -> int:
        """
        Handle memory pressure by adjusting processing parameters.
        
        Returns:
            Adjusted chunk size based on memory pressure
        """
        try:
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            
            if memory_percent >= 95:
                # Extreme memory pressure
                new_chunk_size = max(1_000, self.chunk_size // 4)
                logger.warning(
                    f"Extreme memory pressure ({memory_percent:.1f}%), "
                    f"reducing chunk size to {new_chunk_size:,}"
                )
            elif memory_percent >= 90:
                # High memory pressure
                new_chunk_size = max(1_000, self.chunk_size // 3)
                logger.warning(
                    f"High memory pressure ({memory_percent:.1f}%), "
                    f"reducing chunk size to {new_chunk_size:,}"
                )
            elif memory_percent >= 80:
                # Moderate memory pressure
                new_chunk_size = max(1_000, self.chunk_size // 2)
                logger.info(
                    f"Memory usage high ({memory_percent:.1f}%), "
                    f"adjusting chunk size to {new_chunk_size:,}"
                )
            else:
                # No memory pressure
                new_chunk_size = self.chunk_size
            
            self.chunk_size = new_chunk_size
            return new_chunk_size
            
        except Exception as e:
            logger.error(f"Error handling memory pressure: {e}")
            return self.chunk_size
    
    def validate_parameters(
        self,
        chunk_size: int | None = None,
        aggregations: dict[str, list[str]] | None = None,
        filters: list[tuple[str, str, Any]] | None = None,
        process_func: Any = None,
    ) -> None:
        """
        Validate input parameters for various methods.
        
        Args:
            chunk_size: Optional chunk size to validate
            aggregations: Optional aggregations to validate
            filters: Optional filters to validate
            process_func: Optional processing function to validate
            
        Raises:
            ValueError: If parameters are invalid
            TypeError: If parameters have wrong type
        """
        # Validate chunk_size
        if chunk_size is not None:
            if not isinstance(chunk_size, int):
                raise TypeError(f"chunk_size must be an integer, got {type(chunk_size)}")
            if chunk_size <= 0:
                raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        
        # Validate aggregations
        if aggregations is not None:
            valid_aggs = {'mean', 'sum', 'min', 'max', 'std', 'count', 'first', 'last'}
            for col, funcs in aggregations.items():
                for func in funcs:
                    if func not in valid_aggs:
                        raise ValueError(
                            f"Unsupported aggregation function: {func}. "
                            f"Valid functions: {valid_aggs}"
                        )
        
        # Validate filters
        if filters is not None:
            valid_ops = {'>', '<', '>=', '<=', '==', '!='}
            for col, op, val in filters:
                if op not in valid_ops:
                    raise ValueError(
                        f"Unsupported operator: {op}. "
                        f"Valid operators: {valid_ops}"
                    )
        
        # Validate process_func
        if process_func is not None:
            if not callable(process_func):
                raise TypeError(
                    f"process_func must be callable, got {type(process_func)}"
                )

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
        # Validate input parameters
        if not isinstance(categorical_threshold, (int, float)):
            raise TypeError("categorical_threshold must be a number")
        if not 0 <= categorical_threshold <= 1:
            raise ValueError("categorical_threshold must be between 0 and 1")
        
        # Check for empty DataFrame
        if self.handle_empty_dataframe(df):
            # For empty DataFrame, still apply type conversions to schema
            if len(df) == 0 and len(df.columns) > 0:
                # Convert float columns to Float32 even for empty DataFrame
                for col in df.columns:
                    if df[col].dtype in [pl.Float64, pl.Float32]:
                        df = df.with_columns(pl.col(col).cast(pl.Float32))
            return df
        
        # Validate data types first
        try:
            df = self.validate_datatypes(df)
        except DataTypeError:
            logger.warning("Data type validation failed, continuing with original types")
        
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
        # Check for empty DataFrame
        if self.handle_empty_dataframe(df):
            # Return empty LazyFrame with schema preserved
            return df.lazy()
        
        # First optimize data types
        try:
            df = self.optimize_dtypes(df)
        except Exception as e:
            logger.warning(f"Data type optimization failed: {e}, using original types")

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
        # Validate parameters
        self.validate_parameters(filters=filters)
        
        # Check for empty LazyFrame
        if self.handle_empty_dataframe(lazy_df):
            return lazy_df
        
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
        
        # Validate parameters
        self.validate_parameters(aggregations=aggregations)
        
        # Check for empty LazyFrame
        if self.handle_empty_dataframe(lazy_df):
            # For empty data, return appropriate empty result
            if group_by:
                return lazy_df
            else:
                # For global aggregation on empty data, return single row with nulls
                # Create a single row with null values for all aggregation results
                null_exprs = []
                for column, funcs in aggregations.items():
                    for func in funcs:
                        null_exprs.append(pl.lit(None).alias(f"{column}_{func}"))
                if null_exprs:
                    return pl.DataFrame().lazy().select(null_exprs)
                else:
                    return lazy_df.limit(0)

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

    def process_in_chunks(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        process_func: Callable[[pl.DataFrame], pl.DataFrame],
        chunk_size: int | None = None,
    ) -> pl.DataFrame:
        """
        Process large DataFrames in chunks to control memory usage.

        This method divides data into chunks and processes each chunk separately,
        which is essential for handling datasets larger than available memory.

        Args:
            data: Input DataFrame or LazyFrame to process
            process_func: Function to apply to each chunk
            chunk_size: Optional custom chunk size (uses self.chunk_size if None)

        Returns:
            Processed DataFrame with all chunks combined

        Example:
            def process_chunk(df):
                return df.filter(pl.col("volume") > 5000)

            result = engine.process_in_chunks(large_df, process_chunk)
        """
        # Validate parameters first (before using default)
        if chunk_size is not None:
            self.validate_parameters(chunk_size=chunk_size, process_func=process_func)
        else:
            self.validate_parameters(process_func=process_func)
        
        chunk_size = chunk_size or self.chunk_size
        
        # Check for empty data
        if self.handle_empty_dataframe(data):
            if isinstance(data, pl.LazyFrame):
                return pl.DataFrame()
            return data  # Return empty DataFrame as is

        # Convert LazyFrame to DataFrame if needed
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        n_rows = len(data)
        n_chunks = (n_rows + chunk_size - 1) // chunk_size

        logger.info(
            f"Processing {n_rows:,} rows in {n_chunks} chunks of {chunk_size:,} rows"
        )

        processed_chunks = []
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        for i in range(0, n_rows, chunk_size):
            chunk_end = min(i + chunk_size, n_rows)
            chunk_num = i // chunk_size + 1

            # Extract chunk
            chunk = data.slice(i, chunk_end - i)

            # Process chunk
            try:
                processed_chunk = process_func(chunk)
                processed_chunks.append(processed_chunk)

                # Monitor memory
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory

                logger.debug(
                    f"Chunk {chunk_num}/{n_chunks} processed: "
                    f"{len(processed_chunk):,} rows, "
                    f"memory +{memory_increase:.1f} MB"
                )

                # Adjust chunk size if memory usage is too high
                if memory_increase > 500:  # More than 500MB increase
                    self.adjust_chunk_size(decrease=True)

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_num}: {e}")
                raise

        # Combine all processed chunks
        if processed_chunks:
            result = pl.concat(processed_chunks)
            logger.info(f"Chunk processing complete: {len(result):,} rows in result")
            return result
        else:
            logger.warning("No chunks were successfully processed")
            return pl.DataFrame()

    def adjust_chunk_size(self, decrease: bool = False) -> int:
        """
        Dynamically adjust chunk size based on memory usage.

        This method monitors system memory and adjusts the chunk size
        to optimize processing performance while avoiding memory issues.

        Args:
            decrease: If True, decrease chunk size; if False, increase it

        Returns:
            New chunk size after adjustment
        """
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent

        old_size = self.chunk_size

        if decrease or memory_percent > 80:
            # Decrease chunk size if memory usage is high
            new_size = max(1_000, self.chunk_size // 2)
            if new_size != self.chunk_size:
                self.chunk_size = new_size
                logger.info(
                    f"Decreased chunk size: {old_size:,} → {self.chunk_size:,} "
                    f"(memory usage: {memory_percent:.1f}%)"
                )
        elif memory_percent < 50:
            # Increase chunk size if memory usage is low
            new_size = min(1_000_000, self.chunk_size * 2)
            if new_size != self.chunk_size:
                self.chunk_size = new_size
                logger.info(
                    f"Increased chunk size: {old_size:,} → {self.chunk_size:,} "
                    f"(memory usage: {memory_percent:.1f}%)"
                )
        else:
            logger.debug(
                f"Chunk size unchanged at {self.chunk_size:,} "
                f"(memory usage: {memory_percent:.1f}%)"
            )

        return self.chunk_size

    def stream_csv(
        self,
        file_path: str | Path,
        schema_overrides: dict[str, pl.DataType] | None = None,
        n_rows: int | None = None,
    ) -> pl.LazyFrame | None:
        """
        Stream CSV file using LazyFrame for memory-efficient processing.

        This method uses scan_csv to read CSV files lazily, allowing
        processing of files larger than available memory.

        Args:
            file_path: Path to CSV file
            schema_overrides: Optional schema overrides (defaults to Float32 for numeric columns)
            n_rows: Optional limit on number of rows to read

        Returns:
            LazyFrame for streaming processing

        Example:
            lazy_df = engine.stream_csv("large_data.csv")
            result = lazy_df.filter(pl.col("volume") > 5000).collect()
        """
        file_path = Path(file_path)
        
        # Validate file
        if not self.validate_file(file_path):
            logger.error(f"Invalid CSV file: {file_path}")
            return None

        # Default to Float32 for numeric columns if not specified
        if schema_overrides is None:
            schema_overrides = self._float32_schema.copy()

        logger.info(f"Streaming CSV file: {file_path}")

        try:
            lazy_df = pl.scan_csv(
                file_path,
                schema_overrides=schema_overrides,
                n_rows=n_rows,
            )

            # Get estimated row count (without loading data)
            with pl.StringCache():
                estimated_rows = lazy_df.select(pl.count()).collect()[0, 0]

            logger.info(f"CSV streaming initialized: ~{estimated_rows:,} rows")
            return lazy_df

        except pl.exceptions.ComputeError as e:
            logger.error(f"CSV parsing error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error streaming CSV file: {e}")
            return None

    def stream_parquet(
        self,
        file_path: str | Path,
        columns: list[str] | None = None,
        n_rows: int | None = None,
    ) -> pl.LazyFrame | None:
        """
        Stream Parquet file using LazyFrame for high-performance processing.

        Parquet files support column pruning and predicate pushdown,
        making them ideal for efficient data processing.

        Args:
            file_path: Path to Parquet file
            columns: Optional list of columns to read (reads all if None)
            n_rows: Optional limit on number of rows to read

        Returns:
            LazyFrame for streaming processing

        Example:
            lazy_df = engine.stream_parquet("data.parquet", columns=["timestamp", "close"])
            result = lazy_df.filter(pl.col("close") > 1.08).collect()
        """
        file_path = Path(file_path)
        
        # Validate file
        if not self.validate_file(file_path):
            logger.error(f"Invalid Parquet file: {file_path}")
            return None

        logger.info(f"Streaming Parquet file: {file_path}")

        try:
            # scan_parquetの引数を調整（columnsパラメータは存在しない）
            if n_rows is not None:
                lazy_df = pl.scan_parquet(file_path, n_rows=n_rows)
            else:
                lazy_df = pl.scan_parquet(file_path)

            # カラムの選択は後で行う
            if columns:
                lazy_df = lazy_df.select(columns)

            # Get metadata without loading data
            with pl.StringCache():
                if columns:
                    estimated_rows = lazy_df.select(pl.count()).collect()[0, 0]
                    logger.info(
                        f"Parquet streaming initialized: ~{estimated_rows:,} rows, "
                        f"{len(columns)} columns selected"
                    )
                else:
                    schema = lazy_df.collect_schema()
                    estimated_rows = lazy_df.select(pl.count()).collect()[0, 0]
                    logger.info(
                        f"Parquet streaming initialized: ~{estimated_rows:,} rows, "
                        f"{len(schema)} columns"
                    )

            return lazy_df

        except Exception as e:
            logger.error(f"Error streaming Parquet file: {e}")
            return None

    def process_batches(
        self,
        batches: list[pl.DataFrame | pl.LazyFrame],
        process_func: Callable[[pl.DataFrame], pl.DataFrame],
        preserve_order: bool = True,
    ) -> pl.DataFrame:
        """
        Process multiple batches of data with optional order preservation.

        This method processes multiple data batches and combines the results,
        useful for parallel processing scenarios.

        Args:
            batches: List of DataFrames or LazyFrames to process
            process_func: Function to apply to each batch
            preserve_order: If True, maintain original batch order in result

        Returns:
            Combined DataFrame with all processed batches

        Example:
            batches = [df1, df2, df3]
            result = engine.process_batches(batches, lambda df: df.filter(pl.col("volume") > 5000))
        """
        if not batches:
            logger.warning("No batches provided for processing")
            return pl.DataFrame()

        logger.info(f"Processing {len(batches)} batches")

        processed_batches = []
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        for i, batch in enumerate(batches, 1):
            try:
                # Convert LazyFrame to DataFrame if needed
                if isinstance(batch, pl.LazyFrame):
                    batch = batch.collect()

                # Process batch
                processed_batch = process_func(batch)

                if preserve_order:
                    # Add batch index for order preservation
                    processed_batch = processed_batch.with_columns(
                        pl.lit(i).alias("_batch_idx")
                    )

                processed_batches.append(processed_batch)

                # Monitor memory
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory

                logger.debug(
                    f"Batch {i}/{len(batches)} processed: "
                    f"{len(processed_batch):,} rows, "
                    f"memory +{memory_increase:.1f} MB"
                )

            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                raise

        # Combine all processed batches
        if processed_batches:
            result = pl.concat(processed_batches)

            if preserve_order and "_batch_idx" in result.columns:
                # Sort by batch index and remove the column
                result = result.sort("_batch_idx").drop("_batch_idx")

            logger.info(f"Batch processing complete: {len(result):,} rows in result")
            return result
        else:
            logger.warning("No batches were successfully processed")
            return pl.DataFrame()

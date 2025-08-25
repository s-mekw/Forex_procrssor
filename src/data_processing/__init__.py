"""
データ処理パイプラインモジュール

Polarsベースの高速データ処理、テクニカル指標計算、
RCI計算、リアルタイム処理パイプラインを提供します。
"""

from .processor import PolarsProcessingEngine
from .rci import (
    RCICalculatorEngine,
    DifferentialRCICalculator,
    RCICalculationError,
    InvalidPeriodError,
    InsufficientDataError
)

__all__ = [
    "PolarsProcessingEngine",
    "RCICalculatorEngine",
    "DifferentialRCICalculator",
    "RCICalculationError",
    "InvalidPeriodError",
    "InsufficientDataError"
]

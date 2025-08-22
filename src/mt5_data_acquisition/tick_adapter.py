"""
Tickモデル間の変換アダプター

common.models.TickとTickToBarConverterの間の
データ形式の違いを吸収するアダプター。
"""

from decimal import Decimal

import numpy as np

from src.common.models import Tick as CommonTick


class TickAdapter:
    """common.models.TickをTickToBarConverterで使用できるように変換

    このアダプターは、Float32制約のあるCommonTickと
    Decimal精度を必要とするTickToBarConverterの間で
    データ変換を行います。
    """

    @staticmethod
    def to_decimal_dict(tick: CommonTick) -> dict:
        """CommonTickをDecimal形式の辞書に変換

        Args:
            tick: 変換元のCommonTickインスタンス

        Returns:
            Decimal型の価格データを含む辞書
        """
        return {
            "symbol": tick.symbol,
            "time": tick.timestamp,  # timestamp -> time
            "bid": Decimal(str(tick.bid)),
            "ask": Decimal(str(tick.ask)),
            "volume": Decimal(str(tick.volume)),
        }

    @staticmethod
    def from_decimal_dict(tick_dict: dict) -> CommonTick:
        """Decimal形式の辞書からCommonTickに変換

        Args:
            tick_dict: Decimal型の価格データを含む辞書

        Returns:
            変換されたCommonTickインスタンス
        """
        # timeとtimestamp両方に対応
        timestamp = tick_dict.get("time", tick_dict.get("timestamp"))
        if timestamp is None:
            raise ValueError("time or timestamp key is required")

        return CommonTick(
            timestamp=timestamp,
            symbol=tick_dict["symbol"],
            bid=float(tick_dict["bid"]),
            ask=float(tick_dict["ask"]),
            volume=float(tick_dict.get("volume", 0.0)),
        )

    @staticmethod
    def ensure_decimal_precision(value: float | Decimal) -> Decimal:
        """値をDecimal型に安全に変換

        Args:
            value: 変換する値

        Returns:
            Decimal型の値
        """
        if isinstance(value, Decimal):
            return value
        # Float32精度の値を文字列経由でDecimalに変換
        return Decimal(str(np.float32(value)))

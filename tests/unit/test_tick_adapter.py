"""
TickAdapterクラスのテストスイート

CommonTickとDecimal形式間の変換機能をテストします。
"""

from datetime import datetime
from decimal import Decimal

import numpy as np
import pytest

from src.common.models import Tick as CommonTick
from src.mt5_data_acquisition.tick_adapter import TickAdapter


class TestTickAdapterBasicConversion:
    """基本変換テスト（3ケース）"""

    def test_to_decimal_dict_basic(self):
        """CommonTickからDecimal辞書への基本変換をテスト"""
        # CommonTickを作成
        tick = CommonTick(
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            symbol="EURUSD",
            bid=1.1234,
            ask=1.1236,
            volume=1.0,
        )

        # Decimal辞書に変換
        result = TickAdapter.to_decimal_dict(tick)

        # 検証
        assert result["symbol"] == "EURUSD"
        assert result["time"] == datetime(2025, 1, 1, 12, 0, 0)
        assert isinstance(result["bid"], Decimal)
        assert isinstance(result["ask"], Decimal)
        assert isinstance(result["volume"], Decimal)
        # Float32精度のため、実際のfloat値と比較
        # CommonTickでFloat32変換が適用されるため、tick.bidは既にFloat32精度
        assert result["bid"] == Decimal(str(tick.bid))
        assert result["ask"] == Decimal(str(tick.ask))
        assert result["volume"] == Decimal(str(tick.volume))

    def test_from_decimal_dict_basic(self):
        """Decimal辞書からCommonTickへの基本変換をテスト"""
        # Decimal辞書を作成
        tick_dict = {
            "symbol": "USDJPY",
            "time": datetime(2025, 1, 1, 13, 0, 0),
            "bid": Decimal("110.50"),
            "ask": Decimal("110.52"),
            "volume": Decimal("2.5"),
        }

        # CommonTickに変換
        result = TickAdapter.from_decimal_dict(tick_dict)

        # 検証
        assert result.symbol == "USDJPY"
        assert result.timestamp == datetime(2025, 1, 1, 13, 0, 0)
        assert isinstance(result.bid, float)
        assert isinstance(result.ask, float)
        assert isinstance(result.volume, float)
        # Float32精度での比較
        assert abs(result.bid - 110.50) < 0.01
        assert abs(result.ask - 110.52) < 0.01
        assert abs(result.volume - 2.5) < 0.01

    def test_round_trip_conversion(self):
        """往復変換で元のデータが保持されることをテスト"""
        # オリジナルのCommonTick
        original = CommonTick(
            timestamp=datetime(2025, 1, 1, 14, 30, 45),
            symbol="GBPUSD",
            bid=1.2750,
            ask=1.2752,
            volume=3.3,
        )

        # 往復変換: CommonTick -> Decimal辞書 -> CommonTick
        decimal_dict = TickAdapter.to_decimal_dict(original)
        recovered = TickAdapter.from_decimal_dict(decimal_dict)

        # 検証（Float32精度のため、近似値比較）
        assert recovered.symbol == original.symbol
        assert recovered.timestamp == original.timestamp
        assert abs(recovered.bid - original.bid) < 1e-5
        assert abs(recovered.ask - original.ask) < 1e-5
        assert abs(recovered.volume - original.volume) < 1e-5


class TestTickAdapterAttributeCompatibility:
    """属性名互換性テスト（3ケース）"""

    def test_from_dict_with_time_key(self):
        """timeキーでの辞書からの変換をテスト"""
        tick_dict = {
            "symbol": "AUDUSD",
            "time": datetime(2025, 1, 2, 9, 0, 0),  # timeキーを使用
            "bid": Decimal("0.7500"),
            "ask": Decimal("0.7502"),
            "volume": Decimal("1.5"),
        }

        result = TickAdapter.from_decimal_dict(tick_dict)

        assert result.timestamp == datetime(2025, 1, 2, 9, 0, 0)
        assert result.symbol == "AUDUSD"

    def test_from_dict_with_timestamp_key(self):
        """timestampキーでの辞書からの変換をテスト"""
        tick_dict = {
            "symbol": "NZDUSD",
            "timestamp": datetime(2025, 1, 2, 10, 0, 0),  # timestampキーを使用
            "bid": Decimal("0.6800"),
            "ask": Decimal("0.6802"),
            "volume": Decimal("2.0"),
        }

        result = TickAdapter.from_decimal_dict(tick_dict)

        assert result.timestamp == datetime(2025, 1, 2, 10, 0, 0)
        assert result.symbol == "NZDUSD"

    def test_missing_time_raises_error(self):
        """time/timestampキー不在時にValueErrorが発生することをテスト"""
        tick_dict = {
            "symbol": "USDCAD",
            # timeもtimestampもない
            "bid": Decimal("1.3500"),
            "ask": Decimal("1.3502"),
            "volume": Decimal("1.0"),
        }

        with pytest.raises(ValueError, match="time or timestamp key is required"):
            TickAdapter.from_decimal_dict(tick_dict)


class TestTickAdapterPrecision:
    """精度テスト（3ケース）"""

    def test_float32_precision_maintained(self):
        """Float32制約下での精度維持をテスト"""
        # Float32の精度限界に近い値を使用
        tick = CommonTick(
            timestamp=datetime(2025, 1, 3, 11, 0, 0),
            symbol="XAUUSD",
            bid=1234.5678,  # Float32精度で約7桁
            ask=1234.6789,
            volume=100.123,
        )

        # 変換
        decimal_dict = TickAdapter.to_decimal_dict(tick)
        recovered = TickAdapter.from_decimal_dict(decimal_dict)

        # Float32精度での比較
        np_bid = np.float32(tick.bid)
        np_ask = np.float32(tick.ask)
        np_volume = np.float32(tick.volume)

        assert abs(recovered.bid - np_bid) < 1e-5
        assert abs(recovered.ask - np_ask) < 1e-5
        assert abs(recovered.volume - np_volume) < 1e-5

    def test_decimal_precision_conversion(self):
        """Decimal変換時の精度保持をテスト"""
        # 高精度のDecimal値
        tick_dict = {
            "symbol": "BTCUSD",
            "time": datetime(2025, 1, 3, 12, 0, 0),
            "bid": Decimal("45678.123456789"),  # 高精度
            "ask": Decimal("45678.234567890"),
            "volume": Decimal("0.123456789"),
        }

        # CommonTickに変換（Float32制約が適用される）
        result = TickAdapter.from_decimal_dict(tick_dict)

        # Float32の精度限界内で一致することを確認
        assert abs(result.bid - 45678.123456789) < 10  # Float32では約7桁の精度
        assert abs(result.ask - 45678.234567890) < 10
        assert abs(result.volume - 0.123456789) < 0.0001

    def test_ensure_decimal_precision_helper(self):
        """ヘルパーメソッドの動作確認"""
        # float値のテスト
        float_value = 123.456
        decimal_result = TickAdapter.ensure_decimal_precision(float_value)
        assert isinstance(decimal_result, Decimal)
        assert decimal_result == Decimal(str(np.float32(123.456)))

        # Decimal値のテスト（そのまま返される）
        decimal_value = Decimal("987.654")
        decimal_result = TickAdapter.ensure_decimal_precision(decimal_value)
        assert isinstance(decimal_result, Decimal)
        assert decimal_result == Decimal("987.654")


class TestTickAdapterEdgeCases:
    """エッジケーステスト（3ケース）"""

    def test_zero_values(self):
        """ゼロ値の正確な処理をテスト（volumeのみ）"""
        # bid/askは0より大きい必要があるため、最小値を使用
        tick = CommonTick(
            timestamp=datetime(2025, 1, 4, 8, 0, 0),
            symbol="TESTZV",  # 6文字以上
            bid=0.00001,  # 0より大きい最小値
            ask=0.00002,  # 0より大きい最小値
            volume=0.0,  # volumeは0を許可
        )

        decimal_dict = TickAdapter.to_decimal_dict(tick)

        # Float32精度での検証
        assert abs(float(decimal_dict["bid"]) - 0.00001) < 1e-6
        assert abs(float(decimal_dict["ask"]) - 0.00002) < 1e-6
        assert decimal_dict["volume"] == Decimal("0.0")

        # 逆変換
        recovered = TickAdapter.from_decimal_dict(decimal_dict)
        assert abs(recovered.bid - 0.00001) < 1e-6
        assert abs(recovered.ask - 0.00002) < 1e-6
        assert recovered.volume == 0.0

    def test_large_numbers(self):
        """大きな数値（1e6以上）の処理をテスト"""
        tick = CommonTick(
            timestamp=datetime(2025, 1, 4, 9, 0, 0),
            symbol="BTCUSD",  # 6文字以上
            bid=1234567.89,
            ask=1234568.89,
            volume=9876543.21,
        )

        decimal_dict = TickAdapter.to_decimal_dict(tick)
        recovered = TickAdapter.from_decimal_dict(decimal_dict)

        # Float32精度での比較（大きな数値では相対誤差が重要）
        assert abs((recovered.bid - tick.bid) / tick.bid) < 1e-5
        assert abs((recovered.ask - tick.ask) / tick.ask) < 1e-5
        assert abs((recovered.volume - tick.volume) / tick.volume) < 1e-5

    def test_small_decimals(self):
        """小数点以下多桁（1e-6以下）の処理をテスト"""
        tick_dict = {
            "symbol": "XRPUSD",  # 6文字以上
            "time": datetime(2025, 1, 4, 10, 0, 0),
            "bid": Decimal("0.000012345"),
            "ask": Decimal("0.000012346"),
            "volume": Decimal("0.000000001"),
        }

        result = TickAdapter.from_decimal_dict(tick_dict)

        # 小さな値でも正しく変換されることを確認
        assert result.bid > 0
        assert result.ask > 0
        assert result.volume > 0
        assert abs(result.bid - 0.000012345) < 1e-8
        assert abs(result.ask - 0.000012346) < 1e-8
        # volumeは非常に小さいのでFloat32では0になる可能性がある
        assert result.volume <= 1e-6

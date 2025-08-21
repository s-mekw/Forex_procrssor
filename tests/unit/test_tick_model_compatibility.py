"""Tickモデルの後方互換性テスト

このテストモジュールは、Tickモデルに追加されたtimeプロパティの
後方互換性を確認します。
"""

from datetime import datetime, timezone
import pytest

from src.common.models import Tick


class TestTickBackwardCompatibility:
    """Tickモデルの後方互換性テストクラス"""

    def test_time_property_getter(self):
        """timeプロパティのゲッターがtimestamp属性を返すことを確認"""
        # Arrange
        now = datetime.now(timezone.utc)
        tick = Tick(
            timestamp=now, symbol="USDJPY", bid=150.123, ask=150.125, volume=1000.0
        )

        # Act & Assert
        assert tick.time == tick.timestamp
        assert tick.time == now
        assert isinstance(tick.time, datetime)

    def test_time_property_setter(self):
        """timeプロパティのセッターがtimestamp属性を更新することを確認"""
        # Arrange
        initial_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        new_time = datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc)

        tick = Tick(
            timestamp=initial_time,
            symbol="EURUSD",
            bid=1.1234,
            ask=1.1236,
            volume=500.0,
        )

        # Act
        tick.time = new_time

        # Assert
        assert tick.timestamp == new_time
        assert tick.time == new_time
        assert tick.time == tick.timestamp

    def test_bidirectional_sync(self):
        """timeプロパティとtimestamp属性の双方向同期を確認"""
        # Arrange
        time1 = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        time2 = datetime(2025, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
        time3 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        tick = Tick(
            timestamp=time1, symbol="GBPUSD", bid=1.2500, ask=1.2502, volume=750.0
        )

        # Act & Assert - 初期状態
        assert tick.time == time1
        assert tick.timestamp == time1

        # Act & Assert - timestampを直接更新
        tick.timestamp = time2
        assert tick.time == time2
        assert tick.timestamp == time2

        # Act & Assert - timeプロパティ経由で更新
        tick.time = time3
        assert tick.time == time3
        assert tick.timestamp == time3

    def test_time_property_with_model_validation(self):
        """timeプロパティがモデルのバリデーションと共存することを確認"""
        # Arrange
        now = datetime.now(timezone.utc)

        # Act - 正常なTickインスタンスの作成
        tick = Tick(
            timestamp=now,
            symbol="audusd",  # 小文字でも大文字に変換される
            bid=0.6500,
            ask=0.6502,
            volume=250.0,
        )

        # Assert
        assert tick.time == now
        assert tick.timestamp == now
        assert tick.symbol == "AUDUSD"  # 大文字に正規化されている
        assert tick.spread == pytest.approx(
            0.0002, rel=1e-3
        )  # Float32精度のため許容誤差を調整

    def test_time_property_with_dict_conversion(self):
        """timeプロパティがto_float32_dict()と共存することを確認"""
        # Arrange
        now = datetime.now(timezone.utc)
        tick = Tick(
            timestamp=now, symbol="NZDUSD", bid=0.5800, ask=0.5802, volume=100.0
        )

        # Act
        tick_dict = tick.to_float32_dict()

        # Assert
        assert "timestamp" in tick_dict
        assert tick_dict["timestamp"] == now
        assert tick.time == now  # timeプロパティも引き続き動作

    def test_multiple_time_updates(self):
        """複数回の時刻更新が正しく動作することを確認"""
        # Arrange
        times = [
            datetime(2025, 1, 1, i, 0, 0, tzinfo=timezone.utc) for i in range(10, 15)
        ]

        tick = Tick(
            timestamp=times[0], symbol="USDCAD", bid=1.3500, ask=1.3502, volume=300.0
        )

        # Act & Assert
        for time in times[1:]:
            tick.time = time
            assert tick.time == time
            assert tick.timestamp == time

    def test_time_property_type_preservation(self):
        """timeプロパティがdatetime型を保持することを確認"""
        # Arrange
        now = datetime.now(timezone.utc)
        tick = Tick(
            timestamp=now, symbol="USDCHF", bid=0.8900, ask=0.8902, volume=200.0
        )

        # Act
        retrieved_time = tick.time

        # Assert
        assert isinstance(retrieved_time, datetime)
        assert retrieved_time.tzinfo is not None  # タイムゾーン情報も保持
        assert retrieved_time == now

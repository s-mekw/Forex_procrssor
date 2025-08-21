"""
TickToBarConverter のユニットテスト

このモジュールはティックデータから1分足バー（OHLCV）への変換機能をテストします。
TDDアプローチにより、実装前にテストを作成しています。
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

# 実装されたモジュールをインポート
from src.mt5_data_acquisition.tick_to_bar import Tick, TickToBarConverter


class TestTickToBarConverter:
    """TickToBarConverterクラスのテストスイート"""

    @pytest.fixture
    def converter(self):
        """TickToBarConverterのインスタンスを作成するフィクスチャ"""
        return TickToBarConverter(symbol="EURUSD", timeframe=60)

    @pytest.fixture
    def sample_ticks(self) -> list[dict]:
        """テスト用のサンプルティックデータを生成"""
        base_time = datetime(2025, 1, 20, 10, 0, 0)
        return [
            {
                "symbol": "EURUSD",
                "time": base_time + timedelta(seconds=0),
                "bid": Decimal("1.04200"),
                "ask": Decimal("1.04210"),
                "volume": Decimal("1.0"),
            },
            {
                "symbol": "EURUSD",
                "time": base_time + timedelta(seconds=15),
                "bid": Decimal("1.04220"),
                "ask": Decimal("1.04230"),
                "volume": Decimal("0.5"),
            },
            {
                "symbol": "EURUSD",
                "time": base_time + timedelta(seconds=30),
                "bid": Decimal("1.04180"),
                "ask": Decimal("1.04190"),
                "volume": Decimal("2.0"),
            },
            {
                "symbol": "EURUSD",
                "time": base_time + timedelta(seconds=45),
                "bid": Decimal("1.04250"),
                "ask": Decimal("1.04260"),
                "volume": Decimal("1.5"),
            },
            {
                "symbol": "EURUSD",
                "time": base_time + timedelta(seconds=60),  # 次の分
                "bid": Decimal("1.04240"),
                "ask": Decimal("1.04250"),
                "volume": Decimal("1.0"),
            },
        ]

    def test_converter_initialization(self):
        """TickToBarConverterの初期化テスト"""
        converter = TickToBarConverter(symbol="EURUSD", timeframe=60)
        assert converter.symbol == "EURUSD"
        assert converter.timeframe == 60
        assert converter.current_bar is None
        assert len(converter.completed_bars) == 0

    def test_single_minute_bar_generation(self, converter, sample_ticks):
        """1分足バー生成の基本テスト（複数ティックから1分足を生成）"""
        # 最初の4つのティック（同じ分内）を追加
        for tick_data in sample_ticks[:4]:
            tick = Tick(**tick_data)
            completed_bar = converter.add_tick(tick)
            # 分が完了していないのでNoneが返される
            assert completed_bar is None

        # 5つ目のティック（次の分）を追加
        tick = Tick(**sample_ticks[4])
        completed_bar = converter.add_tick(tick)

        # 前の分のバーが完成して返される
        assert completed_bar is not None
        assert completed_bar.symbol == "EURUSD"
        assert completed_bar.open == Decimal("1.04200")  # 最初のティックのbid
        assert completed_bar.high == Decimal("1.04250")  # 最高値
        assert completed_bar.low == Decimal("1.04180")   # 最安値
        assert completed_bar.close == Decimal("1.04250") # 最後のティックのbid
        assert completed_bar.volume == Decimal("5.0")    # 合計ボリューム

    def test_timestamp_alignment(self, converter, sample_ticks):
        """タイムスタンプ整合性のテスト（バーの開始/終了時刻が正しいこと）"""
        # ティックを追加
        for tick_data in sample_ticks[:4]:
            tick = Tick(**tick_data)
            converter.add_tick(tick)

        # 次の分のティックで前の分が完成
        tick = Tick(**sample_ticks[4])
        completed_bar = converter.add_tick(tick)

        assert completed_bar is not None
        # バーの開始時刻は分の開始時刻（秒を0に正規化）
        expected_start = datetime(2025, 1, 20, 10, 0, 0)
        assert completed_bar.time == expected_start
        # バーの終了時刻は分の終了時刻
        expected_end = datetime(2025, 1, 20, 10, 0, 59, 999999)
        assert completed_bar.end_time == expected_end

    def test_get_current_incomplete_bar(self, converter, sample_ticks):
        """未完成バーの取得テスト"""
        # 最初の2つのティックを追加
        for tick_data in sample_ticks[:2]:
            tick = Tick(**tick_data)
            converter.add_tick(tick)

        # 現在の未完成バーを取得
        current_bar = converter.get_current_bar()

        assert current_bar is not None
        assert current_bar.symbol == "EURUSD"
        assert current_bar.open == Decimal("1.04200")  # 最初のティック
        assert current_bar.high == Decimal("1.04220")  # 現時点での最高値
        assert current_bar.low == Decimal("1.04200")   # 現時点での最安値
        assert current_bar.close == Decimal("1.04220") # 最新のティック
        assert current_bar.volume == Decimal("1.5")    # 合計ボリューム
        assert current_bar.is_complete is False        # 未完成フラグ

    def test_ohlcv_accuracy(self, converter):
        """OHLCV値の正確性テスト"""
        base_time = datetime(2025, 1, 20, 10, 0, 0)

        # 特定のパターンでティックを作成
        ticks_data = [
            {"symbol": "EURUSD", "time": base_time, "bid": Decimal("1.04200"), "ask": Decimal("1.04210"), "volume": Decimal("1.0")},  # Open
            {"symbol": "EURUSD", "time": base_time + timedelta(seconds=10), "bid": Decimal("1.04300"), "ask": Decimal("1.04310"), "volume": Decimal("0.5")},  # High
            {"symbol": "EURUSD", "time": base_time + timedelta(seconds=20), "bid": Decimal("1.04100"), "ask": Decimal("1.04110"), "volume": Decimal("0.5")},  # Low
            {"symbol": "EURUSD", "time": base_time + timedelta(seconds=30), "bid": Decimal("1.04150"), "ask": Decimal("1.04160"), "volume": Decimal("1.0")},  # 中間
            {"symbol": "EURUSD", "time": base_time + timedelta(seconds=50), "bid": Decimal("1.04180"), "ask": Decimal("1.04190"), "volume": Decimal("0.5")},  # Close
        ]

        # ティックを追加
        for tick_data in ticks_data:
            tick = Tick(**tick_data)
            converter.add_tick(tick)

        # 次の分のティックで完成
        next_tick = Tick(
            symbol="EURUSD",
            time=base_time + timedelta(seconds=60),
            bid=Decimal("1.04200"),
            ask=Decimal("1.04210"),
            volume=Decimal("1.0")
        )
        completed_bar = converter.add_tick(next_tick)

        # OHLCV値を検証
        assert completed_bar.open == Decimal("1.04200")   # 最初のティック
        assert completed_bar.high == Decimal("1.04300")   # 最高値
        assert completed_bar.low == Decimal("1.04100")    # 最安値
        assert completed_bar.close == Decimal("1.04180")  # 最後のティック
        assert completed_bar.volume == Decimal("3.5")     # 合計ボリューム

    def test_empty_bar_handling(self, converter):
        """ティックがない場合のバー処理テスト"""
        # 現在のバーを取得（まだティックがない）
        current_bar = converter.get_current_bar()
        assert current_bar is None

        # 完成したバーのリストも空
        assert len(converter.completed_bars) == 0

    def test_multiple_bars_generation(self, converter):
        """複数のバー生成テスト"""
        base_time = datetime(2025, 1, 20, 10, 0, 0)

        # 3分間のティックデータを生成
        for minute in range(3):
            for second in [0, 15, 30, 45]:
                tick = Tick(
                    symbol="EURUSD",
                    time=base_time + timedelta(minutes=minute, seconds=second),
                    bid=Decimal("1.04200") + Decimal(str(minute * 0.0001)),
                    ask=Decimal("1.04210") + Decimal(str(minute * 0.0001)),
                    volume=Decimal("1.0")
                )
                completed_bar = converter.add_tick(tick)

                # 分の最初のティックで前の分が完成（最初の分を除く）
                if minute > 0 and second == 0:
                    assert completed_bar is not None
                else:
                    assert completed_bar is None

        # 3分目を完成させる
        final_tick = Tick(
            symbol="EURUSD",
            time=base_time + timedelta(minutes=3),
            bid=Decimal("1.04230"),
            ask=Decimal("1.04240"),
            volume=Decimal("1.0")
        )
        completed_bar = converter.add_tick(final_tick)
        assert completed_bar is not None

        # 完成したバーは3つ
        assert len(converter.completed_bars) == 3

    def test_volume_aggregation(self, converter, sample_ticks):
        """ボリューム集計の正確性テスト"""
        total_volume = Decimal("0")

        # ティックを追加してボリュームを集計
        for tick_data in sample_ticks[:4]:
            tick = Tick(**tick_data)
            converter.add_tick(tick)
            total_volume += tick.volume

        # 現在のバーのボリュームを確認
        current_bar = converter.get_current_bar()
        assert current_bar.volume == total_volume

    def test_tick_gap_warning(self, converter, caplog):
        """ティック欠損検知のテスト"""
        import logging

        # ログレベルをWARNINGに設定
        caplog.set_level(logging.WARNING)

        base_time = datetime(2025, 1, 20, 10, 0, 0)

        # 最初のティック
        tick1 = Tick(
            symbol="EURUSD",
            time=base_time,
            bid=Decimal("1.04200"),
            ask=Decimal("1.04210"),
            volume=Decimal("1.0")
        )
        converter.add_tick(tick1)

        # 30秒以上後のティック（ギャップあり）
        tick2 = Tick(
            symbol="EURUSD",
            time=base_time + timedelta(seconds=35),
            bid=Decimal("1.04220"),
            ask=Decimal("1.04230"),
            volume=Decimal("1.0")
        )
        converter.add_tick(tick2)

        # 警告ログが出力されたことを確認
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"
        assert "tick_gap_detected" in caplog.records[0].message
        assert "\"gap_seconds\": 35.0" in caplog.records[0].message

    def test_bid_ask_spread_tracking(self, converter):
        """Bid/Askスプレッドの追跡テスト"""
        base_time = datetime(2025, 1, 20, 10, 0, 0)

        tick = Tick(
            symbol="EURUSD",
            time=base_time,
            bid=Decimal("1.04200"),
            ask=Decimal("1.04210"),
            volume=Decimal("1.0")
        )
        converter.add_tick(tick)

        current_bar = converter.get_current_bar()
        # スプレッド情報が保持されているか確認
        assert hasattr(current_bar, 'avg_spread')
        assert current_bar.avg_spread == Decimal("0.00010")

    def test_bar_completion_callback(self, converter):
        """バー完成時のコールバック機能テスト"""
        from unittest.mock import Mock

        # コールバック関数のモック
        callback = Mock()
        converter.on_bar_complete = callback

        base_time = datetime(2025, 1, 20, 10, 0, 0)

        # 1分内のティック
        for second in [0, 15, 30, 45]:
            tick = Tick(
                symbol="EURUSD",
                time=base_time + timedelta(seconds=second),
                bid=Decimal("1.04200"),
                ask=Decimal("1.04210"),
                volume=Decimal("1.0")
            )
            converter.add_tick(tick)

        # コールバックはまだ呼ばれていない
        callback.assert_not_called()

        # 次の分のティックで完成
        next_tick = Tick(
            symbol="EURUSD",
            time=base_time + timedelta(seconds=60),
            bid=Decimal("1.04200"),
            ask=Decimal("1.04210"),
            volume=Decimal("1.0")
        )
        converter.add_tick(next_tick)

        # コールバックが呼ばれたことを確認
        callback.assert_called_once()

    def test_invalid_price_handling(self, converter, caplog):
        """負の価格の処理テスト"""
        import logging

        from pydantic import ValidationError

        # ログレベルをERRORに設定
        caplog.set_level(logging.ERROR)

        base_time = datetime(2025, 1, 20, 10, 0, 0)

        # 負の価格を持つ無効なティック
        with pytest.raises(ValidationError):
            Tick(
                symbol="EURUSD",
                time=base_time,
                bid=Decimal("-1.04200"),  # 負の価格
                ask=Decimal("1.04210"),
                volume=Decimal("1.0")
            )

    def test_timestamp_reversal_handling(self, converter, caplog):
        """タイムスタンプ逆転の処理テスト"""
        import logging

        # ログレベルをERRORに設定
        caplog.set_level(logging.ERROR)

        base_time = datetime(2025, 1, 20, 10, 0, 0)

        # 最初のティック
        tick1 = Tick(
            symbol="EURUSD",
            time=base_time,
            bid=Decimal("1.04200"),
            ask=Decimal("1.04210"),
            volume=Decimal("1.0")
        )
        converter.add_tick(tick1)

        # 過去のタイムスタンプを持つティック
        tick2 = Tick(
            symbol="EURUSD",
            time=base_time - timedelta(seconds=10),  # 10秒前
            bid=Decimal("1.04220"),
            ask=Decimal("1.04230"),
            volume=Decimal("1.0")
        )
        result = converter.add_tick(tick2)

        # ティックは破棄されてNoneが返る
        assert result is None

        # エラーログが出力されたことを確認
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "ERROR"
        assert "timestamp_reversal" in caplog.records[0].message

        # 現在のバーは最初のティックのまま
        current_bar = converter.get_current_bar()
        assert current_bar.tick_count == 1

    def test_zero_price_handling(self, converter):
        """ゼロ価格の処理テスト"""
        from pydantic import ValidationError

        base_time = datetime(2025, 1, 20, 10, 0, 0)

        # ゼロ価格を持つ無効なティック
        with pytest.raises(ValidationError):
            Tick(
                symbol="EURUSD",
                time=base_time,
                bid=Decimal("0"),  # ゼロ価格
                ask=Decimal("1.04210"),
                volume=Decimal("1.0")
            )

    def test_invalid_spread_handling(self, converter):
        """ask < bidの処理テスト"""
        from pydantic import ValidationError

        base_time = datetime(2025, 1, 20, 10, 0, 0)

        # ask < bidの無効なスプレッド
        with pytest.raises(ValidationError):
            Tick(
                symbol="EURUSD",
                time=base_time,
                bid=Decimal("1.04220"),
                ask=Decimal("1.04210"),  # ask < bid
                volume=Decimal("1.0")
            )

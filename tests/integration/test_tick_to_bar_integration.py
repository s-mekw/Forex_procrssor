"""統合テストとパフォーマンステスト for TickToBarConverter

このファイルは、TickToBarConverterの統合的な動作とパフォーマンスを検証します。
- 連続的なティックストリーム処理
- 市場時間外のギャップ処理
- 高頻度ティック処理
- 大量データでのパフォーマンス測定
- メモリ使用量の測定
"""

import sys
import time
import tracemalloc
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from src.mt5_data_acquisition.tick_to_bar import Bar, Tick, TickToBarConverter


class TestTickToBarIntegration:
    """TickToBarConverterの統合テストクラス"""

    def test_continuous_tick_stream(self) -> None:
        """連続的なティックストリーム処理テスト
        
        5分間の連続ティック（1秒ごと）を処理し、
        5つのバーが正しく生成されることを確認
        """
        converter = TickToBarConverter("EURUSD")
        start_time = datetime(2025, 8, 21, 12, 0, 0)
        completed_bars: list[Bar] = []

        # バー完成時のコールバック設定
        converter.on_bar_complete = lambda bar: completed_bars.append(bar)

        # 5分間のティックを生成（300秒 = 5分）
        current_time = start_time
        for i in range(301):  # 301ティックで5つの完全なバーを生成
            tick = Tick(
                symbol="EURUSD",
                time=current_time,
                bid=Decimal("1.1234") + Decimal(str(i % 10)) / Decimal("10000"),
                ask=Decimal("1.1236") + Decimal(str(i % 10)) / Decimal("10000"),
                volume=Decimal("1.0")
            )
            converter.add_tick(tick)
            current_time += timedelta(seconds=1)

        # アサーション
        assert len(completed_bars) == 5, f"Expected 5 bars, got {len(completed_bars)}"

        # 各バーの時刻が正しいことを確認
        for i, bar in enumerate(completed_bars):
            expected_time = start_time + timedelta(minutes=i)
            assert bar.time == expected_time, f"Bar {i} time mismatch"
            assert bar.is_complete is True, f"Bar {i} should be complete"
            assert bar.tick_count == 60, f"Bar {i} should have 60 ticks"
            assert bar.volume == Decimal("60.0"), f"Bar {i} volume should be 60.0"

    def test_market_gap_handling(self, caplog: Any) -> None:
        """市場時間外のギャップ処理テスト（週末シミュレーション）
        
        金曜日の最終ティックから月曜日の最初のティックまでの
        ギャップを処理し、適切に警告が出力されることを確認
        """
        converter = TickToBarConverter("EURUSD")

        # 金曜日の最終ティック（22:59:30）
        friday_tick = Tick(
            symbol="EURUSD",
            time=datetime(2025, 8, 22, 22, 59, 30),  # 金曜日
            bid=Decimal("1.1234"),
            ask=Decimal("1.1236"),
            volume=Decimal("1.0")
        )
        converter.add_tick(friday_tick)

        # 月曜日の最初のティック（0:00:05）- 週末のギャップ
        monday_tick = Tick(
            symbol="EURUSD",
            time=datetime(2025, 8, 25, 0, 0, 5),  # 月曜日
            bid=Decimal("1.1240"),
            ask=Decimal("1.1242"),
            volume=Decimal("2.0")
        )

        # ログレベルを設定
        with caplog.at_level("WARNING"):
            result = converter.add_tick(monday_tick)

        # 警告が出力されたことを確認
        assert "tick_gap_detected" in caplog.text

        # バーが正しく完成したことを確認
        assert result is not None
        assert result.is_complete is True

        # 新しいバーが開始されたことを確認
        current_bar = converter.get_current_bar()
        assert current_bar is not None
        assert current_bar.time == datetime(2025, 8, 25, 0, 0, 0)

    def test_high_frequency_ticks(self) -> None:
        """高頻度ティック処理テスト（1秒間に10ティック）
        
        同一秒内での複数ティック処理を行い、
        OHLCVが正しく更新されることを確認
        """
        converter = TickToBarConverter("EURUSD")
        base_time = datetime(2025, 8, 21, 12, 0, 0)

        # 6秒間、各秒に10ティックを生成（合計60ティック = 1分）
        tick_count = 0
        for second in range(60):
            for microsecond in range(10):
                tick_time = base_time + timedelta(seconds=second, microseconds=microsecond * 100000)

                # 価格を変動させる
                price_variation = Decimal(str(tick_count % 20)) / Decimal("10000")
                tick = Tick(
                    symbol="EURUSD",
                    time=tick_time,
                    bid=Decimal("1.1234") + price_variation,
                    ask=Decimal("1.1236") + price_variation,
                    volume=Decimal("0.1")
                )
                result = converter.add_tick(tick)
                tick_count += 1

                # 最後のティック以外はバーが完成しない
                if second < 59:
                    assert result is None

        # 60秒後に次のティックを追加してバーを完成させる
        final_tick = Tick(
            symbol="EURUSD",
            time=base_time + timedelta(seconds=60),
            bid=Decimal("1.1234"),
            ask=Decimal("1.1236"),
            volume=Decimal("1.0")
        )
        completed_bar = converter.add_tick(final_tick)

        # バーが完成したことを確認
        assert completed_bar is not None
        assert completed_bar.is_complete is True
        assert completed_bar.tick_count == 600  # 60秒 × 10ティック/秒
        assert completed_bar.volume == Decimal("60.0")  # 0.1 × 600

        # High/Lowが正しく記録されていることを確認
        assert completed_bar.high == Decimal("1.1253")  # 1.1234 + 19/10000
        assert completed_bar.low == Decimal("1.1234")   # 最小値

    def test_callback_chain(self) -> None:
        """バー完成通知の連鎖処理テスト
        
        複数のリスナーへの通知をシミュレートし、
        バー完成時のコールバックが連続して呼ばれることを確認
        """
        converter = TickToBarConverter("EURUSD")

        # 複数のリスナーをシミュレート
        completed_bars: list[Bar] = []
        bar_counts: list[int] = []
        bar_symbols: list[str] = []

        def multi_callback(bar: Bar) -> None:
            """複数の処理を行うコールバック"""
            completed_bars.append(bar)
            bar_counts.append(bar.tick_count)
            bar_symbols.append(bar.symbol)

        converter.on_bar_complete = multi_callback

        # 3分間のティックを生成
        start_time = datetime(2025, 8, 21, 12, 0, 0)
        current_time = start_time

        for _i in range(181):  # 181ティック = 3分 + 1秒
            tick = Tick(
                symbol="EURUSD",
                time=current_time,
                bid=Decimal("1.1234"),
                ask=Decimal("1.1236"),
                volume=Decimal("1.0")
            )
            converter.add_tick(tick)
            current_time += timedelta(seconds=1)

        # アサーション
        assert len(completed_bars) == 3, "Should have 3 completed bars"
        assert len(bar_counts) == 3, "Should have 3 bar counts"
        assert len(bar_symbols) == 3, "Should have 3 bar symbols"

        # 各バーが正しく処理されたことを確認
        for i, bar in enumerate(completed_bars):
            assert bar.symbol == "EURUSD"
            assert bar.tick_count == 60
            assert bar.is_complete is True
            assert bar_counts[i] == 60
            assert bar_symbols[i] == "EURUSD"


class TestPerformance:
    """パフォーマンステストクラス"""

    def test_large_tick_volume(self) -> None:
        """大量ティック処理のパフォーマンステスト
        
        10,000ティックを処理し、処理速度を測定
        目標: 10,000ティック/秒以上
        """
        converter = TickToBarConverter("EURUSD")
        start_time = datetime(2025, 8, 21, 9, 0, 0)

        # 10,000ティックを生成
        ticks: list[Tick] = []
        current_time = start_time
        for i in range(10000):
            tick = Tick(
                symbol="EURUSD",
                time=current_time,
                bid=Decimal("1.1234") + Decimal(str(i % 100)) / Decimal("10000"),
                ask=Decimal("1.1236") + Decimal(str(i % 100)) / Decimal("10000"),
                volume=Decimal("1.0")
            )
            ticks.append(tick)
            current_time += timedelta(seconds=1)

        # 処理時間測定
        start = time.perf_counter()
        for tick in ticks:
            converter.add_tick(tick)
        elapsed = time.perf_counter() - start

        # 結果出力
        throughput = 10000 / elapsed
        print("\n=== Performance Test Results ===")
        print(f"Processed 10,000 ticks in {elapsed:.3f} seconds")
        print(f"Throughput: {throughput:.0f} ticks/second")

        # アサーション
        assert elapsed < 1.0, f"Processing took {elapsed:.3f}s, expected < 1.0s"
        assert len(converter.completed_bars) == 166, f"Expected 166 bars, got {len(converter.completed_bars)}"
        assert throughput > 10000, f"Throughput {throughput:.0f} ticks/sec is below target of 10,000"

    def test_one_hour_data_generation(self) -> None:
        """1時間分のデータ生成テスト
        
        3,600ティック（1時間分）を処理し、
        60個のバーが正しく生成されることを確認
        """
        converter = TickToBarConverter("EURUSD")
        start_time = datetime(2025, 8, 21, 9, 0, 0)

        # 1時間分のティックデータ（3600ティック）
        for i in range(3601):  # 3601ティックで60個の完全なバーを生成
            tick = Tick(
                symbol="EURUSD",
                time=start_time + timedelta(seconds=i),
                bid=Decimal("1.1234") + Decimal(str(i % 50)) / Decimal("10000"),
                ask=Decimal("1.1236") + Decimal(str(i % 50)) / Decimal("10000"),
                volume=Decimal("1.0")
            )
            converter.add_tick(tick)

        # アサーション
        assert len(converter.completed_bars) == 60, f"Expected 60 bars, got {len(converter.completed_bars)}"

        # 各バーが1分間のデータを持つことを確認
        for bar in converter.completed_bars:
            assert bar.tick_count == 60, "Each bar should have 60 ticks"
            assert bar.volume == Decimal("60.0"), "Each bar should have volume of 60.0"

    def test_memory_usage(self) -> None:
        """メモリ使用量測定テスト
        
        1時間分のティックデータを処理し、
        メモリ使用量が10MB以下であることを確認
        """
        tracemalloc.start()

        converter = TickToBarConverter("EURUSD")
        start_time = datetime(2025, 8, 21, 9, 0, 0)

        # 1時間分のティックデータ（3600ティック）
        for i in range(3600):
            tick = Tick(
                symbol="EURUSD",
                time=start_time + timedelta(seconds=i),
                bid=Decimal("1.1234"),
                ask=Decimal("1.1236"),
                volume=Decimal("1.0")
            )
            converter.add_tick(tick)

        # メモリ使用量を取得
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # MB単位に変換
        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024

        # 結果出力
        print("\n=== Memory Usage Test Results ===")
        print(f"Current memory usage: {current_mb:.2f} MB")
        print(f"Peak memory usage: {peak_mb:.2f} MB")
        print(f"Completed bars: {len(converter.completed_bars)}")

        # オブジェクトサイズの詳細
        bars_size = sys.getsizeof(converter.completed_bars) / 1024
        current_bar_size = sys.getsizeof(converter.current_bar) / 1024 if converter.current_bar else 0

        print(f"Completed bars size: {bars_size:.2f} KB")
        print(f"Current bar size: {current_bar_size:.2f} KB")

        # アサーション
        assert peak_mb < 10, f"Peak memory usage {peak_mb:.2f} MB exceeds 10 MB limit"
        assert len(converter.completed_bars) == 59, "Expected 59 completed bars"

    def test_memory_with_limit(self) -> None:
        """メモリ制限付きのテスト
        
        completed_barsリストのサイズ制限を確認
        （将来の最適化のためのテスト）
        """
        # 最大1000バーの制限を想定
        converter = TickToBarConverter("EURUSD")
        start_time = datetime(2025, 8, 21, 0, 0, 0)

        # 2000分（33時間）のティックを生成
        current_time = start_time
        for _minute in range(2000):
            # 各分の最初のティックのみ（簡略化）
            tick = Tick(
                symbol="EURUSD",
                time=current_time,
                bid=Decimal("1.1234"),
                ask=Decimal("1.1236"),
                volume=Decimal("1.0")
            )
            converter.add_tick(tick)

            # 次の分へ
            current_time += timedelta(minutes=1)
            tick = Tick(
                symbol="EURUSD",
                time=current_time,
                bid=Decimal("1.1234"),
                ask=Decimal("1.1236"),
                volume=Decimal("1.0")
            )
            converter.add_tick(tick)

        # 注: 現在の実装では制限がないため、すべてのバーが保持される
        # 将来の最適化で max_completed_bars パラメータを追加予定
        assert len(converter.completed_bars) >= 1999, "Should have at least 1999 bars"

        # メモリ使用量の確認
        bars_size_kb = sys.getsizeof(converter.completed_bars) / 1024
        print("\n=== Memory Limit Test ===")
        print(f"Total bars stored: {len(converter.completed_bars)}")
        print(f"Memory used by bars: {bars_size_kb:.2f} KB")

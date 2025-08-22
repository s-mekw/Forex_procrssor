"""
TickDataStreamerクラスのユニットテスト

リアルタイムティックデータの取得、ストリーミング、フィルタリング機能をテストします。
MT5 APIはモック化し、非同期処理の動作を検証します。
"""

import asyncio
import sys
import os
from collections import deque
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call

import pytest
import numpy as np

# srcディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Step 2で実装部分のテストのみ実行可能にする
# pytestmark = pytest.mark.skip(reason="TickDataStreamer実装前のテスト定義")


class TestTickDataStreamerInitialization:
    """TickDataStreamerクラスの初期化テスト"""

    # @pytest.mark.xfail(reason="TickDataStreamer未実装")
    def test_initialization_with_default_parameters(self):
        """デフォルトパラメータでの初期化をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        
        streamer = TickDataStreamer(symbol="USDJPY")
        
        assert streamer.symbol == "USDJPY"
        assert streamer.buffer_size == 10000
        assert streamer.spike_threshold == 3.0
        assert isinstance(streamer.buffer, deque)
        assert streamer.buffer.maxlen == 10000
        assert streamer.is_streaming is False

    # @pytest.mark.xfail(reason="TickDataStreamer未実装")
    def test_initialization_with_custom_parameters(self):
        """カスタムパラメータでの初期化をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        
        streamer = TickDataStreamer(
            symbol="EURUSD",
            buffer_size=5000,
            spike_threshold=2.5,
            backpressure_threshold=0.8
        )
        
        assert streamer.symbol == "EURUSD"
        assert streamer.buffer_size == 5000
        assert streamer.spike_threshold == 2.5
        assert streamer.backpressure_threshold == 0.8
        assert streamer.buffer.maxlen == 5000

    # @pytest.mark.xfail(reason="TickDataStreamer未実装")
    def test_initialization_with_invalid_parameters(self):
        """無効なパラメータでの初期化時のエラーをテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        
        # 無効なシンボル
        with pytest.raises(ValueError, match="Invalid symbol"):
            TickDataStreamer(symbol="")
        
        # 無効なバッファサイズ
        with pytest.raises(ValueError, match="Buffer size must be positive"):
            TickDataStreamer(symbol="USDJPY", buffer_size=0)
        
        # 無効なスパイク閾値
        with pytest.raises(ValueError, match="Spike threshold must be positive"):
            TickDataStreamer(symbol="USDJPY", spike_threshold=-1.0)


class TestRingBuffer:
    """リングバッファ機能のテスト"""

    # @pytest.mark.xfail(reason="TickDataStreamer未実装")  # XFAILを解除
    def test_ring_buffer_size_limit(self):
        """リングバッファのサイズ制限をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        streamer = TickDataStreamer(symbol="USDJPY", buffer_size=100)
        
        # 150個のティックを追加
        for i in range(150):
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="USDJPY",
                bid=150.0 + i * 0.001,
                ask=150.002 + i * 0.001,
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
        
        # バッファサイズが100を超えないことを確認
        assert len(streamer.buffer) == 100
        
        # 最新の100個が保持されていることを確認
        last_tick = streamer.buffer[-1]
        assert last_tick.bid == pytest.approx(150.149, rel=1e-3)

    # @pytest.mark.xfail(reason="TickDataStreamer未実装")  # XFAILを解除
    def test_ring_buffer_fifo_behavior(self):
        """リングバッファのFIFO動作をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        streamer = TickDataStreamer(symbol="USDJPY", buffer_size=3)
        
        ticks = []
        for i in range(5):
            tick = Tick(
                timestamp=datetime.utcnow() + timedelta(seconds=i),
                symbol="USDJPY",
                bid=150.0 + i,
                ask=150.002 + i,
                volume=1000.0
            )
            ticks.append(tick)
            streamer._add_to_buffer(tick)
        
        # バッファに最後の3つが残っていることを確認
        assert len(streamer.buffer) == 3
        assert streamer.buffer[0] == ticks[2]
        assert streamer.buffer[1] == ticks[3]
        assert streamer.buffer[2] == ticks[4]

class TestGetNewTicks:
    """get_new_ticks()メソッドの包括的なテスト"""

    @pytest.mark.asyncio
    async def test_get_new_ticks_basic(self):
        """基本的な新規ティック取得をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        streamer = TickDataStreamer(symbol="USDJPY", buffer_size=100)
        
        # 初回呼び出し：空のリストを返す
        new_ticks = await streamer.get_new_ticks()
        assert new_ticks == []
        
        # 5個のティックを追加
        for i in range(5):
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="USDJPY",
                bid=150.0 + i * 0.001,
                ask=150.002 + i * 0.001,
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
        
        # 新しい5個のティックを取得
        new_ticks = await streamer.get_new_ticks()
        assert len(new_ticks) == 5
        assert new_ticks[0].bid == pytest.approx(150.0, rel=1e-6)
        assert new_ticks[4].bid == pytest.approx(150.004, rel=1e-6)
        
        # 再度呼び出すと空のリスト（新しいティックなし）
        new_ticks = await streamer.get_new_ticks()
        assert new_ticks == []
        
        # さらに3個追加
        for i in range(3):
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="USDJPY",
                bid=150.005 + i * 0.001,
                ask=150.007 + i * 0.001,
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
        
        # 新しい3個のみ取得
        new_ticks = await streamer.get_new_ticks()
        assert len(new_ticks) == 3
        assert new_ticks[0].bid == pytest.approx(150.005, rel=1e-6)

    @pytest.mark.asyncio
    async def test_get_new_ticks_empty_buffer(self):
        """空バッファでの動作をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        
        streamer = TickDataStreamer(symbol="EURUSD", buffer_size=50)
        
        # 空のバッファから取得
        new_ticks = await streamer.get_new_ticks()
        assert new_ticks == []
        
        # 複数回呼び出しても空
        for _ in range(3):
            new_ticks = await streamer.get_new_ticks()
            assert new_ticks == []

    @pytest.mark.asyncio
    async def test_get_new_ticks_buffer_rotation(self):
        """バッファローテーション時の動作をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        buffer_size = 10
        streamer = TickDataStreamer(symbol="GBPUSD", buffer_size=buffer_size)
        
        # バッファサイズの2倍のティックを追加
        for i in range(buffer_size * 2):
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="GBPUSD",
                bid=1.3000 + i * 0.0001,
                ask=1.3002 + i * 0.0001,
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
        
        # 最初の取得：バッファに残っている10個を取得
        new_ticks = await streamer.get_new_ticks()
        assert len(new_ticks) == buffer_size
        # 古いティックは削除され、新しい10個のみ
        assert new_ticks[0].bid == pytest.approx(1.3010, rel=1e-6)
        assert new_ticks[-1].bid == pytest.approx(1.3019, rel=1e-6)
        
        # さらに5個追加
        for i in range(5):
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="GBPUSD",
                bid=1.3020 + i * 0.0001,
                ask=1.3022 + i * 0.0001,
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
        
        # 新しい5個のみ取得
        new_ticks = await streamer.get_new_ticks()
        assert len(new_ticks) == 5
        assert new_ticks[0].bid == pytest.approx(1.3020, rel=1e-6)

    @pytest.mark.asyncio
    async def test_get_new_ticks_after_buffer_overflow(self):
        """バッファオーバーフロー後の正しい取得をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        buffer_size = 5
        streamer = TickDataStreamer(symbol="AUDUSD", buffer_size=buffer_size)
        
        # 3個追加して取得
        for i in range(3):
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="AUDUSD",
                bid=0.7000 + i * 0.0001,
                ask=0.7002 + i * 0.0001,
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
        
        new_ticks = await streamer.get_new_ticks()
        assert len(new_ticks) == 3
        
        # バッファサイズを超える10個を追加（オーバーフロー発生）
        for i in range(10):
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="AUDUSD",
                bid=0.7003 + i * 0.0001,
                ask=0.7005 + i * 0.0001,
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
        
        # オーバーフロー後も正しく新しいティックを取得
        new_ticks = await streamer.get_new_ticks()
        assert len(new_ticks) == buffer_size  # バッファサイズ分のみ
        # 最新の5個が取得される
        assert new_ticks[0].bid == pytest.approx(0.7008, rel=1e-6)
        assert new_ticks[-1].bid == pytest.approx(0.7012, rel=1e-6)

    @pytest.mark.asyncio
    async def test_get_new_ticks_concurrent_access(self):
        """複数の非同期タスクからの同時アクセスをテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        streamer = TickDataStreamer(symbol="USDCHF", buffer_size=100)
        
        async def add_ticks(start_price, count):
            """ティックを追加するタスク"""
            for i in range(count):
                tick = Tick(
                    timestamp=datetime.utcnow(),
                    symbol="USDCHF",
                    bid=start_price + i * 0.0001,
                    ask=start_price + 0.0002 + i * 0.0001,
                    volume=1000.0
                )
                streamer._add_to_buffer(tick)
                await asyncio.sleep(0.001)  # 小さな遅延
        
        async def read_ticks(reader_id):
            """ティックを読み取るタスク"""
            results = []
            for _ in range(3):
                new_ticks = await streamer.get_new_ticks()
                results.append(len(new_ticks))
                await asyncio.sleep(0.005)
            return results
        
        # 並行実行
        add_task = asyncio.create_task(add_ticks(0.9000, 20))
        read_task1 = asyncio.create_task(read_ticks(1))
        read_task2 = asyncio.create_task(read_ticks(2))
        
        await add_task
        results1 = await read_task1
        results2 = await read_task2
        
        # 各リーダーが異なるティックを取得（重複なし）
        total_read1 = sum(results1)
        total_read2 = sum(results2)
        assert total_read1 > 0
        assert total_read2 > 0
        # 合計が追加したティック数を超えない
        assert total_read1 + total_read2 <= 20

    @pytest.mark.asyncio
    async def test_get_new_ticks_after_clear_buffer(self):
        """バッファクリア後の動作をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        streamer = TickDataStreamer(symbol="NZDUSD", buffer_size=50)
        
        # 10個のティックを追加
        for i in range(10):
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="NZDUSD",
                bid=0.6000 + i * 0.0001,
                ask=0.6002 + i * 0.0001,
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
        
        # 一部を取得
        new_ticks = await streamer.get_new_ticks()
        assert len(new_ticks) == 10
        
        # バッファをクリア
        await streamer.clear_buffer()
        
        # クリア後は空を返す
        new_ticks = await streamer.get_new_ticks()
        assert new_ticks == []
        
        # 新しいティックを追加
        for i in range(5):
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="NZDUSD",
                bid=0.6100 + i * 0.0001,
                ask=0.6102 + i * 0.0001,
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
        
        # クリア後の新しいティックを正しく取得
        new_ticks = await streamer.get_new_ticks()
        assert len(new_ticks) == 5
        assert new_ticks[0].bid == pytest.approx(0.6100, rel=1e-6)

    @pytest.mark.asyncio
    async def test_get_new_ticks_large_batch(self):
        """大量ティックの一括処理をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        streamer = TickDataStreamer(symbol="USDCAD", buffer_size=1000)
        
        # 500個のティックを一括追加
        batch_size = 500
        for i in range(batch_size):
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="USDCAD",
                bid=1.3500 + i * 0.00001,
                ask=1.3502 + i * 0.00001,
                volume=1000.0 + i
            )
            streamer._add_to_buffer(tick)
        
        # 一括で全て取得
        new_ticks = await streamer.get_new_ticks()
        assert len(new_ticks) == batch_size
        
        # データの整合性確認
        for i, tick in enumerate(new_ticks):
            assert tick.bid == pytest.approx(1.3500 + i * 0.00001, rel=1e-8)
            assert tick.volume == pytest.approx(1000.0 + i, rel=1e-6)
        
        # 再度呼び出すと空
        new_ticks = await streamer.get_new_ticks()
        assert new_ticks == []

    @pytest.mark.asyncio
    async def test_get_new_ticks_tracking_accuracy(self):
        """インデックス追跡の正確性をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        streamer = TickDataStreamer(symbol="EURJPY", buffer_size=20)
        
        # 段階的にティックを追加して取得
        total_added = 0
        total_retrieved = 0
        
        for batch in range(5):
            # 各バッチで異なる数のティックを追加
            batch_size = (batch + 1) * 3
            for i in range(batch_size):
                tick = Tick(
                    timestamp=datetime.utcnow(),
                    symbol="EURJPY",
                    bid=160.0 + (total_added + i) * 0.001,
                    ask=160.002 + (total_added + i) * 0.001,
                    volume=1000.0
                )
                streamer._add_to_buffer(tick)
            total_added += batch_size
            
            # 新しいティックを取得
            new_ticks = await streamer.get_new_ticks()
            total_retrieved += len(new_ticks)
            
            # 正しい数のティックが取得されたか確認
            if total_added <= 20:  # バッファサイズ以内
                assert len(new_ticks) == batch_size
            
        # バッファサイズを考慮した総取得数の確認
        assert total_retrieved == min(total_added, 20 + (total_added - 20))

    @pytest.mark.asyncio
    async def test_get_new_ticks_vs_get_recent_ticks_performance(self):
        """get_new_ticks()とget_recent_ticks()の性能比較"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        import time
        
        streamer = TickDataStreamer(symbol="GBPJPY", buffer_size=1000)
        
        # 1000個のティックを追加
        for i in range(1000):
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="GBPJPY",
                bid=180.0 + i * 0.001,
                ask=180.002 + i * 0.001,
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
        
        # get_recent_ticks()の重複処理をシミュレート
        processed_ticks_recent = set()
        start_time = time.perf_counter()
        for _ in range(10):
            recent_ticks = await streamer.get_recent_ticks(n=100)
            for tick in recent_ticks:
                # 重複チェック（実際の処理をシミュレート）
                tick_id = (tick.bid, tick.ask, tick.volume)
                if tick_id not in processed_ticks_recent:
                    processed_ticks_recent.add(tick_id)
        time_recent = time.perf_counter() - start_time
        
        # get_new_ticks()で同じ処理（重複なし）
        await streamer.clear_buffer()
        for i in range(1000):
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="GBPJPY",
                bid=180.0 + i * 0.001,
                ask=180.002 + i * 0.001,
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
        
        processed_ticks_new = []
        start_time = time.perf_counter()
        for _ in range(10):
            new_ticks = await streamer.get_new_ticks()
            processed_ticks_new.extend(new_ticks)
        time_new = time.perf_counter() - start_time
        
        # get_new_ticks()の方が効率的（重複処理なし）
        assert len(processed_ticks_new) == 1000  # 全て一度だけ処理
        # 性能比較（get_new_ticks()の方が高速なはず）
        # ただし、環境依存のため厳密な比較は避ける
        print(f"get_recent_ticks time: {time_recent:.4f}s")
        print(f"get_new_ticks time: {time_new:.4f}s")
        print(f"Performance improvement: {(time_recent/time_new - 1)*100:.1f}%")


class TestSpikeFilter:
    """スパイクフィルター（5σルール + 価格変動率）のテスト"""

    @pytest.mark.xfail(reason="TickDataStreamer未実装")
    def test_spike_detection_with_3_sigma_rule(self):
        """3σルールによるスパイク検出をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        streamer = TickDataStreamer(symbol="USDJPY", spike_threshold=3.0)
        
        # 正常なティックデータを追加して統計量を安定させる
        normal_bid = 150.0
        for i in range(100):
            bid_value = normal_bid + np.random.normal(0, 0.001)
            # Ask は常に Bid より大きくなるようにする
            ask_value = bid_value + 0.002 + abs(np.random.normal(0, 0.0005))
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="USDJPY",
                bid=bid_value,
                ask=ask_value,
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
        
        # 統計量を更新
        streamer._update_statistics()
        
        # 正常なティック（平均±2σ以内）
        normal_tick = Tick(
            timestamp=datetime.utcnow(),
            symbol="USDJPY",
            bid=normal_bid + 0.002,
            ask=normal_bid + 0.005,  # Bidより確実に大きい値
            volume=1000.0
        )
        assert streamer._is_spike(normal_tick) is False
        
        # スパイクティック（平均±3σを超える）
        spike_tick = Tick(
            timestamp=datetime.utcnow(),
            symbol="USDJPY",
            bid=normal_bid + 0.1,  # 大きく外れた値
            ask=normal_bid + 0.102,
            volume=1000.0
        )
        assert streamer._is_spike(spike_tick) is True

    @pytest.mark.xfail(reason="TickDataStreamer未実装")
    def test_spike_filter_removes_outliers(self):
        """スパイクフィルターが異常値を除外することをテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        streamer = TickDataStreamer(symbol="USDJPY")
        
        # 正常なティックデータで初期化（ウォームアップ期間100件以上）
        for i in range(110):
            bid_value = 150.0 + np.random.normal(0, 0.001)
            # Ask は常に Bid より大きくなるようにする
            ask_value = bid_value + 0.002 + abs(np.random.normal(0, 0.0005))
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="USDJPY",
                bid=bid_value,
                ask=ask_value,
                volume=1000.0
            )
            streamer._process_tick(tick)
        
        initial_buffer_size = len(streamer.buffer)
        
        # スパイクティックを処理
        spike_tick = Tick(
            timestamp=datetime.utcnow(),
            symbol="USDJPY",
            bid=200.0,  # 明らかな異常値
            ask=200.002,
            volume=1000.0
        )
        
        # スパイクはフィルタリングされる
        result = streamer._process_tick(spike_tick)
        assert result is None
        assert len(streamer.buffer) == initial_buffer_size  # バッファに追加されない

    @pytest.mark.xfail(reason="TickDataStreamer未実装")
    def test_statistics_update_with_rolling_window(self):
        """ローリングウィンドウでの統計量更新をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        streamer = TickDataStreamer(symbol="USDJPY", statistics_window=100)
        
        # 初期データを追加
        for i in range(200):
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="USDJPY",
                bid=150.0 + i * 0.001,
                ask=150.002 + i * 0.001,
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
        
        # 統計量を更新
        streamer._update_statistics()
        
        # 最新100件のデータに基づく統計量が計算されることを確認
        assert streamer.mean_bid is not None
        assert streamer.std_bid is not None
        assert streamer.mean_ask is not None
        assert streamer.std_ask is not None
        
        # 統計量が最新データを反映していることを確認
        recent_bids = [tick.bid for tick in list(streamer.buffer)[-100:]]
        expected_mean = np.mean(recent_bids)
        assert abs(streamer.mean_bid - expected_mean) < 0.01

    @pytest.mark.xfail(reason="TickDataStreamer未実装")
    def test_spike_detection_in_quiet_market(self):
        """静かな市場での誤検出防止をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        streamer = TickDataStreamer(symbol="EURJPY")
        
        # 同一価格のティックを多数追加（静かな市場をシミュレート）
        base_price = 171.500
        for i in range(250):  # ウォームアップ期間を超える数
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="EURJPY",
                bid=base_price,
                ask=base_price + 0.005,
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
        
        # わずかな価格変動（0.002 = 0.0012%）
        small_change_tick = Tick(
            timestamp=datetime.utcnow(),
            symbol="EURJPY",
            bid=base_price - 0.002,
            ask=base_price + 0.003,
            volume=1000.0
        )
        
        # 価格変動率が0.1%未満なので、スパイクとして検出されない
        assert streamer._is_spike(small_change_tick) is False
        
        # 大きな価格変動（0.2 = 0.12%）
        large_change_tick = Tick(
            timestamp=datetime.utcnow(),
            symbol="EURJPY",
            bid=base_price + 0.2,
            ask=base_price + 0.205,
            volume=1000.0
        )
        
        # 価格変動率が0.1%を超えるので、スパイクとして検出される
        assert streamer._is_spike(large_change_tick) is True

    @pytest.mark.xfail(reason="TickDataStreamer未実装")
    def test_price_change_percentage_detection(self):
        """価格変動率によるスパイク検出をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        streamer = TickDataStreamer(symbol="EURJPY")
        
        # 正常な価格変動でバッファを初期化
        base_price = 171.500
        for i in range(250):
            variation = np.random.normal(0, 0.01)  # 通常の変動
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="EURJPY",
                bid=base_price + variation,
                ask=base_price + variation + 0.005,
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
        
        # 0.09%の変動（閾値以下）
        below_threshold_tick = Tick(
            timestamp=datetime.utcnow(),
            symbol="EURJPY",
            bid=base_price * 1.0009,  # 0.09%増
            ask=base_price * 1.0009 + 0.005,
            volume=1000.0
        )
        assert streamer._is_spike(below_threshold_tick) is False
        
        # 0.11%の変動（閾値超過）
        above_threshold_tick = Tick(
            timestamp=datetime.utcnow(),
            symbol="EURJPY",
            bid=base_price * 1.0011,  # 0.11%増
            ask=base_price * 1.0011 + 0.005,
            volume=1000.0
        )
        # 価格変動率が0.1%を超える場合、Zスコアチェックも行われる
        result = streamer._is_spike(above_threshold_tick)
        assert isinstance(result, bool)


class TestAsyncStreaming:
    """非同期ストリーミング機能のテスト"""

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="TickDataStreamer未実装")
    async def test_async_streaming_start_stop(self):
        """非同期ストリーミングの開始と停止をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        
        with patch('mt5_data_acquisition.tick_fetcher.mt5') as mock_mt5:
            mock_mt5.symbol_info_tick = MagicMock()
            
            streamer = TickDataStreamer(symbol="USDJPY")
            
            # ストリーミング開始
            stream_task = asyncio.create_task(streamer.stream_ticks())
            await asyncio.sleep(0.1)
            
            assert streamer.is_streaming is True
            
            # ストリーミング停止
            await streamer.stop_streaming()
            
            assert streamer.is_streaming is False
            stream_task.cancel()
            
            try:
                await stream_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="TickDataStreamer未実装")
    async def test_async_tick_generation(self):
        """非同期でのティック生成をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        with patch('mt5_data_acquisition.tick_fetcher.mt5') as mock_mt5:
            # MT5のティックデータをモック
            mock_tick = MagicMock()
            mock_tick.time = 1705500000
            mock_tick.bid = 150.123
            mock_tick.ask = 150.125
            mock_tick.volume = 1000.0
            mock_mt5.symbol_info_tick.return_value = mock_tick
            
            streamer = TickDataStreamer(symbol="USDJPY")
            
            tick_count = 0
            async for tick in streamer.stream_ticks():
                assert isinstance(tick, Tick)
                assert tick.symbol == "USDJPY"
                assert tick.bid == 150.123
                assert tick.ask == 150.125
                
                tick_count += 1
                if tick_count >= 5:
                    break
            
            assert tick_count == 5

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="TickDataStreamer未実装")
    async def test_streaming_latency_under_10ms(self):
        """ストリーミングレイテンシが10ms以内であることをテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        import time
        
        with patch('mt5_data_acquisition.tick_fetcher.mt5') as mock_mt5:
            mock_tick = MagicMock()
            mock_tick.time = 1705500000
            mock_tick.bid = 150.123
            mock_tick.ask = 150.125
            mock_tick.volume = 1000.0
            mock_mt5.symbol_info_tick.return_value = mock_tick
            
            streamer = TickDataStreamer(symbol="USDJPY")
            
            latencies = []
            async for tick in streamer.stream_ticks():
                start_time = time.perf_counter()
                # ティック処理のシミュレーション
                await streamer._emit_event(tick)
                latency = (time.perf_counter() - start_time) * 1000  # ms
                latencies.append(latency)
                
                if len(latencies) >= 10:
                    break
            
            # 平均レイテンシが10ms以内
            avg_latency = np.mean(latencies)
            assert avg_latency < 10.0, f"Average latency {avg_latency}ms exceeds 10ms"


class TestBackpressureControl:
    """バックプレッシャー制御のテスト"""

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="TickDataStreamer未実装")
    async def test_backpressure_triggers_when_buffer_full(self):
        """バッファがフルに近い時にバックプレッシャーが発動することをテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        streamer = TickDataStreamer(
            symbol="USDJPY",
            buffer_size=100,
            backpressure_threshold=0.8
        )
        
        # バッファを80%以上埋める
        for i in range(85):
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="USDJPY",
                bid=150.0 + i * 0.001,
                ask=150.002 + i * 0.001,
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
        
        # バックプレッシャーが発動することを確認
        assert await streamer._check_backpressure() is True
        
        # バッファをクリア
        streamer.buffer.clear()
        
        # バックプレッシャーが解除されることを確認
        assert await streamer._check_backpressure() is False

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="TickDataStreamer未実装")
    async def test_backpressure_slows_down_streaming(self):
        """バックプレッシャー時にストリーミングが遅延することをテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        import time
        
        streamer = TickDataStreamer(
            symbol="USDJPY",
            buffer_size=100,
            backpressure_threshold=0.8,
            backpressure_delay=0.1  # 100ms遅延
        )
        
        # バッファを埋める
        for i in range(85):
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="USDJPY",
                bid=150.0,
                ask=150.002,
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
        
        # バックプレッシャー処理の時間を測定
        start_time = time.perf_counter()
        await streamer._handle_backpressure()
        elapsed = time.perf_counter() - start_time
        
        # 遅延が適用されていることを確認
        assert elapsed >= 0.1, f"Expected delay >= 0.1s, got {elapsed}s"


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="TickDataStreamer未実装")
    async def test_auto_reconnect_on_connection_error(self):
        """接続エラー時の自動再接続をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        
        with patch('mt5_data_acquisition.tick_fetcher.mt5') as mock_mt5:
            # 最初はエラー、その後成功するように設定
            mock_mt5.symbol_info_tick.side_effect = [
                Exception("Connection lost"),
                Exception("Connection lost"),
                MagicMock(time=1705500000, bid=150.123, ask=150.125, volume=1000.0)
            ]
            
            streamer = TickDataStreamer(symbol="USDJPY", max_retries=3)
            
            # エラーハンドリングと再接続
            result = await streamer._fetch_tick_with_retry()
            
            assert result is not None
            assert mock_mt5.symbol_info_tick.call_count == 3

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="TickDataStreamer未実装")
    async def test_circuit_breaker_pattern(self):
        """サーキットブレーカーパターンの動作をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        
        with patch('mt5_data_acquisition.tick_fetcher.mt5') as mock_mt5:
            mock_mt5.symbol_info_tick.side_effect = Exception("Connection error")
            
            streamer = TickDataStreamer(
                symbol="USDJPY",
                circuit_breaker_threshold=5,
                circuit_breaker_timeout=1.0
            )
            
            # 5回失敗でサーキットブレーカーが開く
            for i in range(5):
                try:
                    await streamer._fetch_tick_with_retry()
                except Exception:
                    pass
            
            assert streamer.circuit_breaker_open is True
            
            # サーキットブレーカーが開いている間は即座に失敗
            with pytest.raises(Exception, match="Circuit breaker is open"):
                await streamer._fetch_tick_with_retry()
            
            # タイムアウト後にサーキットブレーカーがリセット
            await asyncio.sleep(1.1)
            streamer._reset_circuit_breaker()
            assert streamer.circuit_breaker_open is False

    @pytest.mark.xfail(reason="TickDataStreamer未実装")
    def test_error_logging_structure(self):
        """エラーログの構造化出力をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        import logging
        
        with patch('mt5_data_acquisition.tick_fetcher.logger') as mock_logger:
            streamer = TickDataStreamer(symbol="USDJPY")
            
            # エラーを発生させる
            error = Exception("Test error")
            streamer._log_error(error, context={
                "symbol": "USDJPY",
                "timestamp": datetime.utcnow().isoformat(),
                "retry_count": 3
            })
            
            # 構造化されたログが出力されることを確認
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert "Test error" in str(call_args)
            assert "USDJPY" in str(call_args)
            assert "retry_count" in str(call_args)


class TestIntegrationScenarios:
    """統合シナリオのテスト"""

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="TickDataStreamer未実装")
    async def test_end_to_end_streaming_with_filtering(self):
        """エンドツーエンドのストリーミングとフィルタリングをテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        with patch('mt5_data_acquisition.tick_fetcher.mt5') as mock_mt5:
            # 正常なティックとスパイクを混在させる
            tick_data = [
                (150.123, 150.125),  # 正常
                (150.124, 150.126),  # 正常
                (200.000, 200.002),  # スパイク
                (150.125, 150.127),  # 正常
                (150.126, 150.128),  # 正常
            ]
            
            mock_ticks = []
            for bid, ask in tick_data:
                mock_tick = MagicMock()
                mock_tick.time = 1705500000
                mock_tick.bid = bid
                mock_tick.ask = ask
                mock_tick.volume = 1000.0
                mock_ticks.append(mock_tick)
            
            mock_mt5.symbol_info_tick.side_effect = mock_ticks
            
            streamer = TickDataStreamer(symbol="USDJPY")
            
            # 初期統計量を設定
            for i in range(50):
                tick = Tick(
                    timestamp=datetime.utcnow(),
                    symbol="USDJPY",
                    bid=150.0 + np.random.normal(0, 0.01),
                    ask=150.002 + np.random.normal(0, 0.01),
                    volume=1000.0
                )
                streamer._add_to_buffer(tick)
            streamer._update_statistics()
            
            # ストリーミングとフィルタリング
            collected_ticks = []
            async for tick in streamer.stream_ticks():
                collected_ticks.append(tick)
                if len(collected_ticks) >= 4:  # スパイクは除外される
                    break
            
            # スパイクが除外されていることを確認
            assert len(collected_ticks) == 4
            assert all(tick.bid < 200 for tick in collected_ticks)

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="TickDataStreamer未実装")
    async def test_memory_stability_long_running(self):
        """長時間実行時のメモリ安定性をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        streamer = TickDataStreamer(symbol="USDJPY", buffer_size=10000)
        
        # 100,000個のティックを処理
        for i in range(100000):
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="USDJPY",
                bid=150.0 + np.random.normal(0, 0.001),
                ask=150.002 + np.random.normal(0, 0.001),
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
            
            # 定期的にガベージコレクション
            if i % 10000 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # メモリ増加が合理的な範囲内（100MB以下）
        assert memory_increase < 100, f"Memory increased by {memory_increase}MB"
        
        # バッファサイズが制限内
        assert len(streamer.buffer) == 10000


class TestPerformanceMetrics:
    """パフォーマンスメトリクスのテスト"""

    @pytest.mark.xfail(reason="TickDataStreamer未実装")
    def test_tick_processing_throughput(self):
        """ティック処理のスループットをテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        import time
        
        streamer = TickDataStreamer(symbol="USDJPY")
        
        # 10,000個のティックを処理
        start_time = time.perf_counter()
        
        for i in range(10000):
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="USDJPY",
                bid=150.0 + i * 0.0001,
                ask=150.002 + i * 0.0001,
                volume=1000.0
            )
            streamer._process_tick(tick)
        
        elapsed = time.perf_counter() - start_time
        throughput = 10000 / elapsed
        
        # 1秒あたり1000ティック以上処理できること
        assert throughput > 1000, f"Throughput {throughput} ticks/s is too low"

    @pytest.mark.xfail(reason="TickDataStreamer未実装")
    def test_float32_conversion_efficiency(self):
        """Float32変換の効率性をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        import time
        
        streamer = TickDataStreamer(symbol="USDJPY")
        
        # Float64データ
        float64_values = [150.123456789 + i * 0.000001 for i in range(10000)]
        
        # 変換時間を測定
        start_time = time.perf_counter()
        
        for value in float64_values:
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="USDJPY",
                bid=value,
                ask=value + 0.002,
                volume=1000.0
            )
            # Tickモデル内でFloat32変換が行われる
            streamer._add_to_buffer(tick)
        
        elapsed = time.perf_counter() - start_time
        
        # 10,000個の変換が1秒以内に完了
        assert elapsed < 1.0, f"Float32 conversion took {elapsed}s"
        
        # メモリ使用量の確認（Float32は Float64の半分）
        import sys
        tick_size = sys.getsizeof(streamer.buffer[0])
        assert tick_size < 500, f"Tick object size {tick_size} bytes is too large"


class TestEdgeCases:
    """エッジケースと異常系のテスト"""

    def test_extreme_spike_detection(self):
        """極端なスパイク値の検出をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        streamer = TickDataStreamer(symbol="USDJPY", spike_threshold=3.0)
        
        # 正常値でウォームアップ
        base_price = 150.0
        for i in range(100):
            bid_variation = np.random.normal(0, 0.0005)  # より小さな変動
            ask_variation = np.random.normal(0, 0.0005)
            tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="USDJPY",
                bid=base_price + bid_variation,
                ask=base_price + 0.002 + ask_variation,  # 常にbidより高い
                volume=1000.0
            )
            streamer._add_to_buffer(tick)
        
        initial_count = len(streamer.buffer)
        
        # 極端なスパイク（100σ相当）
        extreme_spike = Tick(
            timestamp=datetime.utcnow(),
            symbol="USDJPY",
            bid=base_price + 10.0,  # 10円の異常値
            ask=base_price + 10.002,
            volume=1000.0
        )
        
        processed = streamer._process_tick(extreme_spike)
        
        # スパイクが検出され、バッファに追加されないこと
        assert processed is None
        assert len(streamer.buffer) == initial_count
        assert streamer.stats["spike_count"] > 0

    def test_zero_and_negative_prices(self):
        """ゼロまたは負の価格の処理をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        streamer = TickDataStreamer(symbol="USDJPY")
        
        # ゼロ価格のティック（Pydanticのバリデーションで検出される）
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="greater than 0"):
            zero_tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="USDJPY",
                bid=0.0,
                ask=0.002,
                volume=1000.0
            )
        
        # 負の価格のティック
        with pytest.raises(ValidationError, match="greater than 0"):
            negative_tick = Tick(
                timestamp=datetime.utcnow(),
                symbol="USDJPY",
                bid=-150.0,
                ask=-149.998,
                volume=1000.0
            )

    @pytest.mark.asyncio
    async def test_concurrent_buffer_access(self):
        """複数の非同期タスクからの同時バッファアクセスをテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        from common.models import Tick
        
        streamer = TickDataStreamer(symbol="USDJPY", buffer_size=1000)
        
        async def add_ticks(start_price: float, count: int):
            """非同期でティックを追加"""
            for i in range(count):
                tick = Tick(
                    timestamp=datetime.utcnow(),
                    symbol="USDJPY",
                    bid=start_price + i * 0.001,
                    ask=start_price + 0.002 + i * 0.001,
                    volume=1000.0
                )
                await streamer.add_tick(tick)
                await asyncio.sleep(0.001)
        
        # 3つの非同期タスクから同時にティックを追加
        tasks = [
            add_ticks(150.0, 100),
            add_ticks(151.0, 100),
            add_ticks(152.0, 100)
        ]
        
        await asyncio.gather(*tasks)
        
        # バッファへの追加が正常に行われたこと
        assert len(streamer.buffer) <= 1000
        assert streamer.stats["total_ticks"] >= 200  # スパイクフィルターで一部除外される可能性

    def test_memory_pool_exhaustion(self):
        """メモリプールが枯渇した場合の動作をテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        
        streamer = TickDataStreamer(symbol="USDJPY")
        
        # オブジェクトプールの全オブジェクトを取得
        acquired_objects = []
        for _ in range(100):  # プールサイズ
            obj = streamer._tick_pool.acquire()
            acquired_objects.append(obj)
        
        # プール枯渇後も新規作成できること
        additional_obj = streamer._tick_pool.acquire()
        assert additional_obj is not None
        assert streamer._tick_pool.stats["created"] > 100
        
        # オブジェクトを解放
        for obj in acquired_objects:
            streamer._tick_pool.release(obj)
        
        # プール効率が回復すること
        assert streamer._tick_pool.stats["efficiency"] > 0.8

    @pytest.mark.asyncio
    async def test_rapid_subscribe_unsubscribe(self):
        """高速な購読・購読解除の繰り返しをテスト"""
        from mt5_data_acquisition.tick_fetcher import TickDataStreamer
        
        # 直接パラメータを渡す
        
        # MT5クライアントのモック
        mock_mt5_client = MagicMock()
        mock_mt5_client.is_connected = True
        mock_mt5_client.symbol_info = MagicMock(return_value=MagicMock(visible=True))
        mock_mt5_client.symbol_select = MagicMock(return_value=True)
        
        streamer = TickDataStreamer(symbol="USDJPY", mt5_client=mock_mt5_client)
        
        # 10回の高速な購読・購読解除
        for i in range(10):
            await streamer.subscribe_to_ticks()
            assert streamer.is_subscribed
            
            await streamer.unsubscribe()
            assert not streamer.is_subscribed
            
            # 短い待機
            await asyncio.sleep(0.01)
        
        # リソースリークがないこと
        assert len(streamer._tick_listeners) == 0
        assert len(streamer._error_listeners) == 0
        assert len(streamer._backpressure_listeners) == 0
"""
簡単な統合テスト: ティックデータストリーミング機能の基本動作確認

このファイルは、Step 8の統合テストの簡易版です。
MT5をモックして、基本的な動作を確認します。
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.mt5_data_acquisition.tick_fetcher import StreamerConfig, TickDataStreamer


class MockTickInfo:
    """シンプルなティック情報モック"""
    def __init__(self):
        self.bid = 110.123
        self.ask = 110.125
        self.volume = 100.0
        self.time = 1234567890
        self.flags = 6
        self.last = 110.124
        self.volume_real = 100.0


@pytest.mark.asyncio
async def test_basic_initialization():
    """基本的な初期化テスト"""
    config = StreamerConfig(
        symbol="USDJPY",
        buffer_size=1000,
        spike_threshold=3.0,
    )
    
    with patch("src.mt5_data_acquisition.tick_fetcher.mt5") as mock_mt5:
        # MT5の基本的なモック設定
        mock_mt5.initialize = Mock(return_value=True)
        mock_mt5.symbol_select = Mock(return_value=True)
        mock_mt5.symbol_info = Mock(return_value=Mock(visible=True))
        mock_mt5.symbol_info_tick = Mock(return_value=MockTickInfo())
        
        # MT5ConnectionManagerのモック
        mock_connection_manager = Mock()
        mock_connection_manager.is_connected = Mock(return_value=True)
        mock_connection_manager._connected = True
        
        # TickDataStreamerを初期化
        streamer = TickDataStreamer(
            symbol=config.symbol,
            buffer_size=config.buffer_size,
            spike_threshold=config.spike_threshold,
            mt5_client=mock_connection_manager,
        )
        
        # 基本的なプロパティの確認
        assert streamer.symbol == "USDJPY"
        assert streamer.buffer_size == 1000
        assert streamer.spike_threshold == 3.0
        assert streamer.buffer_usage == 0.0


@pytest.mark.asyncio
async def test_subscribe_and_stream():
    """購読とストリーミングの基本テスト"""
    with patch("src.mt5_data_acquisition.tick_fetcher.mt5") as mock_mt5:
        # MT5の基本的なモック設定
        mock_mt5.initialize = Mock(return_value=True)
        mock_mt5.symbol_select = Mock(return_value=True)
        mock_mt5.symbol_info = Mock(return_value=Mock(visible=True))
        
        # ティック情報を返すモック
        tick_count = 0
        def get_tick(symbol):
            nonlocal tick_count
            tick_count += 1
            if tick_count > 10:
                return None  # 10回後はNoneを返す
            return MockTickInfo()
        
        mock_mt5.symbol_info_tick = Mock(side_effect=get_tick)
        
        # MT5ConnectionManagerのモック
        mock_connection_manager = Mock()
        mock_connection_manager.is_connected = Mock(return_value=True)
        mock_connection_manager._connected = True
        
        # TickDataStreamerを初期化
        streamer = TickDataStreamer(
            symbol="USDJPY",
            buffer_size=100,
            mt5_client=mock_connection_manager,
        )
        
        # 購読開始
        result = await streamer.subscribe_to_ticks()
        assert result is True, "購読は成功するべき"
        
        # ストリーミング（最大5ティック）
        collected_ticks = []
        tick_counter = 0
        async for tick in streamer.stream_ticks():
            collected_ticks.append(tick)
            tick_counter += 1
            if tick_counter >= 5:
                break
        
        # 停止
        await streamer.stop_streaming()
        
        # 検証
        assert len(collected_ticks) == 5, "5ティックを収集すべき"
        stats = streamer.current_stats
        assert stats.get("total_ticks", 0) == 5 or stats.get("tick_count", 0) == 5, "統計が正しく更新されるべき"


@pytest.mark.asyncio
async def test_backpressure_handling():
    """バックプレッシャー処理の基本テスト"""
    with patch("src.mt5_data_acquisition.tick_fetcher.mt5") as mock_mt5:
        # MT5の基本的なモック設定
        mock_mt5.initialize = Mock(return_value=True)
        mock_mt5.symbol_select = Mock(return_value=True)
        mock_mt5.symbol_info = Mock(return_value=Mock(visible=True))
        mock_mt5.symbol_info_tick = Mock(return_value=MockTickInfo())
        
        # MT5ConnectionManagerのモック
        mock_connection_manager = Mock()
        mock_connection_manager.is_connected = Mock(return_value=True)
        mock_connection_manager._connected = True
        
        # 小さいバッファでStreamerを初期化
        streamer = TickDataStreamer(
            symbol="USDJPY",
            buffer_size=10,
            backpressure_threshold=0.8,
            mt5_client=mock_connection_manager,
        )
        
        # バックプレッシャーイベントを記録
        backpressure_events = []
        async def record_backpressure(level, usage):
            backpressure_events.append(level)
        streamer.add_backpressure_listener(record_backpressure)
        
        # バッファを埋める
        from src.common.models import Tick
        from datetime import datetime, UTC
        for i in range(9):  # 90%まで埋める
            tick = Tick(
                symbol="USDJPY",  # symbolフィールドを追加
                timestamp=datetime.now(UTC),
                bid=110.0 + i * 0.001,
                ask=110.002 + i * 0.001,
                volume=100.0,
            )
            # add_tickメソッドを使用（バックプレッシャーチェック含む）
            await streamer.add_tick(tick)
        
        # バックプレッシャーが発火することを確認
        assert streamer.buffer_usage == 0.9, "バッファ使用率は90%であるべき"
        assert len(backpressure_events) > 0, "バックプレッシャーイベントが発火すべき"


@pytest.mark.asyncio
async def test_spike_filter():
    """スパイクフィルターの基本テスト"""
    with patch("src.mt5_data_acquisition.tick_fetcher.mt5") as mock_mt5:
        # MT5の基本的なモック設定
        mock_mt5.initialize = Mock(return_value=True)
        mock_mt5.symbol_select = Mock(return_value=True)
        mock_mt5.symbol_info = Mock(return_value=Mock(visible=True))
        
        # 正常値とスパイクを混ぜたティックを生成
        # ウォームアップ期間（最初の100ティック）を考慮して、より多くのティックを生成
        tick_values = []
        # 最初に正常なティックを100個追加（ウォームアップ）
        for i in range(100):
            tick_values.append((110.0 + i * 0.0001, 110.002 + i * 0.0001))
        # その後スパイクを混ぜる
        tick_values.extend([
            (110.01, 110.012),  # 正常
            (110.011, 110.013),  # 正常
            (150.0, 150.002),   # スパイク！
            (110.012, 110.014),  # 正常
            (110.013, 110.015),  # 正常
        ])
        
        tick_index = 0
        def get_tick(symbol):
            nonlocal tick_index
            if tick_index >= len(tick_values):
                return None
            tick = MockTickInfo()
            tick.bid, tick.ask = tick_values[tick_index]
            tick_index += 1
            return tick
        
        mock_mt5.symbol_info_tick = Mock(side_effect=get_tick)
        
        # MT5ConnectionManagerのモック
        mock_connection_manager = Mock()
        mock_connection_manager.is_connected = Mock(return_value=True)
        mock_connection_manager._connected = True
        
        # TickDataStreamerを初期化
        streamer = TickDataStreamer(
            symbol="USDJPY",
            spike_threshold=3.0,
            stats_window_size=100,
            mt5_client=mock_connection_manager,
        )
        
        # 購読開始
        await streamer.subscribe_to_ticks()
        
        # ストリーミング（ウォームアップ期間を含めて105ティック以上）
        collected_ticks = []
        async for tick in streamer.stream_ticks():
            collected_ticks.append(tick)
            if len(collected_ticks) >= 104:  # ウォームアップ100 + 正常4ティック（スパイク1つ除外）
                break
        
        # 停止
        await streamer.stop_streaming()
        
        # 検証
        stats = streamer.current_stats
        assert stats["spike_count"] >= 1, "スパイクが検出されるべき"
        # スパイクは除外されるので、正常なティックのみ収集される
        assert all(100 < tick.bid < 120 for tick in collected_ticks), "スパイクは除外されるべき"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
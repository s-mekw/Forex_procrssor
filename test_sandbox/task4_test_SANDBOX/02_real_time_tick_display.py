"""
リアルタイムティック表示 - ライブでティックデータを受信・表示
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
from datetime import datetime
from collections import deque
from typing import Dict, Any, Optional
import numpy as np
from rich.live import Live
from rich.console import Console

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer, StreamerConfig
from src.common.models import Tick
from utils.test_config import get_mt5_config, TEST_SYMBOLS, create_tick_streamer, STREAMING_CONFIG, DISPLAY_CONFIG
from utils.display_helpers import (
    print_success, print_error, print_warning, print_info,
    print_section, create_live_display, format_price, format_timestamp,
    get_price_color
)

console = Console()

class TickDisplayManager:
    """ティック表示マネージャー"""
    
    def __init__(self, symbol: str, max_ticks: int = 100):
        self.symbol = symbol
        self.max_ticks = max_ticks
        self.ticks = deque(maxlen=max_ticks)
        self.stats = {
            "total_ticks": 0,
            "spikes_detected": 0,
            "mean_bid": 0.0,
            "mean_ask": 0.0,
            "std_bid": 0.0,
            "std_ask": 0.0,
            "buffer_usage": 0.0,
            "buffer_size": STREAMING_CONFIG["buffer_size"],
            "dropped_ticks": 0,
            "last_update": datetime.now()
        }
        self.connection_info = {
            "is_connected": False,
            "symbol": symbol,
            "server": "Not connected"
        }
        self.errors = deque(maxlen=10)
        self.previous_tick: Optional[Dict] = None
        
    def add_tick(self, tick: Tick):
        """ティックを追加"""
        tick_dict = {
            "timestamp": tick.timestamp,
            "symbol": tick.symbol,
            "bid": tick.bid,
            "ask": tick.ask,
            "volume": tick.volume,
            "spread": tick.ask - tick.bid
        }
        
        # 価格変動を記録
        if self.previous_tick:
            tick_dict["bid_change"] = tick.bid - self.previous_tick["bid"]
            tick_dict["ask_change"] = tick.ask - self.previous_tick["ask"]
        
        self.ticks.append(tick_dict)
        self.previous_tick = tick_dict
        self.stats["total_ticks"] += 1
        self.stats["last_update"] = datetime.now()
        
    def update_stats(self, streamer_stats: Dict[str, Any]):
        """統計情報を更新"""
        self.stats.update(streamer_stats)
        
    def update_connection(self, is_connected: bool, server: str = ""):
        """接続情報を更新"""
        self.connection_info["is_connected"] = is_connected
        if server:
            self.connection_info["server"] = server
            
    def add_error(self, error_type: str, message: str):
        """エラーを追加"""
        self.errors.append({
            "timestamp": datetime.now(),
            "type": error_type,
            "message": message
        })
        
    def get_display_data(self) -> Dict:
        """表示用データを取得"""
        return {
            "ticks": list(self.ticks),
            "stats": self.stats,
            "connection_info": self.connection_info,
            "errors": list(self.errors)
        }

async def stream_and_display(symbol: str, duration: int = 60):
    """ティックをストリーミングして表示"""
    print_section(f"Real-time Tick Display - {symbol}")
    
    # マネージャーを初期化
    display_manager = TickDisplayManager(symbol)
    config = get_mt5_config()
    connection_manager = MT5ConnectionManager(config=config)
    
    # 接続
    print_info("Connecting to MT5...")
    if not connection_manager.connect(config):
        print_error("Failed to connect to MT5")
        return
    
    display_manager.update_connection(True, config["server"])
    print_success(f"Connected to {config['server']}")
    
    # ストリーマーを設定
    streamer = create_tick_streamer(symbol, connection_manager)
    
    # ティックリスナーを追加
    def on_tick(tick: Tick):
        display_manager.add_tick(tick)
        display_manager.update_stats(streamer.current_stats)
    
    def on_error(error: Exception):
        display_manager.add_error(
            error.__class__.__name__,
            str(error)
        )
    
    def on_backpressure(usage: float):
        display_manager.stats["buffer_usage"] = usage
    
    streamer.add_tick_listener(on_tick)
    streamer.add_error_listener(on_error)
    streamer.add_backpressure_listener(on_backpressure)
    
    # 購読開始
    print_info(f"Subscribing to {symbol}...")
    if not await streamer.subscribe_to_ticks():
        print_error(f"Failed to subscribe to {symbol}")
        connection_manager.disconnect()
        return
    
    print_success(f"Subscribed to {symbol}")
    print_info(f"Streaming for {duration} seconds... Press Ctrl+C to stop")
    
    # ストリーミング開始
    await streamer.start_streaming()
    streaming_task = streamer._current_task
    
    # ライブ表示を開始
    try:
        with Live(
            create_live_display(**display_manager.get_display_data()),
            refresh_per_second=2,
            screen=True
        ) as live:
            # 指定時間だけストリーミング
            end_time = asyncio.get_event_loop().time() + duration
            while asyncio.get_event_loop().time() < end_time:
                await asyncio.sleep(DISPLAY_CONFIG["refresh_rate"])
                
                # 表示を更新
                live.update(create_live_display(**display_manager.get_display_data()))
                
                # ストリーミングタスクの状態をチェック
                if streaming_task and streaming_task.done():
                    print_warning("Streaming task ended unexpectedly")
                    break
                    
    except KeyboardInterrupt:
        print_warning("\nStreaming interrupted by user")
    finally:
        # クリーンアップ
        await streamer.stop_streaming()
        await streamer.unsubscribe()
        connection_manager.disconnect()
        
        # 最終統計を表示
        print_section("Final Statistics")
        stats = display_manager.stats
        console.print(f"Total Ticks Received: {stats['total_ticks']}")
        console.print(f"Spikes Detected: {stats['spikes_detected']}")
        console.print(f"Dropped Ticks: {stats['dropped_ticks']}")
        if display_manager.errors:
            console.print(f"Total Errors: {len(display_manager.errors)}")

async def main():
    """メイン関数"""
    # テストパラメータ
    symbol = TEST_SYMBOLS["default"]
    duration = 12  # 秒
    
    try:
        await stream_and_display(symbol, duration)
        print_success("Test completed successfully")
    except Exception as e:
        print_error(f"Test failed: {e}")
        import traceback
        console.print(traceback.format_exc(), style="red")

if __name__ == "__main__":
    asyncio.run(main())
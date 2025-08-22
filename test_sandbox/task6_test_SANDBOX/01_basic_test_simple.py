"""
基本的なティック→バー変換テスト（シンプル版）
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Optional

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer
from src.mt5_data_acquisition.tick_to_bar import TickToBarConverter, Tick, Bar
from src.common.config import BaseConfig

class SimplTickToBarTest:
    """シンプルなティック→バー変換テスト"""
    
    def __init__(self, symbol: str = "EURUSD"):
        self.symbol = symbol
        self.converter = TickToBarConverter(
            symbol=symbol,
            timeframe=60,  # 1分バー
            gap_threshold=30,
            max_completed_bars=10
        )
        self.tick_count = 0
        self.bar_count = 0
        self.last_bar: Optional[Bar] = None
        
        # バー完成時のコールバックを設定
        self.converter.on_bar_complete = self.on_bar_complete
        
    def on_bar_complete(self, bar: Bar):
        """バー完成時のコールバック"""
        self.bar_count += 1
        self.last_bar = bar
        
        # バー情報を表示
        bar_time = bar.time.replace(tzinfo=None) if bar.time.tzinfo else bar.time
        print(f"\n✅ Bar #{self.bar_count} completed at {bar_time.strftime('%H:%M:%S')}")
        print(f"   OHLC: {float(bar.open):.5f} / {float(bar.high):.5f} / {float(bar.low):.5f} / {float(bar.close):.5f}")
        print(f"   Volume: {float(bar.volume):.2f}, Ticks: {bar.tick_count}")
        
    async def process_ticks(self, streamer: TickDataStreamer):
        """ティックを処理"""
        print("\n📊 Starting tick processing...")
        print("   Waiting for ticks...")
        
        processed_ticks = set()  # 処理済みティックを追跡
        
        while True:
            # 最新のティックを取得
            ticks = await streamer.get_recent_ticks(n=100)
            
            if ticks:
                for tick_data in ticks:
                    # ティックのユニークIDを生成（時刻とbid/askの組み合わせ）
                    tick_id = f"{tick_data.timestamp}_{tick_data.bid}_{tick_data.ask}"
                    
                    # すでに処理済みならスキップ
                    if tick_id in processed_ticks:
                        continue
                    
                    processed_ticks.add(tick_id)
                    
                    # メモリ管理: 古い処理済みティックIDを削除
                    if len(processed_ticks) > 10000:
                        # 古いものから削除（setは順序を保持しないので、新しいsetを作成）
                        processed_ticks = set(list(processed_ticks)[-5000:])
                    
                    try:
                        # tick_to_bar.Tickに変換
                        tick = Tick(
                            symbol=tick_data.symbol,
                            time=tick_data.timestamp,
                            bid=Decimal(str(tick_data.bid)),
                            ask=Decimal(str(tick_data.ask)),
                            volume=Decimal(str(tick_data.volume)) if tick_data.volume else Decimal("1.0")
                        )
                        
                        self.tick_count += 1
                        
                        # 10ティックごとに進捗表示
                        if self.tick_count % 10 == 0:
                            current_bar = self.converter.get_current_bar()
                            if current_bar:
                                print(f"\r   Processed {self.tick_count} ticks, {self.bar_count} bars completed. Current bar: {current_bar.tick_count} ticks", end="")
                        
                        # コンバーターに追加
                        self.converter.add_tick(tick)
                        
                    except Exception as e:
                        print(f"\n❌ Error processing tick: {e}")
            
            # CPU負荷を下げるため少し待機
            await asyncio.sleep(0.5)

async def main():
    """メイン関数"""
    print("=" * 60)
    print("Simple Tick to Bar Conversion Test")
    print("=" * 60)
    
    # 設定
    symbol = "EURUSD"
    print(f"\n⚙️ Settings:")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframe: 1 minute bars")
    print(f"   Press Ctrl+C to stop")
    
    # テストインスタンスを作成
    test = SimplTickToBarTest(symbol)
    
    try:
        # MT5設定
        config = BaseConfig()
        mt5_config = {
            "account": config.mt5_login,
            "password": config.mt5_password.get_secret_value() if config.mt5_password else None,
            "server": config.mt5_server,
            "timeout": config.mt5_timeout,
        }
        
        # MT5接続
        print("\n🔗 Connecting to MT5...")
        connection_manager = MT5ConnectionManager()
        
        if not connection_manager.connect(mt5_config):
            print("❌ Failed to connect to MT5")
            return
        
        print("✅ Connected to MT5")
        
        # ティックストリーマー作成
        streamer = TickDataStreamer(
            symbol=symbol,
            buffer_size=5000,  # バッファサイズを増やす
            spike_threshold_percent=0.1,
            backpressure_threshold=0.8,
            mt5_client=connection_manager
        )
        
        # ストリーミング開始
        await streamer.start_streaming()
        print("✅ Tick streaming started")
        
        # ティック処理
        await test.process_ticks(streamer)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Stopping test...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # クリーンアップ
        if 'streamer' in locals():
            await streamer.stop_streaming()
            print("✅ Streaming stopped")
            
        if 'connection_manager' in locals():
            connection_manager.disconnect()
            print("✅ Disconnected from MT5")
        
        # 最終統計を表示
        print("\n" + "=" * 60)
        print("Final Statistics")
        print("=" * 60)
        print(f"   Total ticks processed: {test.tick_count}")
        print(f"   Total bars completed: {test.bar_count}")
        
        if test.bar_count > 0:
            print(f"   Average ticks per bar: {test.tick_count / test.bar_count:.1f}")

if __name__ == "__main__":
    asyncio.run(main())
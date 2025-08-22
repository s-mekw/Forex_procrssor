"""
ティックデータのタイムゾーンを確認するテスト
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
from datetime import datetime
from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer
from src.common.config import BaseConfig

async def main():
    config = BaseConfig()
    mt5_config = {
        "account": config.mt5_login,
        "password": config.mt5_password.get_secret_value() if config.mt5_password else None,
        "server": config.mt5_server,
        "timeout": config.mt5_timeout,
    }
    
    connection_manager = MT5ConnectionManager()
    
    if not connection_manager.connect(mt5_config):
        print("Failed to connect")
        return
    
    print("Connected to MT5")
    
    streamer = TickDataStreamer(
        symbol="EURUSD",
        buffer_size=100,
        mt5_client=connection_manager
    )
    
    await streamer.start_streaming()
    print("Streaming started")
    
    # 少し待ってからティックを取得
    await asyncio.sleep(2)
    
    ticks = await streamer.get_recent_ticks(n=1)
    
    if ticks:
        tick = ticks[0]
        print(f"\nTick received:")
        print(f"  Type: {type(tick)}")
        
        # timestampフィールドを確認
        if hasattr(tick, 'timestamp'):
            print(f"  Timestamp: {tick.timestamp}")
            print(f"  Timestamp type: {type(tick.timestamp)}")
            print(f"  Timestamp tzinfo: {tick.timestamp.tzinfo}")
            
            # 現在時刻と比較
            now = datetime.now()
            print(f"\n  Local now: {now}")
            print(f"  Local now tzinfo: {now.tzinfo}")
            
            try:
                diff = now - tick.timestamp
                print(f"  Time difference: {diff}")
            except TypeError as e:
                print(f"  Cannot compare: {e}")
    
    await streamer.stop_streaming()
    connection_manager.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
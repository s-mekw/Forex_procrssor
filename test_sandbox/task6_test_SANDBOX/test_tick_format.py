"""
ティックデータ形式を確認するテスト
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
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
    
    ticks = await streamer.get_recent_ticks(n=3)
    
    print(f"\nGot {len(ticks)} ticks")
    
    for i, tick in enumerate(ticks):
        print(f"\nTick {i+1}:")
        print(f"  Type: {type(tick)}")
        print(f"  Attributes: {dir(tick)}")
        
        if hasattr(tick, 'symbol'):
            print(f"  Symbol: {tick.symbol}")
        if hasattr(tick, 'time'):
            print(f"  Time: {tick.time}")
        if hasattr(tick, 'bid'):
            print(f"  Bid: {tick.bid}")
        if hasattr(tick, 'ask'):
            print(f"  Ask: {tick.ask}")
        if hasattr(tick, 'volume'):
            print(f"  Volume: {tick.volume}")
    
    await streamer.stop_streaming()
    connection_manager.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
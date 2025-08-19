"""
基本接続テスト - MT5への接続と基本情報の表示
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
import MetaTrader5 as mt5
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer, StreamerConfig
from utils.test_config import get_mt5_config, TEST_SYMBOLS, create_tick_streamer
from utils.display_helpers import (
    print_success, print_error, print_warning, print_info,
    print_section, create_spinner, format_price
)

console = Console()

def display_account_info():
    """アカウント情報を表示"""
    account_info = mt5.account_info()
    if account_info is None:
        print_error("Failed to get account info")
        return
    
    table = Table(title="Account Information", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="magenta")
    table.add_column("Value", justify="right")
    
    table.add_row("Login", str(account_info.login))
    table.add_row("Server", account_info.server)
    table.add_row("Balance", f"${account_info.balance:.2f}")
    table.add_row("Equity", f"${account_info.equity:.2f}")
    table.add_row("Margin", f"${account_info.margin:.2f}")
    table.add_row("Free Margin", f"${account_info.margin_free:.2f}")
    table.add_row("Leverage", f"1:{account_info.leverage}")
    table.add_row("Currency", account_info.currency)
    
    console.print(table)

def display_available_symbols():
    """利用可能なシンボル一覧を表示"""
    symbols = mt5.symbols_get()
    if symbols is None:
        print_error("Failed to get symbols")
        return
    
    # フィルタリング - Forexペアのみ
    forex_symbols = [s for s in symbols if s.visible and "Forex" in s.path][:20]
    
    table = Table(title="Available Forex Symbols (Top 20)", show_header=True, header_style="bold cyan")
    table.add_column("Symbol", style="magenta")
    table.add_column("Bid", justify="right", style="green")
    table.add_column("Ask", justify="right", style="red")
    table.add_column("Spread", justify="right", style="yellow")
    table.add_column("Digits", justify="center")
    
    for symbol in forex_symbols:
        tick = mt5.symbol_info_tick(symbol.name)
        if tick:
            spread = (tick.ask - tick.bid) / symbol.point
            table.add_row(
                symbol.name,
                format_price(tick.bid, symbol.digits),
                format_price(tick.ask, symbol.digits),
                f"{spread:.1f}",
                str(symbol.digits)
            )
    
    console.print(table)

def display_symbol_info(symbol_name: str):
    """特定シンボルの詳細情報を表示"""
    symbol_info = mt5.symbol_info(symbol_name)
    if symbol_info is None:
        print_error(f"Symbol {symbol_name} not found")
        return
    
    table = Table(title=f"Symbol Details: {symbol_name}", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="magenta")
    table.add_column("Value", justify="right")
    
    table.add_row("Description", symbol_info.description)
    table.add_row("Point", str(symbol_info.point))
    table.add_row("Digits", str(symbol_info.digits))
    table.add_row("Spread", str(symbol_info.spread))
    table.add_row("Tick Size", str(symbol_info.trade_tick_size))
    table.add_row("Tick Value", str(symbol_info.trade_tick_value))
    table.add_row("Contract Size", str(symbol_info.trade_contract_size))
    table.add_row("Min Volume", str(symbol_info.volume_min))
    table.add_row("Max Volume", str(symbol_info.volume_max))
    table.add_row("Volume Step", str(symbol_info.volume_step))
    
    console.print(table)

async def test_connection():
    """接続テストを実行"""
    print_section("MT5 Connection Test")
    
    # 接続マネージャーを初期化
    config = get_mt5_config()
    manager = MT5ConnectionManager(config=config)
    
    # 接続を試行
    with create_spinner("Connecting to MT5...") as progress:
        task = progress.add_task("connect", total=None)
        success = manager.connect(config)
        progress.update(task, completed=True)
    
    if success:
        print_success("Successfully connected to MT5")
        
        # アカウント情報を表示
        print_section("Account Information")
        display_account_info()
        
        # 利用可能なシンボルを表示
        print_section("Available Symbols")
        display_available_symbols()
        
        # デフォルトシンボルの詳細を表示
        print_section("Symbol Details")
        display_symbol_info(TEST_SYMBOLS["default"])
        
        # ストリーマーをテスト
        print_section("Tick Streamer Test")
        symbol = TEST_SYMBOLS["default"]
        streamer = create_tick_streamer(symbol, manager)
        
        # 購読をテスト
        print_info(f"Testing subscription to {TEST_SYMBOLS['default']}...")
        subscribed = await streamer.subscribe_to_ticks()
        
        if subscribed:
            print_success(f"Successfully subscribed to {TEST_SYMBOLS['default']}")
            
            # 統計情報を表示
            stats = streamer.current_stats
            stats_table = Table(title="Streamer Statistics", show_header=True, header_style="bold cyan")
            stats_table.add_column("Metric", style="magenta")
            stats_table.add_column("Value", justify="right")
            
            stats_table.add_row("Symbol", streamer.symbol)
            stats_table.add_row("Buffer Size", str(streamer.buffer_size))
            stats_table.add_row("Spike Threshold", f"{streamer.spike_threshold}σ")
            stats_table.add_row("Backpressure Threshold", f"{streamer.backpressure_threshold * 100:.0f}%")
            stats_table.add_row("Statistics Window", str(streamer.statistics_window))
            stats_table.add_row("Circuit Breaker", "Open" if streamer.circuit_breaker_open else "Closed")
            
            console.print(stats_table)
            
            # 購読解除
            await streamer.unsubscribe()
            print_info("Unsubscribed from ticks")
        else:
            print_error("Failed to subscribe to ticks")
        
        # 切断
        manager.disconnect()
        print_info("Disconnected from MT5")
    else:
        print_error("Failed to connect to MT5")
        error = mt5.last_error()
        if error:
            print_error(f"Error: {error}")

async def main():
    """メイン関数"""
    try:
        await test_connection()
        print_section("Test Completed")
        print_success("All tests completed successfully")
    except KeyboardInterrupt:
        print_warning("\nTest interrupted by user")
    except Exception as e:
        print_error(f"Test failed with error: {e}")
        import traceback
        console.print(traceback.format_exc(), style="red")

if __name__ == "__main__":
    asyncio.run(main())
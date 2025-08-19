"""
エラー回復テスト - エラー処理と自動再接続機能のテスト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
import time
from datetime import datetime
from typing import Dict, List
import MetaTrader5 as mt5
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.tick_fetcher import (
    TickDataStreamer, StreamerConfig,
    MT5ConnectionError, SubscriptionError, CircuitBreakerOpenError
)
from src.common.models import Tick
from utils.test_config import get_mt5_config, TEST_SYMBOLS, create_tick_streamer, STREAMING_CONFIG
from utils.display_helpers import (
    print_success, print_error, print_warning, print_info,
    print_section, format_timestamp
)

console = Console()

class ErrorRecoveryTester:
    """エラー回復テスター"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.test_results = []
        self.error_log = []
        self.recovery_attempts = []
        self.tick_count = 0
        self.last_tick_time = None
        self.test_phase = "Initial"
        self.circuit_breaker_trips = 0
        
    def log_error(self, error_type: str, message: str, recovered: bool = False):
        """エラーをログに記録"""
        self.error_log.append({
            "timestamp": datetime.now(),
            "type": error_type,
            "message": message,
            "recovered": recovered,
            "phase": self.test_phase
        })
        
    def log_recovery(self, attempt_type: str, success: bool, duration: float):
        """回復試行をログに記録"""
        self.recovery_attempts.append({
            "timestamp": datetime.now(),
            "type": attempt_type,
            "success": success,
            "duration": duration,
            "phase": self.test_phase
        })
        
    def on_tick(self, tick: Tick):
        """ティック受信時の処理"""
        self.tick_count += 1
        self.last_tick_time = datetime.now()
        
    def on_error(self, error: Exception):
        """エラー発生時の処理"""
        self.log_error(
            error.__class__.__name__,
            str(error),
            False
        )
        
        if isinstance(error, CircuitBreakerOpenError):
            self.circuit_breaker_trips += 1
    
    def create_status_display(self, streamer: TickDataStreamer) -> Layout:
        """ステータス表示を作成"""
        layout = Layout()
        
        # レイアウトを分割
        layout.split_column(
            Layout(name="phase", size=5),
            Layout(name="status", size=8),
            Layout(name="errors", size=10),
            Layout(name="recovery", size=10)
        )
        
        # 現在のフェーズ
        phase_text = Text(f"Test Phase: {self.test_phase}", style="bold cyan")
        time_since_tick = "Never"
        if self.last_tick_time:
            elapsed = (datetime.now() - self.last_tick_time).total_seconds()
            time_since_tick = f"{elapsed:.1f}s ago"
        
        phase_panel = Panel(
            Text.from_markup(
                f"{phase_text}\n"
                f"Ticks Received: {self.tick_count}\n"
                f"Last Tick: {time_since_tick}"
            ),
            title="Test Status",
            border_style="cyan"
        )
        layout["phase"].update(phase_panel)
        
        # 接続状態
        is_connected = streamer.is_connected
        circuit_open = streamer.circuit_breaker_open
        
        status_color = "green" if is_connected else "red"
        circuit_color = "red" if circuit_open else "green"
        
        status_content = Text.from_markup(
            f"MT5 Connection: [{'green' if is_connected else 'red'}]{'Connected' if is_connected else 'Disconnected'}[/]\n"
            f"Subscription: [{'green' if streamer.is_subscribed else 'red'}]{'Active' if streamer.is_subscribed else 'Inactive'}[/]\n"
            f"Circuit Breaker: [{circuit_color}]{'Open' if circuit_open else 'Closed'}[/]\n"
            f"Circuit Trips: {self.circuit_breaker_trips}"
        )
        layout["status"].update(Panel(status_content, title="Connection Status", border_style=status_color))
        
        # エラーログ
        errors_table = Table(title="Recent Errors", show_header=True, header_style="bold red")
        errors_table.add_column("Time", width=12)
        errors_table.add_column("Type", style="magenta")
        errors_table.add_column("Phase")
        errors_table.add_column("Recovered", justify="center")
        
        for error in self.error_log[-5:]:
            errors_table.add_row(
                format_timestamp(error["timestamp"]),
                error["type"],
                error["phase"],
                "✓" if error["recovered"] else "✗"
            )
        
        layout["errors"].update(errors_table)
        
        # 回復試行
        recovery_table = Table(title="Recovery Attempts", show_header=True, header_style="bold green")
        recovery_table.add_column("Time", width=12)
        recovery_table.add_column("Type", style="magenta")
        recovery_table.add_column("Duration", justify="right")
        recovery_table.add_column("Success", justify="center")
        
        for attempt in self.recovery_attempts[-5:]:
            recovery_table.add_row(
                format_timestamp(attempt["timestamp"]),
                attempt["type"],
                f"{attempt['duration']:.2f}s",
                "✓" if attempt["success"] else "✗"
            )
        
        layout["recovery"].update(recovery_table)
        
        return layout

async def test_error_recovery():
    """エラー回復機能をテスト"""
    print_section("Error Recovery Test")
    
    symbol = TEST_SYMBOLS["default"]
    tester = ErrorRecoveryTester(symbol)
    
    # 接続
    config = get_mt5_config()
    connection_manager = MT5ConnectionManager(config=config)
    
    print_info("Connecting to MT5...")
    if not connection_manager.connect(config):
        print_error("Failed to connect to MT5")
        return
    
    print_success("Connected to MT5")
    
    # ストリーマーを設定（エラー回復を有効化）
    streamer = create_tick_streamer(symbol, connection_manager)
    
    # リスナーを設定
    streamer.add_tick_listener(tester.on_tick)
    streamer.add_error_listener(tester.on_error)
    
    # テストシナリオを実行
    try:
        with Live(
            tester.create_status_display(streamer),
            refresh_per_second=2,
            screen=True
        ) as live:
            
            # Phase 1: 正常な接続
            tester.test_phase = "Normal Operation"
            print_info("Phase 1: Testing normal operation...")
            
            if not await streamer.subscribe_to_ticks():
                print_error("Failed to subscribe")
                return
                
            streaming_task = await streamer.start_streaming()
            await asyncio.sleep(5)
            
            # Phase 2: 購読解除と再購読
            tester.test_phase = "Resubscription"
            print_info("Phase 2: Testing unsubscribe/resubscribe...")
            
            await streamer.unsubscribe()
            tester.log_error("Manual", "Unsubscribed", False)
            await asyncio.sleep(2)
            
            start_time = time.time()
            success = await streamer.subscribe_to_ticks()
            duration = time.time() - start_time
            tester.log_recovery("Resubscription", success, duration)
            
            if success:
                streaming_task = await streamer.start_streaming()
            
            await asyncio.sleep(5)
            live.update(tester.create_status_display(streamer))
            
            # Phase 3: 接続エラーのシミュレーション
            tester.test_phase = "Connection Error"
            print_info("Phase 3: Simulating connection error...")
            
            # MT5を一時的に切断
            mt5.shutdown()
            tester.log_error("Manual", "MT5 shutdown", False)
            await asyncio.sleep(3)
            
            # 再接続を試行
            start_time = time.time()
            mt5.initialize()
            success = mt5.login(config["account"], config["password"], config["server"])
            duration = time.time() - start_time
            tester.log_recovery("MT5 Reconnection", success, duration)
            
            await asyncio.sleep(5)
            live.update(tester.create_status_display(streamer))
            
            # Phase 4: Circuit Breaker テスト
            tester.test_phase = "Circuit Breaker"
            print_info("Phase 4: Testing circuit breaker...")
            
            # エラーを連続で発生させる
            for i in range(4):
                try:
                    # 無効なシンボルで購読を試行
                    invalid_streamer = TickDataStreamer(
                        symbol="INVALID_SYMBOL",
                        connection_manager=connection_manager,
                        config=config
                    )
                    await invalid_streamer.subscribe_to_ticks()
                except Exception as e:
                    tester.log_error("InvalidSymbol", str(e), False)
                    await asyncio.sleep(1)
            
            await asyncio.sleep(5)
            live.update(tester.create_status_display(streamer))
            
            # Phase 5: 自動回復
            tester.test_phase = "Auto Recovery"
            print_info("Phase 5: Testing automatic recovery...")
            
            # ストリーマーの自動再購読機能をテスト
            if not streamer.is_subscribed:
                start_time = time.time()
                success = await streamer._auto_resubscribe()
                duration = time.time() - start_time
                tester.log_recovery("Auto Resubscribe", success, duration)
            
            await asyncio.sleep(10)
            live.update(tester.create_status_display(streamer))
            
    except KeyboardInterrupt:
        print_warning("\nTest interrupted by user")
    finally:
        # クリーンアップ
        await streamer.stop_streaming()
        await streamer.unsubscribe()
        connection_manager.disconnect()
        
        # テスト結果を表示
        print_section("Test Results")
        
        # サマリーテーブル
        summary_table = Table(title="Test Summary", show_header=True, header_style="bold cyan")
        summary_table.add_column("Metric", style="magenta")
        summary_table.add_column("Value", justify="right")
        
        summary_table.add_row("Total Ticks Received", str(tester.tick_count))
        summary_table.add_row("Total Errors", str(len(tester.error_log)))
        summary_table.add_row("Recovery Attempts", str(len(tester.recovery_attempts)))
        
        successful_recoveries = sum(1 for r in tester.recovery_attempts if r["success"])
        summary_table.add_row("Successful Recoveries", str(successful_recoveries))
        summary_table.add_row("Circuit Breaker Trips", str(tester.circuit_breaker_trips))
        
        console.print(summary_table)
        
        # エラータイプ別集計
        error_types = {}
        for error in tester.error_log:
            error_type = error["type"]
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1
        
        if error_types:
            error_table = Table(title="Error Type Summary", show_header=True, header_style="bold red")
            error_table.add_column("Error Type", style="magenta")
            error_table.add_column("Count", justify="right")
            
            for error_type, count in error_types.items():
                error_table.add_row(error_type, str(count))
            
            console.print(error_table)

async def main():
    """メイン関数"""
    try:
        await test_error_recovery()
        print_success("Test completed successfully")
    except Exception as e:
        print_error(f"Test failed: {e}")
        import traceback
        console.print(traceback.format_exc(), style="red")

if __name__ == "__main__":
    asyncio.run(main())
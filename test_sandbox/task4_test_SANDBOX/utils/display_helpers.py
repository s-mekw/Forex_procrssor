"""表示ヘルパー関数"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.columns import Columns
import sys

console = Console()

def format_price(price: float, decimal_places: int = 5) -> str:
    """価格をフォーマット"""
    if price is None or np.isnan(price):
        return "N/A"
    return f"{price:.{decimal_places}f}"

def format_percentage(value: float) -> str:
    """パーセンテージをフォーマット"""
    if value is None or np.isnan(value):
        return "N/A"
    return f"{value:.2f}%"

def format_timestamp(timestamp: datetime) -> str:
    """タイムスタンプをフォーマット"""
    return timestamp.strftime("%H:%M:%S.%f")[:-3]

def get_price_color(current: float, previous: float) -> str:
    """価格変動に基づいて色を取得"""
    if current > previous:
        return "green"
    elif current < previous:
        return "red"
    return "white"

def create_tick_table(ticks: List[Dict[str, Any]], max_rows: int = 20) -> Table:
    """ティックデータのテーブルを作成"""
    table = Table(title="Recent Ticks", show_header=True, header_style="bold cyan")
    table.add_column("Time", style="dim", width=12)
    table.add_column("Symbol", style="magenta")
    table.add_column("Bid", justify="right", style="green")
    table.add_column("Ask", justify="right", style="red")
    table.add_column("Spread", justify="right", style="yellow")
    table.add_column("Volume", justify="right")
    
    # 最新のティックから表示
    for tick in ticks[-max_rows:]:
        spread = tick["ask"] - tick["bid"]
        table.add_row(
            format_timestamp(tick["timestamp"]),
            tick["symbol"],
            format_price(tick["bid"]),
            format_price(tick["ask"]),
            format_price(spread, 5),
            str(tick.get("volume", 0))
        )
    
    return table

def create_stats_table(stats: Dict[str, Any]) -> Table:
    """統計情報のテーブルを作成"""
    table = Table(title="Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", justify="right")
    
    # 統計情報を追加
    if "mean_bid" in stats:
        table.add_row("Mean Bid", format_price(stats["mean_bid"]))
    if "mean_ask" in stats:
        table.add_row("Mean Ask", format_price(stats["mean_ask"]))
    if "std_bid" in stats:
        table.add_row("Std Dev Bid", format_price(stats["std_bid"]))
    if "std_ask" in stats:
        table.add_row("Std Dev Ask", format_price(stats["std_ask"]))
    if "total_ticks" in stats:
        table.add_row("Total Ticks", str(stats["total_ticks"]))
    if "spikes_detected" in stats:
        table.add_row("Spikes Detected", str(stats["spikes_detected"]))
    if "buffer_usage" in stats:
        table.add_row("Buffer Usage", format_percentage(stats["buffer_usage"] * 100))
    if "dropped_ticks" in stats:
        table.add_row("Dropped Ticks", str(stats["dropped_ticks"]))
    
    return table

def create_connection_panel(is_connected: bool, symbol: str, server: str) -> Panel:
    """接続状態パネルを作成"""
    if is_connected:
        status = Text("● Connected", style="bold green")
        info = f"Symbol: {symbol}\nServer: {server}"
    else:
        status = Text("● Disconnected", style="bold red")
        info = "Not connected to MT5"
    
    content = Text.from_markup(f"{status}\n{info}")
    return Panel(content, title="Connection Status", border_style="green" if is_connected else "red")

def create_buffer_progress(usage: float, max_size: int) -> Panel:
    """バッファ使用状況のプログレスバーを作成"""
    percentage = usage * 100
    used = int(usage * max_size)
    
    # プログレスバーを作成
    bar_width = 40
    filled = int(bar_width * usage)
    bar = "█" * filled + "░" * (bar_width - filled)
    
    # 色を決定
    if usage < 0.5:
        color = "green"
    elif usage < 0.8:
        color = "yellow"
    else:
        color = "red"
    
    content = Text.from_markup(
        f"[{color}]{bar}[/{color}]\n"
        f"Usage: {percentage:.1f}% ({used}/{max_size})"
    )
    
    return Panel(content, title="Buffer Status", border_style=color)

def create_error_panel(errors: List[Dict[str, Any]], max_errors: int = 5) -> Panel:
    """エラー情報パネルを作成"""
    if not errors:
        content = Text("No errors", style="green")
    else:
        lines = []
        for error in errors[-max_errors:]:
            timestamp = format_timestamp(error["timestamp"])
            error_type = error.get("type", "Unknown")
            message = error.get("message", "No message")
            lines.append(f"[{timestamp}] [{error_type}] {message}")
        content = Text("\n".join(lines), style="red")
    
    return Panel(content, title=f"Recent Errors ({len(errors)} total)", border_style="red" if errors else "green")

def create_live_display(
    ticks: List[Dict[str, Any]],
    stats: Dict[str, Any],
    connection_info: Dict[str, Any],
    errors: List[Dict[str, Any]] = None
) -> Layout:
    """ライブ表示用のレイアウトを作成"""
    layout = Layout()
    
    # レイアウトを分割
    layout.split_column(
        Layout(name="header", size=5),
        Layout(name="body"),
        Layout(name="footer", size=7)
    )
    
    # ヘッダー
    layout["header"].update(
        create_connection_panel(
            connection_info.get("is_connected", False),
            connection_info.get("symbol", "N/A"),
            connection_info.get("server", "N/A")
        )
    )
    
    # ボディを左右に分割
    layout["body"].split_row(
        Layout(create_tick_table(ticks), name="ticks"),
        Layout(create_stats_table(stats), name="stats", ratio=1)
    )
    
    # フッター
    if errors:
        layout["footer"].update(create_error_panel(errors))
    else:
        buffer_usage = stats.get("buffer_usage", 0)
        buffer_size = stats.get("buffer_size", 1000)
        layout["footer"].update(create_buffer_progress(buffer_usage, buffer_size))
    
    return layout

def print_success(message: str):
    """成功メッセージを表示"""
    console.print(f"✅ {message}", style="bold green")

def print_error(message: str):
    """エラーメッセージを表示"""
    console.print(f"❌ {message}", style="bold red")

def print_warning(message: str):
    """警告メッセージを表示"""
    console.print(f"⚠️ {message}", style="bold yellow")

def print_info(message: str):
    """情報メッセージを表示"""
    console.print(f"ℹ️ {message}", style="bold blue")

def print_section(title: str):
    """セクションタイトルを表示"""
    console.print(f"\n{'=' * 50}", style="dim")
    console.print(f"{title}", style="bold cyan")
    console.print(f"{'=' * 50}", style="dim")

def clear_screen():
    """画面をクリア"""
    console.clear()

def create_spinner(text: str) -> Progress:
    """スピナーを作成"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )

def print_tick_comparison(current_tick: Dict, previous_tick: Dict):
    """ティックの比較を表示"""
    if not previous_tick:
        return
    
    bid_color = get_price_color(current_tick["bid"], previous_tick["bid"])
    ask_color = get_price_color(current_tick["ask"], previous_tick["ask"])
    
    bid_arrow = "↑" if current_tick["bid"] > previous_tick["bid"] else "↓" if current_tick["bid"] < previous_tick["bid"] else "→"
    ask_arrow = "↑" if current_tick["ask"] > previous_tick["ask"] else "↓" if current_tick["ask"] < previous_tick["ask"] else "→"
    
    console.print(
        f"Bid: [{bid_color}]{format_price(current_tick['bid'])} {bid_arrow}[/{bid_color}]  "
        f"Ask: [{ask_color}]{format_price(current_tick['ask'])} {ask_arrow}[/{ask_color}]"
    )
"""
バー表示用ヘルパー関数 - TickToBarConverterの出力を視覚的に表示
"""

from typing import List, Optional, Dict, Any
from decimal import Decimal
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich import box
import numpy as np

console = Console()

def print_success(message: str):
    """成功メッセージを表示"""
    console.print(f"[bold green]✓[/bold green] {message}")

def print_error(message: str):
    """エラーメッセージを表示"""
    console.print(f"[bold red]✗[/bold red] {message}")

def print_warning(message: str):
    """警告メッセージを表示"""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")

def print_info(message: str):
    """情報メッセージを表示"""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")

def print_section(title: str):
    """セクションヘッダーを表示"""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]{title.center(60)}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

def format_price(value: Decimal, prev_value: Optional[Decimal] = None) -> str:
    """価格をフォーマットして色付け"""
    if value is None:
        return "N/A"
    
    price_str = f"{float(value):.5f}"
    
    if prev_value is None:
        return price_str
    
    if value > prev_value:
        return f"[green]{price_str}↑[/green]"
    elif value < prev_value:
        return f"[red]{price_str}↓[/red]"
    else:
        return f"[white]{price_str}→[/white]"

def format_volume(volume: Decimal) -> str:
    """ボリュームをフォーマット"""
    vol = float(volume)
    if vol >= 1000000:
        return f"{vol/1000000:.2f}M"
    elif vol >= 1000:
        return f"{vol/1000:.2f}K"
    else:
        return f"{vol:.2f}"

def format_timestamp(timestamp: datetime) -> str:
    """タイムスタンプをフォーマット"""
    return timestamp.strftime("%H:%M:%S.%f")[:-3]

def create_bar_table(bars: List[Any], title: str = "Bar Data") -> Table:
    """バーデータのテーブルを作成"""
    table = Table(title=title, box=box.ROUNDED)
    
    table.add_column("Time", justify="center", style="cyan")
    table.add_column("Open", justify="right", style="white")
    table.add_column("High", justify="right", style="green")
    table.add_column("Low", justify="right", style="red")
    table.add_column("Close", justify="right", style="white")
    table.add_column("Volume", justify="right", style="yellow")
    table.add_column("Ticks", justify="right", style="magenta")
    table.add_column("Spread", justify="right", style="blue")
    table.add_column("Status", justify="center")
    
    for bar in bars[-10:]:  # 最新10本のバーを表示
        status_color = "green" if bar.is_complete else "yellow"
        status_text = "✓ Complete" if bar.is_complete else "⏳ Building"
        
        spread_str = f"{float(bar.avg_spread):.5f}" if bar.avg_spread else "N/A"
        
        table.add_row(
            format_timestamp(bar.time),
            f"{float(bar.open):.5f}",
            f"{float(bar.high):.5f}",
            f"{float(bar.low):.5f}",
            f"{float(bar.close):.5f}",
            format_volume(bar.volume),
            str(bar.tick_count),
            spread_str,
            f"[{status_color}]{status_text}[/{status_color}]"
        )
    
    return table

def create_current_bar_panel(current_bar: Optional[Any], symbol: str) -> Panel:
    """現在のバー情報パネルを作成"""
    if current_bar is None:
        content = "[dim]No active bar[/dim]"
    else:
        lines = []
        lines.append(f"Symbol: [bold cyan]{symbol}[/bold cyan]")
        lines.append(f"Start: {format_timestamp(current_bar.time)}")
        lines.append(f"End: {format_timestamp(current_bar.end_time)}")
        lines.append("")
        lines.append(f"Open:  {float(current_bar.open):.5f}")
        lines.append(f"High:  [green]{float(current_bar.high):.5f}[/green]")
        lines.append(f"Low:   [red]{float(current_bar.low):.5f}[/red]")
        lines.append(f"Close: {float(current_bar.close):.5f}")
        lines.append("")
        lines.append(f"Volume: [yellow]{format_volume(current_bar.volume)}[/yellow]")
        lines.append(f"Ticks: [magenta]{current_bar.tick_count}[/magenta]")
        
        if current_bar.avg_spread:
            lines.append(f"Avg Spread: [blue]{float(current_bar.avg_spread):.5f}[/blue]")
        
        # プログレスバー（バー完成までの進捗）
        now = datetime.now()
        if now < current_bar.end_time:
            elapsed = (now - current_bar.time).total_seconds()
            total = (current_bar.end_time - current_bar.time).total_seconds()
            progress = min(elapsed / total * 100, 100)
            lines.append("")
            lines.append(f"Progress: [{'█' * int(progress/5)}{'-' * (20-int(progress/5))}] {progress:.1f}%")
        
        content = "\n".join(lines)
    
    return Panel(content, title="Current Bar", border_style="green")

def create_statistics_panel(stats: Dict[str, Any]) -> Panel:
    """統計情報パネルを作成"""
    lines = []
    
    lines.append(f"Total Bars: [bold]{stats.get('total_bars', 0)}[/bold]")
    lines.append(f"Total Ticks: [bold]{stats.get('total_ticks', 0)}[/bold]")
    lines.append(f"Conversion Rate: [bold]{stats.get('conversion_rate', 0):.2f}[/bold] ticks/bar")
    lines.append("")
    
    if 'gaps_detected' in stats:
        gap_color = "red" if stats['gaps_detected'] > 0 else "green"
        lines.append(f"Gaps Detected: [{gap_color}]{stats['gaps_detected']}[/{gap_color}]")
        
        if stats.get('max_gap_seconds'):
            lines.append(f"Max Gap: [yellow]{stats['max_gap_seconds']:.1f}s[/yellow]")
    
    if 'errors' in stats:
        error_color = "red" if stats['errors'] > 0 else "green"
        lines.append(f"Errors: [{error_color}]{stats['errors']}[/{error_color}]")
    
    if 'warnings' in stats:
        warning_color = "yellow" if stats['warnings'] > 0 else "green"
        lines.append(f"Warnings: [{warning_color}]{stats['warnings']}[/{warning_color}]")
    
    lines.append("")
    lines.append(f"Memory Used: [dim]{stats.get('memory_bars', 0)} bars in memory[/dim]")
    
    if 'uptime' in stats:
        lines.append(f"Uptime: [dim]{stats['uptime']}[/dim]")
    
    content = "\n".join(lines)
    return Panel(content, title="Statistics", border_style="blue")

def create_mini_chart(bars: List[Any], width: int = 60, height: int = 10) -> str:
    """簡易アスキーチャートを作成"""
    if not bars or len(bars) < 2:
        return "Not enough data for chart"
    
    # 最新のバーから価格データを抽出
    recent_bars = bars[-min(len(bars), width):]
    closes = [float(bar.close) for bar in recent_bars]
    
    if not closes:
        return "No price data"
    
    min_price = min(closes)
    max_price = max(closes)
    price_range = max_price - min_price
    
    if price_range == 0:
        return "Price unchanged"
    
    # チャート作成
    chart_lines = []
    
    # 価格スケール
    for h in range(height, -1, -1):
        price_at_height = min_price + (price_range * h / height)
        line = f"{price_at_height:8.5f} │"
        
        for close in closes:
            normalized = (close - min_price) / price_range * height
            if abs(normalized - h) < 0.5:
                line += "●"
            elif h == 0:
                line += "─"
            else:
                line += " "
        
        chart_lines.append(line)
    
    # X軸
    chart_lines.append(" " * 10 + "└" + "─" * len(closes))
    
    return "\n".join(chart_lines)

def create_tick_to_bar_flow(tick_count: int, bar_count: int) -> str:
    """ティック→バー変換フローの視覚化"""
    flow = []
    flow.append("Tick → Bar Conversion Flow")
    flow.append("")
    flow.append(f"Ticks In: {'●' * min(tick_count, 50)} ({tick_count})")
    flow.append("    ↓")
    flow.append("  [Converter]")
    flow.append("    ↓")
    flow.append(f"Bars Out: {'█' * min(bar_count, 50)} ({bar_count})")
    
    return "\n".join(flow)

def create_bar_progress(current_tick_count: int, expected_ticks: int = 60) -> str:
    """バー形成の進捗バー"""
    progress = min(current_tick_count / expected_ticks * 100, 100)
    filled = int(progress / 5)
    
    bar = f"[{'█' * filled}{'░' * (20 - filled)}] {current_tick_count}/{expected_ticks} ticks"
    
    if progress >= 100:
        return f"[green]{bar} ✓[/green]"
    elif progress >= 75:
        return f"[yellow]{bar}[/yellow]"
    else:
        return f"[cyan]{bar}[/cyan]"

def format_spread(spread: Decimal) -> str:
    """スプレッドを色付きでフォーマット"""
    spread_val = float(spread)
    
    if spread_val < 0.0001:
        return f"[green]{spread_val:.5f}[/green]"
    elif spread_val < 0.0003:
        return f"[yellow]{spread_val:.5f}[/yellow]"
    else:
        return f"[red]{spread_val:.5f}[/red]"

def create_alert_panel(alerts: List[Dict[str, Any]]) -> Panel:
    """アラートパネルを作成"""
    if not alerts:
        content = "[green]No alerts[/green]"
    else:
        lines = []
        for alert in alerts[-5:]:  # 最新5件のアラート
            timestamp = alert.get('timestamp', '')
            level = alert.get('level', 'INFO')
            message = alert.get('message', '')
            
            if level == 'ERROR':
                color = 'red'
                icon = '✗'
            elif level == 'WARNING':
                color = 'yellow'
                icon = '⚠'
            else:
                color = 'blue'
                icon = 'ℹ'
            
            lines.append(f"[{color}]{icon}[/{color}] {timestamp} - {message}")
        
        content = "\n".join(lines)
    
    return Panel(content, title="Alerts", border_style="yellow")

def create_conversion_summary(converter_stats: Dict[str, Any]) -> Table:
    """変換サマリーテーブルを作成"""
    table = Table(title="Conversion Summary", box=box.SIMPLE)
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("Active Converters", str(converter_stats.get('active_converters', 0)))
    table.add_row("Total Ticks Processed", str(converter_stats.get('total_ticks', 0)))
    table.add_row("Total Bars Generated", str(converter_stats.get('total_bars', 0)))
    table.add_row("Avg Ticks per Bar", f"{converter_stats.get('avg_ticks_per_bar', 0):.1f}")
    table.add_row("Gaps Detected", str(converter_stats.get('gaps_detected', 0)))
    table.add_row("Errors", str(converter_stats.get('errors', 0)))
    
    return table
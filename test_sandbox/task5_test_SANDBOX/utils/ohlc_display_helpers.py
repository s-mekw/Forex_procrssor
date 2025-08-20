"""
OHLCデータ表示用ヘルパー関数
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import polars as pl
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
import numpy as np

console = Console()

def print_success(message: str):
    """成功メッセージを表示"""
    console.print(f"[green]✓[/green] {message}")

def print_error(message: str):
    """エラーメッセージを表示"""
    console.print(f"[red]✗[/red] {message}")

def print_warning(message: str):
    """警告メッセージを表示"""
    console.print(f"[yellow]⚠[/yellow] {message}")

def print_info(message: str):
    """情報メッセージを表示"""
    console.print(f"[blue]ℹ[/blue] {message}")

def print_section(title: str):
    """セクションヘッダーを表示"""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]{title.center(60)}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

def format_price(price: float, decimals: int = 5) -> str:
    """価格をフォーマット"""
    return f"{price:.{decimals}f}"

def format_timestamp(timestamp: datetime) -> str:
    """タイムスタンプをフォーマット"""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def format_volume(volume: float) -> str:
    """ボリュームをフォーマット"""
    if volume >= 1_000_000:
        return f"{volume/1_000_000:.2f}M"
    elif volume >= 1_000:
        return f"{volume/1_000:.2f}K"
    else:
        return f"{volume:.0f}"

def get_price_color(current: float, previous: float) -> str:
    """価格変動に基づいて色を取得"""
    if current > previous:
        return "green"
    elif current < previous:
        return "red"
    else:
        return "white"

def create_ohlc_table(data: pl.DataFrame, title: str = "OHLC Data", max_rows: int = 20) -> Table:
    """OHLCデータテーブルを作成"""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    
    # カラムを追加
    table.add_column("Time", style="cyan", no_wrap=True)
    table.add_column("Open", justify="right")
    table.add_column("High", justify="right", style="green")
    table.add_column("Low", justify="right", style="red")
    table.add_column("Close", justify="right")
    table.add_column("Volume", justify="right", style="yellow")
    table.add_column("Spread", justify="right", style="cyan")
    
    # データを追加（最新のmax_rows行のみ）
    rows = data.tail(max_rows)
    
    # カラム名を確認（timeまたはtimestamp）
    time_col = "timestamp" if "timestamp" in data.columns else "time"
    volume_col = "volume" if "volume" in data.columns else "tick_volume"
    
    for row in rows.iter_rows(named=True):
        # 前の行のCloseと比較して色を決定
        close_color = "white"
        if len(rows) > 1:
            prev_close = rows.filter(pl.col(time_col) < row[time_col]).tail(1)
            if not prev_close.is_empty():
                prev_close_val = prev_close["close"][0]
                close_color = get_price_color(row["close"], prev_close_val)
        
        table.add_row(
            format_timestamp(row[time_col]),
            format_price(row["open"]),
            format_price(row["high"]),
            format_price(row["low"]),
            f"[{close_color}]{format_price(row['close'])}[/{close_color}]",
            format_volume(row.get(volume_col, 0)),
            format_price(row.get("spread", 0) / 100000, 1) if "spread" in row else "N/A"
        )
    
    return table

def create_statistics_panel(data: pl.DataFrame, symbol: str, timeframe: str) -> Panel:
    """統計情報パネルを作成"""
    if data.is_empty():
        return Panel("No data available", title=f"{symbol} - {timeframe} Statistics")
    
    # カラム名を確認
    time_col = "timestamp" if "timestamp" in data.columns else "time"
    volume_col = "volume" if "volume" in data.columns else "tick_volume"
    
    # 統計を計算
    stats = {
        "Records": len(data),
        "Period": f"{data[time_col].min()} to {data[time_col].max()}",
        "Avg Open": format_price(data["open"].mean()),
        "Avg High": format_price(data["high"].mean()),
        "Avg Low": format_price(data["low"].mean()),
        "Avg Close": format_price(data["close"].mean()),
        "Max High": format_price(data["high"].max()),
        "Min Low": format_price(data["low"].min()),
        "Total Volume": format_volume(data[volume_col].sum() if volume_col in data.columns else 0),
        "Avg Spread": format_price(data["spread"].mean() / 100000, 1) if "spread" in data.columns else "N/A"
    }
    
    # 統計テキストを作成
    stats_text = "\n".join([f"[bold]{k}:[/bold] {v}" for k, v in stats.items()])
    
    return Panel(stats_text, title=f"{symbol} - {timeframe} Statistics", border_style="blue")

def create_candlestick_chart(data: pl.DataFrame, width: int = 60, height: int = 20) -> str:
    """簡単なローソク足チャートをASCIIで作成"""
    if data.is_empty() or len(data) < 2:
        return "Insufficient data for chart"
    
    # 最新のwidth個のデータを取得
    chart_data = data.tail(width)
    
    # 価格範囲を計算
    price_min = min(chart_data["low"].min(), chart_data["close"].min())
    price_max = max(chart_data["high"].max(), chart_data["close"].max())
    price_range = price_max - price_min
    
    if price_range == 0:
        return "Price range is zero"
    
    # チャートを初期化
    chart = [[" " for _ in range(width)] for _ in range(height)]
    
    # 各バーを描画
    for i, row in enumerate(chart_data.iter_rows(named=True)):
        col = i
        
        # 高値と安値の位置を計算
        high_pos = int((1 - (row["high"] - price_min) / price_range) * (height - 1))
        low_pos = int((1 - (row["low"] - price_min) / price_range) * (height - 1))
        open_pos = int((1 - (row["open"] - price_min) / price_range) * (height - 1))
        close_pos = int((1 - (row["close"] - price_min) / price_range) * (height - 1))
        
        # ヒゲを描画
        for y in range(max(0, high_pos), min(height, low_pos + 1)):
            if y == open_pos or y == close_pos:
                # 実体部分
                if row["close"] >= row["open"]:
                    chart[y][col] = "█"  # 陽線
                else:
                    chart[y][col] = "▒"  # 陰線
            elif high_pos <= y <= low_pos:
                chart[y][col] = "│"  # ヒゲ
    
    # チャートを文字列に変換
    chart_str = "\n".join(["".join(row) for row in chart])
    
    # 価格軸を追加
    price_labels = f"High: {format_price(price_max)}\n{chart_str}\nLow: {format_price(price_min)}"
    
    return price_labels

def create_fetch_progress() -> Progress:
    """データ取得用プログレスバーを作成"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    )

def create_batch_monitor_layout() -> Layout:
    """バッチ処理モニター用レイアウトを作成"""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )
    
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right")
    )
    
    return layout

def format_timeframe(timeframe: str) -> str:
    """タイムフレームを読みやすい形式にフォーマット"""
    timeframe_names = {
        "M1": "1 Minute",
        "M5": "5 Minutes",
        "M15": "15 Minutes",
        "M30": "30 Minutes",
        "H1": "1 Hour",
        "H4": "4 Hours",
        "D1": "Daily",
        "W1": "Weekly",
        "MN1": "Monthly"
    }
    return timeframe_names.get(timeframe, timeframe)

def calculate_price_change(data: pl.DataFrame) -> Dict[str, float]:
    """価格変動を計算"""
    if data.is_empty() or len(data) < 2:
        return {"change": 0.0, "change_pct": 0.0}
    
    first_close = data["close"][0]
    last_close = data["close"][-1]
    change = last_close - first_close
    change_pct = (change / first_close) * 100 if first_close != 0 else 0
    
    return {
        "change": change,
        "change_pct": change_pct
    }

def create_comparison_table(dataframes: Dict[str, pl.DataFrame], symbol: str) -> Table:
    """複数のタイムフレームを比較するテーブルを作成"""
    table = Table(title=f"{symbol} - Timeframe Comparison", show_header=True, header_style="bold magenta")
    
    table.add_column("Timeframe", style="cyan")
    table.add_column("Records", justify="right")
    table.add_column("Latest Close", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Change %", justify="right")
    table.add_column("Volume", justify="right")
    
    for timeframe, df in dataframes.items():
        if df.is_empty():
            table.add_row(
                format_timeframe(timeframe),
                "0",
                "N/A",
                "N/A",
                "N/A",
                "N/A"
            )
        else:
            change_info = calculate_price_change(df)
            change_color = "green" if change_info["change"] >= 0 else "red"
            
            table.add_row(
                format_timeframe(timeframe),
                str(len(df)),
                format_price(df["close"][-1]),
                f"[{change_color}]{format_price(change_info['change'])}[/{change_color}]",
                f"[{change_color}]{change_info['change_pct']:.2f}%[/{change_color}]",
                format_volume(df["tick_volume"].sum() if "tick_volume" in df.columns else 0)
            )
    
    return table

def detect_patterns(data: pl.DataFrame) -> List[str]:
    """簡単なパターン検出"""
    patterns = []
    
    if len(data) < 3:
        return patterns
    
    # 最新の3本のバーを取得
    recent = data.tail(3)
    
    # Doji検出
    last_bar = recent[-1]
    body = abs(last_bar["close"] - last_bar["open"])
    range_hl = last_bar["high"] - last_bar["low"]
    
    if range_hl > 0 and body / range_hl < 0.1:
        patterns.append("Doji")
    
    # Engulfing検出
    if len(recent) >= 2:
        prev_bar = recent[-2]
        if (last_bar["close"] > last_bar["open"] and 
            prev_bar["close"] < prev_bar["open"] and
            last_bar["open"] < prev_bar["close"] and
            last_bar["close"] > prev_bar["open"]):
            patterns.append("Bullish Engulfing")
        elif (last_bar["close"] < last_bar["open"] and 
              prev_bar["close"] > prev_bar["open"] and
              last_bar["open"] > prev_bar["close"] and
              last_bar["close"] < prev_bar["open"]):
            patterns.append("Bearish Engulfing")
    
    return patterns
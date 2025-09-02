"""
Chart Display Helper Utilities
Plotlyを使用したローソク足チャートとEMAの描画
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout

console = Console()

def create_candlestick_chart(
    df: pl.DataFrame,
    ema_data: Optional[Dict[int, List[float]]] = None,
    config: Any = None,
    title: str = "BTCUSD Chart with EMA"
) -> go.Figure:
    """
    ローソク足チャートとEMAを作成
    
    Args:
        df: OHLCデータ（time, open, high, low, close, volume列を含む）
        ema_data: EMA値の辞書 {period: values}
        config: チャート設定
        title: チャートタイトル
        
    Returns:
        Plotlyフィギュア
    """
    # サブプロット作成（メインチャート + ボリューム）
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, "Volume")
    )
    
    # ローソク足チャート
    fig.add_trace(
        go.Candlestick(
            x=df["time"].to_list(),
            open=df["open"].to_list(),
            high=df["high"].to_list(),
            low=df["low"].to_list(),
            close=df["close"].to_list(),
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1, col=1
    )
    
    # EMAラインを追加
    if ema_data:
        for period, values in ema_data.items():
            color = "blue"  # デフォルト色
            if config and hasattr(config, 'chart'):
                color = config.chart.ema_colors.get(period, "gray")
            
            fig.add_trace(
                go.Scatter(
                    x=df["time"].to_list(),
                    y=values,
                    mode="lines",
                    name=f"EMA {period}",
                    line=dict(color=color, width=2)
                ),
                row=1, col=1
            )
    
    # ボリュームバー
    colors = ["#26a69a" if close >= open_ else "#ef5350" 
              for close, open_ in zip(df["close"].to_list(), df["open"].to_list())]
    
    fig.add_trace(
        go.Bar(
            x=df["time"].to_list(),
            y=df["volume"].to_list(),
            name="Volume",
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # レイアウト設定
    fig.update_layout(
        height=config.chart.chart_height if config else 800,
        width=config.chart.chart_width if config else 1200,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    # X軸の設定
    fig.update_xaxes(gridcolor="#333333", row=1, col=1)
    fig.update_xaxes(gridcolor="#333333", row=2, col=1)
    
    # Y軸の設定
    fig.update_yaxes(title_text="Price (USD)", gridcolor="#333333", row=1, col=1)
    fig.update_yaxes(title_text="Volume", gridcolor="#333333", row=2, col=1)
    
    return fig

def create_realtime_chart_update(
    fig: go.Figure,
    new_tick: Dict,
    current_bar: Dict,
    ema_values: Optional[Dict[int, float]] = None
) -> go.Figure:
    """
    リアルタイムチャート更新
    
    Args:
        fig: 既存のPlotlyフィギュア
        new_tick: 新しいティックデータ
        current_bar: 現在形成中のバー
        ema_values: 最新のEMA値
        
    Returns:
        更新されたフィギュア
    """
    # 最後のローソク足を更新
    if current_bar:
        fig.data[0].close[-1] = current_bar["close"]
        fig.data[0].high[-1] = max(fig.data[0].high[-1], current_bar["high"])
        fig.data[0].low[-1] = min(fig.data[0].low[-1], current_bar["low"])
        
        # ボリューム更新
        if len(fig.data) > 1:
            fig.data[-1].y[-1] = current_bar.get("volume", 0)
    
    # EMA値の更新
    if ema_values:
        trace_idx = 1  # EMAトレースの開始インデックス
        for period, value in ema_values.items():
            if trace_idx < len(fig.data) - 1:  # ボリュームトレースを除く
                fig.data[trace_idx].y[-1] = value
                trace_idx += 1
    
    return fig

def format_price(price: float) -> str:
    """価格のフォーマット"""
    return f"${price:,.2f}"

def format_volume(volume: float) -> str:
    """ボリュームのフォーマット"""
    if volume >= 1_000_000:
        return f"{volume/1_000_000:.2f}M"
    elif volume >= 1_000:
        return f"{volume/1_000:.2f}K"
    return f"{volume:.0f}"

def create_statistics_panel(
    df: pl.DataFrame,
    ema_data: Optional[Dict[int, List[float]]] = None
) -> Panel:
    """
    統計情報パネルを作成
    
    Args:
        df: OHLCデータ
        ema_data: EMA値
        
    Returns:
        Rich Panel
    """
    table = Table(title="Market Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="white", width=15)
    
    # 最新価格情報
    latest_price = df["close"][-1]
    table.add_row("Current Price", format_price(latest_price))
    
    # 日次統計
    daily_high = df["high"].max()
    daily_low = df["low"].min()
    daily_range = daily_high - daily_low
    
    table.add_row("Daily High", format_price(daily_high))
    table.add_row("Daily Low", format_price(daily_low))
    table.add_row("Daily Range", format_price(daily_range))
    
    # ボリューム統計
    total_volume = df["volume"].sum()
    avg_volume = df["volume"].mean()
    
    table.add_row("Total Volume", format_volume(total_volume))
    table.add_row("Avg Volume", format_volume(avg_volume))
    
    # EMA値
    if ema_data:
        table.add_row("", "")  # 空行
        for period in sorted(ema_data.keys()):
            ema_value = ema_data[period][-1] if ema_data[period] else 0
            table.add_row(f"EMA {period}", format_price(ema_value))
    
    return Panel(table, title="📊 Statistics", border_style="green")

def create_progress_indicator(
    ticks_received: int,
    bars_completed: int,
    start_time: datetime
) -> Panel:
    """
    進捗インジケーターを作成
    
    Args:
        ticks_received: 受信したティック数
        bars_completed: 完成したバー数
        start_time: 開始時刻
        
    Returns:
        Rich Panel
    """
    elapsed = (datetime.now() - start_time).total_seconds()
    tick_rate = ticks_received / elapsed if elapsed > 0 else 0
    bar_rate = bars_completed / (elapsed / 60) if elapsed > 0 else 0
    
    table = Table(show_header=False)
    table.add_column("Metric", style="yellow", width=20)
    table.add_column("Value", style="white", width=15)
    
    table.add_row("Ticks Received", f"{ticks_received:,}")
    table.add_row("Bars Completed", f"{bars_completed:,}")
    table.add_row("Tick Rate", f"{tick_rate:.1f}/sec")
    table.add_row("Bar Rate", f"{bar_rate:.1f}/min")
    table.add_row("Elapsed Time", f"{elapsed:.0f}s")
    
    return Panel(table, title="⚡ Progress", border_style="yellow")

def validate_ohlc_data(df: pl.DataFrame) -> bool:
    """
    OHLCデータの妥当性を検証
    
    Args:
        df: 検証するデータフレーム
        
    Returns:
        bool: 有効な場合True
    """
    required_columns = ["time", "open", "high", "low", "close", "volume"]
    
    # 必須列の確認
    for col in required_columns:
        if col not in df.columns:
            console.print(f"[red]Missing required column: {col}[/red]")
            return False
    
    # データ型の確認
    numeric_columns = ["open", "high", "low", "close", "volume"]
    for col in numeric_columns:
        if df[col].dtype not in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            console.print(f"[red]Invalid data type for {col}: {df[col].dtype}[/red]")
            return False
    
    # OHLC関係の検証
    invalid_bars = df.filter(
        (pl.col("high") < pl.col("low")) |
        (pl.col("high") < pl.col("open")) |
        (pl.col("high") < pl.col("close")) |
        (pl.col("low") > pl.col("open")) |
        (pl.col("low") > pl.col("close"))
    )
    
    if not invalid_bars.is_empty():
        console.print(f"[red]Found {len(invalid_bars)} invalid OHLC bars[/red]")
        return False
    
    return True

def print_chart_summary(
    df: pl.DataFrame,
    ema_data: Optional[Dict[int, List[float]]] = None,
    symbol: str = "BTCUSD#"
):
    """
    チャートサマリーをコンソールに出力
    
    Args:
        df: OHLCデータ
        ema_data: EMA値
        symbol: シンボル名
    """
    console.print("\n[bold cyan]📈 Chart Summary[/bold cyan]")
    console.print(f"Symbol: {symbol}")
    console.print(f"Timeframe: M1 (1 minute)")
    console.print(f"Total Bars: {len(df)}")
    
    if not df.is_empty():
        start_time = df["time"][0]
        end_time = df["time"][-1]
        console.print(f"Period: {start_time} to {end_time}")
        
        latest = df.row(-1, named=True)
        console.print(f"\n[bold]Latest Bar:[/bold]")
        console.print(f"  Open:  {format_price(latest['open'])}")
        console.print(f"  High:  {format_price(latest['high'])}")
        console.print(f"  Low:   {format_price(latest['low'])}")
        console.print(f"  Close: {format_price(latest['close'])}")
        console.print(f"  Volume: {format_volume(latest['volume'])}")
        
        if ema_data:
            console.print(f"\n[bold]EMA Values:[/bold]")
            for period in sorted(ema_data.keys()):
                if ema_data[period]:
                    value = ema_data[period][-1]
                    console.print(f"  EMA {period}: {format_price(value)}")
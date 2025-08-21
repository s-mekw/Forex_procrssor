"""
Marimo インタラクティブダッシュボード - ティック→バー変換の可視化
使用方法: marimo edit 06_marimo_bar_dashboard.py
"""

import marimo as mo
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
import sys
from pathlib import Path

# プロジェクトパスを追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer, StreamerConfig
from src.mt5_data_acquisition.tick_to_bar import TickToBarConverter, Tick, Bar
from src.common.config import BaseConfig

# Marimo App
__app = mo.App()

@__app.cell
def init():
    """初期化セル"""
    mo.md("# 🎯 Tick to Bar Converter Dashboard")
    
    # グローバル変数の初期化
    global converter, tick_history, bar_history, stats, connection_manager, streamer
    global is_running, selected_symbol, selected_timeframe
    
    converter = None
    tick_history = deque(maxlen=1000)
    bar_history = deque(maxlen=100)
    stats = {
        "total_ticks": 0,
        "total_bars": 0,
        "gaps_detected": 0,
        "errors": 0,
        "start_time": None,
        "last_update": None
    }
    
    connection_manager = None
    streamer = None
    is_running = False
    selected_symbol = "EURUSD"
    selected_timeframe = 60
    
    return mo.md("Dashboard initialized successfully! ✅")

@__app.cell
def controls():
    """コントロールパネル"""
    mo.md("## ⚙️ Control Panel")
    
    # シンボル選択
    symbol_select = mo.ui.dropdown(
        options=["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "AUDUSD"],
        value="EURUSD",
        label="Symbol"
    )
    
    # タイムフレーム選択
    timeframe_slider = mo.ui.slider(
        start=30,
        stop=300,
        step=30,
        value=60,
        label="Timeframe (seconds)"
    )
    
    # ギャップ閾値
    gap_threshold_slider = mo.ui.slider(
        start=5,
        stop=60,
        step=5,
        value=15,
        label="Gap Threshold (seconds)"
    )
    
    # バッファサイズ
    buffer_slider = mo.ui.slider(
        start=10,
        stop=200,
        step=10,
        value=50,
        label="Max Bars in Memory"
    )
    
    # コントロールボタン
    start_button = mo.ui.button(
        label="▶️ Start Streaming",
        kind="success"
    )
    
    stop_button = mo.ui.button(
        label="⏹️ Stop Streaming",
        kind="danger"
    )
    
    clear_button = mo.ui.button(
        label="🗑️ Clear Data",
        kind="secondary"
    )
    
    # レイアウト
    controls_layout = mo.hstack([
        mo.vstack([
            symbol_select,
            timeframe_slider
        ]),
        mo.vstack([
            gap_threshold_slider,
            buffer_slider
        ]),
        mo.vstack([
            start_button,
            stop_button,
            clear_button
        ])
    ])
    
    return controls_layout, symbol_select, timeframe_slider, gap_threshold_slider, buffer_slider, start_button, stop_button, clear_button

@__app.cell
async def streaming_handler(start_button, stop_button, clear_button, symbol_select, timeframe_slider, gap_threshold_slider, buffer_slider):
    """ストリーミング処理"""
    global converter, connection_manager, streamer, is_running, stats
    global selected_symbol, selected_timeframe
    
    if start_button.value:
        if not is_running:
            mo.md("🔄 Starting streaming...")
            
            # パラメータ取得
            selected_symbol = symbol_select.value
            selected_timeframe = timeframe_slider.value
            
            # コンバーター作成
            converter = TickToBarConverter(
                symbol=selected_symbol,
                timeframe=selected_timeframe,
                gap_threshold=gap_threshold_slider.value,
                max_completed_bars=buffer_slider.value
            )
            
            # MT5接続
            config = BaseConfig()
            mt5_config = {
                "login": config.mt5_login,
                "password": config.mt5_password.get_secret_value() if config.mt5_password else None,
                "server": config.mt5_server,
                "timeout": config.mt5_timeout,
                "retry_count": 3,
                "retry_delay": 1
            }
            
            connection_manager = MT5ConnectionManager(mt5_config)
            
            if await connection_manager.connect():
                # ストリーマー設定
                streamer_config = StreamerConfig(
                    symbols=[selected_symbol],
                    buffer_size=1000,
                    max_tick_age=5.0,
                    enable_throttling=False
                )
                
                streamer = TickDataStreamer(connection_manager, streamer_config)
                await streamer.start_streaming()
                
                is_running = True
                stats["start_time"] = datetime.now()
                
                return mo.md("✅ Streaming started successfully!")
            else:
                return mo.md("❌ Failed to connect to MT5")
    
    elif stop_button.value:
        if is_running:
            mo.md("🔄 Stopping streaming...")
            
            if streamer:
                await streamer.stop_streaming()
            
            if connection_manager:
                connection_manager.disconnect()
            
            is_running = False
            
            return mo.md("⏹️ Streaming stopped")
    
    elif clear_button.value:
        tick_history.clear()
        bar_history.clear()
        stats = {
            "total_ticks": 0,
            "total_bars": 0,
            "gaps_detected": 0,
            "errors": 0,
            "start_time": stats.get("start_time"),
            "last_update": None
        }
        
        if converter:
            converter.reset()
        
        return mo.md("🗑️ Data cleared")
    
    return mo.md("")

@__app.cell
async def data_processor():
    """データ処理ループ"""
    global tick_history, bar_history, stats
    
    if is_running and streamer and converter:
        # ティック取得
        ticks = await streamer.get_buffered_ticks(selected_symbol)
        
        for tick_data in ticks:
            try:
                # ティック作成
                tick = Tick(
                    symbol=tick_data["symbol"],
                    time=tick_data["time"],
                    bid=Decimal(str(tick_data["bid"])),
                    ask=Decimal(str(tick_data["ask"])),
                    volume=Decimal(str(tick_data.get("volume", 1.0)))
                )
                
                # ギャップ検出
                gap = converter.check_tick_gap(tick.time)
                if gap:
                    stats["gaps_detected"] += 1
                
                # コンバーターに追加
                completed_bar = converter.add_tick(tick)
                
                # 履歴に追加
                tick_history.append({
                    "time": tick.time,
                    "bid": float(tick.bid),
                    "ask": float(tick.ask),
                    "spread": float(tick.ask - tick.bid),
                    "volume": float(tick.volume)
                })
                
                if completed_bar:
                    bar_history.append({
                        "time": completed_bar.time,
                        "open": float(completed_bar.open),
                        "high": float(completed_bar.high),
                        "low": float(completed_bar.low),
                        "close": float(completed_bar.close),
                        "volume": float(completed_bar.volume),
                        "tick_count": completed_bar.tick_count
                    })
                    stats["total_bars"] += 1
                
                stats["total_ticks"] += 1
                stats["last_update"] = datetime.now()
                
            except Exception as e:
                stats["errors"] += 1
    
    # 自動更新のためのスリープ
    await asyncio.sleep(0.5)
    return mo.md("")

@__app.cell
def charts():
    """チャート表示"""
    mo.md("## 📊 Real-time Charts")
    
    # サブプロット作成
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=("Tick Price Stream", "Bar Chart (OHLC)",
                       "Volume Analysis", "Spread Analysis",
                       "Tick Distribution", "Bar Formation Progress"),
        specs=[[{"type": "scatter"}, {"type": "candlestick"}],
               [{"type": "bar"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "indicator"}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.15
    )
    
    # ティック価格ストリーム
    if tick_history:
        tick_df = pl.DataFrame(list(tick_history))
        
        fig.add_trace(
            go.Scatter(
                x=tick_df["time"].to_list(),
                y=tick_df["bid"].to_list(),
                mode="lines",
                name="Bid",
                line=dict(color="blue", width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=tick_df["time"].to_list(),
                y=tick_df["ask"].to_list(),
                mode="lines",
                name="Ask",
                line=dict(color="red", width=1)
            ),
            row=1, col=1
        )
    
    # バーチャート（OHLC）
    if bar_history:
        bar_df = pl.DataFrame(list(bar_history))
        
        fig.add_trace(
            go.Candlestick(
                x=bar_df["time"].to_list(),
                open=bar_df["open"].to_list(),
                high=bar_df["high"].to_list(),
                low=bar_df["low"].to_list(),
                close=bar_df["close"].to_list(),
                name="OHLC"
            ),
            row=1, col=2
        )
        
        # ボリューム分析
        fig.add_trace(
            go.Bar(
                x=bar_df["time"].to_list(),
                y=bar_df["volume"].to_list(),
                name="Volume",
                marker_color="yellow"
            ),
            row=2, col=1
        )
    
    # スプレッド分析
    if tick_history:
        fig.add_trace(
            go.Scatter(
                x=tick_df["time"].to_list(),
                y=tick_df["spread"].to_list(),
                mode="lines",
                name="Spread",
                line=dict(color="green", width=1)
            ),
            row=2, col=2
        )
        
        # ティック分布ヒストグラム
        fig.add_trace(
            go.Histogram(
                x=tick_df["bid"].to_list(),
                name="Price Distribution",
                marker_color="purple",
                nbinsx=30
            ),
            row=3, col=1
        )
    
    # バー形成進捗インジケーター
    if converter and converter.get_current_bar():
        current_bar = converter.get_current_bar()
        progress = (current_bar.tick_count / 60) * 100  # 60ティックで100%と仮定
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=progress,
                title={"text": f"Bar Progress ({current_bar.tick_count} ticks)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "yellow"},
                        {"range": [80, 100], "color": "lightgreen"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90
                    }
                }
            ),
            row=3, col=2
        )
    
    # レイアウト調整
    fig.update_layout(
        height=900,
        showlegend=True,
        title_text=f"Tick to Bar Converter - {selected_symbol}",
        title_font_size=20
    )
    
    return mo.ui.plotly(fig)

@__app.cell
def statistics():
    """統計表示"""
    mo.md("## 📈 Statistics")
    
    if stats["start_time"]:
        elapsed = (datetime.now() - stats["start_time"]).total_seconds()
        
        # 統計テーブル作成
        stats_data = {
            "Metric": [
                "Total Ticks",
                "Total Bars", 
                "Gaps Detected",
                "Errors",
                "Tick Rate (per sec)",
                "Bar Rate (per min)",
                "Avg Ticks per Bar",
                "Uptime"
            ],
            "Value": [
                stats["total_ticks"],
                stats["total_bars"],
                stats["gaps_detected"],
                stats["errors"],
                f"{stats['total_ticks']/elapsed:.2f}" if elapsed > 0 else "0",
                f"{stats['total_bars']/elapsed*60:.2f}" if elapsed > 0 else "0",
                f"{stats['total_ticks']/stats['total_bars']:.1f}" if stats['total_bars'] > 0 else "0",
                f"{int(elapsed//60)}m {int(elapsed%60)}s"
            ]
        }
        
        stats_df = pl.DataFrame(stats_data)
        
        # カラー付きテーブル
        return mo.ui.table(
            stats_df,
            selection=None,
            pagination=False
        )
    else:
        return mo.md("*No statistics available yet. Start streaming to see data.*")

@__app.cell
def alerts():
    """アラート表示"""
    mo.md("## 🔔 Recent Alerts")
    
    alerts_list = []
    
    if stats["gaps_detected"] > 0:
        alerts_list.append(
            mo.callout(
                f"⚠️ {stats['gaps_detected']} gap(s) detected",
                kind="warn"
            )
        )
    
    if stats["errors"] > 0:
        alerts_list.append(
            mo.callout(
                f"❌ {stats['errors']} error(s) occurred",
                kind="danger"
            )
        )
    
    if converter and converter.get_current_bar():
        current_bar = converter.get_current_bar()
        alerts_list.append(
            mo.callout(
                f"📊 Current bar: {current_bar.tick_count} ticks, "
                f"Range: {float(current_bar.high - current_bar.low):.5f}",
                kind="info"
            )
        )
    
    if not alerts_list:
        alerts_list.append(
            mo.callout(
                "✅ No alerts at this time",
                kind="success"
            )
        )
    
    return mo.vstack(alerts_list)

@__app.cell
def footer():
    """フッター"""
    return mo.md("""
    ---
    ### 📝 Instructions
    
    1. **Configure Settings**: Select symbol, timeframe, and thresholds
    2. **Start Streaming**: Click the Start button to begin receiving ticks
    3. **Monitor Charts**: Watch real-time tick and bar formation
    4. **Check Statistics**: Monitor performance metrics
    5. **Stop/Clear**: Stop streaming or clear data as needed
    
    ### 🔑 Key Features
    
    - **Real-time Visualization**: Live tick and bar charts
    - **Gap Detection**: Automatic detection of tick gaps
    - **Performance Metrics**: Throughput and conversion statistics
    - **Interactive Controls**: Adjust parameters on the fly
    - **Alert System**: Visual notifications for important events
    
    ---
    *Dashboard powered by Marimo, MT5, and TickToBarConverter*
    """)

if __name__ == "__main__":
    __app.run()
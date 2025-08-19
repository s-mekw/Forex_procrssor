"""
Marimo Dashboard - インタラクティブなリアルタイムダッシュボード
"""

import marimo as mo
import asyncio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from collections import deque
import sys
from pathlib import Path

# プロジェクトパスを追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer, StreamerConfig
from src.common.models import Tick
from utils.test_config import get_mt5_credentials, TEST_SYMBOLS, STREAMING_CONFIG

# Marimoアプリケーション
app = mo.App()

@app.cell
def __():
    """初期設定とインポート"""
    import marimo as mo
    import asyncio
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from datetime import datetime, timedelta
    from collections import deque
    import sys
    from pathlib import Path
    
    # プロジェクトパスを追加
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
    from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer, StreamerConfig
    from src.common.models import Tick
    from utils.test_config import get_mt5_config, TEST_SYMBOLS, create_tick_streamer, STREAMING_CONFIG
    
    return mo, asyncio, np, pd, go, make_subplots, datetime, timedelta, deque

@app.cell
def __(mo):
    """UIコンポーネント"""
    
    # シンボル選択
    symbol_select = mo.ui.dropdown(
        options=TEST_SYMBOLS["major"] + TEST_SYMBOLS["minor"],
        value=TEST_SYMBOLS["default"],
        label="Symbol"
    )
    
    # バッファサイズスライダー
    buffer_size_slider = mo.ui.slider(
        start=100,
        stop=2000,
        step=100,
        value=1000,
        label="Buffer Size"
    )
    
    # スパイク閾値スライダー
    spike_threshold_slider = mo.ui.slider(
        start=1.0,
        stop=5.0,
        step=0.5,
        value=3.0,
        label="Spike Threshold (σ)"
    )
    
    # コントロールボタン
    start_button = mo.ui.button(label="Start Streaming", kind="success")
    stop_button = mo.ui.button(label="Stop Streaming", kind="danger")
    clear_button = mo.ui.button(label="Clear Data", kind="warning")
    
    # 自動更新チェックボックス
    auto_update = mo.ui.checkbox(value=True, label="Auto Update Charts")
    
    # 更新間隔スライダー
    update_interval_slider = mo.ui.slider(
        start=500,
        stop=5000,
        step=500,
        value=1000,
        label="Update Interval (ms)"
    )
    
    return symbol_select, buffer_size_slider, spike_threshold_slider, start_button, stop_button, clear_button, auto_update, update_interval_slider

@app.cell
def __(mo, symbol_select, buffer_size_slider, spike_threshold_slider, start_button, stop_button, clear_button, auto_update, update_interval_slider):
    """コントロールパネル"""
    
    control_panel = mo.vstack([
        mo.md("## Control Panel"),
        mo.hstack([
            symbol_select,
            buffer_size_slider,
            spike_threshold_slider
        ]),
        mo.hstack([
            start_button,
            stop_button,
            clear_button
        ]),
        mo.hstack([
            auto_update,
            update_interval_slider
        ])
    ])
    
    return control_panel,

@app.cell
async def __(mo, asyncio, deque, datetime, get_mt5_credentials, MT5ConnectionManager, TickDataStreamer, StreamerConfig, symbol_select, buffer_size_slider, spike_threshold_slider, start_button, stop_button):
    """ストリーミングロジック"""
    
    # データストレージ
    tick_data = deque(maxlen=1000)
    stats_data = {
        "total_ticks": 0,
        "spikes_detected": 0,
        "dropped_ticks": 0,
        "buffer_usage": 0.0,
        "mean_bid": 0.0,
        "mean_ask": 0.0,
        "std_bid": 0.0,
        "std_ask": 0.0
    }
    
    # ストリーマー変数
    connection_manager = None
    streamer = None
    streaming_task = None
    
    async def initialize_streamer():
        """ストリーマーを初期化"""
        global connection_manager, streamer
        
        config = get_mt5_config()
        connection_manager = MT5ConnectionManager(config=config)
        
        if connection_manager.connect(config):
            streamer = create_tick_streamer(symbol_select.value, connection_manager)
            
            # リスナーを追加
            def on_tick(tick):
                tick_data.append({
                    "timestamp": tick.timestamp,
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "spread": tick.ask - tick.bid,
                    "volume": tick.volume
                })
                stats_data["total_ticks"] += 1
                
                # 統計を更新
                current_stats = streamer.current_stats
                stats_data.update(current_stats)
            
            streamer.add_tick_listener(on_tick)
            
            return True
        return False
    
    async def start_streaming():
        """ストリーミングを開始"""
        global streaming_task
        
        if streamer and await streamer.subscribe_to_ticks():
            streaming_task = await streamer.start_streaming()
            return True
        return False
    
    async def stop_streaming():
        """ストリーミングを停止"""
        global streaming_task
        
        if streamer:
            await streamer.stop_streaming()
            await streamer.unsubscribe()
            streaming_task = None
    
    # ボタンイベントハンドラー
    if start_button.value:
        mo.output.replace(mo.md("Initializing streamer..."))
        if await initialize_streamer():
            if await start_streaming():
                mo.output.replace(mo.md("✅ Streaming started"))
            else:
                mo.output.replace(mo.md("❌ Failed to start streaming"))
        else:
            mo.output.replace(mo.md("❌ Failed to connect to MT5"))
    
    if stop_button.value:
        await stop_streaming()
        mo.output.replace(mo.md("⏹️ Streaming stopped"))
    
    return tick_data, stats_data, connection_manager, streamer, streaming_task, initialize_streamer, start_streaming, stop_streaming

@app.cell
def __(pd, tick_data, stats_data, go, make_subplots):
    """チャート作成"""
    
    def create_price_chart():
        """価格チャートを作成"""
        if not tick_data:
            return go.Figure().add_annotation(text="No data available", showarrow=False)
        
        df = pd.DataFrame(list(tick_data))
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Bid/Ask Prices", "Spread"),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        
        # Bid/Ask価格
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["bid"],
                mode="lines",
                name="Bid",
                line=dict(color="green", width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["ask"],
                mode="lines",
                name="Ask",
                line=dict(color="red", width=2)
            ),
            row=1, col=1
        )
        
        # スプレッド
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["spread"],
                mode="lines",
                name="Spread",
                line=dict(color="orange", width=1),
                fill="tozeroy"
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Spread", row=2, col=1)
        
        fig.update_layout(
            title="Real-time Price Data",
            height=600,
            showlegend=True,
            hovermode="x unified"
        )
        
        return fig
    
    def create_stats_gauges():
        """統計ゲージを作成"""
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=("Buffer Usage", "Spike Detection Rate", "Drop Rate")
        )
        
        # バッファ使用率
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=stats_data["buffer_usage"] * 100,
                title={"text": "Buffer %"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "yellow"},
                        {"range": [80, 100], "color": "red"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 80
                    }
                }
            ),
            row=1, col=1
        )
        
        # スパイク検出率
        spike_rate = (stats_data["spikes_detected"] / max(stats_data["total_ticks"], 1)) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=spike_rate,
                title={"text": "Spike %"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 10]},
                    "bar": {"color": "purple"},
                    "steps": [
                        {"range": [0, 1], "color": "lightgray"},
                        {"range": [1, 5], "color": "yellow"},
                        {"range": [5, 10], "color": "red"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # ドロップ率
        drop_rate = (stats_data["dropped_ticks"] / max(stats_data["total_ticks"], 1)) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=drop_rate,
                title={"text": "Drop %"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 5]},
                    "bar": {"color": "orange"},
                    "steps": [
                        {"range": [0, 1], "color": "lightgray"},
                        {"range": [1, 3], "color": "yellow"},
                        {"range": [3, 5], "color": "red"}
                    ]
                }
            ),
            row=1, col=3
        )
        
        fig.update_layout(height=300, showlegend=False)
        
        return fig
    
    # チャートを作成
    price_chart = create_price_chart()
    stats_gauges = create_stats_gauges()
    
    return price_chart, stats_gauges, create_price_chart, create_stats_gauges

@app.cell
def __(mo, stats_data):
    """統計情報パネル"""
    
    stats_panel = mo.vstack([
        mo.md("## Statistics"),
        mo.md(f"""
        | Metric | Value |
        |--------|-------|
        | Total Ticks | {stats_data['total_ticks']} |
        | Spikes Detected | {stats_data['spikes_detected']} |
        | Dropped Ticks | {stats_data['dropped_ticks']} |
        | Mean Bid | {stats_data['mean_bid']:.5f} |
        | Mean Ask | {stats_data['mean_ask']:.5f} |
        | Std Dev Bid | {stats_data['std_bid']:.5f} |
        | Std Dev Ask | {stats_data['std_ask']:.5f} |
        """)
    ])
    
    return stats_panel,

@app.cell
def __(mo, control_panel, price_chart, stats_gauges, stats_panel, clear_button, tick_data, stats_data):
    """メインダッシュボード"""
    
    # データクリア
    if clear_button.value:
        tick_data.clear()
        stats_data.update({
            "total_ticks": 0,
            "spikes_detected": 0,
            "dropped_ticks": 0,
            "buffer_usage": 0.0,
            "mean_bid": 0.0,
            "mean_ask": 0.0,
            "std_bid": 0.0,
            "std_ask": 0.0
        })
    
    dashboard = mo.vstack([
        mo.md("# MT5 Real-time Tick Dashboard"),
        control_panel,
        mo.hstack([
            mo.vstack([price_chart], flex=2),
            stats_panel
        ]),
        stats_gauges
    ])
    
    return dashboard,

@app.cell
def __(dashboard):
    """アプリケーションを表示"""
    dashboard

if __name__ == "__main__":
    app.run()
"""
Marimo Multi-Symbol Comparison - 複数シンボルの同時監視と比較
"""

import marimo as mo
import asyncio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from collections import defaultdict, deque
import sys
from pathlib import Path
from typing import Dict, List

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
    from collections import defaultdict, deque
    import sys
    from pathlib import Path
    from typing import Dict, List
    
    # プロジェクトパスを追加
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
    from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer, StreamerConfig
    from src.common.models import Tick
    from utils.test_config import get_mt5_config, TEST_SYMBOLS, create_tick_streamer
    
    return mo, asyncio, np, pd, go, make_subplots, datetime, timedelta, defaultdict, deque, Dict, List

@app.cell
def __(mo, TEST_SYMBOLS):
    """UIコンポーネント"""
    
    # 複数シンボル選択
    available_symbols = TEST_SYMBOLS["major"] + TEST_SYMBOLS["minor"]
    
    symbol_multiselect = mo.ui.multiselect(
        options=available_symbols,
        value=TEST_SYMBOLS["major"][:3],  # デフォルトで3つ選択
        label="Select Symbols to Monitor"
    )
    
    # 表示期間選択
    time_range_select = mo.ui.dropdown(
        options=["1 min", "5 min", "15 min", "30 min", "1 hour"],
        value="5 min",
        label="Time Range"
    )
    
    # 比較モード選択
    comparison_mode = mo.ui.radio(
        options=["Price", "Spread", "Volatility", "Correlation"],
        value="Price",
        label="Comparison Mode"
    )
    
    # コントロールボタン
    start_monitoring = mo.ui.button(label="Start Monitoring", kind="success")
    stop_monitoring = mo.ui.button(label="Stop Monitoring", kind="danger")
    refresh_button = mo.ui.button(label="Refresh Charts", kind="primary")
    
    # 自動更新設定
    auto_refresh = mo.ui.checkbox(value=True, label="Auto Refresh")
    refresh_interval = mo.ui.slider(
        start=1000,
        stop=10000,
        step=1000,
        value=2000,
        label="Refresh Interval (ms)"
    )
    
    return symbol_multiselect, time_range_select, comparison_mode, start_monitoring, stop_monitoring, refresh_button, auto_refresh, refresh_interval

@app.cell
def __(mo, symbol_multiselect, time_range_select, comparison_mode, start_monitoring, stop_monitoring, refresh_button, auto_refresh, refresh_interval):
    """コントロールパネル"""
    
    control_panel = mo.vstack([
        mo.md("## Multi-Symbol Control Panel"),
        symbol_multiselect,
        mo.hstack([
            time_range_select,
            comparison_mode
        ]),
        mo.hstack([
            start_monitoring,
            stop_monitoring,
            refresh_button
        ]),
        mo.hstack([
            auto_refresh,
            refresh_interval
        ])
    ])
    
    return control_panel,

@app.cell
async def __(mo, asyncio, defaultdict, deque, datetime, get_mt5_credentials, MT5ConnectionManager, TickDataStreamer, StreamerConfig, symbol_multiselect, start_monitoring, stop_monitoring, Dict):
    """マルチシンボルストリーミング"""
    
    # データストレージ
    symbol_data = defaultdict(lambda: deque(maxlen=2000))
    symbol_stats = defaultdict(dict)
    
    # ストリーマー管理
    connection_manager = None
    streamers: Dict[str, TickDataStreamer] = {}
    streaming_tasks = {}
    
    async def initialize_multi_streamers():
        """複数ストリーマーを初期化"""
        global connection_manager, streamers
        
        config = get_mt5_config()
        connection_manager = MT5ConnectionManager(config=config)
        
        if not connection_manager.connect(config):
            return False
        
        # 選択されたシンボルごとにストリーマーを作成
        for symbol in symbol_multiselect.value:
            streamer = create_tick_streamer(symbol, connection_manager)
            
            # シンボル固有のリスナーを追加
            def create_tick_listener(sym):
                def on_tick(tick):
                    symbol_data[sym].append({
                        "timestamp": tick.timestamp,
                        "bid": tick.bid,
                        "ask": tick.ask,
                        "spread": tick.ask - tick.bid,
                        "volume": tick.volume
                    })
                    
                    # 統計を更新
                    if sym in streamers:
                        symbol_stats[sym] = streamers[sym].current_stats
                return on_tick
            
            streamer.add_tick_listener(create_tick_listener(symbol))
            streamers[symbol] = streamer
        
        return True
    
    async def start_multi_streaming():
        """全ストリーマーを開始"""
        global streaming_tasks
        
        for symbol, streamer in streamers.items():
            if await streamer.subscribe_to_ticks():
                streaming_tasks[symbol] = await streamer.start_streaming()
                print(f"Started streaming for {symbol}")
            else:
                print(f"Failed to subscribe to {symbol}")
    
    async def stop_multi_streaming():
        """全ストリーマーを停止"""
        for symbol, streamer in streamers.items():
            if symbol in streaming_tasks:
                await streamer.stop_streaming()
                await streamer.unsubscribe()
                del streaming_tasks[symbol]
                print(f"Stopped streaming for {symbol}")
    
    # イベントハンドラー
    if start_monitoring.value:
        mo.output.replace(mo.md("Initializing streamers..."))
        if await initialize_multi_streamers():
            await start_multi_streaming()
            mo.output.replace(mo.md(f"✅ Monitoring {len(streamers)} symbols"))
        else:
            mo.output.replace(mo.md("❌ Failed to initialize"))
    
    if stop_monitoring.value:
        await stop_multi_streaming()
        streamers.clear()
        mo.output.replace(mo.md("⏹️ Monitoring stopped"))
    
    return symbol_data, symbol_stats, connection_manager, streamers, streaming_tasks, initialize_multi_streamers, start_multi_streaming, stop_multi_streaming

@app.cell
def __(pd, np, symbol_data, symbol_stats, go, make_subplots, comparison_mode, time_range_select):
    """比較チャート作成"""
    
    def parse_time_range(range_str):
        """時間範囲を解析"""
        if "min" in range_str:
            minutes = int(range_str.split()[0])
            return minutes * 60
        elif "hour" in range_str:
            hours = int(range_str.split()[0])
            return hours * 3600
        return 300  # デフォルト5分
    
    def create_comparison_chart():
        """比較チャートを作成"""
        if not symbol_data:
            return go.Figure().add_annotation(text="No data available", showarrow=False)
        
        mode = comparison_mode.value
        
        if mode == "Price":
            return create_price_comparison()
        elif mode == "Spread":
            return create_spread_comparison()
        elif mode == "Volatility":
            return create_volatility_comparison()
        elif mode == "Correlation":
            return create_correlation_matrix()
        
        return go.Figure()
    
    def create_price_comparison():
        """価格比較チャート"""
        fig = go.Figure()
        
        for symbol, data in symbol_data.items():
            if data:
                df = pd.DataFrame(list(data))
                
                # 正規化（最初の価格を100として）
                if len(df) > 0:
                    normalized_bid = (df["bid"] / df["bid"].iloc[0]) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=df["timestamp"],
                        y=normalized_bid,
                        mode="lines",
                        name=symbol,
                        line=dict(width=2)
                    ))
        
        fig.update_layout(
            title="Normalized Price Comparison (Base = 100)",
            xaxis_title="Time",
            yaxis_title="Normalized Price",
            height=500,
            hovermode="x unified"
        )
        
        return fig
    
    def create_spread_comparison():
        """スプレッド比較チャート"""
        fig = go.Figure()
        
        for symbol, data in symbol_data.items():
            if data:
                df = pd.DataFrame(list(data))
                
                fig.add_trace(go.Box(
                    y=df["spread"],
                    name=symbol,
                    boxmean="sd"
                ))
        
        fig.update_layout(
            title="Spread Distribution Comparison",
            yaxis_title="Spread",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_volatility_comparison():
        """ボラティリティ比較チャート"""
        fig = go.Figure()
        
        volatility_data = []
        
        for symbol, data in symbol_data.items():
            if len(data) > 10:
                df = pd.DataFrame(list(data))
                
                # リターンを計算
                returns = df["bid"].pct_change().dropna()
                
                # ローリングボラティリティ（標準偏差）
                rolling_vol = returns.rolling(window=10).std() * np.sqrt(252) * 100
                
                volatility_data.append({
                    "symbol": symbol,
                    "current_vol": rolling_vol.iloc[-1] if len(rolling_vol) > 0 else 0,
                    "avg_vol": rolling_vol.mean() if len(rolling_vol) > 0 else 0,
                    "max_vol": rolling_vol.max() if len(rolling_vol) > 0 else 0
                })
        
        if volatility_data:
            vol_df = pd.DataFrame(volatility_data)
            
            fig = go.Figure(data=[
                go.Bar(name="Current", x=vol_df["symbol"], y=vol_df["current_vol"]),
                go.Bar(name="Average", x=vol_df["symbol"], y=vol_df["avg_vol"]),
                go.Bar(name="Max", x=vol_df["symbol"], y=vol_df["max_vol"])
            ])
            
            fig.update_layout(
                title="Volatility Comparison (%)",
                xaxis_title="Symbol",
                yaxis_title="Volatility (%)",
                barmode="group",
                height=500
            )
        
        return fig
    
    def create_correlation_matrix():
        """相関行列ヒートマップ"""
        # 各シンボルの価格データを集める
        price_dict = {}
        
        for symbol, data in symbol_data.items():
            if len(data) > 50:
                df = pd.DataFrame(list(data))
                price_dict[symbol] = df["bid"].values[-100:]  # 最新100ティック
        
        if len(price_dict) < 2:
            return go.Figure().add_annotation(text="Need at least 2 symbols with data", showarrow=False)
        
        # 最小長に合わせる
        min_len = min(len(prices) for prices in price_dict.values())
        for symbol in price_dict:
            price_dict[symbol] = price_dict[symbol][-min_len:]
        
        # DataFrameを作成
        price_df = pd.DataFrame(price_dict)
        
        # 相関行列を計算
        corr_matrix = price_df.corr()
        
        # ヒートマップを作成
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix.values.round(3),
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Price Correlation Matrix",
            height=500,
            width=600
        )
        
        return fig
    
    # チャートを作成
    comparison_chart = create_comparison_chart()
    
    return comparison_chart, create_comparison_chart, parse_time_range

@app.cell
def __(mo, symbol_stats):
    """統計サマリーテーブル"""
    
    def create_stats_table():
        """統計テーブルを作成"""
        if not symbol_stats:
            return mo.md("No statistics available")
        
        rows = []
        for symbol, stats in symbol_stats.items():
            rows.append(f"""
            | {symbol} | 
            {stats.get('total_ticks', 0)} |
            {stats.get('mean_bid', 0):.5f} |
            {stats.get('std_bid', 0):.5f} |
            {stats.get('spikes_detected', 0)} |
            {stats.get('buffer_usage', 0)*100:.1f}% |
            """)
        
        table_md = f"""
        ## Symbol Statistics
        
        | Symbol | Total Ticks | Mean Bid | Std Dev | Spikes | Buffer |
        |--------|------------|----------|---------|--------|--------|
        {''.join(rows)}
        """
        
        return mo.md(table_md)
    
    stats_table = create_stats_table()
    
    return stats_table, create_stats_table

@app.cell
def __(mo, pd, symbol_data):
    """パフォーマンスメトリクス"""
    
    def calculate_performance_metrics():
        """パフォーマンスメトリクスを計算"""
        metrics = []
        
        for symbol, data in symbol_data.items():
            if len(data) > 1:
                df = pd.DataFrame(list(data))
                
                # 基本メトリクス
                first_price = df["bid"].iloc[0]
                last_price = df["bid"].iloc[-1]
                change = last_price - first_price
                change_pct = (change / first_price) * 100
                
                # ボラティリティ
                returns = df["bid"].pct_change().dropna()
                volatility = returns.std() * 100
                
                # スプレッド統計
                avg_spread = df["spread"].mean()
                max_spread = df["spread"].max()
                
                metrics.append({
                    "Symbol": symbol,
                    "Last Price": f"{last_price:.5f}",
                    "Change": f"{change:.5f}",
                    "Change %": f"{change_pct:.3f}%",
                    "Volatility": f"{volatility:.3f}%",
                    "Avg Spread": f"{avg_spread:.5f}",
                    "Max Spread": f"{max_spread:.5f}"
                })
        
        if metrics:
            df = pd.DataFrame(metrics)
            return mo.md(f"""
            ## Performance Metrics
            
            {df.to_markdown(index=False)}
            """)
        
        return mo.md("No performance data available")
    
    performance_metrics = calculate_performance_metrics()
    
    return performance_metrics, calculate_performance_metrics

@app.cell
def __(mo, control_panel, comparison_chart, stats_table, performance_metrics):
    """メインダッシュボード"""
    
    multi_symbol_dashboard = mo.vstack([
        mo.md("# Multi-Symbol Comparison Dashboard"),
        control_panel,
        comparison_chart,
        mo.hstack([
            stats_table,
            performance_metrics
        ])
    ])
    
    return multi_symbol_dashboard,

@app.cell
def __(multi_symbol_dashboard):
    """アプリケーションを表示"""
    multi_symbol_dashboard

if __name__ == "__main__":
    app.run()
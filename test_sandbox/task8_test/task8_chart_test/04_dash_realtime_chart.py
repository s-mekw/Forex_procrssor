"""
Dash Real-time Chart with Auto-refresh
DashフレームワークによるリアルタイムチャートのWebアプリケーション
ブラウザ側で自動更新されるインタラクティブチャート
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[3]))

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from typing import Dict, List, Optional
import threading
import queue
import json
import time
import socket
import signal
import atexit
import os
from threading import Lock

# プロジェクトのインポート
from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.data_processing.indicators import TechnicalIndicatorEngine
from src.mt5_data_acquisition.tick_to_bar import TickToBarConverter, Bar
from src.common.models import Tick as CommonTick
from utils.config_loader import load_config

# タイムフレーム変換辞書
TIMEFRAME_TO_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400
}

# MT5タイムフレーム変換
MT5_TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1
}

class DashRealtimeChart:
    """Dashによるリアルタイムチャートクラス"""
    
    def __init__(self, config=None):
        """初期化"""
        # TOMLから設定を読み込み
        self.config = config if config else load_config(preset="full")
        
        # MT5とデータ処理の初期化
        self.mt5_manager = None
        self.indicator_engine = TechnicalIndicatorEngine(
            ema_periods=self.config.chart.ema_periods
        )
        
        # TickToBarConverter初期化
        timeframe_seconds = TIMEFRAME_TO_SECONDS.get(self.config.chart.timeframe, 60)
        self.converter = TickToBarConverter(
            symbol=self.config.chart.symbol,
            timeframe=timeframe_seconds
        )
        
        # データ管理（スレッドセーフ用ロック付き）
        self.data_lock = Lock()  # データ更新用ロック
        self.ohlc_data = None
        self.ema_data = {}
        self.current_bar = None
        self.tick_queue = queue.Queue()
        
        # 統計情報
        self.stats = {
            "ticks_received": 0,
            "bars_completed": 0,
            "start_time": None,
            "last_update": None,
            "current_price": 0,
            "ema_values": {}
        }
        
        # スレッド管理
        self.tick_thread = None
        self.is_running = False
        
        # MT5初期化
        self.initialize_mt5()
        
    def initialize_mt5(self):
        """MT5接続を初期化"""
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")
        
        # シンボル確認
        symbol_info = mt5.symbol_info(self.config.chart.symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol {self.config.chart.symbol} not available")
        
        if not symbol_info.visible:
            mt5.symbol_select(self.config.chart.symbol, True)
        
        # 初期データ取得
        self.fetch_initial_data()
        
    def fetch_initial_data(self):
        """初期データを取得"""
        mt5_timeframe = MT5_TIMEFRAMES.get(self.config.chart.timeframe, mt5.TIMEFRAME_M1)
        rates = mt5.copy_rates_from_pos(
            self.config.chart.symbol,
            mt5_timeframe,
            0,
            self.config.chart.initial_bars
        )
        
        if rates is None or len(rates) == 0:
            raise ValueError("Failed to fetch initial data")
        
        self.ohlc_data = pl.DataFrame({
            "time": [datetime.fromtimestamp(r['time']) for r in rates],
            "open": np.array([r['open'] for r in rates], dtype=np.float32),
            "high": np.array([r['high'] for r in rates], dtype=np.float32),
            "low": np.array([r['low'] for r in rates], dtype=np.float32),
            "close": np.array([r['close'] for r in rates], dtype=np.float32),
            "volume": np.array([r['tick_volume'] for r in rates], dtype=np.float32)
        })
        
        # EMA計算
        self.calculate_ema()
        
    def calculate_ema(self):
        """EMAを計算"""
        if self.ohlc_data is None or self.ohlc_data.is_empty():
            return
        
        df_with_ema = self.indicator_engine.calculate_ema(
            self.ohlc_data,
            price_column="close"
        )
        
        for period in self.config.chart.ema_periods:
            col_name = f"ema_{period}"
            if col_name in df_with_ema.columns:
                self.ema_data[period] = df_with_ema[col_name].to_list()
                if len(self.ema_data[period]) > 0:
                    self.stats["ema_values"][period] = self.ema_data[period][-1]
    
    def tick_receiver_thread(self):
        """ティック受信スレッド"""
        last_tick_time = datetime.now()
        
        while self.is_running:
            try:
                # 最新ティック取得
                tick = mt5.symbol_info_tick(self.config.chart.symbol)
                
                if tick is None:
                    time.sleep(0.1)
                    continue
                
                tick_time = datetime.fromtimestamp(tick.time)
                
                # 新しいティックの場合のみ処理
                if tick_time > last_tick_time:
                    # Tickオブジェクトを作成
                    tick_obj = CommonTick(
                        symbol=self.config.chart.symbol,
                        timestamp=tick_time,
                        bid=float(tick.bid),
                        ask=float(tick.ask),
                        volume=float(tick.volume) if hasattr(tick, 'volume') else 1.0
                    )
                    
                    # バーコンバーターに追加
                    bar = self.converter.add_tick(tick_obj)
                    
                    with self.data_lock:
                        if bar:
                            # 新しいバーが完成
                            self.add_new_bar(bar)
                            self.stats["bars_completed"] += 1
                        
                        # 現在のバーを更新（未完成バー）
                        self.current_bar = self.converter.get_current_bar()
                        if self.current_bar:
                            self.stats["current_price"] = float(self.current_bar.close)
                            
                            # 未完成バーをOHLCデータに反映
                            self.update_current_bar_in_ohlc()
                            
                            # EMAを更新
                            self.update_ema_incremental(float(self.current_bar.close))
                        
                        self.stats["ticks_received"] += 1
                        self.stats["last_update"] = tick_time
                    
                    last_tick_time = tick_time
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Tick receiver error: {e}")
                time.sleep(1)
    
    def add_new_bar(self, bar: Bar):
        """新しいバーを追加"""
        new_row = pl.DataFrame({
            "time": [bar.time],
            "open": [np.float32(bar.open)],
            "high": [np.float32(bar.high)],
            "low": [np.float32(bar.low)],
            "close": [np.float32(bar.close)],
            "volume": [np.float32(bar.volume)]
        })
        
        self.ohlc_data = pl.concat([self.ohlc_data, new_row])
        
        # メモリ管理
        max_bars = self.config.chart.initial_bars * 2
        if len(self.ohlc_data) > max_bars:
            self.ohlc_data = self.ohlc_data.tail(max_bars)
        
        # EMA再計算
        self.calculate_ema()
    
    def update_current_bar_in_ohlc(self):
        """未完成バーをOHLCデータの最後のロウに反映"""
        if self.current_bar is None or self.ohlc_data is None:
            return
        
        try:
            # 最後のロウを未完成バーのデータで更新
            last_idx = len(self.ohlc_data) - 1
            if last_idx >= 0:
                # 最後のバーの時刻が現在のバーと同じかチェック
                last_time = self.ohlc_data["time"][last_idx]
                current_bar_time = self.current_bar.time
                
                if last_time == current_bar_time:
                    # 既存のバーを更新
                    self.ohlc_data = self.ohlc_data.with_columns([
                        pl.when(pl.col("time") == current_bar_time)
                        .then(pl.lit(np.float32(self.current_bar.high)))
                        .otherwise(pl.col("high"))
                        .alias("high"),
                        
                        pl.when(pl.col("time") == current_bar_time)
                        .then(pl.lit(np.float32(self.current_bar.low)))
                        .otherwise(pl.col("low"))
                        .alias("low"),
                        
                        pl.when(pl.col("time") == current_bar_time)
                        .then(pl.lit(np.float32(self.current_bar.close)))
                        .otherwise(pl.col("close"))
                        .alias("close"),
                        
                        pl.when(pl.col("time") == current_bar_time)
                        .then(pl.lit(np.float32(self.current_bar.volume)))
                        .otherwise(pl.col("volume"))
                        .alias("volume")
                    ])
                else:
                    # 新しい未完成バーを追加
                    new_row = pl.DataFrame({
                        "time": [self.current_bar.time],
                        "open": [np.float32(self.current_bar.open)],
                        "high": [np.float32(self.current_bar.high)],
                        "low": [np.float32(self.current_bar.low)],
                        "close": [np.float32(self.current_bar.close)],
                        "volume": [np.float32(self.current_bar.volume)]
                    })
                    self.ohlc_data = pl.concat([self.ohlc_data, new_row])
            
        except Exception as e:
            print(f"Error updating current bar in OHLC: {e}")
    
    def update_ema_incremental(self, new_price: float):
        """EMAを増分更新し、OHLCデータに同期"""
        # EMAの増分更新
        for period in self.config.chart.ema_periods:
            if period in self.ema_data and len(self.ema_data[period]) > 0:
                alpha = 2.0 / (period + 1)
                prev_ema = self.ema_data[period][-1]
                new_ema = alpha * new_price + (1 - alpha) * prev_ema
                self.ema_data[period][-1] = new_ema
                self.stats["ema_values"][period] = new_ema
        
        # 未完成バーの場合、EMAを新しいエントリとして追加
        if self.current_bar and len(self.ohlc_data) > len(self.ema_data.get(self.config.chart.ema_periods[0], [])):
            for period in self.config.chart.ema_periods:
                if period in self.ema_data and period in self.stats["ema_values"]:
                    self.ema_data[period].append(self.stats["ema_values"][period])
    
    def start_realtime(self):
        """リアルタイム受信を開始"""
        if not self.is_running:
            self.is_running = True
            self.stats["start_time"] = datetime.now()
            self.tick_thread = threading.Thread(target=self.tick_receiver_thread)
            self.tick_thread.daemon = True
            self.tick_thread.start()
    
    def stop_realtime(self):
        """リアルタイム受信を停止"""
        self.is_running = False
        if self.tick_thread:
            self.tick_thread.join(timeout=2)
    
    def create_chart(self):
        """Plotlyチャートを作成（スレッドセーフ）"""
        with self.data_lock:
            if self.ohlc_data is None or self.ohlc_data.is_empty():
                return go.Figure()
            
            # データのコピーを作成してロックを早めに解放
            ohlc_copy = self.ohlc_data.clone()
            ema_copy = {k: v.copy() for k, v in self.ema_data.items()}
        
        # ロック外でチャートを作成
        # サブプロットの作成
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{self.config.chart.symbol} - {self.config.chart.timeframe}", "Volume")
        )
        
        # ローソク足チャート（コピーしたデータを使用）
        fig.add_trace(
            go.Candlestick(
                x=ohlc_copy["time"].to_list(),
                open=ohlc_copy["open"].to_list(),
                high=ohlc_copy["high"].to_list(),
                low=ohlc_copy["low"].to_list(),
                close=ohlc_copy["close"].to_list(),
                name="OHLC"
            ),
            row=1, col=1
        )
        
        # EMAライン（コピーしたデータを使用）
        colors = ['orange', 'blue', 'green', 'red', 'purple']
        for idx, period in enumerate(self.config.chart.ema_periods):
            if period in ema_copy and len(ema_copy[period]) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=ohlc_copy["time"].to_list()[:len(ema_copy[period])],
                        y=ema_copy[period],
                        mode='lines',
                        name=f'EMA {period}',
                        line=dict(color=colors[idx % len(colors)], width=1)
                    ),
                    row=1, col=1
                )
        
        # ボリューム（コピーしたデータを使用）
        fig.add_trace(
            go.Bar(
                x=ohlc_copy["time"].to_list(),
                y=ohlc_copy["volume"].to_list(),
                name="Volume",
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        # レイアウト設定
        fig.update_layout(
            height=700,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            hovermode='x unified',
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig

# グローバルインスタンス
chart_manager = None

# Dashアプリケーションの初期化（キャッシュ無効化）
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    update_title=None  # タイトルの自動更新を無効化
)

# レイアウト定義
def serve_layout():
    """動的レイアウト生成（キャッシュ回避）"""
    return dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("📈 Real-time Forex Chart Dashboard", className="text-center mb-4"),
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Symbol:", className="d-inline me-2"),
                            html.Span(id="symbol-display", className="badge bg-primary fs-5"),
                        ], width=3),
                        dbc.Col([
                            html.H5("Current Price:", className="d-inline me-2"),
                            html.Span(id="current-price", className="badge bg-success fs-5"),
                        ], width=2),
                        dbc.Col([
                            html.H5("Last Update:", className="d-inline me-2"),
                            html.Span(id="last-update", className="badge bg-warning fs-6"),
                        ], width=2),
                        dbc.Col([
                            html.H5("Ticks:", className="d-inline me-2"),
                            html.Span(id="tick-count", className="badge bg-info fs-5"),
                        ], width=2),
                        dbc.Col([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Start Real-time", id="start-button", color="success", className="me-2"),
                                    dbc.Button("Stop", id="stop-button", color="danger"),
                                ], width=8),
                                dbc.Col([
                                    html.Div([
                                        html.Span("🟢 LIVE", id="status-indicator", 
                                                className="badge bg-success fs-6 ms-2"),
                                    ], className="d-flex align-items-center justify-content-center h-100")
                                ], width=4),
                            ])
                        ], width=3),
                    ])
                ])
            ], className="mb-3")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="live-chart", style={"height": "700px"})
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("EMA Values", className="mb-3"),
                    html.Div(id="ema-values")
                ])
            ])
        ])
    ]),
    
    # 自動更新用のインターバル
    dcc.Interval(
        id='interval-component',
        interval=1000,  # 1秒ごとに更新
        n_intervals=0
    ),
    
    # データストア（初期状態をリアルタイム実行中に設定）
    dcc.Store(id='realtime-status', data={'is_running': True})
    
], fluid=True)

# レイアウトを関数として設定（毎回新しいレイアウトを生成）
app.layout = serve_layout

# コールバック: スタートボタンとステータス表示
@app.callback(
    [Output('realtime-status', 'data'),
     Output('status-indicator', 'children'),
     Output('status-indicator', 'className')],
    [Input('start-button', 'n_clicks'),
     Input('stop-button', 'n_clicks'),
     Input('realtime-status', 'data')],
    [State('realtime-status', 'data')],
    prevent_initial_call=False
)
def toggle_realtime(start_clicks, stop_clicks, status_input, status_state):
    """リアルタイム更新の開始/停止とステータス表示更新"""
    global chart_manager
    
    # 現在のステータスを取得
    current_status = status_state if status_state else {'is_running': True}
    
    ctx = callback_context
    if not ctx.triggered:
        # 初期表示（リアルタイム実行中）
        is_running = current_status.get('is_running', True)
        if is_running:
            return current_status, "🟢 LIVE", "badge bg-success fs-6 ms-2"
        else:
            return current_status, "🔴 STOPPED", "badge bg-danger fs-6 ms-2"
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-button' and chart_manager:
        chart_manager.start_realtime()
        return {'is_running': True}, "🟢 LIVE", "badge bg-success fs-6 ms-2"
    elif button_id == 'stop-button' and chart_manager:
        chart_manager.stop_realtime()
        return {'is_running': False}, "🔴 STOPPED", "badge bg-danger fs-6 ms-2"
    
    # フォールバック
    is_running = current_status.get('is_running', True)
    if is_running:
        return current_status, "🟢 LIVE", "badge bg-success fs-6 ms-2"
    else:
        return current_status, "🔴 STOPPED", "badge bg-danger fs-6 ms-2"

# コールバック: チャート更新
@app.callback(
    [Output('live-chart', 'figure'),
     Output('symbol-display', 'children'),
     Output('current-price', 'children'),
     Output('tick-count', 'children'),
     Output('ema-values', 'children'),
     Output('last-update', 'children')],
    [Input('interval-component', 'n_intervals')],
    [State('realtime-status', 'data')]
)
def update_chart(n, status):
    """チャートと統計情報を更新"""
    global chart_manager
    
    if chart_manager is None:
        return go.Figure(), "", "$0.00", "0", "", "N/A"
    
    # チャート作成
    fig = chart_manager.create_chart()
    
    # 統計情報
    symbol = chart_manager.config.chart.symbol
    current_price = f"${chart_manager.stats['current_price']:,.2f}"
    tick_count = f"{chart_manager.stats['ticks_received']:,}"
    
    # EMA値の表示
    ema_badges = []
    for period, value in chart_manager.stats.get("ema_values", {}).items():
        ema_badges.append(
            html.Span(f"EMA{period}: ${value:,.2f}", 
                     className="badge bg-secondary me-2 fs-6")
        )
    
    # 最終更新時刻
    last_update = chart_manager.stats.get('last_update', None)
    if last_update:
        last_update_str = last_update.strftime("%H:%M:%S")
    else:
        last_update_str = "Waiting..."
    
    return fig, symbol, current_price, tick_count, ema_badges, last_update_str

def find_available_port(start_port=8050, max_attempts=10):
    """利用可能なポートを見つける"""
    for i in range(max_attempts):
        port = start_port + i
        try:
            # ポートが使用可能かテスト
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port+max_attempts}")

def cleanup():
    """終了時のクリーンアップ処理"""
    global chart_manager
    if chart_manager:
        print("\nCleaning up...")
        chart_manager.stop_realtime()
        if mt5.initialize():
            mt5.shutdown()
        print("Cleanup complete")

def signal_handler(sig, frame):
    """シグナルハンドラー"""
    print("\nReceived interrupt signal")
    cleanup()
    sys.exit(0)

def main():
    """メイン関数"""
    global chart_manager
    
    # シグナルハンドラーとクリーンアップの登録
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup)
    
    print("Initializing Dash Real-time Chart...")
    print(f"Process ID: {os.getpid()}")
    
    # TOMLファイルから設定読み込み
    config = load_config(preset="full")
    
    # チャートマネージャー初期化
    chart_manager = DashRealtimeChart(config)
    
    print(f"Symbol: {config.chart.symbol}")
    print(f"Timeframe: {config.chart.timeframe}")
    print(f"EMA periods: {config.chart.ema_periods}")
    
    # リアルタイム更新を自動開始
    print("\n🚀 Starting real-time data feed automatically...")
    chart_manager.start_realtime()
    print("✅ Real-time data feed started")
    
    # Dash設定を取得
    dash_config = getattr(config, 'dash', None)
    host = getattr(dash_config, 'host', '0.0.0.0') if dash_config else '0.0.0.0'
    default_port = getattr(dash_config, 'port', 8050) if dash_config else 8050
    debug = getattr(dash_config, 'debug', False) if dash_config else False
    
    # 環境変数からポート取得またはポート自動検出
    port = int(os.environ.get('DASH_PORT', default_port))
    
    try:
        # ポートが使用中の場合は別のポートを探す
        available_port = find_available_port(port)
        if available_port != port:
            print(f"⚠️  Port {port} is in use, using port {available_port} instead")
            port = available_port
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        print("Please close other Dash applications or specify a different port")
        sys.exit(1)
    
    print(f"\n✅ Starting Dash server on http://{host}:{port}")
    print("📊 Open your browser to view the real-time chart")
    print("Press Ctrl+C to stop")
    
    # Dashサーバー起動（use_reloader=Falseで自動リロードを無効化）
    app.run(
        debug=debug, 
        host=host, 
        port=port,
        use_reloader=False,  # 自動リロードを無効化（重要）
        dev_tools_hot_reload=False  # ホットリロードも無効化
    )

if __name__ == "__main__":
    main()
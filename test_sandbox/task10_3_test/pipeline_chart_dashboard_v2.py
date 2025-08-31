"""
Pipeline Chart Dashboard V2 with MultiTimeframeManager
Task 10.3改良版: MT5から直接マルチタイムフレームデータを取得

主な改善点:
- MT5から直接各タイムフレームのOHLCを取得
- 5分足の正確な更新（2時間ギャップ問題を解決）
- 拡張可能なマルチタイムフレーム対応
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import asyncio
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from typing import Dict, List, Optional, Any
import threading
import time
import signal
import atexit
import os
from threading import Lock
from dataclasses import dataclass, field
import logging
import toml

# プロジェクトのインポート
from src.data_processing.multiframe_manager import MultiTimeframeManager
from src.data_processing.analyzer_v2 import MultiTimeframeAnalyzerV2
from src.data_processing.pipelines import RealtimePipeline, DataPoint, ProcessingResult
from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.common.models import Tick as CommonTick

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ChartData:
    """チャート表示用のデータ管理"""
    m1_ohlc: Optional[pl.DataFrame] = None
    m5_ohlc: Optional[pl.DataFrame] = None
    m1_rci: Dict[int, List[float]] = field(default_factory=dict)
    m5_rci: Dict[int, List[float]] = field(default_factory=dict)
    last_update: Optional[datetime] = None
    current_price: float = 0.0
    current_bars: Dict[str, Dict[str, Any]] = field(default_factory=dict)

class PipelineChartManagerV2:
    """改良版Pipeline Chart Manager with MultiTimeframeManager"""
    
    def __init__(self, config_path: str = None):
        """初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        # 設定ファイルを読み込み
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = toml.load(f)
        else:
            # デフォルト設定
            self.config = {
                'chart': {
                    'symbol': 'EURJPY#',
                    'initial_bars': 200,
                    'display_bars_m1': 100,
                    'display_bars_m5': 100,
                    'update_interval': 0.5,
                    'show_grid': True
                },
                'pipeline': {
                    'queue_size': 1000,
                    'alert_threshold': 1.0,
                    'enable_metrics': True,
                    'max_history_bars': 5000
                },
                'theme': {
                    'background': '#fffbea',
                    'grid': '#e0e0e0',
                    'text': '#000000'
                },
                'dash': {
                    'host': '0.0.0.0',
                    'port': 8053,
                    'debug': False
                },
                'buffer': {
                    'max_m1_bars': 1000,
                    'max_m5_bars': 200
                },
                'mt5': {
                    'login': 75334547,
                    'password': '#Shota1627763',
                    'server': 'XMTrading-MT5 3',
                    'timeout': 60000,
                    'path': 'C:\\Program Files\\XMTrading MT5\\terminal64.exe',
                    'max_retries': 3,
                    'retry_delay': 1.0
                }
            }
        
        # 設定から値を取得
        self.symbol = self.config['chart']['symbol']
        self.initial_bars = self.config['chart']['initial_bars']
        self.display_bars_m1 = self.config['chart']['display_bars_m1']
        self.display_bars_m5 = self.config['chart']['display_bars_m5']
        self.update_interval = self.config['chart']['update_interval']
        self.show_grid = self.config['chart'].get('show_grid', True)
        
        # チャートデータ
        self.chart_data = ChartData()
        self.data_lock = Lock()
        
        # MT5接続管理
        self.mt5_manager = None
        
        # MultiTimeframeManagerを初期化
        self.multiframe_manager = MultiTimeframeManager(
            symbol=self.symbol,
            timeframes=["M1", "M5"],  # 1分足と5分足を管理
            initial_bars=self.initial_bars,
            max_bars=self.config['buffer']['max_m1_bars']
        )
        
        # MultiTimeframeAnalyzerV2を初期化
        self.analyzer = MultiTimeframeAnalyzerV2(
            symbol=self.symbol,
            timeframes=["M1", "M5"],
            initial_bars=self.initial_bars,
            max_history_bars=self.config['pipeline']['max_history_bars']
        )
        
        # RealtimePipelineの初期化（互換性のため保持）
        self.pipeline = RealtimePipeline(
            queue_size=self.config['pipeline']['queue_size'],
            alert_threshold=self.config['pipeline']['alert_threshold'],
            enable_metrics=self.config['pipeline']['enable_metrics']
        )
        
        # 統計情報
        self.stats = {
            "ticks_received": 0,
            "m1_bars_completed": 0,
            "m5_bars_completed": 0,
            "pipeline_queue_size": 0,
            "pipeline_processed": 0,
            "start_time": None,
            "last_update": None,
            "current_price": 0,
            "latency_avg": 0.0
        }
        
        # 非同期タスク管理
        self.is_running = False
        self.tick_task = None
        self.result_task = None
        self.event_loop = None
        self.async_thread = None
        
        # 初期化
        self.initialize()
        
        logger.info(f"PipelineChartManagerV2 initialized with symbol={self.symbol}")
    
    def initialize(self):
        """MT5接続とデータの初期化"""
        logger.info(f"Initializing PipelineChartManagerV2 for {self.symbol}")
        
        # MT5ConnectionManagerを使用して接続
        mt5_config = {
            'account': self.config['mt5'].get('login'),
            'password': self.config['mt5'].get('password'),
            'server': self.config['mt5'].get('server'),
            'timeout': self.config['mt5'].get('timeout', 60000),
            'path': self.config['mt5'].get('path'),
            'max_retries': self.config['mt5'].get('max_retries', 3),
            'retry_delay': self.config['mt5'].get('retry_delay', 1.0)
        }
        
        # MT5ConnectionManagerのインスタンスを作成
        self.mt5_manager = MT5ConnectionManager(mt5_config)
        
        # 接続試行
        if not self.mt5_manager.connect(mt5_config):
            # フォールバック：従来の方法で接続を試みる
            logger.warning("MT5ConnectionManager failed, trying direct initialization")
            if not mt5.initialize():
                raise RuntimeError("MT5 initialization failed")
        else:
            logger.info("MT5 connected successfully via MT5ConnectionManager")
        
        # シンボル確認
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol {self.symbol} not available")
        
        if not symbol_info.visible:
            mt5.symbol_select(self.symbol, True)
        
        # MultiTimeframeManagerとAnalyzerの初期化
        if not self.multiframe_manager.initialize_data():
            raise RuntimeError("Failed to initialize MultiTimeframeManager")
        
        if not self.analyzer.initialize():
            raise RuntimeError("Failed to initialize MultiTimeframeAnalyzerV2")
        
        # 初期データをチャートデータに設定
        self.update_chart_data_from_manager()
        
        # バー完成時のコールバックを設定
        self.multiframe_manager.set_bar_complete_callback("M1", self.on_m1_bar_complete)
        self.multiframe_manager.set_bar_complete_callback("M5", self.on_m5_bar_complete)
        
        logger.info("Initialization complete")
    
    def on_m1_bar_complete(self, timeframe: str, bar_data: Dict[str, Any]):
        """1分足バー完成時のコールバック"""
        self.stats["m1_bars_completed"] += 1
        logger.info(f"M1 bar completed: {bar_data['timestamp']}")
    
    def on_m5_bar_complete(self, timeframe: str, bar_data: Dict[str, Any]):
        """5分足バー完成時のコールバック"""
        self.stats["m5_bars_completed"] += 1
        logger.info(f"✅ M5 bar completed: {bar_data['timestamp']} "
                   f"OHLC=[{bar_data['open']:.5f}, {bar_data['high']:.5f}, "
                   f"{bar_data['low']:.5f}, {bar_data['close']:.5f}]")
    
    def update_chart_data_from_manager(self):
        """MultiTimeframeManagerからチャートデータを更新"""
        with self.data_lock:
            # M1データ取得
            self.chart_data.m1_ohlc = self.multiframe_manager.get_completed_bars("M1")
            self.chart_data.m1_rci = self.analyzer.get_latest_rci("M1")
            
            # M5データ取得
            self.chart_data.m5_ohlc = self.multiframe_manager.get_completed_bars("M5")
            self.chart_data.m5_rci = self.analyzer.get_latest_rci("M5")
            
            # 現在のバー
            self.chart_data.current_bars = self.multiframe_manager.get_all_current_bars()
            
            # 更新時刻
            self.chart_data.last_update = datetime.now()
    
    async def tick_receiver_task(self):
        """MT5からティックを受信して処理する非同期タスク"""
        logger.info("Starting tick receiver task")
        last_tick_time = datetime.now()
        tick_count = 0
        error_count = 0
        
        while self.is_running:
            try:
                # MT5からティック取得（run_in_executorで非同期化）
                loop = asyncio.get_running_loop()
                tick = await loop.run_in_executor(None, mt5.symbol_info_tick, self.symbol)
                
                if tick is None:
                    error_count += 1
                    if error_count % 10 == 0:
                        logger.warning(f"MT5 returned None tick (count: {error_count})")
                    await asyncio.sleep(0.1)
                    continue
                
                tick_time = datetime.fromtimestamp(tick.time)
                tick_count += 1
                
                # デバッグ: ティック受信状況を定期的にログ出力
                if tick_count % 100 == 0:
                    logger.info(f"Ticks received: {tick_count}, Last tick time: {tick_time}, Price: {tick.bid:.5f}")
                
                # 新しいティックの場合のみ処理
                if tick_time > last_tick_time:
                    # MultiTimeframeManagerでティック処理
                    results = self.multiframe_manager.process_tick(tick)
                    
                    # AnalyzerV2でティック分析
                    analysis = self.analyzer.analyze_tick(tick)
                    
                    # チャートデータ更新
                    self.update_chart_data_from_manager()
                    
                    # 統計更新
                    self.stats["ticks_received"] += 1
                    self.stats["current_price"] = float(tick.bid)
                    with self.data_lock:
                        self.chart_data.current_price = float(tick.bid)
                    
                    # 新しいバーが完成した場合のログ
                    for tf_name, tf_result in results.items():
                        if tf_result.get("new_bar"):
                            if tf_name == "M5":
                                logger.info(f"🎯 M5 bar completed at {tick_time}")
                    
                    last_tick_time = tick_time
                    error_count = 0
                
                await asyncio.sleep(0.01)
                
            except Exception as e:
                error_count += 1
                logger.error(f"Tick receiver error: {e}, Error count: {error_count}")
                await asyncio.sleep(1)
    
    async def result_processor_task(self):
        """パイプラインから結果を取得する非同期タスク（互換性のため保持）"""
        logger.info("Starting result processor task")
        
        while self.is_running:
            try:
                # パイプラインから結果取得（タイムアウト付き）
                result = await asyncio.wait_for(
                    self.pipeline.get_result(),
                    timeout=1.0
                )
                
                if result and result.get("status") == "success":
                    self.stats["pipeline_processed"] += 1
                    self.stats["last_update"] = datetime.now()
                
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logger.error(f"Result processor error: {e}")
                await asyncio.sleep(0.1)
    
    def async_main(self):
        """非同期メインループ"""
        async def run():
            # イベントループの設定
            self.event_loop = asyncio.get_running_loop()
            
            # パイプラインを開始
            await self.pipeline.start()
            logger.info("Pipeline started")
            
            # タスクを作成
            self.tick_task = asyncio.create_task(self.tick_receiver_task())
            self.result_task = asyncio.create_task(self.result_processor_task())
            
            # タスクを並行実行
            await asyncio.gather(self.tick_task, self.result_task, return_exceptions=True)
        
        # 非同期実行
        asyncio.run(run())
    
    def start_realtime(self):
        """リアルタイム処理を開始"""
        if not self.is_running:
            self.is_running = True
            self.stats["start_time"] = datetime.now()
            
            # 非同期処理を別スレッドで実行
            self.async_thread = threading.Thread(target=self.async_main)
            self.async_thread.daemon = True
            self.async_thread.start()
            
            logger.info("Realtime processing started")
    
    def stop_realtime(self):
        """リアルタイム処理を停止"""
        if self.is_running:
            self.is_running = False
            
            # タスクをキャンセル
            if self.tick_task:
                self.tick_task.cancel()
            if self.result_task:
                self.result_task.cancel()
            
            # パイプラインを停止
            if self.event_loop and not self.event_loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self.pipeline.stop(),
                    self.event_loop
                )
            
            # スレッドの終了を待つ
            if self.async_thread:
                self.async_thread.join(timeout=5)
            
            logger.info("Realtime processing stopped")
    
    def create_chart(self):
        """チャートを作成"""
        with self.data_lock:
            if self.chart_data.m1_ohlc is None or self.chart_data.m1_ohlc.is_empty():
                return self._create_empty_chart()
            
            # 表示バー数の制限
            display_bars_m1 = self.display_bars_m1
            display_bars_m5 = self.display_bars_m5
            m1_ohlc = self.chart_data.m1_ohlc.tail(display_bars_m1) if self.chart_data.m1_ohlc is not None else None
            m5_ohlc = self.chart_data.m5_ohlc.tail(display_bars_m5) if self.chart_data.m5_ohlc is not None and not self.chart_data.m5_ohlc.is_empty() else None
            
            # RCIデータの準備
            m1_rci = self.chart_data.m1_rci
            m5_rci = self.chart_data.m5_rci
            
            # 現在のバー情報
            current_bars = self.chart_data.current_bars
        
        # サブプロット作成（2列×4行）
        fig = make_subplots(
            rows=4, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.03,
            horizontal_spacing=0.05,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            column_widths=[0.5, 0.5],
            subplot_titles=(
                f"M1 - {self.symbol}", f"M5 - {self.symbol}",
                "M1 RCI [9, 13]", "M5 RCI [24, 33, 48]",
                "M1 RCI [24, 33, 48]", "M5 RCI [66, 108]",
                "M1 RCI [66, 108]", ""
            ),
            specs=[
                [{"type": "xy", "secondary_y": False}, {"type": "xy", "secondary_y": False}],
                [{"type": "xy", "secondary_y": False}, {"type": "xy", "secondary_y": False}],
                [{"type": "xy", "secondary_y": False}, {"type": "xy", "secondary_y": False}],
                [{"type": "xy", "secondary_y": False}, None]
            ]
        )
        
        # M1チャート（左列）
        if m1_ohlc is not None:
            # ローソク足
            fig.add_trace(
                go.Candlestick(
                    x=m1_ohlc["timestamp"].to_list(),
                    open=m1_ohlc["open"].to_list(),
                    high=m1_ohlc["high"].to_list(),
                    low=m1_ohlc["low"].to_list(),
                    close=m1_ohlc["close"].to_list(),
                    name="M1 OHLC",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # 現在のバー[0]を追加（リアルタイム価格）
            if "M1" in current_bars and current_bars["M1"]:
                current_m1 = current_bars["M1"]
                fig.add_trace(
                    go.Scatter(
                        x=[datetime.now()],
                        y=[current_m1.get("close", self.chart_data.current_price)],
                        mode='markers',
                        marker=dict(color='red', size=8),
                        name="Current M1",
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # M5チャート（右列）
        if m5_ohlc is not None:
            # ローソク足
            fig.add_trace(
                go.Candlestick(
                    x=m5_ohlc["timestamp"].to_list(),
                    open=m5_ohlc["open"].to_list(),
                    high=m5_ohlc["high"].to_list(),
                    low=m5_ohlc["low"].to_list(),
                    close=m5_ohlc["close"].to_list(),
                    name="M5 OHLC",
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # 現在のバー[0]を追加（リアルタイム価格）
            if "M5" in current_bars and current_bars["M5"]:
                current_m5 = current_bars["M5"]
                fig.add_trace(
                    go.Scatter(
                        x=[datetime.now()],
                        y=[current_m5.get("close", self.chart_data.current_price)],
                        mode='markers',
                        marker=dict(color='red', size=8),
                        name="Current M5",
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # RCI表示（簡略化のため省略、必要に応じて追加）
        
        # レイアウト設定
        fig.update_layout(
            height=1200,
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
            showlegend=True,
            hovermode='x unified',
            margin=dict(l=50, r=50, t=40, b=40),
            plot_bgcolor=self.config['theme']['background'],
            paper_bgcolor=self.config['theme']['background']
        )
        
        return fig
    
    def _create_empty_chart(self):
        """空のチャートを作成"""
        fig = make_subplots(
            rows=4, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.03,
            horizontal_spacing=0.05,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            column_widths=[0.5, 0.5],
            subplot_titles=(
                "M1 (Loading...)", "M5 (Loading...)",
                "", "", "", "", "", ""
            ),
            specs=[
                [{"type": "xy", "secondary_y": False}, {"type": "xy", "secondary_y": False}],
                [{"type": "xy", "secondary_y": False}, {"type": "xy", "secondary_y": False}],
                [{"type": "xy", "secondary_y": False}, {"type": "xy", "secondary_y": False}],
                [{"type": "xy", "secondary_y": False}, None]
            ]
        )
        
        fig.update_layout(
            height=1200,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            margin=dict(l=50, r=50, t=40, b=40)
        )
        
        return fig
    
    def get_metrics(self):
        """メトリクスを取得"""
        manager_metrics = self.multiframe_manager.get_metrics()
        analyzer_metrics = self.analyzer.get_metrics()
        pipeline_metrics = self.pipeline.get_metrics() if self.pipeline else {}
        
        return {
            **self.stats,
            "manager": manager_metrics,
            "analyzer": analyzer_metrics,
            **pipeline_metrics
        }

# グローバルインスタンス
chart_manager = None

# Dashアプリケーション
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    update_title=None
)

# レイアウト定義
def serve_layout():
    """動的レイアウト生成"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("🚀 Pipeline Chart Dashboard V2", className="text-center mb-4"),
                html.H4("MultiTimeframe with MT5 Direct Data", 
                       className="text-center text-muted mb-4"),
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
                            ], width=2),
                            dbc.Col([
                                html.H5("Price:", className="d-inline me-2"),
                                html.Span(id="current-price", className="badge bg-success fs-5"),
                            ], width=2),
                            dbc.Col([
                                html.H5("M1 Bars:", className="d-inline me-2"),
                                html.Span(id="m1-bars", className="badge bg-info fs-6"),
                            ], width=2),
                            dbc.Col([
                                html.H5("M5 Bars:", className="d-inline me-2"),
                                html.Span(id="m5-bars", className="badge bg-info fs-6"),
                            ], width=2),
                            dbc.Col([
                                dbc.Button("Start", id="start-button", color="success", className="me-2"),
                                dbc.Button("Stop", id="stop-button", color="danger"),
                                html.Span("🟢 LIVE", id="status-indicator", 
                                        className="badge bg-success fs-6 ms-3"),
                            ], width=4),
                        ])
                    ])
                ], className="mb-3")
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="live-chart", style={"height": "1200px"})
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Metrics", className="mb-3"),
                        html.Div(id="metrics-display")
                    ])
                ])
            ])
        ]),
        
        # 自動更新用インターバル
        dcc.Interval(
            id='interval-component',
            interval=chart_manager.update_interval * 1000 if chart_manager else 1000,
            n_intervals=0
        ),
        
        # データストア
        dcc.Store(id='realtime-status', data={'is_running': True})
    ], fluid=True)

app.layout = serve_layout

# コールバック: 開始/停止制御
@app.callback(
    [Output('realtime-status', 'data'),
     Output('status-indicator', 'children'),
     Output('status-indicator', 'className')],
    [Input('start-button', 'n_clicks'),
     Input('stop-button', 'n_clicks')],
    [State('realtime-status', 'data')],
    prevent_initial_call=False
)
def toggle_realtime(start_clicks, stop_clicks, status_state):
    """リアルタイム処理の開始/停止"""
    global chart_manager
    
    ctx = callback_context
    if not ctx.triggered:
        # 初期状態でリアルタイム処理を開始
        if chart_manager:
            chart_manager.start_realtime()
        return {'is_running': True}, "🟢 LIVE", "badge bg-success fs-6 ms-3"
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-button' and chart_manager:
        chart_manager.start_realtime()
        return {'is_running': True}, "🟢 LIVE", "badge bg-success fs-6 ms-3"
    elif button_id == 'stop-button' and chart_manager:
        chart_manager.stop_realtime()
        return {'is_running': False}, "🔴 STOPPED", "badge bg-danger fs-6 ms-3"
    
    return status_state, "⚪ READY", "badge bg-secondary fs-6 ms-3"

# コールバック: チャート更新
@app.callback(
    [Output('live-chart', 'figure'),
     Output('symbol-display', 'children'),
     Output('current-price', 'children'),
     Output('m1-bars', 'children'),
     Output('m5-bars', 'children'),
     Output('metrics-display', 'children')],
    [Input('interval-component', 'n_intervals')],
    [State('realtime-status', 'data')]
)
def update_display(n, status):
    """表示を更新"""
    global chart_manager
    
    if chart_manager is None:
        return go.Figure(), "", "$0.00", "0", "0", ""
    
    # チャート作成
    fig = chart_manager.create_chart()
    
    # メトリクス取得
    metrics = chart_manager.get_metrics()
    
    # 表示データ
    symbol = chart_manager.symbol
    current_price = f"${metrics['current_price']:,.5f}"
    m1_bars = str(metrics['m1_bars_completed'])
    m5_bars = str(metrics['m5_bars_completed'])
    
    # メトリクス表示
    metrics_content = [
        html.P(f"Ticks Received: {metrics['ticks_received']:,}"),
        html.P(f"M1 Bars Completed: {metrics['m1_bars_completed']:,}"),
        html.P(f"M5 Bars Completed: {metrics['m5_bars_completed']:,}"),
        html.P(f"Last Update: {metrics['last_update'].strftime('%H:%M:%S') if metrics['last_update'] else 'N/A'}")
    ]
    
    return fig, symbol, current_price, m1_bars, m5_bars, metrics_content

def cleanup():
    """終了時のクリーンアップ"""
    global chart_manager
    if chart_manager:
        logger.info("Cleaning up...")
        chart_manager.stop_realtime()
        if chart_manager.mt5_manager:
            chart_manager.mt5_manager.disconnect()
        else:
            if mt5.initialize():
                mt5.shutdown()
        logger.info("Cleanup complete")

def signal_handler(sig, frame):
    """シグナルハンドラー"""
    logger.info("Received interrupt signal")
    cleanup()
    sys.exit(0)

def main():
    """メイン関数"""
    global chart_manager
    
    # シグナルハンドラー登録
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup)
    
    print("=" * 60)
    print("Pipeline Chart Dashboard V2")
    print("MultiTimeframe with MT5 Direct Data")
    print("=" * 60)
    print(f"Process ID: {os.getpid()}")
    
    # 設定ファイルのパスを確認
    config_path = os.path.join(os.path.dirname(__file__), 'task10_3_config.toml')
    if os.path.exists(config_path):
        print(f"✅ Using config file: {config_path}")
    else:
        print(f"⚠️ Config file not found: {config_path}, using defaults")
    
    # チャートマネージャー初期化
    try:
        chart_manager = PipelineChartManagerV2(config_path=config_path)
        print(f"✅ Initialized for symbol: {chart_manager.symbol}")
        print(f"✅ MultiTimeframeManager ready")
        print(f"✅ MultiTimeframeAnalyzerV2 ready")
        
        # 自動的にリアルタイム処理を開始
        chart_manager.start_realtime()
        print(f"✅ Realtime processing started automatically")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        sys.exit(1)
    
    # ポート設定
    port = int(os.environ.get('DASH_PORT', chart_manager.config['dash']['port']))
    host = chart_manager.config['dash']['host']
    debug = chart_manager.config['dash']['debug']
    
    print(f"\n✅ Starting Dash server on http://localhost:{port}")
    print("📊 Open your browser to view the pipeline chart")
    print("Press Ctrl+C to stop")
    
    # Dashサーバー起動
    app.run(
        debug=debug,
        host=host,
        port=port,
        use_reloader=False,
        dev_tools_hot_reload=False
    )

if __name__ == "__main__":
    main()
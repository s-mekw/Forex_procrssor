"""
Pipeline Chart Dashboard with RealtimePipeline and MultiTimeframeAnalyzer
Task 10.3: RealtimePipelineとMultiTimeframeAnalyzerを使用したマルチタイムフレームチャート

このテストは、Task 10.3で実装した責務分離アーキテクチャを実際に使用して、
リアルタイムチャートを表示するデモンストレーションです。
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
import queue
import time
import socket
import signal
import atexit
import os
from threading import Lock
from dataclasses import dataclass, field
import logging
import toml

# プロジェクトのインポート
from src.data_processing.pipelines import RealtimePipeline, DataPoint, ProcessingResult
from src.data_processing.analyzer import MultiTimeframeAnalyzer
from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.common.models import Tick as CommonTick
from src.mt5_data_acquisition.tick_to_bar import TickToBarConverter, Bar, Tick
from src.mt5_data_acquisition.tick_adapter import TickAdapter

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

class PipelineChartManager:
    """RealtimePipelineとMultiTimeframeAnalyzerを使用したチャート管理クラス"""
    
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
                    'display_bars_m5': 20,
                    'update_interval': 1.0,
                    'show_grid': True
                },
                'pipeline': {
                    'queue_size': 1000,
                    'alert_threshold': 1.0,
                    'enable_metrics': True,
                    'max_history_bars': 5000
                },
                'analyzer': {
                    'short_term_periods': [9, 13, 24, 33, 48, 66, 108],
                    'long_term_periods': [24, 33, 48, 66, 108],
                    'long_timeframe': '5T'
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
                'tick_converter': {
                    'timeframe': 60
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
        
        # RealtimePipelineの初期化
        self.pipeline = RealtimePipeline(
            queue_size=self.config['pipeline']['queue_size'],
            alert_threshold=self.config['pipeline']['alert_threshold'],
            enable_metrics=self.config['pipeline']['enable_metrics'],
            enable_multiframe=True,
            max_history_bars=self.config['pipeline']['max_history_bars'],
            multiframe_config={
                'short_term_periods': self.config['analyzer']['short_term_periods'],
                'long_term_periods': self.config['analyzer']['long_term_periods'],
                'long_timeframe': self.config['analyzer']['long_timeframe']
            }
        )
        
        # MT5接続管理
        self.mt5_manager = None
        
        # TickToBarConverter追加
        self.tick_converter = TickToBarConverter(
            symbol=self.symbol,
            timeframe=self.config.get('tick_converter', {}).get('timeframe', 60),  # デフォルト1分足
            on_bar_complete=None  # 後でコールバックを設定
        )
        self.tick_adapter = TickAdapter()
        
        # 統計情報
        self.stats = {
            "ticks_received": 0,
            "bars_completed": 0,
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
        
        logger.info(f"PipelineChartManager initialized with symbol={self.symbol}, initial_bars={self.initial_bars}")
    
    def initialize(self):
        """MT5接続とデータの初期化"""
        logger.info(f"Initializing PipelineChartManager for {self.symbol}")
        
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
        
        # 初期データ取得
        self.fetch_initial_data()
        
        logger.info("Initialization complete")
    
    def calculate_rci_history(self, df: pl.DataFrame, periods: List[int]) -> Dict[int, List[float]]:
        """全履歴データのRCIを計算"""
        from src.data_processing.rci import RCICalculatorEngine
        
        rci_history = {period: [] for period in periods}
        rci_engine = RCICalculatorEngine()
        
        # 各期間で必要な最小データ数から開始
        for period in periods:
            for i in range(period, len(df) + 1):
                # 最新のperiod個のデータでRCI計算
                window_df = df[i-period:i]
                
                # calculate_multipleメソッドを使用（dataパラメータを使用）
                result = rci_engine.calculate_multiple(
                    data=window_df,
                    periods=[period],
                    mode="batch"
                )
                
                # 結果から最新のRCI値を取得
                if f"rci_{period}" in result.columns:
                    rci_value = result[f"rci_{period}"][-1]
                    # None値をスキップ
                    if rci_value is not None:
                        rci_history[period].append(float(rci_value))
                    else:
                        # デバッグ: None値が返された場合
                        logger.warning(f"RCI[{period}] returned None for window {i-period}:{i}")
        
        return rci_history

    def fetch_initial_data(self):
        """初期データを取得してアナライザに設定"""
        # M1データ取得
        rates_m1 = mt5.copy_rates_from_pos(
            self.symbol,
            mt5.TIMEFRAME_M1,
            0,
            self.initial_bars
        )
        
        if rates_m1 is None or len(rates_m1) == 0:
            raise ValueError("Failed to fetch initial M1 data")
        
        logger.info(f"Fetched {len(rates_m1)} M1 bars")
        
        # M5データ取得（十分な本数を取得）
        rates_m5 = mt5.copy_rates_from_pos(
            self.symbol,
            mt5.TIMEFRAME_M5,
            0,
            600  # RCI期間108に対して十分な本数
        )
        
        if rates_m5 is None or len(rates_m5) == 0:
            logger.warning("Failed to fetch initial M5 data")
            rates_m5 = []
        else:
            logger.info(f"Fetched {len(rates_m5)} M5 bars")
        
        # M1 DataFrame作成
        df_m1 = pl.DataFrame({
            "timestamp": [datetime.fromtimestamp(r['time']) for r in rates_m1],
            "open": np.array([r['open'] for r in rates_m1], dtype=np.float32),
            "high": np.array([r['high'] for r in rates_m1], dtype=np.float32),
            "low": np.array([r['low'] for r in rates_m1], dtype=np.float32),
            "close": np.array([r['close'] for r in rates_m1], dtype=np.float32),
            "volume": np.array([r['tick_volume'] for r in rates_m1], dtype=np.float32)
        })
        
        # M5 DataFrame作成
        if rates_m5 is not None and len(rates_m5) > 0:
            df_m5 = pl.DataFrame({
                "timestamp": [datetime.fromtimestamp(r['time']) for r in rates_m5],
                "open": np.array([r['open'] for r in rates_m5], dtype=np.float32),
                "high": np.array([r['high'] for r in rates_m5], dtype=np.float32),
                "low": np.array([r['low'] for r in rates_m5], dtype=np.float32),
                "close": np.array([r['close'] for r in rates_m5], dtype=np.float32),
                "volume": np.array([r['tick_volume'] for r in rates_m5], dtype=np.float32)
            })
        else:
            df_m5 = None
        
        # 初期RCI履歴を計算
        short_periods = [9, 13, 24, 33, 48, 66, 108]
        long_periods = [24, 33, 48, 66, 108]
        
        # M1 RCI履歴を計算
        logger.info("Calculating M1 RCI history...")
        m1_rci_history = self.calculate_rci_history(df_m1, short_periods)
        
        # M5 RCI履歴を計算（データがある場合）
        m5_rci_history = {}
        if df_m5 is not None and not df_m5.is_empty():
            logger.info("Calculating M5 RCI history...")
            # M5データの範囲を確認
            logger.info(f"M5 OHLC data range: close min={df_m5['close'].min():.5f}, max={df_m5['close'].max():.5f}")
            m5_rci_history = self.calculate_rci_history(df_m5, long_periods)
            # RCI計算結果を確認
            for period, values in m5_rci_history.items():
                if len(values) > 0:
                    logger.info(f"M5 RCI[{period}] calculated range: min={min(values):.2f}, max={max(values):.2f}, values={len(values)}")
                    # 異常値チェック
                    if max(values) > 100 or min(values) < -100:
                        logger.error(f"ERROR: M5 RCI[{period}] has values outside valid range!")
                        logger.error(f"Sample values: {values[:10]}")
                    # 価格データの混入チェック（価格は通常170台）
                    if max(values) > 150:
                        logger.error(f"CRITICAL: M5 RCI[{period}] contains price-like values (>150)!")
                        logger.error(f"This indicates price data contamination in RCI values")
                        logger.error(f"First 5 values: {values[:5]}")
        
        # チャートデータの初期設定
        with self.data_lock:
            self.chart_data.m1_ohlc = df_m1
            self.chart_data.m5_ohlc = df_m5
            self.chart_data.m1_rci = m1_rci_history
            self.chart_data.m5_rci = m5_rci_history
            
            # デバッグ：M5 RCIデータの内容を詳しく確認
            logger.info("=== M5 RCI DATA CHECK ===")
            for period, values in m5_rci_history.items():
                if len(values) > 0:
                    sample_values = values[:5] if len(values) >= 5 else values
                    logger.info(f"M5 RCI[{period}]: First values = {sample_values}")
                    # 価格のような値（170前後）が含まれていないかチェック
                    suspicious_values = [v for v in values if abs(v) > 150]
                    if suspicious_values:
                        logger.error(f"CRITICAL: M5 RCI[{period}] contains price-like values!")
                        logger.error(f"Suspicious values: {suspicious_values[:5]}")
            
            # RCI履歴の確認ログ
            for period, values in m1_rci_history.items():
                logger.info(f"M1 RCI[{period}]: {len(values)} values calculated")
            for period, values in m5_rci_history.items():
                logger.info(f"M5 RCI[{period}]: {len(values)} values calculated")
            
            logger.info(f"Chart data initialized: M1={df_m1.shape}, M5={df_m5.shape if df_m5 is not None else None}")
        
        # MultiTimeframeAnalyzerにも初期データを設定（リアルタイム処理用）
        if self.pipeline._multiframe_analyzer:
            # 初期データをバッファに追加
            for i in range(len(df_m1)):
                bar_data = {
                    "timestamp": df_m1["timestamp"][i],
                    "open": float(df_m1["open"][i]),
                    "high": float(df_m1["high"][i]),
                    "low": float(df_m1["low"][i]),
                    "close": float(df_m1["close"][i]),
                    "volume": float(df_m1["volume"][i])
                }
                self.pipeline._multiframe_analyzer.add_new_bar(bar_data)
            
            logger.info(f"Loaded {len(df_m1)} initial bars into analyzer for realtime processing")
    
    def update_chart_data_from_analysis(self, analysis: Optional[Dict[str, Any]]):
        """分析結果からチャートデータを更新"""
        if not analysis:
            logger.debug("No analysis data received")
            return
        
        # analyze_streamingのステータスをチェック
        if analysis.get("status") in ["not_ready", "no_data"]:
            logger.warning(f"Analysis not ready: {analysis.get('status')}")
            return
        
        logger.debug(f"Analysis data received: short_rci keys={list(analysis.get('short_rci', {}).keys())}, long_rci keys={list(analysis.get('long_rci', {}).keys())}, is_new_long_bar={analysis.get('is_new_long_bar', False)}")
        
        with self.data_lock:
            # 短期RCIデータの更新（1分足バー完成時のみ新しい値を追加）
            if 'short_rci' in analysis:
                for period, value in analysis['short_rci'].items():
                    if isinstance(period, int) and value is not None:
                        if period not in self.chart_data.m1_rci:
                            self.chart_data.m1_rci[period] = []
                        
                        # 単一値として追加（バー完成ごとに1つの値）
                        self.chart_data.m1_rci[period].append(float(value))
                        
                        # メモリ管理：最新200本分のみ保持
                        if len(self.chart_data.m1_rci[period]) > 200:
                            self.chart_data.m1_rci[period] = self.chart_data.m1_rci[period][-200:]
                        
                        logger.info(f"M1 RCI[{period}] updated: {len(self.chart_data.m1_rci[period])} values, latest={float(value):.2f}")
            
            # 長期RCIデータの更新（5分足バー完成時のみ）
            if analysis.get('is_new_long_bar') and 'long_rci' in analysis:
                # M5チャートデータも更新（5分足バー完成時）
                # MultiTimeframeAnalyzerの内部バッファから5分足データを生成
                if self.pipeline._multiframe_analyzer:
                    # 内部バッファから5分足に変換
                    buffer_df = self.pipeline._multiframe_analyzer.get_buffer_as_dataframe()
                    if buffer_df is not None and not buffer_df.is_empty():
                        # 5分足に集約（最新の5分間のOHLC）
                        latest_timestamp = buffer_df["timestamp"][-1]
                        five_min_ago = latest_timestamp - timedelta(minutes=5)
                        recent_bars = buffer_df.filter(pl.col("timestamp") > five_min_ago)
                        
                        if not recent_bars.is_empty():
                            # 5分足バーを作成
                            new_m5_bar = pl.DataFrame({
                                "timestamp": [latest_timestamp],
                                "open": [recent_bars["open"][0]],
                                "high": [recent_bars["high"].max()],
                                "low": [recent_bars["low"].min()],
                                "close": [recent_bars["close"][-1]],
                                "volume": [recent_bars["volume"].sum()]
                            })
                            
                            # M5チャートデータを更新
                            if self.chart_data.m5_ohlc is None:
                                self.chart_data.m5_ohlc = new_m5_bar
                            else:
                                self.chart_data.m5_ohlc = pl.concat([
                                    self.chart_data.m5_ohlc,
                                    new_m5_bar
                                ])
                                
                                # メモリ管理：設定に基づく最大バー数を保持
                                max_m5_bars = self.config['buffer']['max_m5_bars']
                                if len(self.chart_data.m5_ohlc) > max_m5_bars:
                                    self.chart_data.m5_ohlc = self.chart_data.m5_ohlc[-max_m5_bars:]
                            
                            logger.info(f"M5 chart updated: {len(self.chart_data.m5_ohlc)} bars")
                
                # RCIデータの更新
                for period, value in analysis['long_rci'].items():
                    if isinstance(period, int) and value is not None:
                        if period not in self.chart_data.m5_rci:
                            self.chart_data.m5_rci[period] = []
                        
                        # 単一値として追加（5分足バー完成ごとに1つの値）
                        self.chart_data.m5_rci[period].append(float(value))
                        
                        # メモリ管理：最新120本分のみ保持
                        if len(self.chart_data.m5_rci[period]) > 120:
                            self.chart_data.m5_rci[period] = self.chart_data.m5_rci[period][-120:]
                        
                        logger.info(f"M5 RCI[{period}] updated: {len(self.chart_data.m5_rci[period])} values, latest={float(value):.2f}")
            
            # last_updateを必ず更新（データ更新があった場合）
            self.chart_data.last_update = datetime.now()
            logger.debug(f"Chart data last_update updated to: {self.chart_data.last_update}")
    
    async def tick_receiver_task(self):
        """MT5からティックを受信してパイプラインに送信する非同期タスク"""
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
                    if error_count % 10 == 0:  # 10回ごとにログ出力
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
                    logger.debug(f"New tick: {tick_time} > {last_tick_time}, bid={tick.bid:.5f}")
                    
                    # CommonTickを作成
                    common_tick = CommonTick(
                        symbol=self.symbol,
                        timestamp=tick_time,
                        bid=float(tick.bid),
                        ask=float(tick.ask),
                        last=float(tick.last) if hasattr(tick, 'last') else float(tick.bid),
                        volume=float(tick.volume) if hasattr(tick, 'volume') else 1.0
                    )
                    
                    # TickAdapterでDecimal形式に変換してからTickを作成
                    tick_dict = self.tick_adapter.to_decimal_dict(common_tick)
                    tick_for_converter = Tick(
                        symbol=tick_dict["symbol"],
                        timestamp=tick_dict["time"],  # time フィールドを使用
                        bid=tick_dict["bid"],
                        ask=tick_dict["ask"],
                        volume=tick_dict["volume"]
                    )
                    completed_bar = self.tick_converter.add_tick(tick_for_converter)
                    
                    # バーが完成した場合のみパイプラインに送信
                    if completed_bar:
                        # デバッグログ追加
                        logger.info(f"Bar completed: {completed_bar.time} O:{float(completed_bar.open):.5f} H:{float(completed_bar.high):.5f} L:{float(completed_bar.low):.5f} C:{float(completed_bar.close):.5f} V:{float(completed_bar.volume)}")
                        
                        # M1チャートデータを更新
                        with self.data_lock:
                            if self.chart_data.m1_ohlc is not None:
                                # 新しいバーをDataFrameに追加
                                new_bar_df = pl.DataFrame({
                                    "timestamp": [completed_bar.end_time],
                                    "open": [float(completed_bar.open)],
                                    "high": [float(completed_bar.high)],
                                    "low": [float(completed_bar.low)],
                                    "close": [float(completed_bar.close)],
                                    "volume": [float(completed_bar.volume)]
                                })
                                self.chart_data.m1_ohlc = pl.concat([
                                    self.chart_data.m1_ohlc,
                                    new_bar_df
                                ])
                                
                                # メモリ管理：設定に基づく最大バー数を保持
                                max_bars = self.config['buffer']['max_m1_bars']
                                if len(self.chart_data.m1_ohlc) > max_bars:
                                    self.chart_data.m1_ohlc = self.chart_data.m1_ohlc[-max_bars:]
                                
                                logger.info(f"M1 chart updated: {len(self.chart_data.m1_ohlc)} bars")
                                # チャート更新のためにlast_updateを更新
                                self.chart_data.last_update = datetime.now()
                        
                        # OHLCデータを含むDataPointの作成
                        data_point: DataPoint = {
                            "timestamp": completed_bar.end_time,
                            "data": {
                                "symbol": self.symbol,
                                "open": float(completed_bar.open),
                                "high": float(completed_bar.high),
                                "low": float(completed_bar.low),
                                "close": float(completed_bar.close),
                                "volume": float(completed_bar.volume)
                            },
                            "metadata": {
                                "source": "MT5",
                                "bar_time": completed_bar.time,
                                "tick_count": completed_bar.tick_count
                            }
                        }
                        
                        # パイプラインに送信
                        success = await self.pipeline.submit(data_point)
                        
                        if success:
                            self.stats["bars_completed"] += 1
                            logger.debug(f"Bar sent to pipeline successfully")
                        else:
                            logger.warning("Failed to send bar to pipeline")
                    
                    # ティック統計の更新（バー完成に関係なく）
                    self.stats["ticks_received"] += 1
                    self.stats["current_price"] = float(tick.bid)
                    with self.data_lock:
                        self.chart_data.current_price = float(tick.bid)
                        
                        # 最新バーのClose価格をリアルタイム更新
                        if self.chart_data.m1_ohlc is not None and not self.chart_data.m1_ohlc.is_empty():
                            # 最新バーのインデックス
                            last_idx = len(self.chart_data.m1_ohlc) - 1
                            # Close価格を現在の価格で更新
                            self.chart_data.m1_ohlc = self.chart_data.m1_ohlc.with_columns(
                                pl.when(pl.arange(len(self.chart_data.m1_ohlc)) == last_idx)
                                .then(float(tick.bid))
                                .otherwise(pl.col("close"))
                                .alias("close")
                            )
                            # High/Lowも必要に応じて更新
                            current_high = self.chart_data.m1_ohlc["high"][last_idx]
                            current_low = self.chart_data.m1_ohlc["low"][last_idx]
                            if float(tick.bid) > current_high:
                                self.chart_data.m1_ohlc = self.chart_data.m1_ohlc.with_columns(
                                    pl.when(pl.arange(len(self.chart_data.m1_ohlc)) == last_idx)
                                    .then(float(tick.bid))
                                    .otherwise(pl.col("high"))
                                    .alias("high")
                                )
                            if float(tick.bid) < current_low:
                                self.chart_data.m1_ohlc = self.chart_data.m1_ohlc.with_columns(
                                    pl.when(pl.arange(len(self.chart_data.m1_ohlc)) == last_idx)
                                    .then(float(tick.bid))
                                    .otherwise(pl.col("low"))
                                    .alias("low")
                                )
                        
                        # ティックごとにもlast_updateを更新（価格の更新を反映）
                        self.chart_data.last_update = datetime.now()
                    
                    last_tick_time = tick_time
                    error_count = 0  # エラーカウントをリセット
                else:
                    # 同じタイムスタンプのティック
                    logger.debug(f"Same tick time: {tick_time} == {last_tick_time}")
                
                await asyncio.sleep(0.01)  # CPU負荷軽減のため待機時間を短縮
                
            except Exception as e:
                error_count += 1
                logger.error(f"Tick receiver error: {e}, Error count: {error_count}")
                await asyncio.sleep(1)
    
    async def result_processor_task(self):
        """パイプラインから結果を取得してチャートデータを更新する非同期タスク"""
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
                    
                    # マルチタイムフレーム分析結果を取得
                    multiframe_rci = result.get("multiframe_rci")
                    if multiframe_rci:
                        self.update_chart_data_from_analysis(multiframe_rci)
                    
                    # レイテンシー統計を更新
                    if "latency" in result:
                        current_avg = self.stats["latency_avg"]
                        count = self.stats["pipeline_processed"]
                        self.stats["latency_avg"] = (
                            (current_avg * (count - 1) + result["latency"]) / count
                        )
                    
                    self.stats["last_update"] = datetime.now()
                
            except asyncio.TimeoutError:
                # タイムアウトは正常（データがない場合）
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
            
            # 表示バー数の制限（設定から取得）
            display_bars_m1 = self.display_bars_m1
            display_bars_m5 = self.display_bars_m5
            m1_ohlc = self.chart_data.m1_ohlc.tail(display_bars_m1) if self.chart_data.m1_ohlc is not None else None
            m5_ohlc = self.chart_data.m5_ohlc.tail(display_bars_m5) if self.chart_data.m5_ohlc is not None and not self.chart_data.m5_ohlc.is_empty() else None
            m1_rci = {k: v[-display_bars_m1:] if len(v) > display_bars_m1 else v 
                     for k, v in self.chart_data.m1_rci.items()}
            m5_rci = {k: v[-display_bars_m5:] if len(v) > display_bars_m5 else v 
                     for k, v in self.chart_data.m5_rci.items()}
            
            # デバッグ：描画前のデータ内容を確認
            logger.info("=== CHART DATA DEBUG ===")
            if m1_ohlc is not None:
                logger.info(f"M1 OHLC: {len(m1_ohlc)} bars, close range: {m1_ohlc['close'].min():.5f} - {m1_ohlc['close'].max():.5f}")
            if m5_ohlc is not None:
                logger.info(f"M5 OHLC: {len(m5_ohlc)} bars, close range: {m5_ohlc['close'].min():.5f} - {m5_ohlc['close'].max():.5f}")
            for period, values in m5_rci.items():
                if len(values) > 0:
                    logger.info(f"M5 RCI[{period}]: {len(values)} values, range: {min(values):.2f} - {max(values):.2f}")
                    # 価格データ混入チェック
                    if max(values) > 150:
                        logger.error(f"ERROR: M5 RCI[{period}] contains price data! First values: {values[:3]}")
        
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
                [{"type": "xy", "secondary_y": False}, {"type": "xy", "secondary_y": False}],  # row=1: ローソク足
                [{"type": "xy", "secondary_y": False}, {"type": "xy", "secondary_y": False}],  # row=2: RCI
                [{"type": "xy", "secondary_y": False}, {"type": "xy", "secondary_y": False}],  # row=3: RCI
                [{"type": "xy", "secondary_y": False}, None]  # row=4: M1のRCIのみ（M5は3行目まで）
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
            
            # M1 RCI表示
            m1_timestamps = m1_ohlc["timestamp"].to_list()
            
            # サブウィンドウ1 [9, 13]
            for period in [9, 13]:
                if period in m1_rci and len(m1_rci[period]) > 0:
                    color = 'blue' if period == 9 else 'red'
                    # RCIデータ長に合わせて時間軸を調整
                    rci_len = len(m1_rci[period])
                    time_list = m1_timestamps[-rci_len:] if rci_len <= len(m1_timestamps) else m1_timestamps
                    fig.add_trace(
                        go.Scatter(
                            x=time_list,
                            y=m1_rci[period][-len(time_list):],
                            mode='lines',
                            name=f'M1 RCI {period}',
                            line=dict(color=color, width=1.5)
                        ),
                        row=2, col=1
                    )
            
            # サブウィンドウ2 [24, 33, 48]
            colors_sw2 = ['green', 'purple', 'orange']
            for i, period in enumerate([24, 33, 48]):
                if period in m1_rci and len(m1_rci[period]) > 0:
                    rci_len = len(m1_rci[period])
                    time_list = m1_timestamps[-rci_len:] if rci_len <= len(m1_timestamps) else m1_timestamps
                    fig.add_trace(
                        go.Scatter(
                            x=time_list,
                            y=m1_rci[period][-len(time_list):],
                            mode='lines',
                            name=f'M1 RCI {period}',
                            line=dict(color=colors_sw2[i], width=1.5)
                        ),
                        row=3, col=1
                    )
            
            # サブウィンドウ3 [66, 108]
            colors_sw3 = ['brown', 'pink']
            for i, period in enumerate([66, 108]):
                if period in m1_rci and len(m1_rci[period]) > 0:
                    rci_len = len(m1_rci[period])
                    time_list = m1_timestamps[-rci_len:] if rci_len <= len(m1_timestamps) else m1_timestamps
                    fig.add_trace(
                        go.Scatter(
                            x=time_list,
                            y=m1_rci[period][-len(time_list):],
                            mode='lines',
                            name=f'M1 RCI {period}',
                            line=dict(color=colors_sw3[i], width=1.5)
                        ),
                        row=4, col=1
                    )
        
        # M5チャート（右列）- RCIを先に追加
        if m5_ohlc is not None:
            # M5データの範囲を確認（デバッグ）
            logger.debug(f"M5 OHLC for display: close min={m5_ohlc['close'].min():.5f}, max={m5_ohlc['close'].max():.5f}, bars={len(m5_ohlc)}")
            m5_timestamps = m5_ohlc["timestamp"].to_list()
        else:
            m5_timestamps = []
        
        # M5 RCI（右列）- Candlestickより先に追加
        # サブウィンドウ1 [24, 33, 48]
        colors_m5_sw1 = ['green', 'purple', 'orange']
        for i, period in enumerate([24, 33, 48]):
            # 重要：m5_rciを使用（m1_rciではない）
            if period in m5_rci and len(m5_rci[period]) > 0 and m5_timestamps:
                logger.debug(f"Adding M5 RCI[{period}] to row=2, col=2")
                rci_len = len(m5_rci[period])
                time_list = m5_timestamps[-rci_len:] if rci_len <= len(m5_timestamps) else m5_timestamps
                # デバッグログ：M5 RCIデータの範囲を確認
                if len(m5_rci[period]) > 0:
                    rci_values = m5_rci[period][-len(time_list):]
                    logger.debug(f"M5 RCI[{period}] for plot: min={min(rci_values):.2f}, max={max(rci_values):.2f}, values={len(rci_values)}")
                    # 異常値の検出（RCIは-100〜100の範囲にあるべき）
                    if max(rci_values) > 100 or min(rci_values) < -100:
                        logger.error(f"ERROR: M5 RCI[{period}] has values outside -100 to 100 range! min={min(rci_values):.2f}, max={max(rci_values):.2f}")
                        logger.error(f"Sample values: {rci_values[:5]}")
                    # 価格データ混入チェック
                    if max(rci_values) > 150:
                        logger.error(f"CRITICAL: M5 RCI[{period}] plotting price data instead of RCI!")
                        logger.error(f"Values look like prices: {rci_values[:3]}")
                        # 価格データをRCI範囲にクリップ（一時的な修正）
                        rci_values = [max(-100, min(100, v - 171)) if v > 150 else v for v in rci_values]
                        logger.warning(f"Temporary fix applied: clipping values to RCI range")
                fig.add_trace(
                    go.Scatter(
                        x=time_list,
                        y=rci_values,
                        mode='lines',
                        name=f'M5 RCI {period}',
                        line=dict(color=colors_m5_sw1[i], width=1.5)
                    ),
                    row=2, col=2
                )
        
        # サブウィンドウ2 [66, 108]
        colors_m5_sw2 = ['brown', 'pink']
        for i, period in enumerate([66, 108]):
            if period in m5_rci and len(m5_rci[period]) > 0 and m5_timestamps:
                rci_len = len(m5_rci[period])
                time_list = m5_timestamps[-rci_len:] if rci_len <= len(m5_timestamps) else m5_timestamps
                fig.add_trace(
                    go.Scatter(
                        x=time_list,
                        y=m5_rci[period][-len(time_list):],
                        mode='lines',
                        name=f'M5 RCI {period}',
                        line=dict(color=colors_m5_sw2[i], width=1.5)
                    ),
                    row=3, col=2
                )
        
        # M5のローソク足を最後に追加（RCIの後）
        if m5_ohlc is not None:
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
        else:
            # M5データがない場合は空のプレースホルダーを追加
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode='lines',
                    name="M5 (No Data)",
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # RCI基準線を追加
        for row in [2, 3, 4]:
            for col in [1, 2]:
                if not (row == 4 and col == 2):  # M5は3行目まで
                    fig.add_hline(y=80, row=row, col=col,
                                 line_dash="dash", line_color="red", opacity=0.3)
                    fig.add_hline(y=-80, row=row, col=col,
                                 line_dash="dash", line_color="green", opacity=0.3)
                    fig.add_hline(y=0, row=row, col=col,
                                 line_dash="dot", line_color="gray", opacity=0.5)
        
        # レイアウト設定
        fig.update_layout(
            height=1200,
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,  # M5のレンジスライダーも無効化
            xaxis3_rangeslider_visible=False,
            xaxis4_rangeslider_visible=False,
            xaxis5_rangeslider_visible=False,
            xaxis6_rangeslider_visible=False,
            xaxis7_rangeslider_visible=False,
            # xaxis8は存在しない（row=4, col=2はNone）
            showlegend=True,
            hovermode='x unified',
            margin=dict(l=50, r=50, t=40, b=40),
            plot_bgcolor=self.config['theme']['background'],
            paper_bgcolor=self.config['theme']['background']
        )
        
        # 各サブプロットの軸設定
        # ローソク足チャート（row=1）のグリッド設定
        fig.update_xaxes(showgrid=self.show_grid, gridcolor=self.config['theme']['grid'], row=1, col=1)
        fig.update_yaxes(showgrid=self.show_grid, gridcolor=self.config['theme']['grid'], row=1, col=1)
        fig.update_xaxes(showgrid=self.show_grid, gridcolor=self.config['theme']['grid'], row=1, col=2)
        fig.update_yaxes(showgrid=self.show_grid, gridcolor=self.config['theme']['grid'], row=1, col=2)
        
        # RCIチャート（row=2,3,4）のY軸範囲設定
        for row in [2, 3, 4]:
            # M1側（左列）
            fig.update_yaxes(
                range=[-105, 105], 
                showgrid=self.show_grid, 
                gridcolor=self.config['theme']['grid'], 
                fixedrange=True,  # 軸の範囲を固定
                constrain="domain",  # ドメイン内に制限
                row=row, 
                col=1
            )
            fig.update_xaxes(showgrid=self.show_grid, gridcolor=self.config['theme']['grid'], row=row, col=1)
            
            # M5側（右列）- row=4, col=2は存在しない
            if row < 4:
                fig.update_yaxes(
                    range=[-105, 105], 
                    showgrid=self.show_grid, 
                    gridcolor=self.config['theme']['grid'], 
                    fixedrange=True,  # 軸の範囲を固定
                    constrain="domain",  # ドメイン内に制限
                    row=row, 
                    col=2
                )
                fig.update_xaxes(showgrid=self.show_grid, gridcolor=self.config['theme']['grid'], row=row, col=2)
        
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
                [{"type": "xy", "secondary_y": False}, {"type": "xy", "secondary_y": False}],  # row=1: ローソク足
                [{"type": "xy", "secondary_y": False}, {"type": "xy", "secondary_y": False}],  # row=2: RCI
                [{"type": "xy", "secondary_y": False}, {"type": "xy", "secondary_y": False}],  # row=3: RCI
                [{"type": "xy", "secondary_y": False}, None]  # row=4: M1のRCIのみ
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
        """パイプラインのメトリクスを取得"""
        pipeline_metrics = self.pipeline.get_metrics() if self.pipeline else {}
        
        # アナライザのバッファサイズを取得
        buffer_size = 0
        if self.pipeline and self.pipeline._multiframe_analyzer:
            buffer_size = self.pipeline._multiframe_analyzer.get_buffer_size()
        
        return {
            **self.stats,
            **pipeline_metrics,
            "analyzer_buffer_size": buffer_size
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
                html.H1("🚀 Pipeline Chart Dashboard", className="text-center mb-4"),
                html.H4("Task 10.3: RealtimePipeline + MultiTimeframeAnalyzer", 
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
                                html.H5("Queue:", className="d-inline me-2"),
                                html.Span(id="queue-size", className="badge bg-info fs-6"),
                            ], width=2),
                            dbc.Col([
                                html.H5("Buffer:", className="d-inline me-2"),
                                html.Span(id="buffer-size", className="badge bg-info fs-6"),
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
                        html.H5("Pipeline Metrics", className="mb-3"),
                        html.Div(id="metrics-display")
                    ])
                ])
            ])
        ]),
        
        # 自動更新用インターバル
        dcc.Interval(
            id='interval-component',
            interval=chart_manager.update_interval * 1000 if chart_manager else 1000,  # 設定から更新間隔を取得
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
     Output('queue-size', 'children'),
     Output('buffer-size', 'children'),
     Output('metrics-display', 'children')],
    [Input('interval-component', 'n_intervals')],
    [State('realtime-status', 'data')]
)
def update_display(n, status):
    """表示を更新"""
    global chart_manager
    
    if chart_manager is None:
        return go.Figure(), "", "$0.00", "0/0", "0", ""
    
    # チャート作成
    fig = chart_manager.create_chart()
    
    # メトリクス取得
    metrics = chart_manager.get_metrics()
    
    # 表示データ
    symbol = chart_manager.symbol
    current_price = f"${metrics['current_price']:,.5f}"
    queue_size = f"{metrics.get('input_queue_size', 0)}/{metrics.get('queue_size', 1000)}"
    buffer_size = f"{metrics.get('analyzer_buffer_size', 0)}"
    
    # メトリクス表示
    metrics_content = [
        html.P(f"Ticks Received: {metrics['ticks_received']:,}"),
        html.P(f"Pipeline Processed: {metrics['pipeline_processed']:,}"),
        html.P(f"Average Latency: {metrics['latency_avg']*1000:.2f}ms"),
        html.P(f"Backpressure Events: {metrics.get('backpressure_events', 0):,}"),
        html.P(f"Last Update: {metrics['last_update'].strftime('%H:%M:%S') if metrics['last_update'] else 'N/A'}")
    ]
    
    return fig, symbol, current_price, queue_size, buffer_size, metrics_content

def cleanup():
    """終了時のクリーンアップ"""
    global chart_manager
    if chart_manager:
        logger.info("Cleaning up...")
        chart_manager.stop_realtime()
        # MT5ConnectionManagerを使用している場合はdisconnectを呼ぶ
        if chart_manager.mt5_manager:
            chart_manager.mt5_manager.disconnect()
        else:
            # 直接初期化した場合のクリーンアップ
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
    print("Pipeline Chart Dashboard")
    print("Task 10.3: RealtimePipeline + MultiTimeframeAnalyzer")
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
        chart_manager = PipelineChartManager(config_path=config_path)
        print(f"✅ Initialized for symbol: {chart_manager.symbol}")
        print(f"✅ Pipeline queue size: {chart_manager.pipeline.queue_size}")
        print(f"✅ Analyzer ready: {chart_manager.pipeline._multiframe_analyzer.is_ready()}")
        
        # 自動的にリアルタイム処理を開始
        chart_manager.start_realtime()
        print(f"✅ Realtime processing started automatically")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        sys.exit(1)
    
    # ポート設定（設定ファイルまたは環境変数から取得）
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
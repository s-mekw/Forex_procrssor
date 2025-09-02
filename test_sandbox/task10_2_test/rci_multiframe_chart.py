"""
RCI Multi-Timeframe Chart with Dash
Task 10.2: マルチタイムフレームRCI分析チャート
M1とM5のタイムフレームを並列表示し、異なる期間のRCIを可視化
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from typing import Dict, List, Optional, Tuple
import threading
import queue
import json
import time
import socket
import signal
import atexit
import os
from threading import Lock
from dataclasses import dataclass, field

# プロジェクトのインポート
from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.data_processing.rci import RCIProcessor, DifferentialRCICalculator
from src.mt5_data_acquisition.tick_to_bar import TickToBarConverter, Bar
from src.common.models import Tick as CommonTick
# from utils.config_loader import load_config  # 現在は使用しない

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

@dataclass
class TimeframeData:
    """タイムフレームごとのデータ管理クラス"""
    timeframe: str
    ohlc_data: Optional[pl.DataFrame] = None
    rci_data: Dict[int, List[Optional[float]]] = field(default_factory=dict)
    rci_calculators: Dict[int, DifferentialRCICalculator] = field(default_factory=dict)
    converter: Optional[TickToBarConverter] = None
    current_bar: Optional[Bar] = None
    temp_rci_values: Dict[int, float] = field(default_factory=dict)
    has_incomplete_bar: bool = False
    
    # RCI期間設定
    rci_periods_subwindow1: List[int] = field(default_factory=list)
    rci_periods_subwindow2: List[int] = field(default_factory=list)
    rci_periods_subwindow3: List[int] = field(default_factory=list)
    
    @property
    def all_rci_periods(self) -> List[int]:
        """すべてのRCI期間を取得"""
        return list(set(
            self.rci_periods_subwindow1 + 
            self.rci_periods_subwindow2 + 
            self.rci_periods_subwindow3
        ))

class RCIMultiframeChart:
    """マルチタイムフレームRCIチャートクラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        # 設定読み込み（現在は常にデフォルト設定を使用）
        self.config = self._get_default_config()
        
        # MT5初期化
        self.mt5_manager = None
        
        # RCIProcessor初期化
        self.rci_processor = RCIProcessor(use_float32=True)
        
        # データ管理（スレッドセーフ用ロック付き）
        self.data_lock = Lock()
        
        # M1データ管理
        self.m1_data = TimeframeData(
            timeframe="M1",
            rci_periods_subwindow1=[9, 13],
            rci_periods_subwindow2=[24, 33, 48],
            rci_periods_subwindow3=[66, 108]
        )
        
        # M5データ管理
        self.m5_data = TimeframeData(
            timeframe="M5",
            rci_periods_subwindow1=[24, 33, 48],
            rci_periods_subwindow2=[66, 108]
        )
        
        # タイムフレーム初期化
        self._initialize_timeframes()
        
        # M1からM5への変換用バッファ
        self.m1_to_m5_buffer = []  # M1バーを5本蓄積してM5バーを生成
        
        # 統計情報
        self.stats = {
            "ticks_received": 0,
            "bars_completed_m1": 0,
            "bars_completed_m5": 0,
            "start_time": None,
            "last_update": None,
            "current_price": 0
        }
        
        # スレッド管理
        self.tick_thread = None
        self.is_running = False
        
        # MT5初期化
        self.initialize_mt5()
    
    def _get_default_config(self):
        """デフォルト設定を生成"""
        from types import SimpleNamespace
        return SimpleNamespace(
            chart=SimpleNamespace(
                symbol="EURJPY#",
                initial_bars=200,
                max_bars_m1=10000,  # M1の最大保存バー数
                max_bars_m5=2000,   # M5の最大保存バー数
                update_interval=1.0,
                show_grid=True,
                display_bars_m1=100,  # M1チャート表示バー数
                display_bars_m5=100   # M5チャート表示バー数
            ),
            mt5=SimpleNamespace(
                timeout=60000,
                max_retries=3,
                retry_delay=1.0
            ),
            rci=SimpleNamespace(
                use_float32=True,
                levels=SimpleNamespace(
                    overbought=80,
                    oversold=-80,
                    zero_line=0
                )
            ),
            dash=SimpleNamespace(
                host="0.0.0.0",
                port=8052,
                debug=False
            ),
            theme=SimpleNamespace(
                background="#fffbea",
                grid="#e0e0e0",
                text="#000000"
            )
        )
    
    def _initialize_timeframes(self):
        """タイムフレームごとの初期化"""
        # M1初期化
        self.m1_data.converter = TickToBarConverter(
            symbol=self.config.chart.symbol,
            timeframe=TIMEFRAME_TO_SECONDS["M1"]
        )
        
        # M1のRCICalculator初期化
        for period in self.m1_data.all_rci_periods:
            self.m1_data.rci_calculators[period] = DifferentialRCICalculator(period)
        
        # M5初期化
        self.m5_data.converter = TickToBarConverter(
            symbol=self.config.chart.symbol,
            timeframe=TIMEFRAME_TO_SECONDS["M5"]
        )
        
        # M5のRCICalculator初期化
        for period in self.m5_data.all_rci_periods:
            self.m5_data.rci_calculators[period] = DifferentialRCICalculator(period)
    
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
        # M1データ取得
        rates_m1 = mt5.copy_rates_from_pos(
            self.config.chart.symbol,
            mt5.TIMEFRAME_M1,
            0,
            self.config.chart.initial_bars
        )
        
        if rates_m1 is None or len(rates_m1) == 0:
            raise ValueError("Failed to fetch M1 initial data")
        
        self.m1_data.ohlc_data = pl.DataFrame({
            "time": [datetime.fromtimestamp(r['time']) for r in rates_m1],
            "open": np.array([r['open'] for r in rates_m1], dtype=np.float32),
            "high": np.array([r['high'] for r in rates_m1], dtype=np.float32),
            "low": np.array([r['low'] for r in rates_m1], dtype=np.float32),
            "close": np.array([r['close'] for r in rates_m1], dtype=np.float32),
            "volume": np.array([r['tick_volume'] for r in rates_m1], dtype=np.float32)
        })
        
        # M5データ取得
        # 設定ファイルからM5の初期バー数を取得（デフォルト120本）
        m5_bars_needed = getattr(self.config.chart, 'initial_bars_m5', 120)
        # M5のRCI最大期間が108なので、最低120本は必要
        m5_bars_needed = max(120, m5_bars_needed)
        rates_m5 = mt5.copy_rates_from_pos(
            self.config.chart.symbol,
            mt5.TIMEFRAME_M5,
            0,
            m5_bars_needed
        )
        
        if rates_m5 is not None and len(rates_m5) > 0:
            self.m5_data.ohlc_data = pl.DataFrame({
                "time": [datetime.fromtimestamp(r['time']) for r in rates_m5],
                "open": np.array([r['open'] for r in rates_m5], dtype=np.float32),
                "high": np.array([r['high'] for r in rates_m5], dtype=np.float32),
                "low": np.array([r['low'] for r in rates_m5], dtype=np.float32),
                "close": np.array([r['close'] for r in rates_m5], dtype=np.float32),
                "volume": np.array([r['tick_volume'] for r in rates_m5], dtype=np.float32)
            })
        
        # RCI計算
        self.calculate_rci(self.m1_data)
        if self.m5_data.ohlc_data is not None:
            self.calculate_rci(self.m5_data)
    
    def calculate_rci(self, tf_data: TimeframeData):
        """RCIを計算"""
        if tf_data.ohlc_data is None or tf_data.ohlc_data.is_empty():
            return
        
        # 一時RCI値をクリア
        tf_data.temp_rci_values.clear()
        
        # 初期データをcalculatorに流し込んでRCI計算
        close_prices = tf_data.ohlc_data["close"].to_list()
        
        # 各期間のRCIデータを初期化
        for period in tf_data.all_rci_periods:
            tf_data.rci_data[period] = []
            
            # calculator をリセット
            tf_data.rci_calculators[period].reset()
            
            # 最後のバーを除いて価格データを追加
            for i, price in enumerate(close_prices[:-1]):
                rci_value = tf_data.rci_calculators[period].add(float(price))
                tf_data.rci_data[period].append(rci_value)
            
            # 最後のバー（未完成バー）のRCIをpreviewで計算
            if len(close_prices) > 0:
                last_price = float(close_prices[-1])
                preview_rci = tf_data.rci_calculators[period].preview(last_price)
                tf_data.rci_data[period].append(preview_rci)
                
                if preview_rci is not None:
                    tf_data.temp_rci_values[period] = preview_rci
        
        tf_data.has_incomplete_bar = True
    
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
                    
                    with self.data_lock:
                        # M1バー処理
                        bar_m1 = self.m1_data.converter.add_tick(tick_obj)
                        if bar_m1:
                            self.add_new_bar_m1(bar_m1)
                            self.stats["bars_completed_m1"] += 1
                            
                            # M1バーをバッファに追加してM5バー生成を試みる
                            self.process_m1_to_m5_conversion(bar_m1)
                        
                        # M5バー処理（独立したティック処理）
                        bar_m5 = self.m5_data.converter.add_tick(tick_obj)
                        if bar_m5:
                            self.add_new_bar_m5(bar_m5)
                            self.stats["bars_completed_m5"] += 1
                        
                        # 現在のバーを更新
                        self.m1_data.current_bar = self.m1_data.converter.get_current_bar()
                        self.m5_data.current_bar = self.m5_data.converter.get_current_bar()
                        
                        if self.m1_data.current_bar:
                            self.stats["current_price"] = float(self.m1_data.current_bar.close)
                            
                            # 未完成バー更新
                            self.update_current_bar_in_ohlc(self.m1_data)
                            self.update_rci_incremental(self.m1_data)
                        
                        if self.m5_data.current_bar:
                            self.update_current_bar_in_ohlc(self.m5_data)
                            self.update_rci_incremental(self.m5_data)
                        
                        self.stats["ticks_received"] += 1
                        self.stats["last_update"] = tick_time
                    
                    last_tick_time = tick_time
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Tick receiver error: {e}")
                time.sleep(1)
    
    def process_m1_to_m5_conversion(self, bar_m1: Bar):
        """M1バーからM5バーへの変換処理"""
        # バッファにM1バーを追加
        self.m1_to_m5_buffer.append(bar_m1)
        
        # 5本のM1バーが揃ったらM5バーを生成
        if len(self.m1_to_m5_buffer) >= 5:
            # 最初の5本からM5バーを生成
            m1_bars = self.m1_to_m5_buffer[:5]
            
            # M5バーのOHLCVを計算
            m5_time = m1_bars[0].time  # 最初のM1バーの時刻
            m5_open = m1_bars[0].open
            m5_high = max(bar.high for bar in m1_bars)
            m5_low = min(bar.low for bar in m1_bars)
            m5_close = m1_bars[-1].close
            m5_volume = sum(bar.volume for bar in m1_bars)
            
            # 使用したM1バーをバッファから削除
            self.m1_to_m5_buffer = self.m1_to_m5_buffer[5:]
            
            # M5バーとして追加（手動生成）
            # 注：これは補助的なもので、実際のM5バーはMT5からも取得される
            # ここでは学習用のデモンストレーションとして実装
    
    def add_new_bar_m1(self, bar: Bar):
        """M1新規バーを追加"""
        self.add_new_bar(self.m1_data, bar)
    
    def add_new_bar_m5(self, bar: Bar):
        """M5新規バーを追加"""
        self.add_new_bar(self.m5_data, bar)
    
    def add_new_bar(self, tf_data: TimeframeData, bar: Bar):
        """新しいバーを追加（タイムフレーム共通）"""
        new_row = pl.DataFrame({
            "time": [bar.time],
            "open": [np.float32(bar.open)],
            "high": [np.float32(bar.high)],
            "low": [np.float32(bar.low)],
            "close": [np.float32(bar.close)],
            "volume": [np.float32(bar.volume)]
        })
        
        was_replacement = False
        
        if tf_data.ohlc_data is not None and len(tf_data.ohlc_data) > 0:
            last_time = tf_data.ohlc_data["time"][-1]
            if last_time == bar.time:
                # 同じ時刻のバーなら置き換え
                tf_data.ohlc_data = pl.concat([tf_data.ohlc_data[:-1], new_row])
                was_replacement = True
            else:
                # 新しい時刻のバーなら追加
                tf_data.ohlc_data = pl.concat([tf_data.ohlc_data, new_row])
        else:
            tf_data.ohlc_data = new_row
        
        # RCI増分更新
        new_close = float(bar.close)
        
        for period in tf_data.all_rci_periods:
            rci_value = tf_data.rci_calculators[period].add(new_close)
            
            if period not in tf_data.rci_data:
                tf_data.rci_data[period] = []
            
            if was_replacement and len(tf_data.rci_data[period]) > 0:
                tf_data.rci_data[period][-1] = rci_value
            else:
                tf_data.rci_data[period].append(rci_value)
        
        # メモリ管理
        # M1は10000本、M5は2000本まで保存
        if tf_data.timeframe == "M1":
            max_bars = getattr(self.config.chart, 'max_bars_m1', 10000)
        else:  # M5
            max_bars = getattr(self.config.chart, 'max_bars_m5', 2000)
        
        if len(tf_data.ohlc_data) > max_bars:
            tf_data.ohlc_data = tf_data.ohlc_data.tail(max_bars)
            for period in tf_data.all_rci_periods:
                if period in tf_data.rci_data and len(tf_data.rci_data[period]) > max_bars:
                    tf_data.rci_data[period] = tf_data.rci_data[period][-max_bars:]
    
    def update_current_bar_in_ohlc(self, tf_data: TimeframeData):
        """未完成バーをOHLCデータの最後のロウに反映"""
        if tf_data.current_bar is None or tf_data.ohlc_data is None or tf_data.ohlc_data.is_empty():
            return
        
        try:
            current_bar_time = tf_data.current_bar.time
            data_length = len(tf_data.ohlc_data)
            
            if data_length > 0:
                last_time = tf_data.ohlc_data["time"][-1]
                
                if last_time == current_bar_time:
                    # 既存の未完成バーを更新
                    if data_length > 1:
                        preceding_data = tf_data.ohlc_data[:-1]
                        updated_row = pl.DataFrame({
                            "time": [current_bar_time],
                            "open": [np.float32(tf_data.current_bar.open)],
                            "high": [np.float32(tf_data.current_bar.high)],
                            "low": [np.float32(tf_data.current_bar.low)],
                            "close": [np.float32(tf_data.current_bar.close)],
                            "volume": [np.float32(tf_data.current_bar.volume)]
                        })
                        tf_data.ohlc_data = pl.concat([preceding_data, updated_row])
                else:
                    # 新しい未完成バーを追加
                    new_row = pl.DataFrame({
                        "time": [current_bar_time],
                        "open": [np.float32(tf_data.current_bar.open)],
                        "high": [np.float32(tf_data.current_bar.high)],
                        "low": [np.float32(tf_data.current_bar.low)],
                        "close": [np.float32(tf_data.current_bar.close)],
                        "volume": [np.float32(tf_data.current_bar.volume)]
                    })
                    tf_data.ohlc_data = pl.concat([tf_data.ohlc_data, new_row])
                    
                    # 新しい未完成バーのRCIデータも追加
                    for period in tf_data.all_rci_periods:
                        if period in tf_data.rci_calculators:
                            preview_rci = tf_data.rci_calculators[period].preview(float(tf_data.current_bar.close))
                            if period not in tf_data.rci_data:
                                tf_data.rci_data[period] = []
                            tf_data.rci_data[period].append(preview_rci)
                    
                    tf_data.has_incomplete_bar = True
        
        except Exception as e:
            print(f"Error updating current bar in OHLC: {e}")
    
    def update_rci_incremental(self, tf_data: TimeframeData):
        """RCIを増分更新"""
        if tf_data.current_bar is None:
            return
        
        current_close = float(tf_data.current_bar.close)
        
        for period in tf_data.all_rci_periods:
            if period in tf_data.rci_calculators:
                temp_rci = tf_data.rci_calculators[period].preview(current_close)
                if temp_rci is not None:
                    tf_data.temp_rci_values[period] = temp_rci
                    
                    if tf_data.has_incomplete_bar and len(tf_data.rci_data[period]) > 0:
                        tf_data.rci_data[period][-1] = temp_rci
    
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
        """マルチタイムフレームチャートを作成"""
        with self.data_lock:
            if self.m1_data.ohlc_data is None or self.m1_data.ohlc_data.is_empty():
                return self._create_empty_chart()
            
            try:
                # データのコピーを作成
                m1_ohlc = self.m1_data.ohlc_data.clone()
                m1_rci = {k: v.copy() for k, v in self.m1_data.rci_data.items()}
                
                m5_ohlc = None
                m5_rci = {}
                if self.m5_data.ohlc_data is not None:
                    m5_ohlc = self.m5_data.ohlc_data.clone()
                    m5_rci = {k: v.copy() for k, v in self.m5_data.rci_data.items()}
                
                # 表示バー数制限を適用
                display_bars_m1 = getattr(self.config.chart, 'display_bars_m1', 100)
                display_bars_m5 = getattr(self.config.chart, 'display_bars_m5', 100)
                
                # M1データの表示制限
                if m1_ohlc is not None:
                    m1_ohlc = m1_ohlc.tail(display_bars_m1)
                    # RCIデータも同じ期間に制限
                    for period in m1_rci:
                        if len(m1_rci[period]) > display_bars_m1:
                            m1_rci[period] = m1_rci[period][-display_bars_m1:]
                
                # M5データの表示制限
                if m5_ohlc is not None:
                    m5_ohlc = m5_ohlc.tail(display_bars_m5)
                    # RCIデータも同じ期間に制限
                    for period in m5_rci:
                        if len(m5_rci[period]) > display_bars_m5:
                            m5_rci[period] = m5_rci[period][-display_bars_m5:]
                
            except Exception as e:
                print(f"Error preparing chart data: {e}")
                return self._create_empty_chart()
        
        # サブプロットの作成（2列×4行）
        # 左列：M1、右列：M5
        # specs を明示的に設定して、各サブプロットのタイプを定義
        fig = make_subplots(
            rows=4, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.03,  # 垂直スペースを増やして分離を強化
            horizontal_spacing=0.05,
            row_heights=[0.4, 0.2, 0.2, 0.2],  # メイン40%、RCI各20%
            column_widths=[0.5, 0.5],  # 均等分割
            subplot_titles=(
                f"M1 - {self.config.chart.symbol}", f"M5 - {self.config.chart.symbol}",
                "M1 RCI [9, 13]", "M5 RCI [24, 33, 48]",
                "M1 RCI [24, 33, 48]", "M5 RCI [66, 108]",
                "M1 RCI [66, 108]", ""  # M5は2つのサブウィンドウのみ
            ),
            specs=[
                [{"type": "xy", "secondary_y": False}, {"type": "xy", "secondary_y": False}],  # Row 1: ローソク足用
                [{"type": "xy", "secondary_y": False}, {"type": "xy", "secondary_y": False}],  # Row 2: RCI用
                [{"type": "xy", "secondary_y": False}, {"type": "xy", "secondary_y": False}],  # Row 3: RCI用
                [{"type": "xy", "secondary_y": False}, {"type": "xy", "secondary_y": False}]   # Row 4: RCI用
            ]
        )
        
        # M1チャート（左列）
        if m1_ohlc is not None:
            # ローソク足
            fig.add_trace(
                go.Candlestick(
                    x=m1_ohlc["time"].to_list(),
                    open=m1_ohlc["open"].to_list(),
                    high=m1_ohlc["high"].to_list(),
                    low=m1_ohlc["low"].to_list(),
                    close=m1_ohlc["close"].to_list(),
                    name="M1 OHLC",
                    showlegend=False,
                    increasing_line_color='black',
                    increasing_fillcolor='white',
                    decreasing_line_color='black',
                    decreasing_fillcolor='black',
                    zorder=1  # レイヤー順序を低く設定
                ),
                row=1, col=1
            )
            
            # M1 RCI サブウィンドウ1 [9, 13]
            for period in self.m1_data.rci_periods_subwindow1:
                if period in m1_rci and len(m1_rci[period]) > 0:
                    color = 'blue' if period == 9 else 'red'
                    fig.add_trace(
                        go.Scatter(
                            x=m1_ohlc["time"].to_list()[:len(m1_rci[period])],
                            y=m1_rci[period],
                            mode='lines',
                            name=f'M1 RCI {period}',
                            line=dict(color=color, width=1.5),
                            showlegend=True
                        ),
                        row=2, col=1
                    )
            
            # M1 RCI サブウィンドウ2 [24, 33, 48]
            colors_sw2 = ['green', 'purple', 'orange']
            for i, period in enumerate(self.m1_data.rci_periods_subwindow2):
                if period in m1_rci and len(m1_rci[period]) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=m1_ohlc["time"].to_list()[:len(m1_rci[period])],
                            y=m1_rci[period],
                            mode='lines',
                            name=f'M1 RCI {period}',
                            line=dict(color=colors_sw2[i % len(colors_sw2)], width=1.5),
                            showlegend=True
                        ),
                        row=3, col=1
                    )
            
            # M1 RCI サブウィンドウ3 [66, 108]
            colors_sw3 = ['brown', 'pink']
            for i, period in enumerate(self.m1_data.rci_periods_subwindow3):
                if period in m1_rci and len(m1_rci[period]) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=m1_ohlc["time"].to_list()[:len(m1_rci[period])],
                            y=m1_rci[period],
                            mode='lines',
                            name=f'M1 RCI {period}',
                            line=dict(color=colors_sw3[i % len(colors_sw3)], width=1.5),
                            showlegend=True
                        ),
                        row=4, col=1
                    )
        
        # M5チャート（右列）
        if m5_ohlc is not None:
            # M5 RCI サブウィンドウ1 [24, 33, 48]
            colors_m5_sw1 = ['green', 'purple', 'orange']
            for i, period in enumerate(self.m5_data.rci_periods_subwindow1):
                if period in m5_rci and len(m5_rci[period]) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=m5_ohlc["time"].to_list()[:len(m5_rci[period])],
                            y=m5_rci[period],
                            mode='lines',
                            name=f'M5 RCI {period}',
                            line=dict(color=colors_m5_sw1[i % len(colors_m5_sw1)], width=1.5),
                            showlegend=True,
                            zorder=2  # レイヤー順序を高く設定
                        ),
                        row=2, col=2
                    )
            
            # M5 RCI サブウィンドウ2 [66, 108]
            colors_m5_sw2 = ['brown', 'pink']
            for i, period in enumerate(self.m5_data.rci_periods_subwindow2):
                if period in m5_rci and len(m5_rci[period]) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=m5_ohlc["time"].to_list()[:len(m5_rci[period])],
                            y=m5_rci[period],
                            mode='lines',
                            name=f'M5 RCI {period}',
                            line=dict(color=colors_m5_sw2[i % len(colors_m5_sw2)], width=1.5),
                            showlegend=True
                        ),
                        row=3, col=2
                    )
            
            # M5のローソク足を最後に追加（RCIの後）
            fig.add_trace(
                go.Candlestick(
                    x=m5_ohlc["time"].to_list(),
                    open=m5_ohlc["open"].to_list(),
                    high=m5_ohlc["high"].to_list(),
                    low=m5_ohlc["low"].to_list(),
                    close=m5_ohlc["close"].to_list(),
                    name="M5 OHLC",
                    showlegend=False,
                    increasing_line_color='black',
                    increasing_fillcolor='white',
                    decreasing_line_color='black',
                    decreasing_fillcolor='black',
                    zorder=1  # レイヤー順序を低く設定
                ),
                row=1, col=2
            )
        
        # RCI基準線を追加（すべてのRCIサブウィンドウ）
        # M1のRCIサブウィンドウ（行2,3,4の列1）
        for row in [2, 3, 4]:
            self._add_rci_reference_lines(fig, row, 1)
        
        # M5のRCIサブウィンドウ（行2,3の列2）
        for row in [2, 3]:
            self._add_rci_reference_lines(fig, row, 2)
        
        # レイアウト設定
        fig.update_layout(
            height=1200,  # 高さを増やして4行に対応
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,  # M5のレンジスライダーも無効化
            xaxis3_rangeslider_visible=False,
            xaxis4_rangeslider_visible=False,
            xaxis5_rangeslider_visible=False,
            xaxis6_rangeslider_visible=False,
            xaxis7_rangeslider_visible=False,
            xaxis8_rangeslider_visible=False,
            showlegend=True,
            hovermode='x unified',
            margin=dict(l=50, r=50, t=40, b=40),  # マージンを調整
            plot_bgcolor=self.config.theme.background,
            paper_bgcolor=self.config.theme.background,
            font=dict(color=self.config.theme.text)
        )
        
        # 軸の設定
        fig.update_xaxes(title_text="Time", row=4, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=2)  # M5は3行まで
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=2)
        
        # RCI軸の範囲設定と固定
        for row in [2, 3, 4]:
            fig.update_yaxes(
                title_text="RCI", 
                row=row, 
                col=1, 
                range=[-105, 105],
                fixedrange=True,  # 軸の範囲を固定
                constrain="domain"  # ドメイン内に制限
            )
        for row in [2, 3]:
            fig.update_yaxes(
                title_text="RCI", 
                row=row, 
                col=2, 
                range=[-105, 105],
                fixedrange=True,  # 軸の範囲を固定
                constrain="domain"  # ドメイン内に制限
            )
        
        # グリッドの設定
        if self.config.chart.show_grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=self.config.theme.grid)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=self.config.theme.grid)
        
        return fig
    
    def _add_rci_reference_lines(self, fig, row, col):
        """RCI基準線を追加"""
        # ±100境界線
        fig.add_hline(y=100, row=row, col=col,
                     line_dash="solid", line_color="black", line_width=1)
        fig.add_hline(y=-100, row=row, col=col,
                     line_dash="solid", line_color="black", line_width=1)
        # 買われすぎ・売られすぎライン
        fig.add_hline(y=self.config.rci.levels.overbought, row=row, col=col,
                     line_dash="dash", line_color="red", opacity=0.3)
        fig.add_hline(y=self.config.rci.levels.oversold, row=row, col=col,
                     line_dash="dash", line_color="green", opacity=0.3)
        fig.add_hline(y=self.config.rci.levels.zero_line, row=row, col=col,
                     line_dash="dot", line_color="gray", opacity=0.5)
    
    def _create_empty_chart(self):
        """空のチャートを作成"""
        fig = make_subplots(
            rows=4, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.02,
            horizontal_spacing=0.05,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            column_widths=[0.5, 0.5],
            subplot_titles=(
                "M1 (Loading...)", "M5 (Loading...)",
                "", "", "", "", "", ""
            )
        )
        
        fig.update_layout(
            height=1200,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            hovermode='x unified',
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig

# グローバルインスタンス
chart_manager = None

# Dashアプリケーションの初期化
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
            html.H1("📊 RCI Multi-Timeframe Chart Dashboard", className="text-center mb-4"),
            html.H4("Task 10.2: M1 & M5 Parallel Analysis", className="text-center text-muted mb-4"),
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
                    html.H5("Statistics", className="mb-3"),
                    html.Div(id="stats-display")
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
    
    # データストア
    dcc.Store(id='realtime-status', data={'is_running': True})
    
], fluid=True)

# レイアウトを関数として設定
app.layout = serve_layout

# コールバック: スタートボタンとステータス表示
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
    """リアルタイム更新の開始/停止"""
    global chart_manager
    
    current_status = status_state if status_state else {'is_running': True}
    
    ctx = callback_context
    if not ctx.triggered:
        is_running = current_status.get('is_running', True)
        if is_running:
            return current_status, "🟢 LIVE", "badge bg-success fs-6 ms-3"
        else:
            return current_status, "🔴 STOPPED", "badge bg-danger fs-6 ms-3"
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-button' and chart_manager:
        chart_manager.start_realtime()
        return {'is_running': True}, "🟢 LIVE", "badge bg-success fs-6 ms-3"
    elif button_id == 'stop-button' and chart_manager:
        chart_manager.stop_realtime()
        return {'is_running': False}, "🔴 STOPPED", "badge bg-danger fs-6 ms-3"
    
    is_running = current_status.get('is_running', True)
    if is_running:
        return current_status, "🟢 LIVE", "badge bg-success fs-6 ms-3"
    else:
        return current_status, "🔴 STOPPED", "badge bg-danger fs-6 ms-3"

# コールバック: チャート更新
@app.callback(
    [Output('live-chart', 'figure'),
     Output('symbol-display', 'children'),
     Output('current-price', 'children'),
     Output('m1-bars', 'children'),
     Output('m5-bars', 'children'),
     Output('stats-display', 'children')],
    [Input('interval-component', 'n_intervals')],
    [State('realtime-status', 'data')]
)
def update_chart(n, status):
    """チャートと統計情報を更新"""
    global chart_manager
    
    if chart_manager is None:
        return go.Figure(), "", "$0.00", "0", "0", ""
    
    # チャート作成
    fig = chart_manager.create_chart()
    
    # 統計情報
    symbol = chart_manager.config.chart.symbol
    current_price = f"${chart_manager.stats['current_price']:,.5f}"
    m1_bars = f"{chart_manager.stats['bars_completed_m1']:,}"
    m5_bars = f"{chart_manager.stats['bars_completed_m5']:,}"
    
    # 統計表示
    stats_content = [
        html.P(f"Ticks Received: {chart_manager.stats['ticks_received']:,}"),
        html.P(f"Last Update: {chart_manager.stats['last_update'].strftime('%H:%M:%S') if chart_manager.stats['last_update'] else 'N/A'}")
    ]
    
    return fig, symbol, current_price, m1_bars, m5_bars, stats_content

def find_available_port(start_port=8052, max_attempts=10):
    """利用可能なポートを見つける"""
    for i in range(max_attempts):
        port = start_port + i
        try:
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
    
    print("Initializing RCI Multi-Timeframe Chart...")
    print("Task 10.2: Testing multi-timeframe RCI analysis")
    print(f"Process ID: {os.getpid()}")
    
    # 設定ファイルのパスを確認
    config_path = Path(__file__).parent / "task10_2_config.toml"
    if config_path.exists():
        print(f"Loading config from: {config_path}")
        chart_manager = RCIMultiframeChart(str(config_path))
    else:
        print("Using default configuration")
        chart_manager = RCIMultiframeChart()
    
    print(f"Symbol: {chart_manager.config.chart.symbol}")
    print("Timeframes: M1, M5")
    print(f"M1 RCI periods: {chart_manager.m1_data.all_rci_periods}")
    print(f"M5 RCI periods: {chart_manager.m5_data.all_rci_periods}")
    
    # リアルタイム更新を自動開始
    print("\n🚀 Starting real-time data feed...")
    chart_manager.start_realtime()
    print("✅ Real-time data feed started")
    
    # Dash設定
    host = chart_manager.config.dash.host
    default_port = chart_manager.config.dash.port
    debug = chart_manager.config.dash.debug
    
    # ポート取得
    port = int(os.environ.get('DASH_PORT', default_port))
    
    try:
        available_port = find_available_port(port)
        if available_port != port:
            print(f"⚠️  Port {port} is in use, using port {available_port} instead")
            port = available_port
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        print("Please close other Dash applications or specify a different port")
        sys.exit(1)
    
    print(f"\n✅ Starting Dash server on http://{host}:{port}")
    print("📊 Open your browser to view the multi-timeframe RCI chart")
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
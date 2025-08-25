"""
RCI Real-time Chart with Dash
RCIインジケーターを含むリアルタイムチャートのWebアプリケーション
ローソク足チャート + EMA + 2つのRCIサブウィンドウ
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
from src.data_processing.rci import RCIProcessor
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

class RCIRealtimeChart:
    """RCIを含むリアルタイムチャートクラス"""
    
    def __init__(self, config=None):
        """初期化"""
        # TOMLから設定を読み込み
        self.config = config if config else load_config(preset="full")
        
        # MT5とデータ処理の初期化
        self.mt5_manager = None
        self.indicator_engine = TechnicalIndicatorEngine(
            ema_periods=self.config.chart.ema_periods
        )
        
        # RCIProcessor初期化
        self.rci_processor = RCIProcessor(use_float32=self.config.rci.use_float32)
        
        # DifferentialRCICalculator初期化（ストリーミング用）
        from src.data_processing.rci import DifferentialRCICalculator
        self.rci_calculators = {}  # 各期間のDifferentialRCICalculatorインスタンス
        for period in self.config.all_rci_periods:
            self.rci_calculators[period] = DifferentialRCICalculator(period)
        
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
        self.rci_data = {}  # RCIデータを保持（完成バーのみ）
        self.temp_ema_values = {}  # 未完成バー用の一時EMA値
        self.temp_rci_values = {}  # 未完成バー用の一時RCI値
        self.current_bar = None
        self.tick_queue = queue.Queue()
        
        # 統計情報
        self.stats = {
            "ticks_received": 0,
            "bars_completed": 0,
            "start_time": None,
            "last_update": None,
            "current_price": 0,
            "ema_values": {},
            "rci_values": {}
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
        
        # RCI計算
        self.calculate_rci()
        
    def calculate_ema(self):
        """EMAを計算（完成バーのみ対象）"""
        if self.ohlc_data is None or self.ohlc_data.is_empty():
            return
        
        # 一時EMA値をクリア
        self.temp_ema_values.clear()
        
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
    
    def calculate_rci(self):
        """RCIを計算（初期化時のみ使用、ストリーミング方式）"""
        if self.ohlc_data is None or self.ohlc_data.is_empty():
            return
        
        # 一時RCI値をクリア
        self.temp_rci_values.clear()
        
        # 初期データをcalculatorに流し込んでRCI計算
        close_prices = self.ohlc_data["close"].to_list()
        
        # 各期間のRCIデータを初期化
        for period in self.config.all_rci_periods:
            self.rci_data[period] = []
            
            # calculator をリセット（初期化時のみ）
            self.rci_calculators[period].reset()
            
            # 価格データを順次追加してRCI計算
            for price in close_prices:
                rci_value = self.rci_calculators[period].add(float(price))
                # None（ウィンドウが満たない）または実際のRCI値を保存
                self.rci_data[period].append(rci_value)
            
            # 最後の有効なRCI値を統計情報に保存
            if self.rci_data[period]:
                last_value = self.rci_data[period][-1]
                if last_value is not None:
                    self.stats["rci_values"][period] = last_value
            
            # デバッグ情報（必要に応じてコメントアウト）
            # print(f"Period {period}: Initialized with {len(close_prices)} prices, "
            #       f"RCI values count: {len(self.rci_data[period])}")
    
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
                            
                            # 未完成バー用のEMA一時値を更新
                            self.update_ema_incremental(float(self.current_bar.close))
                            
                            # 未完成バー用のRCI一時値を更新
                            self.update_rci_incremental()
                        
                        self.stats["ticks_received"] += 1
                        self.stats["last_update"] = tick_time
                    
                    last_tick_time = tick_time
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Tick receiver error: {e}")
                time.sleep(1)
    
    def add_new_bar(self, bar: Bar):
        """新しいバーを追加（完成バー用）"""
        new_row = pl.DataFrame({
            "time": [bar.time],
            "open": [np.float32(bar.open)],
            "high": [np.float32(bar.high)],
            "low": [np.float32(bar.low)],
            "close": [np.float32(bar.close)],
            "volume": [np.float32(bar.volume)]
        })
        
        self.ohlc_data = pl.concat([self.ohlc_data, new_row])
        
        # RCI増分更新（新しいバーのcloseだけをcalculatorに追加）
        new_close = float(bar.close)
        for period in self.config.all_rci_periods:
            # 新しい価格をcalculatorに追加してRCI計算
            rci_value = self.rci_calculators[period].add(new_close)
            
            # RCIデータリストが存在しない場合は初期化
            if period not in self.rci_data:
                self.rci_data[period] = []
            
            # 新しいRCI値を追加
            self.rci_data[period].append(rci_value)
            
            # 統計情報を更新
            if rci_value is not None:
                self.stats["rci_values"][period] = rci_value
                # デバッグ情報（コメントアウト可能）
                # print(f"New bar - Period {period}: RCI={rci_value:.2f}, "
                #       f"Close={new_close:.5f}, Total bars: {len(self.ohlc_data)}")
        
        # メモリ管理
        max_bars = self.config.chart.initial_bars * 2
        if len(self.ohlc_data) > max_bars:
            self.ohlc_data = self.ohlc_data.tail(max_bars)
            # EMAデータも同じ長さに調整
            for period in self.config.chart.ema_periods:
                if period in self.ema_data and len(self.ema_data[period]) > max_bars:
                    self.ema_data[period] = self.ema_data[period][-max_bars:]
            # RCIデータも同じ長さに調整
            for period in self.config.all_rci_periods:
                if period in self.rci_data and len(self.rci_data[period]) > max_bars:
                    self.rci_data[period] = self.rci_data[period][-max_bars:]
        
        # EMA再計算（EMAは全体再計算が必要）
        self.calculate_ema()
        
        # データ同期を確認・修復
        self.sync_data_integrity()
    
    def update_current_bar_in_ohlc(self):
        """未完成バーをOHLCデータの最後のロウに反映"""
        if self.current_bar is None or self.ohlc_data is None or self.ohlc_data.is_empty():
            return
        
        try:
            current_bar_time = self.current_bar.time
            data_length = len(self.ohlc_data)
            
            if data_length > 0:
                # 最後のバーの時刻を取得
                last_time = self.ohlc_data["time"][-1]
                
                if last_time == current_bar_time:
                    # 既存の未完成バーを更新
                    if data_length > 1:
                        preceding_data = self.ohlc_data[:-1]
                        updated_row = pl.DataFrame({
                            "time": [current_bar_time],
                            "open": [np.float32(self.current_bar.open)],
                            "high": [np.float32(self.current_bar.high)],
                            "low": [np.float32(self.current_bar.low)],
                            "close": [np.float32(self.current_bar.close)],
                            "volume": [np.float32(self.current_bar.volume)]
                        })
                        self.ohlc_data = pl.concat([preceding_data, updated_row])
                    else:
                        # 最初のバーの場合
                        self.ohlc_data = pl.DataFrame({
                            "time": [current_bar_time],
                            "open": [np.float32(self.current_bar.open)],
                            "high": [np.float32(self.current_bar.high)],
                            "low": [np.float32(self.current_bar.low)],
                            "close": [np.float32(self.current_bar.close)],
                            "volume": [np.float32(self.current_bar.volume)]
                        })
                else:
                    # 新しい未完成バーを追加
                    new_row = pl.DataFrame({
                        "time": [current_bar_time],
                        "open": [np.float32(self.current_bar.open)],
                        "high": [np.float32(self.current_bar.high)],
                        "low": [np.float32(self.current_bar.low)],
                        "close": [np.float32(self.current_bar.close)],
                        "volume": [np.float32(self.current_bar.volume)]
                    })
                    self.ohlc_data = pl.concat([self.ohlc_data, new_row])
            else:
                # OHLCデータが空の場合
                self.ohlc_data = pl.DataFrame({
                    "time": [current_bar_time],
                    "open": [np.float32(self.current_bar.open)],
                    "high": [np.float32(self.current_bar.high)],
                    "low": [np.float32(self.current_bar.low)],
                    "close": [np.float32(self.current_bar.close)],
                    "volume": [np.float32(self.current_bar.volume)]
                })
            
        except Exception as e:
            print(f"Error updating current bar in OHLC: {e}")
    
    def sync_data_integrity(self):
        """データの整合性をチェックし、必要に応じて修復"""
        if self.ohlc_data is None or self.ohlc_data.is_empty():
            return
        
        try:
            ohlc_length = len(self.ohlc_data)
            
            # 各EMA期間のデータ長をチェック
            for period in self.config.chart.ema_periods:
                if period in self.ema_data:
                    ema_length = len(self.ema_data[period])
                    
                    if ema_length > ohlc_length:
                        # EMAデータが長すぎる場合はトリム
                        self.ema_data[period] = self.ema_data[period][:ohlc_length]
                    elif ema_length < ohlc_length - 1:  # -1は未完成バーを考慮
                        # EMAデータが短すぎる場合はEMAを再計算
                        self.calculate_ema()
                        break
            
            # 各RCI期間のデータ長をチェック
            for period in self.config.all_rci_periods:
                if period in self.rci_data:
                    rci_length = len(self.rci_data[period])
                    
                    if rci_length > ohlc_length:
                        # RCIデータが長すぎる場合はトリム
                        self.rci_data[period] = self.rci_data[period][:ohlc_length]
                    elif rci_length < ohlc_length - 1:  # -1は未完成バーを考慮
                        # RCIデータが短すぎる場合は、Noneで埋める（再計算はしない）
                        # calculatorの状態を保持するため
                        missing = ohlc_length - rci_length
                        self.rci_data[period].extend([None] * missing)
                        print(f"Warning: RCI data sync - Period {period}: Added {missing} None values")
                        
        except Exception as e:
            print(f"Error in data sync: {e}")
            # エラー時はEMAのみ再計算（RCIはcalculatorの状態を保持）
            self.calculate_ema()
    
    def update_ema_incremental(self, new_price: float):
        """EMAを増分更新（未完成バー用の一時値を計算）"""
        # 未完成バー用の一時EMA値を計算
        for period in self.config.chart.ema_periods:
            if period in self.ema_data and len(self.ema_data[period]) > 0:
                alpha = 2.0 / (period + 1)
                prev_ema = self.ema_data[period][-1]
                new_ema = alpha * new_price + (1 - alpha) * prev_ema
                # 既存のEMAデータは変更せず、一時値として保存
                self.temp_ema_values[period] = new_ema
                self.stats["ema_values"][period] = new_ema
    
    def update_rci_incremental(self):
        """RCIを増分更新（未完成バー用の一時値を計算）"""
        if self.current_bar is None:
            return
        
        # 未完成バーの価格で一時的なRCI値を計算
        current_close = float(self.current_bar.close)
        
        for period in self.config.all_rci_periods:
            # DifferentialRCICalculatorのpreviewメソッドを使用
            if period in self.rci_calculators:
                temp_rci = self.rci_calculators[period].preview(current_close)
                if temp_rci is not None:
                    self.temp_rci_values[period] = temp_rci
                    self.stats["rci_values"][period] = temp_rci
                    # デバッグ情報
                    # print(f"Preview - Period {period}: RCI={temp_rci:.2f} for price={current_close:.5f}")
    
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
        """Plotlyチャートを作成（スレッドセーフ・RCI対応）"""
        with self.data_lock:
            if self.ohlc_data is None or self.ohlc_data.is_empty():
                return self._create_empty_chart()
            
            try:
                # データのコピーを作成してロックを早めに解放
                ohlc_copy = self.ohlc_data.clone()
                ema_copy = {k: v.copy() for k, v in self.ema_data.items()}
                rci_copy = {k: v.copy() for k, v in self.rci_data.items()}
                temp_ema_copy = self.temp_ema_values.copy()
                temp_rci_copy = self.temp_rci_values.copy()
                
                # データ検証と整合性チェック
                ohlc_copy, ema_copy, rci_copy = self._validate_and_sync_data(
                    ohlc_copy, ema_copy, rci_copy, temp_ema_copy, temp_rci_copy
                )
                
            except Exception as e:
                print(f"Error preparing chart data: {e}")
                return self._create_empty_chart()
        
        # ロック外でチャートを作成
        # サブプロットの作成（3行：メインチャート、RCI短期、RCI長期）
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],  # メイン50%、RCI各25%
            subplot_titles=(
                f"{self.config.chart.symbol} - {self.config.chart.timeframe}",
                f"RCI Short ({', '.join(map(str, self.config.rci.periods_short))})",
                f"RCI Long ({', '.join(map(str, self.config.rci.periods_long))})"
            )
        )
        
        # ローソク足チャート（コピーしたデータを使用）
        fig.add_trace(
            go.Candlestick(
                x=ohlc_copy["time"].to_list(),
                open=ohlc_copy["open"].to_list(),
                high=ohlc_copy["high"].to_list(),
                low=ohlc_copy["low"].to_list(),
                close=ohlc_copy["close"].to_list(),
                name="OHLC",
                increasing_line_color='black',  # 陽線の枠線
                increasing_fillcolor='white',   # 陽線の塗りつぶし
                decreasing_line_color='black',  # 陰線の枠線
                decreasing_fillcolor='black'    # 陰線の塗りつぶし
            ),
            row=1, col=1
        )
        
        # EMAライン（一時値を含む結合データを使用）
        colors = ['orange', 'blue', 'green', 'red', 'purple']
        for idx, period in enumerate(self.config.chart.ema_periods):
            if period in ema_copy and len(ema_copy[period]) > 0:
                # 検証済みのEMAデータを使用
                ema_length = min(len(ema_copy[period]), len(ohlc_copy))
                time_data = ohlc_copy["time"].to_list()[:ema_length]
                ema_data_values = ema_copy[period][:ema_length]
                
                fig.add_trace(
                    go.Scatter(
                        x=time_data,
                        y=ema_data_values,
                        mode='lines',
                        name=f'EMA {period}',
                        line=dict(color=colors[idx % len(colors)], width=1)
                    ),
                    row=1, col=1
                )
        
        # RCI短期（サブウィンドウ1）
        for period in self.config.rci.periods_short:
            if period in rci_copy and len(rci_copy[period]) > 0:
                rci_length = min(len(rci_copy[period]), len(ohlc_copy))
                time_data = ohlc_copy["time"].to_list()[:rci_length]
                rci_data_values = rci_copy[period][:rci_length]
                
                color = self.config.rci.colors.get(period, 'gray')
                fig.add_trace(
                    go.Scatter(
                        x=time_data,
                        y=rci_data_values,
                        mode='lines',
                        name=f'RCI {period}',
                        line=dict(color=color, width=1.5),
                        line_shape='spline'  # 曲線を滑らかに
                    ),
                    row=2, col=1
                )
        
        # RCI長期（サブウィンドウ2）
        for period in self.config.rci.periods_long:
            if period in rci_copy and len(rci_copy[period]) > 0:
                rci_length = min(len(rci_copy[period]), len(ohlc_copy))
                time_data = ohlc_copy["time"].to_list()[:rci_length]
                rci_data_values = rci_copy[period][:rci_length]
                
                color = self.config.rci.colors.get(period, 'gray')
                fig.add_trace(
                    go.Scatter(
                        x=time_data,
                        y=rci_data_values,
                        mode='lines',
                        name=f'RCI {period}',
                        line=dict(color=color, width=1.5),
                        line_shape='spline'  # 曲線を滑らかに
                    ),
                    row=3, col=1
                )
        
        # RCI基準線を追加（買われすぎ、売られすぎ、ゼロライン、±100境界線）
        time_range = ohlc_copy["time"].to_list()
        if time_range:
            # 短期RCIパネル
            # ±100境界線
            fig.add_hline(y=100, row=2, col=1,
                         line_dash="solid", line_color="black", line_width=1)
            fig.add_hline(y=-100, row=2, col=1,
                         line_dash="solid", line_color="black", line_width=1)
            # 買われすぎ・売られすぎライン
            fig.add_hline(y=self.config.rci.levels['overbought'], row=2, col=1,
                         line_dash="dash", line_color="red", opacity=0.3)
            fig.add_hline(y=self.config.rci.levels['oversold'], row=2, col=1,
                         line_dash="dash", line_color="green", opacity=0.3)
            fig.add_hline(y=self.config.rci.levels['zero_line'], row=2, col=1,
                         line_dash="dot", line_color="gray", opacity=0.5)
            
            # 長期RCIパネル
            # ±100境界線
            fig.add_hline(y=100, row=3, col=1,
                         line_dash="solid", line_color="black", line_width=1)
            fig.add_hline(y=-100, row=3, col=1,
                         line_dash="solid", line_color="black", line_width=1)
            # 買われすぎ・売られすぎライン
            fig.add_hline(y=self.config.rci.levels['overbought'], row=3, col=1,
                         line_dash="dash", line_color="red", opacity=0.3)
            fig.add_hline(y=self.config.rci.levels['oversold'], row=3, col=1,
                         line_dash="dash", line_color="green", opacity=0.3)
            fig.add_hline(y=self.config.rci.levels['zero_line'], row=3, col=1,
                         line_dash="dot", line_color="gray", opacity=0.5)
        
        # レイアウト設定
        fig.update_layout(
            height=900,  # 高さを増やして3つのパネルに対応
            xaxis_rangeslider_visible=False,
            showlegend=True,
            hovermode='x unified',
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor=self.config.theme.get('background', '#fffbea'),
            paper_bgcolor=self.config.theme.get('background', '#fffbea'),
            font=dict(color=self.config.theme.get('text', '#000000'))
        )
        
        # 軸の設定
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RCI", row=2, col=1, range=[-105, 105])  # 境界線表示のため範囲拡張
        fig.update_yaxes(title_text="RCI", row=3, col=1, range=[-105, 105])  # 境界線表示のため範囲拡張
        
        # グリッドの設定
        if self.config.chart.show_grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=self.config.theme.get('grid', '#e0e0e0'))
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=self.config.theme.get('grid', '#e0e0e0'))
        
        return fig
    
    def _validate_and_sync_data(self, ohlc_data, ema_data, rci_data, temp_ema_values, temp_rci_values):
        """チャート用データの検証と同期（RCI対応）"""
        try:
            # OHLCデータの基本検証
            if ohlc_data.is_empty():
                return ohlc_data, ema_data, rci_data
            
            # 数値データのNaN・Infチェック
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in ohlc_data.columns:
                    # NaNやInfを除去
                    ohlc_data = ohlc_data.filter(pl.col(col).is_finite())
            
            if ohlc_data.is_empty():
                return ohlc_data, ema_data, rci_data
            
            ohlc_length = len(ohlc_data)
            
            # EMAデータと一時EMA値を結合・検証
            for period in self.config.chart.ema_periods:
                if period in ema_data:
                    # 現在のEMAデータの長さをチェック
                    ema_length = len(ema_data[period])
                    
                    # 一時EMA値を追加（未完成バー用）
                    if period in temp_ema_values and ohlc_length > ema_length:
                        ema_data[period] = ema_data[period] + [temp_ema_values[period]]
                    
                    # EMAデータの長さをOHLCに合わせる
                    if len(ema_data[period]) > ohlc_length:
                        ema_data[period] = ema_data[period][:ohlc_length]
                    elif len(ema_data[period]) < ohlc_length:
                        # EMAデータが不足の場合は最後の値で埋める
                        if len(ema_data[period]) > 0:
                            last_ema = ema_data[period][-1]
                            missing_count = ohlc_length - len(ema_data[period])
                            ema_data[period].extend([last_ema] * missing_count)
                        else:
                            # EMAデータが空の場合は除去
                            del ema_data[period]
                    
                    # NaNやInfのチェックと除去
                    if period in ema_data:
                        ema_data[period] = [x for x in ema_data[period] if x is not None and not np.isnan(x) and np.isfinite(x)]
                        if len(ema_data[period]) == 0:
                            del ema_data[period]
            
            # RCIデータと一時RCI値を結合・検証
            for period in self.config.all_rci_periods:
                if period in rci_data:
                    # 現在のRCIデータの長さをチェック
                    rci_length = len(rci_data[period])
                    
                    # 一時RCI値を追加（未完成バー用）
                    if period in temp_rci_values and ohlc_length > rci_length:
                        rci_data[period] = rci_data[period] + [temp_rci_values[period]]
                    
                    # RCIデータの長さをOHLCに合わせる
                    if len(rci_data[period]) > ohlc_length:
                        rci_data[period] = rci_data[period][:ohlc_length]
                    elif len(rci_data[period]) < ohlc_length:
                        # RCIデータが不足の場合
                        # RCIはNoneを許容するので、Noneで埋める
                        missing_count = ohlc_length - len(rci_data[period])
                        rci_data[period].extend([None] * missing_count)
                    
                    # 有効な値のみを保持（NaNやInfは除去、Noneは保持）
                    if period in rci_data:
                        cleaned_rci = []
                        for x in rci_data[period]:
                            if x is None:
                                cleaned_rci.append(None)
                            elif not np.isnan(x) and np.isfinite(x):
                                cleaned_rci.append(x)
                            else:
                                cleaned_rci.append(None)
                        rci_data[period] = cleaned_rci
            
            return ohlc_data, ema_data, rci_data
            
        except Exception as e:
            print(f"Error in data validation: {e}")
            return ohlc_data, ema_data, rci_data
    
    def _create_empty_chart(self):
        """空のチャートを作成"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(
                f"{self.config.chart.symbol} - {self.config.chart.timeframe} (Loading...)",
                "RCI Short (Loading...)",
                "RCI Long (Loading...)"
            )
        )
        
        fig.update_layout(
            height=900,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            hovermode='x unified',
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
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
            html.H1("📈 RCI Real-time Forex Chart Dashboard", className="text-center mb-4"),
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
                        ], width=4),
                    ])
                ])
            ], className="mb-3")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="live-chart", style={"height": "900px"})
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
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("RCI Values", className="mb-3"),
                    html.Div(id="rci-values")
                ])
            ])
        ], width=6)
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
     Output('rci-values', 'children'),
     Output('last-update', 'children')],
    [Input('interval-component', 'n_intervals')],
    [State('realtime-status', 'data')]
)
def update_chart(n, status):
    """チャートと統計情報を更新"""
    global chart_manager
    
    if chart_manager is None:
        return go.Figure(), "", "$0.00", "0", "", "", "N/A"
    
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
    
    # RCI値の表示
    rci_badges = []
    for period, value in chart_manager.stats.get("rci_values", {}).items():
        color = "bg-danger" if value > 80 else "bg-success" if value < -80 else "bg-secondary"
        rci_badges.append(
            html.Span(f"RCI{period}: {value:.1f}", 
                     className=f"badge {color} me-2 fs-6")
        )
    
    # 最終更新時刻
    last_update = chart_manager.stats.get('last_update', None)
    if last_update:
        last_update_str = last_update.strftime("%H:%M:%S")
    else:
        last_update_str = "Waiting..."
    
    return fig, symbol, current_price, tick_count, ema_badges, rci_badges, last_update_str

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
    
    print("Initializing RCI Real-time Chart...")
    print(f"Process ID: {os.getpid()}")
    
    # TOMLファイルから設定読み込み
    config = load_config(preset="full")
    
    # チャートマネージャー初期化
    chart_manager = RCIRealtimeChart(config)
    
    print(f"Symbol: {config.chart.symbol}")
    print(f"Timeframe: {config.chart.timeframe}")
    print(f"EMA periods: {config.chart.ema_periods}")
    print(f"RCI periods: {config.all_rci_periods}")
    
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
    print("📊 Open your browser to view the RCI real-time chart")
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
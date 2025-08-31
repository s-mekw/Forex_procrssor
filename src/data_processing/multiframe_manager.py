"""
マルチタイムフレーム管理モジュール

各タイムフレームのOHLCデータをMT5から直接取得し、
リアルタイムティックとの整合性を保ちながら管理します。

主な機能:
- 複数タイムフレーム（1M, 5M, 15M, 30M, 1H, 4H, D1）の同時管理
- 形成中バー[0]のティックベース更新
- 完成バー[1]のMT5 API経由取得
- MT5データとの完全な整合性保証
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import MetaTrader5 as mt5
import polars as pl
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TimeframeData:
    """
    各タイムフレームのデータ管理クラス
    
    Attributes:
        timeframe: タイムフレーム識別子（"M1", "M5", "M15"など）
        mt5_timeframe: MT5のタイムフレーム定数
        interval_seconds: タイムフレームの秒数
        last_bar_time: 最後に完成したバーのタイムスタンプ
        current_bar: 形成中のバー[0]のOHLC
        completed_bars: 完成済みバーのDataFrame
        max_bars: 保持する最大バー数
        on_bar_complete: バー完成時のコールバック
    """
    timeframe: str
    mt5_timeframe: int
    interval_seconds: int
    last_bar_time: Optional[datetime] = None
    current_bar: Dict[str, Any] = field(default_factory=dict)
    completed_bars: Optional[pl.DataFrame] = None
    max_bars: int = 5000
    on_bar_complete: Optional[Callable] = None
    
    def __post_init__(self):
        """初期化後の処理"""
        if self.completed_bars is None:
            # 空のDataFrameを作成
            self.completed_bars = pl.DataFrame({
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": []
            })
    
    def add_completed_bar(self, bar_data: Dict[str, Any]):
        """完成したバーを追加"""
        new_bar = pl.DataFrame([bar_data])
        if self.completed_bars is not None and not self.completed_bars.is_empty():
            self.completed_bars = pl.concat([self.completed_bars, new_bar])
        else:
            self.completed_bars = new_bar
        
        # 最大バー数を超えた場合は古いバーを削除
        if len(self.completed_bars) > self.max_bars:
            self.completed_bars = self.completed_bars.tail(self.max_bars)
        
        # コールバック実行
        if self.on_bar_complete:
            self.on_bar_complete(self.timeframe, bar_data)
    
    def update_current_bar(self, tick_price: float, tick_volume: float = 1.0, 
                          tick_time: Optional[datetime] = None):
        """現在のバー[0]を更新"""
        if not self.current_bar:
            # 新しいバーの開始
            self.current_bar = {
                "open": tick_price,
                "high": tick_price,
                "low": tick_price,
                "close": tick_price,
                "volume": tick_volume,
                "start_time": tick_time or datetime.now()
            }
        else:
            # 既存バーの更新
            self.current_bar["high"] = max(self.current_bar["high"], tick_price)
            self.current_bar["low"] = min(self.current_bar["low"], tick_price)
            self.current_bar["close"] = tick_price
            self.current_bar["volume"] += tick_volume
    
    def reset_current_bar(self):
        """現在のバーをリセット"""
        self.current_bar = {}


class MultiTimeframeManager:
    """
    マルチタイムフレーム統合管理クラス
    
    複数のタイムフレームを同時に管理し、ティック更新とMT5データ取得を統合します。
    """
    
    # MT5タイムフレームマッピング
    TIMEFRAME_MAP = {
        "M1": (mt5.TIMEFRAME_M1, 60),
        "M5": (mt5.TIMEFRAME_M5, 300),
        "M15": (mt5.TIMEFRAME_M15, 900),
        "M30": (mt5.TIMEFRAME_M30, 1800),
        "H1": (mt5.TIMEFRAME_H1, 3600),
        "H4": (mt5.TIMEFRAME_H4, 14400),
        "D1": (mt5.TIMEFRAME_D1, 86400),
    }
    
    def __init__(self, symbol: str, timeframes: List[str], 
                 initial_bars: int = 200, max_bars: int = 5000):
        """
        初期化
        
        Args:
            symbol: 取引シンボル
            timeframes: 管理するタイムフレームのリスト（["M1", "M5", "M15"]など）
            initial_bars: 初期化時に取得するバー数
            max_bars: 各タイムフレームで保持する最大バー数
        """
        self.symbol = symbol
        self.timeframes: Dict[str, TimeframeData] = {}
        self.initial_bars = initial_bars
        self.max_bars = max_bars
        self.is_initialized = False
        
        # タイムフレームデータを初期化
        for tf in timeframes:
            if tf not in self.TIMEFRAME_MAP:
                logger.warning(f"Unsupported timeframe: {tf}")
                continue
            
            mt5_tf, interval_seconds = self.TIMEFRAME_MAP[tf]
            self.timeframes[tf] = TimeframeData(
                timeframe=tf,
                mt5_timeframe=mt5_tf,
                interval_seconds=interval_seconds,
                max_bars=max_bars
            )
        
        logger.info(f"MultiTimeframeManager initialized for {symbol} with timeframes: {timeframes}")
    
    def initialize_data(self) -> bool:
        """
        各タイムフレームの初期データをMT5から取得
        
        Returns:
            成功した場合True
        """
        try:
            for tf_name, tf_data in self.timeframes.items():
                # MT5から初期データを取得
                rates = mt5.copy_rates_from_pos(
                    self.symbol,
                    tf_data.mt5_timeframe,
                    0,
                    self.initial_bars
                )
                
                if rates is None or len(rates) == 0:
                    logger.error(f"Failed to fetch initial data for {tf_name}")
                    continue
                
                # DataFrameに変換
                df = pl.DataFrame({
                    "timestamp": [datetime.fromtimestamp(r['time']) for r in rates],
                    "open": np.array([r['open'] for r in rates], dtype=np.float32),
                    "high": np.array([r['high'] for r in rates], dtype=np.float32),
                    "low": np.array([r['low'] for r in rates], dtype=np.float32),
                    "close": np.array([r['close'] for r in rates], dtype=np.float32),
                    "volume": np.array([r['tick_volume'] for r in rates], dtype=np.float32)
                })
                
                tf_data.completed_bars = df
                tf_data.last_bar_time = df["timestamp"][-1] if not df.is_empty() else None
                
                logger.info(f"Initialized {tf_name} with {len(df)} bars")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize data: {e}")
            return False
    
    def process_tick(self, tick: Any) -> Dict[str, Dict[str, Any]]:
        """
        ティックを処理して全タイムフレームを更新
        
        Args:
            tick: MT5のティックオブジェクト（bid, ask, time属性を持つ）
        
        Returns:
            各タイムフレームの更新結果
        """
        if not self.is_initialized:
            logger.warning("Manager not initialized. Call initialize_data() first.")
            return {}
        
        results = {}
        tick_time = datetime.fromtimestamp(tick.time) if hasattr(tick, 'time') else datetime.now()
        tick_price = float(tick.bid) if hasattr(tick, 'bid') else float(tick.last)
        tick_volume = float(tick.volume) if hasattr(tick, 'volume') else 1.0
        
        for tf_name, tf_data in self.timeframes.items():
            # バー完成チェック
            new_bar_started = self._check_new_bar(tick_time, tf_data)
            
            if new_bar_started:
                # MT5から完成したバー[1]を取得
                completed_bar = self._finalize_bar_from_mt5(tf_data)
                if completed_bar:
                    results[tf_name] = {
                        "new_bar": True,
                        "completed_bar": completed_bar,
                        "timestamp": tick_time
                    }
                
                # 現在のバーをリセット
                tf_data.reset_current_bar()
            
            # 現在のバー[0]を更新
            tf_data.update_current_bar(tick_price, tick_volume, tick_time)
            
            # 結果に現在のバー情報を追加
            if tf_name not in results:
                results[tf_name] = {
                    "new_bar": False,
                    "current_bar": tf_data.current_bar.copy(),
                    "timestamp": tick_time
                }
            else:
                results[tf_name]["current_bar"] = tf_data.current_bar.copy()
        
        return results
    
    def _check_new_bar(self, current_time: datetime, tf_data: TimeframeData) -> bool:
        """
        新しいバーが開始したかチェック
        
        Args:
            current_time: 現在時刻
            tf_data: タイムフレームデータ
        
        Returns:
            新しいバーが開始した場合True
        """
        if tf_data.last_bar_time is None:
            # 初回
            tf_data.last_bar_time = self._get_bar_start_time(current_time, tf_data.interval_seconds)
            logger.debug(f"[{tf_data.timeframe}] Initial bar time set to: {tf_data.last_bar_time}")
            return False
        
        # 現在のバー開始時刻を計算
        current_bar_time = self._get_bar_start_time(current_time, tf_data.interval_seconds)
        
        # 前回のバー時刻と比較
        if current_bar_time > tf_data.last_bar_time:
            logger.info(f"✅ [{tf_data.timeframe}] New bar detected! "
                       f"Previous: {tf_data.last_bar_time}, Current: {current_bar_time}, "
                       f"Tick time: {current_time}")
            tf_data.last_bar_time = current_bar_time
            return True
        
        return False
    
    def _get_bar_start_time(self, time: datetime, interval_seconds: int) -> datetime:
        """
        指定時刻が属するバーの開始時刻を計算
        
        Args:
            time: 対象時刻
            interval_seconds: バーの間隔（秒）
        
        Returns:
            バーの開始時刻
        """
        timestamp = int(time.timestamp())
        bar_start_timestamp = (timestamp // interval_seconds) * interval_seconds
        return datetime.fromtimestamp(bar_start_timestamp)
    
    def _finalize_bar_from_mt5(self, tf_data: TimeframeData) -> Optional[Dict[str, Any]]:
        """
        MT5から完成したバー[1]のデータを取得
        
        Args:
            tf_data: タイムフレームデータ
        
        Returns:
            完成したバーのデータ、取得失敗時はNone
        """
        try:
            # MT5から最新の完成バー[1]を取得
            rates = mt5.copy_rates_from_pos(
                self.symbol,
                tf_data.mt5_timeframe,
                1,  # [1]の位置（完成したバー）
                1   # 1本取得
            )
            
            if rates is None or len(rates) == 0:
                logger.warning(f"Failed to fetch completed bar for {tf_data.timeframe}")
                return None
            
            # バーデータを辞書形式に変換
            bar = rates[0]
            bar_data = {
                "timestamp": datetime.fromtimestamp(bar['time']),
                "open": float(bar['open']),
                "high": float(bar['high']),
                "low": float(bar['low']),
                "close": float(bar['close']),
                "volume": float(bar['tick_volume'])
            }
            
            # 完成バーとして追加
            tf_data.add_completed_bar(bar_data)
            
            logger.debug(f"{tf_data.timeframe} bar completed: {bar_data['timestamp']} "
                        f"OHLC=[{bar_data['open']:.5f}, {bar_data['high']:.5f}, "
                        f"{bar_data['low']:.5f}, {bar_data['close']:.5f}]")
            
            return bar_data
            
        except Exception as e:
            logger.error(f"Error fetching completed bar for {tf_data.timeframe}: {e}")
            return None
    
    def get_timeframe_data(self, timeframe: str) -> Optional[TimeframeData]:
        """
        指定タイムフレームのデータを取得
        
        Args:
            timeframe: タイムフレーム（"M1", "M5"など）
        
        Returns:
            TimeframeDataオブジェクト、存在しない場合None
        """
        return self.timeframes.get(timeframe)
    
    def get_completed_bars(self, timeframe: str, limit: Optional[int] = None) -> Optional[pl.DataFrame]:
        """
        指定タイムフレームの完成済みバーを取得
        
        Args:
            timeframe: タイムフレーム
            limit: 取得する最大バー数
        
        Returns:
            完成済みバーのDataFrame
        """
        tf_data = self.get_timeframe_data(timeframe)
        if tf_data is None:
            return None
        
        if limit and tf_data.completed_bars is not None:
            return tf_data.completed_bars.tail(limit)
        
        return tf_data.completed_bars
    
    def get_current_bar(self, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        指定タイムフレームの現在形成中のバー[0]を取得
        
        Args:
            timeframe: タイムフレーム
        
        Returns:
            現在のバーデータ
        """
        tf_data = self.get_timeframe_data(timeframe)
        if tf_data is None:
            return None
        
        return tf_data.current_bar.copy()
    
    def get_all_current_bars(self) -> Dict[str, Dict[str, Any]]:
        """
        全タイムフレームの現在形成中のバーを取得
        
        Returns:
            タイムフレーム名をキーとした現在バーの辞書
        """
        return {
            tf_name: tf_data.current_bar.copy()
            for tf_name, tf_data in self.timeframes.items()
            if tf_data.current_bar
        }
    
    def set_bar_complete_callback(self, timeframe: str, callback: Callable):
        """
        バー完成時のコールバックを設定
        
        Args:
            timeframe: タイムフレーム
            callback: コールバック関数
        """
        tf_data = self.get_timeframe_data(timeframe)
        if tf_data:
            tf_data.on_bar_complete = callback
            logger.info(f"Callback set for {timeframe} bar completion")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        管理メトリクスを取得
        
        Returns:
            各タイムフレームのバー数などのメトリクス
        """
        metrics = {
            "symbol": self.symbol,
            "is_initialized": self.is_initialized,
            "timeframes": {}
        }
        
        for tf_name, tf_data in self.timeframes.items():
            metrics["timeframes"][tf_name] = {
                "completed_bars": len(tf_data.completed_bars) if tf_data.completed_bars is not None else 0,
                "has_current_bar": bool(tf_data.current_bar),
                "last_bar_time": tf_data.last_bar_time.isoformat() if tf_data.last_bar_time else None
            }
        
        return metrics
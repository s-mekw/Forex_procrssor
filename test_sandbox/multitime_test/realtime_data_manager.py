"""
Realtime Data Manager Module
リアルタイムデータ管理機能を提供するモジュール
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import logging
from datetime import datetime
import threading
import time
from typing import Dict, Any, Optional
import MetaTrader5 as mt5
import polars as pl

from src.data_processing.multiframe_manager import MultiTimeframeManager

logger = logging.getLogger(__name__)


class RealtimeDataManager:
    """リアルタイムデータ管理クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.symbol = config['trading']['symbol']
        self.timeframes = config['trading']['timeframes']
        
        # MultiTimeframeManager
        self.manager = None
        
        # データ格納
        self.m1_data = None
        self.m5_data = None
        self.m1_current_bar = None
        self.m5_current_bar = None
        self.current_price = 0.0
        self.last_update = datetime.now()
        
        # スレッド制御
        self.is_running = False
        self.tick_thread = None
        
        logger.info(f"RealtimeDataManager initialized for {self.symbol}")
    
    def connect_mt5(self) -> bool:
        """
        MT5に接続
        
        Returns:
            接続成功の場合True
        """
        try:
            mt5_config = self.config['mt5']
            
            # MT5初期化
            if not mt5.initialize(
                path=mt5_config['path'],
                login=mt5_config['account'],
                password=mt5_config['password'],
                server=mt5_config['server'],
                timeout=mt5_config['timeout']
            ):
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            logger.info(f"MT5 connected: Account {mt5_config['account']}")
            
            # シンボル確認
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"Symbol {self.symbol} not found")
                return False
            
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    logger.error(f"Failed to select symbol {self.symbol}")
                    return False
            
            logger.info(f"Symbol {self.symbol} selected")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def initialize_manager(self) -> bool:
        """
        MultiTimeframeManagerを初期化
        
        Returns:
            初期化成功の場合True
        """
        try:
            # マネージャー作成
            self.manager = MultiTimeframeManager(
                symbol=self.symbol,
                timeframes=self.timeframes,
                initial_bars=self.config['chart']['initial_bars'],
                max_bars=self.config['chart']['max_bars']
            )
            
            # 初期データ取得
            if not self.manager.initialize_data():
                logger.error("Failed to initialize manager data")
                return False
            
            # 初期データを保存
            self.m1_data = self.manager.get_completed_bars("M1")
            self.m5_data = self.manager.get_completed_bars("M5")
            self.m1_current_bar = self.manager.get_current_bar("M1")
            self.m5_current_bar = self.manager.get_current_bar("M5")
            
            # 現在価格を取得
            tick = mt5.symbol_info_tick(self.symbol)
            if tick:
                self.current_price = tick.bid
            
            logger.info("Manager initialized successfully")
            logger.info(f"M1 bars: {len(self.m1_data) if self.m1_data is not None else 0}")
            logger.info(f"M5 bars: {len(self.m5_data) if self.m5_data is not None else 0}")
            logger.info(f"M1 current bar: {self.m1_current_bar is not None}")
            logger.info(f"M5 current bar: {self.m5_current_bar is not None}")
            
            return True
            
        except Exception as e:
            logger.error(f"Manager initialization error: {e}")
            return False
    
    def tick_receiver_loop(self):
        """ティック受信ループ（別スレッド）"""
        logger.info("Tick receiver started")
        
        tick_count = 0
        m1_bars_completed = 0
        m5_bars_completed = 0
        
        while self.is_running:
            try:
                # ティック取得
                tick = mt5.symbol_info_tick(self.symbol)
                if tick is None:
                    time.sleep(0.1)
                    continue
                
                # ティック処理
                results = self.manager.process_tick(tick)
                tick_count += 1
                
                # 現在価格更新
                self.current_price = tick.bid
                self.last_update = datetime.now()
                
                # 新しいバーが完成したかチェック
                for tf_name, tf_result in results.items():
                    if tf_result.get("new_bar"):
                        if tf_name == "M1":
                            m1_bars_completed += 1
                            logger.info(f"✅ M1 bar completed #{m1_bars_completed} at {tf_result['timestamp']}")
                        elif tf_name == "M5":
                            m5_bars_completed += 1
                            logger.info(f"✅ M5 bar completed #{m5_bars_completed} at {tf_result['timestamp']}")
                
                # 最新データを取得
                self.m1_data = self.manager.get_completed_bars("M1")
                self.m5_data = self.manager.get_completed_bars("M5")
                self.m1_current_bar = self.manager.get_current_bar("M1")
                self.m5_current_bar = self.manager.get_current_bar("M5")
                
                # 10ティックごとにログ
                if tick_count % 10 == 0:
                    logger.debug(f"Processed {tick_count} ticks, M1 bars: {m1_bars_completed}, M5 bars: {m5_bars_completed}")
                    if tick_count % 50 == 0:  # 50ティックごとに現在のバー情報をログ
                        if self.m1_current_bar:
                            logger.info(f"M1 current bar - O:{self.m1_current_bar['open']:.2f} H:{self.m1_current_bar['high']:.2f} L:{self.m1_current_bar['low']:.2f} C:{self.m1_current_bar['close']:.2f}")
                        if self.m5_current_bar:
                            logger.info(f"M5 current bar - O:{self.m5_current_bar['open']:.2f} H:{self.m5_current_bar['high']:.2f} L:{self.m5_current_bar['low']:.2f} C:{self.m5_current_bar['close']:.2f}")
                
                time.sleep(0.1)  # CPU負荷軽減
                
            except Exception as e:
                logger.error(f"Tick receiver error: {e}")
                time.sleep(1)
        
        logger.info(f"Tick receiver stopped. Total ticks: {tick_count}, M1 bars: {m1_bars_completed}, M5 bars: {m5_bars_completed}")
    
    def start_tick_receiver(self):
        """ティック受信を開始"""
        if not self.is_running:
            self.is_running = True
            self.tick_thread = threading.Thread(target=self.tick_receiver_loop)
            self.tick_thread.daemon = True
            self.tick_thread.start()
            logger.info("Tick receiver thread started")
    
    def stop_tick_receiver(self):
        """ティック受信を停止"""
        if self.is_running:
            self.is_running = False
            if self.tick_thread:
                self.tick_thread.join(timeout=2)
            logger.info("Tick receiver stopped")
    
    def get_chart_data(self) -> Dict[str, Any]:
        """
        チャート描画用データを取得
        
        Returns:
            チャートデータ辞書
        """
        return {
            'm1_data': self.m1_data,
            'm5_data': self.m5_data,
            'm1_current': self.m1_current_bar,
            'm5_current': self.m5_current_bar,
            'current_price': self.current_price,
            'last_update': self.last_update
        }
    
    def connect_and_initialize(self) -> bool:
        """
        MT5接続とマネージャー初期化を一括実行
        
        Returns:
            成功の場合True
        """
        if not self.connect_mt5():
            logger.error("Failed to connect to MT5")
            return False
        
        if not self.initialize_manager():
            logger.error("Failed to initialize manager")
            return False
        
        return True
    
    def cleanup(self):
        """クリーンアップ処理"""
        self.stop_tick_receiver()
        mt5.shutdown()
        logger.info("RealtimeDataManager cleaned up")
"""
MT5接続サンドボックス（設定ファイル対応版）

設定ファイルを使用してMT5接続をテストする改良版サンドボックス
MT5データ取得モジュールを使用して接続、データ取得、切断をテストします。
"""

import sys
import os
import time
import logging
import json
from typing import Dict, Any, Optional, List

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from mt5_data_acquisition.mt5_client import MT5ConnectionManager
from mt5_config_example import MT5ConfigHelper

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MT5SandboxWithConfig:
    """設定ファイル対応MT5接続テスト用サンドボックスクラス"""
    
    def __init__(self, config_file: str = "test_sandbox/mt5_config.json"):
        """初期化
        
        Args:
            config_file: 設定ファイルのパス
        """
        self.connection_manager = MT5ConnectionManager()
        self.is_connected = False
        self.config_helper = MT5ConfigHelper(config_file)
        self.config = {}
    
    def load_config(self) -> bool:
        """設定ファイルから設定を読み込み
        
        Returns:
            bool: 読み込み成功時True
        """
        self.config = self.config_helper.load_config()
        
        if not self.config:
            logger.error("設定ファイルが読み込めませんでした")
            return False
        
        # 設定の検証
        if not self.config_helper.validate_config(self.config):
            logger.error("設定が無効です")
            return False
        
        logger.info("設定ファイルを正常に読み込みました")
        return True
    
    def connect_to_mt5(self) -> bool:
        """MT5への接続を試行
        
        Returns:
            bool: 接続成功時True
        """
        if not self.config:
            logger.error("設定が読み込まれていません。load_config()を先に実行してください")
            return False
        
        logger.info("MT5接続を開始します...")
        
        try:
            # 接続実行
            success = self.connection_manager.connect(self.config)
            
            if success:
                self.is_connected = True
                logger.info("MT5接続に成功しました！")
                
                # 接続情報を表示
                self._display_connection_info()
                
            else:
                logger.error("MT5接続に失敗しました")
                
            return success
            
        except Exception as e:
            logger.error(f"接続中にエラーが発生しました: {e}")
            return False
    
    def _display_connection_info(self):
        """接続情報を表示"""
        try:
            # ターミナル情報の表示
            terminal_info = self.connection_manager.terminal_info
            if terminal_info:
                logger.info("=== ターミナル情報 ===")
                logger.info(f"会社: {getattr(terminal_info, 'company', 'N/A')}")
                logger.info(f"ビルド: {getattr(terminal_info, 'build', 'N/A')}")
                logger.info(f"パス: {getattr(terminal_info, 'path', 'N/A')}")
                logger.info(f"データパス: {getattr(terminal_info, 'data_path', 'N/A')}")
                logger.info(f"共通パス: {getattr(terminal_info, 'common_path', 'N/A')}")
            
            # アカウント情報の表示
            account_info = self.connection_manager.account_info
            if account_info:
                logger.info("=== アカウント情報 ===")
                logger.info(f"ログイン: {getattr(account_info, 'login', 'N/A')}")
                logger.info(f"残高: {getattr(account_info, 'balance', 'N/A')}")
                logger.info(f"レバレッジ: {getattr(account_info, 'leverage', 'N/A')}")
                logger.info(f"通貨: {getattr(account_info, 'currency', 'N/A')}")
                logger.info(f"サーバー: {getattr(account_info, 'server', 'N/A')}")
                
        except Exception as e:
            logger.warning(f"接続情報の表示中にエラーが発生しました: {e}")
    
    def get_symbols_info(self, symbols: List[str]) -> Dict[str, Any]:
        """シンボル情報を取得
        
        Args:
            symbols: 取得するシンボルのリスト
        
        Returns:
            Dict[str, Any]: シンボル情報の辞書
        """
        if not self.is_connected:
            logger.error("MT5に接続されていません")
            return {}
        
        try:
            import MetaTrader5 as mt5
            
            symbols_info = {}
            
            for symbol in symbols:
                logger.info(f"シンボル情報を取得中: {symbol}")
                
                # シンボル情報を取得
                symbol_info = mt5.symbol_info(symbol)
                
                if symbol_info is not None:
                    symbols_info[symbol] = {
                        'name': symbol_info.name,
                        'point': symbol_info.point,
                        'digits': symbol_info.digits,
                        'spread': symbol_info.spread,
                        'spread_float': symbol_info.spread_float,
                        'ticks_bookdepth': symbol_info.ticks_bookdepth,
                        'trade_calc_mode': symbol_info.trade_calc_mode,
                        'trade_mode': symbol_info.trade_mode,
                        'start_time': symbol_info.start_time,
                        'expiration_time': symbol_info.expiration_time,
                        'trade_stops_level': symbol_info.trade_stops_level,
                        'trade_freeze_level': symbol_info.trade_freeze_level,
                        'volume_min': symbol_info.volume_min,
                        'volume_max': symbol_info.volume_max,
                        'volume_step': symbol_info.volume_step,
                        'volume_limit': symbol_info.volume_limit,
                        'swap_long': symbol_info.swap_long,
                        'swap_short': symbol_info.swap_short,
                        'margin_initial': symbol_info.margin_initial,
                        'margin_maintenance': symbol_info.margin_maintenance,
                        'session_volume': symbol_info.session_volume,
                        'session_turnover': symbol_info.session_turnover,
                        'session_interest': symbol_info.session_interest,
                        'session_buy_orders_volume': symbol_info.session_buy_orders_volume,
                        'session_sell_orders_volume': symbol_info.session_sell_orders_volume,
                        'session_open': symbol_info.session_open,
                        'session_close': symbol_info.session_close,
                        'session_aw': symbol_info.session_aw,
                        'session_price_settlement': symbol_info.session_price_settlement,
                        'session_price_limit_min': symbol_info.session_price_limit_min,
                        'session_price_limit_max': symbol_info.session_price_limit_max,
                    }
                    logger.info(f"シンボル {symbol} の情報を取得しました")
                else:
                    logger.warning(f"シンボル {symbol} の情報を取得できませんでした")
                    symbols_info[symbol] = None
            
            return symbols_info
            
        except Exception as e:
            logger.error(f"シンボル情報取得中にエラーが発生しました: {e}")
            return {}
    
    def get_ohlc_data(self, symbol: str, timeframe: int, count: int = 100) -> Optional[List[Dict[str, Any]]]:
        """OHLCデータを取得
        
        Args:
            symbol: シンボル名
            timeframe: タイムフレーム（MT5定数）
            count: 取得するバー数
        
        Returns:
            Optional[List[Dict[str, Any]]]: OHLCデータのリスト
        """
        if not self.is_connected:
            logger.error("MT5に接続されていません")
            return None
        
        try:
            import MetaTrader5 as mt5
            
            logger.info(f"OHLCデータを取得中: {symbol}, タイムフレーム: {timeframe}, バー数: {count}")
            
            # OHLCデータを取得
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            
            if rates is not None and len(rates) > 0:
                # データを辞書のリストに変換
                ohlc_data = []
                for rate in rates:
                    ohlc_data.append({
                        'time': rate['time'],
                        'open': rate['open'],
                        'high': rate['high'],
                        'low': rate['low'],
                        'close': rate['close'],
                        'tick_volume': rate['tick_volume'],
                        'spread': rate['spread'],
                        'real_volume': rate['real_volume'],
                    })
                
                logger.info(f"OHLCデータを取得しました: {len(ohlc_data)} バー")
                return ohlc_data
            else:
                logger.warning(f"シンボル {symbol} のOHLCデータを取得できませんでした")
                return None
                
        except Exception as e:
            logger.error(f"OHLCデータ取得中にエラーが発生しました: {e}")
            return None
    
    def get_tick_data(self, symbol: str, count: int = 100) -> Optional[List[Dict[str, Any]]]:
        """ティックデータを取得
        
        Args:
            symbol: シンボル名
            count: 取得するティック数
        
        Returns:
            Optional[List[Dict[str, Any]]]: ティックデータのリスト
        """
        if not self.is_connected:
            logger.error("MT5に接続されていません")
            return None
        
        try:
            import MetaTrader5 as mt5
            
            logger.info(f"ティックデータを取得中: {symbol}, ティック数: {count}")
            
            # ティックデータを取得
            ticks = mt5.copy_ticks_from(symbol, 0, count, mt5.COPY_TICKS_ALL)
            
            if ticks is not None and len(ticks) > 0:
                # データを辞書のリストに変換
                tick_data = []
                for tick in ticks:
                    tick_data.append({
                        'time': tick['time'],
                        'bid': tick['bid'],
                        'ask': tick['ask'],
                        'last': tick['last'],
                        'volume': tick['volume'],
                        'time_msc': tick['time_msc'],
                        'flags': tick['flags'],
                        'volume_real': tick['volume_real'],
                    })
                
                logger.info(f"ティックデータを取得しました: {len(tick_data)} ティック")
                return tick_data
            else:
                logger.warning(f"シンボル {symbol} のティックデータを取得できませんでした")
                return None
                
        except Exception as e:
            logger.error(f"ティックデータ取得中にエラーが発生しました: {e}")
            return None
    
    def health_check(self) -> bool:
        """接続の健全性を確認
        
        Returns:
            bool: 接続が健全な場合True
        """
        if not self.is_connected:
            logger.warning("接続されていないため、ヘルスチェックをスキップします")
            return False
        
        logger.info("ヘルスチェックを実行中...")
        is_healthy = self.connection_manager.health_check()
        
        if is_healthy:
            logger.info("ヘルスチェック成功: 接続は正常です")
        else:
            logger.warning("ヘルスチェック失敗: 接続に問題があります")
        
        return is_healthy
    
    def disconnect(self):
        """MT5から切断"""
        if self.is_connected:
            logger.info("MT5から切断中...")
            self.connection_manager.disconnect()
            self.is_connected = False
            logger.info("MT5から切断しました")
        else:
            logger.info("既に切断されています")


def main():
    """メイン関数 - 設定ファイルを使用したサンドボックスの実行例"""
    
    # テスト用シンボル
    test_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    # サンドボックスの作成
    sandbox = MT5SandboxWithConfig()
    
    try:
        # 1. 設定ファイルの読み込み
        logger.info("=== 設定ファイル読み込み ===")
        
        if not sandbox.load_config():
            logger.error("設定ファイルの読み込みに失敗したため、テストを終了します")
            logger.info("mt5_config_example.pyを実行して設定ファイルを作成してください")
            return
        
        # 2. MT5への接続
        logger.info("\n=== MT5接続テスト開始 ===")
        
        if not sandbox.connect_to_mt5():
            logger.error("接続に失敗したため、テストを終了します")
            return
        
        # 3. ヘルスチェック
        logger.info("\n=== ヘルスチェック ===")
        sandbox.health_check()
        
        # 4. シンボル情報の取得
        logger.info("\n=== シンボル情報取得 ===")
        symbols_info = sandbox.get_symbols_info(test_symbols)
        
        for symbol, info in symbols_info.items():
            if info:
                logger.info(f"{symbol}: ポイント={info['point']}, スプレッド={info['spread']}")
            else:
                logger.warning(f"{symbol}: 情報取得失敗")
        
        # 5. OHLCデータの取得
        logger.info("\n=== OHLCデータ取得 ===")
        import MetaTrader5 as mt5
        
        for symbol in test_symbols[:1]:  # 最初のシンボルのみテスト
            ohlc_data = sandbox.get_ohlc_data(symbol, mt5.TIMEFRAME_H1, 10)
            if ohlc_data:
                logger.info(f"{symbol} の最新10バー:")
                for i, bar in enumerate(ohlc_data[-5:]):  # 最新5バーのみ表示
                    logger.info(f"  バー{i+1}: O={bar['open']:.5f}, H={bar['high']:.5f}, L={bar['low']:.5f}, C={bar['close']:.5f}")
        
        # 6. ティックデータの取得
        logger.info("\n=== ティックデータ取得 ===")
        for symbol in test_symbols[:1]:  # 最初のシンボルのみテスト
            tick_data = sandbox.get_tick_data(symbol, 5)
            if tick_data:
                logger.info(f"{symbol} の最新5ティック:")
                for i, tick in enumerate(tick_data[-3:]):  # 最新3ティックのみ表示
                    logger.info(f"  ティック{i+1}: Bid={tick['bid']:.5f}, Ask={tick['ask']:.5f}, Time={tick['time']}")
        
        # 7. 再接続テスト
        logger.info("\n=== 再接続テスト ===")
        if sandbox.connection_manager.reconnect():
            logger.info("再接続に成功しました")
        else:
            logger.error("再接続に失敗しました")
        
    except KeyboardInterrupt:
        logger.info("ユーザーによって中断されました")
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
    finally:
        # 8. 切断
        logger.info("\n=== 切断 ===")
        sandbox.disconnect()
        logger.info("サンドボックステストを終了しました")


if __name__ == "__main__":
    main()

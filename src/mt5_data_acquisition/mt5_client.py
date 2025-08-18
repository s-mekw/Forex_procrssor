"""
MT5接続管理モジュール

MetaTrader 5への接続管理、再接続ロジック、接続プール管理を提供します。
"""

import logging
import time
from typing import Optional, Dict, Any

# structlogを使用（可能な場合）
try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    # structlogが利用できない場合は標準のloggingを使用
    import logging
    logger = logging.getLogger(__name__)

# MT5パッケージのインポート（テスト時はモックされる）
try:
    import MetaTrader5 as mt5
except ImportError:
    # テスト環境や開発環境でMT5がインストールされていない場合のフォールバック
    mt5 = None


class MT5ConnectionManager:
    """MT5接続を管理するクラス
    
    MetaTrader 5への接続を管理し、信頼性の高い取引環境を提供します。
    指数バックオフアルゴリズムによる自動再接続、接続プール管理、
    ヘルスチェック機能を搭載しています。
    
    Attributes:
        DEFAULT_MAX_RETRIES: デフォルトの最大再試行回数
        DEFAULT_TIMEOUT: デフォルトのタイムアウト時間（ミリ秒）
        DEFAULT_RETRY_DELAY: デフォルトの初期待機時間（秒）
        DEFAULT_MAX_DELAY: デフォルトの最大待機時間（秒）
        BACKOFF_FACTOR: バックオフ係数（指数バックオフで使用）
    """
    
    # クラス定数
    DEFAULT_MAX_RETRIES = 5
    DEFAULT_TIMEOUT = 60000  # 60秒
    DEFAULT_RETRY_DELAY = 1  # 初期待機時間（秒）
    DEFAULT_MAX_DELAY = 16  # 最大待機時間（秒）
    BACKOFF_FACTOR = 2  # バックオフ係数
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初期化メソッド
        
        Args:
            config: 接続設定の辞書（オプション）
                - account: アカウント番号
                - password: パスワード
                - server: サーバー名
                - timeout: タイムアウト時間（ミリ秒）
                - path: MT5のパス（オプション）
                - max_retries: 最大再試行回数（オプション）
                - retry_delay: 初期待機時間（オプション）
        """
        # 接続管理
        self._connected: bool = False
        self._config: Optional[Dict[str, Any]] = config
        
        # 再試行管理
        self._retry_count: int = 0
        self._max_retries: int = self._get_config_value('max_retries', self.DEFAULT_MAX_RETRIES)
        self._retry_delay: int = self._get_config_value('retry_delay', self.DEFAULT_RETRY_DELAY)
        
        # MT5情報
        self._terminal_info: Optional[Dict[str, Any]] = None
        self._account_info: Optional[Dict[str, Any]] = None
        
        # ロガー（グローバルロガーを使用）
        self.logger = logger
        
    def connect(self, config: Dict[str, Any]) -> bool:
        """MT5への接続を確立
        
        Args:
            config: 接続設定の辞書
                - account: アカウント番号
                - password: パスワード
                - server: サーバー名
                - timeout: タイムアウト時間（ミリ秒）
                - path: MT5のパス（オプション）
        
        Returns:
            bool: 接続成功時True、失敗時False
        """
        # 設定を保存
        self._config = config
        
        # 接続開始ログ
        self.logger.info("MT5接続を開始します...")
        
        # MT5が利用できない場合（テスト環境など）
        if mt5 is None:
            self.logger.error("MetaTrader5パッケージがインストールされていません")
            return False
        
        try:
            # 1. MT5の初期化
            path = config.get('path')
            if path:
                # パスが指定されている場合
                init_result = mt5.initialize(path=path)
            else:
                # パスが指定されていない場合（デフォルトパス使用）
                init_result = mt5.initialize()
            
            if not init_result:
                # 初期化失敗
                error = mt5.last_error()
                self.logger.error(f"MT5の初期化に失敗しました: {error}")
                return False
            
            # 2. MT5へのログイン
            login = config.get('account')
            password = config.get('password')
            server = config.get('server')
            timeout = config.get('timeout', self.DEFAULT_TIMEOUT)
            
            # ログイン実行
            login_result = mt5.login(
                login=login,
                password=password,
                server=server,
                timeout=timeout
            )
            
            if not login_result:
                # ログイン失敗
                error = mt5.last_error()
                self.logger.error(f"MT5へのログインに失敗しました: {error}")
                # 初期化は成功したが、ログインに失敗した場合はshutdownする
                mt5.shutdown()
                return False
            
            # 3. 接続成功時の処理
            # ターミナル情報の取得
            self._terminal_info = mt5.terminal_info()
            if self._terminal_info:
                self.logger.info(f"ターミナル情報: 会社={self._terminal_info.company}, ビルド={self._terminal_info.build}")
            
            # アカウント情報の取得
            self._account_info = mt5.account_info()
            if self._account_info:
                self.logger.info(f"アカウント情報: ログイン={self._account_info.login}, 残高={self._account_info.balance}, レバレッジ={self._account_info.leverage}")
            
            # 接続状態を更新
            self._connected = True
            
            # 再試行カウンタをリセット
            self._reset_retry_count()
            
            # 成功ログ
            self.logger.info("MT5への接続に成功しました")
            
            return True
            
        except Exception as e:
            # 予期しない例外の処理
            self.logger.error(f"MT5接続中に予期しないエラーが発生しました: {e}")
            # 念のためshutdownを呼び出す
            try:
                if mt5:
                    mt5.shutdown()
            except:
                pass
            return False
    
    def disconnect(self) -> None:
        """MT5から切断
        
        MT5との接続を安全に切断し、内部状態をリセットします。
        """
        # MT5が利用できない場合は何もしない
        if mt5 is None:
            self.logger.warning("MetaTrader5パッケージがインストールされていないため、切断処理をスキップします")
            return
        
        try:
            # MT5のシャットダウン
            if self._connected:
                mt5.shutdown()
                self.logger.info("MT5から切断しました")
            
            # 内部状態のリセット
            self._connected = False
            self._terminal_info = None
            self._account_info = None
            
        except Exception as e:
            # 切断時のエラーは警告として記録
            self.logger.warning(f"MT5切断中にエラーが発生しました: {e}")
            # それでも内部状態はリセットする
            self._connected = False
            self._terminal_info = None
            self._account_info = None
    
    def is_connected(self) -> bool:
        """接続状態を返す
        
        Returns:
            bool: 接続中の場合True
        """
        return self._connected
    
    def reconnect(self) -> bool:
        """再接続を試みる
        
        指数バックオフアルゴリズムを使用した再接続
        
        Returns:
            bool: 再接続成功時True
        """
        # 実装はStep 7で行う
        pass
    
    def health_check(self) -> bool:
        """ヘルスチェックを実行
        
        Returns:
            bool: 正常な場合True
        """
        # 実装はStep 9で行う
        pass
    
    # ========== プロパティメソッド ==========
    
    @property
    def terminal_info(self) -> Optional[Dict[str, Any]]:
        """ターミナル情報を取得
        
        Returns:
            ターミナル情報の辞書、未接続の場合None
        """
        return self._terminal_info
    
    @property
    def account_info(self) -> Optional[Dict[str, Any]]:
        """アカウント情報を取得
        
        Returns:
            アカウント情報の辞書、未接続の場合None
        """
        return self._account_info
    
    @property
    def retry_count(self) -> int:
        """現在の再試行回数を取得
        
        Returns:
            現在の再試行回数
        """
        return self._retry_count
    
    @property
    def max_retries(self) -> int:
        """最大再試行回数を取得
        
        Returns:
            最大再試行回数
        """
        return self._max_retries
    
    @max_retries.setter
    def max_retries(self, value: int):
        """最大再試行回数を設定
        
        Args:
            value: 新しい最大再試行回数（1以上の整数）
        
        Raises:
            ValueError: 無効な値が指定された場合
        """
        if value < 1:
            raise ValueError("最大再試行回数は1以上である必要があります")
        self._max_retries = value
    
    # ========== ユーティリティメソッド ==========
    
    def _get_config_value(self, key: str, default: Any) -> Any:
        """設定値を取得（デフォルト値付き）
        
        Args:
            key: 設定キー
            default: デフォルト値
        
        Returns:
            設定値またはデフォルト値
        """
        if self._config and key in self._config:
            return self._config[key]
        return default
    
    def _reset_retry_count(self):
        """再試行カウンタをリセット
        
        接続成功時や新しい接続試行の開始時に呼び出します。
        """
        self._retry_count = 0
        self.logger.debug("再試行カウンタをリセットしました")
    
    def _increment_retry_count(self) -> int:
        """再試行カウンタをインクリメント
        
        Returns:
            インクリメント後の再試行回数
        """
        self._retry_count += 1
        self.logger.debug(f"再試行カウンタをインクリメント: {self._retry_count}/{self._max_retries}")
        return self._retry_count
    
    def _calculate_backoff_delay(self) -> float:
        """現在の再試行回数から待機時間を計算
        
        指数バックオフアルゴリズムを使用して待機時間を計算します。
        待機時間 = min(初期待機時間 * (バックオフ係数 ^ 再試行回数), 最大待機時間)
        
        Returns:
            計算された待機時間（秒）
        """
        delay = min(
            self._retry_delay * (self.BACKOFF_FACTOR ** self._retry_count),
            self.DEFAULT_MAX_DELAY
        )
        self.logger.debug(f"バックオフ待機時間を計算: {delay}秒 (試行回数: {self._retry_count})")
        return delay
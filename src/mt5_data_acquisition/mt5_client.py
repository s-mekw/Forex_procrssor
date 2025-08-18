"""
MT5接続管理モジュール

MetaTrader 5への接続管理、再接続ロジック、接続プール管理を提供します。
"""

import logging
import time
import threading
from typing import Optional, Dict, Any, List

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
            bool: 再接続成功時True、全試行失敗時False
        """
        # 既存の接続がある場合は切断
        if self._connected:
            self.disconnect()
        
        # 再試行カウンタをリセット
        self._reset_retry_count()
        
        # 最大再試行回数までループ
        for attempt in range(1, self._max_retries + 1):
            # 再試行カウンタをインクリメント
            self._increment_retry_count()
            
            # 再接続試行のログ
            self.logger.info(f"再接続を試行します... (試行回数: {attempt}/{self._max_retries})")
            
            # 接続を試みる
            if self.connect(self._config):
                # 接続成功
                self.logger.info(f"再接続に成功しました (試行回数: {attempt})")
                return True
            
            # 接続失敗 - 最後の試行でない場合は待機
            if attempt < self._max_retries:
                # バックオフ待機時間を計算
                delay = self._calculate_backoff_delay()
                self.logger.warning(f"接続失敗。{int(delay)}秒後に再試行します...")
                
                # 指数バックオフによる待機
                time.sleep(delay)
        
        # 全試行失敗
        self.logger.error("再接続に失敗しました。最大試行回数に達しました")
        return False
    
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
        待機時間 = min(初期待機時間 * (バックオフ係数 ^ (再試行回数-1)), 最大待機時間)
        
        Returns:
            計算された待機時間（秒）
        """
        # 再試行回数-1を使用（最初の失敗後は1秒、2回目の失敗後は2秒...）
        delay = min(
            self._retry_delay * (self.BACKOFF_FACTOR ** (self._retry_count - 1)),
            self.DEFAULT_MAX_DELAY
        )
        self.logger.debug(f"バックオフ待機時間を計算: {delay}秒 (試行回数: {self._retry_count})")
        return delay


class ConnectionPool:
    """MT5接続プールの管理
    
    複数のMT5ConnectionManagerインスタンスを効率的に管理し、
    接続の再利用とリソース管理を行います。
    スレッドセーフな実装により、マルチスレッド環境での使用も可能です。
    
    Attributes:
        DEFAULT_MAX_CONNECTIONS: デフォルトの最大接続数
        DEFAULT_ACQUIRE_TIMEOUT: 接続取得のデフォルトタイムアウト時間（秒）
    """
    
    DEFAULT_MAX_CONNECTIONS = 3
    DEFAULT_ACQUIRE_TIMEOUT = 30  # 30秒
    
    def __init__(self, config: Dict[str, Any], max_connections: int = None):
        """初期化メソッド
        
        Args:
            config: MT5接続設定の辞書
            max_connections: 最大接続数（デフォルト: 3）
        """
        self._config = config
        self._max_connections = max_connections or self.DEFAULT_MAX_CONNECTIONS
        
        # プール管理
        self._idle_connections: List[MT5ConnectionManager] = []  # アイドル接続
        self._active_connections: List[MT5ConnectionManager] = []  # アクティブ接続
        
        # スレッドセーフティのためのロック
        self._lock = threading.Lock()
        self._connection_available = threading.Condition(self._lock)
        
        # ロガー
        self.logger = logger
        
        # 状態フラグ
        self._closed = False
        
        self.logger.info(f"接続プールを初期化しました (最大接続数: {self._max_connections})")
    
    def _create_connection(self) -> Optional[MT5ConnectionManager]:
        """新しい接続を作成
        
        Returns:
            新しいMT5ConnectionManagerインスタンス、作成失敗時はNone
        """
        try:
            connection = MT5ConnectionManager(self._config)
            if connection.connect(self._config):
                self.logger.info("新しい接続を作成しました")
                return connection
            else:
                self.logger.error("新しい接続の作成に失敗しました")
                return None
        except Exception as e:
            self.logger.error(f"接続作成中にエラーが発生しました: {e}")
            return None
    
    def acquire(self, timeout: Optional[float] = None) -> Optional[MT5ConnectionManager]:
        """利用可能な接続を取得
        
        アイドルプールから接続を取得するか、新規作成します。
        最大接続数に達している場合は待機またはNoneを返します。
        
        Args:
            timeout: 接続取得のタイムアウト時間（秒）。Noneの場合はデフォルト値を使用
        
        Returns:
            MT5ConnectionManagerインスタンス、取得失敗時はNone
        """
        if self._closed:
            self.logger.error("プールがクローズされています")
            return None
        
        timeout = timeout if timeout is not None else self.DEFAULT_ACQUIRE_TIMEOUT
        deadline = time.time() + timeout if timeout > 0 else None
        
        with self._connection_available:
            while True:
                # アイドル接続があれば使用
                if self._idle_connections:
                    connection = self._idle_connections.pop(0)
                    # 接続の健全性チェック
                    if connection.is_connected():
                        self._active_connections.append(connection)
                        self.logger.debug(f"アイドルプールから接続を取得 (アクティブ: {len(self._active_connections)}, アイドル: {len(self._idle_connections)})")
                        return connection
                    else:
                        # 接続が切れている場合は新しく作成を試みる
                        self.logger.warning("アイドル接続が切断されていました。再作成を試みます")
                        continue
                
                # 新規作成可能か確認
                total_connections = len(self._active_connections) + len(self._idle_connections)
                if total_connections < self._max_connections:
                    # 新規作成
                    connection = self._create_connection()
                    if connection:
                        self._active_connections.append(connection)
                        self.logger.debug(f"新規接続を作成してアクティブに追加 (アクティブ: {len(self._active_connections)}, アイドル: {len(self._idle_connections)})")
                        return connection
                    else:
                        self.logger.error("新規接続の作成に失敗しました")
                        return None
                
                # 最大接続数に達している場合は待機
                self.logger.debug(f"最大接続数に達しています。待機中... (最大: {self._max_connections})")
                
                # タイムアウトチェック
                if deadline:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        self.logger.warning("接続取得がタイムアウトしました")
                        return None
                    # 待機（タイムアウト付き）
                    if not self._connection_available.wait(timeout=remaining):
                        self.logger.warning("接続取得がタイムアウトしました")
                        return None
                else:
                    # 無限待機
                    self._connection_available.wait()
    
    def release(self, connection: MT5ConnectionManager) -> bool:
        """使用済み接続をプールに返却
        
        Args:
            connection: 返却するMT5ConnectionManagerインスタンス
        
        Returns:
            bool: 返却成功時True、失敗時False
        """
        if not connection:
            self.logger.warning("Noneの接続を返却しようとしました")
            return False
        
        with self._connection_available:
            # アクティブ接続から削除
            if connection in self._active_connections:
                self._active_connections.remove(connection)
                
                # 接続の健全性をチェック
                if connection.is_connected() and not self._closed:
                    # アイドルプールに追加
                    self._idle_connections.append(connection)
                    self.logger.debug(f"接続をアイドルプールに返却 (アクティブ: {len(self._active_connections)}, アイドル: {len(self._idle_connections)})")
                    # 待機中のスレッドに通知
                    self._connection_available.notify()
                    return True
                else:
                    # 接続が切れているか、プールがクローズされている場合は切断
                    try:
                        connection.disconnect()
                        self.logger.info("不健全な接続を切断しました")
                    except Exception as e:
                        self.logger.error(f"接続の切断中にエラーが発生しました: {e}")
                    return True
            else:
                self.logger.warning("アクティブプールに存在しない接続を返却しようとしました")
                return False
    
    def close_all(self) -> None:
        """全接続を切断してプールをクローズ
        
        アクティブおよびアイドルの全接続を切断し、
        プールをクローズ状態にします。
        """
        with self._lock:
            self._closed = True
            
            # アクティブ接続を切断
            for connection in self._active_connections:
                try:
                    connection.disconnect()
                    self.logger.debug("アクティブ接続を切断しました")
                except Exception as e:
                    self.logger.error(f"アクティブ接続の切断中にエラーが発生しました: {e}")
            
            # アイドル接続を切断
            for connection in self._idle_connections:
                try:
                    connection.disconnect()
                    self.logger.debug("アイドル接続を切断しました")
                except Exception as e:
                    self.logger.error(f"アイドル接続の切断中にエラーが発生しました: {e}")
            
            # プールをクリア
            total_closed = len(self._active_connections) + len(self._idle_connections)
            self._active_connections.clear()
            self._idle_connections.clear()
            
            self.logger.info(f"接続プールをクローズしました (切断した接続数: {total_closed})")
    
    def get_status(self) -> Dict[str, int]:
        """プールの現在の状態を取得
        
        Returns:
            プール状態の辞書
                - active: アクティブ接続数
                - idle: アイドル接続数
                - max_connections: 最大接続数
                - total: 現在の総接続数
        """
        with self._lock:
            status = {
                'active': len(self._active_connections),
                'idle': len(self._idle_connections),
                'max_connections': self._max_connections,
                'total': len(self._active_connections) + len(self._idle_connections)
            }
            self.logger.debug(f"プール状態: アクティブ={status['active']}, アイドル={status['idle']}, 最大={status['max_connections']}")
            return status
    
    @property
    def is_closed(self) -> bool:
        """プールがクローズされているか確認
        
        Returns:
            bool: クローズされている場合True
        """
        return self._closed
    
    def __enter__(self):
        """コンテキストマネージャーのエンター"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーのイグジット"""
        self.close_all()
        return False
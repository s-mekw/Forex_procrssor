"""
MT5接続管理のユニットテスト

MT5ConnectionManagerクラスの機能をテストします。
MT5パッケージはモックで代替し、外部依存なしでテスト可能にします。
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import pytest
from datetime import datetime
import logging
import sys
import os
import time

# srcディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# MT5ConnectionManagerとConnectionPoolのインポート
from mt5_data_acquisition.mt5_client import MT5ConnectionManager, ConnectionPool


class TestMT5ConnectionManager(unittest.TestCase):
    """MT5ConnectionManagerクラスのテストスイート"""

    def setUp(self):
        """各テストメソッドの前に実行される初期化処理"""
        # MT5モジュールのモック作成
        self.mock_mt5 = MagicMock()
        
        # パッチの適用（MT5パッケージをモック化）
        self.mt5_patcher = patch('mt5_data_acquisition.mt5_client.mt5', self.mock_mt5)
        self.mt5_patcher.start()  # パッチャーを開始
        
        # ロガーのモック
        self.mock_logger = MagicMock(spec=logging.Logger)
        self.logger_patcher = patch('mt5_data_acquisition.mt5_client.logger', self.mock_logger)
        self.logger_patcher.start()  # パッチャーを開始
        
        # MT5ConnectionManagerのインスタンスを作成
        self.connection_manager = MT5ConnectionManager()
        
        # テスト用の設定
        self.test_config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'TestServer-Demo',
            'timeout': 30000,
            'path': None
        }

    def tearDown(self):
        """各テストメソッドの後に実行されるクリーンアップ処理"""
        # パッチを停止
        if hasattr(self, 'mt5_patcher'):
            try:
                self.mt5_patcher.stop()
            except:
                pass
        
        if hasattr(self, 'logger_patcher'):
            try:
                self.logger_patcher.stop()
            except:
                pass
        
        # モックのリセット
        if hasattr(self, 'mock_mt5'):
            self.mock_mt5.reset_mock()
        
        if hasattr(self, 'mock_logger'):
            self.mock_logger.reset_mock()

    # Step 2以降で実装するテストケースのプレースホルダー
    def test_connection_success(self):
        """接続成功時のテスト
        
        正常な接続フローをテスト:
        1. MT5の初期化が成功
        2. ログインが成功
        3. ターミナル情報とアカウント情報が取得できる
        4. 適切なログが出力される
        """
        # MT5初期化とログインの成功をモック
        self.mock_mt5.initialize.return_value = True
        self.mock_mt5.login.return_value = True
        
        # ターミナル情報のモックオブジェクト作成
        mock_terminal_info = MagicMock()
        mock_terminal_info.company = "Test Broker"
        mock_terminal_info.name = "MetaTrader 5"
        mock_terminal_info.build = 3320
        mock_terminal_info.data_path = "/path/to/mt5"
        mock_terminal_info.connected = True
        self.mock_mt5.terminal_info.return_value = mock_terminal_info
        
        # アカウント情報のモックオブジェクト作成
        mock_account_info = MagicMock()
        mock_account_info.login = 12345678
        mock_account_info.server = "TestServer-Demo"
        mock_account_info.name = "Test User"
        mock_account_info.balance = 10000.0
        mock_account_info.equity = 10000.0
        mock_account_info.margin = 0.0
        mock_account_info.margin_free = 10000.0
        mock_account_info.leverage = 100
        mock_account_info.currency = "USD"
        self.mock_mt5.account_info.return_value = mock_account_info
        
        # テスト実行
        result = self.connection_manager.connect(self.test_config)
        
        # モック呼び出しの検証
        self.mock_mt5.initialize.assert_called_once()
        self.mock_mt5.login.assert_called_once_with(
            login=self.test_config['account'],
            password=self.test_config['password'],
            server=self.test_config['server'],
            timeout=self.test_config['timeout']
        )
        
        # ログ出力の検証
        self.mock_logger.info.assert_any_call("MT5接続を開始します...")
        self.mock_logger.info.assert_any_call("MT5への接続に成功しました")
        self.mock_logger.info.assert_any_call(f"ターミナル情報: 会社={mock_terminal_info.company}, ビルド={mock_terminal_info.build}")
        self.mock_logger.info.assert_any_call(f"アカウント情報: ログイン={mock_account_info.login}, 残高={mock_account_info.balance}, レバレッジ={mock_account_info.leverage}")
        
        # 接続状態の検証
        self.assertTrue(result)
        self.assertTrue(self.connection_manager.is_connected())
        
        # ターミナル情報とアカウント情報が保存されているか確認
        self.assertIsNotNone(self.connection_manager.terminal_info)
        self.assertIsNotNone(self.connection_manager.account_info)
        self.assertEqual(self.connection_manager.terminal_info.company, "Test Broker")
        self.assertEqual(self.connection_manager.account_info.balance, 10000.0)

    def test_connection_failure(self):
        """接続失敗時のテスト
        
        以下の失敗シナリオをテスト:
        1. MT5初期化失敗
        2. MT5初期化成功後のログイン失敗
        3. エラーメッセージのログ出力確認
        4. 接続状態がFalseであることの確認
        """
        # シナリオ1: MT5初期化失敗のテスト
        self.mock_mt5.initialize.return_value = False
        self.mock_mt5.last_error.return_value = (500, "MT5初期化エラー: ターミナルが見つかりません")
        
        # テスト実行
        result = self.connection_manager.connect(self.test_config)
        
        # 初期化失敗時の検証
        self.assertFalse(result)
        self.assertFalse(self.connection_manager.is_connected())
        self.mock_logger.error.assert_any_call("MT5の初期化に失敗しました: (500, 'MT5初期化エラー: ターミナルが見つかりません')")
        self.mock_mt5.login.assert_not_called()  # ログインは呼ばれない
        
        # モックをリセット
        self.mock_mt5.reset_mock()
        self.mock_logger.reset_mock()
        
        # シナリオ2: ログイン失敗のテスト
        self.mock_mt5.initialize.return_value = True  # 初期化は成功
        self.mock_mt5.login.return_value = False  # ログインは失敗
        self.mock_mt5.last_error.return_value = (10004, "認証失敗: アカウントまたはパスワードが正しくありません")
        
        # テスト実行
        result = self.connection_manager.connect(self.test_config)
        
        # ログイン失敗時の検証
        self.assertFalse(result)
        self.assertFalse(self.connection_manager.is_connected())
        self.mock_mt5.initialize.assert_called_once()
        self.mock_mt5.login.assert_called_once_with(
            login=self.test_config['account'],
            password=self.test_config['password'],
            server=self.test_config['server'],
            timeout=self.test_config['timeout']
        )
        self.mock_logger.error.assert_any_call("MT5へのログインに失敗しました: (10004, '認証失敗: アカウントまたはパスワードが正しくありません')")
        # shutdownが呼ばれたか確認
        self.mock_mt5.shutdown.assert_called_once()

    @patch('time.sleep')  # time.sleepをモック化
    def test_reconnection_with_exponential_backoff(self, mock_sleep):
        """指数バックオフによる再接続テスト
        
        再接続メカニズムが指数バックオフアルゴリズムで動作することを検証:
        1. 初回接続失敗後、再接続を試行
        2. 各試行間の待機時間が指数的に増加（1, 2, 4, 8, 16秒）
        3. 3回目の試行で成功するシナリオ
        4. 適切なログ出力の確認
        """
        # 初回接続を失敗に設定
        self.mock_mt5.initialize.return_value = True
        self.mock_mt5.login.return_value = False  # 初回は失敗
        self.mock_mt5.last_error.return_value = (10004, "ネットワークエラー: 一時的な接続障害")
        
        # 接続試行（初回失敗）
        result = self.connection_manager.connect(self.test_config)
        
        # 再接続用に3回目で成功するようside_effectを設定
        # side_effect: [1回目失敗, 2回目失敗, 3回目成功]
        self.mock_mt5.login.side_effect = [False, False, True]
        
        # ターミナル情報とアカウント情報のモック（成功時用）
        mock_terminal_info = MagicMock()
        mock_terminal_info.company = "Test Broker"
        mock_terminal_info.connected = True
        self.mock_mt5.terminal_info.return_value = mock_terminal_info
        
        mock_account_info = MagicMock()
        mock_account_info.login = 12345678
        mock_account_info.balance = 10000.0
        self.mock_mt5.account_info.return_value = mock_account_info
        
        # 再接続の実行
        reconnect_result = self.connection_manager.reconnect()
        
        # 再接続成功の検証
        self.assertTrue(reconnect_result)
        self.assertTrue(self.connection_manager.is_connected())
        
        # time.sleepの呼び出し検証
        # 期待される呼び出し: sleep(1), sleep(2) の2回（3回目で成功するため）
        expected_sleep_calls = [call(1), call(2)]
        mock_sleep.assert_has_calls(expected_sleep_calls)
        self.assertEqual(mock_sleep.call_count, 2)
        
        # ログイン試行回数の検証
        # 初回接続で1回 + 再接続で3回 = 合計4回のログイン試行
        self.assertEqual(self.mock_mt5.login.call_count, 4)
        
        # ログ出力の検証
        self.mock_logger.info.assert_any_call("再接続を試行します... (試行回数: 1/5)")
        self.mock_logger.info.assert_any_call("再接続を試行します... (試行回数: 2/5)")
        self.mock_logger.info.assert_any_call("再接続を試行します... (試行回数: 3/5)")
        self.mock_logger.info.assert_any_call("再接続に成功しました (試行回数: 3)")
        self.mock_logger.warning.assert_any_call("接続失敗。1秒後に再試行します...")
        self.mock_logger.warning.assert_any_call("接続失敗。2秒後に再試行します...")
        
        # side_effectの動作確認テストは削除しても良いが、検証のために残しておく
        self.assertIsNotNone(self.connection_manager)

    @patch('time.sleep')
    def test_max_retry_attempts(self, mock_sleep):
        """最大再試行回数のテスト
        
        最大5回の試行で全て失敗するケースをテスト:
        1. 全5回の試行が失敗
        2. 指数バックオフの完全な動作確認
        3. 最終的にFalseが返されることを確認
        4. 適切なエラーログが出力されることを確認
        """
        # 初回接続を失敗に設定
        self.mock_mt5.initialize.return_value = True
        self.mock_mt5.login.return_value = False
        self.mock_mt5.last_error.return_value = (10004, "ネットワークエラー: 接続できません")
        
        # 初回接続試行（失敗）
        result = self.connection_manager.connect(self.test_config)
        self.assertFalse(result)
        
        # 再接続も全て失敗するように設定
        # side_effect: 5回全て失敗
        self.mock_mt5.login.side_effect = [False, False, False, False, False]
        
        # 再接続の実行
        reconnect_result = self.connection_manager.reconnect()
        
        # 再接続失敗の検証
        self.assertFalse(reconnect_result)
        self.assertFalse(self.connection_manager.is_connected())
        
        # time.sleepの呼び出し検証
        # 期待される呼び出し: sleep(1), sleep(2), sleep(4), sleep(8)の4回
        # 注意: 5回目は最後の試行なのでsleepしない
        expected_sleep_calls = [call(1), call(2), call(4), call(8)]
        mock_sleep.assert_has_calls(expected_sleep_calls)
        self.assertEqual(mock_sleep.call_count, 4)
        
        # ログイン試行回数の検証
        # 初回接続で1回 + 再接続で5回 = 合計6回のログイン試行
        self.assertEqual(self.mock_mt5.login.call_count, 6)
        
        # ログ出力の検証
        self.mock_logger.info.assert_any_call("再接続を試行します... (試行回数: 1/5)")
        self.mock_logger.info.assert_any_call("再接続を試行します... (試行回数: 2/5)")
        self.mock_logger.info.assert_any_call("再接続を試行します... (試行回数: 3/5)")
        self.mock_logger.info.assert_any_call("再接続を試行します... (試行回数: 4/5)")
        self.mock_logger.info.assert_any_call("再接続を試行します... (試行回数: 5/5)")
        self.mock_logger.error.assert_any_call("再接続に失敗しました。最大試行回数に達しました")
        
        # 再試行カウンタが最大値になっていることを確認
        self.assertEqual(self.connection_manager.retry_count, 5)

    def test_connection_pool_management(self):
        """接続プール管理のテスト
        
        ConnectionPoolクラスの基本機能をテストします：
        1. 接続の取得（acquire）
        2. 接続の返却（release）
        3. 最大接続数制限の動作
        4. プール状態の取得（get_status）
        5. 全接続のクローズ（close_all）
        """
        # ConnectionPoolのインスタンス作成
        pool = ConnectionPool(self.test_config, max_connections=2)
        
        # 初期状態の確認
        status = pool.get_status()
        self.assertEqual(status['active'], 0)
        self.assertEqual(status['idle'], 0)
        self.assertEqual(status['max_connections'], 2)
        self.assertEqual(status['total'], 0)
        
        # 1. 最初の接続取得
        # MT5のモックを成功状態に設定
        self.mock_mt5.initialize.return_value = True
        self.mock_mt5.login.return_value = True
        
        # ターミナル情報のモックオブジェクト作成
        mock_terminal_info = MagicMock()
        mock_terminal_info.company = "Test Broker"
        mock_terminal_info.build = 3320
        self.mock_mt5.terminal_info.return_value = mock_terminal_info
        
        # アカウント情報のモックオブジェクト作成
        mock_account_info = MagicMock()
        mock_account_info.login = 12345678
        mock_account_info.balance = 10000.0
        mock_account_info.leverage = 100
        self.mock_mt5.account_info.return_value = mock_account_info
        
        conn1 = pool.acquire()
        self.assertIsNotNone(conn1)
        self.assertIsInstance(conn1, MT5ConnectionManager)
        
        # 接続取得後の状態確認
        status = pool.get_status()
        self.assertEqual(status['active'], 1)
        self.assertEqual(status['idle'], 0)
        self.assertEqual(status['total'], 1)
        
        # 2. 2番目の接続取得
        conn2 = pool.acquire()
        self.assertIsNotNone(conn2)
        self.assertIsInstance(conn2, MT5ConnectionManager)
        self.assertIsNot(conn1, conn2)  # 異なるインスタンスであることを確認
        
        # 最大接続数に達した状態の確認
        status = pool.get_status()
        self.assertEqual(status['active'], 2)
        self.assertEqual(status['idle'], 0)
        self.assertEqual(status['total'], 2)
        
        # 3. 最大接続数を超える取得試行（タイムアウト付き）
        conn3 = pool.acquire(timeout=0.1)  # 0.1秒でタイムアウト
        self.assertIsNone(conn3)  # 最大接続数に達しているため取得失敗
        
        # 4. 接続の返却
        result = pool.release(conn1)
        self.assertTrue(result)
        
        # 返却後の状態確認
        status = pool.get_status()
        self.assertEqual(status['active'], 1)
        self.assertEqual(status['idle'], 1)
        self.assertEqual(status['total'], 2)
        
        # 5. 返却された接続の再取得
        conn3 = pool.acquire()
        self.assertIsNotNone(conn3)
        self.assertIs(conn3, conn1)  # 同じインスタンスが再利用されることを確認
        
        # 6. 全接続の返却
        pool.release(conn2)
        pool.release(conn3)
        
        status = pool.get_status()
        self.assertEqual(status['active'], 0)
        self.assertEqual(status['idle'], 2)
        self.assertEqual(status['total'], 2)
        
        # 7. 全接続のクローズ
        pool.close_all()
        self.assertTrue(pool.is_closed)
        
        # クローズ後の状態確認
        status = pool.get_status()
        self.assertEqual(status['active'], 0)
        self.assertEqual(status['idle'], 0)
        self.assertEqual(status['total'], 0)
        
        # 8. クローズ後の接続取得試行
        conn4 = pool.acquire()
        self.assertIsNone(conn4)  # プールがクローズされているため取得失敗
        
        # コンテキストマネージャーのテスト
        with ConnectionPool(self.test_config, max_connections=1) as test_pool:
            # プール内での操作
            conn = test_pool.acquire()
            self.assertIsNotNone(conn)
            test_pool.release(conn)
        # withブロックを出ると自動的にclose_all()が呼ばれる
        self.assertTrue(test_pool.is_closed)

    def test_health_check(self):
        """ヘルスチェック機能のテスト"""
        from mt5_data_acquisition.mt5_client import HealthChecker
        
        # モックをリセット
        self.mock_mt5.reset_mock()
        
        # ========== シナリオ1: 正常な接続時のhealth_check ==========
        # 接続を確立
        self.mock_mt5.initialize.return_value = True
        self.mock_mt5.login.return_value = True
        
        # terminal_infoとaccount_infoのモック
        mock_terminal_info = type('TerminalInfo', (), {
            'community_account': False,
            'connected': True,
            'dlls_allowed': True,
            'trade_allowed': True,
            'tradeapi_disabled': False,
            'path': 'C:/Program Files/MetaTrader 5',
            'data_path': 'C:/Users/Test/AppData/Roaming/MetaQuotes/Terminal',
            'company': 'Test Broker',  # companyを追加
            'build': 3000,  # buildを追加
        })()
        
        mock_account_info = type('AccountInfo', (), {
            'login': 12345678,
            'server': 'TestServer',
            'balance': 100000.0,
            'profit': 0.0,
            'equity': 100000.0,
            'margin': 0.0,
            'currency': 'USD',
            'leverage': 100,  # leverageを追加
        })()
        
        self.mock_mt5.terminal_info.return_value = mock_terminal_info
        self.mock_mt5.account_info.return_value = mock_account_info
        
        # 接続を確立
        result = self.connection_manager.connect(self.test_config)
        self.assertTrue(result)
        self.assertTrue(self.connection_manager.is_connected())
        
        # ヘルスチェック実行（成功）
        health_result = self.connection_manager.health_check()
        self.assertTrue(health_result)
        
        # MT5 APIが呼ばれたことを確認
        self.mock_mt5.terminal_info.assert_called()
        self.mock_mt5.account_info.assert_called()
        
        # ========== シナリオ2: 切断状態でのhealth_check ==========
        # 接続を切断
        self.connection_manager.disconnect()
        self.assertFalse(self.connection_manager.is_connected())
        
        # ヘルスチェック実行（失敗）
        health_result = self.connection_manager.health_check()
        self.assertFalse(health_result)
        
        # ========== シナリオ3: MT5 API呼び出し失敗時 ==========
        # 再度接続を確立
        self.mock_mt5.login.return_value = True
        result = self.connection_manager.connect(self.test_config)
        self.assertTrue(result)
        
        # terminal_infoがNoneを返すように設定
        self.mock_mt5.terminal_info.return_value = None
        
        # ヘルスチェック実行（失敗）
        health_result = self.connection_manager.health_check()
        self.assertFalse(health_result)
        
        # ========== シナリオ4: HealthCheckerクラスのテスト ==========
        # 新しい接続マネージャーインスタンスを作成
        test_manager = MT5ConnectionManager(self.test_config)
        
        # HealthCheckerインスタンスを作成（チェック間隔1秒、自動再接続有効）
        health_checker = HealthChecker(
            connection_manager=test_manager,
            interval=1,
            auto_reconnect=True
        )
        
        # HealthCheckerの初期状態を確認
        self.assertFalse(health_checker.is_running)
        self.assertEqual(health_checker.interval, 1)
        self.assertTrue(health_checker.auto_reconnect)
        
        # HealthCheckerを開始
        health_checker.start()
        self.assertTrue(health_checker.is_running)
        
        # 少し待機してからチェック実行を確認（モックで動作確認）
        import time
        time.sleep(0.1)  # 短い待機
        
        # HealthCheckerを停止
        health_checker.stop()
        self.assertFalse(health_checker.is_running)
        
        # ========== シナリオ5: 自動再接続のテスト ==========
        # HealthCheckerで自動再接続が動作することを確認
        test_manager2 = MT5ConnectionManager(self.test_config)
        
        # 最初は接続失敗するが、reconnectで成功するように設定
        test_manager2._connected = True  # 接続済みと仮定
        
        # health_checkメソッドをモック（失敗を返す）
        with patch.object(test_manager2, 'health_check', return_value=False):
            with patch.object(test_manager2, 'reconnect', return_value=True) as mock_reconnect:
                # HealthCheckerを作成（自動再接続有効）
                health_checker2 = HealthChecker(
                    connection_manager=test_manager2,
                    interval=1,
                    auto_reconnect=True
                )
                
                # 単一のチェックを実行
                health_checker2._perform_check()
                
                # reconnectが呼ばれたことを確認
                mock_reconnect.assert_called_once()
        
        # ========== シナリオ6: 自動再接続無効時のテスト ==========
        test_manager3 = MT5ConnectionManager(self.test_config)
        test_manager3._connected = True
        
        with patch.object(test_manager3, 'health_check', return_value=False):
            with patch.object(test_manager3, 'reconnect', return_value=True) as mock_reconnect:
                # HealthCheckerを作成（自動再接続無効）
                health_checker3 = HealthChecker(
                    connection_manager=test_manager3,
                    interval=1,
                    auto_reconnect=False  # 自動再接続無効
                )
                
                # 単一のチェックを実行
                health_checker3._perform_check()
                
                # reconnectが呼ばれていないことを確認
                mock_reconnect.assert_not_called()
    
    def test_property_methods(self):
        """プロパティメソッドのテスト（Step 5で実装）"""
        # terminal_infoプロパティのテスト
        self.assertIsNone(self.connection_manager.terminal_info)
        test_terminal_info = {"version": "5.0.0", "build": 3000}
        self.connection_manager._terminal_info = test_terminal_info
        self.assertEqual(self.connection_manager.terminal_info, test_terminal_info)
        
        # account_infoプロパティのテスト
        self.assertIsNone(self.connection_manager.account_info)
        test_account_info = {"login": 12345678, "server": "TestServer"}
        self.connection_manager._account_info = test_account_info
        self.assertEqual(self.connection_manager.account_info, test_account_info)
        
        # retry_countプロパティのテスト
        self.assertEqual(self.connection_manager.retry_count, 0)
        self.connection_manager._retry_count = 3
        self.assertEqual(self.connection_manager.retry_count, 3)
        
        # max_retriesプロパティのテスト（読み取り）
        self.assertEqual(self.connection_manager.max_retries, 5)
        
        # max_retriesプロパティのテスト（設定）
        self.connection_manager.max_retries = 10
        self.assertEqual(self.connection_manager.max_retries, 10)
        
        # max_retriesプロパティのテスト（無効な値）
        with self.assertRaises(ValueError) as context:
            self.connection_manager.max_retries = 0
        self.assertIn("最大再試行回数は1以上である必要があります", str(context.exception))
    
    def test_utility_methods(self):
        """ユーティリティメソッドのテスト（Step 5で実装）"""
        # _reset_retry_countメソッドのテスト
        self.connection_manager._retry_count = 5
        self.connection_manager._reset_retry_count()
        self.assertEqual(self.connection_manager.retry_count, 0)
        
        # _increment_retry_countメソッドのテスト
        self.assertEqual(self.connection_manager._increment_retry_count(), 1)
        self.assertEqual(self.connection_manager.retry_count, 1)
        self.assertEqual(self.connection_manager._increment_retry_count(), 2)
        self.assertEqual(self.connection_manager.retry_count, 2)
        
        # _calculate_backoff_delayメソッドのテスト
        # retry_count - 1の計算式になったため、期待値を更新
        self.connection_manager._retry_count = 1
        self.assertAlmostEqual(self.connection_manager._calculate_backoff_delay(), 1.0)  # 2^(1-1) = 1
        
        self.connection_manager._retry_count = 2
        self.assertAlmostEqual(self.connection_manager._calculate_backoff_delay(), 2.0)  # 2^(2-1) = 2
        
        self.connection_manager._retry_count = 3
        self.assertAlmostEqual(self.connection_manager._calculate_backoff_delay(), 4.0)  # 2^(3-1) = 4
        
        self.connection_manager._retry_count = 4
        self.assertAlmostEqual(self.connection_manager._calculate_backoff_delay(), 8.0)  # 2^(4-1) = 8
        
        self.connection_manager._retry_count = 5
        self.assertAlmostEqual(self.connection_manager._calculate_backoff_delay(), 16.0)  # 2^(5-1) = 16
        
        # 最大待機時間のテスト
        self.connection_manager._retry_count = 10  # 非常に大きな再試行回数
        self.assertAlmostEqual(self.connection_manager._calculate_backoff_delay(), 16.0)  # 最大値でクリップ
    
    def test_config_with_custom_values(self):
        """カスタム設定値でのインスタンス化テスト（Step 5で実装）"""
        custom_config = {
            'account': 99999999,
            'password': 'custom_password',
            'server': 'CustomServer',
            'max_retries': 10,
            'retry_delay': 2
        }
        
        custom_manager = MT5ConnectionManager(config=custom_config)
        
        # カスタム設定が適用されているか確認
        self.assertEqual(custom_manager.max_retries, 10)
        self.assertEqual(custom_manager._retry_delay, 2)
        self.assertEqual(custom_manager._config, custom_config)
        
        # _get_config_valueメソッドのテスト
        self.assertEqual(custom_manager._get_config_value('account', None), 99999999)
        self.assertEqual(custom_manager._get_config_value('nonexistent', 'default'), 'default')


class TestMT5ConnectionManagerIntegration(unittest.TestCase):
    """MT5ConnectionManagerの統合テスト"""
    
    def setUp(self):
        """統合テスト用のセットアップ"""
        # MT5モジュールのモック作成
        self.mock_mt5 = MagicMock()
        
        # パッチの適用（MT5パッケージをモック化）
        self.mt5_patcher = patch('mt5_data_acquisition.mt5_client.mt5', self.mock_mt5)
        self.mt5_patcher.start()  # パッチャーを開始
        
        # ロガーのモック
        self.mock_logger = MagicMock(spec=logging.Logger)
        self.logger_patcher = patch('mt5_data_acquisition.mt5_client.logger', self.mock_logger)
        self.logger_patcher.start()  # パッチャーを開始
        
        # MT5ConnectionManagerのインスタンスを作成
        self.connection_manager = MT5ConnectionManager()
        
        # テスト用の設定
        self.test_config = {
            'account': 12345678,
            'password': 'test_password',
            'server': 'TestServer-Demo',
            'timeout': 30000,
            'path': None
        }
    
    def tearDown(self):
        """統合テスト用のクリーンアップ"""
        # パッチを停止
        if hasattr(self, 'mt5_patcher'):
            try:
                self.mt5_patcher.stop()
            except:
                pass
        
        if hasattr(self, 'logger_patcher'):
            try:
                self.logger_patcher.stop()
            except:
                pass
        
        # モックのリセット
        if hasattr(self, 'mock_mt5'):
            self.mock_mt5.reset_mock()
        
        if hasattr(self, 'mock_logger'):
            self.mock_logger.reset_mock()
    
    def test_integration_flow(self):
        """完全な接続フローの統合テスト
        
        以下のフローをテスト:
        1. 初期接続の確立
        2. ヘルスチェックの実行
        3. 接続プールでの管理
        4. 意図的な切断
        5. 再接続の実行
        6. HealthCheckerとの統合
        7. 最終的なクリーンアップ
        """
        from mt5_data_acquisition.mt5_client import ConnectionPool, HealthChecker
        
        # ========== 1. 初期接続の確立 ==========
        # MT5モックを成功状態に設定
        self.mock_mt5.initialize.return_value = True
        self.mock_mt5.login.return_value = True
        
        # ターミナル情報とアカウント情報のモック
        mock_terminal_info = MagicMock()
        mock_terminal_info.company = "Test Broker"
        mock_terminal_info.connected = True
        mock_terminal_info.build = 3320
        self.mock_mt5.terminal_info.return_value = mock_terminal_info
        
        mock_account_info = MagicMock()
        mock_account_info.login = 12345678
        mock_account_info.balance = 10000.0
        mock_account_info.leverage = 100
        self.mock_mt5.account_info.return_value = mock_account_info
        
        # 接続を確立
        result = self.connection_manager.connect(self.test_config)
        self.assertTrue(result, "初期接続が成功すべき")
        self.assertTrue(self.connection_manager.is_connected())
        
        # ========== 2. ヘルスチェックの実行 ==========
        health_result = self.connection_manager.health_check()
        self.assertTrue(health_result, "接続済みでヘルスチェックは成功すべき")
        
        # ========== 3. 接続プールでの管理 ==========
        # ConnectionPoolを作成して接続を管理
        pool = ConnectionPool(self.test_config, max_connections=2)
        
        # 接続を取得
        conn1 = pool.acquire()
        self.assertIsNotNone(conn1, "プールから接続が取得できるべき")
        self.assertIsInstance(conn1, MT5ConnectionManager)
        
        # プール状態の確認
        status = pool.get_status()
        self.assertEqual(status['active'], 1)
        self.assertEqual(status['idle'], 0)
        self.assertEqual(status['total'], 1)
        
        # 接続を返却
        pool.release(conn1)
        status = pool.get_status()
        self.assertEqual(status['active'], 0)
        self.assertEqual(status['idle'], 1)
        
        # ========== 4. 意図的な切断 ==========
        self.connection_manager.disconnect()
        self.assertFalse(self.connection_manager.is_connected(), "切断後は未接続状態になるべき")
        
        # 切断状態でのヘルスチェック（失敗するはず）
        health_result = self.connection_manager.health_check()
        self.assertFalse(health_result, "切断状態でヘルスチェックは失敗すべき")
        
        # ========== 5. 再接続の実行 ==========
        # 再接続は成功するように設定
        self.mock_mt5.login.side_effect = None  # side_effectをクリア
        self.mock_mt5.login.return_value = True
        
        reconnect_result = self.connection_manager.reconnect()
        self.assertTrue(reconnect_result, "再接続が成功すべき")
        self.assertTrue(self.connection_manager.is_connected())
        
        # ========== 6. HealthCheckerとの統合 ==========
        # HealthCheckerを作成して動作確認
        health_checker = HealthChecker(
            connection_manager=self.connection_manager,
            interval=1,
            auto_reconnect=True
        )
        
        # HealthCheckerを開始
        health_checker.start()
        self.assertTrue(health_checker.is_running, "HealthCheckerが実行中であるべき")
        
        # 少し待機してからチェック実行を確認
        import time
        time.sleep(0.1)
        
        # HealthCheckerを停止
        health_checker.stop()
        self.assertFalse(health_checker.is_running, "HealthCheckerが停止しているべき")
        
        # ========== 7. 最終的なクリーンアップ ==========
        # プールの全接続をクローズ
        pool.close_all()
        self.assertTrue(pool.is_closed, "プールがクローズされているべき")
        
        # 最終的な切断
        self.connection_manager.disconnect()
        self.assertFalse(self.connection_manager.is_connected())
        
        # MT5のshutdownが呼ばれたことを確認
        self.mock_mt5.shutdown.assert_called()
        
        # ログ出力の確認（主要なもののみ）
        self.mock_logger.info.assert_any_call("MT5への接続に成功しました")
        self.mock_logger.info.assert_any_call("MT5から切断しました")
        self.mock_logger.info.assert_any_call("再接続に成功しました (試行回数: 1)")


@pytest.mark.unit
@pytest.mark.skip(reason="MT5ConnectionManagerクラスが未実装")
def test_mt5_client_imports():
    """必要なモジュールがインポート可能かテスト"""
    # 後でMT5ConnectionManagerのインポートをテスト
    pass


if __name__ == '__main__':
    # ユニットテストの実行
    unittest.main(verbosity=2)
# mt5_client.py

## 概要
MetaTrader 5プラットフォームへの接続管理を担当するモジュール。接続プール、ヘルスチェック、自動再接続機能を提供し、安定したMT5接続を維持します。

## 依存関係
- 外部ライブラリ: MetaTrader5, asyncio, structlog, typing
- 内部モジュール: common.config (BaseConfig)

## 主要コンポーネント

### クラス

#### MT5ConnectionManager
**目的**: MT5プラットフォームへの接続を管理するシングルトンクラス

**属性**:
- `_instance` (Optional[MT5ConnectionManager]): シングルトンインスタンス
- `_connected` (bool): 接続状態フラグ
- `_config` (BaseConfig): 設定インスタンス
- `_lock` (asyncio.Lock): 接続操作用のロック
- `_last_ping` (datetime): 最後のping時刻

**メソッド**:
- `connect()`: MT5への接続を確立
- `disconnect()`: MT5から切断
- `reconnect()`: 再接続を実行
- `is_connected()`: 接続状態を確認
- `execute_with_retry(func, *args, **kwargs)`: リトライ付きで関数を実行
- `get_account_info()`: アカウント情報を取得
- `ping()`: 接続確認のping送信

#### ConnectionPool
**目的**: 複数のMT5接続を管理するプール

**属性**:
- `_connections` (List[MT5ConnectionManager]): 接続のリスト
- `_max_connections` (int): 最大接続数
- `_available` (Queue): 利用可能な接続のキュー
- `_in_use` (Set): 使用中の接続セット

**メソッド**:
- `acquire()`: 接続を取得
- `release(connection)`: 接続を返却
- `close_all()`: すべての接続を閉じる
- `get_stats()`: プールの統計情報を取得

#### HealthChecker
**目的**: MT5接続の健全性を監視

**属性**:
- `_connection_manager` (MT5ConnectionManager): 監視対象の接続マネージャー
- `_check_interval` (int): チェック間隔（秒）
- `_max_failures` (int): 最大失敗回数
- `_failure_count` (int): 現在の失敗回数
- `_running` (bool): 実行状態フラグ

**メソッド**:
- `start()`: ヘルスチェックを開始
- `stop()`: ヘルスチェックを停止
- `check_health()`: 健全性をチェック
- `handle_failure()`: 失敗を処理
- `reset_failure_count()`: 失敗カウントをリセット

### 関数

#### connect_to_mt5
**目的**: MT5への接続を確立するヘルパー関数

**入力**:
- `config` (Optional[BaseConfig]): 設定オブジェクト
- `retry_count` (int): リトライ回数、デフォルト3
- `retry_delay` (float): リトライ間隔（秒）、デフォルト1.0

**出力**:
- bool: 接続成功時True

**処理フロー**:
1. 設定を検証
2. MT5を初期化
3. ログイン情報で認証
4. 接続成功を確認
5. 失敗時はリトライ

**例外**:
- `ConnectionError`: 接続に失敗した場合
- `ValueError`: 無効な設定の場合

#### disconnect_from_mt5
**目的**: MT5から切断

**入力**:
- なし

**出力**:
- bool: 切断成功時True

**処理フロー**:
1. 現在の接続を確認
2. MT5をシャットダウン
3. リソースをクリーンアップ

## 使用例
```python
from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.common.config import get_config

# 接続マネージャーの取得（シングルトン）
manager = MT5ConnectionManager()

# 接続を確立
config = get_config()
if await manager.connect():
    print("MT5に接続しました")
    
    # アカウント情報を取得
    account_info = await manager.get_account_info()
    print(f"Balance: {account_info['balance']}")
    
    # リトライ付きで操作を実行
    result = await manager.execute_with_retry(
        some_mt5_operation,
        retry_count=3
    )
    
    # 切断
    await manager.disconnect()

# ヘルスチェッカーの使用
checker = HealthChecker(manager)
await checker.start()
# ... 処理 ...
await checker.stop()

# 接続プールの使用
pool = ConnectionPool(max_connections=5)
connection = await pool.acquire()
try:
    # 接続を使用
    pass
finally:
    await pool.release(connection)
```

## 注意事項
- **シングルトン**: MT5ConnectionManagerはアプリケーション全体で1インスタンス
- **非同期処理**: すべての接続操作は非同期で実行
- **自動再接続**: 接続断時は自動的に再接続を試行
- **リトライ機能**: 一時的な失敗に対して自動リトライ
- **ヘルスチェック**: 定期的な接続確認で安定性を向上
- **リソース管理**: 適切なクリーンアップで リソースリークを防止
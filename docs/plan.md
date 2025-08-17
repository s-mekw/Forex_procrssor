## 汎用実装計画（リンク集・導線テンプレート）

このファイルは、../.kiro/specs/Forex_procrssor/tasks.md に定義された任意のタスクを実装するための最小限の導線です。具体的な設計・仕様は各ドキュメントへ直接リンクし、本ファイルには詳細を書きません。

### タスク選択
- 対象タスクは `../.kiro/specs/Forex_procrssor/tasks.md` を参照し、対応するチェックリスト/説明/要件番号を確認してください。
- 現在の対象タスク: 
  - [ ] 3. MT5接続管理のテスト駆動実装
  - tests/unit/test_mt5_client.pyに接続成功/失敗/再接続のテストケースを作成
  - src/mt5_data_acquisition/mt5_client.pyにMT5ConnectionManagerクラスを実装
  - 指数バックオフによる再接続ロジックを実装（最大5回試行）
  - 接続プール管理とヘルスチェック機能を追加
  - _要件: 1.1_
- 
### 参照ドキュメント（必読）
- 実装タスク一覧: `../.kiro/specs/Forex_procrssor/tasks.md`
- 要件定義: `../.kiro/specs/Forex_procrssor/requirements.md`
- 詳細設計: `../.kiro/specs/Forex_procrssor/design.md`
- スペック概要: `../.kiro/specs/Forex_procrssor/spec.json`
- 技術方針: `../.kiro/steering/tech.md`
- 構造/モジュール方針: `../.kiro/steering/structure.md`
- Python開発ガイドライン: `../.kiro/steering/Python_Development_Guidelines.md`
- プロダクト方針: `../.kiro/steering/product.md`

### 実装の置き場所（指針のみ）
- 実装するディレクトリ/モジュールは `../.kiro/steering/structure.md` の方針に従い選定してください。
- 例: `src/common/`、`src/mt5_data_acquisition/`、`src/data_processing/`、`src/storage/`、`src/patchTST_model/`、`src/app/`、`src/production/` など（詳細は設計参照）。
  
### テストの置き場所（指針のみ）
- `tests/unit/`（ユニット）、`tests/integration/`（統合）、`tests/e2e/`（E2E）配下に配置。
- テスト観点・項目は各タスクの記述に従い、詳細は `../.kiro/specs/Forex_procrssor/design.md` および `requirements.md` を参照。

### 完了条件（DoD の参照）
- 当該タスクのチェック項目が満たされ、関連する要件の受け入れ条件に適合していること。
- ビルド/テストがグリーンであること（`pyproject.toml` の設定に準拠）。
- 
### 作業メモ欄（自由記述）
- ここには「選択タスク」「対象ファイル」「追加の参照リンク」「決定事項」などを簡潔に記録してください。

## 実装詳細計画

### テスト駆動開発（TDD）フロー
1. **テストを先に書く** - 失敗するテストから始める
2. **最小限の実装** - テストが通る最小限のコードを書く
3. **リファクタリング** - コードを改善する

### Step 1-4: テストケース作成（tests/unit/test_mt5_client.py）
```python
# 必要なテストケース
- test_connection_success: 接続成功シナリオ
- test_connection_failure: 接続失敗シナリオ
- test_connection_timeout: タイムアウトシナリオ
- test_reconnection_with_backoff: 指数バックオフ再接続
- test_max_retry_attempts: 最大5回の再試行
- test_terminal_info_logging: ターミナル情報のログ出力
- test_account_info_logging: アカウント情報のログ出力
```

### Step 5-7: MT5ConnectionManager実装（src/mt5_data_acquisition/mt5_client.py）
```python
class MT5ConnectionManager:
    # 設定
    - login: int
    - password: str
    - server: str
    - timeout: int = 30000
    - max_retries: int = 5
    - base_delay: float = 1.0
    
    # メソッド
    - async def connect() -> bool
    - async def disconnect() -> None
    - async def _exponential_backoff(attempt: int) -> float
    - def _log_terminal_info() -> None
    - def _log_account_info() -> None
```

### Step 8-9: 高度な機能実装
```python
# 接続プール管理
class ConnectionPool:
    - connections: Dict[str, MT5ConnectionManager]
    - max_connections: int = 5
    - async def get_connection(symbol: str) -> MT5ConnectionManager
    - async def release_connection(symbol: str) -> None

# ヘルスチェック
class HealthChecker:
    - check_interval: float = 10.0
    - async def start_monitoring() -> None
    - async def check_connection_health() -> HealthStatus
    - async def trigger_reconnection() -> None
```

### 依存ライブラリ
- MetaTrader5（MT5 API）
- pytest（テスト）
- pytest-asyncio（非同期テスト）
- unittest.mock（モック）
- structlog（構造化ログ）

### エラーハンドリング
```python
class MT5ConnectionError(Exception):
    """MT5接続エラーの基底クラス"""
    pass

class MT5TimeoutError(MT5ConnectionError):
    """接続タイムアウトエラー"""
    pass

class MT5AuthenticationError(MT5ConnectionError):
    """認証エラー"""
    pass
```
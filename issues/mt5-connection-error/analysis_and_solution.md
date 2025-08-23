# MT5接続エラー分析と解決策

## 問題概要

**発生日時**: 2025-08-22 23:17:10  
**エラーメッセージ**: `(-6, 'Terminal: Authorization failed')`  
**影響範囲**: test_sandbox/task4_test_SANDBOX内のすべてのテストスクリプト

## エラー分析

### 1. 根本原因

MT5接続時に`Authorization failed`エラーが発生。調査の結果、以下の2つの問題を特定：

1. **環境変数名の不一致**
   - `.env`ファイル: `FOREX_MT5_*`プレフィックスを使用
   - `test_config.py`: `MT5_DEMO_*`プレフィックスを期待
   
2. **MT5初期化方法の問題**
   - Axiory MT5では`mt5.initialize()`にログイン情報を同時に渡す必要がある
   - 現在の実装は2段階（initialize → login）で実行

### 2. 調査で判明した事実

#### 環境変数の状態
```
# .envファイル（存在する）
FOREX_MT5_LOGIN=20046505
FOREX_MT5_PASSWORD=***MASKED***
FOREX_MT5_SERVER=Axiory-Demo
FOREX_MT5_TIMEOUT=60000
```

#### MT5実行ファイルの場所
- パス: `C:\Program Files\Axiory MetaTrader 5\terminal64.exe`
- ファイルは存在し、アクセス可能

#### 初期化方法の違い
```python
# 失敗するパターン
mt5.initialize()  # または mt5.initialize(path=path)
mt5.login(login=login, password=password, server=server)

# 成功するパターン（Axiory MT5）
mt5.initialize(path=path, login=login, password=password, server=server)
```

## 実施済みの修正

### test_config.pyの修正
```python
# 変更前
MT5_CONFIG = {
    "login": int(os.getenv("MT5_DEMO_LOGIN", "5025869601")),
    "password": os.getenv("MT5_DEMO_PASSWORD", ""),
    "server": os.getenv("MT5_DEMO_SERVER", "MetaQuotes-Demo"),
    "timeout": 60000,
}

# 変更後
MT5_CONFIG = {
    "login": int(os.getenv("FOREX_MT5_LOGIN", "20046505")),
    "password": os.getenv("FOREX_MT5_PASSWORD", ""),
    "server": os.getenv("FOREX_MT5_SERVER", "Axiory-Demo"),
    "timeout": int(os.getenv("FOREX_MT5_TIMEOUT", "60000")),
    "path": r"C:\Program Files\Axiory MetaTrader 5\terminal64.exe",
}
```

## 必要な追加修正（src/mt5_data_acquisition/mt5_client.py）

### 修正プラン

`MT5ConnectionManager.connect()`メソッドの修正が必要。

#### 現在の実装（問題あり）
```python
def connect(self, config: dict[str, Any]) -> bool:
    try:
        # 1. MT5の初期化
        path = config.get("path")
        if path:
            init_result = mt5.initialize(path=path)
        else:
            init_result = mt5.initialize()
        
        if not init_result:
            error = mt5.last_error()
            self.logger.error(f"MT5の初期化に失敗しました: {error}")
            return False
        
        # 2. MT5へのログイン（別途実行）
        login_result = mt5.login(
            login=login, password=password, server=server, timeout=timeout
        )
```

#### 提案する修正
```python
def connect(self, config: dict[str, Any]) -> bool:
    try:
        # 認証情報を取得
        login = config.get("account")
        password = config.get("password")
        server = config.get("server")
        timeout = config.get("timeout", self.DEFAULT_TIMEOUT)
        path = config.get("path")

        # MT5の初期化とログインを同時に実行
        # Axiory MT5などでは、この方法が必須の場合がある
        if path and login and password and server:
            # すべての情報がある場合は、初期化時にログイン情報も渡す
            init_result = mt5.initialize(
                path=path,
                login=login,
                password=password,
                server=server,
                timeout=timeout
            )
        elif path:
            # パスのみ指定されている場合
            init_result = mt5.initialize(path=path)
        else:
            # パスが指定されていない場合（デフォルトパス使用）
            init_result = mt5.initialize()

        if not init_result:
            error = mt5.last_error()
            self.logger.error(f"MT5の初期化に失敗しました: {error}")
            return False

        # 初期化のみ成功した場合、別途ログインを試行
        # （initialize時にログイン情報を渡した場合は、すでにログイン済み）
        if not (path and login and password and server):
            login_result = mt5.login(
                login=login, password=password, server=server, timeout=timeout
            )
            if not login_result:
                error = mt5.last_error()
                self.logger.error(f"MT5へのログインに失敗しました: {error}")
                mt5.shutdown()
                return False
```

### 修正の利点

1. **互換性の維持**: 既存の2段階初期化も引き続きサポート
2. **Axiory MT5対応**: 特定のブローカーの要件に対応
3. **柔軟性**: パス、ログイン情報の有無に応じた適切な処理

## テスト結果

### デバッグスクリプトでの検証結果
```python
# test_mt5_advanced.pyの実行結果
=== Attempt 1: Initialize without path ===
Result: False
Error: (-6, 'Terminal: Authorization failed')

=== Attempt 2: Initialize with path ===
Result: False
Error: (-6, 'Terminal: Authorization failed')

=== Attempt 3: Initialize + Login ===
Result: True  # ✅ 成功
Success!
Terminal info:
  Company: Axiory Global Ltd.
  Connected: True
Account info:
  Login: 20046505
  Balance: 1000000.0
```

## 推奨される今後の対応

1. **短期対応**
   - `mt5_client.py`の`connect`メソッドを修正
   - すべてのテストスクリプトで動作確認

2. **中期対応**
   - 環境変数名の統一（`FOREX_MT5_*`に統一）
   - `.env.example`ファイルの作成
   - エラーハンドリングの改善

3. **長期対応**
   - 複数のMT5ブローカーに対応できる柔軟な設計
   - 接続設定のバリデーション強化
   - より詳細なエラーメッセージの提供

## セキュリティ考慮事項

- パスワードや認証情報は常にマスクして表示
- `.env`ファイルは`.gitignore`に含める
- デバッグログでも認証情報は出力しない
- 本番環境では環境変数または安全な認証情報管理システムを使用

## 参考資料

### MetaTrader5パッケージのinitialize関数
```python
def initialize(
    path: str = None,
    login: int = None,
    password: str = None,
    server: str = None,
    timeout: int = None,
    portable: bool = False
) -> bool:
    """
    MT5ターミナルを初期化し、必要に応じてログインも実行
    すべてのパラメータはオプショナル
    """
```

### エラーコード
- `-6`: Terminal: Authorization failed（認証失敗）
- `-1`: Terminal: Invalid parameters（パラメータ不正）
- `-2`: Terminal: Internal error（内部エラー）
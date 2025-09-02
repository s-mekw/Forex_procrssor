# MT5接続エラー修正 実装報告書

**実装日時**: 2025-08-23  
**実装者**: Claude Code  
**対象エラー**: `(-6, 'Terminal: Authorization failed')`  
**影響範囲**: test_sandbox/task4_test_SANDBOX内のすべてのテストスクリプト

## 実装概要

Axiory MT5への接続時に発生していた`Authorization failed`エラーを解決するため、MT5の初期化方法を改善し、関連する設定ファイルとテストスクリプトを修正しました。

## 実装内容

### 1. MT5ConnectionManager.connect()メソッドの改善

**ファイル**: `src/mt5_data_acquisition/mt5_client.py` (85-180行目)

#### 変更前の問題点
```python
# 初期化とログインを別々に実行（Axiory MT5では失敗）
if path:
    init_result = mt5.initialize(path=path)
else:
    init_result = mt5.initialize()

# 別途ログインを試行
login_result = mt5.login(login=login, password=password, server=server, timeout=timeout)
```

#### 実装した修正
```python
# Axiory MT5対応: 認証情報がすべて揃っている場合は初期化時に同時に渡す
if path and login and password and server:
    # すべての情報がある場合は、初期化時にログイン情報も渡す
    init_result = mt5.initialize(
        path=path,
        login=login,
        password=password,
        server=server,
        timeout=timeout
    )
    need_separate_login = False
elif path:
    # パスのみ指定されている場合（既存の処理を維持）
    init_result = mt5.initialize(path=path)
    need_separate_login = True
else:
    # デフォルトパス使用（既存の処理を維持）
    init_result = mt5.initialize()
    need_separate_login = True

# 必要に応じて別途ログイン
if need_separate_login:
    login_result = mt5.login(login=login, password=password, server=server, timeout=timeout)
```

**改善点**:
- Axiory MT5の要件に対応（初期化時に認証情報を同時に渡す）
- 既存の2段階初期化との後方互換性を維持
- 他のブローカーとの互換性も確保

### 2. test_config.pyの修正

**ファイル**: `test_sandbox/task4_test_SANDBOX/utils/test_config.py`

#### 2.1 MT5パスパラメータの追加
```python
def get_mt5_config() -> dict:
    """MT5ConnectionManager用の設定を取得"""
    return {
        "account": MT5_CONFIG["login"],
        "password": MT5_CONFIG["password"],
        "server": MT5_CONFIG["server"],
        "timeout": MT5_CONFIG["timeout"],
        "path": MT5_CONFIG["path"],  # Axiory MT5のパスを追加
    }

def get_mt5_credentials() -> dict:
    """MT5認証情報を取得"""
    return {
        "account": MT5_CONFIG["login"],
        "password": MT5_CONFIG["password"],
        "server": MT5_CONFIG["server"],
        "timeout": MT5_CONFIG["timeout"],
        "path": MT5_CONFIG["path"],  # Axiory MT5のパスを追加
    }
```

#### 2.2 TickDataStreamerパラメータの修正
```python
# ストリーミング設定
STREAMING_CONFIG = {
    "buffer_size": 1000,
    "spike_threshold_percent": 0.1,  # spike_threshold → spike_threshold_percent
    "backpressure_threshold": 0.8,
    # stats_window_sizeパラメータを削除（TickDataStreamerで使用されない）
}
```

### 3. TickDataStreamerの初期化バグ修正

**ファイル**: `src/mt5_data_acquisition/tick_fetcher.py` (504-515行目)

#### stats属性の初期化追加
```python
def __init__(self, ...):
    # ... 既存の初期化コード ...
    
    # 新しいティック追跡用のインデックス
    self._last_read_index: int = 0
    self._tick_counter: int = 0
    
    # 統計情報の初期化（追加）
    self.stats: dict[str, Any] = {
        "mean": 0.0,
        "std": 0.0,
        "sample_count": 0,
        "spike_count": 0,
        "last_update": None,
    }
```

### 4. テストスクリプトの更新

**ファイル**: `test_sandbox/task4_test_SANDBOX/01_basic_connection_test.py` (151-155行目)

#### 新しいAPIに対応
```python
# 変更前（エラーが発生）
stats_table.add_row("Symbol", streamer.symbol)
stats_table.add_row("Buffer Size", str(streamer.buffer_size))
stats_table.add_row("Spike Threshold", f"{streamer.spike_threshold}σ")

# 変更後（正しいAPI使用）
stats_table.add_row("Symbol", streamer.config.symbol)
stats_table.add_row("Buffer Size", str(streamer.config.buffer_size))
stats_table.add_row("Spike Threshold", f"{streamer.config.spike_threshold_percent:.1f}%")
stats_table.add_row("Backpressure Threshold", f"{streamer.config.backpressure_threshold * 100:.0f}%")
stats_table.add_row("Circuit Breaker", "Open" if streamer.circuit_breaker.is_open else "Closed")
```

## テスト結果

### 1. 基本接続テスト (`01_basic_connection_test.py`)
```
✅ Successfully connected to MT5
✅ Account Information取得成功
✅ Available Symbols取得成功
✅ Symbol Details取得成功
✅ Tick Streamer初期化成功
✅ All tests completed successfully
```

### 2. リアルタイムティック表示テスト (`02_real_time_tick_display.py`)
```
✅ Connected to Axiory-Demo
✅ Subscribed to EURJPY
✅ Streaming started successfully
```

## 技術的な洞察

### 1. ブローカー固有の実装差異
- **MetaQuotes-Demo**: 2段階初期化（initialize → login）で動作
- **Axiory MT5**: 初期化時に認証情報を同時に渡す必要がある
- この違いは、MT5ターミナルのビルドやブローカーのカスタマイズに起因

### 2. 環境変数の命名規則
- プロジェクト全体で`FOREX_MT5_*`プレフィックスに統一
- 一貫性のある設定管理を実現

### 3. APIの進化と後方互換性
- TickDataStreamerのAPIが進化（個別プロパティ → configオブジェクト）
- 後方互換性を保ちながら、より構造化されたアプローチを採用

## セキュリティ考慮事項

1. **認証情報の保護**
   - パスワードは環境変数から読み込み
   - ログ出力時はマスク処理
   - `.env`ファイルは`.gitignore`に含める

2. **エラーメッセージの取り扱い**
   - 詳細なエラー情報はログに記録
   - ユーザー向けメッセージは汎用的な内容に

## 今後の推奨事項

### 短期的改善
1. エラーリカバリーメカニズムの強化
2. 接続タイムアウト処理の改善
3. より詳細なログ出力の実装

### 中長期的改善
1. 複数ブローカー対応の設定管理システム
2. 接続プロファイルの動的切り替え機能
3. 自動再接続とフェイルオーバー機能

## まとめ

MT5接続エラー`(-6, 'Terminal: Authorization failed')`を完全に解決しました。実装した修正により：

1. **Axiory MT5への接続が成功**
2. **既存のコードとの後方互換性を維持**
3. **将来の拡張性を考慮した設計**

この修正により、プロジェクトはAxiory MT5を使用したリアルタイムデータ取得が可能になり、開発を継続できる状態になりました。
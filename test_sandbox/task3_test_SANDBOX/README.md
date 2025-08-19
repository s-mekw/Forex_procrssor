# MT5接続サンドボックス

このディレクトリには、MT5データ取得モジュールを使用した実際のMT5接続をテストするためのサンドボックスコードが含まれています。

## ファイル構成

- `mt5_connection_sandbox.py` - 基本的なMT5接続テスト用サンドボックス
- `mt5_config_example.py` - MT5接続設定の例とヘルパー関数
- `mt5_sandbox_with_config.py` - 設定ファイル対応版のMT5接続サンドボックス
- `README.md` - このファイル

## 前提条件

1. **MetaTrader 5**がインストールされていること
2. **MetaTrader5 Pythonパッケージ**がインストールされていること
   ```bash
   pip install MetaTrader5
   ```
3. **MT5アカウント**（デモまたはライブ）があること

## 使用方法

### 1. 設定ファイルの作成

まず、MT5接続設定を作成します：

```bash
cd tests/test_sandbox
python mt5_config_example.py
```

対話的に設定を入力するか、手動で`mt5_config.json`ファイルを作成します：

```json
{
  "account": 12345678,
  "password": "your_password",
  "server": "YourBroker-Demo",
  "timeout": 60000,
  "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
}
```

### 2. 基本的な接続テスト

設定ファイルを使用した接続テスト：

```bash
python mt5_sandbox_with_config.py
```

### 3. 直接設定での接続テスト

コード内で直接設定を指定した接続テスト：

```bash
python mt5_connection_sandbox.py
```

## 機能

### MT5ConnectionSandbox / MT5SandboxWithConfig

- **接続管理**: MT5への接続・切断
- **ヘルスチェック**: 接続の健全性確認
- **シンボル情報取得**: 通貨ペアの詳細情報
- **OHLCデータ取得**: 履歴価格データ
- **ティックデータ取得**: リアルタイムティックデータ
- **再接続機能**: 自動再接続テスト

### MT5ConfigHelper

- **設定ファイル管理**: JSON形式での設定保存・読み込み
- **設定検証**: 接続設定の妥当性チェック
- **設定例提供**: デモ・ライブ用の設定テンプレート

## 実行例

### 接続テストの実行

```python
from mt5_sandbox_with_config import MT5SandboxWithConfig

# サンドボックスの作成
sandbox = MT5SandboxWithConfig()

# 設定ファイルの読み込み
if sandbox.load_config():
    # MT5への接続
    if sandbox.connect_to_mt5():
        # ヘルスチェック
        sandbox.health_check()
        
        # シンボル情報の取得
        symbols_info = sandbox.get_symbols_info(['EURUSD', 'GBPUSD'])
        
        # OHLCデータの取得
        ohlc_data = sandbox.get_ohlc_data('EURUSD', mt5.TIMEFRAME_H1, 10)
        
        # ティックデータの取得
        tick_data = sandbox.get_tick_data('EURUSD', 5)
        
        # 切断
        sandbox.disconnect()
```

### 設定ファイルの作成

```python
from mt5_config_example import MT5ConfigHelper

helper = MT5ConfigHelper()

# デモ用設定の作成
demo_config = helper.create_demo_config()
helper.save_config(demo_config)

# 設定の読み込み
config = helper.load_config()
```

## トラブルシューティング

### よくある問題

1. **MT5パッケージが見つからない**
   ```
   pip install MetaTrader5
   ```

2. **接続に失敗する**
   - アカウント番号、パスワード、サーバー名を確認
   - MT5が起動していることを確認
   - ネットワーク接続を確認

3. **シンボルが見つからない**
   - シンボル名のスペルを確認
   - ブローカーで利用可能なシンボルか確認

4. **データが取得できない**
   - 取引時間外の可能性
   - シンボルが有効か確認

### ログの確認

すべてのサンドボックスは詳細なログを出力します。エラーが発生した場合は、ログメッセージを確認してください。

## 注意事項

- **デモアカウントの使用を推奨**: テスト時はデモアカウントを使用してください
- **パスワードの管理**: 設定ファイルにパスワードを含める場合は、適切に保護してください
- **取引時間**: データ取得は取引時間内に行ってください
- **レート制限**: 過度なデータ取得は避けてください

## カスタマイズ

サンドボックスコードは、プロジェクトの要件に合わせてカスタマイズできます：

- 新しいデータ取得機能の追加
- エラーハンドリングの改善
- ログレベルの調整
- 設定の拡張

## ライセンス

このサンドボックスコードは、プロジェクトの開発・テスト目的で使用してください。

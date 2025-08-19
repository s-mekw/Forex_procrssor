# MT5 Tick Fetcher サンドボックステスト

このディレクトリには、MT5接続とティックデータストリーミング機能を視覚的に確認・テストするためのサンドボックステストが含まれています。

## 📋 概要

`src/mt5_data_acquisition/tick_fetcher.py`の実装を使用して、実際のMT5接続を通じてリアルタイムデータを取得し、様々な側面から動作を確認できます。

## 🚀 クイックスタート

### 前提条件

1. MT5がインストールされていること
2. デモアカウントまたはライブアカウントの認証情報
3. Python環境（uv推奨）

### 環境設定

`.env`ファイルをプロジェクトルートに作成：

```env
MT5_DEMO_LOGIN=your_demo_login
MT5_DEMO_PASSWORD=your_demo_password
MT5_DEMO_SERVER=MetaQuotes-Demo
```

## 📁 ファイル構成

### ユーティリティ
- `utils/test_config.py` - テスト用の共通設定
- `utils/display_helpers.py` - 表示用ヘルパー関数（richライブラリ使用）

### CLIテスト

#### 1. `01_basic_connection_test.py`
基本的な接続テストと情報表示

```bash
uv run python test_sandbox/task4_test_SANDBOX/01_basic_connection_test.py
```

**機能：**
- MT5への接続確認
- アカウント情報の表示
- 利用可能なシンボル一覧
- ストリーマーの初期化テスト

#### 2. `02_real_time_tick_display.py`
リアルタイムティックデータの視覚的表示

```bash
uv run python test_sandbox/task4_test_SANDBOX/02_real_time_tick_display.py
```

**機能：**
- ライブティックデータの受信と表示
- 価格変動の色分け表示（上昇:緑、下降:赤）
- 統計情報のリアルタイム更新
- スパイク検出の可視化

#### 3. `03_buffer_monitoring.py`
バッファ管理とパフォーマンス監視

```bash
uv run python test_sandbox/task4_test_SANDBOX/03_buffer_monitoring.py
```

**機能：**
- バッファ使用率のプログレスバー表示
- バックプレッシャー発生の監視
- メモリ使用量とCPU使用率の追跡
- スループット測定（ティック/秒）

#### 4. `04_error_recovery_test.py`
エラー処理と自動回復機能のテスト

```bash
uv run python test_sandbox/task4_test_SANDBOX/04_error_recovery_test.py
```

**機能：**
- 接続エラーのシミュレーション
- 自動再接続の動作確認
- Circuit Breakerのテスト
- エラー統計とリカバリー成功率の表示

### Marimo WebUIテスト

#### 5. `05_marimo_dashboard.py`
インタラクティブなリアルタイムダッシュボード

```bash
uv run marimo run test_sandbox/task4_test_SANDBOX/05_marimo_dashboard.py
```

**機能：**
- リアルタイム価格チャート（Plotly）
- スプレッドグラフ
- 統計ゲージ（バッファ使用率、スパイク率、ドロップ率）
- インタラクティブなコントロール
  - シンボル選択
  - バッファサイズ調整
  - スパイク閾値設定

#### 6. `06_marimo_multi_symbol.py`
複数シンボルの同時監視と比較

```bash
uv run marimo run test_sandbox/task4_test_SANDBOX/06_marimo_multi_symbol.py
```

**機能：**
- 複数通貨ペアの同時監視
- 比較モード：
  - 正規化価格比較
  - スプレッド分布
  - ボラティリティ分析
  - 相関行列ヒートマップ
- パフォーマンスメトリクス表示

## 🎯 主な特徴

### リアルタイムモニタリング
- ティックデータのライブストリーミング
- 動的な統計情報更新
- 視覚的なフィードバック

### エラー処理
- 自動再接続機能
- Circuit Breaker実装
- エラーログとリカバリー統計

### パフォーマンス最適化
- バックプレッシャー管理
- メモリプール使用
- 効率的なバッファリング

### スパイク検出
- 3σルールによる異常値検出
- リアルタイムフィルタリング
- 統計的分析

## 🔧 トラブルシューティング

### 接続エラー
1. MT5が起動していることを確認
2. 認証情報が正しいことを確認
3. ネットワーク接続を確認

### データが表示されない
1. 市場が開いている時間帯か確認（Forex市場は週末休場）
2. 選択したシンボルが利用可能か確認
3. デモアカウントの有効期限を確認

### Marimoが起動しない
```bash
# Marimoの再インストール
uv pip install --force-reinstall marimo

# ポートを指定して起動
uv run marimo run --port 8080 05_marimo_dashboard.py
```

## 📈 パフォーマンス目安

- **ティック処理能力**: 1000+ ticks/秒
- **レイテンシ**: < 10ms（ローカル環境）
- **メモリ使用量**: 50-100MB（通常動作時）
- **CPU使用率**: 5-15%（シングルシンボル）

## 🔍 デバッグ

詳細なログを有効化：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 関連ドキュメント

- [tick_fetcher.py API](../../docs/api/tick_fetcher.md)
- [トラブルシューティングガイド](../../docs/troubleshooting/tick_fetcher.md)
- [サンプルコード](../../docs/examples/tick_streaming_example.py)

## ⚠️ 注意事項

- これらのテストは**実際のMT5接続**を使用します
- デモアカウントの使用を推奨
- ライブアカウントで実行する場合は注意してください
- 長時間実行する場合はメモリ使用量を監視してください

  🎯 推奨テスト順序

  Phase 1: 基礎確認

  # 1. 基本接続テスト（最初に実行）
  uv run python test_sandbox/task4_test_SANDBOX/01_basic_connection_test.py
  目的： MT5接続、認証情報、利用可能シンボルの確認

  Phase 2: 動作確認

  # 2. リアルタイムティック表示（30秒程度）
  uv run python test_sandbox/task4_test_SANDBOX/02_real_time_tick_display.py
  目的： データストリーミングの基本動作確認

  Phase 3: パフォーマンス検証

  # 3. バッファモニタリング（30秒程度）
  uv run python test_sandbox/task4_test_SANDBOX/03_buffer_monitoring.py
  目的： メモリ使用量、スループット、バックプレッシャーの確認

  Phase 4: 信頼性検証

  # 4. エラー回復テスト（少し長め）
  uv run python test_sandbox/task4_test_SANDBOX/04_error_recovery_test.py
  目的： 接続断、自動再接続、Circuit Breakerの動作確認

  Phase 5: WebUI体験

  # 5. Marimoダッシュボード
  uv run marimo run test_sandbox/task4_test_SANDBOX/05_marimo_dashboard.py
  # ブラウザで http://localhost:2718 にアクセス

  # 6. 複数シンボル比較（余裕があれば）
  uv run marimo run test_sandbox/task4_test_SANDBOX/06_marimo_multi_symbol.py

  ⚠️ 事前準備

  # 環境変数の設定（.envファイル作成）
  echo "MT5_DEMO_LOGIN=your_demo_login" > .env
  echo "MT5_DEMO_PASSWORD=your_password" >> .env
  echo "MT5_DEMO_SERVER=MetaQuotes-Demo" >> .env

  💡 各段階での確認ポイント

  1. 接続テスト: エラーなく完了するか
  2. リアルタイム表示: ティックが流れているか、色の変化
  3. バッファ監視: メモリ使用量が適切か、ドロップが少ないか
  4. エラー回復: 自動回復が機能しているか
  5. WebUI: インタラクティブな操作ができるか
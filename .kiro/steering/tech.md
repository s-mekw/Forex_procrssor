# Technology Stack

## Architecture

### システムアーキテクチャ
- **設計パターン**: モジュラーモノリス
- **主要原則**: 依存性注入（DI）、単一責任の原則、設定の外部化
- **データフロー**: MT5 → データ処理 → ML推論 → ストレージ → ダッシュボード
- **処理モデル**: イベントドリブン + バッチ処理のハイブリッド

### コンポーネント構成
1. **データ取得層**: MT5クライアント
2. **処理層**: Polarsベースの高速データ処理
3. **ML層**: PatchTSTモデルによる予測
4. **永続化層**: InfluxDB時系列データベース
5. **プレゼンテーション層**: Dashウェブアプリケーション

## Backend

### プログラミング言語
- **Python 3.11+**: メイン開発言語
- **型ヒント**: 全コードで必須

### 主要フレームワーク・ライブラリ
- **MetaTrader5**: FXデータ取得API
- **Polars**: 高速データフレーム処理（Pandasは使用禁止）
- **polars-ta-extension**: テクニカル指標計算
- **PyTorch + CUDA**: 機械学習フレームワーク
- **neuralforecast**: 時系列予測ライブラリ
  - PatchTSTの運用方針: 1分粒度の入力から1〜15分先をマルチホライズン予測。入力はOHLC＋指標（Volume除外）。出力はデフォルトで終値のみ＋80%/90%/95%の分位区間
- **InfluxDB**: 時系列データベース
- **Redis**: クエリキャッシュ
- **Dash**: Webダッシュボードフレームワーク
- **dash-extensions**: WebSocket統合
- **Prometheus**: システム監視・メトリクス

### パッケージ管理
- **uv**: Pythonパッケージマネージャー（pip使用禁止）

## Development Environment

### 必須ツール
```bash
# uvのインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# プロジェクトセットアップ
uv sync
```

### 開発ツール
- **Ruff**: コードフォーマッター・リンター
- **pytest**: テストフレームワーク
- **pre-commit**: Git hooks管理
- **TensorBoard**: MLモデル可視化

### IDE推奨設定
- VSCode + Python拡張
- 型チェック: strict mode
- フォーマッター: Ruff

## Common Commands

### 開発コマンド
```bash
# 予測モデル依存の追加
uv add neuralforecast pandas pytorch-lightning

# 依存関係インストール
uv sync

# パッケージ追加
uv add <package-name>

# 開発用パッケージ追加
uv add --dev <package-name>

# コードフォーマット
uv run --frozen ruff format .

# リント実行
uv run --frozen ruff check .

# リント自動修正
uv run --frozen ruff check . --fix

# テスト実行
uv run --frozen pytest

# テスト（カバレッジ付き）
uv run --frozen pytest --cov
```

### アプリケーション実行
```bash
# MT5データ取得開始
uv run python -m src.mt5_data_acquisition.main

# ダッシュボード起動
uv run python -m src.app.main

# ML学習実行
uv run python -m src.patchTST_model.train

# バックテスト実行
uv run python -m src.backtesting.run
```

## Environment Variables

### 必須環境変数
```env
# MT5接続情報
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server

# InfluxDB設定
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_token
INFLUXDB_ORG=your_org
INFLUXDB_BUCKET=fx_data

# Redis設定
REDIS_HOST=localhost
REDIS_PORT=6379

# アプリケーション設定
DASH_HOST=0.0.0.0
DASH_PORT=8050
DEBUG_MODE=false
```

### オプション環境変数
```env
# ML設定
CUDA_VISIBLE_DEVICES=0
MODEL_CHECKPOINT_DIR=./checkpoints

# ロギング
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

## Port Configuration

### サービスポート
- **8050**: Dashダッシュボード（Web UI）
- **8086**: InfluxDB API
- **6379**: Redisサーバー
- **9090**: Prometheusサーバー
- **6006**: TensorBoard（ML可視化）

### 内部通信
- **5555**: WebSocketサーバー（リアルタイム更新）
- **5556**: データ処理パイプライン

### 外部接続
- **MT5**: ブローカー指定のサーバーポート
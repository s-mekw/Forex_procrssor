# Project Structure

## Root Directory Organization

```
Route_to_Botter_gamma/
├── src/                      # ソースコード
├── tests/                    # テストファイル
├── docs/                     # 技術ドキュメント
├── data/                     # ローカルデータ（gitignore）
├── models/                   # 学習済みモデル（gitignore）
├── logs/                     # ログファイル（gitignore）
├── .kiro/                    # Kiroステアリング・仕様
├── implementation_plan/      # 実装計画ドキュメント
├── pyproject.toml           # プロジェクト設定
├── .env.example             # 環境変数テンプレート
├── .gitignore               # Git除外設定
├── .pre-commit-config.yaml  # Pre-commitフック設定
├── CLAUDE.md                # Claude Code指示
└── README.md                # プロジェクト概要
```

## Subdirectory Structures

### src/ - アプリケーションコード
```
src/
├── __init__.py
├── common/                   # 共通ユーティリティ
│   ├── __init__.py
│   ├── config.py            # 設定管理
│   ├── logger.py            # ロギング設定
│   └── exceptions.py        # カスタム例外
│
├── mt5_data_acquisition/     # 機能1: データ取得
│   ├── __init__.py
│   ├── base.py              # 基底クラス
│   ├── mt5_client.py        # MT5接続管理
│   ├── ohlc_fetcher.py      # OHLC取得
│   ├── tick_fetcher.py      # ティック取得
│   ├── tick_to_bar.py       # バー変換
│   └── main.py              # エントリーポイント
│
├── data_processing/          # 機能2: データ処理
│   ├── __init__.py
│   ├── processor.py         # メイン処理クラス
│   ├── indicators.py        # テクニカル指標
│   ├── rci.py              # RCI計算エンジン
│   └── pipelines.py         # 処理パイプライン
│
├── storage/                  # 機能3: データ保存
│   ├── __init__.py
│   ├── base.py              # ストレージ基底クラス
│   ├── influx_handler.py    # InfluxDB操作
│   ├── parquet_handler.py   # Parquetファイル操作
│   └── query_engine.py      # データクエリ
│
├── patchTST_model/          # 機能4: ML予測
│   ├── __init__.py
│   ├── datasets.py          # データセット準備
│   ├── model.py             # PatchTSTモデル定義
│   ├── train.py             # 学習スクリプト
│   ├── predict.py           # 推論エンジン
│   └── evaluate.py          # モデル評価
│
├── app/                     # 機能5: ダッシュボード
│   ├── __init__.py
│   ├── main.py              # Dashアプリケーション
│   ├── layouts.py           # UI レイアウト
│   ├── callbacks.py         # インタラクティブ機能
│   ├── components/          # UIコンポーネント
│   └── assets/              # 静的ファイル
│
└── production/              # 機能6: 本番運用
    ├── __init__.py
    ├── monitoring.py        # システム監視
    ├── alerts.py            # アラート管理
    └── health_check.py      # ヘルスチェック
```

### tests/ - テストコード
```
tests/
├── __init__.py
├── conftest.py              # pytest設定
├── unit/                    # ユニットテスト
│   ├── test_mt5_client.py
│   ├── test_indicators.py
│   └── test_rci.py
├── integration/             # 統合テスト
│   ├── test_data_pipeline.py
│   └── test_storage.py
└── fixtures/                # テストデータ
    └── sample_data.parquet
```

## Code Organization Patterns

### モジュール設計原則
- **単一責任**: 各モジュールは1つの明確な責任を持つ
- **依存性逆転**: 抽象に依存し、具象に依存しない
- **インターフェース分離**: 必要最小限のインターフェースを定義

### レイヤー構造
1. **プレゼンテーション層**: Dashアプリケーション
2. **ビジネスロジック層**: データ処理、ML推論
3. **データアクセス層**: MT5、InfluxDB接続
4. **共通層**: ユーティリティ、設定、ロギング

## File Naming Conventions

### Pythonファイル
- **モジュール**: `snake_case.py`（例: `mt5_client.py`）
- **クラス定義**: ファイル名と対応（例: `mt5_client.py` → `class MT5Client`）
- **テスト**: `test_<module_name>.py`（例: `test_mt5_client.py`）

### 設定ファイル
- **環境変数**: `.env`（本番）、`.env.example`（テンプレート）
- **Python設定**: `config.py`（アプリケーション設定）
- **プロジェクト設定**: `pyproject.toml`

### データファイル
- **時系列データ**: `YYYYMMDD_<pair>_<timeframe>.parquet`
- **モデル**: `model_<version>_<timestamp>.pth`
- **ログ**: `app_YYYYMMDD.log`

## Import Organization

### インポート順序
```python
# 1. 標準ライブラリ
import os
import sys
from datetime import datetime

# 2. サードパーティライブラリ
import polars as pl
import torch
from influxdb_client import InfluxDBClient

# 3. ローカルモジュール
from src.common.config import Config
from src.mt5_data_acquisition.mt5_client import MT5Client
```

### 相対/絶対インポート
- **プロジェクト内**: 絶対インポート（`from src.module import`）
- **パッケージ内**: 相対インポート（`from .module import`）

## Key Architectural Principles

### データ処理
- **Polars優先**: 全てのデータフレーム操作でPolarsを使用
- **遅延評価**: 可能な限りLazyFrameを活用
- **型安全性**: 全関数で型ヒントを使用

### エラーハンドリング
- **早期失敗**: 問題を早期に検出して失敗
- **明示的エラー**: カスタム例外で意図を明確化
- **復旧可能性**: 可能な限り自動復旧を実装

### パフォーマンス
- **非同期処理**: I/O操作は非同期で実装
- **バッチ処理**: 大量データは適切なバッチサイズで処理
- **キャッシング**: 頻繁にアクセスするデータはキャッシュ

### セキュリティ
- **認証情報**: 環境変数で管理、コードにハードコーディングしない
- **入力検証**: 全ての外部入力を検証
- **最小権限**: 各コンポーネントは必要最小限の権限で動作
- **設定の外部化**: パラメータは設定ファイルで管理し、コードから分離する

### テスト可能性
- **依存性注入**: テスト時にモックを注入可能に
- **純粋関数**: 副作用を最小限に
- **境界明確化**: 外部システムとの境界を明確に定義
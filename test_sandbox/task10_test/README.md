# RealtimePipeline E2Eテストスイート

Task 10.1で開発したRealtimePipelineクラスの包括的なEnd-to-Endテストスイートです。

## 概要

このテストスイートは、実際のFX取引環境を模擬した5つのシナリオテストを提供し、RealtimePipelineの信頼性、パフォーマンス、回復力を検証します。

## テストシナリオ

### 1. リアルタイムFX市場シミュレーション (`01_realtime_fx_market_simulation.py`)

**目的**: 実際の市場環境での動作検証

**テスト内容**:
- 市場オープン時の高頻度データ処理
- 通常取引時の継続的なデータフロー
- ニュースリリース時のデータバースト
- 高ボラティリティ市場での処理
- 遅延プロバイダーのアラート機能

**成功基準**:
- 成功率: 95%以上
- 平均遅延: 1秒未満
- 安定性スコア: 80/100以上

### 2. マルチプロバイダー統合 (`02_multi_provider_integration.py`)

**目的**: 複数データソースの統合管理

**テスト内容**:
- 3つのプロバイダー（Premium/Standard/Backup）の優先度処理
- プロバイダー障害時のフェイルオーバー
- データ重複排除機能
- データ整合性チェック

**成功基準**:
- 成功率: 90%以上
- 重複排除機能の正常動作
- 全プロバイダーの稼働確認

### 3. バックプレッシャー制御と回復 (`03_backpressure_recovery_scenario.py`)

**目的**: システムの自己防衛と回復力検証

**テスト内容**:
- 段階的な負荷増加シミュレーション
- アラートエスカレーション（medium→high→critical）
- 自動負荷調整機能
- システム回復プロセス

**成功基準**:
- バックプレッシャーイベントの発生
- アラートの適切な発出
- 回復率: 95%以上
- 安定性スコア: 70/100以上

### 4. 24時間連続稼働シミュレーション (`04_24h_continuous_operation.py`)

**目的**: 長時間安定稼働の検証（時間圧縮版）

**テスト内容**:
- 市場時間帯別の負荷変動（アジア→欧州→米国）
- メモリリーク検出
- パフォーマンス劣化監視
- システムリソース使用状況追跡

**成功基準**:
- メモリリークなし（50MB未満の増加）
- 成功率: 98%以上
- 安定性スコア: 85/100以上

### 5. 障害復旧とデータ整合性 (`05_disaster_recovery_scenario.py`)

**目的**: 障害時の処理継続性とデータ保全

**テスト内容**:
- 突然のシャットダウンと再起動
- データ破損シミュレーション
- ネットワーク分断と復旧
- リカバリー後のパフォーマンス検証

**成功基準**:
- データ整合性スコア: 80%以上
- リカバリー後のパフォーマンス劣化: 20%以内
- シーケンス順序の維持

## ディレクトリ構成

```
test_sandbox/task10_test/
├── README.md                              # このファイル
├── 01_realtime_fx_market_simulation.py    # リアルタイム市場シミュレーション
├── 02_multi_provider_integration.py       # マルチプロバイダー統合
├── 03_backpressure_recovery_scenario.py   # バックプレッシャー制御
├── 04_24h_continuous_operation.py         # 24時間連続稼働
├── 05_disaster_recovery_scenario.py       # 障害復旧シナリオ
├── utils/                                  # 共通ユーティリティ
│   ├── __init__.py
│   ├── data_generator.py                  # FXデータ生成器
│   ├── metrics_analyzer.py                # メトリクス分析器
│   └── report_generator.py                # レポート生成器
└── reports/                                # テストレポート出力先（自動生成）
```

## 実行方法

### 個別テストの実行

```bash
# 各テストを個別に実行
python test_sandbox/task10_test/01_realtime_fx_market_simulation.py
python test_sandbox/task10_test/02_multi_provider_integration.py
python test_sandbox/task10_test/03_backpressure_recovery_scenario.py
python test_sandbox/task10_test/04_24h_continuous_operation.py
python test_sandbox/task10_test/05_disaster_recovery_scenario.py
```

### 全テストの連続実行

```bash
# バッチスクリプトで全テスト実行（作成予定）
python -m test_sandbox.task10_test.run_all_tests
```

## 必要なパッケージ

```python
# 標準ライブラリ
import asyncio
import logging
import sys
import gc
from datetime import datetime, timedelta
from pathlib import Path

# 外部パッケージ（24時間テストのみ）
import psutil  # pip install psutil

# プロジェクト内部
from src.data_processing.pipelines import RealtimePipeline, DataPoint
```

## テストレポート

各テスト実行後、`reports/`ディレクトリに以下の形式でレポートが生成されます：

- **Markdownレポート** (`.md`): GitHubなどで閲覧可能
- **HTMLレポート** (`.html`): ブラウザで詳細確認
- **JSONレポート** (`.json`): プログラムでの処理用

### レポート内容

- テスト実行日時
- 成功/失敗の判定
- パフォーマンスメトリクス
  - スループット（messages/sec）
  - 遅延統計（avg/max/min）
  - 成功率
- エラーメッセージ（失敗時）
- 詳細ログ

## パフォーマンス目標

| メトリクス | 目標値 | 備考 |
|-----------|--------|------|
| スループット | ≥800 msgs/sec | 通常負荷時 |
| 平均遅延 | <1000ms | 全シナリオ |
| 成功率 | ≥95% | データ送信成功率 |
| メモリリーク | <50MB | 24時間運用 |
| 回復時間 | <5秒 | 障害からの復旧 |
| データ整合性 | ≥80% | 障害復旧後 |

## 市場条件の定義

```python
class MarketCondition(Enum):
    NORMAL = "normal"                # 通常状態
    HIGH_VOLATILITY = "high_volatility"  # 高ボラティリティ
    NEWS_RELEASE = "news_release"    # ニュースリリース時
    MARKET_OPEN = "market_open"      # 市場オープン時
    MARKET_CLOSE = "market_close"    # 市場クローズ時
    LOW_LIQUIDITY = "low_liquidity"  # 低流動性
```

## トラブルシューティング

### メモリ不足エラー
- `queue_size`パラメータを小さくする
- テストの実行時間を短縮する

### タイムアウトエラー
- `alert_threshold`を大きくする
- ネットワーク遅延を考慮した設定に変更

### psutilインストールエラー
```bash
pip install psutil
```

## カスタマイズ

### データ生成パラメータ

`utils/data_generator.py`で以下を調整可能：
- 通貨ペアと基準レート
- ボラティリティレベル
- スプレッド設定
- ボリューム範囲

### パイプライン設定

各テストファイルで以下を調整可能：
- `queue_size`: キューサイズ
- `alert_threshold`: アラート閾値（秒）
- `enable_metrics`: メトリクス収集の有効/無効

## 注意事項

1. **24時間テストについて**: 実際は時間圧縮版（約7分）で実行されます
2. **リソース使用**: 一部のテストは高負荷をシミュレートするため、CPU/メモリを多く使用します
3. **並行実行**: 各テストは独立しているため、並行実行は推奨されません

## 今後の拡張

- [ ] CI/CD統合用のGitHub Actionsワークフロー
- [ ] パフォーマンスベンチマークの自動比較
- [ ] グラフィカルなダッシュボード
- [ ] ストレステストの追加シナリオ
- [ ] 実際の市場データとの統合テスト

## ライセンス

プロジェクトのライセンスに準拠

## 作成者

Task 10.1 - RealtimePipeline E2Eテストスイート
作成日: 2025年8月26日
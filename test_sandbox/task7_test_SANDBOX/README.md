# Task7 MT5 + Polars データ処理統合テスト

このディレクトリには、Task7で実装したPolarsデータ処理基盤とMT5接続システムを組み合わせた視覚的なテストデモが含まれています。

## 🎯 概要

これらのテストは、実際のMT5接続を使用してPolarsの高性能データ処理機能を体験できる小規模なE2Eテストです。各テストはリアルタイムでCLIにパフォーマンス指標、メモリ使用量、処理結果を表示します。

## 📁 ファイル構成

```
test_sandbox/task7_test_SANDBOX/
├── 01_polars_mt5_data_showcase.py      # 基本的なデータ処理デモ
├── 02_realtime_polars_processing.py    # リアルタイム処理デモ
├── 03_large_dataset_streaming.py       # 大容量データストリーミング
├── 04_data_aggregation_pipeline.py     # データ集約パイプライン
├── display_helpers.py                  # 共通表示ユーティリティ
└── README.md                           # このファイル
```

## 🚀 テスト概要

### 1. 01_polars_mt5_data_showcase.py
**基本的なPolars + MT5データ処理デモ**

- **実行時間**: 約2分
- **機能**:
  - MT5からリアルタイムティックデータを収集（30秒）
  - Polarsによるデータ型最適化
  - メモリ使用量の最適化効果を可視化
  - 基本的な統計情報とデータサンプルの表示

```bash
# 実行方法
cd test_sandbox/task7_test_SANDBOX
python 01_polars_mt5_data_showcase.py
```

### 2. 02_realtime_polars_processing.py
**リアルタイム処理とパフォーマンス監視**

- **実行時間**: 約3分
- **機能**:
  - リアルタイムティックストリーム処理
  - 動的バッチサイズ調整
  - チャンク処理とメモリ圧迫時の自動調整
  - バックグラウンド処理とフォアグラウンド表示の分離

```bash
python 02_realtime_polars_processing.py
```

### 3. 03_large_dataset_streaming.py  
**大容量データのストリーミング処理**

- **実行時間**: 約5-10分（データ量により変動）
- **機能**:
  - 複数銘柄の大量履歴データ処理
  - Polarsの遅延評価（LazyFrame）機能
  - メモリ制限下でのチャンク分割処理
  - ガベージコレクションとメモリ最適化

```bash
python 03_large_dataset_streaming.py
```

### 4. 04_data_aggregation_pipeline.py
**多段階データ集約パイプライン**

- **実行時間**: 約3分
- **機能**:
  - 7段階の処理パイプライン実行
  - データクリーニング → 時間軸集約 → テクニカル指標計算
  - 統計分析 → 異常値検出 → 最終レポート生成
  - 各段階のパフォーマンス監視と可視化

```bash
python 04_data_aggregation_pipeline.py
```

## 🛠 前提条件

### システム要件
- Python 3.8以上
- MT5ターミナルがインストール済み
- MT5ターミナルにログイン済み
- 約200MB以上の利用可能メモリ

### 依存パッケージ
以下のパッケージがインストールされている必要があります：

```bash
# メインパッケージ
polars>=0.20.0
rich>=13.0.0
psutil>=5.9.0
asyncio

# MT5関連
MetaTrader5>=5.0.0

# 既存プロジェクト依存関係
# (pyproject.tomlで管理されているパッケージ)
```

## 🎮 実行方法

### 基本実行
プロジェクトルートから実行：

```bash
# プロジェクトルートで
cd test_sandbox/task7_test_SANDBOX

# 各テストを個別実行
python 01_polars_mt5_data_showcase.py
python 02_realtime_polars_processing.py  
python 03_large_dataset_streaming.py
python 04_data_aggregation_pipeline.py
```

### uvを使用した実行（推奨）
```bash
# プロジェクトルートで
uv run python test_sandbox/task7_test_SANDBOX/01_polars_mt5_data_showcase.py
uv run python test_sandbox/task7_test_SANDBOX/02_realtime_polars_processing.py
# ... 以下同様
```

### 中断方法
実行中に `Ctrl+C` で安全に中断できます。各テストは適切なクリーンアップ処理を行います。

## 📊 各テストで確認できること

### パフォーマンス指標
- **処理速度**: ティック/秒、行/秒
- **メモリ効率**: 最適化前後の比較
- **レイテンシ**: 平均・最大・最小処理時間
- **スループット**: バッチ処理能力

### 可視化要素
- **リアルタイムダッシュボード**: Rich libraryによる美しいCLI表示
- **プログレスバー**: 処理進行状況の可視化
- **メモリ使用量グラフ**: メモリ傾向の表示
- **統計テーブル**: 詳細な処理統計

### エラーハンドリング
- **接続エラー**: MT5接続失敗時の対応
- **メモリ制限**: メモリ不足時の動的調整
- **データ品質**: 異常値・欠損値の検出と処理

## 🔧 カスタマイズ

### 設定変更
各テストファイルの先頭部分で以下の設定を変更できます：

```python
# 例：01_polars_mt5_data_showcase.py
class PolarsDataShowcase:
    def __init__(self):
        # これらの値を変更可能
        self.collection_duration = 30  # データ収集時間（秒）
        self.symbol = "EURUSD"         # 取引シンボル
        self.buffer_size = 1000        # バッファサイズ
```

### 追加シンボル
```python
# 複数銘柄での実行例
self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
```

### メモリ制限調整
```python
# Polarsエンジンの設定
self.polars_engine = PolarsProcessingEngine(
    max_memory_mb=100,   # メモリ制限を調整
    chunk_size=50        # チャンクサイズを調整
)
```

## 🚨 トラブルシューティング

### よくある問題と解決方法

#### 1. MT5接続エラー
```
❌ MT5への接続に失敗しました
```
**解決方法**:
- MT5ターミナルが起動していることを確認
- MT5にアカウントでログインしていることを確認
- ターミナルでアルゴリズム取引が許可されていることを確認

#### 2. メモリ不足エラー
```
❌ MemoryLimitError: メモリ制限を超過しました
```
**解決方法**:
- 他のアプリケーションを終了してメモリを確保
- テストファイル内の `max_memory_mb` を小さい値に設定
- `chunk_size` を小さい値に設定

#### 3. データが取得できない
```
⚠️ 収集されたデータが不十分です
```
**解決方法**:
- 市場が開いている時間に実行
- 異なるシンボル（通貨ペア）を試す
- データ収集時間を長くする

#### 4. パッケージエラー
```
ModuleNotFoundError: No module named 'polars'
```
**解決方法**:
```bash
# 必要パッケージのインストール
pip install polars rich psutil MetaTrader5

# または uvを使用
uv add polars rich psutil MetaTrader5
```

## 📈 期待される結果

### 正常実行時の出力例

#### 01_polars_mt5_data_showcase.py
```
🚀 Polars + MT5 データ処理デモンストレーション

✅ 接続とエンジン初期化完了
📊 30秒間ティックデータを収集します...
✅ 収集完了: 1,247ティック
🔄 Polarsデータフレーム作成中...
⚡ データ型最適化実行中...

🎉 データ処理結果サマリー
• 処理されたティック数: 1,247
• メモリ削減率: 23.45%
• データフレームサイズ: 1247 x 6

📋 データサンプル（最初の5行）:
...（データテーブル表示）...
```

#### パフォーマンス目標値
- **処理速度**: 1,000ティック/秒以上
- **メモリ削減率**: 15-30%
- **レスポンス時間**: 100ms以下（小バッチ）
- **メモリ使用量**: 150MB以下（通常時）

## 🔗 関連ファイル

### プロジェクト内の関連コード
- `src/data_processing/processor.py` - Polarsエンジン本体
- `src/mt5_data_acquisition/` - MT5データ取得システム
- `src/common/models.py` - データモデル定義

### 参考ドキュメント
- [Polars公式ドキュメント](https://pola-rs.github.io/polars/)
- [Rich公式ドキュメント](https://rich.readthedocs.io/)
- [MetaTrader5 Python API](https://www.mql5.com/en/docs/integration/python_metatrader5)

## 📝 注意事項

1. **市場時間**: FX市場が開いている時間に実行することを推奨
2. **データ量**: 大容量テストは通信量が多くなる可能性があります
3. **リソース**: CPUとメモリを集約的に使用します
4. **ネットワーク**: 安定したインターネット接続が必要です
5. **MT5設定**: アルゴリズム取引とDLLの使用が許可されている必要があります

## 🎉 完了後の確認

各テストが正常に完了すると以下が確認できます：

✅ MT5システムとの接続・データ取得  
✅ Polarsの高速データ処理性能  
✅ メモリ効率的な大容量データ処理  
✅ リアルタイム処理とストリーミング  
✅ 複雑な集約処理パイプライン  
✅ エラーハンドリングと回復処理

これらのテストを通じて、Task7で実装したPolarsデータ処理基盤の実用性と性能を体験できます。
# MT5 OHLC Fetcher - Visual Sandbox Tests

Phase2_task5で開発した`HistoricalDataFetcher`クラスを使用した視覚的な確認テスト集です。
人間が視覚的に確認できるE2Eテストとして、様々な角度からOHLCデータ取得機能を検証します。

## 📁 ファイル構成

```
task5_test_SANDBOX/
├── utils/
│   └── ohlc_display_helpers.py    # 表示用ヘルパー関数
├── 01_basic_ohlc_fetch.py         # 基本的なOHLCデータ取得
├── 02_multi_timeframe_display.py  # 複数時間枠の比較表示
├── 03_batch_fetch_monitor.py      # バッチ処理のパフォーマンス監視
├── 04_missing_data_detector.py    # 欠損データの検出と分析
├── 05_comparison_with_ticks.py    # ティックデータとの比較検証
├── 06_marimo_ohlc_dashboard.py    # インタラクティブダッシュボード
└── README.md                       # このファイル
```

## 🚀 実行方法

### 前提条件

1. MT5がインストールされていること
2. デモアカウントでログイン可能なこと
3. 必要なPythonパッケージがインストールされていること

```bash
# 必要なパッケージのインストール
uv pip install rich polars altair marimo psutil
```

### 各テストの実行

#### 1. 基本的なOHLCデータ取得テスト

```bash
python test_sandbox/task5_test_SANDBOX/01_basic_ohlc_fetch.py
```

**機能:**
- EURJPYの5分足データを過去3日分取得
- ローソク足チャートのASCII表示
- データ統計の表示
- データ品質チェック（時間の連続性、価格の妥当性）

**確認ポイント:**
- データが正しく取得できているか
- 価格関係が正しいか（Low ≤ Open/Close ≤ High）
- 時間間隔に異常がないか

#### 2. 複数時間枠表示テスト

```bash
python test_sandbox/task5_test_SANDBOX/02_multi_timeframe_display.py
```

**機能:**
- M1, M5, M15, H1, H4, D1の6つの時間枠を並列取得
- タイムフレーム間の相関分析
- データ整合性チェック（M1×5 = M5の検証）

**確認ポイント:**
- 各タイムフレームのトレンド方向
- ボラティリティの違い
- データの整合性

#### 3. バッチ処理監視テスト

```bash
python test_sandbox/task5_test_SANDBOX/03_batch_fetch_monitor.py
```

**機能:**
- 複数シンボルの大量データ（30日分の1分足）を並列取得
- リアルタイムでCPU/メモリ使用量を監視
- 取得速度とパフォーマンス統計の表示

**確認ポイント:**
- データ取得速度（records/sec）
- メモリ使用量の推移
- エラー発生の有無

#### 4. 欠損データ検出テスト

```bash
python test_sandbox/task5_test_SANDBOX/04_missing_data_detector.py
```

**機能:**
- 2週間分のデータから欠損期間を検出
- マーケット休場時間とデータギャップを分類
- データ品質スコアの算出
- タイムライン視覚化

**確認ポイント:**
- 週末ギャップの識別
- 異常なデータギャップの検出
- データカバレッジ率

#### 5. ティックデータ比較テスト

```bash
python test_sandbox/task5_test_SANDBOX/05_comparison_with_ticks.py
```

**機能:**
- リアルタイムティックを3分間収集
- ティックからOHLCバーを生成
- MT5の公式OHLCデータと比較
- 精度分析

**確認ポイント:**
- ティック集約の正確性
- MT5データとの一致率
- ボリュームカウントの差異

#### 6. Marimoダッシュボード

```bash
marimo run test_sandbox/task5_test_SANDBOX/06_marimo_ohlc_dashboard.py
```

**機能:**
- インタラクティブなローソク足チャート
- 移動平均線とボリンジャーバンド
- パターン検出（Doji、Engulfing）
- 自動更新機能

**確認ポイント:**
- チャートの視覚的確認
- テクニカル指標の動作
- リアルタイム更新

## 📊 表示される情報

### 共通の表示要素

- **Rich Console出力**: カラフルなターミナル表示
- **進捗バー**: データ取得の進行状況
- **統計テーブル**: 各種メトリクスの表形式表示
- **エラーハンドリング**: 接続エラーや取得失敗の詳細表示

### データ品質指標

- **時間連続性**: 期待される間隔からの逸脱を検出
- **価格妥当性**: OHLC関係の整合性チェック
- **カバレッジ率**: 期待されるデータ量に対する実際の取得率
- **精度スコア**: ティックデータとの一致度

## 🔧 トラブルシューティング

### 接続エラーが発生する場合

1. MT5が起動していることを確認
2. デモアカウントの認証情報が正しいか確認
3. ネットワーク接続を確認

### データが取得できない場合

1. シンボル名が正しいか確認（EURJPY, GBPJPY等）
2. 指定期間にマーケットが開いていたか確認
3. MT5のヒストリーデータ設定を確認

### パフォーマンスが遅い場合

1. `batch_size`を調整（デフォルト: 5000）
2. `max_workers`を環境に合わせて調整（デフォルト: 4）
3. 取得期間を短くする

## 📈 期待される出力例

### 01_basic_ohlc_fetch.py

```
============================================================
                 Basic OHLC Data Fetch Test                 
============================================================

ℹ Symbol: EURJPY
ℹ Timeframe: 5 Minutes
✓ Connected to MT5 successfully!
✓ Data fetched successfully in 2.34 seconds!
ℹ Total records: 864

[Chart Display]
[Table with recent OHLC data]
[Statistics panel]

✓ No time anomalies detected
✓ All price relationships are valid
```

### 03_batch_fetch_monitor.py

```
┌─────────────────────────────────────┐
│     Batch Processing Monitor        │
├─────────────────────────────────────┤
│ Total Batches: 45                   │
│ Completed: 42                       │
│ Progress: 93.3%                     │
│                                     │
│ CPU Usage: ████████░░░░░░░░ 42.3%  │
│ Memory: 156.8 MB                    │
│ Fetch Speed: 1234.5 rec/s          │
└─────────────────────────────────────┘
```

## 🎯 学習ポイント

1. **非同期処理**: ティックストリーミングでの`asyncio`活用
2. **並列処理**: `ThreadPoolExecutor`による複数タイムフレーム同時取得
3. **メモリ効率**: Polarsの`LazyFrame`による遅延評価
4. **視覚化**: Rich/Marimoによる対話的な表示
5. **データ検証**: 時系列データの品質チェック手法

## 📝 注意事項

- デモアカウントを使用するため、実際の取引には使用できません
- 大量データ取得時はネットワーク帯域とメモリ使用量に注意
- マーケット休場時間はデータが取得できません
- ティック比較テストは市場が活発な時間帯に実行することを推奨
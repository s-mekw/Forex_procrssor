## 汎用実装計画（リンク集・導線テンプレート）

このファイルは、../.kiro/specs/Forex_procrssor/tasks.md に定義された任意のタスクを実装するための最小限の導線です。具体的な設計・仕様は各ドキュメントへ直接リンクし、本ファイルには詳細を書きません。

### タスク選択
- 対象タスクは `../.kiro/specs/Forex_procrssor/tasks.md` を参照し、対応するチェックリスト/説明/要件番号を確認してください。
- 現在の対象タスク: 
  - [ ] 10.1. リアルタイム処理パイプライン基盤の構築
    - tests/integration/test_data_pipeline.pyに非同期処理とバックプレッシャーのテストを作成
    - src/data_processing/pipelines.pyにRealtimePipelineクラスの骨格を実装
    - asyncioベースの非同期データフロー処理を実装（1分足データをパススルー）
    - 1秒を超える遅延時のアラート機能を追加
    - _要件: 2.4_ of `../.kiro/specs/Forex_procrssor/requirements.md`
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

## タスク10.1 実装計画

### 📍 現在の状態
- ステップ: 1/7 完了
- 最終更新: 2025-08-26
- 現在作業中: Step 2（RealtimePipelineクラスの骨格実装）

### 📋 実装ステップ

#### Step 1: テストファイルの作成と基本構造
- ファイル: `tests/integration/test_data_pipeline.py`
- 作業: テストファイルを作成し、基本的なimportとテストクラスの骨格を実装
- 内容:
  - pytest-asyncioの設定
  - RealtimePipelineのテストクラス作成
  - 基本的なセットアップ/ティアダウンメソッド
- 完了: [x] ✅ 2025-08-26 完了

#### Step 2: RealtimePipelineクラスの骨格実装
- ファイル: `src/data_processing/pipelines.py` (新規作成)
- 作業: RealtimePipelineクラスの基本構造を実装
- 内容:
  - asyncioベースの基本クラス定義
  - 初期化メソッド（__init__）
    - queue_size: int = 1000（キューの最大サイズ）
    - alert_threshold: float = 1.0（遅延アラート閾値、秒）
    - enable_metrics: bool = True（メトリクス収集フラグ）
  - 基本的な型定義とプロトコル定義
    - DataPoint: TypedDict（timestamp, data, metadata）
    - ProcessingResult: TypedDict（processed_data, latency, status）
  - インスタンス変数の定義
    - self._input_queue: asyncio.Queue（入力キュー）
    - self._output_queue: asyncio.Queue（出力キュー）
    - self._metrics: Dict（メトリクス収集用）
    - self._is_running: bool（実行状態フラグ）
  - 基本メソッドのスタブ実装
    - async def start() -> None
    - async def stop() -> None
    - async def submit(data: DataPoint) -> None
    - async def get_result() -> ProcessingResult
- 完了: [ ]

#### Step 3: 非同期データフロー処理の実装
- ファイル: `src/data_processing/pipelines.py`
- 作業: 1分足データのパススルー処理を実装
- 内容:
  - async def _process_loop() メソッド実装
    - 入力キューからデータを取得
    - タイムスタンプ情報を付与
    - 簡単な変換処理（パススルー）
    - 出力キューへ送信
  - async def process_data(data: DataPoint) メソッド実装
    - 単一データの処理ロジック
    - 遅延計測の追加
  - 基本的なエラーハンドリング
    - try-except による例外処理
    - ログ出力
  - asyncio.create_task()でのループ起動
- 完了: [ ]

#### Step 4: バックプレッシャー制御の実装
- ファイル: `src/data_processing/pipelines.py`
- 作業: キューサイズ管理とバックプレッシャー機能を追加
- 内容:
  - キューの最大サイズ設定（maxsize パラメータ）
    - asyncio.Queue(maxsize=self.queue_size)
  - キューフル時の待機ロジック
    - await queue.put() での自動待機
    - タイムアウト処理の追加
  - メトリクス収集（キューサイズ、処理待ち件数）
    - queue.qsize() での現在サイズ取得
    - 最大/平均キューサイズの記録
    - バックプレッシャー発生回数のカウント
  - async def is_backpressure_active() メソッド追加
    - キューの状態チェック
- 完了: [ ]

#### Step 5: 遅延監視とアラート機能の実装
- ファイル: `src/data_processing/pipelines.py`
- 作業: 1秒を超える遅延時のアラート機能を追加
- 内容:
  - タイムスタンプベースの遅延計測
  - 閾値（1秒）を超えた場合のアラート発出
  - ログとメトリクスへの記録
- 完了: [ ]

#### Step 6: 統合テストの実装
- ファイル: `tests/integration/test_data_pipeline.py`
- 作業: 非同期処理とバックプレッシャーのテストケース実装
- 内容:
  - test_realtime_pipeline_basic_flow: 基本的なデータフロー
  - test_backpressure_control: バックプレッシャー制御
  - test_latency_alert: 遅延アラート機能
  - test_concurrent_processing: 並行処理の正常動作
- 完了: [ ]

#### Step 7: パフォーマンステストと最適化
- ファイル: `tests/integration/test_data_pipeline.py`, `src/data_processing/pipelines.py`
- 作業: パフォーマンステストの追加と必要に応じた最適化
- 内容:
  - test_throughput: スループット測定テスト
  - test_memory_usage: メモリ使用量の監視
  - 必要に応じてバッファサイズやワーカー数の最適化
- 完了: [ ]

### 技術的決定事項
1. **非同期フレームワーク**: Python標準のasyncioを使用
2. **キュー実装**: asyncio.Queueを使用（maxsizeでバックプレッシャー制御）
3. **遅延計測**: データオブジェクトにタイムスタンプを付与し、処理時に経過時間を計算
4. **アラート方式**: 初期実装ではloggingモジュールを使用（将来的にはPrometheusメトリクス連携）
5. **テスト方針**: pytest-asyncioを使用した非同期テスト

### 依存関係
- 既存: `src/data_processing/pipeline.py` (IndicatorPipeline)
- 新規作成: `src/data_processing/pipelines.py` (RealtimePipeline)
- テスト: `pytest-asyncio` パッケージが必要
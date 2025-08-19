## 汎用実装計画（リンク集・導線テンプレート）

このファイルは、../.kiro/specs/Forex_procrssor/tasks.md に定義された任意のタスクを実装するための最小限の導線です。具体的な設計・仕様は各ドキュメントへ直接リンクし、本ファイルには詳細を書きません。

### タスク選択
- 対象タスクは `../.kiro/specs/Forex_procrssor/tasks.md` を参照し、対応するチェックリスト/説明/要件番号を確認してください。
- 現在の対象タスク: 
  - [ ] 4. リアルタイムティックデータ取得の実装
    - tests/unit/test_tick_fetcher.pyにティックデータ取得とバリデーションのテストを作成
    - src/mt5_data_acquisition/tick_fetcher.pyにTickDataStreamerクラスを実装
    - 非同期ストリーミングとリングバッファ（10,000件）を実装
    - スパイクフィルター（3σルール）による異常値除外を追加
    - _要件: 1.2_
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

---

## 📋 タスク4 実装計画

### Step 1: テストケースの作成
- ファイル: tests/unit/test_tick_fetcher.py
- 作業: TickDataStreamerクラスのテストケースを作成
  - 正常なストリーミング開始のテスト
  - リングバッファのサイズ制限テスト
  - スパイクフィルター（3σ）のテスト
  - バックプレッシャー制御のテスト
  - エラー時の再購読テスト
- 完了: [x]

### Step 2: TickDataStreamerクラスの基本実装 ✅
- ファイル: src/mt5_data_acquisition/tick_fetcher.py
- 作業: クラスの基本構造と初期化メソッドを実装
  - StreamerConfigデータクラスの定義 ✅
  - __init__メソッドの実装（設定パラメータ定義） ✅
  - リングバッファ（collections.deque）の初期化 ✅
  - 統計量（平均、標準偏差）の初期化 ✅
  - プロパティメソッド（buffer_usage, current_stats, is_connected） ✅
  - ロガー設定 ✅
- 完了: [x] (2025-08-19 09:01)

### Step 3: MT5ティックデータ取得メソッドの実装 ✅
- ファイル: src/mt5_data_acquisition/tick_fetcher.py
- 作業: MT5 APIを使用したティック取得機能を実装
  - 軽微な改善点の修正（backpressure_threshold重複、デッドコード） [x]
  - subscribe_to_ticksメソッド（MT5購読開始） [x]
  - unsubscribeメソッド（購読解除） [x]
  - _process_tickメソッド（Tickモデル変換） [x]
  - _fetch_latest_tickメソッド（最新ティック取得） [x]
  - stream_ticksメソッドのスタブ実装 [x]
- 完了: [x] (2025-08-19 09:20)

### Step 4: 非同期ストリーミング機能の実装
- ファイル: src/mt5_data_acquisition/tick_fetcher.py
- 作業: asyncioベースの非同期ストリーミングを実装
  - stream_ticksメソッド（非同期ジェネレータ）
  - _handle_backpressureメソッド（バックプレッシャー制御）
  - _emit_eventメソッド（10ms以内のイベント発火）
- 完了: [ ]

### Step 5: スパイクフィルターの実装
- ファイル: src/mt5_data_acquisition/tick_fetcher.py
- 作業: 3σルールによる異常値検出と除外機能
  - _update_statisticsメソッド（平均・標準偏差の更新）
  - _is_spikeメソッド（3σルール判定）
  - _filter_spikeメソッド（異常値除外処理）
  - 統計量の動的更新（ローリングウィンドウ）
- 完了: [ ]

### Step 6: エラーハンドリングと再購読機能
- ファイル: src/mt5_data_acquisition/tick_fetcher.py
- 作業: 堅牢なエラー処理と自動復旧機能
  - _handle_stream_errorメソッド（エラー処理）
  - _auto_resubscribeメソッド（自動再購読）
  - サーキットブレーカーパターンの実装
  - エラーログの構造化出力
- 完了: [ ]

### Step 7: パフォーマンス最適化
- ファイル: src/mt5_data_acquisition/tick_fetcher.py
- 作業: 10ms以内のレイテンシ達成
  - メモリプールの実装（事前割り当て）
  - Float32への効率的な変換
  - 非同期処理の最適化
  - プロファイリングとベンチマーク
- 完了: [ ]

### Step 8: 統合テストの実装
- ファイル: tests/integration/test_tick_streaming.py
- 作業: エンドツーエンドの統合テスト
  - MT5ConnectionManagerとの統合テスト
  - 実際のストリーミングシミュレーション
  - レイテンシ測定テスト
  - 長時間実行の安定性テスト
- 完了: [ ]

### Step 9: ドキュメント作成
- ファイル: src/mt5_data_acquisition/tick_fetcher.py
- 作業: docstringとコメントの追加
  - クラス・メソッドのdocstring作成
  - 使用例の追加
  - パラメータ説明の詳細化
  - 設計上の注意点の記載
- 完了: [ ]

### Step 10: 最終確認とレビュー
- ファイル: 全関連ファイル
- 作業: 品質保証とパフォーマンス確認
  - 全テストケースの実行と確認
  - メモリリークチェック
  - パフォーマンス測定（10ms以内）
  - コードレビューと改善
- 完了: [ ]

## 🔧 技術的な決定事項
- **リングバッファ**: `collections.deque(maxlen=10000)`を使用 ✅
- **統計計算**: NumPyを使用せず、Pythonネイティブで実装（パフォーマンス優先） ✅
- **非同期処理**: `asyncio`と`async/await`パターンを使用
- **スパイクフィルター**: ローリングウィンドウ（1000件）で統計量を更新
- **エラー処理**: サーキットブレーカーパターン（失敗5回で一時停止）

## 📈 進捗サマリ
- **完了**: Step 1-4 (テスト作成, 基本実装, MT5連携, 非同期ストリーミング)
- **現在**: Step 5 (スパイクフィルター実装) - 開始
- **残り**: Step 5-10 (6ステップ)
- **完了率**: 40% (4/10ステップ)
- **テスト状態**: 9個PASSED (45%)、残り11個XFAIL/SKIP

## 📚 参考リンク
- MT5 Python API: https://www.mql5.com/en/docs/python_metatrader5
- asyncio ドキュメント: https://docs.python.org/3/library/asyncio.html
- collections.deque: https://docs.python.org/3/library/collections.html#collections.deque
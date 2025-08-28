# Task 10.3 実装計画

## 現在の対象タスク
- [ ] 10.3. パイプラインのリファクタリングと責務の明確化
  - src/data_processing/analyzer.py に MultiTimeframeAnalyzer クラス実装 ✅ (実装済み)
  - RealtimePipeline からマルチタイムフレーム分析ロジックを MultiTimeframeAnalyzer に移譲
  - RealtimePipeline はデータフロー管理に専念し、MultiTimeframeAnalyzer をコンポーネントとして利用する構成に変更
  - Analyzer専門のユニットテスト(tests/unit/test_analyzer.py)を作成し、ロジックの堅牢性を保証
  - _要件: 2.4, 2.5 (リファクタリング)_ of `../.kiro/specs/Forex_procrssor/requirements.md`

## 実装ステップ

### Step 1: 現状分析と設計確認
- ファイル: src/data_processing/pipelines.py, src/data_processing/analyzer.py
- 作業: 現在のコード構造を確認し、責務分離の境界を明確化
- 完了: [x]

### Step 2: RealtimePipelineのリファクタリング準備
- ファイル: src/data_processing/pipelines.py
- 作業: マルチタイムフレーム分析ロジックの抽出対象を特定
- 完了: [x]

### Step 3: MultiTimeframeAnalyzerへの責務移譲
- ファイル: src/data_processing/analyzer.py
- 作業: バッファ管理機能をMultiTimeframeAnalyzerに実装
  - バッファ管理プロパティの追加（_data_buffer, _max_history_bars）
  - add_new_bar()メソッドの実装
  - _manage_buffer_size()メソッドの実装
  - get_buffer_size(), is_ready(), get_buffer_as_dataframe()メソッドの実装
  - analyze_streaming()メソッドの改良（内部バッファ対応）
- 完了: [x]

### Step 4: RealtimePipelineの簡素化
- ファイル: src/data_processing/pipelines.py
- 作業: データフロー管理に特化した実装に変更
  - __init__メソッド: max_history_barsをMultiTimeframeAnalyzerに渡すように修正（L83）
  - _data_bufferの削除（L79）
  - バッファ管理コードの削除（L203, L205-207）
  - 最小バー数チェックとDataFrame変換の削除（L210-213）
  - analyze_streaming()の呼び出しをパラメータなしに変更（L216-220）
  - MultiTimeframeAnalyzerのadd_new_bar()とis_ready()を使用
  - ログメッセージをget_buffer_size()を使用するように更新（L238-241）
- 完了: [x]

### Step 5: インターフェース設計の改善
- ファイル: src/data_processing/pipelines.py, src/data_processing/analyzer.py
- 作業: 依存性注入パターンの適用、モジュール間の結合度低減
- 完了: [x]

### Step 6: ユニットテストの作成（基本テスト）
- ファイル: tests/unit/test_multiframe_analyzer.py
- 作業: MultiTimeframeAnalyzerのバッファ管理機能に対する基本的な単体テスト実装
  - TestBufferManagement: バッファ管理の基本機能テスト（4テストケース）
  - TestAnalysisReadiness: 分析準備状態のテスト（3テストケース）
  - TestDataFrameConversion: DataFrame変換のテスト（3テストケース）
  - TestInternalBufferMode: 内部バッファモードのテスト（3テストケース）
- 完了: [x]

### Step 7: ユニットテストの作成（エッジケース）
- ファイル: tests/unit/test_multiframe_analyzer.py
- 作業: エッジケース、エラーハンドリングテストの追加
  - TestEdgeCasesExtendedクラス：
    - test_add_bar_with_none: None値のバー追加テスト ✅
    - test_add_bar_with_invalid_data: 無効データ型のテスト ✅
    - test_negative_values_in_bar: 負の価格データテスト ✅
    - test_nan_values_handling: NaN値処理テスト ✅
    - test_infinity_values_handling: 無限大値処理テスト ✅
  - TestErrorHandlingExtendedクラス：
    - test_buffer_overflow_protection: バッファオーバーフロー保護 ✅
    - test_corrupt_data_handling: 破損データ処理 ✅
    - test_type_mismatch_handling: データ型不一致処理 ✅
    - test_memory_efficiency_large_buffer: 巨大バッファのメモリ効率 ✅
    - test_timestamp_validation: タイムスタンプ検証 ✅
- 完了: [x]

### Step 8: 統合テストの更新
- ファイル: tests/integration/test_data_pipeline.py
- 作業: リファクタリング後の動作確認テスト
  - TestMultiframeIntegration: RealtimePipelineとMultiTimeframeAnalyzerの連携テスト（4テストケース）
    - test_pipeline_multiframe_data_flow: データフロー全体の検証
    - test_buffer_synchronization: バッファ管理の同期確認
    - test_analyzer_state_consistency: 分析器の状態一貫性
    - test_pipeline_restart_recovery: パイプライン再起動時の復旧
  - TestEndToEndIntegration: エンドツーエンドの検証（4テストケース）
    - test_realtime_data_processing: リアルタイムデータ処理の検証
    - test_large_volume_processing: 大量データ処理のパフォーマンス
    - test_multiframe_analysis_accuracy: マルチタイムフレーム分析精度
    - test_error_recovery_flow: エラー復旧フローの確認
  - TestPerformanceIntegration: パフォーマンス検証（4テストケース）
    - test_throughput_measurement: スループット測定
    - test_latency_monitoring: レイテンシー監視
    - test_memory_efficiency: メモリ効率検証
    - test_cpu_utilization: CPU使用率測定
- 完了: [ ]

### Step 9: ドキュメント更新
- ファイル: src/data_processing/README.md（必要に応じて）
- 作業: アーキテクチャ変更の記録
- 完了: [ ]

### Step 10: 最終検証とクリーンアップ
- ファイル: 全体
- 作業: コードレビュー、不要なコードの削除、最終動作確認
- 完了: [ ]

## 技術的な注意点

### 現在の実装状況
- `MultiTimeframeAnalyzer`クラスは既に`analyzer.py`に実装済み
- `RealtimePipeline`は既に`MultiTimeframeAnalyzer`を使用している（`_multiframe_analyzer`として）
- しかし、分析ロジックの一部がまだ`RealtimePipeline`に残っている

### リファクタリングの焦点
1. **責務の明確化**
   - RealtimePipeline: データフロー管理、キューイング、バックプレッシャー制御
   - MultiTimeframeAnalyzer: マルチタイムフレームRCI計算、データ変換

2. **重複コードの削除**
   - データバッファリングロジックの統一
   - RCI計算の一元化

3. **テスタビリティの向上**
   - 各コンポーネントの独立性を高める
   - モックしやすい設計にする

## 進捗メトリクス
- 総ステップ数: 10
- 完了ステップ: 7
- 実行中ステップ: 1 (Step 8)
- 進捗率: 75%

## Step 8 実装チェックリスト
- [ ] test_data_pipeline.pyの既存テスト確認
- [ ] TestMultiframeIntegrationクラスの実装
- [ ] TestEndToEndIntegrationクラスの実装
- [ ] TestPerformanceIntegrationクラスの実装
- [ ] 新テストの実行と検証
- [ ] パフォーマンス測定結果の記録
- [ ] 既存13テストとの互換性確認

## Step 3 実装チェックリスト
- [x] __init__メソッドにバッファ管理プロパティを追加
- [x] add_new_bar()メソッドの実装
- [x] _manage_buffer_size()メソッドの実装
- [x] get_buffer_size()メソッドの実装
- [x] is_ready()メソッドの実装
- [x] get_buffer_as_dataframe()メソッドの実装
- [x] analyze_streaming()メソッドのリファクタリング
- [x] _analyze_with_external_history()メソッドの実装
- [x] _calculate_rci_metrics()メソッドの抽出
- [ ] 実装後の動作確認
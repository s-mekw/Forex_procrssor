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
- 作業: RealtimePipelineから移譲するメソッドの実装・調整
- 完了: [ ]

### Step 4: RealtimePipelineの簡素化
- ファイル: src/data_processing/pipelines.py
- 作業: データフロー管理に特化した実装に変更
- 完了: [ ]

### Step 5: インターフェース設計の改善
- ファイル: src/data_processing/pipelines.py, src/data_processing/analyzer.py
- 作業: 依存性注入パターンの適用、モジュール間の結合度低減
- 完了: [ ]

### Step 6: ユニットテストの作成（基本テスト）
- ファイル: tests/unit/test_analyzer.py
- 作業: MultiTimeframeAnalyzerの基本的な単体テスト実装
- 完了: [ ]

### Step 7: ユニットテストの作成（エッジケース）
- ファイル: tests/unit/test_analyzer.py
- 作業: エラーハンドリング、境界値テストの実装
- 完了: [ ]

### Step 8: 統合テストの更新
- ファイル: tests/integration/test_data_pipeline.py
- 作業: リファクタリング後の動作確認テスト
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
- 完了ステップ: 2
- 進捗率: 20%
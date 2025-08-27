# ワークフローコンテキスト

## 📍 現在の状態
- ステップ: 1/10
- 最終更新: 2025-08-27 13:30
- タスク: Task 10.3 パイプラインのリファクタリングと責務の明確化

## 📋 計画ステータス
計画策定完了

## 🎯 タスクの目的
RealtimePipelineとMultiTimeframeAnalyzerの責務を明確に分離し、以下を達成する：
1. RealtimePipelineをデータフロー管理に特化
2. MultiTimeframeAnalyzerに分析ロジックを集約
3. テスタビリティとメンテナンス性の向上

## 📊 現状分析（Step 1: 完了）
### 既存実装の確認
- `MultiTimeframeAnalyzer`クラス: 実装済み（analyzer.py）
  - 短期・長期RCI計算ロジック実装済み
  - analyze()とanalyze_streaming()メソッドあり
  - _calculate_single_rci()でRCI計算を実装
- `RealtimePipeline`: MultiTimeframeAnalyzerを部分的に使用中
  - _multiframe_analyzerとして保持
  - _data_bufferで独自にバッファリング
  - 最小バー数チェックを独自に実装

### 発見した責務の重複
1. **データバッファリング**: 両方で実装
   - RealtimePipeline: _data_bufferで管理（L79, L194-207）
   - MultiTimeframeAnalyzer: analyze_streaming内でも履歴管理（L472-478）

2. **最小バー数チェック**: 両方で実装
   - RealtimePipeline: L210でチェック
   - MultiTimeframeAnalyzer: L310でチェック

3. **RCI計算**: 部分的に分散
   - RealtimePipelineは MultiTimeframeAnalyzer.analyze_streaming()を呼び出し
   - ただし、バッファ管理とメトリクス更新はパイプライン側で実施

### 責務分離の境界（明確化済み）
- **RealtimePipeline**: データフロー管理に専念
  - キュー管理、バックプレッシャー制御
  - メトリクス収集（パフォーマンス関連）
  - アラート管理
- **MultiTimeframeAnalyzer**: 分析ロジックに専念
  - データバッファリング
  - タイムフレーム変換
  - RCI計算

## 🔄 次のアクション
Step 2: RealtimePipelineのリファクタリング準備

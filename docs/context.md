# ワークフローコンテキスト

## 📍 現在の状態
- ステップ: 2/10 ✅
- 最終更新: 2025-08-27 14:15
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
### Step 2: RealtimePipelineのリファクタリング準備（完了 ✅）

#### 作業内容
**目的**: RealtimePipelineから移譲すべきコードブロックを特定し、リファクタリング計画を明確化

**作業対象ファイル**: `src/data_processing/pipelines.py`

**特定した移譲対象コード**:

1. **データバッファ管理ロジック（L194-207）**
   ```python
   # 現在: RealtimePipeline._process_message()内
   self._data_buffer.append(new_bar)
   if len(self._data_buffer) > self._max_history_bars:
       self._data_buffer = self._data_buffer[-self._max_history_bars:]
   ```
   → MultiTimeframeAnalyzerに移譲

2. **最小バー数チェックロジック（L210-211）**
   ```python
   # 現在: RealtimePipeline._process_message()内
   min_required_bars = 200
   if len(self._data_buffer) >= min_required_bars:
   ```
   → MultiTimeframeAnalyzerに移譲

3. **DataFrame変換処理（L213）**
   ```python
   # 現在: RealtimePipeline._process_message()内
   history_df = pl.DataFrame(self._data_buffer)
   ```
   → MultiTimeframeAnalyzerに移譲

**実装方針**:
1. MultiTimeframeAnalyzerにバッファ管理機能を追加
   - `add_new_bar()` メソッド: 新しいバーをバッファに追加
   - `_manage_buffer_size()` プライベートメソッド: バッファサイズ管理
   - バッファ状態のgetter: `get_buffer_size()`, `is_ready()`

2. RealtimePipelineの変更内容
   - `_data_buffer`の削除（L79）
   - `_max_history_bars`の削除（MultiTimeframeAnalyzerに移譲）
   - バッファ管理をMultiTimeframeAnalyzerに委譲

3. インターフェースの変更
   - 現在: `analyzer.analyze_streaming(new_bar, history, min_history_bars)`
   - 変更後: `analyzer.add_new_bar(new_bar)` → `analyzer.analyze_streaming()`

**リファクタリングの影響範囲**:
- RealtimePipeline._process_message() メソッド（L190-244）
- RealtimePipeline.__init__() メソッド（L79: _data_buffer初期化部分）
- MultiTimeframeAnalyzer.analyze_streaming() メソッド（パラメータ変更）

**詳細な変更箇所の特定**:
1. **pipelines.py L79**: `self._data_buffer: list[dict[str, Any]] = []` → 削除予定
2. **pipelines.py L71**: `self._max_history_bars = max_history_bars` → MultiTimeframeAnalyzerのconfigへ移動
3. **pipelines.py L194-207**: バッファ管理コード全体 → MultiTimeframeAnalyzerへ移譲
4. **pipelines.py L210-213**: 最小バー数チェックとDataFrame変換 → MultiTimeframeAnalyzerへ移譲
5. **pipelines.py L216-220**: analyze_streaming()の呼び出し方法変更

**次のステップへの準備**:
- Step 3でMultiTimeframeAnalyzerにバッファ管理機能を実装
- Step 4でRealtimePipelineから該当コードを削除

---

## 👁️ レビュー結果

### Step 2 レビュー
#### 良い点
- ✅ 責務分離の境界が明確に特定されている
- ✅ 移譲対象コードが具体的に特定されている（行番号付き）
- ✅ インターフェース変更の設計が適切
- ✅ 影響範囲が詳細に分析されている
- ✅ ドキュメントが適切に更新されている

#### 改善点
- ⚠️ 既存のユニットテストに11件の失敗がある（テストカバレッジ8.41%）
- 優先度: 高（Step 3実装前に修正必要）
- ⚠️ ruffによる未使用変数の警告（`start_time`）が残っている
- 優先度: 低（軽微な問題）

#### 評価総合点数
- 分析の正確性と網羅性から、評価総合点数をつけます
- 92/100 (100点満点)

#### 判定
- [x] 合格（次へ進む）
- [ ] 要修正

### Step 3: MultiTimeframeAnalyzerへの責務移譲（次のステップ）

#### 作業内容
**目的**: バッファ管理機能をMultiTimeframeAnalyzerに実装

**実装予定の機能**:
1. **バッファ管理機能**
   - `_data_buffer` プロパティ: 履歴データの保持
   - `_max_history_bars` プロパティ: 最大バッファサイズ
   - `add_new_bar()` メソッド: 新しいバーの追加
   - `_manage_buffer_size()` メソッド: バッファサイズの管理
   
2. **状態確認機能**
   - `get_buffer_size()` メソッド: 現在のバッファサイズ取得
   - `is_ready()` メソッド: 分析準備完了状態の確認
   
3. **analyze_streaming()メソッドの改良**
   - 引数を省略可能にし、内部バッファから自動的にデータを取得
   - 後方互換性の維持

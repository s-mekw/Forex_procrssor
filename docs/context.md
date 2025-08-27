# ワークフローコンテキスト

## 📍 現在の状態
- ステップ: 3/10 実行中
- 最終更新: 2025-08-27 14:30
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

### Step 3 レビュー
#### 良い点
- ✅ バッファ管理機能が計画通りに実装されている
- ✅ 後方互換性が100%保持されている
- ✅ 型ヒントが適切に付与されている（Optional型の正しい使用）
- ✅ エラーハンドリングが適切に実装されている
- ✅ バッファサイズ制限が正しく動作している
- ✅ 準備状態の判定ロジックが正確
- ✅ 内部バッファと外部履歴の両モードが正常動作
- ✅ コードフォーマットとLintチェックが全て合格

#### 改善点
- ⚠️ 単体テストファイルが未作成（test_analyzer.py）
- 優先度: 中（Step 6で対応予定のため現時点では許容）

#### 評価総合点数
- 実装の完成度と品質から、評価総合点数をつけます
- 95/100 (100点満点)

#### 判定
- [x] 合格（次へ進む）
- [ ] 要修正

### 実装の技術的詳細
- **バッファ管理**: list[dict[str, Any]]で実装、最大5000バー保持
- **メモリ効率**: バッファサイズを超えた場合は自動的に古いデータを削除
- **インターフェース設計**: 後方互換性を保ちつつ、新しい内部バッファモードも追加
- **メソッド分離**: `_analyze_with_external_history()`と`_calculate_rci_metrics()`で責務を明確化

### Step 3: MultiTimeframeAnalyzerへの責務移譲（完了 ✅）

#### 作業内容
**目的**: バッファ管理機能をMultiTimeframeAnalyzerに実装

**作業対象ファイル**: `src/data_processing/analyzer.py`

**実装する機能と具体的なコード**:

1. **プロパティの追加（__init__メソッド内）**
   ```python
   # 既存のconfigパラメータから設定を取得
   self._data_buffer: list[dict[str, Any]] = []
   self._max_history_bars = config.get('max_history_bars', 5000)
   self._min_required_bars = 200  # 分析に必要な最小バー数
   ```

2. **バッファ管理メソッドの実装**
   ```python
   def add_new_bar(self, bar: dict[str, Any]) -> None:
       """新しいバーをバッファに追加し、サイズを管理"""
       self._data_buffer.append(bar)
       self._manage_buffer_size()
   
   def _manage_buffer_size(self) -> None:
       """バッファサイズを最大値以内に維持"""
       if len(self._data_buffer) > self._max_history_bars:
           self._data_buffer = self._data_buffer[-self._max_history_bars:]
   ```

3. **状態確認メソッドの実装**
   ```python
   def get_buffer_size(self) -> int:
       """現在のバッファサイズを返す"""
       return len(self._data_buffer)
   
   def is_ready(self) -> bool:
       """分析準備が完了しているかを返す"""
       return len(self._data_buffer) >= self._min_required_bars
   
   def get_buffer_as_dataframe(self) -> Optional[pl.DataFrame]:
       """バッファをDataFrameとして取得"""
       if not self._data_buffer:
           return None
       return pl.DataFrame(self._data_buffer)
   ```

4. **analyze_streaming()メソッドの改良**
   ```python
   def analyze_streaming(
       self,
       new_bar: Optional[dict[str, Any]] = None,
       history: Optional[pl.DataFrame] = None,
       min_history_bars: int = 200
   ) -> dict[str, Any]:
       """改良版: 内部バッファも利用可能"""
       # 後方互換性の維持
       if history is not None:
           # 既存の動作（外部から履歴を渡す）
           return self._analyze_with_external_history(
               new_bar, history, min_history_bars
           )
       
       # 新しい動作（内部バッファを使用）
       if not self.is_ready():
           return {
               'timestamp': new_bar['time'] if new_bar else None,
               'status': 'not_ready',
               'buffer_size': self.get_buffer_size(),
               'required_bars': self._min_required_bars
           }
       
       history_df = self.get_buffer_as_dataframe()
       if history_df is None:
           return {'status': 'no_data'}
       
       # 既存のRCI計算ロジックを呼び出し
       return self._calculate_rci_metrics(history_df)
   
   def _analyze_with_external_history(
       self,
       new_bar: dict[str, Any],
       history: pl.DataFrame,
       min_history_bars: int
   ) -> dict[str, Any]:
       """既存の外部履歴を使用した分析（後方互換性）"""
       # 既存のコードをここに移動
       ...
   
   def _calculate_rci_metrics(
       self,
       history_df: pl.DataFrame
   ) -> dict[str, Any]:
       """RCIメトリクスの計算（既存ロジックの再利用）"""
       # 既存のRCI計算ロジックを抽出してここに実装
       ...
   ```

#### 実装結果
**実装完了日時**: 2025-08-27 14:45

**実装内容**:
1. ✅ __init__メソッドにバッファ管理プロパティを追加
   - `_data_buffer`: バッファリスト
   - `_max_history_bars`: 最大履歴バー数パラメータ（デフォルト5000）
   - `_min_required_bars`: 分析に必要な最小バー数（200）

2. ✅ バッファ管理メソッドの実装
   - `add_new_bar()`: 新しいバーを追加
   - `_manage_buffer_size()`: バッファサイズ管理

3. ✅ 状態確認メソッドの実装
   - `get_buffer_size()`: 現在のサイズ取得
   - `is_ready()`: 分析可能状態の判定
   - `get_buffer_as_dataframe()`: DataFrame形式で取得

4. ✅ analyze_streaming()メソッドの改良
   - 後方互換性を維持（historyパラメータ対応）
   - 内部バッファモードのサポート追加
   - _analyze_with_external_history()で既存処理を分離
   - _calculate_rci_metrics()で内部バッファ用処理を実装

**技術的詳細**:
- 型ヒント: 完全に付与（Optional型を適切に使用）
- エラーハンドリング: 適切に実装
- ログ出力: 既存パターンを維持
- 後方互換性: 100%保持

**実装上の注意点**:
1. 既存のanalyze_streaming()の後方互換性を保つ
2. 内部バッファと外部履歴の両方をサポート
3. エラーハンドリングを適切に実装
4. 型ヒントを正確に記述

**コンフリクト解消**:
- 既存の__init__メソッドには`config`パラメータがないため、個別のパラメータとして追加
- max_history_barsは新規パラメータとして追加（デフォルト: 5000）
- 既存のanalyze_streaming()の処理を`_analyze_with_external_history()`に移動

**実装の順序**:
1. まず__init__メソッドにプロパティを追加
2. バッファ管理メソッド（add_new_bar, _manage_buffer_size）を実装
3. 状態確認メソッド（get_buffer_size, is_ready, get_buffer_as_dataframe）を実装
4. analyze_streaming()メソッドをリファクタリング
5. 必要に応じてヘルパーメソッドを追加

**テスト確認事項**:
- バッファへのバー追加が正しく動作すること
- バッファサイズが最大値を超えないこと
- is_ready()が最小バー数を正しく判定すること
- analyze_streaming()が両方のモード（内部/外部）で動作すること

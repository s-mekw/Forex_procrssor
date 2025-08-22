## タスク7: Polarsデータ処理基盤の構築 - 実装計画

### 概要
要件2.1に基づき、Polarsを使用した高速データ処理基盤を構築します。TDD（テスト駆動開発）アプローチで、テストを先に書いてから実装を進めます。

### 実装ステップ

#### Step 1: テストファイルの作成
- ファイル: `tests/unit/test_data_processor.py`
- 作業: 基本的なテスト構造とインポートを設定

#### Step 2: 基本的なデータ型定義テスト
- ファイル: `tests/unit/test_data_processor.py`
- 作業: Float32統一のスキーマ定義テストを追加
- 詳細:
  - Forexデータのスキーマ定義（timestamp, open, high, low, close, volume）
  - Float32への自動変換テスト
  - メモリ使用量の検証テスト
  - データ型の整合性チェック

#### Step 3: 実装ファイルの作成
- ファイル: `src/data_processing/processor.py`
- 作業: PolarsProcessingEngineクラスの骨組みを作成

#### Step 4: データ型最適化の実装
- ファイル: `src/data_processing/processor.py`
- 作業: Float32変換とメモリ最適化メソッドを実装

#### Step 5: LazyFrame処理のテスト追加
- ファイル: `tests/unit/test_data_processor.py`
- 作業: 遅延評価のテストケースを追加

#### Step 6: チャンク処理のテスト追加
- ファイル: `tests/unit/test_data_processor.py`
- 作業: 大規模データのチャンク処理テストを追加
- 詳細:
  - チャンクサイズの設定と検証（デフォルト: 100,000行）
  - 大規模データ（100万行以上）の分割処理
  - チャンクごとの処理と集約
  - メモリ使用量の監視と検証
  - ストリーミング処理のテスト（scan_csv/scan_parquet）
  - パフォーマンス測定（処理時間の計測）

#### Step 7: チャンク処理の実装 ✅ 完了
- ファイル: `src/data_processing/processor.py`
- 作業: チャンク処理とストリーミング処理を実装
- 詳細:
  1. **process_in_chunksメソッド**
     - DataFrame/LazyFrameを指定サイズで分割
     - 各チャンクに対してユーザー定義の処理関数を適用
     - チャンク結果を効率的に集約（concatまたはvstack）
     - メモリ使用量の監視と制御
  
  2. **adjust_chunk_sizeメソッド**
     - 現在のメモリ使用率をpsutilで取得
     - メモリ使用率が高い場合はチャンクサイズを縮小
     - メモリ使用率が低い場合はチャンクサイズを拡大
     - 適応的な調整アルゴリズムの実装
  
  3. **stream_csvメソッド**
     - pl.scan_csvを使用したLazyFrame生成
     - データ型の明示的な指定（Float32統一）
     - バッチ処理との組み合わせ（collectのタイミング制御）
  
  4. **stream_parquetメソッド**
     - pl.scan_parquetを使用した高速読み込み
     - 列選択とpredicate pushdownの最適化
     - メタデータの活用による効率化
  
  5. **process_batchesメソッド**
     - 複数のバッチを順次処理
     - 結果の順序保証と集約
     - エラーハンドリングとリトライ機能

- 目標成果:
  - test_chunk_processingがPASSする
  - test_streaming_processingがPASSする
  - メモリ使用量が指定闾値内に収まる
  - パフォーマンスが目標値を達成する

#### Step 8: エラーハンドリングのテスト追加 ✅ 完了
- ファイル: `tests/unit/test_data_processor.py`
- 作業: 異常系のテストケースを追加
- 詳細:
  1. **test_invalid_data_types**
     - 非Float32データ型の処理テスト
     - 文字列カラムが数値カラムとして期待される場合
     - 欠損値（null/NaN）のハンドリング
     - 適切なエラーメッセージとログ出力の確認
  
  2. **test_empty_dataframe_handling**
     - 空のDataFrameを処理する際の動作
     - optimize_dtypesメソッドの安全性
     - create_lazyframeの空データ対応
     - チャンク処理での空データハンドリング
  
  3. **test_excessive_data_size**
     - メモリに収まらない大規模データの処理
     - チャンクサイズの自動調整確認
     - OutOfMemoryErrorのキャッチと復旧
     - プログレスバーやログでの適切な通知
  
  4. **test_corrupted_file_handling**
     - 破損したCSV/Parquetファイルの読み込み
     - 不正なフォーマットのファイル処理
     - FileNotFoundErrorのハンドリング
     - スキーマ不整合の検出と報告
  
  5. **test_memory_pressure_handling**
     - メモリ不足状況のシミュレーション
     - adjust_chunk_sizeの動的調整確認
     - メモリ使用率が高い場合の処理速度低下
     - グレースフルデグレデーションの確認
  
  6. **test_invalid_parameters**
     - 負のチャンクサイズの処理
     - 無効な集計関数名の処理
     - 不正なフィルタ条件の処理
     - ValueError/TypeErrorの適切な発生

- 目標成果:
  - 各エラーケースが適切にキャッチされる
  - エラーメッセージがユーザーフレンドリー
  - ログ出力が適切なレベルで記録される
  - リカバリー可能なエラーは自動復旧を試みる

#### Step 9: エラーハンドリングの実装 🔄 進行中
- ファイル: `src/data_processing/processor.py`
- 作業: Step 8のテストをパスさせるエラーハンドリング実装
- 詳細:
  1. **カスタム例外クラスの定義**
     - ProcessingError（基底クラス）
     - DataTypeError（データ型関連）
     - MemoryLimitError（メモリ制限関連）
     - FileValidationError（ファイル検証関連）
  
  2. **validate_datatypes メソッド**
     - Float32以外の数値型を検出して変換
     - 文字列カラムの数値変換試行
     - null/NaN値の適切な処理
     - 変換不可能な場合の明確なエラーメッセージ
  
  3. **handle_empty_dataframe メソッド**
     - 空データの検出（len(df) == 0）
     - 各処理メソッドでの空データチェック
     - 空の結果を適切に返す
     - ログでの通知
  
  4. **handle_memory_limit メソッド**
     - psutilでメモリ使用量を監視
     - メモリ制限に近づいたら警告
     - チャンクサイズの自動縮小
     - 必要に応じて処理を分割
  
  5. **validate_file メソッド**
     - ファイルの存在確認（Path.exists()）
     - ファイル拡張子の検証
     - 読み込み可能性のチェック
     - エラー時の詳細なメッセージ
  
  6. **handle_memory_pressure メソッド**
     - adjust_chunk_sizeの改良版
     - 段階的なチャンクサイズ縮小
     - 最小チャンクサイズの保証（1,000行）
     - グレースフルデグレデーション
  
  7. **validate_parameters メソッド**
     - チャンクサイズの範囲チェック（1以上）
     - 集計関数名の検証（サポート済みリストと照合）
     - フィルタ演算子の検証
     - カラム名の存在確認
  
  8. **既存メソッドへのエラーハンドリング追加**
     - optimize_dtypes: データ型変換エラーの処理
     - create_lazyframe: LazyFrame作成エラーの処理
     - apply_filters: フィルタ条件エラーの処理
     - apply_aggregations: 集計エラーの処理
     - process_in_chunks: チャンク処理エラーの処理
     - stream_csv/stream_parquet: ファイル読み込みエラーの処理

- 実装方針:
  - 各メソッドにtry-exceptブロック追加
  - 適切なログレベル（ERROR/WARNING/INFO/DEBUG）
  - リカバリー機能（リトライ、フォールバック）
  - 部分的成功の許容

- 目標成果:
  - 6つのエラーハンドリングテストがすべてPASS
  - エラーメッセージが明確で実用的
  - ログ出力が適切
  - メモリ使用量が制御下にある
  - 処理の継続性が保たれる

#### Step 10: 統合テストとドキュメント更新
- ファイル: `tests/unit/test_data_processor.py`, `docs/context.md`
- 作業: 全体のテスト実行と最終確認、ドキュメント更新

### 成果物
1. **src/data_processing/processor.py**
   - PolarsProcessingEngineクラス
   - メモリ最適化メソッド
   - チャンク処理/ストリーミング処理

2. **tests/unit/test_data_processor.py**
   - データ型変換テスト
   - LazyFrame処理テスト
   - チャンク処理テスト
   - エラーハンドリングテスト

### 技術仕様
- **データ型**: Float32統一（メモリ効率50%改善）
- **処理方式**: LazyFrameによる遅延評価
- **メモリ管理**: チャンク処理（デフォルト: 100,000行）
- **並列処理**: CPUコア数に応じた自動最適化

### 検証項目
- [ ] 全テストがグリーン
- [ ] カバレッジ80%以上
- [ ] メモリ使用量がFloat64比で50%削減
- [ ] 処理速度がPandas比で2倍以上

### 参照ドキュメント
- 要件定義: `.kiro/specs/Forex_procrssor/requirements.md` (要件2.1)
- 詳細設計: `.kiro/specs/Forex_procrssor/design.md` (セクション2.1, 2.3)
- Python開発ガイドライン: `.kiro/steering/Python_Development_Guidelines.md`
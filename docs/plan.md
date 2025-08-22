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

#### Step 3: 実装ファイルの作成
- ファイル: `src/data_processing/processor.py`
- 作業: PolarsProcessingEngineクラスの骨組みを作成

#### Step 4: データ型最適化の実装
- ファイル: `src/data_processing/processor.py`
- 作業: Float32変換とメモリ最適化メソッドを実装

#### Step 5: LazyFrame処理のテスト追加
- ファイル: `tests/unit/test_data_processor.py`
- 作業: 遅延評価のテストケースを追加

#### Step 6: LazyFrame処理の実装
- ファイル: `src/data_processing/processor.py`
- 作業: LazyFrameを使った遅延評価処理を実装

#### Step 7: チャンク処理のテスト追加
- ファイル: `tests/unit/test_data_processor.py`
- 作業: 大規模データのチャンク処理テストを追加

#### Step 8: チャンク処理の実装
- ファイル: `src/data_processing/processor.py`
- 作業: チャンク処理とストリーミング処理を実装

#### Step 9: エラーハンドリングのテスト追加
- ファイル: `tests/unit/test_data_processor.py`
- 作業: 異常系のテストケースを追加

#### Step 10: エラーハンドリングの実装
- ファイル: `src/data_processing/processor.py`
- 作業: 例外処理とロギングを実装

#### Step 11: 統合テストとドキュメント更新
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
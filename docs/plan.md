## 汎用実装計画（リンク集・導線テンプレート）

このファイルは、../.kiro/specs/Forex_procrssor/tasks.md に定義された任意のタスクを実装するための最小限の導線です。具体的な設計・仕様は各ドキュメントへ直接リンクし、本ファイルには詳細を書きません。

### タスク選択
- 対象タスクは `../.kiro/specs/Forex_procrssor/tasks.md` を参照し、対応するチェックリスト/説明/要件番号を確認してください。
- 現在の対象タスク: 
  - [ ] 6. ティック→バー変換エンジンの実装
    - tests/unit/test_tick_to_bar.pyにバー生成とタイムスタンプ整合性のテストを作成
    - src/mt5_data_acquisition/tick_to_bar.pyにTickToBarConverterクラスを実装
    - リアルタイム1分足生成と未完成バーの継続更新機能を実装
    - 30秒以上のティック欠損時の警告機能を追加
    - _要件: 1.4_ of `../.kiro/specs/Forex_procrssor/requirements.md`
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

#### 現在のタスク
**タスク6: ティック→バー変換エンジンの実装**
- 要件1.4の実装（リアルタイムティック→1分足OHLC変換）
- テスト駆動開発（TDD）アプローチを採用

#### 実装対象ファイル
1. `tests/unit/test_tick_to_bar.py` - ユニットテスト
2. `src/mt5_data_acquisition/tick_to_bar.py` - 本体実装
3. `tests/integration/test_tick_to_bar_integration.py` - 統合テスト（Step 7）

#### 技術的決定事項
- **データ処理**: Polarsを使用（pandas禁止）
- **データ構造**: Pydanticでデータモデル定義
- **時間管理**: datetimeでタイムスタンプ管理、1分足の境界判定
- **ロギング**: 構造化ログ（JSON形式）で欠損検知などを記録
- **エラー処理**: カスタム例外クラスで明確なエラー分類

#### 実装の優先順位
1. 基本的な1分足生成機能（Step 1-3）
2. リアルタイム更新機能（Step 4）  
3. 品質・信頼性機能（Step 5-6）
4. パフォーマンス最適化（Step 7）
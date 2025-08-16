# Claude Code Spec-Driven Development

KiroスタイルのSpec Driven Developmentを、claude codeスラッシュコマンド、フック、エージェントを使用して実装。

## プロジェクトコンテキスト

### パス
- ステアリング: `.kiro/steering/`
- 仕様: `.kiro/specs/`
- コマンド: `.claude/commands/`

### ステアリング vs 仕様

**ステアリング** (`.kiro/steering/`) - プロジェクト全体のルールとコンテキストでAIをガイド  
**仕様** (`.kiro/specs/`) - 個別機能の開発プロセスを形式化

### アクティブな仕様
- `.kiro/specs/`でアクティブな仕様を確認
- `/kiro:spec-status [feature-name]`を使用して進捗を確認

## 開発ガイドライン
- 英語で思考し、日本語で回答を生成する（思考は英語、回答の生成は日本語で行うように）

## ワークフロー

### フェーズ0: ステアリング（オプション）
`/kiro:steering` - ステアリングドキュメントの作成/更新
`/kiro:steering-custom` - 専門的なコンテキスト用のカスタムステアリングの作成

**注意**: 新機能や小さな追加ではオプション。spec-initに直接進むことが可能。

### フェーズ1: 仕様作成
1. `/kiro:spec-init [詳細な説明]` - 詳細なプロジェクト説明で仕様を初期化
2. `/kiro:spec-requirements [feature]` - 要件ドキュメントの生成
3. `/kiro:spec-design [feature]` - インタラクティブ: "requirements.mdをレビューしましたか？ [y/N]"
4. `/kiro:spec-tasks [feature]` - インタラクティブ: 要件とデザインの両方のレビューを確認

### フェーズ2: 進捗追跡
`/kiro:spec-status [feature]` - 現在の進捗とフェーズを確認

## 開発ルール
1. **ステアリングを考慮**: 主要な開発前に`/kiro:steering`を実行（新機能ではオプション）
2. **3フェーズ承認ワークフローに従う**: 要件 → デザイン → タスク → 実装
3. **承認が必要**: 各フェーズは人間のレビューが必要（インタラクティブプロンプトまたは手動）
4. **フェーズのスキップ禁止**: デザインには承認された要件が必要；タスクには承認されたデザインが必要
5. **タスクステータスの更新**: 作業時にタスクを完了としてマーク
6. **ステアリングを最新に保つ**: 重要な変更後に`/kiro:steering`を実行
7. **仕様コンプライアンスの確認**: `/kiro:spec-status`を使用して整合性を確認

## ステアリング設定

### 現在のステアリングファイル
`/kiro:steering`コマンドで管理。ここでの更新はコマンドの変更を反映。

### アクティブなステアリングファイル
- `product.md`: 常に含まれる - プロダクトコンテキストとビジネス目標
- `tech.md`: 常に含まれる - 技術スタックとアーキテクチャの決定
- `structure.md`: 常に含まれる - ファイル構成とコードパターン
- `Python_Development_Guidelines.md`: 常に含まれる - Python開発ルールとスタンダード

### カスタムステアリングファイル
<!-- /kiro:steering-customコマンドで追加 -->
<!-- 形式: 
- `filename.md`: モード - パターン - 説明
  モード: Always|Conditional|Manual
  パターン: Conditionalモード用のファイルパターン（例: `"*.test.js"`）
-->

### 含めるモード
- **Always**: すべてのインタラクションで読み込まれる（デフォルト）
- **Conditional**: 特定のファイルパターンで読み込まれる（例: `"*.test.js"`）
- **Manual**: `@filename.md`構文で参照

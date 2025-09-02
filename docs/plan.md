## 汎用実装計画（リンク集・導線テンプレート）

このファイルは、../.kiro/specs/Forex_procrssor/tasks.md に定義された任意のタスクを実装するための最小限の導線です。具体的な設計・仕様は各ドキュメントへ直接リンクし、本ファイルには詳細を書きません。

### タスク選択
- 対象タスクは `../.kiro/specs/Forex_procrssor/tasks.md` を参照し、対応するチェックリスト/説明/要件番号を確認してください。
- 現在の対象タスク: 
  - [ ] 11. InfluxDB接続とデータモデル設定
  - tests/unit/test_influx_handler.pyに接続とクエリのテストを作成
  - src/storage/influx_handler.pyにInfluxDBHandlerクラスを実装
  - 環境変数による接続設定と認証を実装
  - タグ（通貨ペア、時間足）とフィールド（OHLC、Volume）のスキーマを定義
  - _要件: 3.1_ of `../.kiro/specs/Forex_procrssor/requirements.md`
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

#### タスク11: InfluxDB接続とデータモデル設定 - 実装計画

##### 実装ステップ詳細

**Step 1: テスト環境の準備とInfluxDBクライアントパッケージの追加**
- influxdb-client-python パッケージを uv add で追加
- 開発環境用のInfluxDBモックライブラリを追加（必要に応じて）

**Step 2: InfluxDBHandlerクラスの接続設定とヘルスチェック機能の実装**
- src/storage/influx_handler.py を新規作成
- 基本的な接続管理（connect, disconnect）メソッドの実装
- health_check メソッドで接続状態を確認する機能を実装
- 非同期対応（async/await）での実装

**Step 3: データモデル（タグ・フィールド）のスキーマ定義**
- Line Protocolフォーマットの定義
- タグ: symbol（通貨ペア）、timeframe（時間足）、broker（ブローカー識別子）
- フィールド: open、high、low、close、volume、spread（すべてFloat32）
- OHLCデータ用のDataPointクラスまたは辞書形式の定義

**Step 4: 接続とヘルスチェックのユニットテスト作成**
- tests/unit/test_influx_handler.py を新規作成
- 接続成功/失敗のテストケース
- ヘルスチェックの正常/異常時のテストケース
- 環境変数が設定されていない場合のエラーハンドリングテスト

**Step 5: 書き込み機能（単一ポイント）の実装とテスト**
- write_point メソッドの実装（単一のOHLCデータポイントを書き込み）
- 書き込み成功/失敗のテストケース追加
- タグとフィールドの正しいフォーマットのテスト

**Step 6: クエリ機能（基本的な読み取り）の実装とテスト**
- query メソッドの実装（Flux言語による基本的なクエリ）
- 時間範囲を指定したデータ取得
- 通貨ペアと時間足でフィルタリングする機能
- クエリ結果をPolars DataFrameに変換する機能

**Step 7: 環境変数による接続設定の実装**
- 以下の環境変数をサポート:
  - INFLUXDB_URL（デフォルト: http://localhost:8086）
  - INFLUXDB_TOKEN（必須）
  - INFLUXDB_ORG（必須）
  - INFLUXDB_BUCKET（必須）
- src/common/config.py との統合（必要に応じて）

**Step 8: エラーハンドリングと再接続ロジックの実装**
- カスタム例外クラスの定義（InfluxDBConnectionError など）
- 指数バックオフによる再接続ロジック
- サーキットブレーカーパターンの基本実装（オプション）

##### 各ステップの実装時間目安
- Step 1: 5分
- Step 2: 30分
- Step 3: 15分
- Step 4: 20分
- Step 5: 20分
- Step 6: 30分
- Step 7: 15分
- Step 8: 20分

##### 依存関係
- Step 1が完了しないと他のステップは進められない
- Step 2-3は並行実装可能
- Step 4はStep 2の後に実装
- Step 5-6はStep 3の後に実装
- Step 7-8は最後に実装

##### 決定事項
- 非同期処理を前提とした実装（asyncio使用）
- データ型はFloat32で統一（メモリ効率最適化）
- Polarsとの統合を考慮した設計
- 環境変数による設定管理
## 汎用実装計画（リンク集・導線テンプレート）

このファイルは、../.kiro/specs/Forex_procrssor/tasks.md に定義された任意のタスクを実装するための最小限の導線です。具体的な設計・仕様は各ドキュメントへ直接リンクし、本ファイルには詳細を書きません。

### タスク選択
- 対象タスクは `../.kiro/specs/Forex_procrssor/tasks.md` を参照し、対応するチェックリスト/説明/要件番号を確認してください。
- 現在の対象タスク: 
  - [ ] 5. 履歴OHLCデータ取得とバッチ処理
    - tests/unit/test_ohlc_fetcher.pyに履歴データ取得と欠損検出のテストを作成
    - src/mt5_data_acquisition/ohlc_fetcher.pyにHistoricalDataFetcherクラスを実装
    - 10,000バー単位のバッチ処理と並列フェッチ機能を実装
    - 複数時間足（1分〜日足）のサポートを追加
    - _要件: 1.3_ of `../.kiro/specs/Forex_procrssor/requirements.md`

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

### 作業メモ欄（自由記述）
- ここには「選択タスク」「対象ファイル」「追加の参照リンク」「決定事項」などを簡潔に記録してください。

---

## タスク5実装詳細計画

### 実装方針
1. **TDD（テスト駆動開発）アプローチ**
   - まずテストを作成し、その後実装を進める
   - モックを活用してMT5依存を分離

2. **段階的実装**
   - 基本機能から始めて、徐々に高度な機能を追加
   - 各ステップでテストが通ることを確認

3. **パフォーマンス最適化**
   - Polars LazyFrameによる遅延評価
   - ThreadPoolExecutorによる並列処理
   - メモリ効率を考慮したバッチサイズ設定

### クラス設計概要

```python
class HistoricalDataFetcher:
    """履歴OHLCデータ取得クラス"""
    
    def __init__(self, connection_manager: MT5ConnectionManager):
        """初期化"""
        self.connection_manager = connection_manager
        self.chunk_size = 10000  # バッチサイズ
        self.max_workers = 4     # 並列ワーカー数
    
    async def fetch_ohlc_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_time: datetime,
        end_time: datetime
    ) -> pl.LazyFrame:
        """OHLCデータを取得（メインメソッド）"""
        pass
    
    def _fetch_chunk(self, params: dict) -> List[dict]:
        """単一チャンクのデータ取得（内部メソッド）"""
        pass
    
    def _detect_missing_periods(self, df: pl.DataFrame) -> List[tuple]:
        """欠損期間を検出"""
        pass
    
    def _convert_timeframe(self, timeframe: TimeFrame) -> int:
        """TimeFrameをMT5定数に変換"""
        pass
```

### エラーハンドリング方針
- 接続エラー: 指数バックオフでリトライ
- データ欠損: ログに記録して処理継続
- 無効なパラメータ: ValueErrorを発生

### テスト方針
- MT5接続はモックで代替
- 各メソッドを個別にテスト
- 境界値テスト（大量データ、空データ等）
- 並列処理の正確性を検証

### 実装優先順位
1. 基本的なOHLCデータ取得機能
2. バッチ処理によるメモリ効率化
3. 並列処理による高速化
4. データ品質チェック機能
5. エラーハンドリングとリトライ
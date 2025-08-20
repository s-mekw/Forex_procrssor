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
- 現在: Step 2/7 実装中

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

#### Step 2 実装詳細（現在作業中）

##### データモデル設計
```python
# Tickクラス（Pydantic BaseModel）
- symbol: str  # 通貨ペア（例: "EURUSD"）
- time: datetime  # ティックのタイムスタンプ
- bid: Decimal  # Bid価格
- ask: Decimal  # Ask価格  
- volume: Decimal  # 取引量

# Barクラス（Pydantic BaseModel）
- symbol: str  # 通貨ペア
- time: datetime  # バー開始時刻（分の00秒）
- end_time: datetime  # バー終了時刻（分の59.999999秒）
- open: Decimal  # 始値（最初のティックのbid）
- high: Decimal  # 高値（期間中の最高bid）
- low: Decimal  # 安値（期間中の最低bid）
- close: Decimal  # 終値（最後のティックのbid）
- volume: Decimal  # 期間中の総取引量
- is_complete: bool  # バー完成フラグ
- tick_count: int  # ティック数（オプション）
- avg_spread: Decimal | None  # 平均スプレッド（オプション）
```

##### TickToBarConverterクラス設計
```python
class TickToBarConverter:
    def __init__(self, symbol: str, timeframe: int = 60):
        """初期化
        Args:
            symbol: 通貨ペア
            timeframe: バーの時間枠（秒単位、デフォルト60秒=1分）
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.current_bar: Bar | None = None  # 現在作成中のバー
        self.completed_bars: list[Bar] = []  # 完成したバーのリスト
        self.on_bar_complete: Callable[[Bar], None] | None = None  # コールバック
        self._current_ticks: list[Tick] = []  # 現在のバー用のティック
```

##### 実装順序
1. **フェーズ1**: データモデル（Tick, Bar）をPydanticで定義
2. **フェーズ2**: TickToBarConverterクラスの基本構造
   - __init__メソッド
   - プロパティとフィールドの定義
3. **フェーズ3**: メソッドスケルトン（NotImplementedError）
   - add_tick() - メインのティック処理
   - get_current_bar() - 現在のバー取得
   - 内部ヘルパーメソッド群

##### テスト活性化計画
- Step 2完了時点: `test_converter_initialization`のみ有効化
- Step 3で: `test_single_minute_bar_generation`等を有効化
- 段階的にテストを有効化していく

##### 技術的注意点
- Decimalを使用して浮動小数点の精度問題を回避
- datetimeのタイムゾーン処理は後のステップで考慮
- Polarsは集計処理が必要になるStep 7で導入予定
- 現段階では純粋なPythonで実装
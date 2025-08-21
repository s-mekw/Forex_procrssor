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
- 現在: Step 4/7 実装中
- 状況: Step 1-3完了（4テストPASSED、6テストSKIPPED）

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
- Step 2完了時点: `test_converter_initialization`のみ有効化（完了）
- Step 3で: 
  - `test_single_minute_bar_generation` - 1分足バーの基本生成
  - `test_timestamp_alignment` - タイムスタンプの正確性
  - `test_ohlcv_calculation` - OHLCV値の計算精度
- 段階的にテストを有効化していく

##### 技術的注意点
- Decimalを使用して浮動小数点の精度問題を回避
- datetimeのタイムゾーン処理は後のステップで考慮
- Polarsは集計処理が必要になるStep 7で導入予定
- 現段階では純粋なPythonで実装

#### Step 3 実装詳細（現在作業中）

##### add_tick()メソッドの実装フロー
```python
def add_tick(self, tick: Tick) -> Bar | None:
    """ティックを追加し、完成したバーを返す
    
    処理フロー:
    1. ティックのタイムスタンプからバー開始時刻を計算
    2. 新しいバーが必要か判定
    3. 必要なら現在のバーを完成させて新しいバーを作成
    4. 現在のバーを更新
    5. 完成したバーを返却（またはNone）
    """
    bar_start = self._get_bar_start_time(tick.time)
    completed_bar = None
    
    # 新しいバーが必要か判定
    if self.current_bar is None or self.current_bar.time != bar_start:
        # 現在のバーを完成させる
        if self.current_bar:
            self.current_bar.is_complete = True
            self.completed_bars.append(self.current_bar)
            if self.on_bar_complete:
                self.on_bar_complete(self.current_bar)
            completed_bar = self.current_bar
        
        # 新しいバーを作成
        self._create_new_bar(tick)
    else:
        # 現在のバーを更新
        self._update_bar(tick)
    
    return completed_bar
```

##### _create_new_bar()メソッド
```python
def _create_new_bar(self, tick: Tick) -> None:
    """新しいバーを作成して初期化"""
    bar_start = self._get_bar_start_time(tick.time)
    bar_end = self._get_bar_end_time(bar_start)
    
    self.current_bar = Bar(
        symbol=self.symbol,
        time=bar_start,
        end_time=bar_end,
        open=tick.bid,
        high=tick.bid,
        low=tick.bid,
        close=tick.bid,
        volume=tick.volume,
        tick_count=1,
        avg_spread=tick.ask - tick.bid,
        is_complete=False
    )
    
    self._current_ticks = [tick]
```

##### _update_bar()メソッド
```python
def _update_bar(self, tick: Tick) -> None:
    """現在のバーのOHLCVを更新"""
    if not self.current_bar:
        return
    
    # OHLCVの更新
    self.current_bar.high = max(self.current_bar.high, tick.bid)
    self.current_bar.low = min(self.current_bar.low, tick.bid)
    self.current_bar.close = tick.bid
    self.current_bar.volume += tick.volume
    self.current_bar.tick_count += 1
    
    # 平均スプレッドの更新（累積平均）
    if self.current_bar.avg_spread:
        total_spread = self.current_bar.avg_spread * (self.current_bar.tick_count - 1)
        new_spread = tick.ask - tick.bid
        self.current_bar.avg_spread = (total_spread + new_spread) / self.current_bar.tick_count
    
    self._current_ticks.append(tick)
```

##### ヘルパーメソッド
```python
def _get_bar_start_time(self, tick_time: datetime) -> datetime:
    """ティックタイムからバー開始時刻を計算（分単位に切り捨て）"""
    # timeframeが60秒（1分）の場合、分単位に切り捨て
    if self.timeframe == 60:
        return tick_time.replace(second=0, microsecond=0)
    # 他のタイムフレームの場合の処理（将来的に拡張）
    else:
        # timeframe秒単位で切り捨て
        epoch = datetime(1970, 1, 1, tzinfo=tick_time.tzinfo)
        seconds_since_epoch = (tick_time - epoch).total_seconds()
        bar_start_seconds = (seconds_since_epoch // self.timeframe) * self.timeframe
        return epoch + timedelta(seconds=bar_start_seconds)

def _get_bar_end_time(self, bar_start: datetime) -> datetime:
    """バー開始時刻から終了時刻を計算"""
    return bar_start + timedelta(seconds=self.timeframe) - timedelta(microseconds=1)
```

##### 実装時のポイント
1. **ティックの順序保証**: ティックが時間順に来ることを前提とする（Step 6で逆転処理を追加）
2. **スプレッド計算**: Bid/Askの差を累積平均で計算
3. **ボリューム累積**: 各ティックのボリュームを単純加算
4. **コールバック処理**: バー完成時に外部に通知

##### テスト有効化の準備
- `test_single_minute_bar_generation`: 基本的な1分足生成を検証
- `test_timestamp_alignment`: バーの開始/終了時刻が正しいことを検証
- `test_ohlcv_calculation`: OHLCV値の正確性を検証

#### Step 4 実装詳細（現在作業中）

##### 実装対象
Step 4では、既に実装済みの`get_current_bar()`メソッドの動作確認と、関連テストの有効化を行います。

##### 有効化するテスト
1. **test_get_current_incomplete_bar** - 未完成バーの取得テスト
   - 現在作成中のバーが正しく取得できることを確認
   - バーが存在しない場合はNoneを返すことを確認

2. **test_empty_bar_handling** - 空のバー処理テスト
   - ティックがない場合のバー処理を確認
   - 初期状態でcurrent_barがNoneであることを確認
   - completed_barsが空であることを確認

3. **test_multiple_bars_generation** - 複数バー生成テスト
   - 複数分のティックデータから正しく複数のバーが生成されることを確認
   - バー境界を越えた際の処理が正しく動作することを確認
   - 各バーのOHLCV値が正確であることを確認

##### 実装の確認ポイント
- `get_current_bar()`が未完成バーを正しく返す
- バー境界を越えた際の処理が正確
- 空のバー処理（初期状態）が適切
- High/Low値の動的更新が正しく動作している（Step 3で実装済み）

##### エッジケース
- ティックがない状態でのget_current_bar()呼び出し → None
- 単一ティックのバー → Open=High=Low=Close
- 複数分にまたがるティックデータ → 正しくバーが分割される

##### 実装手順
1. test_get_current_incomplete_barのskipマークを削除
2. test_empty_bar_handlingのskipマークを削除 
3. test_multiple_bars_generationのskipマークを削除
4. テストを実行して動作確認（7 passed, 3 skipped目標）
5. 必要に応じて既存実装の微調整
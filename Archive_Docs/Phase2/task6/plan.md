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

#### Step 4 実装詳細（完了）

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

##### 実装結果
- ✅ 7 tests passed, 3 skipped達成
- ✅ 既存実装で全てのテストが正常動作
- ✅ エッジケースも適切に処理

#### Step 5 実装詳細（完了）

##### 実装対象
ティック欠損検知と警告機能を追加し、30秒以上のギャップがある場合に構造化ログで警告を出力します。

##### 1. クラスプロパティの追加
```python
class TickToBarConverter:
    def __init__(self, symbol: str, timeframe: int = 60, on_bar_complete: Callable[[Bar], None] | None = None):
        # 既存のプロパティ...
        self.last_tick_time: datetime | None = None  # 最後のティック受信時刻
        self.logger = self._setup_logger()  # JSONロガーの設定
```

##### 2. ロガー設定メソッド
```python
def _setup_logger(self) -> logging.Logger:
    """構造化ログのためのロガー設定"""
    logger = logging.getLogger(f"TickToBarConverter.{self.symbol}")
    logger.setLevel(logging.WARNING)
    
    # JSONフォーマットのハンドラー設定
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger
```

##### 3. ティック欠損検知メソッド
```python
def check_tick_gap(self, tick: Tick) -> bool:
    """30秒以上のティック欠損を検知
    
    Returns:
        bool: 欠損が検出された場合True
    """
    if self.last_tick_time is None:
        return False
    
    gap_seconds = (tick.time - self.last_tick_time).total_seconds()
    
    if gap_seconds > 30:
        # 構造化ログ出力
        log_data = {
            "timestamp": tick.time.isoformat(),
            "level": "WARNING",
            "message": "Tick gap detected",
            "symbol": self.symbol,
            "gap_seconds": gap_seconds,
            "last_tick": self.last_tick_time.isoformat(),
            "current_tick": tick.time.isoformat()
        }
        self.logger.warning(json.dumps(log_data))
        return True
    
    return False
```

##### 4. add_tick()メソッドの更新
```python
def add_tick(self, tick: Tick) -> Bar | None:
    """ティックを追加し、完成したバーを返す（更新版）"""
    # 欠損検知を追加
    self.check_tick_gap(tick)
    
    # 既存の処理...
    bar_start = self._get_bar_start_time(tick.time)
    completed_bar = None
    
    # バー処理ロジック（既存）...
    
    # 最後のティック時刻を更新
    self.last_tick_time = tick.time
    
    return completed_bar
```

##### 5. テストの有効化と更新
- **test_tick_gap_warning**: 
  - 30秒以上のギャップでWARNINGログ出力を確認
  - ログのモック化とアサーション追加
  
- **test_volume_aggregation**:
  - ボリューム累積の正確性を確認
  - 複数ティックでのボリューム合計値の検証

##### 6. 実装手順
1. TickToBarConverterクラスにlast_tick_timeプロパティを追加
2. _setup_logger()メソッドを実装
3. check_tick_gap()メソッドを実装
4. add_tick()メソッドに欠損検知と時刻更新を統合
5. json, loggingモジュールのインポート追加
6. test_tick_gap_warningのskipマーク削除とモック設定
7. test_volume_aggregationのskipマーク削除
8. テスト実行（9 passed, 1 skipped目標）

##### 技術的考慮事項
- logging.WARNING レベルを使用
- JSON形式で構造化ログを出力
- テストではunittest.mockを使用してログ出力を検証
- 欠損検知は非破壊的（警告のみ、処理は継続）

#### Step 6 実装詳細（完了）

##### 実装対象
エッジケースとエラーハンドリングを強化し、無効なデータやタイムスタンプ逆転に対処します。

##### 1. Tickモデルのバリデーション強化
```python
from pydantic import field_validator

class Tick(BaseModel):
    """ティックデータモデル（バリデーション強化版）"""
    symbol: str
    time: datetime
    bid: Decimal
    ask: Decimal
    volume: Decimal
    
    @field_validator('bid', 'ask')
    @classmethod
    def validate_prices(cls, v: Decimal, info) -> Decimal:
        """価格のバリデーション"""
        if v is None:
            raise ValueError(f"{info.field_name} cannot be None")
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive, got {v}")
        return v
    
    @field_validator('ask')
    @classmethod
    def validate_spread(cls, v: Decimal, info) -> Decimal:
        """スプレッドのバリデーション"""
        bid = info.data.get('bid')
        if bid is not None and v < bid:
            raise ValueError(f"Ask ({v}) cannot be less than bid ({bid})")
        return v
    
    @field_validator('volume')
    @classmethod
    def validate_volume(cls, v: Decimal) -> Decimal:
        """ボリュームのバリデーション"""
        if v < 0:
            raise ValueError(f"Volume must be non-negative, got {v}")
        return v
```

##### 2. タイムスタンプ逆転検知の実装
```python
def add_tick(self, tick: Tick) -> Bar | None:
    """ティックを追加（エラーハンドリング強化版）"""
    try:
        # バリデーション（Pydanticが自動実行）
        # タイムスタンプ逆転チェック
        if self.last_tick_time and tick.time < self.last_tick_time:
            log_data = {
                "event": "timestamp_reversal_detected",
                "level": "ERROR",
                "symbol": self.symbol,
                "current_tick_time": tick.time.isoformat(),
                "last_tick_time": self.last_tick_time.isoformat(),
                "reversal_seconds": (self.last_tick_time - tick.time).total_seconds()
            }
            self.logger.error(json.dumps(log_data))
            return None  # ティックを破棄
        
        # 欠損検知（既存）
        self.check_tick_gap(tick)
        
        # 通常のバー処理（既存）
        bar_start = self._get_bar_start_time(tick.time)
        completed_bar = None
        
        # ... バー処理ロジック ...
        
        self.last_tick_time = tick.time
        return completed_bar
        
    except ValidationError as e:
        # バリデーションエラーのログ出力
        log_data = {
            "event": "invalid_tick_data",
            "level": "ERROR",
            "symbol": self.symbol,
            "error": str(e),
            "tick_time": tick.time.isoformat() if hasattr(tick, 'time') else None
        }
        self.logger.error(json.dumps(log_data))
        return None  # ティックを破棄
```

##### 3. ロガー設定の更新
```python
def _setup_logger(self) -> logging.Logger:
    """構造化ログのためのロガー設定（ERROR追加）"""
    logger = logging.getLogger(f"TickToBarConverter.{self.symbol}")
    logger.setLevel(logging.DEBUG)  # ERRORもキャッチ
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger
```

##### 4. 既存テストの有効化
- **test_bid_ask_spread_tracking**:
  - avg_spreadプロパティの累積平均計算を検証
  - 複数ティックでのスプレッド追跡を確認
  - skipマーク削除

- **test_bar_completion_callback**:
  - バー完成時のコールバック実行を検証
  - on_bar_completeが正しく呼ばれることを確認
  - skipマーク削除

##### 5. 新規テストケースの追加
```python
def test_invalid_price_handling():
    """負の価格やゼロ価格の処理テスト"""
    converter = TickToBarConverter("EURUSD")
    
    # 負の価格
    with pytest.raises(ValidationError):
        tick = Tick(
            symbol="EURUSD",
            time=datetime(2025, 8, 21, 12, 0, 0),
            bid=Decimal("-1.1234"),
            ask=Decimal("1.1236"),
            volume=Decimal("1.0")
        )
    
    # ゼロ価格
    with pytest.raises(ValidationError):
        tick = Tick(
            symbol="EURUSD",
            time=datetime(2025, 8, 21, 12, 0, 0),
            bid=Decimal("0"),
            ask=Decimal("1.1236"),
            volume=Decimal("1.0")
        )

def test_timestamp_reversal_handling(caplog):
    """タイムスタンプ逆転の処理テスト"""
    converter = TickToBarConverter("EURUSD")
    
    # 正常なティック
    tick1 = create_test_tick(datetime(2025, 8, 21, 12, 0, 30))
    converter.add_tick(tick1)
    
    # 過去のタイムスタンプのティック
    tick2 = create_test_tick(datetime(2025, 8, 21, 12, 0, 10))
    result = converter.add_tick(tick2)
    
    assert result is None  # ティックが破棄される
    assert "timestamp_reversal_detected" in caplog.text
    assert converter.current_bar.tick_count == 1  # カウントが増えない

def test_invalid_spread_handling():
    """異常スプレッド（ask < bid）の処理テスト"""
    with pytest.raises(ValidationError) as exc_info:
        tick = Tick(
            symbol="EURUSD",
            time=datetime(2025, 8, 21, 12, 0, 0),
            bid=Decimal("1.1236"),
            ask=Decimal("1.1234"),  # bid > ask
            volume=Decimal("1.0")
        )
    assert "Ask" in str(exc_info.value)
    assert "cannot be less than bid" in str(exc_info.value)

def test_zero_volume_handling():
    """ゼロボリュームの処理テスト"""
    # ゼロボリュームは許可される
    tick = Tick(
        symbol="EURUSD",
        time=datetime(2025, 8, 21, 12, 0, 0),
        bid=Decimal("1.1234"),
        ask=Decimal("1.1236"),
        volume=Decimal("0")  # ゼロは許可
    )
    assert tick.volume == Decimal("0")
    
    # 負のボリュームは拒否
    with pytest.raises(ValidationError):
        tick = Tick(
            symbol="EURUSD",
            time=datetime(2025, 8, 21, 12, 0, 0),
            bid=Decimal("1.1234"),
            ask=Decimal("1.1236"),
            volume=Decimal("-1.0")
        )
```

##### 6. 実装手順
1. Tickモデルにfield_validatorを追加
2. ロガー設定をDEBUGレベルに変更（ERRORもキャッチ）
3. add_tick()にtry-except追加とタイムスタンプ逆転検知
4. pydantic.ValidationErrorのインポート追加
5. test_bid_ask_spread_trackingのskipマーク削除
6. test_bar_completion_callbackのskipマーク削除
7. 新規テストケース4つを追加
8. テスト実行（15 tests passed目標）

##### 技術的考慮事項
- ValidationErrorは自動的に発生（Pydanticの機能）
- エラー時もアプリケーションは継続（非破壊的）
- 構造化ログでデバッグしやすい
- タイムスタンプ逆転は実際のMT5データで発生する可能性あり

#### Step 7 実装詳細（現在作業中）

##### 実装対象
統合テストとパフォーマンステストを作成し、コードの最適化とドキュメント改善を行います。

##### 1. 統合テストファイルの構成
```python
# tests/integration/test_tick_to_bar_integration.py

import time
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from src.mt5_data_acquisition.tick_to_bar import TickToBarConverter, Tick, Bar

class TestTickToBarIntegration:
    """統合テストクラス"""
    
    def test_continuous_tick_stream(self):
        """連続的なティックストリーム処理"""
        # 5分間の連続ティックを生成（1秒ごと）
        # 5つのバーが生成されることを確認
        
    def test_market_gap_handling(self):
        """市場時間外のギャップ処理（週末シミュレーション）"""
        # 金曜日の最終ティック→月曜日の最初のティック
        # 適切に警告が出力され、処理が継続することを確認
        
    def test_high_frequency_ticks(self):
        """高頻度ティック処理（1秒間に10ティック）"""
        # 同一秒内での複数ティック処理
        # OHLCVが正しく更新されることを確認
        
    def test_callback_chain(self):
        """バー完成通知の連鎖処理"""
        # バー完成時のコールバックが連続して呼ばれることを確認
        # 複数のリスナーへの通知をシミュレート
```

##### 2. パフォーマンステストの実装
```python
class TestPerformance:
    """パフォーマンステストクラス"""
    
    def test_large_tick_volume(self):
        """大量ティック処理のパフォーマンス"""
        converter = TickToBarConverter("EURUSD")
        start_time = datetime(2025, 8, 21, 9, 0, 0)
        
        # 10,000ティックを生成（約2.7時間分）
        ticks = []
        current_time = start_time
        for i in range(10000):
            tick = Tick(
                symbol="EURUSD",
                time=current_time,
                bid=Decimal("1.1234") + Decimal(str(i % 100)) / Decimal("10000"),
                ask=Decimal("1.1236") + Decimal(str(i % 100)) / Decimal("10000"),
                volume=Decimal("1.0")
            )
            ticks.append(tick)
            current_time += timedelta(seconds=1)
        
        # 処理時間測定
        start = time.perf_counter()
        for tick in ticks:
            converter.add_tick(tick)
        elapsed = time.perf_counter() - start
        
        # アサーション
        assert elapsed < 1.0  # 1秒以内に処理完了
        assert len(converter.completed_bars) == 166  # 166バー生成（10000秒 / 60）
        print(f"Processed 10,000 ticks in {elapsed:.3f} seconds")
        print(f"Throughput: {10000 / elapsed:.0f} ticks/second")
    
    def test_memory_usage(self):
        """メモリ使用量の測定"""
        import tracemalloc
        
        tracemalloc.start()
        converter = TickToBarConverter("EURUSD")
        
        # 1時間分のティックデータ（3600ティック）
        start_time = datetime(2025, 8, 21, 9, 0, 0)
        for i in range(3600):
            tick = Tick(
                symbol="EURUSD",
                time=start_time + timedelta(seconds=i),
                bid=Decimal("1.1234"),
                ask=Decimal("1.1236"),
                volume=Decimal("1.0")
            )
            converter.add_tick(tick)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # メモリ使用量をMBで表示
        print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
        print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
        
        # 10MB以下であることを確認
        assert peak / 1024 / 1024 < 10
```

##### 3. コード最適化の検討
```python
# src/mt5_data_acquisition/tick_to_bar.py への改善案

class TickToBarConverter:
    def __init__(self, symbol: str, timeframe: int = 60, 
                 on_bar_complete: Callable[[Bar], None] | None = None,
                 max_completed_bars: int = 1000):  # メモリ管理用
        """
        Args:
            max_completed_bars: 保持する完成バーの最大数（メモリ管理）
        """
        self.max_completed_bars = max_completed_bars
        # ... 既存の初期化 ...
    
    def add_tick(self, tick: Tick) -> Bar | None:
        """改善版: メモリ管理を追加"""
        # ... 既存の処理 ...
        
        # メモリ管理: 古いバーを削除
        if len(self.completed_bars) > self.max_completed_bars:
            # FIFOで古いバーを削除
            self.completed_bars = self.completed_bars[-self.max_completed_bars:]
        
        return completed_bar
    
    def clear_completed_bars(self) -> list[Bar]:
        """完成したバーをクリアして返す（メモリ解放用）"""
        bars = self.completed_bars.copy()
        self.completed_bars.clear()
        return bars
```

##### 4. ドキュメント改善案
```python
class TickToBarConverter:
    """ティックデータを時系列バー（OHLCV）に変換するコンバーター
    
    このクラスはリアルタイムのティックデータを受け取り、
    指定された時間枠（デフォルト1分）のOHLCVバーに変換します。
    
    Features:
        - リアルタイムティック処理
        - 未完成バーの継続更新
        - ティック欠損検知（30秒以上）
        - エラーハンドリング（無効データ、タイムスタンプ逆転）
        - バー完成時のコールバック通知
    
    Example:
        >>> converter = TickToBarConverter("EURUSD")
        >>> converter.on_bar_complete = lambda bar: print(f"Bar completed: {bar}")
        >>> 
        >>> tick = Tick(
        ...     symbol="EURUSD",
        ...     time=datetime.now(),
        ...     bid=Decimal("1.1234"),
        ...     ask=Decimal("1.1236"),
        ...     volume=Decimal("1.0")
        ... )
        >>> completed_bar = converter.add_tick(tick)
        >>> 
        >>> # 未完成バーの取得
        >>> current_bar = converter.get_current_bar()
        >>> if current_bar:
        ...     print(f"Current bar: {current_bar}")
    
    Attributes:
        symbol: 処理対象の通貨ペア
        timeframe: バーの時間枠（秒単位）
        current_bar: 現在作成中のバー
        completed_bars: 完成したバーのリスト
        on_bar_complete: バー完成時のコールバック関数
        last_tick_time: 最後に受信したティックの時刻
        gap_threshold: ティック欠損警告の閾値（秒）
    
    Note:
        - ティックは時間順に処理される必要があります
        - タイムスタンプが逆転したティックは破棄されます
        - 無効な価格データ（負値、ゼロ）は拒否されます
    """
```

##### 5. 最終確認項目
- [ ] 全テスト（unit + integration）がPASSED
- [ ] Ruffチェックでエラーなし
- [ ] 型ヒントの完全性（mypy互換）
- [ ] パフォーマンス目標達成（10,000ティック/秒以上）
- [ ] メモリ使用量が適切（10MB以下）
- [ ] ドキュメントが十分

##### 6. 実装手順
1. tests/integration/ディレクトリの確認・作成
2. test_tick_to_bar_integration.pyファイルの作成
3. 基本的な統合テストの実装
4. パフォーマンステストの実装
5. メモリプロファイリングテストの実装
6. 必要に応じてコード最適化
7. ドキュメントストリングの改善
8. 最終的なRuffチェック
9. 全テスト実行と確認
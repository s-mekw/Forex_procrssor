# Tickモデル統一リファクタリング計画

## 📋 概要
プロジェクト内に2つの異なるTickモデルが存在し、統一が必要です。このドキュメントでは、段階的な移行計画と実装手順を定義します。

## 🔍 現状分析

### 1. 既存のTickモデル

#### A. src/common/models.py のTickモデル（プロジェクト標準）
```python
class Tick(BaseModel):
    timestamp: datetime  # 時刻属性名が「timestamp」
    symbol: str
    bid: float          # Float32制約付き
    ask: float          # Float32制約付き  
    volume: float       # Float32制約付き
```

**特徴:**
- プロジェクト全体の共通モデルとして設計
- Float32制約によるメモリ効率最適化
- Pydanticのバリデーション機能付き
- 15箇所のファイルで使用中

#### B. src/mt5_data_acquisition/tick_to_bar.py のTickモデル（ローカル実装）
```python
class Tick(BaseModel):
    time: datetime      # 時刻属性名が「time」
    symbol: str
    bid: Decimal        # Decimal型（高精度）
    ask: Decimal        # Decimal型（高精度）
    volume: Decimal     # Decimal型（高精度）
```

**特徴:**
- TickToBarConverterクラス専用
- Decimal型による高精度計算
- 13箇所のテストファイルで使用中

### 2. 影響範囲分析

#### 直接影響を受けるファイル
- **TickToBarConverter使用箇所:** 16ファイル
  - src/mt5_data_acquisition/tick_to_bar.py
  - tests/unit/test_tick_to_bar.py
  - tests/integration/test_tick_to_bar_integration.py

- **common/models.Tick使用箇所:** 15ファイル
  - src/mt5_data_acquisition/tick_fetcher.py
  - tests/unit/test_tick_fetcher.py
  - tests/integration/test_tick_streaming.py

## 🎯 移行目標

### 最終状態
1. **統一されたTickモデル**: src/common/models.pyのTickを全体で使用
2. **属性名の統一**: `timestamp`を標準属性名として使用
3. **型の統一**: Float32制約を維持（プロジェクト標準）
4. **後方互換性**: 既存コードへの影響を最小化

### 制約事項
- Float32制約の維持（プロジェクト全体の要件）
- Polarsの使用（Pandas禁止）
- Pydanticベースのモデル
- テストカバレッジ80%以上の維持

## 📊 移行戦略

### アプローチ: アダプターパターンによる段階的移行

1. **Phase 1**: 互換性レイヤーの追加（リスク: 低）
2. **Phase 2**: TickToBarConverterの内部実装を更新（リスク: 中）
3. **Phase 3**: テストコードの移行（リスク: 低）
4. **Phase 4**: 古いモデルの削除（リスク: 低）

## 🔄 実装ステップ

### Step 1: 互換性プロパティの追加（作業時間: 30分）
**ファイル:** src/common/models.py
**作業内容:**

#### 1.1 実装内容
```python
class Tick(BaseModel):
    timestamp: datetime = Field(...)
    # ... 既存のフィールド ...
    
    @property
    def time(self) -> datetime:
        """後方互換性のためのプロパティ
        
        TickToBarConverterとの互換性を保つため、
        time属性でtimestampにアクセスできるようにする。
        
        Note:
            このプロパティはStep 6で削除予定。
            新規コードではtimestamp属性を使用すること。
        """
        return self.timestamp
    
    @time.setter
    def time(self, value: datetime):
        """後方互換性のためのセッター
        
        Args:
            value: 設定する時刻値
        """
        self.timestamp = value
```

#### 1.2 実装箇所
- **挿入位置**: to_float32_dict()メソッドの後（108行目以降）
- **インポート不要**: datetimeは既にインポート済み

#### 1.3 テスト計画
1. **単体テスト作成**（tests/unit/test_tick_model_compatibility.py）
   - timeプロパティの読み取りテスト
   - timeプロパティの書き込みテスト
   - timestamp属性との同期確認

2. **既存テストの確認**
   - tests/unit/test_models.py が通ることを確認
   - 既存のTickモデル使用箇所に影響がないことを確認

#### 1.4 検証項目
- [ ] timeプロパティでtimestampの値が取得できる
- [ ] time = valueでtimestamp属性が更新される
- [ ] 既存のコードが破壊されない
- [ ] docstringで非推奨であることを明記

### Step 2: TickToBarConverterアダプターの作成（作業時間: 1時間）
**ファイル:** src/mt5_data_acquisition/tick_adapter.py（新規）
**作業内容:**

#### 2.0 実装準備チェックリスト
- [x] Step 1が完了していることを確認
- [x] CommonTickにtimeプロパティが実装されている
- [ ] tick_adapter.pyの作成場所を確認（mt5_data_acquisitionディレクトリ）
- [ ] テストファイルの作成場所を確認（tests/unitディレクトリ）

#### 2.1 実装内容
```python
"""
Tickモデル間の変換アダプター

common.models.TickとTickToBarConverterの間の
データ形式の違いを吸収するアダプター。
"""

from decimal import Decimal
from typing import Union
import numpy as np

from src.common.models import Tick as CommonTick


class TickAdapter:
    """common.models.TickをTickToBarConverterで使用できるように変換
    
    このアダプターは、Float32制約のあるCommonTickと
    Decimal精度を必要とするTickToBarConverterの間で
    データ変換を行います。
    """
    
    @staticmethod
    def to_decimal_dict(tick: CommonTick) -> dict:
        """CommonTickをDecimal形式の辞書に変換
        
        Args:
            tick: 変換元のCommonTickインスタンス
            
        Returns:
            Decimal型の価格データを含む辞書
        """
        return {
            'symbol': tick.symbol,
            'time': tick.timestamp,  # timestamp -> time
            'bid': Decimal(str(tick.bid)),
            'ask': Decimal(str(tick.ask)),
            'volume': Decimal(str(tick.volume))
        }
    
    @staticmethod
    def from_decimal_dict(tick_dict: dict) -> CommonTick:
        """Decimal形式の辞書からCommonTickに変換
        
        Args:
            tick_dict: Decimal型の価格データを含む辞書
            
        Returns:
            変換されたCommonTickインスタンス
        """
        # timeとtimestamp両方に対応
        timestamp = tick_dict.get('time', tick_dict.get('timestamp'))
        if timestamp is None:
            raise ValueError("time or timestamp key is required")
            
        return CommonTick(
            timestamp=timestamp,
            symbol=tick_dict['symbol'],
            bid=float(tick_dict['bid']),
            ask=float(tick_dict['ask']),
            volume=float(tick_dict.get('volume', 0.0))
        )
    
    @staticmethod
    def ensure_decimal_precision(value: Union[float, Decimal]) -> Decimal:
        """値をDecimal型に安全に変換
        
        Args:
            value: 変換する値
            
        Returns:
            Decimal型の値
        """
        if isinstance(value, Decimal):
            return value
        # Float32精度の値を文字列経由でDecimalに変換
        return Decimal(str(np.float32(value)))
```

#### 2.2 テスト計画（詳細）
1. **単体テスト作成**（tests/unit/test_tick_adapter.py）
   
   **基本変換テスト（3ケース）**
   - `test_to_decimal_dict_basic`: CommonTick → Decimal辞書の標準変換
   - `test_from_decimal_dict_basic`: Decimal辞書 → CommonTickの標準変換
   - `test_round_trip_conversion`: 往復変換で元のデータが保持されるか
   
   **属性名互換性テスト（3ケース）**
   - `test_from_dict_with_time_key`: timeキーでの辞書からの変換
   - `test_from_dict_with_timestamp_key`: timestampキーでの辞書からの変換
   - `test_missing_time_raises_error`: time/timestampキー不在時のValueError
   
   **精度テスト（3ケース）**
   - `test_float32_precision_maintained`: Float32制約下での精度維持
   - `test_decimal_precision_conversion`: Decimal変換時の精度保持
   - `test_ensure_decimal_precision_helper`: ヘルパーメソッドの動作確認
   
   **エッジケーステスト（3ケース）**
   - `test_zero_values`: ゼロ値（0.0）の正確な処理
   - `test_large_numbers`: 大きな数値（1e6以上）の処理
   - `test_small_decimals`: 小数点以下多桁（1e-6以下）の処理

2. **統合テスト**
   - CommonTickからDecimalへの往復変換テスト
   - 1000回の連続変換での精度損失測定
   - メモリリークのチェック

#### 2.3 検証項目
- [ ] CommonTickからDecimal辞書への変換が正しい
- [ ] Decimal辞書からCommonTickへの変換が正しい
- [ ] timestamp/time両方の属性名に対応
- [ ] 数値精度が保持される

### Step 3: TickToBarConverterの更新（作業時間: 2時間）⚠️重要度: 高
**ファイル:** src/mt5_data_acquisition/tick_to_bar.py
**作業内容:**
1. ローカルTickクラスを削除
2. common.models.Tickをインポート
3. 内部実装をFloat32/Decimalハイブリッドに更新

#### 3.0 前提条件
- ✅ Step 1完了: common.models.Tickにtimeプロパティ実装済み
- ✅ Step 2完了: TickAdapterクラスが利用可能
- 📊 影響範囲: 16ファイルで使用されているコアコンポーネント

#### 3.1 実装詳細

##### A. インポート変更
```python
# 削除するインポート
# from pydantic import BaseModel, Field, ValidationError, field_validator

# 追加するインポート
from src.common.models import Tick
from src.mt5_data_acquisition.tick_adapter import TickAdapter
from pydantic import ValidationError  # ValidationErrorは残す
```

##### B. ローカルTickクラスの削除（17-49行目）
```python
# 以下のクラス定義を完全に削除
# class Tick(BaseModel):
#     """ティックデータのモデル"""
#     ...（省略）...
```

##### C. add_tick()メソッドの修正（144-220行目）
```python
def add_tick(self, tick: Tick) -> Bar | None:
    """
    ティックを追加し、バーが完成した場合はそれを返す
    
    Args:
        tick: 追加するティックデータ（common.models.Tick）
    
    Returns:
        完成したバー（完成していない場合はNone）
    """
    try:
        # タイムスタンプ逆転チェック（tick.time → tick.timestamp）
        if self.last_tick_time and tick.timestamp < self.last_tick_time:
            error_data = {
                "event": "timestamp_reversal",
                "symbol": self.symbol,
                "current_time": tick.timestamp.isoformat(),
                "last_tick_time": self.last_tick_time.isoformat()
            }
            self.logger.error(json.dumps(error_data))
            return None
        
        # ティック欠損を検知
        self.check_tick_gap(tick.timestamp)
        
        # 現在のバーがない場合は新規作成
        if self.current_bar is None:
            self._create_new_bar(tick)
            self.last_tick_time = tick.timestamp
            return None
        
        # バー完成判定
        if self._check_bar_completion(tick.timestamp):
            # ... 既存のロジック ...
            self.last_tick_time = tick.timestamp
            return completed_bar
        else:
            # 現在のバーを更新
            self._update_bar(tick)
            self.last_tick_time = tick.timestamp
            return None
            
    except ValidationError as e:
        # エラーハンドリング（tick.timestamp使用）
        error_data = {
            "event": "invalid_tick_data",
            "symbol": self.symbol,
            "error": str(e),
            "tick_data": {
                "time": tick.timestamp.isoformat() if tick.timestamp else None,
                "bid": str(tick.bid) if tick.bid else None,
                "ask": str(tick.ask) if tick.ask else None,
                "volume": str(tick.volume) if tick.volume else None
            }
        }
        self.logger.error(json.dumps(error_data))
        return None
```

##### D. _create_new_bar()メソッドの修正（317-346行目）
```python
def _create_new_bar(self, tick: Tick) -> None:
    """
    新しいバーを作成（プライベートメソッド）
    
    Args:
        tick: バーの最初のティック（common.models.Tick）
    """
    # TickAdapterを使用してDecimal形式に変換
    tick_decimal = TickAdapter.to_decimal_dict(tick)
    
    # ティックの時刻を分単位に正規化
    bar_start = self._get_bar_start_time(tick.timestamp)
    bar_end = self._get_bar_end_time(bar_start)
    
    # OHLC値を最初のティックで初期化（Decimal精度維持）
    self.current_bar = Bar(
        symbol=self.symbol,
        time=bar_start,
        end_time=bar_end,
        open=tick_decimal['bid'],
        high=tick_decimal['bid'],
        low=tick_decimal['bid'],
        close=tick_decimal['bid'],
        volume=tick_decimal['volume'],
        tick_count=1,
        avg_spread=tick_decimal['ask'] - tick_decimal['bid'],
        is_complete=False,
    )
    
    # ティックリストをクリアして新しいティックを追加
    self._current_ticks = [tick]
```

##### E. _update_bar()メソッドの修正（348-383行目）
```python
def _update_bar(self, tick: Tick) -> None:
    """
    現在のバーを更新（プライベートメソッド）
    
    Args:
        tick: バーに追加するティック（common.models.Tick）
    """
    if not self.current_bar:
        return
    
    # TickAdapterを使用してDecimal形式に変換
    tick_decimal = TickAdapter.to_decimal_dict(tick)
    
    # High/Lowの更新（Decimal精度で比較）
    self.current_bar.high = max(self.current_bar.high, tick_decimal['bid'])
    self.current_bar.low = min(self.current_bar.low, tick_decimal['bid'])
    
    # Closeを最新のティック価格に
    self.current_bar.close = tick_decimal['bid']
    
    # ボリューム累積
    self.current_bar.volume += tick_decimal['volume']
    
    # ティックカウント増加
    self.current_bar.tick_count += 1
    
    # スプレッドの累積平均計算
    if self.current_bar.avg_spread is not None:
        total_spread = self.current_bar.avg_spread * (
            self.current_bar.tick_count - 1
        )
        new_spread = tick_decimal['ask'] - tick_decimal['bid']
        self.current_bar.avg_spread = (
            total_spread + new_spread
        ) / self.current_bar.tick_count
    
    # ティックリストに追加
    self._current_ticks.append(tick)
```

##### F. その他のメソッドの修正
```python
def _check_bar_completion(self, tick_time: datetime) -> bool:
    # tick_time引数の型はdatetimeのままでOK（tick.timestampを渡す）
    
def _get_bar_start_time(self, tick_time: datetime) -> datetime:
    # tick_time引数の型はdatetimeのままでOK（tick.timestampを渡す）
```

#### 3.2 テスト対応（簡易修正）

Step 3では最小限の修正でテストを通すことを目標とする（詳細はStep 4で実施）:

```python
# tests/unit/test_tick_to_bar.py の修正例
from src.common.models import Tick  # インポート変更

# テストデータの修正
{
    "symbol": "EURUSD",
    "timestamp": base_time + timedelta(seconds=0),  # time → timestamp
    "bid": 1.04200,  # Decimal → float
    "ask": 1.04210,  # Decimal → float
    "volume": 1.0,   # Decimal → float
}
```

### Step 4: テストコードの移行（作業時間: 2時間）
**対象ファイル:** 
- tests/unit/test_tick_to_bar.py
- tests/integration/test_tick_to_bar_integration.py

**作業内容:**
1. Tick生成部分を共通モデルに変更
2. `time`から`timestamp`への属性名変更
3. Decimal型からfloat型への変更

```python
# Before
from src.mt5_data_acquisition.tick_to_bar import Tick
tick = Tick(
    symbol="EURUSD",
    time=datetime.now(),
    bid=Decimal("1.1234"),
    ask=Decimal("1.1236"),
    volume=Decimal("1.0")
)

# After
from src.common.models import Tick
tick = Tick(
    symbol="EURUSD",
    timestamp=datetime.now(),
    bid=1.1234,
    ask=1.1236,
    volume=1.0
)
```

### Step 5: 性能最適化（作業時間: 1時間）
**ファイル:** src/mt5_data_acquisition/tick_to_bar.py
**作業内容:**
- Decimal計算をnp.float32に置き換え（精度要件を確認後）
- メモリプロファイリングの実施
- パフォーマンステストの追加

### Step 6: 後方互換性プロパティの削除（作業時間: 30分）
**ファイル:** src/common/models.py
**作業内容:**
- timeプロパティを削除（全てのコードが移行完了後）
- deprecation warningの削除

## 📝 各ステップのチェックリスト

### Step 1 チェックリスト
- [ ] timeプロパティを追加
- [ ] 既存テストが全て通る
- [ ] 新規プロパティテストを追加

### Step 2 チェックリスト
**実装タスク**
- [ ] tick_adapter.pyファイルを新規作成
- [ ] TickAdapterクラスの基本構造を実装
- [ ] to_decimal_dict()メソッドを実装
- [ ] from_decimal_dict()メソッドを実装
- [ ] ensure_decimal_precision()ヘルパーを実装

**テストタスク**
- [ ] test_tick_adapter.pyファイルを新規作成
- [ ] 基本変換テスト（3ケース）を実装
- [ ] 属性名互換性テスト（3ケース）を実装
- [ ] 精度テスト（3ケース）を実装
- [ ] エッジケーステスト（3ケース）を実装

**検証タスク**
- [ ] 全12個のテストケースが成功
- [ ] 既存のテストに影響がない
- [ ] パフォーマンスへの影響が5%以内
- [ ] メモリ使用量の増加が10%以内

### Step 3 チェックリスト

**実装タスク**
- [ ] ローカルTickクラスをコメントアウト（一時的）
- [ ] インポート文の変更
  - [ ] common.models.Tickをインポート
  - [ ] TickAdapterをインポート
  - [ ] 不要なインポートを削除
- [ ] add_tick()メソッドの修正
  - [ ] tick.time → tick.timestampへの変更
  - [ ] エラーハンドリング部分の修正
- [ ] _create_new_bar()メソッドの修正
  - [ ] TickAdapter.to_decimal_dict()の使用
  - [ ] tick.timestamp使用への変更
- [ ] _update_bar()メソッドの修正
  - [ ] TickAdapter.to_decimal_dict()の使用
  - [ ] Decimal計算の維持
- [ ] check_tick_gap()メソッドの引数確認
- [ ] ローカルTickクラスを完全削除

**テストタスク**
- [ ] test_tick_to_bar.pyの最小限修正
  - [ ] インポートの変更
  - [ ] time → timestampの変更
  - [ ] Decimal → floatの変更
- [ ] ユニットテストの実行
- [ ] エラー箇所の特定と修正

**検証タスク**
- [ ] 基本的な変換動作の確認
- [ ] Decimal精度が維持されているか
- [ ] 既存の16ファイルへの影響確認
- [ ] パフォーマンス測定（変換オーバーヘッド）

### Step 4 チェックリスト
- [ ] 全テストファイルでcommon.models.Tickを使用
- [ ] timestamp属性名に統一
- [ ] float型への変更
- [ ] テストカバレッジ80%以上を維持

### Step 5 チェックリスト
- [ ] Decimal計算の必要性を評価
- [ ] Float32での精度テスト実施
- [ ] パフォーマンス比較テスト作成
- [ ] メモリ使用量の測定

### Step 6 チェックリスト
- [ ] 全コードでtimestamp使用を確認
- [ ] timeプロパティを削除
- [ ] 最終的な統合テスト実施

## 🚨 リスクと対策

### リスク1: 精度の低下
**問題:** DecimalからFloat32への変換で精度が低下する可能性
**対策:** 
- 金融計算の精度要件を確認
- 必要に応じて内部計算のみDecimalを使用
- 精度テストケースを追加

### リスク2: 既存コードの破壊
**問題:** 属性名変更により既存コードが動作しなくなる
**対策:**
- 後方互換性プロパティの提供
- 段階的な移行
- 各ステップでの回帰テスト実施

### リスク3: パフォーマンス劣化
**問題:** 型変換処理によるオーバーヘッド
**対策:**
- パフォーマンステストの実施
- ボトルネックの特定と最適化
- 必要に応じてキャッシュ機構の導入

## 🔄 ロールバック計画

各ステップは独立しており、問題が発生した場合は以下の手順でロールバック可能:

1. **Step 1のロールバック:** timeプロパティを削除
2. **Step 2のロールバック:** TickAdapterを削除
3. **Step 3のロールバック:** tick_to_bar.pyを以前のバージョンに戻す
4. **Step 4のロールバック:** テストコードを以前のバージョンに戻す
5. **Step 5のロールバック:** 最適化前のコードに戻す

## 📈 成功指標

- [ ] 全てのテストが通る（テストカバレッジ80%以上）
- [ ] パフォーマンスの劣化が5%以内
- [ ] メモリ使用量の増加が10%以内
- [ ] 統一されたTickモデルの使用
- [ ] コードの重複削除

## 🗓️ タイムライン

**推定作業時間:** 合計 7時間

1. **Day 1 (2時間)**
   - Step 1: 互換性プロパティの追加（30分）
   - Step 2: TickAdapterの作成（1時間）
   - テスト実行と確認（30分）

2. **Day 2 (3時間)**
   - Step 3: TickToBarConverterの更新（2時間）
   - Step 4の一部: 主要テストコードの移行（1時間）

3. **Day 3 (2時間)**
   - Step 4の完了: 残りのテストコード移行（1時間）
   - Step 5: 性能最適化（1時間）

4. **Day 4 (30分)**
   - Step 6: クリーンアップ（30分）
   - 最終テストと文書化

## 📚 参考資料

- [Pydantic Migration Guide](https://docs.pydantic.dev/latest/migration/)
- [NumPy Float32 Documentation](https://numpy.org/doc/stable/user/basics.types.html)
- [Python Decimal Module](https://docs.python.org/3/library/decimal.html)

## ✅ 承認とレビュー

このリファクタリング計画は以下のステークホルダーによるレビューが必要です:

- [ ] 技術リード
- [ ] QAチーム
- [ ] プロダクトオーナー

---

*最終更新: 2025-08-21*
*作成者: Claude AI Assistant*
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

#### 2.2 テスト計画
1. **単体テスト作成**（tests/unit/test_tick_adapter.py）
   - to_decimal_dict()の変換テスト
   - from_decimal_dict()の変換テスト
   - 精度保持の確認テスト
   - エラーケースのテスト

2. **統合テスト**
   - CommonTickからDecimalへの往復変換テスト
   - 精度損失がないことの確認

#### 2.3 検証項目
- [ ] CommonTickからDecimal辞書への変換が正しい
- [ ] Decimal辞書からCommonTickへの変換が正しい
- [ ] timestamp/time両方の属性名に対応
- [ ] 数値精度が保持される

### Step 3: TickToBarConverterの更新（作業時間: 2時間）
**ファイル:** src/mt5_data_acquisition/tick_to_bar.py
**作業内容:**
1. ローカルTickクラスを削除
2. common.models.Tickをインポート
3. 内部実装をFloat32/Decimalハイブリッドに更新

```python
from src.common.models import Tick
from decimal import Decimal
import numpy as np

class TickToBarConverter:
    def add_tick(self, tick: Tick) -> Bar | None:
        # 内部計算用にDecimalに変換（精度保持）
        bid_decimal = Decimal(str(tick.bid))
        ask_decimal = Decimal(str(tick.ask))
        volume_decimal = Decimal(str(tick.volume))
        
        # timestamp属性を使用（timeプロパティは後方互換性用）
        tick_time = tick.timestamp
        
        # 既存のロジックを維持...
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
- [ ] TickAdapterクラスを作成
- [ ] 変換メソッドのユニットテスト作成
- [ ] 双方向変換の動作確認

### Step 3 チェックリスト
- [ ] ローカルTickクラスを削除
- [ ] 共通Tickモデルをインポート
- [ ] 全メソッドでtimestamp属性を使用
- [ ] 既存テストが通る

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
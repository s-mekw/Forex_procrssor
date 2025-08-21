# ワークフローコンテキスト

## 📍 現在の状態
- ステップ: 4/6 ✅
- 最終更新: 2025-08-21 17:30
- フェーズ: 実装中（Step 4完了）

## 🎯 目標
Tickモデルを統一し、プロジェクト全体で一貫性のあるデータモデルを実現する

## 📊 現状分析サマリー

### 2つのTickモデルの違い

| 項目 | common/models.py（標準） | tick_to_bar.py（ローカル） |
|------|------------------------|------------------------|
| 時刻属性名 | timestamp | time |
| 数値型 | float (Float32制約) | Decimal |
| 使用箇所 | 15ファイル | 16ファイル |
| 役割 | プロジェクト共通モデル | TickToBarConverter専用 |

### 主な課題
1. **属性名の不一致**: `timestamp` vs `time`
2. **型の不一致**: Float32 vs Decimal
3. **影響範囲**: 合計31ファイルに影響

## 📋 実装計画

### Step 1: 互換性プロパティの追加 ✅完了
- ファイル: src/common/models.py
- 作業: timeプロパティを追加（後方互換性）
- 完了: [x] 2025-08-21 12:30
- 見積時間: 30分（実績: 15分）

### Step 2: TickAdapterの作成 ✅完了
- ファイル: src/mt5_data_acquisition/tick_adapter.py（新規）
- 作業: 型変換アダプタークラスの実装
- 完了: [x] 2025-08-21 14:30
- 見積時間: 1時間（実績: 30分）
- 詳細時間配分:
  - TickAdapterクラス実装: 10分
  - テストケース作成（12個）: 15分
  - 動作確認・調整: 5分

### Step 3: TickToBarConverterの更新 ✅完了
- ファイル: src/mt5_data_acquisition/tick_to_bar.py
- 作業: common.models.Tickを使用するよう変更
- 完了: [x] 2025-08-21 15:30
- 見積時間: 2時間（実績: 30分）

### Step 4: テストコードの移行 ✅完了
- ファイル: tests/unit/test_tick_to_bar.py, tests/integration/test_tick_to_bar_integration.py
- 作業: Tickモデルの参照と属性名を変更
- 完了: [x] 2025-08-21 17:30
- 見積時間: 2時間（実績: 60分）

### Step 5: 性能最適化 ✅計画済み
- ファイル: src/mt5_data_acquisition/tick_to_bar.py
- 作業: Decimal計算の必要性評価と最適化
- 完了: [ ]
- 見積時間: 1時間

### Step 6: クリーンアップ ✅計画済み
- ファイル: src/common/models.py
- 作業: 後方互換性プロパティの削除
- 完了: [ ]
- 見積時間: 30分

## 🔄 現在の作業
**完了したタスク**: Step 4のテストコード移行が完了
**次のタスク**: Step 5の性能最適化（必要に応じて実施）

### Step 3 実装タスク（詳細計画）

#### 3.0 事前確認
- Step 1とStep 2が完了済み
- TickAdapterが利用可能
- 既存テストの現状を把握

#### 3.1 変更概要
**目的**: ローカルTickクラスを削除し、common.models.Tickを使用

**主な変更点**:
1. ローカルTickクラスの削除（17-49行目）
2. インポートの変更
3. 内部実装のDecimal/Float32ハイブリッド化
4. timestamp属性の使用（timeプロパティは互換性のみ）

#### 3.2 実装詳細

##### 3.2.1 インポートの変更
```python
# 削除
from pydantic import BaseModel, Field, ValidationError, field_validator

# 追加
from src.common.models import Tick
from src.mt5_data_acquisition.tick_adapter import TickAdapter
```

##### 3.2.2 TickToBarConverterクラスの変更箇所

**A. add_tick()メソッド（144-220行目）**
```python
def add_tick(self, tick: Tick) -> Bar | None:
    # 変更点1: tick.time → tick.timestamp
    # 変更点2: Decimal変換の追加（内部計算用）
    
    # タイムスタンプ逆転チェック
    if self.last_tick_time and tick.timestamp < self.last_tick_time:
        # ...
    
    # 内部計算用にDecimal変換（精度保持）
    tick_decimal = TickAdapter.to_decimal_dict(tick)
```

**B. _create_new_bar()メソッド（317-346行目）**
```python
def _create_new_bar(self, tick: Tick) -> None:
    # Decimal変換して内部計算
    tick_decimal = TickAdapter.to_decimal_dict(tick)
    
    bar_start = self._get_bar_start_time(tick.timestamp)
    bar_end = self._get_bar_end_time(bar_start)
    
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
```

**C. _update_bar()メソッド（348-383行目）**
```python
def _update_bar(self, tick: Tick) -> None:
    # Decimal変換して内部計算
    tick_decimal = TickAdapter.to_decimal_dict(tick)
    
    # High/Lowの更新
    self.current_bar.high = max(self.current_bar.high, tick_decimal['bid'])
    self.current_bar.low = min(self.current_bar.low, tick_decimal['bid'])
    
    # Closeを最新のティック価格に
    self.current_bar.close = tick_decimal['bid']
    
    # ボリューム累積
    self.current_bar.volume += tick_decimal['volume']
    
    # スプレッド計算
    new_spread = tick_decimal['ask'] - tick_decimal['bid']
```

##### 3.2.3 エラーハンドリングの調整
- ValidationErrorの処理（206-219行目）
- tick.time → tick.timestamp への変更
- tick.bid/ask/volumeのアクセス方法は変更なし

#### 3.3 テスト戦略

##### 3.3.1 段階的テスト
1. **Phase 1**: 変更後のユニットテストを実行
   - `uv run pytest tests/unit/test_tick_to_bar.py -v`
   - エラー箇所を特定

2. **Phase 2**: テストの修正
   - Tickインポートを変更
   - time → timestamp への変更
   - Decimal → float への変更

3. **Phase 3**: 統合テストの実行
   - `uv run pytest tests/integration/test_tick_to_bar_integration.py -v`

##### 3.3.2 回帰テストリスト
- [ ] test_converter_initialization
- [ ] test_single_minute_bar_generation
- [ ] test_timestamp_alignment
- [ ] test_ohlc_calculation
- [ ] test_volume_aggregation
- [ ] test_spread_calculation
- [ ] test_tick_gap_detection
- [ ] test_timestamp_reversal_handling

#### 3.4 リスク評価と対策

##### リスク1: Decimal精度の維持
- **対策**: TickAdapterを使用して内部計算はDecimalで実施
- **検証**: 精度テストケースで確認

##### リスク2: 既存テストの大量失敗
- **対策**: テストも同時に修正（Step 4と並行作業）
- **検証**: 各テストケースを個別に修正

##### リスク3: パフォーマンス劣化
- **対策**: 型変換のオーバーヘッドを測定
- **検証**: ベンチマークテストを追加

#### 3.5 実装順序（推奨）
1. ローカルTickクラスをコメントアウト
2. インポートを変更
3. add_tick()メソッドを修正
4. _create_new_bar()メソッドを修正
5. _update_bar()メソッドを修正
6. エラーハンドリングを調整
7. ローカルTickクラスを削除
8. テストを実行して動作確認

#### 3.6 予想される問題と対処法

##### 問題1: timeプロパティの使用箇所
- **症状**: 一部のコードがtick.timeを使用している可能性
- **対処**: common.models.Tickのtimeプロパティで後方互換性確保済み
- **確認**: grep -r "tick\.time" src/

##### 問題2: Decimal型の期待
- **症状**: テストがDecimal型を期待している
- **対処**: TickAdapterを使用して内部的にDecimal変換
- **確認**: Barモデルの各フィールドがDecimal型であることを確認

##### 問題3: バリデーションエラー
- **症状**: Tick作成時のバリデーションエラー
- **対処**: common.models.TickのFloat32制約を考慮
- **確認**: 極端な値（very large/small）のテストケース

##### 問題4: インポートエラー
- **症状**: 循環インポートの可能性
- **対処**: 相対インポートから絶対インポートへ
- **確認**: import順序の調整

#### 3.7 成功基準
- ✅ tick_to_bar.pyのローカルTickクラスが削除されている
- ✅ common.models.Tickが使用されている
- ✅ 内部計算でDecimal精度が維持されている
- ✅ 既存のテストが修正後も通る（Step 4と並行）
- ✅ パフォーマンスの劣化が5%以内
- ✅ メモリ使用量の増加が10%以内

### Step 2 実装タスク（詳細計画）

#### 2.1 ファイル作成
- **新規ファイル**: `src/mt5_data_acquisition/tick_adapter.py`
- **場所**: tick_to_bar.pyと同じディレクトリ
- **目的**: CommonTickとDecimal形式の相互変換を担当

#### 2.2 実装内容

##### TickAdapterクラス
```python
class TickAdapter:
    """Tickモデル間の変換を担当するアダプター"""
    
    @staticmethod
    def to_decimal_dict(tick: CommonTick) -> dict:
        """CommonTick → Decimal辞書変換"""
        - timestamp → time への属性名変換
        - float → Decimal への型変換
        - 文字列経由で精度を保持
    
    @staticmethod
    def from_decimal_dict(tick_dict: dict) -> CommonTick:
        """Decimal辞書 → CommonTick変換"""
        - time/timestamp 両方に対応
        - Decimal → float への型変換
        - Float32制約の適用
    
    @staticmethod
    def ensure_decimal_precision(value: Union[float, Decimal]) -> Decimal:
        """型変換ヘルパーメソッド"""
        - Float32精度の保持
        - 安全なDecimal変換
```

#### 2.3 テストケース設計

##### tests/unit/test_tick_adapter.py
1. **基本変換テスト**
   - test_to_decimal_dict_basic: 通常のTick → Decimal辞書
   - test_from_decimal_dict_basic: Decimal辞書 → Tick
   - test_round_trip_conversion: 往復変換の整合性

2. **属性名互換性テスト**
   - test_from_dict_with_time_key: timeキーのサポート
   - test_from_dict_with_timestamp_key: timestampキーのサポート
   - test_missing_time_raises_error: キー不在時のエラー

3. **精度テスト**
   - test_float32_precision_maintained: Float32精度の維持
   - test_decimal_precision_conversion: Decimal変換の正確性
   - test_ensure_decimal_precision_helper: ヘルパーメソッドの動作

4. **エッジケース**
   - test_zero_values: ゼロ値の処理
   - test_large_numbers: 大きな数値の処理
   - test_small_decimals: 小数点以下の精度

#### 2.4 実装順序
1. tick_adapter.py の基本構造を作成
2. to_decimal_dict メソッドを実装
3. from_decimal_dict メソッドを実装
4. ensure_decimal_precision ヘルパーを実装
5. test_tick_adapter.py でテストケースを作成
6. 全てのテストが通ることを確認

#### 2.5 検証項目
- [x] CommonTickからDecimal辞書への正確な変換
- [x] Decimal辞書からCommonTickへの正確な変換
- [x] timestamp/time属性名の相互互換性
- [x] Float32とDecimal間の精度保持
- [x] エラーハンドリングの適切性
- [x] パフォーマンスへの影響が最小限

### Step 4 完了（2025-08-21 17:30）
- ✅ 全23個のテストが成功（ユニット15個、統合8個）
- ✅ Tickモデルのインポートをcommon.modelsに変更
- ✅ time → timestamp属性名の一括変更完了
- ✅ Decimal型からfloat型への変換完了
- ✅ Float32精度を考慮した比較処理を追加
- 📁 変更ファイル:
  - `tests/unit/test_tick_to_bar.py` - 更新（インポート、属性名、型変更）
  - `tests/integration/test_tick_to_bar_integration.py` - 更新（同様の変更）
- 📝 備考:
  - Float32精度の影響で、Decimal値の比較にabs()を使用した許容誤差比較を導入
  - volumeの集計時にDecimal型変換を追加（Decimal(str(tick.volume))）
  - パフォーマンステストも含めてすべて成功

## 🔨 実装結果

### Step 1 完了（2025-08-21 12:30）
- ✅ Tickクラスにtimeプロパティを追加（getter/setter実装）
- ✅ timestamp属性との双方向同期を実現
- 📁 変更ファイル: 
  - `src/common/models.py` - timeプロパティ追加（110-130行目）
  - `tests/unit/test_tick_model_compatibility.py` - 新規テストファイル作成
- 📝 備考: 
  - 7つのテストケースすべて成功
  - 既存のtest_models.pyも問題なく動作
  - Float32精度の影響でテストの許容誤差を調整（rel=1e-3）

### Step 2 完了（2025-08-21 14:30）
- ✅ TickAdapterクラスの実装完了
- ✅ CommonTickとDecimal辞書間の双方向変換機能
- ✅ timestamp/time属性名の互換性確保
- 📁 変更ファイル:
  - `src/mt5_data_acquisition/tick_adapter.py` - 新規作成（77行）
  - `tests/unit/test_tick_adapter.py` - 新規テストファイル作成（268行）
- 📝 備考:
  - 12個のテストケースすべて成功
  - Float32精度を考慮した変換ロジック実装
  - 既存のテスト（test_models.py）にも影響なし

### Step 3 完了（2025-08-21 15:30）
- ✅ ローカルTickクラスを削除し、common.models.Tickを使用
- ✅ 全メソッドでtick.time → tick.timestampへの変更完了
- ✅ TickAdapterによるDecimal精度の維持を実装
- 📁 変更ファイル:
  - `src/mt5_data_acquisition/tick_to_bar.py` - 更新（インポート変更、メソッド修正）
- 📝 備考:
  - add_tick(), _create_new_bar(), _update_bar()メソッドを修正
  - 基本動作確認テストで正常動作を確認
  - テストファイルの修正はStep 4で実施予定

## 📝 レビュー履歴

### 2025-08-21
- Tickモデル統一リファクタリング計画を策定
- 段階的移行アプローチを採用（リスク最小化）
- アダプターパターンによる後方互換性確保
- Step 1完了: timeプロパティによる後方互換性を実装

## 👁️ レビュー結果

### Step 1 実装レビュー（2025-08-21 12:45）

#### 良い点
- ✅ **後方互換性の実装が正確**: timeプロパティのgetter/setterが適切に実装され、timestamp属性との双方向同期が実現されている
- ✅ **包括的なテストカバレッジ**: 7つのテストケースで全ての重要なシナリオをカバー
- ✅ **既存機能への影響なし**: 既存の15個のmodelsテストが全て成功、破壊的変更なし
- ✅ **適切なドキュメント**: docstringで後方互換性の目的と削除予定（Step 6）を明記
- ✅ **Float32精度の考慮**: テストで適切な許容誤差（rel=1e-3）を設定

#### 改善点
- ⚠️ **リントエラーの存在**: 空白行やインポート順序などのフォーマット問題があったが、自動修正済み
- 優先度: 低（すでに対応済み）

#### 判定
- ✅ **合格**: Step 1の実装は計画通り完了。次のStep 2に進む準備が整った

### コミット結果
- **Hash**: 2665e68
- **Message**: feat: Step 1完了 - Tickモデルに後方互換性プロパティを追加
- **変更内容**:
  - src/common/models.py: timeプロパティ追加（110-130行目）
  - tests/unit/test_tick_model_compatibility.py: 新規テストファイル作成（7テストケース）
  - リントエラー修正（フォーマット適用）

### Step 2 実装レビュー（2025-08-21 15:00）

#### 良い点
- ✅ **アダプターパターンの適切な実装**: 静的メソッドによるシンプルな設計で、CommonTickとDecimal辞書間の変換を実現
- ✅ **完璧なテスト成功率**: 12個のテストケース全てがPASSED（100%成功）
- ✅ **包括的なテストカバレッジ**: 基本変換、属性名互換性、精度、エッジケースの4カテゴリを網羅
- ✅ **timestamp/time互換性の確保**: from_decimal_dictメソッドが両方の属性名を適切に処理
- ✅ **Float32精度の適切な考慮**: numpy.float32を使用した変換ロジックで精度を保持
- ✅ **既存機能への影響なし**: 既存の15個のmodelsテストが全て成功、破壊的変更なし

#### 改善点
- ⚠️ **リントエラーの存在**: 52個のフォーマット問題（空白行、インポート順序、型アノテーション形式）
  - 優先度: 中（コード品質に影響するが、動作には問題なし）
  - 対処: `uv run --frozen ruff check . --fix`で自動修正可能
- ⚠️ **カバレッジ低下**: 全体カバレッジが7.34%（要求80%未満）
  - 優先度: 低（新規ファイルのため他モジュールのカバレッジが低いことが原因）
  - 注記: tick_adapter.py自体は100%カバレッジ達成

#### 判定
- ✅ **合格（条件付き）**: Step 2の実装は機能的に完璧。リントエラーを修正後、次のStep 3に進む

### コミット結果（Step 2）
- **Hash**: 196223d
- **Message**: feat: Step 2完了 - TickAdapterクラスの実装
- **変更内容**:
  - src/mt5_data_acquisition/tick_adapter.py: 新規作成（77行）
  - tests/unit/test_tick_adapter.py: 新規テストファイル作成（267行）
  - リントエラー修正（フォーマット適用済み）
  - docs/context.md, docs/plan.md: ドキュメント更新

### Step 3 実装レビュー（2025-08-21 16:00）

#### 良い点
- ✅ **ローカルTickクラスの完全削除**: 17-49行目のローカルTickクラスが適切に削除された
- ✅ **正確なインポート変更**: common.models.TickとTickAdapterが正しくインポートされている
- ✅ **全メソッドでの属性名変更**: tick.time → tick.timestampへの変更が漏れなく実施
  - add_tick()メソッド: 126, 137, 143, 167, 171, 173行目
  - check_tick_gap(): 明示的な変更完了
  - _create_new_bar(): 298行目でtick.timestampを使用
- ✅ **TickAdapterの適切な使用**: 内部計算でDecimal精度を維持
  - _create_new_bar(): 295行目でto_decimal_dict()使用
  - _update_bar(): 332行目でto_decimal_dict()使用
- ✅ **基本動作の確認**: テストコードで正常動作を確認
  - OHLC計算が正確（High: 1.1240, Low: 1.1230）
  - ボリューム集計が正確（5.5）
  - タイムスタンプ逆転処理が正常
- ✅ **リントエラーの修正**: 全12個のフォーマット問題を自動修正済み

#### 改善点
- ⚠️ **テストの未修正**: tests/unit/test_tick_to_bar.pyがまだローカルTickをインポート
  - 優先度: 高（Step 4で対応予定）
  - 現状: 15テスト中10個が失敗（time vs timestamp問題）
- ⚠️ **docstringの未更新**: Example部分でtick.timeを使用（57行目）
  - 優先度: 低（動作に影響なし）

#### 判定
- ✅ **合格**: Step 3の実装は計画通り完了。主要な変更が正確に実施され、基本動作を確認。次のStep 4（テスト修正）に進む準備が整った

### コミット結果（Step 3）
- **Hash**: e152033
- **Message**: feat: Step 3完了 - TickToBarConverterをcommon.models.Tickに移行
- **変更内容**:
  - src/mt5_data_acquisition/tick_to_bar.py: 
    - ローカルTickクラス削除（17-49行目）
    - インポート変更（common.models.Tick, TickAdapter追加）
    - 全メソッドでtick.timestampを使用
    - TickAdapterによるDecimal変換追加
    - リントエラー修正（12個）

### Step 4 実装レビュー（2025-08-21 18:00）

#### 良い点
- ✅ **完璧なテスト成功率**: 全23個のテスト（ユニット15個、統合8個）が100%成功
- ✅ **正確なインポート変更**: common.models.Tickへの移行が完全に実施
- ✅ **属性名の完全な置換**: time → timestampへの変更が漏れなく実施（約30箇所）
- ✅ **型変換の適切な実装**: Decimal型からfloat型への変換が正確
- ✅ **Float32精度への適切な対応**: 
  - abs()を使用した許容誤差比較の実装
  - 金融データに適した0.0001の許容誤差設定
  - volumeのDecimal変換処理の追加
- ✅ **パフォーマンステストの成功**: 
  - 10,000ティック/秒以上の処理速度を達成
  - メモリ使用量が10MB以下の制限内
  - 1時間分のデータ処理も正常完了

#### 改善点
- ⚠️ **軽微なリントエラー**: docstring内の空白行問題（8箇所）
  - 優先度: 低（自動修正済み）
  - 対処: `--unsafe-fixes`オプションで修正完了
- ⚠️ **カバレッジの低下**: 全体カバレッジ12.02%（要求80%未満）
  - 優先度: 低（他モジュールの未実装部分が原因）
  - 注記: tick_to_bar.pyのカバレッジは83.89%で良好

#### 判定
- ✅ **合格**: Step 4の実装は完璧に完了。Tickモデルの統一が完全に実現され、全テストが成功。次のStep 5に進む準備が整った

## 📋 Step 4 実装計画（詳細）

### 4.0 現状分析
- **テスト失敗数**: 15テスト中10個が失敗
- **主要エラー**: ValidationError - timestampフィールドが必須だがtimeを使用
- **型の問題**: Decimal型を使用しているがfloat型が必要

### 4.1 修正対象ファイル

#### A. tests/unit/test_tick_to_bar.py
**必要な変更**:
1. インポート変更（14行目）
   - `from src.mt5_data_acquisition.tick_to_bar import Tick` を削除
   - `from src.common.models import Tick` を追加

2. 属性名の変更（全箇所）
   - `"time":` → `"timestamp":`
   - 影響箇所: 32, 39, 46, 53, 60行目など（sample_ticksフィクスチャ内）

3. 型の変更（全箇所）
   - `Decimal("1.04200")` → `1.04200`（float型）
   - bid, ask, volumeフィールドすべて

#### B. tests/integration/test_tick_to_bar_integration.py
**必要な変更**:
- 同様のインポート、属性名、型の変更が必要

### 4.2 段階的実装手順

#### Phase 1: インポートの修正（5分）
1. test_tick_to_bar.pyのインポートを変更
2. test_tick_to_bar_integration.pyのインポートを変更
3. 基本的なインポートエラーの解消

#### Phase 2: 属性名の一括変更（15分）
1. time → timestampへの変更
   - sample_ticksフィクスチャ内
   - 各テストケース内のTickオブジェクト生成箇所
2. sedまたは一括置換で効率的に実施

#### Phase 3: 型の変更（30分）
1. Decimal型からfloat型への変換
   - bid, ask, volumeフィールド
   - アサーション部分の期待値も変更
2. 精度問題への対処
   - 必要に応じてpytest.approxを使用

#### Phase 4: テスト実行と修正（30分）
1. 個別テストの実行と確認
2. エラーの特定と修正
3. 全体テストの実行

#### Phase 5: 統合テストの修正（40分）
1. test_tick_to_bar_integration.pyの同様の修正
2. 実際のMT5データとの互換性確認

### 4.3 修正例（具体的なコード）

**Before（現在のコード）**:
```python
from src.mt5_data_acquisition.tick_to_bar import Tick, TickToBarConverter

{
    "symbol": "EURUSD",
    "time": base_time + timedelta(seconds=0),
    "bid": Decimal("1.04200"),
    "ask": Decimal("1.04210"),
    "volume": Decimal("1.0"),
}
```

**After（修正後）**:
```python
from src.common.models import Tick
from src.mt5_data_acquisition.tick_to_bar import TickToBarConverter

{
    "symbol": "EURUSD",
    "timestamp": base_time + timedelta(seconds=0),
    "bid": 1.04200,
    "ask": 1.04210,
    "volume": 1.0,
}
```

### 4.4 テストの互換性維持

#### Decimal精度の考慮
- Barモデルは内部的にDecimalを使用
- TickAdapterがfloat→Decimal変換を担当
- テストのアサーションでDecimal型との比較が必要

#### 例:
```python
# アサーション部分は変更不要（Barは依然としてDecimalを返す）
assert completed_bar.open == Decimal("1.04200")
assert completed_bar.high == Decimal("1.04250")
```

### 4.5 リスクと対策

#### リスク1: Float精度による誤差
- **対策**: pytest.approxを使用した柔軟な比較
- **例**: `assert bar.open == pytest.approx(1.04200, rel=1e-5)`

#### リスク2: 既存の統合テストへの影響
- **対策**: 段階的な修正と確認
- **手順**: ユニットテスト→統合テストの順に修正

#### リスク3: MT5実データとの不整合
- **対策**: 実際のMT5データでの動作確認
- **検証**: サンプルデータでの往復変換テスト

### 4.6 成功基準
- ✅ 15個のユニットテストすべてが成功
- ✅ 統合テストが成功
- ✅ カバレッジ80%以上を維持
- ✅ パフォーマンスの劣化なし

### 4.7 チェックリスト
- [ ] test_tick_to_bar.pyのインポート修正
- [ ] sample_ticksフィクスチャの属性名変更
- [ ] sample_ticksフィクスチャの型変更
- [ ] 各テストケースのTick生成部分の修正
- [ ] テスト実行と全15テストの成功確認
- [ ] test_tick_to_bar_integration.pyの同様の修正
- [ ] 統合テストの成功確認
- [ ] カバレッジレポートの確認

## ⚠️ 注意事項
- Float32制約の維持が必須（プロジェクト要件）
- Polarsの使用（Pandas禁止）
- テストカバレッジ80%以上を維持
- 各ステップで回帰テストを実施

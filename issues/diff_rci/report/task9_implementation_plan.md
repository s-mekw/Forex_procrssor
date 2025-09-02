# task9 RCI計算エンジン実装計画書

## 1. 実装概要

本実装計画書は、Forex_procrssorプロジェクトのtask9「RCI計算エンジンの高速実装」について、Pythonサンプルコード（`rci_differential.py`）の差分計算アルゴリズムを基にした実装方針を定義します。

### 1.1 実装方針
- **アルゴリズム**: 差分計算による高速RCI実装（O(n)更新）
- **データ型**: Float32精度による メモリ効率化（50%削減）
- **フレームワーク**: Polars統合による高速データ処理
- **拡張性**: 任意期間（3〜200）対応の汎用エンジン

### 1.2 技術選定の根拠
| 技術要素 | 選定理由 |
|---------|---------|
| 差分計算アルゴリズム | 従来のO(n²)からO(n)への計算量削減により、リアルタイム処理に最適 |
| dequeバッファ | 固定サイズのスライディングウィンドウで メモリ効率的 |
| Float32精度 | 金融計算に十分な精度を保ちつつ、メモリ使用量を50%削減 |
| Polars Expression | ネイティブPolars統合により、データパイプラインとのシームレスな連携 |

## 2. クラス設計

### 2.1 クラス構成図

```
┌─────────────────────────────────────────────────────────┐
│                    RCICalculatorEngine                  │
│  (メインエンジン：複数期間の並列RCI計算を管理)         │
└─────────────────────────────────────────────────────────┘
                            │
                  ┌─────────┴─────────┐
                  │                   │
        ┌─────────▼─────────┐ ┌──────▼──────────┐
        │DifferentialRCICalc│ │  RCIProcessor   │
        │(単一期間計算)     │ │ (Polars統合)    │
        └───────────────────┘ └─────────────────┘
```

### 2.2 クラス詳細設計

#### 2.2.1 DifferentialRCICalculator
```python
class DifferentialRCICalculator:
    """単一期間のストリーミングRCI計算クラス
    
    Attributes:
        period (int): RCI計算期間（3〜200）
        prices (deque): 価格データのスライディングウィンドウ
        time_ranks (np.ndarray): 時間順位（事前計算）
        denominator (float): RCI式の分母（事前計算）
    
    Methods:
        add(price: float) -> Optional[float]: 
            新価格を追加してRCIを計算
        _optimized_ranking(prices: np.ndarray) -> np.ndarray:
            Float32精度での最適化ランキング計算
    """
```

#### 2.2.2 RCICalculatorEngine
```python
class RCICalculatorEngine:
    """複数期間対応の汎用RCI計算エンジン
    
    Constants:
        DEFAULT_PERIODS = [9, 13, 24, 33, 48, 66, 108]
        MIN_PERIOD = 3
        MAX_PERIOD = 200  # 仕様書に合わせて540から200に変更
    
    Methods:
        calculate_multiple(
            data: pl.DataFrame,
            periods: Optional[List[int]] = None,
            column_name: str = 'close',
            mode: Literal['batch', 'streaming'] = 'batch'
        ) -> pl.DataFrame:
            複数期間のRCI計算（メインインターフェース）
        
        validate_periods(periods: List[int]) -> None:
            期間パラメータのバリデーション
        
        _process_streaming(data, periods, column_name) -> pl.DataFrame:
            ストリーミングモードでの処理
        
        _process_batch(data, periods, column_name) -> pl.DataFrame:
            バッチモードでの処理（Polars最適化）
    """
```

#### 2.2.3 RCIProcessor
```python
class RCIProcessor:
    """Polars Expression統合用ラッパークラス
    
    Methods:
        create_rci_expression(
            column: str,
            period: int
        ) -> pl.Expr:
            Polars Expressionとして RCI計算を定義
        
        apply_to_dataframe(
            df: pl.DataFrame,
            periods: List[int],
            column_name: str = 'close'
        ) -> pl.DataFrame:
            DataFrameに直接RCI計算を適用
    """
```

### 2.3 エラー処理設計

```python
class RCICalculationError(Exception):
    """RCI計算固有のエラー"""
    pass

class InvalidPeriodError(RCICalculationError):
    """無効な期間パラメータエラー"""
    pass

class InsufficientDataError(RCICalculationError):
    """データ不足エラー"""
    pass
```

## 3. 既存仕様との整合性

### 3.1 要件適合性チェック

| 要件ID | 要件内容 | 実装方針 | 適合性 |
|--------|---------|---------|--------|
| 2.3.1 | ユーザー設定可能な期間リスト | `periods`パラメータで任意リスト指定可能 | ✓ |
| 2.3.2 | 動的な期間リスト更新 | 実行時にperiodsパラメータで変更可能 | ✓ |
| 2.3.3 | Polars Expression統合 | RCIProcessorクラスで実装 | ✓ |
| 2.3.4 | 3〜200期間の汎用対応 | MIN_PERIOD=3, MAX_PERIOD=200で制約 | ✓ |
| 2.3.5 | -100〜+100の正規化 | RCI式に基づく標準計算 | ✓ |
| 2.3.6 | 信頼性フラグ | `rci_{period}_reliable`カラムで実装 | ✓ |
| 2.3.7 | 無効期間の制限と警告 | validate_periods()で実装 | ✓ |

### 3.2 技術スタック準拠

| 仕様 | 実装内容 | 準拠状況 |
|------|---------|----------|
| Polars必須 | 全データ処理でPolars使用 | ✓ |
| Float32精度 | np.float32とpl.Float32統一 | ✓ |
| 型ヒント必須 | 全メソッドで型ヒント実装 | ✓ |
| 非同期処理対応 | async/awaitインターフェース提供 | 予定 |

### 3.3 ファイル配置

```
src/
└── data_processing/
    ├── __init__.py        # RCICalculatorEngineをエクスポート追加
    ├── rci.py            # 新規作成（メイン実装）
    └── processor.py      # 既存（統合ポイント）
```

## 4. 実装ステップ

### Phase 1: 基礎実装（2日）
- [X] Step 1.1: DifferentialRCICalculatorクラスの実装
  - dequeによるスライディングウィンドウ
  - Float32精度での計算
  - 基本的な差分更新ロジック

- [X] Step 1.2: ランキング計算の最適化
  - scipy.stats.rankdataによる正確なタイ処理
  - NumPyフォールバック実装
  - パフォーマンステスト

### Phase 2: エンジン実装（2日）
- [X] Step 2.1: RCICalculatorEngineクラスの実装
  - 複数期間の並列計算
  - バリデーションロジック
  - エラーハンドリング

- [X] Step 2.2: バッチ/ストリーミングモード実装
  - モード切り替えロジック
  - 大規模データ用バッチ処理
  - リアルタイム用ストリーミング処理

### Phase 3: Polars統合（1日）
- [X] Step 3.1: RCIProcessorクラスの実装
  - Polars Expression生成
  - LazyFrame対応
  - 最適化されたグループ処理

- [X] Step 3.2: 既存processor.pyとの統合
  - PolarsProcessingEngineへの組み込み
  - パイプライン統合テスト

### Phase 4: テストとドキュメント（1日）
- [X] Step 4.1: ユニットテスト作成
  - tests/unit/test_rci.pyの実装
  - 各クラスの単体テスト
  - エッジケースのテスト

- [X] Step 4.2: 統合テストとベンチマーク
  - 大規模データでのパフォーマンステスト
  - メモリ使用量の測定
  - 精度検証
  
### Phase 5: テストカバレッジ向上（2日）

#### 目標
- **rci.py**: 現在67.00% → 85%以上
- **全体カバレッジ**: 現在9.66% → 15%以上
- **品質指標**: mutation testing導入によるテスト品質向上

#### Step 5.1: コアロジックの追加テスト
- [ ] 未カバーコードパスの網羅的テスト
  - エラーハンドリングの全分岐パス
  - 並列処理モード（batch/streaming/auto）の境界条件
  - キャッシュ機構の有効性検証
  - Float32/Float64精度切り替えロジック

- [ ] エッジケース の拡充
  - 極小期間（period=3）での動作
  - 最大期間（period=200）での メモリ効率
  - 同一値連続データでのランキング処理
  - NaN/Inf混入データの処理

#### Step 5.2: モックベースの統合テスト
- [ ] 外部依存関係のモック化
  ```python
  # tests/unit/test_rci_mocked.py
  @patch('psutil.Process')
  def test_memory_monitoring_with_mock(mock_process):
      """メモリ監視機能のモックテスト"""
      mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 100
      # メモリ制限下での動作確認
  
  @patch('scipy.stats.rankdata')
  def test_ranking_fallback(mock_rankdata):
      """ランキング計算のフォールバック動作"""
      mock_rankdata.side_effect = ImportError
      # NumPyフォールバック実装の検証
  ```

- [ ] ログ出力の検証
  - 各ログレベルでの適切な出力
  - エラー時の詳細情報記録
  - パフォーマンス統計の記録

#### Step 5.3: プロパティベーステスト
- [ ] hypothesis使用による数学的性質の検証
  ```python
  # tests/unit/test_rci_properties.py
  from hypothesis import given, strategies as st
  
  @given(
      prices=st.lists(
          st.floats(min_value=0.01, max_value=1000000),
          min_size=3, max_size=200
      ),
      period=st.integers(min_value=3, max_value=200)
  )
  def test_rci_range_property(prices, period):
      """RCI値が常に-100〜100の範囲内であることを検証"""
      if len(prices) >= period:
          rci = calculate_rci(prices, period)
          assert -100 <= rci <= 100
  ```

- [ ] 不変条件の検証
  - 同一データで同一結果（決定性）
  - 順位相関の数学的性質
  - 差分更新の正確性

#### Step 5.4: パフォーマンス回帰テスト
- [ ] ベンチマークスイートの作成
  ```python
  # tests/performance/test_rci_regression.py
  class TestRCIPerformanceRegression:
      """パフォーマンス劣化を検出する回帰テスト"""
      
      BASELINE_METRICS = {
          'single_update': 0.1,  # ms
          'batch_10k': 1.0,      # seconds
          'memory_10k': 100,     # MB
      }
      
      def test_single_update_performance(self):
          """単一更新のパフォーマンス基準"""
          # 基準値からの乖離が10%以内であることを確認
      
      def test_batch_processing_scalability(self):
          """バッチ処理のスケーラビリティ"""
          # O(n)の計算量を維持していることを確認
  ```

- [ ] CI/CDパイプラインへの統合
  - GitHub Actionsでの自動実行
  - パフォーマンス劣化時の自動通知
  - ベンチマーク結果の可視化

#### Step 5.5: 統合テストの拡充
- [ ] 実データを使用した End-to-End テスト
  ```python
  # tests/integration/test_rci_e2e.py
  def test_full_pipeline_with_real_data():
      """実際のFXデータでの完全パイプラインテスト"""
      # 1. データ取得
      # 2. RCI計算
      # 3. 結果検証
      # 4. パフォーマンス測定
  ```

- [ ] 他モジュールとの連携テスト
  - processor.pyとの統合
  - indicators.pyとの組み合わせ
  - pipeline.pyでの使用

#### Step 5.6: カバレッジ分析と改善
- [ ] カバレッジレポートの詳細分析
  - 未カバー行の特定と分類
  - ブランチカバレッジの向上
  - 条件カバレッジの測定

- [ ] テスト品質メトリクスの導入
  - Mutation Testing（mutmut使用）
  - Cyclomatic Complexityの測定
  - テストの保守性評価

#### 実装優先順位
1. **高優先度**: Step 5.1（コアロジックのテスト）
2. **中優先度**: Step 5.3（プロパティベーステスト）、Step 5.5（統合テスト）
3. **低優先度**: Step 5.2（モックテスト）、Step 5.4（回帰テスト）

#### 成功基準
- [ ] rci.pyのカバレッジ85%以上達成
- [ ] 全エッジケースのテスト完了
- [ ] パフォーマンス回帰テストの自動化
- [ ] mutation scoreが80%以上
- [ ] CI/CDパイプラインでの全テスト通過

## 5. インターフェース設計

### 5.1 基本使用例

```python
from src.data_processing.rci import RCICalculatorEngine
import polars as pl

# エンジンの初期化
engine = RCICalculatorEngine()

# データの準備
df = pl.DataFrame({
    'timestamp': [...],
    'close': [100.5, 101.2, 99.8, ...]
})

# RCI計算（デフォルト期間）
result = engine.calculate_multiple(
    data=df,
    column_name='close'
)

# カスタム期間でのRCI計算
result = engine.calculate_multiple(
    data=df,
    periods=[9, 13, 24, 50, 100],
    column_name='close',
    mode='streaming'  # リアルタイムモード
)
```

### 5.2 Polars統合例

```python
from src.data_processing.rci import RCIProcessor

processor = RCIProcessor()

# Polars Expressionとして使用
df = df.with_columns([
    processor.create_rci_expression('close', period=9).alias('rci_9'),
    processor.create_rci_expression('close', period=13).alias('rci_13')
])

# LazyFrameでの使用
lazy_df = df.lazy().with_columns(
    processor.create_rci_expression('close', 24)
).collect()
```

### 5.3 ストリーミング処理例

```python
from src.data_processing.rci import DifferentialRCICalculator

# ストリーミング計算
calculator = DifferentialRCICalculator(period=9)

async def process_tick_stream(tick_stream):
    async for tick in tick_stream:
        rci = calculator.add(tick.close)
        if rci is not None:
            print(f"RCI(9): {rci:.2f}")
```

## 6. パフォーマンス目標

### 6.1 計算性能
| メトリクス | 目標値 | 測定方法 |
|-----------|--------|----------|
| 単一期間RCI計算 | < 0.1ms/update | タイマー計測 |
| 12期間並列計算 | < 1ms/update | プロファイリング |
| バッチ処理（10万件） | < 1秒 | ベンチマーク |
| メモリ使用量 | < 100MB（10万件） | memory_profiler |

### 6.2 最適化戦略
1. **NumPy/Polarsベクトル化**: ループの最小化
2. **事前計算**: 定数部分のキャッシュ
3. **Float32精度**: メモリ帯域幅の削減
4. **並列処理**: 複数期間の同時計算

## 7. 提案する仕様変更・追加

### 7.1 追加機能提案

#### 提案1: 適応的期間選択
```python
class AdaptiveRCIEngine(RCICalculatorEngine):
    """市場ボラティリティに応じて期間を自動調整"""
    
    def calculate_adaptive(
        self,
        data: pl.DataFrame,
        base_periods: List[int],
        volatility_factor: float = 1.0
    ) -> pl.DataFrame:
        # ボラティリティに基づく期間調整ロジック
        pass
```

**利点**: 市場状況に応じた動的な分析が可能

#### 提案2: 増分更新API
```python
class IncrementalRCIEngine:
    """既存のRCI値を基に差分更新のみ実行"""
    
    def update_incremental(
        self,
        previous_state: Dict,
        new_data: pl.DataFrame
    ) -> Tuple[pl.DataFrame, Dict]:
        # 状態を保持して増分更新
        pass
```

**利点**: リアルタイム処理の更なる高速化

### 7.2 設定ファイル拡張提案

`config/rci_settings.toml`の追加：
```toml
[rci]
default_periods = [9, 13, 24, 33, 48, 66, 108]
max_period = 200
min_period = 3

[rci.performance]
use_float32 = true
enable_parallel = true
batch_size = 10000

[rci.realtime]
streaming_buffer_size = 1000
update_interval_ms = 100
```

**利点**: 運用時の柔軟な設定変更が可能

## 8. リスクと対策

### 8.1 技術的リスク

| リスク | 影響度 | 発生確率 | 対策 |
|--------|--------|----------|------|
| Float32精度不足 | 中 | 低 | 重要な計算部分のみFloat64使用オプション |
| メモリ不足（大規模データ） | 高 | 中 | チャンク処理とストリーミング処理の自動切替 |
| 並列処理のオーバーヘッド | 低 | 中 | 期間数に応じた動的な並列度調整 |

### 8.2 実装上の注意点

1. **スレッドセーフティ**: 
   - dequeの操作をスレッドセーフに
   - 共有状態の最小化

2. **数値安定性**:
   - ゼロ除算の回避
   - NaN/Inf値の適切な処理

3. **後方互換性**:
   - 既存のprocessor.pyとの互換性維持
   - 段階的な移行パスの提供

## 9. 成功基準

### 9.1 機能要件の達成
- [ ] 全てのユニットテストがパス（カバレッジ80%以上）
- [ ] 要件2.3の全項目を満たす
- [ ] パフォーマンス目標の達成

### 9.2 品質基準
- [ ] コードレビューの完了
- [ ] ドキュメントの完備
- [ ] パフォーマンステストの合格

## 10. まとめ

本実装計画は、Pythonサンプルコードの差分計算アルゴリズムを基に、既存のForex_procrssorプロジェクトの仕様と完全に整合する形でRCI計算エンジンを実装します。

### 主要な特徴
- **高速性**: O(n)の差分更新による高速計算
- **効率性**: Float32精度による メモリ使用量50%削減
- **拡張性**: 3〜200期間の任意設定に対応
- **統合性**: 既存のPolarsパイプラインとシームレスに統合

### 次のステップ
1. 本計画書のレビューと承認
2. Phase 1の基礎実装開始
3. 継続的なテストとフィードバック

---

*作成日: 2025年8月25日*  
*対象タスク: task9 - RCI計算エンジンの高速実装*  
*参考実装: rci_differential.py*
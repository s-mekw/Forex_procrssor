# 差分RCI計算アルゴリズム分析レポート

## 1. 概要

本レポートでは、RCI（Rank Correlation Index）の高速計算を実現する2つの異なる実装アプローチを分析します。両実装とも「差分計算」という革新的な手法を採用し、従来のRCI計算の計算量を大幅に削減しています。

### 1.1 差分RCI計算手法の特徴

**従来のRCI計算の問題点:**
- 新しいデータポイントが追加されるたびに、全期間のランキングを再計算
- 計算量: O(n² × m) （n: 期間、m: データ長）
- リアルタイムトレーディングには不向き

**差分計算による改善:**
- 前回の計算結果を活用し、差分のみを更新
- 計算量: O(n) per update
- メモリ効率的なスライディングウィンドウ実装

## 2. MT5実装（diff_RCI.mq5）の詳細分析

### 2.1 アルゴリズムの核心概念

MT5実装は、整数配列を使用した独自のランキング差分管理システムを採用しています。

```mql5
int D0[];  // 基準ランキング配列
int D1[];  // 作業用ランキング配列
```

### 2.2 データ構造と初期化

```mql5
// 初期化処理
for(i=0; i<tPrMax; i++) {
    D0[i] = -i*2;  // 基準値として-2iを設定
}
n1 = tPr1 * (tPr1*tPr1 - 1) * 2 / 3;  // RCI計算の分母を事前計算
```

**重要な設計上の工夫:**
- D0配列に-2iを格納することで、ランキング差分の計算を簡略化
- 分母の事前計算により、実行時の計算負荷を軽減

### 2.3 初期RCI計算（CalculateInitialRCI）

```mql5
// 全ペア比較によるランキング計算
for(i = 0; i < rci_period; i++) {
    for(j = i + 1; j < rci_period; j++) {
        if(close[n + i] < close[n + j]) {
            Dx[i] += 2;
        } else if(close[n + i] == close[n + j]) {
            Dx[i]++;
            Dx[j]++;
        } else {
            Dx[j] += 2;
        }
    }
    dx += Dx[i] * Dx[i];
}
RCI = (1 - (dx / nx)) * 100;
```

**アルゴリズムの特徴:**
- ペアワイズ比較によるランキング計算
- タイ（同値）の場合は両方のランクを調整
- 整数演算による高速化

### 2.4 差分更新（UpdateRCI）

```mql5
// 既存ランキングの差分更新
for(i = rci_period - 1; i >= 1; i--) {
    Dx[i] = Dx[i - 1] - 2;  // 時間経過によるランクシフト
    
    // 新データとの比較
    if(close[n + i] < close[n])
        Dx[i] += 2;
    else if(close[n + i] == close[n])
        Dx[i]++;
    
    // 古いデータとの比較調整
    if(close[n + i] < close[n + rci_period])
        Dx[i] -= 2;
    else if(close[n + i] == close[n + rci_period])
        Dx[i]--;
        
    dx += Dx[i] * Dx[i];
}
```

**差分計算の巧妙さ:**
- 時間経過による自然なランクシフト（-2）
- 新データと既存データの比較による調整
- ウィンドウから外れるデータの影響を除去

### 2.5 パフォーマンス特性

| 特性 | 詳細 |
|------|------|
| 計算量（初期） | O(n²) |
| 計算量（更新） | O(n) |
| メモリ使用量 | O(n) |
| 数値精度 | 整数演算による高精度 |
| リアルタイム性 | 非常に高い |

## 3. Python実装（rci_differential.py）の詳細分析

### 3.1 設計思想

Python実装は、モダンなデータ処理ライブラリ（Polars、NumPy）を活用し、大規模データ処理に適した設計となっています。

### 3.2 核心データ構造

```python
class DifferentialRCICalculator:
    def __init__(self, period: int):
        self.prices = deque(maxlen=period)  # スライディングウィンドウ
        self.time_ranks = np.arange(0, period, dtype=np.float32)
        self.denominator = np.float32(period * (period**2 - 1))
```

**設計上の特徴:**
- `deque`による効率的なFIFOバッファ
- Float32精度による メモリ使用量50%削減
- 時間ランクの事前計算

### 3.3 ランキング計算の最適化

```python
def _optimized_ranking(self, prices: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import rankdata
        # scipy使用時：タイの正確な処理
        return (rankdata(prices, method='average') - 1).astype(np.float32)
    except ImportError:
        # フォールバック：NumPyによる実装
        return np.argsort(np.argsort(prices)).astype(np.float32)
```

**実装の工夫:**
- scipyによる高精度なタイ処理（平均ランク法）
- NumPyフォールバックによる依存性の軽減
- 0ベースランキングへの変換

### 3.4 バッチ処理とLazyFrame最適化

```python
def _process_with_batching(self, df, periods, column_name, batch_size, vectorized):
    overlap = max_period - 1  # バッチ間の継続性確保
    
    for start_idx in range(0, data_length, batch_size):
        # オーバーラップを含むバッチ抽出
        batch_df = df.slice(start_idx, end_idx - start_idx)
        batch_result = self.fast_calculate_multiple(batch_df, periods, column_name)
        
        # オーバーラップ除去
        if start_idx > 0:
            batch_result = batch_result.slice(overlap)
```

**大規模データ処理の工夫:**
- バッチ処理による メモリ効率化
- オーバーラップによる境界での連続性確保
- LazyFrame評価による遅延実行

### 3.5 パフォーマンス特性

| 特性 | 詳細 |
|------|------|
| 計算量 | O(n log n) per window |
| メモリ効率 | Float32による50%削減 |
| 並列処理 | 複数期間の並列計算対応 |
| スケーラビリティ | バッチ処理で大規模データ対応 |
| 依存性 | Polars、NumPy、オプションでSciPy |

## 4. 共通点と相違点の比較分析

### 4.1 共通点

| 項目 | 説明 |
|------|------|
| **差分計算** | 両実装とも前回の結果を活用した差分更新 |
| **スライディングウィンドウ** | 固定サイズのウィンドウで効率的なデータ管理 |
| **事前計算** | RCI式の分母など、定数部分の事前計算 |
| **リアルタイム性** | ストリーミングデータへの対応 |
| **メモリ効率** | O(n)のメモリ使用量 |

### 4.2 相違点

| 項目 | MT5実装 | Python実装 |
|------|---------|------------|
| **ランキング計算** | ペアワイズ比較（整数演算） | ソート＋順位付け（浮動小数点） |
| **タイ処理** | インクリメント方式 | 平均ランク法（scipy） |
| **データ型** | 整数配列 | Float32配列 |
| **更新方式** | 配列要素の直接更新 | deque+新規計算 |
| **複数期間対応** | 単一期間特化 | 複数期間の並列処理 |
| **バッチ処理** | なし（リアルタイム特化） | 大規模データ用バッチ処理 |
| **拡張性** | MQL5環境に特化 | Python生態系との統合 |

### 4.3 計算精度の違い

**MT5実装:**
- 整数演算による丸め誤差の排除
- ペアワイズ比較による厳密なランキング
- タイの処理がシンプル

**Python実装:**
- Float32による計算速度の向上
- scipyによる統計的に正確なタイ処理
- 数値精度と速度のバランス

## 5. task9実装への推奨事項

### 5.1 ハイブリッドアプローチの提案

両実装の長所を組み合わせた最適な実装戦略：

```python
class OptimizedRCIEngine:
    """
    MT5の差分更新アルゴリズム + Pythonの並列処理能力
    """
    
    def __init__(self, periods: List[int]):
        # MT5方式：整数配列による差分管理
        self.rank_diffs = {p: np.zeros(p, dtype=np.int32) for p in periods}
        
        # Python方式：効率的なバッファ管理
        self.price_buffers = {p: deque(maxlen=p) for p in periods}
        
        # 事前計算値のキャッシュ
        self.denominators = {p: p * (p**2 - 1) for p in periods}
    
    def update_differential(self, new_price: float):
        """MT5方式の差分更新ロジック"""
        # 実装詳細...
        
    def batch_process(self, data: pl.DataFrame):
        """Python方式のバッチ処理"""
        # 実装詳細...
```

### 5.2 パフォーマンス最適化戦略

1. **計算の最適化**
   - 小規模期間（< 50）: MT5方式の整数演算
   - 大規模期間（>= 50）: Python方式のベクトル化
   - クリティカルパス: Cython/Numbaによる高速化

2. **メモリ最適化**
   - Float32/Int32の使い分け
   - 共有メモリプールの活用
   - 不要なコピーの削減

3. **並列処理**
   - 複数期間の並列計算
   - SIMD命令の活用
   - GPU計算の検討（大規模バッチ処理時）

### 5.3 実装上の注意点

1. **数値精度**
   - 金融計算には適切な精度が必要
   - Float32とFloat64の使い分け
   - オーバーフロー/アンダーフローの考慮

2. **エラーハンドリング**
   - NaN/Inf値の適切な処理
   - データ欠損への対応
   - 期間境界での継続性保証

3. **テスト戦略**
   - 単体テスト: 各計算ロジックの検証
   - 統合テスト: ストリーミング処理の検証
   - パフォーマンステスト: 大規模データでの性能評価

## 6. まとめ

両実装とも差分計算による革新的な高速化を実現していますが、それぞれ異なる強みを持っています：

- **MT5実装**: リアルタイム性と計算精度を重視した軽量実装
- **Python実装**: スケーラビリティと拡張性を重視した汎用実装

task9の実装では、これらの特徴を理解し、要件に応じて適切なアプローチを選択または組み合わせることが重要です。特に、リアルタイムトレーディングでは MT5方式の軽量性が、バックテストや大規模分析ではPython方式の並列処理能力が有効となるでしょう。

---

*作成日: 2025年8月25日*  
*分析対象: diff_RCI.mq5, rci_differential.py*
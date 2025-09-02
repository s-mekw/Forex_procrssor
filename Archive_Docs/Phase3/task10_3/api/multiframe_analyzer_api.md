# MultiTimeframeAnalyzer API仕様書

## 概要
MultiTimeframeAnalyzerクラスは、マルチタイムフレームRCI分析を提供するコンポーネントです。Task 10.3のリファクタリングにより、バッファ管理機能が追加され、より独立したコンポーネントとなりました。

## クラス定義

```python
class MultiTimeframeAnalyzer:
    """マルチタイムフレームRCI分析クラス"""
    
    def __init__(
        self,
        short_rci_period: int = 9,
        long_rci_period: int = 26,
        signal_period: int = 13,
        mt5_client: Optional[Any] = None,
        max_history_bars: int = 5000
    ):
        """
        Args:
            short_rci_period: 短期RCI計算期間
            long_rci_period: 長期RCI計算期間
            signal_period: シグナル計算期間
            mt5_client: MT5クライアント（オプション）
            max_history_bars: 保持する最大履歴バー数
        """
```

## 新規追加API（Task 10.3）

### add_new_bar()
```python
def add_new_bar(self, bar: dict[str, Any]) -> None:
    """
    新しいバーをバッファに追加する
    
    Args:
        bar: 追加するバーデータ
            必須フィールド:
            - timestamp: datetime
            - open: float
            - high: float
            - low: float
            - close: float
            - volume: float
    
    Note:
        - None値の場合は処理をスキップ
        - バッファサイズが最大値を超えた場合、古いデータを自動削除
    """
```

### is_ready()
```python
def is_ready(self) -> bool:
    """
    分析準備が完了しているかを返す
    
    Returns:
        bool: 200バー以上のデータがある場合True
    
    Example:
        >>> analyzer = MultiTimeframeAnalyzer()
        >>> analyzer.is_ready()
        False
        >>> # 200バー追加後
        >>> analyzer.is_ready()
        True
    """
```

### get_buffer_size()
```python
def get_buffer_size(self) -> int:
    """
    現在のバッファサイズを返す
    
    Returns:
        int: バッファ内のバー数
    
    Example:
        >>> analyzer = MultiTimeframeAnalyzer()
        >>> analyzer.get_buffer_size()
        0
        >>> analyzer.add_new_bar(bar_data)
        >>> analyzer.get_buffer_size()
        1
    """
```

### get_buffer_as_dataframe()
```python
def get_buffer_as_dataframe(self) -> Optional[pl.DataFrame]:
    """
    バッファをPolars DataFrameとして取得
    
    Returns:
        Optional[pl.DataFrame]: バッファデータ、空の場合None
    
    Schema:
        - timestamp: datetime
        - open: float64
        - high: float64
        - low: float64
        - close: float64
        - volume: float64
    """
```

## 更新されたAPI

### analyze_streaming()
```python
def analyze_streaming(
    self,
    new_bar: Optional[dict[str, Any]] = None,
    history: Optional[pl.DataFrame] = None,
    min_history_bars: int = 200
) -> dict[str, Any]:
    """
    ストリーミングデータの分析を実行
    
    Args:
        new_bar: 新しいバーデータ（後方互換性のため、deprecated）
        history: 履歴データ（外部バッファモード）
        min_history_bars: 最小必要バー数（外部バッファモード）
    
    Returns:
        dict: 分析結果
            - timestamp: datetime
            - rci_short: float（短期RCI）
            - rci_long: float（長期RCI）
            - signal: float（シグナル値）
            - trend: str（'uptrend'/'downtrend'/'neutral'）
            - strength: float（トレンド強度）
            - status: str（'ready'/'not_ready'）
    
    Note:
        - historyがNoneの場合、内部バッファを使用（推奨）
        - historyが指定された場合、外部バッファモード（後方互換性）
    """
```

## Protocol定義

```python
from typing import Protocol

class AnalyzerProtocol(Protocol):
    """Analyzerインターフェースの定義"""
    
    def add_new_bar(self, bar: dict[str, Any]) -> None: ...
    def is_ready(self) -> bool: ...
    def get_buffer_size(self) -> int: ...
    def analyze_streaming(self) -> dict[str, Any]: ...
```

## 使用例

### 基本的な使用方法
```python
# 初期化
analyzer = MultiTimeframeAnalyzer(
    short_rci_period=9,
    long_rci_period=26,
    max_history_bars=5000
)

# データの追加
for bar in historical_data:
    analyzer.add_new_bar(bar)

# 準備状態の確認
if analyzer.is_ready():
    # 分析の実行
    result = analyzer.analyze_streaming()
    print(f"RCI Short: {result['rci_short']}")
    print(f"RCI Long: {result['rci_long']}")
    print(f"Trend: {result['trend']}")
```

### RealtimePipelineとの統合
```python
class RealtimePipeline:
    def __init__(self, analyzer: Optional[AnalyzerProtocol] = None):
        self._multiframe_analyzer = analyzer or MultiTimeframeAnalyzer()
    
    def _process_message(self, data_point):
        # データ処理
        new_bar = self._create_bar(data_point)
        
        # Analyzerに追加
        self._multiframe_analyzer.add_new_bar(new_bar)
        
        # 分析実行
        if self._multiframe_analyzer.is_ready():
            result = self._multiframe_analyzer.analyze_streaming()
            self._handle_analysis_result(result)
```

## 後方互換性

### 移行ガイド
現在のコードは完全な後方互換性を維持しています。段階的な移行が可能です。

#### 旧スタイル（外部バッファ）
```python
# 従来の使用方法（引き続き動作）
history_df = pl.DataFrame(data_buffer)
result = analyzer.analyze_streaming(
    new_bar=new_bar,
    history=history_df,
    min_history_bars=200
)
```

#### 新スタイル（内部バッファ）
```python
# 推奨される新しい使用方法
analyzer.add_new_bar(new_bar)
if analyzer.is_ready():
    result = analyzer.analyze_streaming()
```

## エラーハンドリング

### None値の処理
```python
analyzer.add_new_bar(None)  # 処理をスキップ、エラーなし
```

### 異常値の処理
```python
# NaN値を含むバー
bar_with_nan = {
    "timestamp": datetime.now(),
    "close": float('nan'),
    # ...
}
analyzer.add_new_bar(bar_with_nan)  # 正常に処理
```

### バッファオーバーフロー
```python
# 最大5000バーを超えた場合、自動的に古いデータを削除
for i in range(10000):
    analyzer.add_new_bar(create_bar(i))
assert analyzer.get_buffer_size() == 5000  # 最大値でキャップ
```

## パフォーマンス特性

### 時間計算量
- `add_new_bar()`: O(1) - 定数時間
- `is_ready()`: O(1) - 定数時間
- `get_buffer_size()`: O(1) - 定数時間
- `analyze_streaming()`: O(n) - nは分析期間

### 空間計算量
- バッファメモリ: O(max_history_bars)
- 約1.66 KB/バー

## ベストプラクティス

### 1. 内部バッファの使用
```python
# 推奨
analyzer.add_new_bar(bar)
result = analyzer.analyze_streaming()

# 非推奨（後方互換性のため維持）
result = analyzer.analyze_streaming(new_bar, history)
```

### 2. エラーハンドリング
```python
try:
    analyzer.add_new_bar(bar)
    if analyzer.is_ready():
        result = analyzer.analyze_streaming()
        process_result(result)
except Exception as e:
    logger.error(f"Analysis failed: {e}")
```

### 3. メモリ管理
```python
# 適切なmax_history_barsの設定
analyzer = MultiTimeframeAnalyzer(
    max_history_bars=2000  # 必要最小限に設定
)
```

## 今後の拡張予定

### v2.0で予定される機能
- 非同期API対応
- ストリーミングイテレータ
- カスタム指標のプラグイン機構
- メモリマップドバッファ対応

### 廃止予定のAPI
- `analyze_streaming(new_bar, history)` - v2.0で廃止予定
- 代替: 内部バッファモードの使用を推奨
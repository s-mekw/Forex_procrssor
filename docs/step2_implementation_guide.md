# Step 2 実装ガイド

## 実装概要
RealtimePipelineから以下のロジックをMultiTimeframeAnalyzerに移譲します。

## 1. MultiTimeframeAnalyzerに追加する機能

### バッファ管理機能の追加
```python
class MultiTimeframeAnalyzer:
    def __init__(self, config: Optional[Config] = None):
        # 既存のコード...
        
        # バッファ管理用の新しい属性を追加
        self._data_buffer = []
        self._max_history_bars = 1500  # configから取得する
        
    def add_new_bar(self, new_bar: dict) -> None:
        """新しいバーをバッファに追加する
        
        Args:
            new_bar: {
                "timestamp": pd.Timestamp,
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float
            }
        """
        self._data_buffer.append(new_bar)
        self._manage_buffer_size()
        
    def _manage_buffer_size(self) -> None:
        """バッファサイズを管理する"""
        if len(self._data_buffer) > self._max_history_bars:
            self._data_buffer = self._data_buffer[-self._max_history_bars:]
            
    def get_buffer_size(self) -> int:
        """現在のバッファサイズを返す"""
        return len(self._data_buffer)
        
    def is_ready(self) -> bool:
        """分析に必要な最小バー数が揃っているかチェック"""
        min_required_bars = 200  # configから取得する
        return len(self._data_buffer) >= min_required_bars
        
    def analyze_streaming(self, new_bar: Optional[dict] = None) -> dict:
        """
        変更: new_barは内部バッファから取得するため、引数を省略可能にする
        """
        if not self.is_ready():
            return {
                "short_rci": None,
                "long_rci": None,
                "is_new_long_bar": False,
                "error": "Insufficient history for analysis"
            }
            
        # DataFrame変換を内部で行う
        history_df = pl.DataFrame(self._data_buffer)
        
        # 既存の分析ロジック...
        # (historyとnew_barパラメータは内部バッファから生成)
```

## 2. RealtimePipelineの変更点

### 削除するコード（L79, L194-213）
```python
# 削除対象:
# L79: self._data_buffer = []  # __init__から削除
# L194-207: バッファ管理ロジックを削除
# L210-213: 最小バー数チェックとDataFrame変換を削除
```

### 新しい実装
```python
def _process_message(self, message: str) -> dict:
    # 既存の処理...
    
    if self._enable_multiframe and self._multiframe_analyzer:
        try:
            multiframe_start = time.time()
            
            # バッファ管理をanalyzerに委譲
            new_bar = {
                "timestamp": data_point["timestamp"],
                "open": processed_data.get("open", 0),
                "high": processed_data.get("high", 0),
                "low": processed_data.get("low", 0),
                "close": processed_data.get("close", 0),
                "volume": processed_data.get("volume", 0),
            }
            
            # MultiTimeframeAnalyzerにバーを追加
            self._multiframe_analyzer.add_new_bar(new_bar)
            
            # 分析実行の準備ができているかチェック
            if self._multiframe_analyzer.is_ready():
                # ストリーミング分析の実行（引数なしで呼び出し）
                multiframe_rci = self._multiframe_analyzer.analyze_streaming()
                
                # メトリクス更新...（既存コード維持）
            else:
                self._logger.debug(
                    f"Insufficient history for multi-timeframe analysis: "
                    f"{self._multiframe_analyzer.get_buffer_size()} bars collected"
                )
```

## 3. テストケースの準備

Step 6-7で作成するテストケースのアウトライン:

```python
# tests/unit/test_analyzer.py

def test_buffer_management():
    """バッファ管理機能のテスト"""
    analyzer = MultiTimeframeAnalyzer(config)
    
    # バーを追加
    for i in range(250):
        analyzer.add_new_bar(create_test_bar(i))
    
    # バッファサイズの確認
    assert analyzer.get_buffer_size() == 250
    assert analyzer.is_ready() == True
    
def test_buffer_size_limit():
    """バッファサイズ制限のテスト"""
    analyzer = MultiTimeframeAnalyzer(config)
    analyzer._max_history_bars = 100
    
    # 制限を超える数のバーを追加
    for i in range(150):
        analyzer.add_new_bar(create_test_bar(i))
    
    # バッファが制限内に収まることを確認
    assert analyzer.get_buffer_size() == 100
```

## 実装順序

1. **Step 3で実行**: MultiTimeframeAnalyzerにバッファ管理メソッドを追加
2. **Step 4で実行**: RealtimePipelineから該当コードを削除し、新しいインターフェースを使用
3. **Step 5で実行**: 依存性注入の改善とインターフェース調整
4. **Step 6-7で実行**: テストケースの実装

## 注意事項

- configからのパラメータ取得方法を統一する
- 後方互換性を保つため、analyze_streamingメソッドは引数なしでも動作するようにする
- エラーハンドリングを適切に実装する
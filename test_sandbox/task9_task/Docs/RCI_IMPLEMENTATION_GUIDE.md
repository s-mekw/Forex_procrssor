# RCIリアルタイム実装ガイド

## 概要

本ドキュメントは、RCI（Rank Correlation Index）のリアルタイム計算実装で得られた知見をまとめ、本番フロントエンド開発の参考資料として作成されました。

## 1. アーキテクチャ設計

### 1.1 コンポーネント構成

```
┌─────────────────────────────────────────────────────┐
│                  フロントエンド                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │   Dash App  │  │ Chart Display │  │Statistics│  │
│  └──────┬──────┘  └──────┬───────┘  └─────┬────┘  │
└─────────┼─────────────────┼────────────────┼───────┘
          │                 │                │
┌─────────▼─────────────────▼────────────────▼───────┐
│                   データ処理層                       │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────┐  │
│  │RCI Calculator│  │Tick Converter│  │EMA Engine│  │
│  └──────────────┘  └─────────────┘  └──────────┘  │
└─────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────┐
│                  データソース (MT5)                   │
└─────────────────────────────────────────────────────┘
```

### 1.2 データフロー

1. **初期データ取得**: MT5から過去データを取得
2. **リアルタイムストリーミング**: ティックデータを継続的に受信
3. **バー変換**: ティックデータをOHLCバーに変換
4. **インジケーター計算**: RCI、EMAなどを計算
5. **表示更新**: チャートとダッシュボードを更新

## 2. 核心技術：DifferentialRCICalculator

### 2.1 設計思想

```python
class DifferentialRCICalculator:
    """
    差分更新アルゴリズムによる高速RCI計算
    
    特徴:
    - O(n)の計算複雑度
    - メモリ効率的なスライディングウィンドウ
    - 内部状態の一貫性保証
    """
```

### 2.2 重要メソッド

#### add() - 完成バーの追加
```python
def add(self, price: float) -> Optional[float]:
    """
    新しい完成バーを追加してRCIを計算
    - 内部状態（prices deque）を更新
    - ウィンドウサイズを維持（最古のデータを削除）
    """
```

#### preview() - 未完成バーの一時計算
```python
def preview(self, temp_price: float) -> Optional[float]:
    """
    未完成バーの一時的なRCI値を計算
    - 内部状態は変更しない
    - 最古のバーを削除して最新のperiod個で計算
    """
```

### 2.3 状態管理のベストプラクティス

```python
# 良い例：明確な状態分離
self.prices = deque(maxlen=period)  # 完成バーのみ
temp_rci = calculator.preview(current_price)  # 未完成バー

# 悪い例：状態の混在
all_prices.append(incomplete_price)  # 完成/未完成が混在
```

## 3. リアルタイムデータ処理パターン

### 3.1 未完成バーと完成バーの管理

```python
class RealtimeChartManager:
    def __init__(self):
        # データを明確に分離
        self.completed_bars = []      # 完成バーのみ
        self.current_bar = None        # 未完成バー
        self.temp_indicator_values = {}  # 一時的な値
        self.has_incomplete_bar = False  # 未完成バーの状態フラグ
```

**重要な実装パターン：未完成バーのRCI更新**

```python
def update_rci_incremental(self):
    """未完成バー用のRCI更新（修正版）"""
    if self.current_bar is None:
        return
    
    current_close = float(self.current_bar.close)
    
    for period in self.rci_periods:
        # preview計算（内部状態は変更しない）
        temp_rci = self.rci_calculators[period].preview(current_close)
        
        if temp_rci is not None:
            self.temp_rci_values[period] = temp_rci
            
            # 重要：rci_data配列の最後の値を直接更新
            # これによりデータ長の整合性を保つ
            if self.has_incomplete_bar and len(self.rci_data[period]) > 0:
                self.rci_data[period][-1] = temp_rci
```

### 3.2 データ更新フロー

```python
def process_tick(self, tick):
    """ティック処理の標準フロー"""
    # 1. バー変換
    bar = self.converter.add_tick(tick)
    
    if bar:  # バーが完成
        # 2. 完成バーを正式に追加
        self.add_completed_bar(bar)
    else:
        # 3. 未完成バーを更新
        self.update_current_bar()
        
    # 4. インジケーターを更新
    self.update_indicators()
```

### 3.3 初期データの処理

```python
def initialize_data(self, historical_data):
    """初期データ処理の重要ポイント"""
    # MT5から取得した最後のバーは未完成
    completed_bars = historical_data[:-1]
    incomplete_bar = historical_data[-1]
    
    # 完成バーのみを処理
    for bar in completed_bars:
        self.calculator.add(bar.close)
    
    # 未完成バーは別途処理
    self.current_bar_rci = self.calculator.preview(incomplete_bar.close)
```

## 4. パフォーマンス最適化

### 4.1 メモリ管理

```python
# dequeによる効率的なウィンドウ管理
self.prices = deque(maxlen=period)

# Float32精度でメモリ使用量を削減
prices_array = np.array(prices, dtype=np.float32)

# キャッシュによる計算削減
self._ranking_cache = {}  # 頻出パターンをキャッシュ
```

### 4.2 計算最適化

```python
# 事前計算による高速化
self.time_ranks = np.arange(period, dtype=np.float32)
self.denominator = np.float32(period * (period**2 - 1))

# 条件分岐による最適化
if self._check_for_ties(prices):
    ranks = self._optimized_ranking(prices)  # 同値対応
else:
    ranks = self._fast_ranking_no_ties(prices)  # 高速版
```

## 5. フロントエンド統合

### 5.1 Dashアプリケーション構成

```python
# データ更新の分離
app.layout = serve_layout  # 動的レイアウト生成

@app.callback(
    Output('live-chart', 'figure'),
    Input('interval-component', 'n_intervals'),
    State('realtime-status', 'data')
)
def update_chart(n, status):
    """チャート更新ロジック"""
    with chart_manager.data_lock:  # スレッドセーフ
        return chart_manager.create_chart()
```

### 5.2 リアルタイム表示の工夫

```python
# Plotlyの設定
fig.update_layout(
    hovermode='x unified',  # 統一ホバー表示
    xaxis_rangeslider_visible=False,  # パフォーマンス向上
)

# スムージング（line_shape='spline'の代替）
# 未完成バーのpreview値により自然な曲線を実現
```

## 6. エラーハンドリング

### 6.1 データ検証

```python
def validate_and_sync_data(self, ohlc_data, indicator_data):
    """データ整合性の確保"""
    # NaN/Inf値のチェック
    if not np.isfinite(price):
        return None
    
    # データ長の同期
    if len(indicator_data) != len(ohlc_data):
        self.resync_data()
```

**重要：_validate_and_sync_dataの正しい実装**

```python
def _validate_and_sync_data(self, ohlc_data, ema_data, rci_data, temp_ema_values, temp_rci_values):
    """チャート用データの検証と同期（修正版）"""
    ohlc_length = len(ohlc_data)
    
    # RCIデータの検証（temp値の追加処理は削除）
    for period in self.rci_periods:
        if period in rci_data:
            # 注意：temp_rci_valuesを追加してはいけない
            # update_rci_incrementalで既に更新済み
            
            # データ長の調整のみ行う
            if len(rci_data[period]) > ohlc_length:
                rci_data[period] = rci_data[period][:ohlc_length]
            elif len(rci_data[period]) < ohlc_length:
                # 不足分はNoneで埋める
                missing_count = ohlc_length - len(rci_data[period])
                rci_data[period].extend([None] * missing_count)
    
    return ohlc_data, ema_data, rci_data
```

**データ同期チェックの実装**

```python
def check_data_sync(self):
    """データ同期状態の確認"""
    ohlc_len = len(self.ohlc_data)
    
    for period in self.rci_periods:
        rci_len = len(self.rci_data[period])
        if ohlc_len != rci_len:
            print(f"[SYNC ERROR] Period {period}: OHLC={ohlc_len}, RCI={rci_len}")
            return False
    
    return True
```

### 6.2 リカバリー戦略

```python
try:
    rci_value = calculator.add(price)
except Exception as e:
    logger.error(f"RCI calculation failed: {e}")
    # フォールバック処理
    calculator.reset()
    self.reinitialize_from_checkpoint()
```

## 7. テスト戦略

### 7.1 ユニットテスト

```python
def test_incomplete_bar_handling():
    """未完成バー処理のテスト"""
    calculator = DifferentialRCICalculator(period=9)
    
    # 完成バーを追加
    for price in initial_prices[:-1]:
        calculator.add(price)
    
    # 未完成バーのpreview
    preview_rci = calculator.preview(current_price)
    
    # バー完成時の処理
    final_rci = calculator.add(final_price)
    
    assert preview_rci != final_rci  # 異なる計算ロジック
```

### 7.2 統合テスト

```python
def test_mt5_compatibility():
    """MT5との互換性テスト"""
    # MT5の結果と比較
    mt5_rci = get_mt5_rci_values()
    our_rci = calculate_our_rci()
    
    assert np.allclose(mt5_rci, our_rci, rtol=1e-5)
```

## 8. 本番デプロイメント考慮事項

### 8.1 スケーラビリティ

- **マルチシンボル対応**: 各シンボルに独立したCalculatorインスタンス
- **並列処理**: ThreadPoolExecutorによる複数期間の並列計算
- **メモリ管理**: 古いデータの自動削除とガベージコレクション

### 8.2 モニタリング

```python
def get_statistics(self):
    """パフォーマンス統計"""
    return {
        'calculation_count': self._calculation_count,
        'cache_hit_rate': self._cache_hits / total_attempts,
        'memory_usage': self._process.memory_info().rss
    }
```

### 8.3 設定管理

```toml
# config.toml
[rci]
periods_short = [9, 13]
periods_long = [26, 33]
use_float32 = true

[chart]
initial_bars = 200
timeframe = "M1"
```

## 9. まとめ

### 成功要因

1. **明確な状態管理**: 完成バーと未完成バーの分離
2. **効率的なアルゴリズム**: 差分更新による計算最適化
3. **MT5互換性**: 既存システムとの完全な互換性確保
4. **テスタビリティ**: 単体・統合テストの充実

### 推奨事項

1. **段階的な実装**: コア機能から順次実装
2. **継続的なテスト**: 各段階でMT5との比較検証
3. **パフォーマンス監視**: メモリ使用量と計算時間の追跡
4. **ドキュメント**: 実装の意図と制約を明文化

## 10. 参考実装

完全な実装例は以下のファイルを参照：

- `src/data_processing/rci.py` - DifferentialRCICalculator
- `test_sandbox/task9_task/rci_realtime_chart.py` - リアルタイムチャート実装
- `test_sandbox/task9_task/test_incomplete_bar.py` - テストケース

## 11. よくある落とし穴と解決策

### 11.1 RCIグラフの不連続性問題

**問題の症状**
- RCIグラフが途切れて表示される
- データ更新時にグラフが不自然にジャンプする

**原因**
temp_rci_valuesの二重追加によるデータ長の不整合

```python
# 問題のあるコード
def _validate_and_sync_data(self, ...):
    # NG: temp値を毎回追加してしまう
    if period in temp_rci_values and ohlc_length > rci_length:
        rci_data[period] = rci_data[period] + [temp_rci_values[period]]
```

**解決策**
update_rci_incrementalで直接rci_data配列を更新

```python
# 正しい実装
def update_rci_incremental(self):
    temp_rci = self.rci_calculators[period].preview(current_close)
    if self.has_incomplete_bar and len(self.rci_data[period]) > 0:
        # 最後の値を直接更新（追加ではない）
        self.rci_data[period][-1] = temp_rci
```

### 11.2 新しい未完成バー追加時の同期ずれ

**問題の症状**
- 新しいバーが始まったときにRCIデータが欠落
- OHLCデータとRCIデータの長さが一致しない

**原因**
update_current_bar_in_ohlc()でRCIデータの追加が漏れている

```python
# 問題のあるコード
def update_current_bar_in_ohlc(self):
    if last_time != current_bar_time:
        # OHLCデータは追加するが、RCIデータは追加していない
        self.ohlc_data = pl.concat([self.ohlc_data, new_row])
```

**解決策**
新しい未完成バーの追加時にRCIデータも同時に追加

```python
# 正しい実装
def update_current_bar_in_ohlc(self):
    if last_time != current_bar_time:
        # OHLCデータを追加
        self.ohlc_data = pl.concat([self.ohlc_data, new_row])
        
        # RCIデータも追加
        for period in self.rci_periods:
            preview_rci = self.rci_calculators[period].preview(current_close)
            self.rci_data[period].append(preview_rci)
```

### 11.3 初期データ処理での未完成バー誤認

**問題の症状**
- MT5から取得した初期データで、最後のバーのRCI値が不正確

**原因**
MT5の最新バーは常に未完成バーだが、完成バーとして処理してしまう

```python
# 問題のあるコード
for price in close_prices:  # 全て完成バーとして処理
    rci_value = calculator.add(price)
```

**解決策**
最後のバーを未完成バーとして処理

```python
# 正しい実装
# 最後のバーを除いて完成バーとして処理
for price in close_prices[:-1]:
    rci_value = calculator.add(price)

# 最後のバーは未完成バーとして処理
if len(close_prices) > 0:
    preview_rci = calculator.preview(close_prices[-1])
    rci_data.append(preview_rci)
```

### 11.4 デバッグのベストプラクティス

**データ同期の監視**

```python
# デバッグログの追加
if period == 9:  # 特定期間のみログ出力
    print(f"[RCI Update] Period {period}: Preview={temp_rci:.2f}, "
          f"OHLC len={len(self.ohlc_data)}, RCI len={len(self.rci_data[period])}")
```

**定期的な同期チェック**

```python
def periodic_sync_check(self):
    """定期的にデータ同期をチェック"""
    if time.time() - self.last_check >= 5:  # 5秒ごと
        if not self.check_data_sync():
            self.log_sync_error()
        self.last_check = time.time()
```

---

*Last Updated: 2025-01-26*
*Version: 1.1.0*
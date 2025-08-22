# バッファオーバーフロー問題の根本解決実装計画書

## 1. 問題の概要

### 現状の問題
- **症状**: TickDataStreamerのバッファ（サイズ5000）がオーバーフローし、9000件以上のティックがドロップ
- **エラーログ**: `Buffer full - dropping ticks buffer_size=5000 dropped_count=9320`

### 根本原因
1. **プッシュ型アーキテクチャの限界**
   - `stream_ticks()`が3ms間隔で継続的にティックを生成
   - 消費側の処理速度に関係なく、自動的にバッファに追加
   
2. **速度のミスマッチ**
   - 生成速度: 約333ティック/秒（3ms間隔）
   - 消費速度: テストでは100ティック/秒未満
   - 結果: 生成速度 > 消費速度でバッファが溢れる

3. **バックプレッシャーの不完全性**
   - 現在の実装は警告とスリープのみ
   - 生成自体を止める機構がない

## 2. 提案する解決策

### 方法A: アダプティブストリーミング（推奨）

#### 概要
バッファ使用率に基づいて、動的に生成速度を調整する仕組み

#### 実装内容
```python
# src/mt5_data_acquisition/tick_fetcher.py

class TickDataStreamer:
    def __init__(self, ...):
        # 新規パラメータ
        self.adaptive_streaming = True
        self.min_stream_delay = 0.001  # 1ms
        self.max_stream_delay = 0.05   # 50ms
        
    async def stream_ticks(self):
        while self.is_streaming:
            # バッファ使用率に基づく動的待機時間
            delay = self._calculate_adaptive_delay()
            await asyncio.sleep(delay)
            
    def _calculate_adaptive_delay(self) -> float:
        """バッファ使用率に基づいて待機時間を計算"""
        usage = self.buffer_usage
        
        if usage < 0.3:  # 30%未満
            return self.min_stream_delay  # 1ms（高速）
        elif usage < 0.5:  # 50%未満  
            return 0.003  # 3ms（標準）
        elif usage < 0.8:  # 80%未満
            return 0.01   # 10ms（中速）
        else:  # 80%以上
            # 線形補間で20-50ms
            return 0.02 + (usage - 0.8) * 0.15
```

### 方法B: 非同期キューベースアーキテクチャ

#### 概要
`asyncio.Queue`を使用した生産者-消費者パターン

#### 実装内容
```python
class TickDataStreamer:
    def __init__(self, buffer_size: int = 10000):
        # dequeからasyncio.Queueへ移行
        self.buffer = asyncio.Queue(maxsize=buffer_size)
        
    async def _producer_task(self):
        """ティック生成タスク"""
        while self.is_streaming:
            tick = self._fetch_latest_tick()
            if tick:
                # キューがフルの場合、自動的にブロック
                await self.buffer.put(tick)
            await asyncio.sleep(0.003)
            
    async def get_new_ticks(self) -> list[Tick]:
        """非ブロッキングで利用可能なティックを全て取得"""
        ticks = []
        while not self.buffer.empty():
            try:
                tick = self.buffer.get_nowait()
                ticks.append(tick)
            except asyncio.QueueEmpty:
                break
        return ticks
```

## 3. 実装手順

### Phase 1: アダプティブストリーミングの実装
1. `TickDataStreamer`クラスに動的待機時間計算メソッドを追加
2. `stream_ticks()`メソッドを改修して動的待機を適用
3. 設定パラメータを追加（adaptive_streaming, min/max_delay）

### Phase 2: モニタリング機能の追加
1. バッファ使用率のリアルタイム監視
2. 生成速度と消費速度のメトリクス収集
3. パフォーマンス統計のログ出力

### Phase 3: テストと検証
1. 単体テストの作成
2. 負荷テストの実施
3. パフォーマンス比較（改善前後）

## 4. 設定パラメータ

### 新規追加パラメータ
| パラメータ | 型 | デフォルト値 | 説明 |
|---------|---|----------|------|
| `adaptive_streaming` | bool | True | アダプティブモード有効化 |
| `min_stream_delay` | float | 0.001 | 最小待機時間（秒） |
| `max_stream_delay` | float | 0.05 | 最大待機時間（秒） |
| `target_buffer_usage` | float | 0.5 | 目標バッファ使用率 |
| `backpressure_mode` | str | "adaptive" | "adaptive" \| "fixed" \| "off" |

## 5. 期待される効果

### パフォーマンス改善
- **データ損失**: 9000+件のドロップ → 0件
- **バッファ使用率**: 常に100%（オーバーフロー） → 50-80%で安定
- **処理遅延**: 不安定 → 10ms以内で安定

### システムへの影響
- **CPU使用率**: 高負荷時のみ高速処理、低負荷時は省エネ
- **メモリ使用量**: バッファサイズに応じた固定量
- **後方互換性**: 既存APIは完全互換

## 6. リスクと対策

### リスク
1. **過度の速度制限**: 消費が追いつかない場合、生成が遅くなりすぎる
2. **レイテンシ増加**: 動的調整による遅延

### 対策
1. **最小速度の保証**: min_stream_delayで下限を設定
2. **モニタリング**: メトリクスで常時監視
3. **フォールバック**: adaptive_streaming=Falseで従来動作

## 7. 暫定対応（実施済み）

テストスクリプト側での暫定対応として以下を実施済み：

### 実施内容
1. **バッファサイズ拡張**: 5,000 → 20,000（4倍）
2. **バックプレッシャー閾値調整**: 90% → 95%
3. **処理速度最適化**: 
   - 基本待機時間: 10ms → 1ms
   - 動的待機時間調整の実装

### 結果
- バッファオーバーフロー: 9000+件 → 0件
- すべてのティックを正常に処理

## 8. まとめ

本実装計画により、TickDataStreamerのバッファオーバーフロー問題を根本的に解決できます。アダプティブストリーミングにより、消費側の処理速度に自動的に適応し、データ損失を防ぎながら最適なパフォーマンスを実現します。

実装は段階的に行い、各フェーズでテストを実施することで、安定性を確保しながら改善を進めます。

---

作成日: 2025-08-21  
作成者: Claude Code Assistant  
バージョン: 1.0
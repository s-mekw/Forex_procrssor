"""
マルチタイムフレーム機能の動作確認テストスクリプト
"""

import asyncio
import polars as pl
from datetime import datetime, timedelta
import numpy as np
from src.data_processing.pipelines import RealtimePipeline, DataPoint

async def test_multiframe_batch():
    """バッチデータでマルチタイムフレーム機能をテスト"""
    
    # パイプラインを有効化されたマルチタイムフレーム機能で初期化
    pipeline = RealtimePipeline(
        enable_multiframe=True,
        multiframe_config={
            "short_term_periods": [9, 13, 24],
            "long_term_periods": [24, 33],
            "long_timeframe": "5T",
        },
        max_history_bars=300
    )
    
    await pipeline.start()
    
    print("=== マルチタイムフレーム統合テスト ===")
    print(f"Config: {pipeline.get_multiframe_config()}")
    
    # テストデータを生成（300本の1分足データ）
    base_time = datetime.now().replace(second=0, microsecond=0) - timedelta(minutes=300)
    prices = 100 + np.cumsum(np.random.randn(300) * 0.1)  # ランダムウォーク
    
    print(f"\nGenerating {len(prices)} bars of test data...")
    
    # データを順次送信
    for i in range(len(prices)):
        data_point: DataPoint = {
            "timestamp": base_time + timedelta(minutes=i),
            "data": {
                "open": float(prices[max(0, i-1)] if i > 0 else prices[i]),
                "high": float(prices[i] + abs(np.random.randn() * 0.05)),
                "low": float(prices[i] - abs(np.random.randn() * 0.05)),
                "close": float(prices[i]),
                "volume": float(1000 + np.random.randn() * 100),
            },
            "metadata": {"bar_index": i}
        }
        
        await pipeline.submit(data_point)
        
        # 5分ごとに進捗を表示
        if (i + 1) % 5 == 0:
            print(f"  Submitted {i + 1}/{len(prices)} bars...")
    
    # 少し待つ
    await asyncio.sleep(1)
    
    # 結果を確認
    print("\n=== 結果の確認 ===")
    
    # 最後の結果を取得
    results_count = 0
    multiframe_results = []
    
    while True:
        try:
            result = await asyncio.wait_for(pipeline.get_result(), timeout=0.1)
            results_count += 1
            
            if result.get("multiframe_rci"):
                multiframe_results.append(result)
                
        except asyncio.TimeoutError:
            break
    
    print(f"Total results received: {results_count}")
    print(f"Results with multiframe RCI: {len(multiframe_results)}")
    
    # 最後のマルチタイムフレームRCI結果を表示
    if multiframe_results:
        last_result = multiframe_results[-1]
        multiframe_rci = last_result["multiframe_rci"]
        
        print(f"\n=== 最後のRCI値 ===")
        print(f"Timestamp: {multiframe_rci['timestamp']}")
        print(f"Short-term RCI (1min): {multiframe_rci['short_rci']}")
        print(f"Long-term RCI (5min): {multiframe_rci['long_rci']}")
        print(f"New 5-min bar: {multiframe_rci['is_new_long_bar']}")
    
    # メトリクスを表示
    metrics = pipeline.get_metrics()
    print(f"\n=== パイプラインメトリクス ===")
    print(f"Processed count: {metrics['processed_count']}")
    print(f"Average latency: {metrics['avg_latency']*1000:.2f}ms")
    print(f"Multiframe processing count: {metrics['multiframe_processing_count']}")
    print(f"Multiframe avg latency: {metrics.get('multiframe_avg_latency', 0)*1000:.2f}ms")
    print(f"Data buffer size: {metrics.get('data_buffer_size', 0)} bars")
    
    await pipeline.stop()
    print("\n✅ テスト完了")

async def test_multiframe_streaming():
    """ストリーミングモードでマルチタイムフレーム機能をテスト"""
    
    pipeline = RealtimePipeline(
        enable_multiframe=True,
        multiframe_config={
            "short_term_periods": [9, 13],
            "long_term_periods": [24],
        },
        max_history_bars=100
    )
    
    await pipeline.start()
    
    print("\n=== ストリーミングモードテスト ===")
    
    # リアルタイムシミュレーション
    base_time = datetime.now().replace(second=0, microsecond=0)
    base_price = 100.0
    
    for i in range(20):
        # 価格を少し動かす
        price_change = np.random.randn() * 0.1
        current_price = base_price + price_change
        
        data_point: DataPoint = {
            "timestamp": base_time + timedelta(minutes=i),
            "data": {
                "open": float(base_price),
                "high": float(max(base_price, current_price) + 0.05),
                "low": float(min(base_price, current_price) - 0.05),
                "close": float(current_price),
                "volume": 1000.0,
            },
            "metadata": None
        }
        
        await pipeline.submit(data_point)
        base_price = current_price
        
        # 5分ごとの区切りでログを出力
        if (i + 1) % 5 == 0:
            print(f"  Streamed {i + 1} bars - checking for 5-min completion...")
    
    await asyncio.sleep(0.5)
    
    # 結果を確認
    five_min_bars = 0
    while True:
        try:
            result = await asyncio.wait_for(pipeline.get_result(), timeout=0.1)
            if result.get("multiframe_rci") and result["multiframe_rci"].get("is_new_long_bar"):
                five_min_bars += 1
        except asyncio.TimeoutError:
            break
    
    print(f"Detected {five_min_bars} completed 5-minute bars")
    
    await pipeline.stop()
    print("✅ ストリーミングテスト完了")

if __name__ == "__main__":
    asyncio.run(test_multiframe_batch())
    asyncio.run(test_multiframe_streaming())
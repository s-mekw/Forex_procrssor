import asyncio
import logging
from datetime import datetime, timedelta
from src.data_processing.pipelines import RealtimePipeline, DataPoint

# Configure logging to see warnings
logging.basicConfig(level=logging.WARNING)

async def test_latency_alert():
    """遅延アラート機能のテスト"""
    pipeline = RealtimePipeline(
        queue_size=100,
        alert_threshold=1.0,
        enable_metrics=True
    )
    
    await pipeline.start()
    
    try:
        # 2秒前のタイムスタンプでデータポイントを作成（遅延をシミュレート）
        old_time = datetime.now() - timedelta(seconds=2)
        
        data_point: DataPoint = {
            'timestamp': old_time,
            'data': {'test': 'data'},
            'metadata': None
        }
        
        # データを送信
        await pipeline.submit(data_point)
        
        # 少し待機
        await asyncio.sleep(0.5)
        
        # 結果を取得
        result = await pipeline.get_result()
        
        # メトリクスを確認
        metrics = pipeline.get_metrics()
        
        print(f"Latency: {result['latency']:.3f}s")
        print(f"Alert count: {metrics['alert_count']}")
        print(f"Result status: {result['status']}")
        
        assert result['latency'] > 1.0, "Latency should be > 1 second"
        assert metrics['alert_count'] == 1, "Should have triggered 1 alert"
        assert result['status'] == 'success'
        
        print("✅ Latency alert test passed!")
        
    finally:
        await pipeline.stop()

if __name__ == "__main__":
    asyncio.run(test_latency_alert())

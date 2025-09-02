import asyncio
from datetime import datetime
from src.data_processing.pipelines import RealtimePipeline, DataPoint

async def producer(pipeline, producer_id, num_items=10):
    """データを生成してパイプラインに送信"""
    for i in range(num_items):
        data_point: DataPoint = {
            'timestamp': datetime.now(),
            'data': {
                'producer': producer_id,
                'item': i,
                'value': f'data_{producer_id}_{i}'
            },
            'metadata': {'source': f'producer_{producer_id}'}
        }
        await pipeline.submit(data_point)
        await asyncio.sleep(0.01)  # 少し待機

async def consumer(pipeline, num_items):
    """パイプラインから結果を取得"""
    results = []
    for _ in range(num_items):
        result = await pipeline.get_result()
        results.append(result)
    return results

async def test_concurrent():
    """並行処理のテスト"""
    pipeline = RealtimePipeline(
        queue_size=100,
        alert_threshold=1.0,
        enable_metrics=True
    )
    
    await pipeline.start()
    
    try:
        # 3つのプロデューサーを同時に起動
        num_producers = 3
        items_per_producer = 5
        total_items = num_producers * items_per_producer
        
        # プロデューサータスクを作成
        producer_tasks = [
            asyncio.create_task(producer(pipeline, i, items_per_producer))
            for i in range(num_producers)
        ]
        
        # コンシューマータスクを作成
        consumer_task = asyncio.create_task(consumer(pipeline, total_items))
        
        # 全プロデューサーの完了を待つ
        await asyncio.gather(*producer_tasks)
        
        # コンシューマーの完了を待つ
        results = await consumer_task
        
        # 検証
        assert len(results) == total_items, f"Expected {total_items} results, got {len(results)}"
        
        # 各プロデューサーからのデータが含まれているか確認
        producer_counts = {}
        for result in results:
            prod_id = result['processed_data']['producer']
            producer_counts[prod_id] = producer_counts.get(prod_id, 0) + 1
        
        assert len(producer_counts) == num_producers, f"Expected data from {num_producers} producers"
        
        for prod_id, count in producer_counts.items():
            assert count == items_per_producer, f"Producer {prod_id} should have {items_per_producer} items"
        
        # メトリクスを確認
        metrics = pipeline.get_metrics()
        assert metrics['processed_count'] == total_items
        assert metrics['alert_count'] == 0  # 遅延はないはず
        
        print(f"✅ Concurrent processing test passed!")
        print(f"  Total items processed: {total_items}")
        print(f"  Producer distribution: {producer_counts}")
        print(f"  Average latency: {metrics['avg_latency']:.6f}s")
        
    finally:
        await pipeline.stop()

if __name__ == "__main__":
    asyncio.run(test_concurrent())

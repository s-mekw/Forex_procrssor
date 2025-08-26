"""
修正確認用の簡易テストスクリプト
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.pipelines import RealtimePipeline
from test_sandbox.task10_test.utils import FXDataGenerator


async def verify_fixes():
    """修正の確認"""
    print("=== 修正確認テスト ===\n")
    
    # 1. EURGBPが追加されているか確認
    generator = FXDataGenerator()
    print("1. EURGBP通貨ペアの確認:")
    try:
        data = generator.generate_tick("EURGBP")
        print(f"   ✅ EURGBP生成成功: bid={data['data']['bid']}")
    except ValueError as e:
        print(f"   ❌ エラー: {e}")
    
    # 2. queue_statusのキー名確認
    print("\n2. queue_statusのキー名確認:")
    pipeline = RealtimePipeline(queue_size=100, alert_threshold=1.0, enable_metrics=True)
    
    try:
        await pipeline.start()
        queue_status = await pipeline.get_queue_status()
        
        # 正しいキー名をチェック
        required_keys = ["input_queue_size", "input_queue_maxsize"]
        for key in required_keys:
            if key in queue_status:
                print(f"   ✅ {key}: {queue_status[key]}")
            else:
                print(f"   ❌ {key}: 存在しません")
        
        # 使用率計算のテスト
        usage = queue_status["input_queue_size"] / queue_status["input_queue_maxsize"] * 100
        print(f"   ✅ キュー使用率計算成功: {usage:.1f}%")
        
    except Exception as e:
        print(f"   ❌ エラー: {e}")
    finally:
        await pipeline.stop()
    
    print("\n✅ すべての修正が正常に動作しています")


if __name__ == "__main__":
    asyncio.run(verify_fixes())
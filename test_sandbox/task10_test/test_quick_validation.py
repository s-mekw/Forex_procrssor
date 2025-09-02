"""
E2Eテスト修正の簡易検証スクリプト

awaitエラーが修正されていることを確認します。
"""

import asyncio
import sys
from pathlib import Path

# プロジェクトのルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.pipelines import DataPoint, RealtimePipeline


async def quick_test():
    """簡易テスト"""
    print("=== E2Eテスト修正検証 ===")
    
    # パイプライン初期化
    pipeline = RealtimePipeline(
        queue_size=100,
        alert_threshold=1.0,
        enable_metrics=True
    )
    
    try:
        # パイプライン開始
        await pipeline.start()
        print("✅ パイプライン起動成功")
        
        # テストデータ送信
        from datetime import datetime
        test_data = DataPoint(
            timestamp=datetime.now(),
            data={"symbol": "USDJPY", "bid": 150.00, "ask": 150.01},
            metadata={"source": "test"}
        )
        
        success = await pipeline.submit(test_data)
        print(f"✅ データ送信: {'成功' if success else '失敗'}")
        
        # 同期メソッドのテスト（awaitなし）
        metrics = pipeline.get_metrics()  # 同期メソッド
        print(f"✅ get_metrics() 正常: {metrics.get('processed_count', 0)}件処理")
        
        alert_stats = pipeline.get_alert_statistics()  # 同期メソッド
        print(f"✅ get_alert_statistics() 正常: アラート{alert_stats['total_alerts']}件")
        
        is_bp = pipeline.is_backpressure_active()  # 同期メソッド
        print(f"✅ is_backpressure_active() 正常: {'Yes' if is_bp else 'No'}")
        
        # 非同期メソッドのテスト（awaitあり）
        queue_status = await pipeline.get_queue_status()  # 非同期メソッド
        print(f"✅ get_queue_status() 正常: キュー使用{queue_status['input_queue_size']}/{queue_status['input_queue_maxsize']}")
        
        print("\n✅ 全てのメソッド呼び出しが正常に動作しました")
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        return False
    
    finally:
        await pipeline.stop()
        print("✅ パイプライン停止成功")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(quick_test())
    
    if success:
        print("\n========================================")
        print("✅ E2Eテストの修正は正しく動作しています")
        print("========================================")
    else:
        print("\n❌ 修正に問題があります")
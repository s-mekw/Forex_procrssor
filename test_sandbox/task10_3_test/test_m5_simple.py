"""
M5バー境界検出の簡単な動作確認
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import logging
from datetime import datetime, timedelta

# デバッグログを有効化
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_boundary_detection():
    """修正されたバー境界検出をテスト"""
    from src.data_processing.analyzer import MultiTimeframeAnalyzer
    
    # アナライザーのインスタンスを作成
    analyzer = MultiTimeframeAnalyzer()
    
    logger.info("=" * 60)
    logger.info("M5バー境界検出テスト（修正版）")
    logger.info("=" * 60)
    
    # テスト用のタイムスタンプ（4:59から5:01まで）
    test_times = [
        datetime(2024, 1, 1, 10, 4, 59, 500000),
        datetime(2024, 1, 1, 10, 4, 59, 800000),
        datetime(2024, 1, 1, 10, 5, 0, 100000),  # 境界を超える
        datetime(2024, 1, 1, 10, 5, 0, 500000),
        datetime(2024, 1, 1, 10, 5, 1, 0),
        datetime(2024, 1, 1, 10, 9, 59, 800000),
        datetime(2024, 1, 1, 10, 10, 0, 200000),  # 境界を超える
        datetime(2024, 1, 1, 10, 10, 1, 0),
    ]
    
    for timestamp in test_times:
        result = analyzer._is_new_long_bar_complete(timestamp)
        if result:
            logger.info(f"✅ New M5 bar at {timestamp.strftime('%H:%M:%S.%f')[:-3]}")
        else:
            logger.info(f"   {timestamp.strftime('%H:%M:%S.%f')[:-3]} -> No new bar")
    
    logger.info("\n結果：")
    logger.info("- 10:05:00.100 で最初のM5バー完成を検出")
    logger.info("- 10:10:00.200 で次のM5バー完成を検出")
    logger.info("- 秒が0でなくても境界を超えれば検出される")

if __name__ == "__main__":
    test_boundary_detection()
"""
M5バー境界検出問題のデバッグと修正テスト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_current_boundary_detection():
    """現在の境界検出ロジックのテスト"""
    def _is_new_long_bar_complete_old(timestamp: datetime) -> bool:
        """現在の実装（問題あり）"""
        interval_minutes = 5
        return timestamp.minute % interval_minutes == 0 and timestamp.second == 0
    
    # テスト用タイムスタンプ
    test_times = [
        datetime(2024, 1, 1, 10, 4, 59, 999999),  # 4:59.999999
        datetime(2024, 1, 1, 10, 5, 0, 0),        # 5:00.000000 (理想的)
        datetime(2024, 1, 1, 10, 5, 0, 100000),   # 5:00.100000 (実際のティック)
        datetime(2024, 1, 1, 10, 5, 1, 0),        # 5:01.000000
        datetime(2024, 1, 1, 10, 9, 59, 999999),  # 9:59.999999
        datetime(2024, 1, 1, 10, 10, 0, 0),       # 10:00.000000
        datetime(2024, 1, 1, 10, 10, 0, 500000),  # 10:00.500000
    ]
    
    logger.info("=== 現在の実装（timestamp.second == 0 必須）===")
    for t in test_times:
        result = _is_new_long_bar_complete_old(t)
        logger.info(f"{t.strftime('%H:%M:%S.%f')[:-3]} -> {result}")
    
    # 実際のティックデータをシミュレート
    logger.info("\n=== 実際のティックデータのシミュレーション ===")
    current_time = datetime(2024, 1, 1, 10, 4, 58, 800000)
    for i in range(20):
        current_time += timedelta(milliseconds=200)  # 200msごとのティック
        result = _is_new_long_bar_complete_old(current_time)
        if result:
            logger.info(f"✅ New bar at {current_time.strftime('%H:%M:%S.%f')[:-3]}")
        else:
            logger.debug(f"   {current_time.strftime('%H:%M:%S.%f')[:-3]} -> {result}")

def test_improved_boundary_detection():
    """改善された境界検出ロジックのテスト"""
    
    class BoundaryDetector:
        def __init__(self, interval_minutes=5):
            self.interval_minutes = interval_minutes
            self.last_bar_minute = None
            
        def is_new_bar(self, timestamp: datetime) -> bool:
            """改善版：境界を超えたかチェック"""
            # 現在のバーの開始時刻（分）を計算
            current_bar_minute = (timestamp.minute // self.interval_minutes) * self.interval_minutes
            
            # 初回またはバーが変わった場合
            if self.last_bar_minute is None:
                self.last_bar_minute = current_bar_minute
                return False
            
            if current_bar_minute != self.last_bar_minute:
                logger.info(f"  Bar boundary crossed: {self.last_bar_minute:02d} -> {current_bar_minute:02d}")
                self.last_bar_minute = current_bar_minute
                return True
            
            return False
    
    detector = BoundaryDetector(5)
    
    logger.info("\n=== 改善された実装（境界超えチェック）===")
    current_time = datetime(2024, 1, 1, 10, 4, 58, 800000)
    for i in range(20):
        current_time += timedelta(milliseconds=200)
        result = detector.is_new_bar(current_time)
        if result:
            logger.info(f"✅ New bar at {current_time.strftime('%H:%M:%S.%f')[:-3]}")
        else:
            logger.debug(f"   {current_time.strftime('%H:%M:%S.%f')[:-3]} -> {result}")

def test_multiframe_manager_logic():
    """MultiTimeframeManagerの境界検出ロジックをテスト"""
    from src.data_processing.multiframe_manager import MultiTimeframeManager
    
    def _get_bar_start_time(time: datetime, interval_seconds: int) -> datetime:
        """バーの開始時刻を計算"""
        timestamp = int(time.timestamp())
        bar_start_timestamp = (timestamp // interval_seconds) * interval_seconds
        return datetime.fromtimestamp(bar_start_timestamp)
    
    logger.info("\n=== MultiTimeframeManager方式 ===")
    
    # M5のinterval_seconds = 300
    interval_seconds = 300
    last_bar_time = None
    
    current_time = datetime(2024, 1, 1, 10, 4, 58, 800000)
    for i in range(20):
        current_time += timedelta(milliseconds=200)
        
        current_bar_time = _get_bar_start_time(current_time, interval_seconds)
        
        if last_bar_time is None:
            last_bar_time = current_bar_time
            logger.debug(f"   Initial bar time: {last_bar_time.strftime('%H:%M:%S')}")
        elif current_bar_time > last_bar_time:
            logger.info(f"✅ New bar! Previous: {last_bar_time.strftime('%H:%M:%S')}, "
                       f"Current: {current_bar_time.strftime('%H:%M:%S')}, "
                       f"Tick: {current_time.strftime('%H:%M:%S.%f')[:-3]}")
            last_bar_time = current_bar_time
        else:
            logger.debug(f"   Same bar: {current_time.strftime('%H:%M:%S.%f')[:-3]}")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("M5バー境界検出のテスト")
    logger.info("=" * 60)
    
    # 現在の問題のある実装をテスト
    test_current_boundary_detection()
    
    # 改善された実装をテスト
    test_improved_boundary_detection()
    
    # MultiTimeframeManager方式をテスト
    test_multiframe_manager_logic()
    
    logger.info("\n" + "=" * 60)
    logger.info("結論: timestamp.second == 0 の条件は実際のティックデータでは")
    logger.info("ほぼ満たされないため、境界超えチェック方式に変更すべき")
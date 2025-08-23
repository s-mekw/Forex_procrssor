"""メモリ管理モジュールのテスト"""

import time
import unittest
from unittest.mock import MagicMock, patch

import psutil

from src.common.memory_manager import (
    AdaptiveChunkSize,
    MemoryMonitor,
    MemoryPressureHandler,
    MemoryPressureLevel,
    MemoryStatus,
)


class TestMemoryStatus(unittest.TestCase):
    """MemoryStatusクラスのテスト"""
    
    def test_memory_status_properties(self):
        """プロパティの動作確認"""
        status = MemoryStatus(
            total_mb=16000.0,
            available_mb=8000.0,
            used_mb=8000.0,
            percent=50.0,
            pressure_level=MemoryPressureLevel.NORMAL
        )
        
        self.assertEqual(status.free_mb, 8000.0)
        self.assertFalse(status.is_under_pressure)
    
    def test_is_under_pressure(self):
        """メモリ圧力判定のテスト"""
        # 通常状態
        status = MemoryStatus(
            total_mb=16000.0,
            available_mb=10000.0,
            used_mb=6000.0,
            percent=37.5,
            pressure_level=MemoryPressureLevel.NORMAL
        )
        self.assertFalse(status.is_under_pressure)
        
        # 高圧力状態
        status = MemoryStatus(
            total_mb=16000.0,
            available_mb=1600.0,
            used_mb=14400.0,
            percent=90.0,
            pressure_level=MemoryPressureLevel.CRITICAL
        )
        self.assertTrue(status.is_under_pressure)


class TestMemoryMonitor(unittest.TestCase):
    """MemoryMonitorクラスのテスト"""
    
    def setUp(self):
        """テストのセットアップ"""
        self.monitor = MemoryMonitor()
    
    @patch('psutil.virtual_memory')
    def test_get_status(self, mock_memory):
        """メモリ状態取得のテスト"""
        # モックのメモリ情報
        mock_memory.return_value = MagicMock(
            total=16 * 1024 * 1024 * 1024,  # 16GB
            available=8 * 1024 * 1024 * 1024,  # 8GB
            used=8 * 1024 * 1024 * 1024,  # 8GB
            percent=50.0
        )
        
        status = self.monitor.get_status()
        
        self.assertAlmostEqual(status.total_mb, 16384.0, places=0)
        self.assertAlmostEqual(status.available_mb, 8192.0, places=0)
        self.assertAlmostEqual(status.used_mb, 8192.0, places=0)
        self.assertEqual(status.percent, 50.0)
        self.assertEqual(status.pressure_level, MemoryPressureLevel.NORMAL)
    
    def test_determine_pressure_level(self):
        """圧力レベル判定のテスト"""
        monitor = MemoryMonitor()
        
        # 各レベルのテスト
        test_cases = [
            (30.0, MemoryPressureLevel.NORMAL),
            (65.0, MemoryPressureLevel.MODERATE),
            (85.0, MemoryPressureLevel.HIGH),
            (92.0, MemoryPressureLevel.CRITICAL),
            (96.0, MemoryPressureLevel.EXTREME),
        ]
        
        for percent, expected_level in test_cases:
            level = monitor._determine_pressure_level(percent)
            self.assertEqual(
                level, expected_level,
                f"Failed for {percent}%: expected {expected_level}, got {level}"
            )
    
    @patch('psutil.virtual_memory')
    def test_check_available_memory(self, mock_memory):
        """利用可能メモリチェックのテスト"""
        # 十分なメモリがある場合
        mock_memory.return_value = MagicMock(
            total=16 * 1024 * 1024 * 1024,
            available=8 * 1024 * 1024 * 1024,
            used=8 * 1024 * 1024 * 1024,
            percent=50.0
        )
        
        self.assertTrue(self.monitor.check_available_memory(1000))
        
        # メモリ不足の場合
        mock_memory.return_value = MagicMock(
            total=16 * 1024 * 1024 * 1024,
            available=500 * 1024 * 1024,  # 500MB
            used=15.5 * 1024 * 1024 * 1024,
            percent=97.0
        )
        
        self.assertFalse(self.monitor.check_available_memory(1000))


class TestAdaptiveChunkSize(unittest.TestCase):
    """AdaptiveChunkSizeクラスのテスト"""
    
    def setUp(self):
        """テストのセットアップ"""
        self.adaptive = AdaptiveChunkSize(
            initial_size=100_000,
            min_size=1_000,
            max_size=1_000_000
        )
    
    @patch('psutil.virtual_memory')
    def test_get_optimal_size_normal(self, mock_memory):
        """通常時の最適サイズ取得テスト"""
        # 通常のメモリ状態
        mock_memory.return_value = MagicMock(
            total=16 * 1024 * 1024 * 1024,
            available=10 * 1024 * 1024 * 1024,
            percent=37.5
        )
        
        size = self.adaptive.get_optimal_size()
        self.assertEqual(size, 100_000)  # 初期サイズのまま
    
    @patch('psutil.virtual_memory')
    def test_get_optimal_size_high_pressure(self, mock_memory):
        """高メモリ圧力時の最適サイズ取得テスト"""
        # 高メモリ圧力
        mock_memory.return_value = MagicMock(
            total=16 * 1024 * 1024 * 1024,
            available=1.6 * 1024 * 1024 * 1024,
            percent=90.0
        )
        
        size = self.adaptive.get_optimal_size()
        self.assertEqual(size, 33_000)  # 初期サイズの33%
    
    def test_increase_size(self):
        """チャンクサイズ増加のテスト"""
        initial = self.adaptive.current_size
        new_size = self.adaptive.increase_size(factor=2.0)
        
        self.assertEqual(new_size, min(initial * 2, self.adaptive.max_size))
    
    def test_decrease_size(self):
        """チャンクサイズ減少のテスト"""
        initial = self.adaptive.current_size
        new_size = self.adaptive.decrease_size(factor=0.5)
        
        self.assertEqual(new_size, max(initial * 0.5, self.adaptive.min_size))
    
    @patch('psutil.virtual_memory')
    def test_update_based_on_performance(self, mock_memory):
        """パフォーマンスベースの更新テスト"""
        # 良好なパフォーマンス、メモリに余裕
        mock_memory.return_value = MagicMock(
            total=16 * 1024 * 1024 * 1024,
            available=10 * 1024 * 1024 * 1024,
            percent=37.5
        )
        
        initial = self.adaptive.current_size
        self.adaptive.update_based_on_performance(
            processing_time=0.5,
            memory_used_mb=100,
            success=True
        )
        
        # サイズが増加することを確認
        self.assertGreater(self.adaptive.current_size, initial)
        
        # 失敗時はサイズが減少
        self.adaptive.update_based_on_performance(
            processing_time=5.0,
            memory_used_mb=1000,
            success=False
        )
        
        self.assertLess(self.adaptive.current_size, initial)
    
    def test_reset(self):
        """リセット機能のテスト"""
        # サイズを変更
        self.adaptive.increase_size(factor=3.0)
        self.adaptive._performance_history = [(100_000, 1.0, 100.0)]
        
        # リセット
        self.adaptive.reset(size=50_000)
        
        self.assertEqual(self.adaptive.current_size, 50_000)
        self.assertEqual(len(self.adaptive._performance_history), 0)


class TestMemoryPressureHandler(unittest.TestCase):
    """MemoryPressureHandlerクラスのテスト"""
    
    def setUp(self):
        """テストのセットアップ"""
        self.handler = MemoryPressureHandler()
    
    @patch('psutil.virtual_memory')
    def test_should_gc(self, mock_memory):
        """GC実行判定のテスト"""
        # 高メモリ圧力時
        mock_memory.return_value = MagicMock(
            total=16 * 1024 * 1024 * 1024,
            available=1.6 * 1024 * 1024 * 1024,
            percent=90.0
        )
        
        self.assertTrue(self.handler.should_gc())
        
        # 通常時（前回GCからの増加なし）
        mock_memory.return_value = MagicMock(
            total=16 * 1024 * 1024 * 1024,
            available=10 * 1024 * 1024 * 1024,
            percent=37.5
        )
        self.handler._last_gc_percent = 35.0
        
        self.assertFalse(self.handler.should_gc(threshold_percent=10.0))
    
    @patch('gc.collect')
    @patch('psutil.Process')
    @patch('psutil.virtual_memory')
    def test_force_garbage_collection(self, mock_memory, mock_process_cls, mock_gc):
        """強制GC実行のテスト"""
        # メモリ使用量のモック
        mock_process = MagicMock()
        mock_process_cls.return_value = mock_process
        
        # GC前後のメモリ使用量
        mock_process.memory_info.side_effect = [
            MagicMock(rss=1000 * 1024 * 1024),  # 1000MB
            MagicMock(rss=800 * 1024 * 1024),   # 800MB
        ]
        
        mock_memory.return_value = MagicMock(percent=50.0)
        
        before, after = self.handler.force_garbage_collection()
        
        self.assertAlmostEqual(before, 1000.0, places=0)
        self.assertAlmostEqual(after, 800.0, places=0)
        mock_gc.assert_called_once()
    
    @patch('psutil.virtual_memory')
    def test_get_recommended_action(self, mock_memory):
        """推奨アクション取得のテスト"""
        test_cases = [
            (30.0, MemoryPressureLevel.NORMAL, "normal"),
            (70.0, MemoryPressureLevel.MODERATE, "Consider"),
            (85.0, MemoryPressureLevel.HIGH, "Reduce"),
            (92.0, MemoryPressureLevel.CRITICAL, "garbage"),
            (96.0, MemoryPressureLevel.EXTREME, "CRITICAL"),
        ]
        
        for percent, level, expected_keyword in test_cases:
            mock_memory.return_value = MagicMock(
                total=16 * 1024 * 1024 * 1024,
                available=(100 - percent) * 0.16 * 1024 * 1024 * 1024,
                percent=percent
            )
            
            # 新しいハンドラーを作成（状態をリセット）
            handler = MemoryPressureHandler()
            action = handler.get_recommended_action()
            
            self.assertIn(
                expected_keyword.lower(), action.lower(),
                f"Failed for {percent}%: action '{action}' doesn't contain '{expected_keyword}'"
            )
    
    @patch('time.sleep')
    @patch('psutil.virtual_memory')
    def test_wait_for_memory(self, mock_memory, mock_sleep):
        """メモリ待機のテスト"""
        # 最初は不足、その後利用可能に
        # check_available_memoryはget_statusを呼び、そこでpsutil.virtual_memory()が呼ばれる
        mock_memory.side_effect = [
            # 1回目のチェック（不足）
            MagicMock(available=500 * 1024 * 1024, percent=97.0, total=16 * 1024 * 1024 * 1024, used=15.5 * 1024 * 1024 * 1024),  # 500MB
            # should_gc内のget_status
            MagicMock(available=500 * 1024 * 1024, percent=97.0, total=16 * 1024 * 1024 * 1024, used=15.5 * 1024 * 1024 * 1024),
            # force_garbage_collection内のget_memory_usage_mb (2回)とget_status
            MagicMock(available=500 * 1024 * 1024, percent=97.0, total=16 * 1024 * 1024 * 1024, used=15.5 * 1024 * 1024 * 1024),
            # 2回目のチェック（まだ不足）  
            MagicMock(available=500 * 1024 * 1024, percent=97.0, total=16 * 1024 * 1024 * 1024, used=15.5 * 1024 * 1024 * 1024),
            # should_gc内のget_status
            MagicMock(available=500 * 1024 * 1024, percent=97.0, total=16 * 1024 * 1024 * 1024, used=15.5 * 1024 * 1024 * 1024),
            # force_garbage_collection内のget_memory_usage_mb (2回)とget_status
            MagicMock(available=500 * 1024 * 1024, percent=97.0, total=16 * 1024 * 1024 * 1024, used=15.5 * 1024 * 1024 * 1024),
            # 3回目のチェック（利用可能）
            MagicMock(available=2000 * 1024 * 1024, percent=87.5, total=16 * 1024 * 1024 * 1024, used=14 * 1024 * 1024 * 1024),  # 2000MB、利用可能
        ]
        
        result = self.handler.wait_for_memory(
            required_mb=1500,
            max_wait_seconds=10,
            check_interval=1.0
        )
        
        self.assertTrue(result)
        self.assertEqual(mock_sleep.call_count, 2)
    
    @patch('time.sleep')
    @patch('psutil.virtual_memory')
    def test_wait_for_memory_timeout(self, mock_memory, mock_sleep):
        """メモリ待機タイムアウトのテスト"""
        # 常にメモリ不足
        mock_memory.return_value = MagicMock(
            available=500 * 1024 * 1024,
            percent=97.0
        )
        
        result = self.handler.wait_for_memory(
            required_mb=1500,
            max_wait_seconds=3,
            check_interval=1.0
        )
        
        self.assertFalse(result)
        self.assertEqual(mock_sleep.call_count, 3)


if __name__ == '__main__':
    unittest.main()
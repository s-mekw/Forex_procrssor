"""統一的なメモリ管理モジュール

このモジュールは、Forex Processor全体で使用される
メモリ管理機能を提供します。リアルタイム処理とバッチ処理の
両方で使用可能な汎用的な設計になっています。
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import psutil


class MemoryPressureLevel(Enum):
    """メモリ圧力レベルの定義"""
    
    NORMAL = "normal"       # < 60%
    MODERATE = "moderate"   # 60-80%
    HIGH = "high"          # 80-90%
    CRITICAL = "critical"   # 90-95%
    EXTREME = "extreme"     # >= 95%


@dataclass
class MemoryStatus:
    """メモリ状態を表すデータクラス"""
    
    total_mb: float
    available_mb: float
    used_mb: float
    percent: float
    pressure_level: MemoryPressureLevel
    
    @property
    def free_mb(self) -> float:
        """空きメモリ（MB）"""
        return self.available_mb
    
    @property
    def is_under_pressure(self) -> bool:
        """メモリ圧力下にあるかどうか"""
        return self.pressure_level not in (
            MemoryPressureLevel.NORMAL, 
            MemoryPressureLevel.MODERATE
        )


class MemoryMonitor:
    """システムメモリの監視と状態レポート
    
    このクラスは、システムメモリの状態を監視し、
    メモリ圧力レベルを判定する機能を提供します。
    
    Example:
        >>> monitor = MemoryMonitor()
        >>> status = monitor.get_status()
        >>> if status.is_under_pressure:
        ...     print(f"Memory pressure: {status.pressure_level.value}")
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        thresholds: Optional[dict[str, float]] = None
    ):
        """
        初期化
        
        Args:
            logger: ロガーインスタンス（省略時は新規作成）
            thresholds: メモリ圧力レベルの閾値（％）
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # デフォルトの閾値
        self.thresholds = thresholds or {
            "moderate": 60.0,
            "high": 80.0,
            "critical": 90.0,
            "extreme": 95.0
        }
    
    def get_status(self) -> MemoryStatus:
        """現在のメモリ状態を取得
        
        Returns:
            MemoryStatus: メモリ状態情報
        """
        memory_info = psutil.virtual_memory()
        
        total_mb = memory_info.total / (1024 * 1024)
        available_mb = memory_info.available / (1024 * 1024)
        used_mb = memory_info.used / (1024 * 1024)
        percent = memory_info.percent
        
        # メモリ圧力レベルの判定
        pressure_level = self._determine_pressure_level(percent)
        
        return MemoryStatus(
            total_mb=total_mb,
            available_mb=available_mb,
            used_mb=used_mb,
            percent=percent,
            pressure_level=pressure_level
        )
    
    def _determine_pressure_level(self, percent: float) -> MemoryPressureLevel:
        """メモリ使用率から圧力レベルを判定
        
        Args:
            percent: メモリ使用率（％）
            
        Returns:
            MemoryPressureLevel: 圧力レベル
        """
        if percent >= self.thresholds["extreme"]:
            return MemoryPressureLevel.EXTREME
        elif percent >= self.thresholds["critical"]:
            return MemoryPressureLevel.CRITICAL
        elif percent >= self.thresholds["high"]:
            return MemoryPressureLevel.HIGH
        elif percent >= self.thresholds["moderate"]:
            return MemoryPressureLevel.MODERATE
        else:
            return MemoryPressureLevel.NORMAL
    
    def check_available_memory(self, required_mb: float) -> bool:
        """必要なメモリが利用可能かチェック
        
        Args:
            required_mb: 必要なメモリ量（MB）
            
        Returns:
            bool: 利用可能な場合True
        """
        status = self.get_status()
        
        if required_mb > status.available_mb:
            self.logger.warning(
                f"Insufficient memory: required {required_mb:.1f}MB, "
                f"available {status.available_mb:.1f}MB"
            )
            return False
        
        # メモリ圧力が高い場合も警告
        if status.is_under_pressure:
            self.logger.warning(
                f"Memory pressure is {status.pressure_level.value}: "
                f"{status.percent:.1f}% used"
            )
        
        return True
    
    def get_memory_usage_mb(self) -> float:
        """現在のプロセスのメモリ使用量を取得（MB）
        
        Returns:
            float: メモリ使用量（MB）
        """
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)


class AdaptiveChunkSize:
    """メモリ状況に応じた動的チャンクサイズ調整
    
    このクラスは、システムメモリの状況に応じて
    処理チャンクサイズを動的に調整する機能を提供します。
    
    Example:
        >>> adaptive = AdaptiveChunkSize(initial_size=100_000)
        >>> chunk_size = adaptive.get_optimal_size()
        >>> # 処理後にフィードバック
        >>> adaptive.update_based_on_performance(
        ...     processing_time=1.5, 
        ...     memory_used_mb=500
        ... )
    """
    
    def __init__(
        self,
        initial_size: int = 100_000,
        min_size: int = 1_000,
        max_size: int = 1_000_000,
        memory_monitor: Optional[MemoryMonitor] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初期化
        
        Args:
            initial_size: 初期チャンクサイズ
            min_size: 最小チャンクサイズ
            max_size: 最大チャンクサイズ
            memory_monitor: メモリモニターインスタンス
            logger: ロガーインスタンス
        """
        self.current_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.logger = logger or logging.getLogger(__name__)
        
        # パフォーマンス履歴（調整の参考用）
        self._performance_history: list[tuple[int, float, float]] = []
        self._adjustment_factor = 1.0
    
    def get_optimal_size(self) -> int:
        """現在のメモリ状況に基づく最適なチャンクサイズを取得
        
        Returns:
            int: 推奨チャンクサイズ
        """
        status = self.memory_monitor.get_status()
        
        # メモリ圧力に基づく調整
        if status.pressure_level == MemoryPressureLevel.EXTREME:
            self._adjustment_factor = 0.25
        elif status.pressure_level == MemoryPressureLevel.CRITICAL:
            self._adjustment_factor = 0.33
        elif status.pressure_level == MemoryPressureLevel.HIGH:
            self._adjustment_factor = 0.5
        elif status.pressure_level == MemoryPressureLevel.MODERATE:
            self._adjustment_factor = 0.75
        else:
            self._adjustment_factor = 1.0
        
        # 調整後のサイズを計算
        adjusted_size = int(self.current_size * self._adjustment_factor)
        
        # 範囲内に収める
        optimal_size = max(self.min_size, min(adjusted_size, self.max_size))
        
        if optimal_size != self.current_size:
            self.logger.info(
                f"Chunk size adjusted: {self.current_size:,} → {optimal_size:,} "
                f"(memory: {status.percent:.1f}%, pressure: {status.pressure_level.value})"
            )
            self.current_size = optimal_size
        
        return optimal_size
    
    def update_based_on_performance(
        self,
        processing_time: float,
        memory_used_mb: float,
        success: bool = True
    ) -> None:
        """処理パフォーマンスに基づいてチャンクサイズを更新
        
        Args:
            processing_time: 処理時間（秒）
            memory_used_mb: 使用メモリ量（MB）
            success: 処理が成功したかどうか
        """
        # パフォーマンス履歴に追加
        self._performance_history.append(
            (self.current_size, processing_time, memory_used_mb)
        )
        
        # 履歴は最新10件のみ保持
        if len(self._performance_history) > 10:
            self._performance_history = self._performance_history[-10:]
        
        # 失敗した場合はサイズを減らす
        if not success:
            self.decrease_size()
            return
        
        # メモリ使用量が多すぎる場合はサイズを減らす
        status = self.memory_monitor.get_status()
        if memory_used_mb > status.available_mb * 0.5:
            self.decrease_size()
        # パフォーマンスが良好でメモリに余裕がある場合はサイズを増やす
        elif processing_time < 1.0 and status.percent < 50:
            self.increase_size()
    
    def increase_size(self, factor: float = 1.5) -> int:
        """チャンクサイズを増やす
        
        Args:
            factor: 増加倍率
            
        Returns:
            int: 新しいチャンクサイズ
        """
        old_size = self.current_size
        new_size = min(int(self.current_size * factor), self.max_size)
        
        if new_size != old_size:
            self.current_size = new_size
            self.logger.info(f"Chunk size increased: {old_size:,} → {new_size:,}")
        
        return self.current_size
    
    def decrease_size(self, factor: float = 0.5) -> int:
        """チャンクサイズを減らす
        
        Args:
            factor: 減少倍率
            
        Returns:
            int: 新しいチャンクサイズ
        """
        old_size = self.current_size
        new_size = max(int(self.current_size * factor), self.min_size)
        
        if new_size != old_size:
            self.current_size = new_size
            self.logger.info(f"Chunk size decreased: {old_size:,} → {new_size:,}")
        
        return self.current_size
    
    def reset(self, size: Optional[int] = None) -> None:
        """チャンクサイズをリセット
        
        Args:
            size: リセット後のサイズ（省略時は初期値）
        """
        if size is not None:
            self.current_size = max(self.min_size, min(size, self.max_size))
        else:
            # 初期値は min_size と max_size の幾何平均
            self.current_size = int((self.min_size * self.max_size) ** 0.5)
        
        self._performance_history.clear()
        self._adjustment_factor = 1.0
        
        self.logger.info(f"Chunk size reset to {self.current_size:,}")


class MemoryPressureHandler:
    """メモリ圧力への統一的な対応
    
    このクラスは、メモリ圧力が高い状況での
    適切な対応策を提供します。
    
    Example:
        >>> handler = MemoryPressureHandler()
        >>> if handler.should_gc():
        ...     handler.force_garbage_collection()
    """
    
    def __init__(
        self,
        memory_monitor: Optional[MemoryMonitor] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初期化
        
        Args:
            memory_monitor: メモリモニターインスタンス
            logger: ロガーインスタンス
        """
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.logger = logger or logging.getLogger(__name__)
        self._last_gc_percent = 0.0
    
    def should_gc(self, threshold_percent: float = 10.0) -> bool:
        """ガベージコレクションを実行すべきか判定
        
        Args:
            threshold_percent: 前回のGCからのメモリ増加閾値（％）
            
        Returns:
            bool: GCを実行すべき場合True
        """
        status = self.memory_monitor.get_status()
        
        # メモリ圧力が高い場合
        if status.is_under_pressure:
            return True
        
        # 前回のGCから一定以上メモリが増加した場合
        if status.percent - self._last_gc_percent >= threshold_percent:
            return True
        
        return False
    
    def force_garbage_collection(self) -> Tuple[float, float]:
        """強制的にガベージコレクションを実行
        
        Returns:
            Tuple[float, float]: (実行前のメモリ使用量MB, 実行後のメモリ使用量MB)
        """
        import gc
        
        # 実行前のメモリ使用量
        before_mb = self.memory_monitor.get_memory_usage_mb()
        
        # ガベージコレクション実行
        gc.collect()
        
        # 実行後のメモリ使用量
        after_mb = self.memory_monitor.get_memory_usage_mb()
        
        # 現在のメモリ使用率を記録
        status = self.memory_monitor.get_status()
        self._last_gc_percent = status.percent
        
        freed_mb = before_mb - after_mb
        if freed_mb > 0:
            self.logger.info(
                f"Garbage collection freed {freed_mb:.1f}MB "
                f"({before_mb:.1f}MB → {after_mb:.1f}MB)"
            )
        
        return before_mb, after_mb
    
    def get_recommended_action(self) -> str:
        """現在のメモリ状況に基づく推奨アクション
        
        Returns:
            str: 推奨アクション
        """
        status = self.memory_monitor.get_status()
        
        if status.pressure_level == MemoryPressureLevel.EXTREME:
            return "CRITICAL: Immediately reduce memory usage or abort processing"
        elif status.pressure_level == MemoryPressureLevel.CRITICAL:
            return "Reduce chunk size significantly and run garbage collection"
        elif status.pressure_level == MemoryPressureLevel.HIGH:
            return "Reduce chunk size and monitor closely"
        elif status.pressure_level == MemoryPressureLevel.MODERATE:
            return "Consider reducing chunk size for stability"
        else:
            return "Memory usage is normal, continue processing"
    
    def wait_for_memory(
        self,
        required_mb: float,
        max_wait_seconds: int = 30,
        check_interval: float = 1.0
    ) -> bool:
        """必要なメモリが利用可能になるまで待機
        
        Args:
            required_mb: 必要なメモリ量（MB）
            max_wait_seconds: 最大待機時間（秒）
            check_interval: チェック間隔（秒）
            
        Returns:
            bool: メモリが利用可能になった場合True
        """
        import time
        
        elapsed = 0.0
        
        while elapsed < max_wait_seconds:
            if self.memory_monitor.check_available_memory(required_mb):
                return True
            
            # ガベージコレクションを試みる
            if self.should_gc():
                self.force_garbage_collection()
            
            time.sleep(check_interval)
            elapsed += check_interval
        
        self.logger.error(
            f"Timeout waiting for {required_mb:.1f}MB memory "
            f"after {max_wait_seconds} seconds"
        )
        return False
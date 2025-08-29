"""
Realtime data processing pipeline implementation.

This module provides asyncio-based real-time data processing pipeline
with backpressure control and latency monitoring.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Protocol, TypedDict

from src.data_processing.analyzer import MultiTimeframeAnalyzer


# Interface definitions using Protocol
class AnalyzerProtocol(Protocol):
    """Protocol for analyzer implementations.

    This protocol defines the interface that any analyzer must implement
    to be used with the RealtimePipeline. This enables dependency injection
    and makes the pipeline testable with mock analyzers.
    """

    def add_new_bar(self, bar: dict[str, Any]) -> None:
        """Add a new bar to the analyzer's buffer."""
        ...

    def is_ready(self) -> bool:
        """Check if the analyzer has enough data to perform analysis."""
        ...

    def analyze_streaming(self) -> dict[str, Any]:
        """Perform streaming analysis on the buffered data."""
        ...

    def get_buffer_size(self) -> int:
        """Get the current buffer size."""
        ...


# Type definitions
class DataPoint(TypedDict):
    """Data point structure for pipeline processing."""

    timestamp: datetime
    data: dict[str, Any]
    metadata: dict[str, Any] | None


class ProcessingResult(TypedDict):
    """Processing result structure."""

    processed_data: dict[str, Any]
    latency: float
    status: str
    multiframe_rci: dict[str, Any] | None  # マルチタイムフレームRCI結果


class RealtimePipeline:
    """
    Asyncio-based real-time data processing pipeline.

    Attributes:
        queue_size: Maximum size of input/output queues
        alert_threshold: Latency threshold in seconds for alerts
        enable_metrics: Flag to enable/disable metrics collection
    """

    def __init__(
        self,
        queue_size: int = 1000,
        alert_threshold: float = 1.0,
        enable_metrics: bool = True,
        enable_multiframe: bool = False,
        multiframe_config: dict[str, Any] | None = None,
        max_history_bars: int = 1440,  # 24時間分の1分足データ
        analyzer: AnalyzerProtocol | None = None,  # 依存性注入用
    ):
        """
        Initialize RealtimePipeline.

        Args:
            queue_size: Maximum size of input/output queues (default: 1000)
            alert_threshold: Latency threshold in seconds for alerts (default: 1.0)
            enable_metrics: Enable/disable metrics collection (default: True)
            enable_multiframe: Enable multi-timeframe analysis (default: False)
            multiframe_config: Configuration for multi-timeframe analyzer (optional)
            max_history_bars: Maximum number of historical bars to keep (default: 1440)
            analyzer: Optional analyzer instance for dependency injection. If provided,
                     this will be used instead of creating a new MultiTimeframeAnalyzer
        """
        self.queue_size = queue_size
        self.alert_threshold = alert_threshold
        self._enable_metrics = enable_metrics
        self._enable_multiframe = enable_multiframe
        self._max_history_bars = max_history_bars

        # Queueは遅延初期化（イベントループ内で作成）
        self._input_queue: asyncio.Queue | None = None
        self._output_queue: asyncio.Queue | None = None

        # Initialize multi-timeframe analyzer
        self._multiframe_analyzer: AnalyzerProtocol | None = None

        # Use injected analyzer if provided, otherwise create a new one if enabled
        if analyzer is not None:
            self._multiframe_analyzer = analyzer
            self._enable_multiframe = True  # Force enable if analyzer is provided
            self._logger = logging.getLogger(__name__)
            self._logger.info("Using injected analyzer instance")
        elif enable_multiframe:
            multiframe_config = multiframe_config or {}
            # max_history_barsパラメータをMultiTimeframeAnalyzerに渡す
            analyzer_config = multiframe_config.copy()
            analyzer_config["max_history_bars"] = self._max_history_bars
            self._multiframe_analyzer = MultiTimeframeAnalyzer(**analyzer_config)
            self._logger = logging.getLogger(__name__)
            self._logger.info(
                f"Multi-timeframe analysis enabled with config: {multiframe_config}"
            )

        # Initialize metrics
        self._metrics: dict[str, Any] = {
            "processed_count": 0,
            "total_latency": 0.0,
            "max_latency": 0.0,
            "min_latency": float("inf"),
            "alert_count": 0,
            "backpressure_events": 0,
            "queue_full_count": 0,
            "max_queue_size": 0,
            "rejected_items": 0,
            "dropped_results": 0,
            "last_alert_time": None,
            "max_consecutive_alerts": 0,
            "auto_pause_triggered": False,
            # Multi-timeframe specific metrics
            "multiframe_processing_count": 0,
            "multiframe_total_latency": 0.0,
            "multiframe_max_latency": 0.0,
        }

        # Alert management
        self._alert_history: list[dict[str, Any]] = []  # アラート履歴を保持
        self._alert_callback: Any = None  # カスタムアラート処理用コールバック
        self._consecutive_alerts: int = 0  # 連続アラート数カウント
        self._alert_escalation_threshold: int = 5  # エスカレーション閾値

        # Pipeline state
        self._is_running: bool = False
        self._processing_task: asyncio.Task | None = None

        # Logger
        self._logger: logging.Logger = logging.getLogger(__name__)

    async def _process_loop(self) -> None:
        """非同期処理ループ（1分足データを継続的に処理）

        入力キューからDataPointを取得し、処理して出力キューへ送信します。
        """
        # Queueが初期化されていない場合のチェック
        if self._input_queue is None or self._output_queue is None:
            self._logger.error("Queues not initialized in process loop")
            return
            
        while self._is_running:
            try:
                # 入力キューからDataPointを取得（タイムアウト設定）
                data_point = await asyncio.wait_for(
                    self._input_queue.get(), timeout=1.0
                )

                # データ処理（1分足データのパススルー）
                result = await self._process_data(data_point)

                # 出力キューへ送信（バックプレッシャー考慮）
                try:
                    if self._output_queue.full():
                        self._logger.warning("Output queue is full, waiting...")

                    await asyncio.wait_for(
                        self._output_queue.put(result),
                        timeout=1.0,  # 1秒タイムアウト
                    )
                except TimeoutError:
                    self._logger.error("Output queue timeout, dropping result")
                    self._metrics["dropped_results"] = (
                        self._metrics.get("dropped_results", 0) + 1
                    )

            except TimeoutError:
                # タイムアウト時は続行（graceful handling）
                continue
            except Exception as e:
                self._logger.error(f"Processing error: {e}")

    async def _process_data(self, data_point: DataPoint) -> ProcessingResult:
        """1分足データの処理（マルチタイムフレーム分析対応）

        Args:
            data_point: 処理対象のデータポイント

        Returns:
            ProcessingResult: 処理結果（マルチタイムフレームRCI含む）
        """

        # 1分足データをそのままパススルー（将来的に変換処理を追加）
        processed_data = data_point["data"]

        # 遅延計測 (timestampをdatetimeからfloatに変換)
        # timestampからの遅延計算は実際のタイムスタンプでのみ行う
        if isinstance(data_point["timestamp"], datetime):
            # 現在時刻とtimestampの差を計算（リアルタイムデータ用）
            current_time = datetime.now()
            if data_point["timestamp"] > current_time:
                # 未来のタイムスタンプの場合は処理時間のみ計測
                latency = 0.001  # デフォルト値
            else:
                latency = (current_time - data_point["timestamp"]).total_seconds()
        else:
            # タイムスタンプがfloat型の場合
            latency = time.time() - data_point["timestamp"]

        # マルチタイムフレーム分析の実行（有効な場合）
        multiframe_rci = await self._perform_multiframe_analysis(
            data_point, processed_data
        )

        # 1秒を超える遅延をチェック（アラート機能統合）
        if latency > self.alert_threshold:
            await self._check_latency_alert(latency, data_point)
        else:
            # アラート解除
            if self._consecutive_alerts > 0:
                self._logger.info(
                    f"Latency returned to normal after {self._consecutive_alerts} alerts"
                )
                self._consecutive_alerts = 0

        # メトリクス更新
        if self._enable_metrics:
            self._update_metrics(latency)

        return {
            "processed_data": processed_data,
            "latency": latency,
            "status": "success",
            "multiframe_rci": multiframe_rci,
        }

    async def _perform_multiframe_analysis(
        self, data_point: DataPoint, processed_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """マルチタイムフレーム分析を実行する

        このメソッドを独立させることで、以下の利点があります：
        - テストしやすくなる（モック化が容易）
        - 責務が明確になる
        - 将来的な拡張が容易

        Args:
            data_point: 処理対象のデータポイント
            processed_data: 処理済みデータ

        Returns:
            分析結果またはNone
        """
        if not (self._enable_multiframe and self._multiframe_analyzer):
            return None

        try:
            multiframe_start = time.time()

            # 新しいバーの作成
            new_bar = {
                "timestamp": data_point["timestamp"],
                "open": processed_data.get("open", 0),
                "high": processed_data.get("high", 0),
                "low": processed_data.get("low", 0),
                "close": processed_data.get("close", 0),
                "volume": processed_data.get("volume", 0),
            }

            # Analyzerにバーを追加
            self._multiframe_analyzer.add_new_bar(new_bar)

            # 分析準備チェック
            if not self._multiframe_analyzer.is_ready():
                self._logger.debug(
                    f"Insufficient history for multi-timeframe analysis: "
                    f"{self._multiframe_analyzer.get_buffer_size()}/200"
                )
                return None

            # 内部バッファを使用して分析実行
            multiframe_rci = self._multiframe_analyzer.analyze_streaming()

            # マルチタイムフレーム処理のメトリクス更新
            multiframe_latency = time.time() - multiframe_start
            if self._enable_metrics:
                self._update_multiframe_metrics(multiframe_latency)

            # ログ出力（デバッグ用）
            if multiframe_rci.get("is_new_long_bar"):
                self._logger.debug(
                    f"New 5-minute bar completed. Long-term RCI: {multiframe_rci.get('long_rci')}"
                )

            return multiframe_rci

        except Exception as e:
            self._logger.error(f"Multi-timeframe analysis error: {e}")
            # エラーが発生してもパイプラインは継続
            return None

    def _update_multiframe_metrics(self, latency: float) -> None:
        """マルチタイムフレーム分析のメトリクスを更新する

        Args:
            latency: 分析にかかった時間（秒）
        """
        self._metrics["multiframe_processing_count"] += 1
        self._metrics["multiframe_total_latency"] += latency
        self._metrics["multiframe_max_latency"] = max(
            self._metrics["multiframe_max_latency"], latency
        )

    def _update_metrics(self, latency: float) -> None:
        """メトリクスの更新（遅延情報の記録）

        Args:
            latency: 計測された遅延時間（秒）
        """
        self._metrics["processed_count"] += 1
        self._metrics["total_latency"] += latency
        self._metrics["max_latency"] = max(self._metrics.get("max_latency", 0), latency)
        self._metrics["min_latency"] = min(
            self._metrics.get("min_latency", float("inf")), latency
        )

        # 移動平均の更新
        if "latency_samples" not in self._metrics:
            self._metrics["latency_samples"] = []

        self._metrics["latency_samples"].append(latency)
        # 最新100サンプルのみ保持
        if len(self._metrics["latency_samples"]) > 100:
            self._metrics["latency_samples"].pop(0)

    async def start(self) -> None:
        """
        Start the pipeline processing.

        This method initializes the processing loop and starts
        consuming data from the input queue.
        """
        if self._is_running:
            raise RuntimeError("Pipeline already running")

        # 現在のイベントループ内でQueueを作成
        self._input_queue = asyncio.Queue(maxsize=self.queue_size)
        self._output_queue = asyncio.Queue(maxsize=self.queue_size)
        
        self._is_running = True
        self._processing_task = asyncio.create_task(self._process_loop())
        self._logger.info("RealtimePipeline started with queues initialized")

    async def stop(self) -> None:
        """
        Stop the pipeline processing.

        This method gracefully shuts down the processing loop
        and clears any remaining data in queues.
        """
        if not self._is_running:
            self._logger.warning("Pipeline is not running")
            return

        self._is_running = False

        # 処理タスクの終了を待つ
        if self._processing_task:
            await self._processing_task
            self._processing_task = None

        self._logger.info("RealtimePipeline stopped")

    async def submit(self, data: DataPoint) -> bool:
        """
        Submit data to the pipeline for processing (with backpressure control).

        Args:
            data: DataPoint to be processed

        Returns:
            bool: True if submission succeeded, False if timed out

        Raises:
            RuntimeError: If the pipeline is not running
        """
        if not self._is_running:
            raise RuntimeError("Pipeline is not running")
        
        # Queueが初期化されていない場合のチェック
        if self._input_queue is None:
            raise RuntimeError("Pipeline queues not initialized. Call start() first.")

        try:
            # Check if queue is full
            if self._input_queue.full():
                self._metrics["queue_full_count"] += 1
                self._metrics["backpressure_events"] += 1
                self._logger.warning(
                    f"Input queue is full ({self._input_queue.qsize()}/{self._input_queue.maxsize})"
                )

                # Wait with timeout
                await asyncio.wait_for(
                    self._input_queue.put(data),
                    timeout=0.1,  # 100ms timeout
                )
                return True
            else:
                # Normal submission
                await self._input_queue.put(data)

                # Update queue size metrics
                current_size = self._input_queue.qsize()
                self._metrics["max_queue_size"] = max(
                    self._metrics["max_queue_size"], current_size
                )
                return True

        except TimeoutError:
            self._metrics["rejected_items"] += 1
            self._logger.error("Failed to submit data: queue timeout")
            return False

    async def get_result(self) -> ProcessingResult:
        """
        Get processed result from the pipeline.

        Returns:
            ProcessingResult from the output queue

        Raises:
            asyncio.QueueEmpty: If no results are available
        """
        if not self._is_running:
            raise RuntimeError("Pipeline is not running")
        
        # Queueが初期化されていない場合のチェック
        if self._output_queue is None:
            raise RuntimeError("Pipeline queues not initialized. Call start() first.")

        result = await self._output_queue.get()
        return result

    def get_metrics(self) -> dict[str, Any]:
        """
        Get current pipeline metrics.

        Returns:
            Dictionary containing pipeline metrics:
            - processed_count: Number of processed data points
            - total_latency: Cumulative latency
            - max_latency: Maximum observed latency
            - min_latency: Minimum observed latency
            - alert_count: Number of latency alerts triggered
            - backpressure_events: Number of backpressure events
            - multiframe_processing_count: Number of multi-timeframe analyses
            - multiframe_avg_latency: Average multi-timeframe processing latency
        """
        if not self._enable_metrics:
            return {}

        metrics = self._metrics.copy()

        # Add current queue sizes (Queueが初期化されている場合のみ)
        if self._input_queue is not None:
            metrics["input_queue_size"] = self._input_queue.qsize()
            metrics["queue_size"] = self._input_queue.maxsize
        else:
            metrics["input_queue_size"] = 0
            metrics["queue_size"] = self.queue_size
            
        if self._output_queue is not None:
            metrics["output_queue_size"] = self._output_queue.qsize()
        else:
            metrics["output_queue_size"] = 0

        # Calculate average latency
        if metrics["processed_count"] > 0:
            metrics["avg_latency"] = (
                metrics["total_latency"] / metrics["processed_count"]
            )
        else:
            metrics["avg_latency"] = 0.0

        # Calculate multi-timeframe average latency
        if (
            self._enable_multiframe
            and metrics.get("multiframe_processing_count", 0) > 0
        ):
            metrics["multiframe_avg_latency"] = (
                metrics["multiframe_total_latency"]
                / metrics["multiframe_processing_count"]
            )
        else:
            metrics["multiframe_avg_latency"] = 0.0

        # Add buffer size if multi-timeframe is enabled
        if self._enable_multiframe and self._multiframe_analyzer:
            metrics["data_buffer_size"] = self._multiframe_analyzer.get_buffer_size()

        return metrics

    def is_backpressure_active(self) -> bool:
        """
        Check if backpressure is currently active.

        Returns:
            bool: True if input queue usage exceeds 80% threshold
        """
        if not self._is_running or self._input_queue is None:
            return False

        # Backpressure active if queue is 80% or more full
        threshold = self._input_queue.maxsize * 0.8
        return self._input_queue.qsize() >= threshold

    async def get_queue_status(self) -> dict[str, Any]:
        """
        Get detailed queue status information.

        Returns:
            Dictionary containing:
            - is_running: Whether the pipeline is currently running
            - input_queue_size: Current input queue size
            - input_queue_maxsize: Maximum input queue capacity
            - output_queue_size: Current output queue size
            - output_queue_maxsize: Maximum output queue capacity
            - backpressure_active: Whether backpressure is active
            - backpressure_events: Total backpressure events
            - rejected_items: Total rejected items
            - max_queue_size: Maximum queue size observed
        """
        # Queueが初期化されていない場合
        if self._input_queue is None or self._output_queue is None:
            return {
                "is_running": self._is_running,
                "input_queue_size": 0,
                "input_queue_maxsize": self.queue_size,
                "output_queue_size": 0,
                "output_queue_maxsize": self.queue_size,
                "backpressure_active": False,
                "backpressure_events": self._metrics.get("backpressure_events", 0),
                "rejected_items": self._metrics.get("rejected_items", 0),
                "max_queue_size": self._metrics.get("max_queue_size", 0),
            }
        
        return {
            "is_running": self._is_running,
            "input_queue_size": self._input_queue.qsize(),
            "input_queue_maxsize": self._input_queue.maxsize,
            "output_queue_size": self._output_queue.qsize(),
            "output_queue_maxsize": self._output_queue.maxsize,
            "backpressure_active": self.is_backpressure_active(),
            "backpressure_events": self._metrics.get("backpressure_events", 0),
            "rejected_items": self._metrics.get("rejected_items", 0),
            "max_queue_size": self._metrics.get("max_queue_size", 0),
        }

    async def _check_latency_alert(self, latency: float, data_point: DataPoint) -> None:
        """遅延をチェックしてアラートを発出

        Args:
            latency: 遅延時間（秒）
            data_point: 処理中のデータポイント
        """
        self._consecutive_alerts += 1
        alert_info = {
            "timestamp": datetime.now(),
            "latency": latency,
            "data_point": data_point,
            "severity": self._get_alert_severity(latency),
            "consecutive_count": self._consecutive_alerts,
        }

        # アラート履歴に追加（最新100件を保持）
        self._alert_history.append(alert_info)
        if len(self._alert_history) > 100:
            self._alert_history.pop(0)

        # アラートメトリクス更新
        self._metrics["alert_count"] += 1
        self._metrics["last_alert_time"] = time.time()
        self._metrics["max_consecutive_alerts"] = max(
            self._metrics.get("max_consecutive_alerts", 0), self._consecutive_alerts
        )

        # ログ出力（重要度によって変更）
        if alert_info["severity"] == "critical":
            self._logger.critical(f"CRITICAL: Latency {latency:.3f}s exceeds threshold")
        elif alert_info["severity"] == "high":
            self._logger.error(f"HIGH: Latency alert - {latency:.3f}s")
        else:
            self._logger.warning(f"Latency alert: {latency:.3f}s")

        # エスカレーション処理
        if self._consecutive_alerts >= self._alert_escalation_threshold:
            await self._escalate_alert(alert_info)

        # カスタムコールバック実行
        if self._alert_callback:
            if asyncio.iscoroutinefunction(self._alert_callback):
                await self._alert_callback(alert_info)
            else:
                self._alert_callback(alert_info)

    def _get_alert_severity(self, latency: float) -> str:
        """遅延時間に基づいてアラートの重要度を判定

        Args:
            latency: 遅延時間（秒）

        Returns:
            重要度文字列 (low/medium/high/critical)
        """
        if latency > 10.0:  # 10秒超
            return "critical"
        elif latency > 5.0:  # 5秒超
            return "high"
        elif latency > self.alert_threshold:  # 1秒超
            return "medium"
        else:
            return "low"

    async def _escalate_alert(self, alert_info: dict) -> None:
        """アラートをエスカレーション（連続発生時の特別処理）

        Args:
            alert_info: アラート情報
        """
        self._logger.critical(
            f"ESCALATION: {self._consecutive_alerts} consecutive alerts detected! "
            f"Latest latency: {alert_info['latency']:.3f}s"
        )

        # パイプライン一時停止の検討
        if self._consecutive_alerts >= 10:
            self._logger.critical(
                "Automatic pipeline pause triggered due to persistent high latency"
            )
            # 自動停止フラグを設定（オプション）
            self._metrics["auto_pause_triggered"] = True

    def get_alert_statistics(self) -> dict[str, Any]:
        """アラート統計情報を取得

        Returns:
            アラート統計情報を含む辞書
        """
        if not self._alert_history:
            return {
                "total_alerts": 0,
                "recent_alerts": [],
                "avg_latency": 0,
                "max_latency": 0,
            }

        recent_alerts = self._alert_history[-10:]  # 最新10件
        latencies = [a["latency"] for a in self._alert_history]

        return {
            "total_alerts": self._metrics.get("alert_count", 0),
            "recent_alerts": recent_alerts,
            "avg_latency": sum(latencies) / len(latencies),
            "max_latency": max(latencies),
            "consecutive_alerts": self._consecutive_alerts,
            "last_alert_time": self._metrics.get("last_alert_time"),
            "severity_distribution": self._get_severity_distribution(),
        }

    def _get_severity_distribution(self) -> dict[str, int]:
        """アラートの重要度分布を取得

        Returns:
            重要度別のアラート数
        """
        distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for alert in self._alert_history:
            severity = alert.get("severity", "medium")
            distribution[severity] += 1
        return distribution

    def set_alert_callback(self, callback: Any) -> None:
        """カスタムアラート処理のコールバックを設定

        Args:
            callback: アラート発生時に呼び出されるコールバック関数
        """
        self._alert_callback = callback

    def get_multiframe_config(self) -> dict[str, Any] | None:
        """
        Get multi-timeframe analyzer configuration.

        Returns:
            Multi-timeframe analyzer configuration if enabled, None otherwise
        """
        if self._enable_multiframe and self._multiframe_analyzer:
            return self._multiframe_analyzer.get_analyzer_info()
        return None

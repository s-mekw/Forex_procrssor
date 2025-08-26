"""
Realtime data processing pipeline implementation.

This module provides asyncio-based real-time data processing pipeline
with backpressure control and latency monitoring.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, TypedDict


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
    ):
        """
        Initialize RealtimePipeline.

        Args:
            queue_size: Maximum size of input/output queues (default: 1000)
            alert_threshold: Latency threshold in seconds for alerts (default: 1.0)
            enable_metrics: Enable/disable metrics collection (default: True)
        """
        self.queue_size = queue_size
        self.alert_threshold = alert_threshold
        self._enable_metrics = enable_metrics

        # Initialize queues
        self._input_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._output_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)

        # Initialize metrics
        self._metrics: dict[str, Any] = {
            "processed_count": 0,
            "total_latency": 0.0,
            "max_latency": 0.0,
            "min_latency": float("inf"),
            "alert_count": 0,
            "backpressure_events": 0,
        }

        # Pipeline state
        self._is_running: bool = False
        self._processing_task: asyncio.Task | None = None

        # Logger
        self._logger: logging.Logger = logging.getLogger(__name__)

    async def _process_loop(self) -> None:
        """非同期処理ループ（1分足データを継続的に処理）

        入力キューからDataPointを取得し、処理して出力キューへ送信します。
        """
        while self._is_running:
            try:
                # 入力キューからDataPointを取得（タイムアウト設定）
                data_point = await asyncio.wait_for(
                    self._input_queue.get(),
                    timeout=1.0
                )

                # データ処理（1分足データのパススルー）
                result = await self._process_data(data_point)

                # 出力キューへ送信
                await self._output_queue.put(result)

            except TimeoutError:
                # タイムアウト時は続行（graceful handling）
                continue
            except Exception as e:
                self._logger.error(f"Processing error: {e}")

    async def _process_data(self, data_point: DataPoint) -> ProcessingResult:
        """1分足データの処理（現在はパススルー）

        Args:
            data_point: 処理対象のデータポイント

        Returns:
            ProcessingResult: 処理結果
        """

        # 1分足データをそのままパススルー（将来的に変換処理を追加）
        processed_data = data_point['data']

        # 遅延計測 (timestampをdatetimeからfloatに変換)
        if isinstance(data_point['timestamp'], datetime):
            timestamp = data_point['timestamp'].timestamp()
        else:
            timestamp = data_point['timestamp']

        latency = time.time() - timestamp

        # 1秒を超える遅延をチェック（アラート準備）
        if latency > self.alert_threshold:
            self._logger.warning(f"High latency detected: {latency:.3f}s")
            if self._enable_metrics:
                self._metrics['alert_count'] += 1

        # メトリクス更新
        if self._enable_metrics:
            self._update_metrics(latency)

        return {
            'processed_data': processed_data,
            'latency': latency,
            'status': 'success'
        }

    def _update_metrics(self, latency: float) -> None:
        """メトリクスの更新（遅延情報の記録）

        Args:
            latency: 計測された遅延時間（秒）
        """
        self._metrics['processed_count'] += 1
        self._metrics['total_latency'] += latency
        self._metrics['max_latency'] = max(self._metrics.get('max_latency', 0), latency)
        self._metrics['min_latency'] = min(self._metrics.get('min_latency', float('inf')), latency)

        # 移動平均の更新
        if 'latency_samples' not in self._metrics:
            self._metrics['latency_samples'] = []

        self._metrics['latency_samples'].append(latency)
        # 最新100サンプルのみ保持
        if len(self._metrics['latency_samples']) > 100:
            self._metrics['latency_samples'].pop(0)

    async def start(self) -> None:
        """
        Start the pipeline processing.

        This method initializes the processing loop and starts
        consuming data from the input queue.
        """
        if self._is_running:
            self._logger.warning("Pipeline is already running")
            return

        self._is_running = True
        self._processing_task = asyncio.create_task(self._process_loop())
        self._logger.info("RealtimePipeline started")

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

    async def submit(self, data: DataPoint) -> None:
        """
        Submit data to the pipeline for processing.

        Args:
            data: DataPoint to be processed

        Raises:
            asyncio.QueueFull: If the input queue is full (backpressure)
        """
        if not self._is_running:
            raise RuntimeError("Pipeline is not running")

        await self._input_queue.put(data)
        # TODO: Add metrics tracking in Step 4

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
        """
        if not self._enable_metrics:
            return {}

        metrics = self._metrics.copy()

        # Add current queue sizes
        metrics["input_queue_size"] = self._input_queue.qsize()
        metrics["output_queue_size"] = self._output_queue.qsize()

        # Calculate average latency
        if metrics["processed_count"] > 0:
            metrics["avg_latency"] = (
                metrics["total_latency"] / metrics["processed_count"]
            )
        else:
            metrics["avg_latency"] = 0.0

        return metrics

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
            "queue_full_count": 0,
            "max_queue_size": 0,
            "rejected_items": 0,
            "dropped_results": 0,
            "last_alert_time": None,
            "max_consecutive_alerts": 0,
            "auto_pause_triggered": False,
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
        while self._is_running:
            try:
                # 入力キューからDataPointを取得（タイムアウト設定）
                data_point = await asyncio.wait_for(
                    self._input_queue.get(),
                    timeout=1.0
                )

                # データ処理（1分足データのパススルー）
                result = await self._process_data(data_point)

                # 出力キューへ送信（バックプレッシャー考慮）
                try:
                    if self._output_queue.full():
                        self._logger.warning("Output queue is full, waiting...")

                    await asyncio.wait_for(
                        self._output_queue.put(result),
                        timeout=1.0  # 1秒タイムアウト
                    )
                except TimeoutError:
                    self._logger.error("Output queue timeout, dropping result")
                    self._metrics['dropped_results'] = self._metrics.get('dropped_results', 0) + 1

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

        # 1秒を超える遅延をチェック（アラート機能統合）
        if latency > self.alert_threshold:
            await self._check_latency_alert(latency, data_point)
        else:
            # アラート解除
            if self._consecutive_alerts > 0:
                self._logger.info(f"Latency returned to normal after {self._consecutive_alerts} alerts")
                self._consecutive_alerts = 0

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

        try:
            # Check if queue is full
            if self._input_queue.full():
                self._metrics['queue_full_count'] += 1
                self._metrics['backpressure_events'] += 1
                self._logger.warning(
                    f"Input queue is full ({self._input_queue.qsize()}/{self._input_queue.maxsize})"
                )

                # Wait with timeout
                await asyncio.wait_for(
                    self._input_queue.put(data),
                    timeout=0.1  # 100ms timeout
                )
                return True
            else:
                # Normal submission
                await self._input_queue.put(data)

                # Update queue size metrics
                current_size = self._input_queue.qsize()
                self._metrics['max_queue_size'] = max(
                    self._metrics['max_queue_size'],
                    current_size
                )
                return True

        except TimeoutError:
            self._metrics['rejected_items'] += 1
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

    def is_backpressure_active(self) -> bool:
        """
        Check if backpressure is currently active.

        Returns:
            bool: True if input queue usage exceeds 80% threshold
        """
        if not self._is_running:
            return False

        # Backpressure active if queue is 80% or more full
        threshold = self._input_queue.maxsize * 0.8
        return self._input_queue.qsize() >= threshold

    async def get_queue_status(self) -> dict[str, Any]:
        """
        Get detailed queue status information.

        Returns:
            Dictionary containing:
            - input_queue_size: Current input queue size
            - input_queue_maxsize: Maximum input queue capacity
            - output_queue_size: Current output queue size
            - output_queue_maxsize: Maximum output queue capacity
            - backpressure_active: Whether backpressure is active
            - backpressure_events: Total backpressure events
            - rejected_items: Total rejected items
        """
        return {
            'input_queue_size': self._input_queue.qsize(),
            'input_queue_maxsize': self._input_queue.maxsize,
            'output_queue_size': self._output_queue.qsize(),
            'output_queue_maxsize': self._output_queue.maxsize,
            'backpressure_active': self.is_backpressure_active(),
            'backpressure_events': self._metrics.get('backpressure_events', 0),
            'rejected_items': self._metrics.get('rejected_items', 0)
        }

    async def _check_latency_alert(self, latency: float, data_point: DataPoint) -> None:
        """遅延をチェックしてアラートを発出

        Args:
            latency: 遅延時間（秒）
            data_point: 処理中のデータポイント
        """
        self._consecutive_alerts += 1
        alert_info = {
            'timestamp': datetime.now(),
            'latency': latency,
            'data_point': data_point,
            'severity': self._get_alert_severity(latency),
            'consecutive_count': self._consecutive_alerts
        }

        # アラート履歴に追加（最新100件を保持）
        self._alert_history.append(alert_info)
        if len(self._alert_history) > 100:
            self._alert_history.pop(0)

        # アラートメトリクス更新
        self._metrics['alert_count'] += 1
        self._metrics['last_alert_time'] = time.time()
        self._metrics['max_consecutive_alerts'] = max(
            self._metrics.get('max_consecutive_alerts', 0),
            self._consecutive_alerts
        )

        # ログ出力（重要度によって変更）
        if alert_info['severity'] == 'critical':
            self._logger.critical(f"CRITICAL: Latency {latency:.3f}s exceeds threshold")
        elif alert_info['severity'] == 'high':
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
            return 'critical'
        elif latency > 5.0:  # 5秒超
            return 'high'
        elif latency > self.alert_threshold:  # 1秒超
            return 'medium'
        else:
            return 'low'

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
            self._logger.critical("Automatic pipeline pause triggered due to persistent high latency")
            # 自動停止フラグを設定（オプション）
            self._metrics['auto_pause_triggered'] = True

    def get_alert_statistics(self) -> dict[str, Any]:
        """アラート統計情報を取得

        Returns:
            アラート統計情報を含む辞書
        """
        if not self._alert_history:
            return {
                'total_alerts': 0,
                'recent_alerts': [],
                'avg_latency': 0,
                'max_latency': 0
            }

        recent_alerts = self._alert_history[-10:]  # 最新10件
        latencies = [a['latency'] for a in self._alert_history]

        return {
            'total_alerts': self._metrics.get('alert_count', 0),
            'recent_alerts': recent_alerts,
            'avg_latency': sum(latencies) / len(latencies),
            'max_latency': max(latencies),
            'consecutive_alerts': self._consecutive_alerts,
            'last_alert_time': self._metrics.get('last_alert_time'),
            'severity_distribution': self._get_severity_distribution()
        }

    def _get_severity_distribution(self) -> dict[str, int]:
        """アラートの重要度分布を取得

        Returns:
            重要度別のアラート数
        """
        distribution = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for alert in self._alert_history:
            severity = alert.get('severity', 'medium')
            distribution[severity] += 1
        return distribution

    def set_alert_callback(self, callback: Any) -> None:
        """カスタムアラート処理のコールバックを設定

        Args:
            callback: アラート発生時に呼び出されるコールバック関数
        """
        self._alert_callback = callback

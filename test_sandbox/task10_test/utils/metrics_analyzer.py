"""
メトリクスアナライザー

パイプラインのパフォーマンスメトリクスを分析するユーティリティです。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional


@dataclass
class PerformanceReport:
    """パフォーマンスレポート"""
    
    test_name: str
    duration_seconds: float
    total_messages: int
    successful_messages: int
    failed_messages: int
    avg_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    throughput_per_sec: float
    success_rate_percent: float
    backpressure_events: int
    alert_count: int
    alert_distribution: dict[str, int]
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "test_name": self.test_name,
            "duration_seconds": round(self.duration_seconds, 2),
            "total_messages": self.total_messages,
            "successful_messages": self.successful_messages,
            "failed_messages": self.failed_messages,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "throughput_per_sec": round(self.throughput_per_sec, 2),
            "success_rate_percent": round(self.success_rate_percent, 2),
            "backpressure_events": self.backpressure_events,
            "alert_count": self.alert_count,
            "alert_distribution": self.alert_distribution,
            "memory_usage_mb": round(self.memory_usage_mb, 2) if self.memory_usage_mb else None,
            "cpu_usage_percent": round(self.cpu_usage_percent, 2) if self.cpu_usage_percent else None,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def get_summary(self) -> str:
        """サマリーテキストを生成"""
        return f"""
Performance Report: {self.test_name}
=====================================
Duration: {self.duration_seconds:.2f} seconds
Total Messages: {self.total_messages}
Successful: {self.successful_messages} ({self.success_rate_percent:.1f}%)
Failed: {self.failed_messages}

Latency:
  Average: {self.avg_latency_ms:.2f} ms
  Max: {self.max_latency_ms:.2f} ms
  Min: {self.min_latency_ms:.2f} ms

Throughput: {self.throughput_per_sec:.2f} messages/sec
Backpressure Events: {self.backpressure_events}
Alert Count: {self.alert_count}

Alert Distribution:
{self._format_alert_distribution()}
"""
    
    def _format_alert_distribution(self) -> str:
        """アラート分布のフォーマット"""
        if not self.alert_distribution:
            return "  None"
        
        lines = []
        for severity, count in sorted(self.alert_distribution.items()):
            lines.append(f"  {severity}: {count}")
        return "\n".join(lines)


class MetricsAnalyzer:
    """メトリクス分析器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.message_count = 0
        self.success_count = 0
        self.fail_count = 0
        self.latencies = []
        self.backpressure_events = 0
        self.alerts = []
        self.metrics_history = []
        
    def start_analysis(self):
        """分析開始"""
        self.start_time = datetime.now()
        self.reset_counters()
    
    def end_analysis(self):
        """分析終了"""
        self.end_time = datetime.now()
    
    def reset_counters(self):
        """カウンターをリセット"""
        self.message_count = 0
        self.success_count = 0
        self.fail_count = 0
        self.latencies = []
        self.backpressure_events = 0
        self.alerts = []
        self.metrics_history = []
    
    def record_message(self, success: bool, latency_ms: float = 0):
        """メッセージ処理を記録"""
        self.message_count += 1
        if success:
            self.success_count += 1
            if latency_ms > 0:
                self.latencies.append(latency_ms)
        else:
            self.fail_count += 1
    
    def record_backpressure(self):
        """バックプレッシャーイベントを記録"""
        self.backpressure_events += 1
    
    def record_alert(self, severity: str, latency_ms: float):
        """アラートを記録"""
        self.alerts.append({
            "severity": severity,
            "latency_ms": latency_ms,
            "timestamp": datetime.now()
        })
    
    def record_metrics_snapshot(self, metrics: dict[str, Any]):
        """メトリクススナップショットを記録"""
        self.metrics_history.append({
            "timestamp": datetime.now(),
            "metrics": metrics.copy()
        })
    
    def analyze_pipeline_metrics(self, pipeline_metrics: dict[str, Any]):
        """パイプラインメトリクスを分析"""
        # 処理数の更新
        if "processed_count" in pipeline_metrics:
            self.message_count = pipeline_metrics["processed_count"]
            self.success_count = pipeline_metrics["processed_count"]
        
        # 遅延情報の更新
        if "avg_latency" in pipeline_metrics:
            avg_latency_seconds = pipeline_metrics["avg_latency"]
            if avg_latency_seconds and avg_latency_seconds > 0:
                self.latencies.append(avg_latency_seconds * 1000)  # 秒からミリ秒へ
        
        # バックプレッシャーイベントの更新
        if "backpressure_events" in pipeline_metrics:
            self.backpressure_events = pipeline_metrics["backpressure_events"]
        
        # アラート数の更新
        if "alert_count" in pipeline_metrics:
            alert_count = pipeline_metrics["alert_count"]
            # 新規アラート分を記録
            while len(self.alerts) < alert_count:
                self.record_alert("unknown", 0)
    
    def generate_report(self, test_name: str) -> PerformanceReport:
        """パフォーマンスレポートを生成"""
        if not self.start_time:
            self.start_time = datetime.now()
        if not self.end_time:
            self.end_time = datetime.now()
        
        duration = (self.end_time - self.start_time).total_seconds()
        
        # 遅延統計の計算
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        max_latency = max(self.latencies) if self.latencies else 0
        min_latency = min(self.latencies) if self.latencies else 0
        
        # スループットの計算
        throughput = self.message_count / duration if duration > 0 else 0
        
        # 成功率の計算
        success_rate = (self.success_count / self.message_count * 100) if self.message_count > 0 else 0
        
        # アラート分布の計算
        alert_distribution = self._calculate_alert_distribution()
        
        return PerformanceReport(
            test_name=test_name,
            duration_seconds=duration,
            total_messages=self.message_count,
            successful_messages=self.success_count,
            failed_messages=self.fail_count,
            avg_latency_ms=avg_latency,
            max_latency_ms=max_latency,
            min_latency_ms=min_latency,
            throughput_per_sec=throughput,
            success_rate_percent=success_rate,
            backpressure_events=self.backpressure_events,
            alert_count=len(self.alerts),
            alert_distribution=alert_distribution,
        )
    
    def _calculate_alert_distribution(self) -> dict[str, int]:
        """アラート分布を計算"""
        distribution = {}
        for alert in self.alerts:
            severity = alert["severity"]
            distribution[severity] = distribution.get(severity, 0) + 1
        return distribution
    
    def get_time_series_metrics(self) -> list[dict[str, Any]]:
        """時系列メトリクスを取得"""
        return self.metrics_history
    
    def calculate_stability_score(self) -> float:
        """
        安定性スコアを計算（0-100）
        
        スコア要素:
        - 成功率: 40%
        - 遅延安定性: 30%
        - バックプレッシャー頻度: 20%
        - アラート頻度: 10%
        """
        score = 0.0
        
        # 成功率スコア（40点満点）
        if self.message_count > 0:
            success_rate = self.success_count / self.message_count
            score += success_rate * 40
        
        # 遅延安定性スコア（30点満点）
        if self.latencies:
            avg_latency = sum(self.latencies) / len(self.latencies)
            # 1秒以下で満点、10秒で0点
            if avg_latency <= 1000:  # 1秒以下
                score += 30
            elif avg_latency <= 10000:  # 10秒以下
                score += 30 * (1 - (avg_latency - 1000) / 9000)
        
        # バックプレッシャー頻度スコア（20点満点）
        if self.message_count > 0:
            bp_rate = self.backpressure_events / self.message_count
            # 10%以下で満点、50%で0点
            if bp_rate <= 0.1:
                score += 20
            elif bp_rate <= 0.5:
                score += 20 * (1 - (bp_rate - 0.1) / 0.4)
        
        # アラート頻度スコア（10点満点）
        if self.message_count > 0:
            alert_rate = len(self.alerts) / self.message_count
            # 1%以下で満点、10%で0点
            if alert_rate <= 0.01:
                score += 10
            elif alert_rate <= 0.1:
                score += 10 * (1 - (alert_rate - 0.01) / 0.09)
        
        return min(100.0, max(0.0, score))
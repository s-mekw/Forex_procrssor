"""
24時間連続稼働シミュレーション（短縮版）

実際には数分で実行される短縮版シミュレーションです。
市場時間帯による負荷変動、メモリリーク、パフォーマンス劣化を監視します。
"""

import asyncio
import gc
import logging
import psutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# プロジェクトのルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.pipelines import RealtimePipeline
from test_sandbox.task10_test.utils import (
    FXDataGenerator,
    MarketCondition,
    MetricsAnalyzer,
    ReportGenerator,
    TestResult,
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MarketSession:
    """市場セッション"""
    
    ASIAN = "Asian"      # 東京市場（低～中ボリューム）
    EUROPEAN = "European"  # ロンドン市場（高ボリューム）
    US = "US"           # ニューヨーク市場（高ボリューム）
    OVERLAP = "Overlap"  # 市場重複時間（最高ボリューム）
    QUIET = "Quiet"     # 静穏時間（最低ボリューム）


class SystemMonitor:
    """システムリソースモニター"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = None
        self.memory_samples = []
        self.cpu_samples = []
        
    def start_monitoring(self):
        """監視開始"""
        gc.collect()  # ガベージコレクション実行
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"初期メモリ使用量: {self.initial_memory:.2f} MB")
        
    def sample_resources(self) -> dict[str, float]:
        """リソース使用状況をサンプリング"""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent(interval=0.1)
        
        self.memory_samples.append(memory_mb)
        self.cpu_samples.append(cpu_percent)
        
        return {
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent,
            "memory_delta_mb": memory_mb - self.initial_memory if self.initial_memory else 0
        }
    
    def check_memory_leak(self) -> tuple[bool, float]:
        """メモリリークをチェック"""
        if len(self.memory_samples) < 10:
            return False, 0.0
        
        # 最初の10サンプルと最後の10サンプルの平均を比較
        early_avg = sum(self.memory_samples[:10]) / 10
        late_avg = sum(self.memory_samples[-10:]) / len(self.memory_samples[-10:])
        
        leak_mb = late_avg - early_avg
        has_leak = leak_mb > 50  # 50MB以上の増加をリークと判定
        
        return has_leak, leak_mb
    
    def get_statistics(self) -> dict[str, Any]:
        """統計情報を取得"""
        if not self.memory_samples:
            return {}
        
        return {
            "memory": {
                "initial_mb": self.initial_memory,
                "current_mb": self.memory_samples[-1] if self.memory_samples else 0,
                "max_mb": max(self.memory_samples),
                "min_mb": min(self.memory_samples),
                "avg_mb": sum(self.memory_samples) / len(self.memory_samples),
            },
            "cpu": {
                "max_percent": max(self.cpu_samples) if self.cpu_samples else 0,
                "avg_percent": sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0,
            }
        }


def get_market_session(hour: int) -> MarketSession:
    """時間帯から市場セッションを判定（簡略化）"""
    if 0 <= hour < 3:
        return MarketSession.ASIAN
    elif 3 <= hour < 8:
        return MarketSession.QUIET
    elif 8 <= hour < 12:
        return MarketSession.EUROPEAN
    elif 12 <= hour < 16:
        return MarketSession.OVERLAP
    elif 16 <= hour < 20:
        return MarketSession.US
    else:
        return MarketSession.QUIET


def get_session_parameters(session: MarketSession) -> dict[str, Any]:
    """セッションに応じたパラメータを取得"""
    params = {
        MarketSession.ASIAN: {
            "volume_multiplier": 1.0,
            "volatility": MarketCondition.NORMAL,
            "tick_rate": 20,  # ticks/sec
            "active_pairs": ["USDJPY", "EURJPY", "AUDJPY"],
        },
        MarketSession.EUROPEAN: {
            "volume_multiplier": 2.0,
            "volatility": MarketCondition.HIGH_VOLATILITY,
            "tick_rate": 50,
            "active_pairs": ["EURUSD", "GBPUSD", "EURGBP", "USDCHF"],
        },
        MarketSession.US: {
            "volume_multiplier": 2.5,
            "volatility": MarketCondition.HIGH_VOLATILITY,
            "tick_rate": 60,
            "active_pairs": ["EURUSD", "GBPUSD", "USDJPY", "USDCAD"],
        },
        MarketSession.OVERLAP: {
            "volume_multiplier": 3.0,
            "volatility": MarketCondition.NEWS_RELEASE,
            "tick_rate": 100,
            "active_pairs": ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY"],
        },
        MarketSession.QUIET: {
            "volume_multiplier": 0.3,
            "volatility": MarketCondition.LOW_LIQUIDITY,
            "tick_rate": 5,
            "active_pairs": ["AUDUSD", "NZDUSD"],
        },
    }
    
    return params.get(session, params[MarketSession.QUIET])


async def simulate_market_session(
    pipeline: RealtimePipeline,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer,
    session: MarketSession,
    duration_seconds: int
):
    """市場セッションのシミュレーション"""
    
    params = get_session_parameters(session)
    logger.info(f"=== {session}セッション開始 (Rate: {params['tick_rate']} ticks/sec) ===")
    
    start_time = datetime.now()
    tick_interval = 1.0 / params["tick_rate"] if params["tick_rate"] > 0 else 1.0
    
    while (datetime.now() - start_time).total_seconds() < duration_seconds:
        # アクティブな通貨ペアからランダムに選択
        import random
        symbol = random.choice(params["active_pairs"])
        
        # データ生成
        data_point = generator.generate_tick(
            symbol=symbol,
            condition=params["volatility"]
        )
        
        # 送信
        success = await pipeline.submit(data_point)
        analyzer.record_message(success, tick_interval * 1000)
        
        await asyncio.sleep(tick_interval)
    
    # セッション終了時のメトリクス
    metrics = await pipeline.get_metrics()
    logger.info(f"{session}セッション終了: {metrics['processed_count']}件処理")


async def continuous_monitoring(
    pipeline: RealtimePipeline,
    monitor: SystemMonitor,
    analyzer: MetricsAnalyzer,
    report_interval_seconds: int = 30
):
    """継続的な監視とレポート生成"""
    
    while True:
        await asyncio.sleep(report_interval_seconds)
        
        # システムリソースをサンプリング
        resources = monitor.sample_resources()
        
        # パイプラインメトリクスを取得
        metrics = await pipeline.get_metrics()
        queue_status = await pipeline.get_queue_status()
        
        # 分析器に記録
        analyzer.analyze_pipeline_metrics(metrics)
        analyzer.record_metrics_snapshot({
            **metrics,
            **resources,
            "queue_usage_percent": queue_status["current_input_size"] / queue_status["max_input_size"] * 100
        })
        
        # 定期レポート
        logger.info(f"定期レポート:")
        logger.info(f"  処理数: {metrics['processed_count']}")
        logger.info(f"  平均遅延: {metrics['avg_latency']:.3f}秒")
        logger.info(f"  メモリ: {resources['memory_mb']:.2f}MB (Δ{resources['memory_delta_mb']:+.2f}MB)")
        logger.info(f"  CPU: {resources['cpu_percent']:.1f}%")
        
        # メモリリークチェック
        has_leak, leak_mb = monitor.check_memory_leak()
        if has_leak:
            logger.warning(f"潜在的なメモリリーク検出: {leak_mb:.2f}MB増加")


async def simulate_24h_operation(
    pipeline: RealtimePipeline,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer,
    monitor: SystemMonitor,
    simulated_hours: int = 24,
    time_scale: float = 0.01  # 1時間を0.01倍（36秒）で実行
):
    """24時間運用のシミュレーション（時間圧縮版）"""
    
    logger.info(f"24時間シミュレーション開始（{simulated_hours}時間を{simulated_hours * time_scale * 3600:.0f}秒で実行）")
    
    # 監視タスクを開始
    monitor_task = asyncio.create_task(
        continuous_monitoring(pipeline, monitor, analyzer, 30)
    )
    
    try:
        for hour in range(simulated_hours):
            # 現在の市場セッションを判定
            session = get_market_session(hour)
            
            # セッションごとの処理
            session_duration = 3600 * time_scale  # 実時間での長さ
            await simulate_market_session(
                pipeline, generator, analyzer, 
                session, session_duration
            )
            
            # ガベージコレクション（メモリ管理）
            if hour % 6 == 0:
                gc.collect()
                logger.info(f"ガベージコレクション実行（{hour}時間経過）")
        
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass


async def main():
    """メインテスト実行"""
    logger.info("=" * 60)
    logger.info("24時間連続稼働シミュレーション E2Eテスト開始")
    logger.info("=" * 60)
    
    # テストコンポーネントの初期化
    report_generator = ReportGenerator("test_sandbox/task10_test/reports")
    
    # パイプラインの初期化
    pipeline = RealtimePipeline(
        queue_size=5000,
        alert_threshold=2.0,
        enable_metrics=True
    )
    
    # モニターとジェネレーター
    monitor = SystemMonitor()
    generator = FXDataGenerator(seed=789)
    analyzer = MetricsAnalyzer()
    
    test_start = datetime.now()
    
    try:
        # システム監視開始
        monitor.start_monitoring()
        
        # パイプライン開始
        await pipeline.start()
        logger.info("パイプライン起動完了")
        
        # 分析開始
        analyzer.start_analysis()
        
        # 24時間シミュレーション実行（実際は数分）
        await simulate_24h_operation(
            pipeline, generator, analyzer, monitor,
            simulated_hours=24,
            time_scale=0.005  # 1時間を18秒で実行（合計7.2分）
        )
        
        # 最終メトリクス取得
        await asyncio.sleep(2)
        final_metrics = await pipeline.get_metrics()
        system_stats = monitor.get_statistics()
        
        # 分析終了
        analyzer.end_analysis()
        
        # メモリリークチェック
        has_leak, leak_mb = monitor.check_memory_leak()
        
        # パフォーマンスレポート生成
        performance_report = analyzer.generate_report("24時間連続稼働")
        
        # 安定性スコア計算
        stability_score = analyzer.calculate_stability_score()
        
        # テスト結果の判定
        test_passed = (
            not has_leak and  # メモリリークなし
            stability_score >= 85.0 and  # 高い安定性
            performance_report.success_rate_percent >= 98.0  # 高成功率
        )
        
        test_result = TestResult(
            test_name="24時間連続稼働シミュレーション",
            status="PASSED" if test_passed else "FAILED",
            duration_seconds=(datetime.now() - test_start).total_seconds(),
            performance_report=performance_report,
            logs=[
                f"シミュレート時間: 24時間",
                f"実行時間: {(datetime.now() - test_start).total_seconds():.1f}秒",
                f"総処理メッセージ: {final_metrics['processed_count']}",
                f"メモリリーク: {'検出' if has_leak else '未検出'}" + (f" ({leak_mb:.2f}MB)" if has_leak else ""),
                f"初期メモリ: {system_stats['memory']['initial_mb']:.2f}MB",
                f"最大メモリ: {system_stats['memory']['max_mb']:.2f}MB",
                f"平均CPU: {system_stats['cpu']['avg_percent']:.1f}%",
                f"安定性スコア: {stability_score:.1f}/100",
            ]
        )
        
        # レポート生成
        report_generator.add_test_result(test_result)
        
        # 結果表示
        logger.info("=" * 60)
        logger.info("テスト結果サマリー")
        logger.info("=" * 60)
        logger.info(f"総処理数: {final_metrics['processed_count']}")
        logger.info(f"成功率: {performance_report.success_rate_percent:.1f}%")
        logger.info(f"平均遅延: {performance_report.avg_latency_ms:.2f}ms")
        logger.info(f"メモリ使用量: 初期{system_stats['memory']['initial_mb']:.2f}MB → 最大{system_stats['memory']['max_mb']:.2f}MB")
        logger.info(f"メモリリーク: {'❌ 検出' if has_leak else '✅ 未検出'}")
        logger.info(f"安定性スコア: {stability_score:.1f}/100")
        logger.info(f"テスト結果: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        
        # レポートファイル生成
        md_path = report_generator.generate_markdown_report("24h_operation.md")
        html_path = report_generator.generate_html_report("24h_operation.html")
        
        logger.info(f"レポート生成完了:")
        logger.info(f"  - Markdown: {md_path}")
        logger.info(f"  - HTML: {html_path}")
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}", exc_info=True)
        
        test_result = TestResult(
            test_name="24時間連続稼働シミュレーション",
            status="FAILED",
            duration_seconds=(datetime.now() - test_start).total_seconds(),
            error_message=str(e)
        )
        report_generator.add_test_result(test_result)
        
    finally:
        # クリーンアップ
        await pipeline.stop()
        logger.info("パイプライン停止完了")


if __name__ == "__main__":
    asyncio.run(main())
"""
バックプレッシャー制御と回復シナリオ

システムの自己防衛機能と回復力を検証します。
段階的な負荷増加、アラートエスカレーション、自動回復プロセスをテストします。
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

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


class LoadController:
    """負荷制御クラス"""
    
    def __init__(self):
        self.current_load = 1.0  # 初期負荷レベル
        self.max_load = 10.0
        self.min_load = 0.1
        
    def increase_load(self, factor: float = 1.5):
        """負荷を増加"""
        self.current_load = min(self.current_load * factor, self.max_load)
        logger.info(f"負荷レベル増加: {self.current_load:.1f}")
        
    def decrease_load(self, factor: float = 0.7):
        """負荷を減少"""
        self.current_load = max(self.current_load * factor, self.min_load)
        logger.info(f"負荷レベル減少: {self.current_load:.1f}")
        
    def get_send_interval(self) -> float:
        """送信間隔を取得（負荷に反比例）"""
        return 0.1 / self.current_load  # 負荷が高いほど間隔が短い


async def monitor_pipeline_health(
    pipeline: RealtimePipeline,
    load_controller: LoadController,
    analyzer: MetricsAnalyzer
):
    """パイプラインの健全性を監視し、負荷を自動調整"""
    
    while True:
        await asyncio.sleep(5)  # 5秒ごとに監視
        
        # メトリクス取得
        metrics = await pipeline.get_metrics()
        queue_status = await pipeline.get_queue_status()
        
        # バックプレッシャー状態を確認
        is_backpressure = await pipeline.is_backpressure_active()
        
        # メトリクスを記録
        analyzer.record_metrics_snapshot(metrics)
        
        # 健全性評価
        queue_usage = queue_status["current_input_size"] / queue_status["max_input_size"]
        
        if is_backpressure or queue_usage > 0.8:
            logger.warning(f"バックプレッシャー検出: キュー使用率 {queue_usage:.1%}")
            load_controller.decrease_load()
            analyzer.record_backpressure()
            
        elif queue_usage < 0.3 and load_controller.current_load < 5.0:
            logger.info(f"システム余裕あり: キュー使用率 {queue_usage:.1%}")
            load_controller.increase_load(1.2)  # 緩やかに増加
            
        # アラート状態の確認
        alert_stats = await pipeline.get_alert_statistics()
        if alert_stats["total_alerts"] > 0:
            severity_dist = alert_stats["severity_distribution"]
            
            # クリティカルアラートがある場合は大幅に負荷を下げる
            if severity_dist.get("critical", 0) > 0:
                logger.critical("クリティカルアラート検出！負荷を大幅削減")
                load_controller.decrease_load(0.3)
            elif severity_dist.get("high", 0) > 0:
                logger.warning("高レベルアラート検出。負荷を削減")
                load_controller.decrease_load(0.5)


async def generate_variable_load(
    pipeline: RealtimePipeline,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer,
    load_controller: LoadController,
    duration_seconds: int
):
    """可変負荷を生成"""
    
    start_time = datetime.now()
    symbols = list(FXDataGenerator.CURRENCY_PAIRS.keys())
    
    while (datetime.now() - start_time).total_seconds() < duration_seconds:
        # 現在の負荷レベルに応じたデータ生成
        batch_size = int(load_controller.current_load * 10)
        
        tasks = []
        for _ in range(batch_size):
            import random
            symbol = random.choice(symbols)
            
            # 負荷が高いときは高ボラティリティ
            condition = (
                MarketCondition.HIGH_VOLATILITY 
                if load_controller.current_load > 5 
                else MarketCondition.NORMAL
            )
            
            data_point = generator.generate_tick(symbol, condition)
            tasks.append(pipeline.submit(data_point))
        
        results = await asyncio.gather(*tasks)
        
        # 結果を記録
        for success in results:
            analyzer.record_message(success)
            if not success:
                analyzer.record_backpressure()
        
        # 送信間隔
        await asyncio.sleep(load_controller.get_send_interval())


async def simulate_gradual_load_increase(
    pipeline: RealtimePipeline,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer
):
    """段階的な負荷増加シミュレーション"""
    logger.info("=== 段階的負荷増加シミュレーション開始 ===")
    
    load_controller = LoadController()
    
    # 監視タスクを開始
    monitor_task = asyncio.create_task(
        monitor_pipeline_health(pipeline, load_controller, analyzer)
    )
    
    try:
        # フェーズ1: 低負荷（10秒）
        logger.info("フェーズ1: 低負荷")
        await generate_variable_load(pipeline, generator, analyzer, load_controller, 10)
        
        # フェーズ2: 負荷を段階的に増加（20秒）
        logger.info("フェーズ2: 負荷増加")
        for _ in range(4):
            load_controller.increase_load(2.0)
            await generate_variable_load(pipeline, generator, analyzer, load_controller, 5)
        
        # フェーズ3: 高負荷維持（10秒）
        logger.info("フェーズ3: 高負荷維持")
        await generate_variable_load(pipeline, generator, analyzer, load_controller, 10)
        
        # フェーズ4: 自然回復（15秒）
        logger.info("フェーズ4: 自然回復待機")
        await asyncio.sleep(15)
        
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass


async def simulate_alert_escalation(
    pipeline: RealtimePipeline,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer
):
    """アラートエスカレーションシミュレーション"""
    logger.info("=== アラートエスカレーションシミュレーション開始 ===")
    
    # エスカレーション状態を追跡
    escalation_state = {"level": "none", "count": 0}
    
    async def escalation_callback(alert_info):
        escalation_state["count"] += 1
        severity = alert_info["severity"]
        
        if severity == "critical":
            escalation_state["level"] = "critical"
            logger.critical(f"エスカレーション: CRITICAL - {alert_info['consecutive_count']}回連続")
        elif severity == "high":
            escalation_state["level"] = "high"
            logger.warning(f"エスカレーション: HIGH - {alert_info['consecutive_count']}回連続")
        else:
            escalation_state["level"] = "medium"
            logger.info(f"エスカレーション: MEDIUM")
    
    pipeline.set_alert_callback(escalation_callback)
    
    # 段階的に遅延を増加させたデータを送信
    delays = [0.5, 1.5, 3.0, 6.0, 11.0]  # 徐々に遅延を増加
    
    for delay in delays:
        logger.info(f"遅延 {delay}秒のデータを送信")
        
        # 遅延データを10件送信
        delayed_data = generator.generate_delayed_data(
            symbol="USDJPY",
            delay_seconds=delay,
            count=10
        )
        
        for data_point in delayed_data:
            success = await pipeline.submit(data_point)
            analyzer.record_message(success)
            
            if delay > 1.0:  # 1秒以上の遅延でアラート記録
                analyzer.record_alert(escalation_state["level"], delay * 1000)
        
        await asyncio.sleep(2)  # 各段階の間に待機
    
    # 正常データで回復
    logger.info("正常データで回復処理")
    for _ in range(20):
        data_point = generator.generate_tick("USDJPY", MarketCondition.NORMAL)
        await pipeline.submit(data_point)
        await asyncio.sleep(0.1)
    
    logger.info(f"エスカレーション総数: {escalation_state['count']}")
    logger.info(f"最高エスカレーションレベル: {escalation_state['level']}")


async def simulate_recovery_process(
    pipeline: RealtimePipeline,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer
):
    """回復プロセスのシミュレーション"""
    logger.info("=== 回復プロセスシミュレーション開始 ===")
    
    # ステップ1: 過負荷状態を作る
    logger.info("ステップ1: 過負荷状態の生成")
    burst_data = generator.generate_burst(
        symbol="EURUSD",
        count=500,
        interval_ms=1,
        condition=MarketCondition.NEWS_RELEASE
    )
    
    submit_tasks = []
    for data_point in burst_data:
        submit_tasks.append(pipeline.submit(data_point))
    
    results = await asyncio.gather(*submit_tasks)
    rejected = sum(1 for r in results if not r)
    logger.info(f"過負荷: {rejected}/{len(results)}件が拒否されました")
    
    # ステップ2: 負荷を段階的に減少
    logger.info("ステップ2: 段階的な負荷減少")
    for load_level in [100, 50, 20, 10, 5]:
        logger.info(f"負荷レベル: {load_level}msg/batch")
        
        for _ in range(5):  # 各レベルで5バッチ
            batch_data = []
            for _ in range(load_level):
                data_point = generator.generate_tick("EURUSD", MarketCondition.NORMAL)
                batch_data.append(pipeline.submit(data_point))
            
            results = await asyncio.gather(*batch_data)
            success_rate = sum(1 for r in results if r) / len(results) * 100
            logger.info(f"  成功率: {success_rate:.1f}%")
            
            for success in results:
                analyzer.record_message(success)
            
            await asyncio.sleep(1)
    
    # ステップ3: 完全回復の確認
    logger.info("ステップ3: 完全回復の確認")
    await asyncio.sleep(5)  # パイプラインの安定化待機
    
    # 正常負荷でのテスト
    normal_test_data = []
    for _ in range(100):
        data_point = generator.generate_tick("EURUSD", MarketCondition.NORMAL)
        normal_test_data.append(pipeline.submit(data_point))
    
    results = await asyncio.gather(*normal_test_data)
    recovery_rate = sum(1 for r in results if r) / len(results) * 100
    
    logger.info(f"回復率: {recovery_rate:.1f}%")
    return recovery_rate >= 95.0  # 95%以上で回復成功


async def main():
    """メインテスト実行"""
    logger.info("=" * 60)
    logger.info("バックプレッシャー制御と回復シナリオ E2Eテスト開始")
    logger.info("=" * 60)
    
    # テストコンポーネントの初期化
    report_generator = ReportGenerator("test_sandbox/task10_test/reports")
    
    # パイプラインの初期化（小さいキューサイズでバックプレッシャーを誘発）
    pipeline = RealtimePipeline(
        queue_size=200,  # 小さいキューサイズ
        alert_threshold=1.0,
        enable_metrics=True
    )
    
    # ジェネレーターと分析器
    generator = FXDataGenerator(seed=456)
    analyzer = MetricsAnalyzer()
    
    test_start = datetime.now()
    
    try:
        # パイプライン開始
        await pipeline.start()
        logger.info("パイプライン起動完了")
        
        # 分析開始
        analyzer.start_analysis()
        
        # 1. 段階的負荷増加シミュレーション
        await simulate_gradual_load_increase(pipeline, generator, analyzer)
        
        # 2. アラートエスカレーションシミュレーション
        await simulate_alert_escalation(pipeline, generator, analyzer)
        
        # 3. 回復プロセスシミュレーション
        recovery_success = await simulate_recovery_process(pipeline, generator, analyzer)
        
        # 最終メトリクス取得
        await asyncio.sleep(2)
        final_metrics = await pipeline.get_metrics()
        queue_status = await pipeline.get_queue_status()
        alert_stats = await pipeline.get_alert_statistics()
        
        # 分析終了
        analyzer.end_analysis()
        
        # パフォーマンスレポート生成
        performance_report = analyzer.generate_report("バックプレッシャー制御と回復")
        
        # 安定性スコア計算
        stability_score = analyzer.calculate_stability_score()
        
        # テスト結果の判定
        test_passed = (
            final_metrics["backpressure_events"] > 0 and  # バックプレッシャーが発生
            alert_stats["total_alerts"] > 0 and  # アラートが発生
            recovery_success and  # 回復成功
            stability_score >= 70.0  # 安定性スコア70%以上
        )
        
        test_result = TestResult(
            test_name="バックプレッシャー制御と回復",
            status="PASSED" if test_passed else "FAILED",
            duration_seconds=(datetime.now() - test_start).total_seconds(),
            performance_report=performance_report,
            logs=[
                f"バックプレッシャーイベント: {final_metrics['backpressure_events']}",
                f"拒否されたアイテム: {final_metrics['rejected_items']}",
                f"アラート総数: {alert_stats['total_alerts']}",
                f"アラート分布: {alert_stats['severity_distribution']}",
                f"最大キューサイズ: {final_metrics['max_queue_size']}",
                f"安定性スコア: {stability_score:.1f}/100",
                f"回復成功: {recovery_success}",
            ]
        )
        
        # レポート生成
        report_generator.add_test_result(test_result)
        
        # 結果表示
        logger.info("=" * 60)
        logger.info("テスト結果サマリー")
        logger.info("=" * 60)
        logger.info(f"バックプレッシャーイベント: {final_metrics['backpressure_events']}")
        logger.info(f"アラート総数: {alert_stats['total_alerts']}")
        logger.info(f"安定性スコア: {stability_score:.1f}/100")
        logger.info(f"回復テスト: {'✅ 成功' if recovery_success else '❌ 失敗'}")
        logger.info(f"テスト結果: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        
        # レポートファイル生成
        md_path = report_generator.generate_markdown_report("backpressure_recovery.md")
        html_path = report_generator.generate_html_report("backpressure_recovery.html")
        
        logger.info(f"レポート生成完了:")
        logger.info(f"  - Markdown: {md_path}")
        logger.info(f"  - HTML: {html_path}")
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}", exc_info=True)
        
        test_result = TestResult(
            test_name="バックプレッシャー制御と回復",
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
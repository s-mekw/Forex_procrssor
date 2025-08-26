"""
リアルタイムFX市場データ処理シミュレーション

実際の市場環境を模擬したE2Eテストです。
複数通貨ペアのリアルタイムデータストリーミング、市場ボラティリティ変動、
データバースト、遅延監視機能をテストします。
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
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


async def simulate_market_open(
    pipeline: RealtimePipeline,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer,
    symbols: list[str],
    duration_seconds: int = 10
):
    """市場オープン時のシミュレーション"""
    logger.info("=== 市場オープンシミュレーション開始 ===")
    
    # 市場オープン時の高ボラティリティデータを生成
    for symbol in symbols[:3]:  # 主要3通貨ペア
        burst_data = generator.generate_burst(
            symbol=symbol,
            count=50,
            interval_ms=20,
            condition=MarketCondition.MARKET_OPEN
        )
        
        for data_point in burst_data:
            success = await pipeline.submit(data_point)
            analyzer.record_message(success)
            await asyncio.sleep(0.001)  # 1msの遅延
    
    # パイプラインが処理を完了するまで待機
    await asyncio.sleep(2)
    
    # メトリクスを取得して分析
    metrics = pipeline.get_metrics()
    analyzer.analyze_pipeline_metrics(metrics)
    logger.info(f"市場オープン処理完了: {metrics['processed_count']}件処理")


async def simulate_normal_trading(
    pipeline: RealtimePipeline,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer,
    symbols: list[str],
    duration_seconds: int = 30
):
    """通常取引時のシミュレーション"""
    logger.info("=== 通常取引シミュレーション開始 ===")
    
    start_time = datetime.now()
    message_count = 0
    
    while (datetime.now() - start_time).total_seconds() < duration_seconds:
        # ランダムな通貨ペアのデータを生成
        import random
        symbol = random.choice(symbols)
        
        data_point = generator.generate_tick(
            symbol=symbol,
            condition=MarketCondition.NORMAL
        )
        
        success = await pipeline.submit(data_point)
        analyzer.record_message(success)
        message_count += 1
        
        # リアルなレート更新間隔（10-100ms）
        await asyncio.sleep(random.uniform(0.01, 0.1))
    
    # メトリクスを取得
    await asyncio.sleep(1)
    metrics = pipeline.get_metrics()
    analyzer.analyze_pipeline_metrics(metrics)
    logger.info(f"通常取引処理完了: {message_count}件送信, {metrics['processed_count']}件処理")


async def simulate_news_release(
    pipeline: RealtimePipeline,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer,
    symbol: str = "USDJPY"
):
    """ニュースリリース時のデータバーストシミュレーション"""
    logger.info(f"=== ニュースリリースシミュレーション開始 ({symbol}) ===")
    
    # ニュースリリース時の急激なデータ増加
    burst_data = generator.generate_burst(
        symbol=symbol,
        count=200,  # 大量のデータ
        interval_ms=5,  # 高頻度
        condition=MarketCondition.NEWS_RELEASE
    )
    
    submit_tasks = []
    for data_point in burst_data:
        submit_tasks.append(pipeline.submit(data_point))
    
    # 全データを非同期で送信
    results = await asyncio.gather(*submit_tasks)
    success_count = sum(1 for r in results if r)
    
    for success in results:
        analyzer.record_message(success)
        if not success:
            analyzer.record_backpressure()
    
    logger.info(f"ニュースリリース: {success_count}/{len(burst_data)}件送信成功")
    
    # 処理完了待機
    await asyncio.sleep(3)
    
    # アラート統計を確認
    alert_stats = pipeline.get_alert_statistics()
    if alert_stats["total_alerts"] > 0:
        logger.warning(f"アラート発生: {alert_stats['total_alerts']}件")
        logger.info(f"アラート分布: {alert_stats['severity_distribution']}")


async def simulate_high_volatility(
    pipeline: RealtimePipeline,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer,
    symbols: list[str],
    duration_seconds: int = 20
):
    """高ボラティリティ市場のシミュレーション"""
    logger.info("=== 高ボラティリティシミュレーション開始 ===")
    
    start_time = datetime.now()
    
    while (datetime.now() - start_time).total_seconds() < duration_seconds:
        # 複数通貨ペアで同時に高ボラティリティ
        tasks = []
        for symbol in symbols[:5]:  # 5通貨ペア
            data_point = generator.generate_tick(
                symbol=symbol,
                condition=MarketCondition.HIGH_VOLATILITY
            )
            tasks.append(pipeline.submit(data_point))
        
        results = await asyncio.gather(*tasks)
        for success in results:
            analyzer.record_message(success)
        
        # 短い間隔で次のデータ
        await asyncio.sleep(0.05)
    
    # メトリクス確認
    await asyncio.sleep(2)
    metrics = pipeline.get_metrics()
    analyzer.analyze_pipeline_metrics(metrics)
    logger.info(f"高ボラティリティ処理: {metrics['processed_count']}件処理")


async def simulate_delayed_provider(
    pipeline: RealtimePipeline,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer
):
    """遅延プロバイダーのシミュレーション（アラートテスト）"""
    logger.info("=== 遅延プロバイダーシミュレーション開始 ===")
    
    # アラートコールバックの設定
    alert_count = {"count": 0}
    
    async def alert_callback(alert_info):
        alert_count["count"] += 1
        logger.warning(f"カスタムアラート受信: {alert_info['severity']} - {alert_info['latency']:.2f}秒")
    
    pipeline.set_alert_callback(alert_callback)
    
    # 遅延データの生成（2秒、5秒、10秒の遅延）
    delays = [2.0, 5.0, 10.0]
    for delay in delays:
        delayed_data = generator.generate_delayed_data(
            symbol="EURUSD",
            delay_seconds=delay,
            count=5
        )
        
        for data_point in delayed_data:
            success = await pipeline.submit(data_point)
            analyzer.record_message(success)
            await asyncio.sleep(0.1)
    
    # 処理完了待機
    await asyncio.sleep(3)
    
    # アラート統計確認
    alert_stats = pipeline.get_alert_statistics()
    logger.info(f"遅延アラート発生: {alert_stats['total_alerts']}件")
    logger.info(f"カスタムコールバック呼び出し: {alert_count['count']}回")
    
    # アラートを分析に記録
    for severity, count in alert_stats["severity_distribution"].items():
        for _ in range(count):
            analyzer.record_alert(severity, 0)


async def main():
    """メインテスト実行"""
    logger.info("=" * 60)
    logger.info("リアルタイムFX市場シミュレーション E2Eテスト開始")
    logger.info("=" * 60)
    
    # テストコンポーネントの初期化
    report_generator = ReportGenerator("test_sandbox/task10_test/reports")
    
    # 使用する通貨ペア
    symbols = list(FXDataGenerator.CURRENCY_PAIRS.keys())
    
    # パイプラインの初期化
    pipeline = RealtimePipeline(
        queue_size=1000,
        alert_threshold=1.0,  # 1秒閾値
        enable_metrics=True
    )
    
    # データジェネレーターと分析器の初期化
    generator = FXDataGenerator(seed=42)  # 再現性のためシード固定
    analyzer = MetricsAnalyzer()
    
    test_start = datetime.now()
    
    try:
        # パイプライン開始
        await pipeline.start()
        logger.info("パイプライン起動完了")
        
        # 分析開始
        analyzer.start_analysis()
        
        # 1. 市場オープンシミュレーション
        await simulate_market_open(pipeline, generator, analyzer, symbols, 10)
        
        # 2. 通常取引シミュレーション
        await simulate_normal_trading(pipeline, generator, analyzer, symbols, 30)
        
        # 3. ニュースリリースシミュレーション
        await simulate_news_release(pipeline, generator, analyzer, "USDJPY")
        
        # 4. 高ボラティリティシミュレーション
        await simulate_high_volatility(pipeline, generator, analyzer, symbols, 20)
        
        # 5. 遅延プロバイダーシミュレーション
        await simulate_delayed_provider(pipeline, generator, analyzer)
        
        # 最終メトリクス取得
        await asyncio.sleep(3)
        final_metrics = pipeline.get_metrics()
        queue_status = await pipeline.get_queue_status()
        
        # 分析終了
        analyzer.end_analysis()
        
        # レポート生成
        performance_report = analyzer.generate_report("リアルタイムFX市場シミュレーション")
        
        # 安定性スコア計算
        stability_score = analyzer.calculate_stability_score()
        
        # テスト結果の判定
        test_passed = (
            performance_report.success_rate_percent >= 95.0 and
            performance_report.avg_latency_ms < 1000 and
            stability_score >= 80.0
        )
        
        test_result = TestResult(
            test_name="リアルタイムFX市場シミュレーション",
            status="PASSED" if test_passed else "FAILED",
            duration_seconds=(datetime.now() - test_start).total_seconds(),
            performance_report=performance_report,
            logs=[
                f"総メッセージ数: {final_metrics['processed_count']}",
                f"平均遅延: {final_metrics['avg_latency']:.3f}秒",
                f"アラート数: {final_metrics['alert_count']}",
                f"バックプレッシャーイベント: {final_metrics['backpressure_events']}",
                f"安定性スコア: {stability_score:.1f}/100",
            ]
        )
        
        # レポート生成
        report_generator.add_test_result(test_result)
        
        # 結果表示
        logger.info("=" * 60)
        logger.info("テスト結果サマリー")
        logger.info("=" * 60)
        logger.info(performance_report.get_summary())
        logger.info(f"安定性スコア: {stability_score:.1f}/100")
        logger.info(f"テスト結果: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        
        # レポートファイル生成
        md_path = report_generator.generate_markdown_report("realtime_fx_simulation.md")
        html_path = report_generator.generate_html_report("realtime_fx_simulation.html")
        json_path = report_generator.generate_json_report("realtime_fx_simulation.json")
        
        logger.info(f"レポート生成完了:")
        logger.info(f"  - Markdown: {md_path}")
        logger.info(f"  - HTML: {html_path}")
        logger.info(f"  - JSON: {json_path}")
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}", exc_info=True)
        
        test_result = TestResult(
            test_name="リアルタイムFX市場シミュレーション",
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
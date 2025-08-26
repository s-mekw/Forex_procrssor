"""
マルチプロバイダー統合シナリオ

複数のデータプロバイダーからの同時データ処理、優先度処理、
フェイルオーバー、データ重複排除をテストします。
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# プロジェクトのルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.pipelines import DataPoint, RealtimePipeline
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


class Provider:
    """データプロバイダーのシミュレーション"""
    
    def __init__(self, name: str, latency_ms: float, reliability: float, priority: int):
        """
        初期化
        
        Args:
            name: プロバイダー名
            latency_ms: 遅延（ミリ秒）
            reliability: 信頼性（0.0-1.0）
            priority: 優先度（低い値が高優先度）
        """
        self.name = name
        self.latency_ms = latency_ms
        self.reliability = reliability
        self.priority = priority
        self.is_active = True
        self.message_count = 0
        self.error_count = 0
        
    async def send_data(self, data_point: DataPoint) -> bool:
        """
        データ送信シミュレーション
        
        Args:
            data_point: 送信データ
            
        Returns:
            bool: 送信成功/失敗
        """
        if not self.is_active:
            self.error_count += 1
            return False
        
        # 遅延シミュレーション
        await asyncio.sleep(self.latency_ms / 1000)
        
        # 信頼性に基づく成功/失敗
        import random
        if random.random() > self.reliability:
            self.error_count += 1
            return False
        
        self.message_count += 1
        return True
    
    def fail(self):
        """プロバイダー障害をシミュレート"""
        logger.warning(f"プロバイダー {self.name} が障害")
        self.is_active = False
    
    def recover(self):
        """プロバイダー復旧をシミュレート"""
        logger.info(f"プロバイダー {self.name} が復旧")
        self.is_active = True


class MultiProviderManager:
    """マルチプロバイダー管理"""
    
    def __init__(self, pipeline: RealtimePipeline):
        """
        初期化
        
        Args:
            pipeline: RealtimePipeline インスタンス
        """
        self.pipeline = pipeline
        self.providers = []
        self.processed_ticks = set()  # 重複排除用
        self.duplicate_count = 0
        
    def add_provider(self, provider: Provider):
        """プロバイダーを追加"""
        self.providers.append(provider)
        self.providers.sort(key=lambda p: p.priority)  # 優先度順にソート
        
    async def process_data(self, data_point: DataPoint) -> bool:
        """
        データ処理（重複排除とプロバイダー選択）
        
        Args:
            data_point: 処理するデータ
            
        Returns:
            bool: 処理成功/失敗
        """
        # 重複チェック用のキー生成
        tick_id = data_point["data"].get("tick_id")
        provider = data_point["data"].get("provider")
        duplicate_key = f"{provider}_{tick_id}"
        
        # 重複排除
        if duplicate_key in self.processed_ticks:
            self.duplicate_count += 1
            logger.debug(f"重複データ検出: {duplicate_key}")
            return False
        
        self.processed_ticks.add(duplicate_key)
        
        # パイプラインに送信
        return await self.pipeline.submit(data_point)
    
    async def send_with_failover(
        self,
        data_point: DataPoint,
        analyzer: MetricsAnalyzer
    ) -> bool:
        """
        フェイルオーバー機能付きでデータ送信
        
        Args:
            data_point: 送信データ
            analyzer: メトリクス分析器
            
        Returns:
            bool: 送信成功/失敗
        """
        for provider in self.providers:
            if await provider.send_data(data_point):
                # プロバイダー情報を追加
                data_point["data"]["provider"] = provider.name
                data_point["metadata"]["latency_ms"] = provider.latency_ms
                
                # パイプラインに送信
                success = await self.process_data(data_point)
                analyzer.record_message(success, provider.latency_ms)
                return success
        
        # すべてのプロバイダーで失敗
        analyzer.record_message(False)
        return False
    
    def get_statistics(self) -> dict:
        """統計情報を取得"""
        stats = {
            "providers": [],
            "duplicate_count": self.duplicate_count,
            "processed_unique": len(self.processed_ticks)
        }
        
        for provider in self.providers:
            stats["providers"].append({
                "name": provider.name,
                "active": provider.is_active,
                "message_count": provider.message_count,
                "error_count": provider.error_count,
                "reliability": provider.reliability,
                "latency_ms": provider.latency_ms,
                "priority": provider.priority
            })
        
        return stats


async def simulate_normal_operation(
    manager: MultiProviderManager,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer,
    duration_seconds: int = 30
):
    """通常運用のシミュレーション"""
    logger.info("=== 通常運用シミュレーション開始 ===")
    
    start_time = datetime.now()
    symbols = ["USDJPY", "EURUSD", "GBPUSD"]
    
    while (datetime.now() - start_time).total_seconds() < duration_seconds:
        # 各プロバイダーから同じデータを送信（重複テスト）
        import random
        symbol = random.choice(symbols)
        base_data = generator.generate_tick(symbol, MarketCondition.NORMAL)
        
        tasks = []
        for _ in manager.providers:
            # 同じティックIDで複数プロバイダーから送信
            data_copy = DataPoint(
                timestamp=base_data["timestamp"],
                data=base_data["data"].copy(),
                metadata=base_data["metadata"].copy()
            )
            tasks.append(manager.send_with_failover(data_copy, analyzer))
        
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.1)
    
    logger.info(f"通常運用完了: 重複排除数 = {manager.duplicate_count}")


async def simulate_provider_failure(
    manager: MultiProviderManager,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer
):
    """プロバイダー障害シミュレーション"""
    logger.info("=== プロバイダー障害シミュレーション開始 ===")
    
    # プライマリプロバイダー（最優先）を障害にする
    primary_provider = manager.providers[0]
    primary_provider.fail()
    
    # 障害中のデータ送信
    for i in range(50):
        data_point = generator.generate_tick("USDJPY", MarketCondition.HIGH_VOLATILITY)
        success = await manager.send_with_failover(data_point, analyzer)
        
        if success:
            logger.debug(f"フェイルオーバー成功: メッセージ {i+1}")
        else:
            logger.warning(f"全プロバイダー失敗: メッセージ {i+1}")
        
        await asyncio.sleep(0.05)
    
    # プロバイダー復旧
    primary_provider.recover()
    logger.info("プライマリプロバイダー復旧")
    
    # 復旧後のデータ送信
    for i in range(20):
        data_point = generator.generate_tick("USDJPY", MarketCondition.NORMAL)
        await manager.send_with_failover(data_point, analyzer)
        await asyncio.sleep(0.05)


async def simulate_priority_processing(
    manager: MultiProviderManager,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer
):
    """優先度処理のシミュレーション"""
    logger.info("=== 優先度処理シミュレーション開始 ===")
    
    # 異なる優先度のプロバイダーから同時送信
    symbols = ["EURUSD", "GBPUSD", "AUDUSD"]
    
    for _ in range(30):
        tasks = []
        for symbol in symbols:
            data_point = generator.generate_tick(symbol, MarketCondition.NORMAL)
            tasks.append(manager.send_with_failover(data_point, analyzer))
        
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.1)
    
    # 統計を確認
    stats = manager.get_statistics()
    for provider_stat in stats["providers"]:
        logger.info(
            f"プロバイダー {provider_stat['name']}: "
            f"送信数={provider_stat['message_count']}, "
            f"エラー数={provider_stat['error_count']}"
        )


async def simulate_data_consistency(
    manager: MultiProviderManager,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer
):
    """データ整合性テスト"""
    logger.info("=== データ整合性テスト開始 ===")
    
    # 同じデータを複数回送信して重複排除を確認
    test_data = generator.generate_tick("USDJPY", MarketCondition.NORMAL)
    
    duplicate_attempts = 0
    for _ in range(10):
        # 同じデータを複数回送信
        data_copy = DataPoint(
            timestamp=test_data["timestamp"],
            data=test_data["data"].copy(),
            metadata=test_data["metadata"].copy()
        )
        
        success = await manager.process_data(data_copy)
        if not success:
            duplicate_attempts += 1
        
        await asyncio.sleep(0.01)
    
    logger.info(f"重複送信テスト: {duplicate_attempts}/9 件が正しく排除されました")
    
    # 異なるプロバイダーから同じティックIDのデータ
    base_tick_id = 99999
    for provider in manager.providers:
        data_point = DataPoint(
            timestamp=datetime.now(),
            data={
                "tick_id": base_tick_id,
                "symbol": "EURUSD",
                "bid": 1.0800,
                "ask": 1.0802,
                "provider": provider.name
            },
            metadata={"source": "consistency_test"}
        )
        
        await manager.process_data(data_point)
        await asyncio.sleep(0.01)
    
    logger.info(f"プロバイダー間重複テスト完了")


async def main():
    """メインテスト実行"""
    logger.info("=" * 60)
    logger.info("マルチプロバイダー統合 E2Eテスト開始")
    logger.info("=" * 60)
    
    # テストコンポーネントの初期化
    report_generator = ReportGenerator("test_sandbox/task10_test/reports")
    
    # パイプラインの初期化
    pipeline = RealtimePipeline(
        queue_size=2000,
        alert_threshold=0.5,  # 500ms閾値
        enable_metrics=True
    )
    
    # マルチプロバイダーマネージャーの初期化
    manager = MultiProviderManager(pipeline)
    
    # プロバイダーの設定
    providers = [
        Provider("Premium", latency_ms=5, reliability=0.99, priority=1),
        Provider("Standard", latency_ms=20, reliability=0.95, priority=2),
        Provider("Backup", latency_ms=50, reliability=0.90, priority=3),
    ]
    
    for provider in providers:
        manager.add_provider(provider)
        logger.info(f"プロバイダー追加: {provider.name} (優先度={provider.priority})")
    
    # データジェネレーターと分析器
    generator = FXDataGenerator(seed=123)
    analyzer = MetricsAnalyzer()
    
    test_start = datetime.now()
    
    try:
        # パイプライン開始
        await pipeline.start()
        logger.info("パイプライン起動完了")
        
        # 分析開始
        analyzer.start_analysis()
        
        # 1. 通常運用シミュレーション
        await simulate_normal_operation(manager, generator, analyzer, 20)
        
        # 2. プロバイダー障害シミュレーション
        await simulate_provider_failure(manager, generator, analyzer)
        
        # 3. 優先度処理シミュレーション
        await simulate_priority_processing(manager, generator, analyzer)
        
        # 4. データ整合性テスト
        await simulate_data_consistency(manager, generator, analyzer)
        
        # 最終メトリクス取得
        await asyncio.sleep(2)
        final_metrics = pipeline.get_metrics()
        provider_stats = manager.get_statistics()
        
        # 分析終了
        analyzer.end_analysis()
        
        # パフォーマンスレポート生成
        performance_report = analyzer.generate_report("マルチプロバイダー統合")
        
        # テスト結果の判定
        test_passed = (
            performance_report.success_rate_percent >= 90.0 and
            provider_stats["duplicate_count"] > 0 and  # 重複排除が機能
            all(p["message_count"] > 0 for p in provider_stats["providers"])  # 全プロバイダー稼働
        )
        
        test_result = TestResult(
            test_name="マルチプロバイダー統合",
            status="PASSED" if test_passed else "FAILED",
            duration_seconds=(datetime.now() - test_start).total_seconds(),
            performance_report=performance_report,
            logs=[
                f"プロバイダー数: {len(providers)}",
                f"重複排除数: {provider_stats['duplicate_count']}",
                f"ユニークメッセージ数: {provider_stats['processed_unique']}",
                f"総処理数: {final_metrics['processed_count']}",
                "=== プロバイダー統計 ===",
            ] + [
                f"{p['name']}: 送信={p['message_count']}, エラー={p['error_count']}"
                for p in provider_stats["providers"]
            ]
        )
        
        # レポート生成
        report_generator.add_test_result(test_result)
        
        # 結果表示
        logger.info("=" * 60)
        logger.info("テスト結果サマリー")
        logger.info("=" * 60)
        logger.info(f"総処理メッセージ数: {final_metrics['processed_count']}")
        logger.info(f"重複排除数: {provider_stats['duplicate_count']}")
        logger.info(f"成功率: {performance_report.success_rate_percent:.1f}%")
        
        for provider_stat in provider_stats["providers"]:
            logger.info(
                f"{provider_stat['name']}: "
                f"送信={provider_stat['message_count']}, "
                f"エラー={provider_stat['error_count']}, "
                f"稼働={provider_stat['active']}"
            )
        
        logger.info(f"テスト結果: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        
        # レポートファイル生成
        md_path = report_generator.generate_markdown_report("multi_provider.md")
        html_path = report_generator.generate_html_report("multi_provider.html")
        
        logger.info(f"レポート生成完了:")
        logger.info(f"  - Markdown: {md_path}")
        logger.info(f"  - HTML: {html_path}")
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}", exc_info=True)
        
        test_result = TestResult(
            test_name="マルチプロバイダー統合",
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
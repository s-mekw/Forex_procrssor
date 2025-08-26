"""
障害復旧とデータ整合性シナリオ

障害発生時の処理継続性とデータ整合性の確保をテストします。
予期しない停止、データ欠損チェック、重複防止、リカバリー後の正常動作を検証します。
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


class DataIntegrityChecker:
    """データ整合性チェッカー"""
    
    def __init__(self):
        self.sent_data = []  # 送信データの記録
        self.received_data = []  # 受信データの記録
        self.expected_sequence = {}  # 期待されるシーケンス番号
        
    def record_sent(self, data_point: DataPoint):
        """送信データを記録"""
        self.sent_data.append({
            "tick_id": data_point["data"].get("tick_id"),
            "symbol": data_point["data"].get("symbol"),
            "timestamp": data_point["timestamp"],
            "bid": data_point["data"].get("bid"),
        })
        
    def record_received(self, result: dict):
        """受信データを記録"""
        if result and "processed_data" in result:
            self.received_data.append({
                "tick_id": result["processed_data"]["data"].get("tick_id"),
                "symbol": result["processed_data"]["data"].get("symbol"),
                "timestamp": result["processed_data"]["timestamp"],
                "bid": result["processed_data"]["data"].get("bid"),
            })
    
    def check_integrity(self) -> dict:
        """データ整合性をチェック"""
        sent_ids = {d["tick_id"] for d in self.sent_data if d["tick_id"]}
        received_ids = {d["tick_id"] for d in self.received_data if d["tick_id"]}
        
        missing = sent_ids - received_ids
        duplicates = self._find_duplicates()
        
        integrity_score = 100.0
        if missing:
            integrity_score -= len(missing) / len(sent_ids) * 50 if sent_ids else 0
        if duplicates:
            integrity_score -= len(duplicates) / len(self.received_data) * 50 if self.received_data else 0
        
        return {
            "sent_count": len(self.sent_data),
            "received_count": len(self.received_data),
            "missing_count": len(missing),
            "missing_ids": list(missing)[:10],  # 最初の10件
            "duplicate_count": len(duplicates),
            "integrity_score": max(0, integrity_score),
        }
    
    def _find_duplicates(self) -> list:
        """重複データを検出"""
        seen = set()
        duplicates = []
        
        for data in self.received_data:
            tick_id = data["tick_id"]
            if tick_id in seen:
                duplicates.append(tick_id)
            seen.add(tick_id)
        
        return duplicates
    
    def validate_sequence(self, symbol: str) -> bool:
        """シーケンスの連続性を検証"""
        symbol_data = [d for d in self.received_data if d["symbol"] == symbol]
        
        if len(symbol_data) < 2:
            return True
        
        # タイムスタンプの順序をチェック
        for i in range(1, len(symbol_data)):
            if symbol_data[i]["timestamp"] < symbol_data[i-1]["timestamp"]:
                logger.warning(f"シーケンス違反検出: {symbol}")
                return False
        
        return True


async def simulate_sudden_shutdown(
    pipeline: RealtimePipeline,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer,
    integrity_checker: DataIntegrityChecker
):
    """突然のシャットダウンシミュレーション"""
    logger.info("=== 突然のシャットダウンシミュレーション開始 ===")
    
    # 通常処理開始
    logger.info("通常処理を開始")
    symbols = ["USDJPY", "EURUSD", "GBPUSD"]
    
    # 100件のデータを送信
    for i in range(100):
        import random
        symbol = random.choice(symbols)
        data_point = generator.generate_tick(symbol, MarketCondition.NORMAL)
        
        # 送信記録
        integrity_checker.record_sent(data_point)
        
        # 送信
        success = await pipeline.submit(data_point)
        analyzer.record_message(success)
        
        # 50件目で突然停止
        if i == 50:
            logger.warning("⚠️ 突然のシャットダウンをシミュレート")
            await pipeline.stop()
            await asyncio.sleep(2)  # 停止時間
            
            logger.info("システム再起動")
            await pipeline.start()
            logger.info("再起動完了")
        
        await asyncio.sleep(0.01)
    
    # 処理結果を取得
    await asyncio.sleep(2)
    
    # 受信データを記録（キューから取得）
    received_count = 0
    while True:
        try:
            result = await asyncio.wait_for(pipeline.get_result(), timeout=0.1)
            if result:
                integrity_checker.record_received(result)
                received_count += 1
        except asyncio.TimeoutError:
            break
    
    logger.info(f"受信データ数: {received_count}")


async def simulate_data_corruption(
    pipeline: RealtimePipeline,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer
):
    """データ破損シミュレーション"""
    logger.info("=== データ破損シミュレーション開始 ===")
    
    corrupted_count = 0
    normal_count = 0
    
    for i in range(50):
        data_point = generator.generate_tick("USDJPY", MarketCondition.NORMAL)
        
        # 10%の確率でデータを破損させる
        import random
        if random.random() < 0.1:
            # データ破損をシミュレート（不正な値）
            data_point["data"]["bid"] = -999999  # 不正な価格
            data_point["data"]["ask"] = None  # 欠損値
            corrupted_count += 1
        else:
            normal_count += 1
        
        try:
            success = await pipeline.submit(data_point)
            analyzer.record_message(success)
        except Exception as e:
            logger.warning(f"データ送信エラー: {e}")
            analyzer.record_message(False)
        
        await asyncio.sleep(0.05)
    
    logger.info(f"正常データ: {normal_count}件, 破損データ: {corrupted_count}件")


async def simulate_network_partition(
    pipeline: RealtimePipeline,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer,
    integrity_checker: DataIntegrityChecker
):
    """ネットワーク分断シミュレーション"""
    logger.info("=== ネットワーク分断シミュレーション開始 ===")
    
    # 2つの独立したデータストリームをシミュレート
    stream1_data = []
    stream2_data = []
    
    # ストリーム1: 通常のデータフロー
    logger.info("ストリーム1: 通常処理")
    for i in range(30):
        data_point = generator.generate_tick("EURUSD", MarketCondition.NORMAL)
        integrity_checker.record_sent(data_point)
        stream1_data.append(data_point)
        
        success = await pipeline.submit(data_point)
        analyzer.record_message(success)
        await asyncio.sleep(0.05)
    
    # ネットワーク分断発生
    logger.warning("⚠️ ネットワーク分断発生")
    
    # ストリーム2: 分断中のデータ（バッファリング）
    logger.info("ストリーム2: 分断中（データバッファリング）")
    for i in range(20):
        data_point = generator.generate_tick("EURUSD", MarketCondition.HIGH_VOLATILITY)
        stream2_data.append(data_point)
        # 分断中なので送信できない
        analyzer.record_message(False)
    
    # ネットワーク復旧
    logger.info("ネットワーク復旧")
    
    # バッファされたデータを一括送信
    logger.info("バッファデータの一括送信")
    for data_point in stream2_data:
        integrity_checker.record_sent(data_point)
        success = await pipeline.submit(data_point)
        analyzer.record_message(success)
        
        # 受信可能なデータを取得
        try:
            result = await asyncio.wait_for(pipeline.get_result(), timeout=0.01)
            if result:
                integrity_checker.record_received(result)
        except asyncio.TimeoutError:
            pass
    
    await asyncio.sleep(1)


async def simulate_recovery_validation(
    pipeline: RealtimePipeline,
    generator: FXDataGenerator,
    analyzer: MetricsAnalyzer
):
    """リカバリー後の正常動作検証"""
    logger.info("=== リカバリー後の正常動作検証 ===")
    
    # ベースラインパフォーマンスの測定
    baseline_latencies = []
    
    logger.info("ベースライン測定中...")
    for i in range(100):
        start_time = datetime.now()
        data_point = generator.generate_tick("GBPUSD", MarketCondition.NORMAL)
        
        success = await pipeline.submit(data_point)
        if success:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            baseline_latencies.append(latency)
            analyzer.record_message(True, latency)
        else:
            analyzer.record_message(False)
        
        await asyncio.sleep(0.01)
    
    baseline_avg = sum(baseline_latencies) / len(baseline_latencies) if baseline_latencies else 0
    logger.info(f"ベースライン平均遅延: {baseline_avg:.2f}ms")
    
    # 障害シミュレーション
    logger.warning("障害を発生させます")
    await pipeline.stop()
    await asyncio.sleep(3)
    
    # 復旧
    logger.info("システム復旧中...")
    await pipeline.start()
    
    # 復旧後のパフォーマンス測定
    recovery_latencies = []
    
    logger.info("復旧後パフォーマンス測定中...")
    for i in range(100):
        start_time = datetime.now()
        data_point = generator.generate_tick("GBPUSD", MarketCondition.NORMAL)
        
        success = await pipeline.submit(data_point)
        if success:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            recovery_latencies.append(latency)
            analyzer.record_message(True, latency)
        else:
            analyzer.record_message(False)
        
        await asyncio.sleep(0.01)
    
    recovery_avg = sum(recovery_latencies) / len(recovery_latencies) if recovery_latencies else 0
    logger.info(f"復旧後平均遅延: {recovery_avg:.2f}ms")
    
    # パフォーマンス劣化の判定
    degradation = ((recovery_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
    
    is_recovered = abs(degradation) < 20  # 20%以内の変動は許容
    logger.info(f"パフォーマンス変動: {degradation:+.1f}%")
    logger.info(f"復旧判定: {'✅ 成功' if is_recovered else '❌ 失敗'}")
    
    return is_recovered, baseline_avg, recovery_avg


async def main():
    """メインテスト実行"""
    logger.info("=" * 60)
    logger.info("障害復旧とデータ整合性 E2Eテスト開始")
    logger.info("=" * 60)
    
    # テストコンポーネントの初期化
    report_generator = ReportGenerator("test_sandbox/task10_test/reports")
    
    # パイプラインの初期化
    pipeline = RealtimePipeline(
        queue_size=1000,
        alert_threshold=1.0,
        enable_metrics=True
    )
    
    # コンポーネント初期化
    generator = FXDataGenerator(seed=999)
    analyzer = MetricsAnalyzer()
    integrity_checker = DataIntegrityChecker()
    
    test_start = datetime.now()
    
    try:
        # パイプライン開始
        await pipeline.start()
        logger.info("パイプライン起動完了")
        
        # 分析開始
        analyzer.start_analysis()
        
        # 1. 突然のシャットダウンシミュレーション
        await simulate_sudden_shutdown(pipeline, generator, analyzer, integrity_checker)
        
        # 2. データ破損シミュレーション
        await simulate_data_corruption(pipeline, generator, analyzer)
        
        # 3. ネットワーク分断シミュレーション
        await simulate_network_partition(pipeline, generator, analyzer, integrity_checker)
        
        # 4. リカバリー後の正常動作検証
        recovery_success, baseline_latency, recovery_latency = await simulate_recovery_validation(
            pipeline, generator, analyzer
        )
        
        # データ整合性チェック
        integrity_result = integrity_checker.check_integrity()
        
        # シーケンス検証
        sequence_valid = all([
            integrity_checker.validate_sequence(symbol)
            for symbol in ["USDJPY", "EURUSD", "GBPUSD"]
        ])
        
        # 最終メトリクス取得
        await asyncio.sleep(2)
        final_metrics = pipeline.get_metrics()
        
        # 分析終了
        analyzer.end_analysis()
        
        # パフォーマンスレポート生成
        performance_report = analyzer.generate_report("障害復旧とデータ整合性")
        
        # テスト結果の判定
        test_passed = (
            integrity_result["integrity_score"] >= 80.0 and  # データ整合性80%以上
            recovery_success and  # リカバリー成功
            sequence_valid and  # シーケンス維持
            performance_report.success_rate_percent >= 85.0  # 成功率85%以上
        )
        
        test_result = TestResult(
            test_name="障害復旧とデータ整合性",
            status="PASSED" if test_passed else "FAILED",
            duration_seconds=(datetime.now() - test_start).total_seconds(),
            performance_report=performance_report,
            logs=[
                "=== データ整合性 ===",
                f"送信数: {integrity_result['sent_count']}",
                f"受信数: {integrity_result['received_count']}",
                f"欠損数: {integrity_result['missing_count']}",
                f"重複数: {integrity_result['duplicate_count']}",
                f"整合性スコア: {integrity_result['integrity_score']:.1f}%",
                "=== リカバリー性能 ===",
                f"ベースライン遅延: {baseline_latency:.2f}ms",
                f"復旧後遅延: {recovery_latency:.2f}ms",
                f"リカバリー成功: {recovery_success}",
                f"シーケンス検証: {sequence_valid}",
            ]
        )
        
        # レポート生成
        report_generator.add_test_result(test_result)
        
        # 結果表示
        logger.info("=" * 60)
        logger.info("テスト結果サマリー")
        logger.info("=" * 60)
        logger.info(f"データ整合性スコア: {integrity_result['integrity_score']:.1f}%")
        logger.info(f"データ欠損: {integrity_result['missing_count']}件")
        logger.info(f"データ重複: {integrity_result['duplicate_count']}件")
        logger.info(f"リカバリー: {'✅ 成功' if recovery_success else '❌ 失敗'}")
        logger.info(f"シーケンス検証: {'✅ 有効' if sequence_valid else '❌ 無効'}")
        logger.info(f"テスト結果: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        
        # レポートファイル生成
        md_path = report_generator.generate_markdown_report("disaster_recovery.md")
        html_path = report_generator.generate_html_report("disaster_recovery.html")
        json_path = report_generator.generate_json_report("disaster_recovery.json")
        
        logger.info(f"レポート生成完了:")
        logger.info(f"  - Markdown: {md_path}")
        logger.info(f"  - HTML: {html_path}")
        logger.info(f"  - JSON: {json_path}")
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}", exc_info=True)
        
        test_result = TestResult(
            test_name="障害復旧とデータ整合性",
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